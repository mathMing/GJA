"""Paired G2 threshold locking with independent certification."""
import time,json
import numpy as np,pandas as pd
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
import v2_pipeline as v

O=v.ROOT/'results/locked_real_v4';O.mkdir(parents=True,exist_ok=True)
def md(d):
 return '| '+' | '.join(d.columns)+' |\n|'+'|'.join(['---']*len(d.columns))+'|\n'+'\n'.join('| '+' | '.join('NA' if pd.isna(x) else f'{x:.4g}' if isinstance(x,(float,np.floating)) else str(x) for x in r)+' |' for r in d.itertuples(index=False,name=None))
def main():
 start=time.time();d=pd.read_csv(v.O/'derived_patients_LOCAL.csv',dtype={'patient_id':str});y=d.true_label.to_numpy();g=d.mask_group.to_numpy()
 rawhash=v.old.digest(v.SOURCE);derivedhash=v.old.digest(v.O/'derived_patients_LOCAL.csv')
 splits=[];ix=np.random.default_rng(20260917).permutation(len(d));fit,cal,test=np.split(ix,[len(d)//2,len(d)//2+len(d)//4]);splits.append(('primary',-1,-1,fit,cal,test))
 for seed in range(5):
  for fold,(train,test) in enumerate(StratifiedKFold(5,shuffle=True,random_state=seed).split(d,y)):
   fit,cal=np.split(np.random.default_rng(73000+seed*10+fold).permutation(train),[int(.75*len(train))]);splits.append(('repeated',seed,fold,fit,cal,test))
 rows=[];actions=[];rules=[]
 for split,seed,fold,fit,cal,test in splits:
  assert not set(fit)&set(cal) and not set(fit)&set(test) and not set(cal)&set(test)
  for kind in ['lr','gbdt']:
   for name,features in {'clinical':v.BASE,'preop':v.BASE+v.MRI+['hpv_normalized','tct_normalized']}.items():
    pf=np.full(len(fit),np.nan)
    with threadpool_limits(limits=2):
     for tr,te in StratifiedKFold(3,shuffle=True,random_state=4801).split(fit,y[fit]):
      inner=v.build(features,kind);inner.fit(d.iloc[fit[tr]],y[fit[tr]]);pf[te]=inner.predict_proba(d.iloc[fit[te]])[:,1]
     model=v.build(features,kind);model.fit(d.iloc[fit],y[fit]);pc=model.predict_proba(d.iloc[cal])[:,1];pt=model.predict_proba(d.iloc[test])[:,1]
    assert np.isfinite(pf).all()
    ev126=v.old.evidence(pc,y[cal],g[cal]);ev63={s:(n,k,v.old.cp(k,n,.05/63)) for s,(n,k,u) in ev126.items()}
    for alpha in v.old.ALPHAS:
     locked={}
     for scope in v.old.GROUPS[:3]:
      sm=g[fit]==scope;sel=pf[sm,None]<=v.old.GRID[None,:];ns=sel.sum(0);ks=(sel*y[fit][sm,None]).sum(0)
      eligible=np.flatnonzero((ns>=10)&(ks<=alpha*.5*ns));t=float(v.old.GRID[eligible[-1]]) if len(eligible) else -np.inf
      take=(g[cal]==scope)&(pc<=t);n=int(take.sum());k=int(y[cal][take].sum());bound=float(v.old.cp(np.array([k]),np.array([n]),.05/3)[0]);active=n>0 and bound<=alpha
      locked[scope]={'threshold':t if active else -np.inf,'bound':bound if active else np.nan,'active':active}
      rules.append(dict(split=split,seed=seed,fold=fold,model=kind,features=name,risk_budget=alpha,scope=scope,proposed_threshold=t,selection_source='fit_internal_3fold_OOF',cal_accepted=n,cal_positive=k,cal_bound=bound,active=active))
     for policy,rr in [('G2_search126',v.old.rules_at(ev126,alpha)),('G2_fixed63',v.old.rules_at(ev63,alpha)),('G2_locked3',locked)]:
      auto,t,b,sc=v.old.apply(pt,g[test],rr,'G2');assert np.all(b[auto]<=alpha)
      meta=dict(split=split,seed=seed,fold=fold,model=kind,features=name,policy=policy,partition='G2',risk_budget=alpha)
      rows+=v.old.metrics_rows(y[test],g[test],auto,b,meta)
      actions.append(pd.DataFrame({**meta,'patient_id':d.patient_id.iloc[test].to_numpy(),'true_label':y[test],'mask_group':g[test],'score':pt,'action':np.where(auto,'AUTO_NEGATIVE','ABSTAIN'),'threshold':t,'risk_bound':b}))
  print(split,seed,fold,'complete',flush=True)
 r=pd.DataFrame(rows);r.to_csv(O/'fold_metrics.csv',index=False);pd.DataFrame(rules).to_csv(O/'locked_rules.csv',index=False)
 a=pd.concat(actions,ignore_index=True);assert not a.duplicated(['split','seed','model','features','policy','risk_budget','patient_id']).any();a.to_csv(O/'actions_LOCAL.csv.gz',index=False,compression='gzip')
 old=pd.read_csv(v.O/'e3_repeated_seed_metrics.csv');rs=r[r.split=='repeated'].groupby(['seed','model','features','policy','risk_budget','mask_group'])[['n','positive_n','auto_n','auto_positive_n']].sum().reset_index();rs['auto_rate']=rs.auto_n/rs.n;rs['risk']=rs.auto_positive_n/rs.auto_n.replace(0,np.nan);rs.to_csv(O/'seed_metrics.csv',index=False)
 paired=rs[rs.policy=='G2_search126'].merge(old[old.policy=='G2'],on=['seed','model','features','risk_budget','mask_group'],suffixes=('_new','_old'))
 assert len(paired)==300 and (paired.auto_n_new==paired.auto_n_old).all() and (paired.auto_positive_n_new==paired.auto_positive_n_old).all()
 primary=r[r.split=='primary'].merge(pd.read_csv(v.O/'e3_metrics.csv').query("policy=='G2'"),on=['model','features','risk_budget','mask_group'],suffixes=('_new','_old'))
 assert (primary[primary.policy_new=='G2_search126'].auto_n_new==primary[primary.policy_new=='G2_search126'].auto_n_old).all()
 summary=rs.groupby(['model','features','policy','risk_budget','mask_group']).agg(auto_n_mean=('auto_n','mean'),positive_in_auto_mean=('auto_positive_n','mean'),auto_rate_mean=('auto_rate','mean'),risk_mean_served=('risk','mean')).reset_index();summary.to_csv(O/'summary.csv',index=False)
 pivot=rs.pivot(index=['seed','model','features','risk_budget','mask_group'],columns='policy',values='auto_rate').reset_index();pivot['locked_minus_search126']=pivot.G2_locked3-pivot.G2_search126;pivot['fixed63_minus_search126']=pivot.G2_fixed63-pivot.G2_search126;pivot.to_csv(O/'paired_service_deltas.csv',index=False)
 assert v.old.digest(v.SOURCE)==rawhash and v.old.digest(v.O/'derived_patients_LOCAL.csv')==derivedhash
 db=summary[(summary.mask_group=='missing_both')&(summary.risk_budget==.2)];overall=summary[(summary.mask_group=='all')&(summary.risk_budget==.2)]
 prim=r[(r.split=='primary')&(r.mask_group=='missing_both')][['model','features','policy','risk_budget','auto_n','auto_positive_n','empirical_risk']]
 ruleframe=pd.DataFrame(rules)
 diagnostic=ruleframe[(ruleframe.scope=='missing_both')&(ruleframe.risk_budget==.2)].copy()
 diagnostic['failure_reason']=np.select([~np.isfinite(diagnostic.proposed_threshold),diagnostic.cal_accepted<19,~diagnostic.active],['no_fit_OOF_candidate','below_zero_error_min19','bound_failed_despite_at_least19'],default='certified')
 counts=diagnostic.groupby(['split','failure_reason']).size().rename('model_split_count').reset_index()
 counts.to_csv(O/'double_missing_failure_counts.csv',index=False)
 all_double=r[r.mask_group=='missing_both']
 verdict=('本轮所有预算、所有分数器、主划分和重复折的双缺失组自动人数均为零。合法减少候选数量和训练集预锁定阈值，都未在当前实验中恢复该组服务。' if all_double.auto_n.sum()==0 else '部分配置出现双缺失服务，须结合逐折风险与接收阳性数判断，不能直接称为稳定改善。')
 report='''# V4：训练集预锁定阈值能否挽救真实子群服务？

## 本轮实际结论

{verdict}

20% 风险预算下，locked3 失败分解如下。19 是 delta=0.05/3 下零阳性的最低接收校准人数；达到 19 并不保证通过。no_fit_OOF_candidate 表示拟合集内没有满足预先设定条件的阈值，并非证明不存在其他可行方法。

{failures}

Material Passport：Origin Skill=experiment-agent；Mode=run/validate；Version=locked_real_v4；Verification Status=ANALYZED（对照计数与 v2 复现一致，新的方法尚无独立外部复现）。

## 方法

同一四个分数器、主划分及 5 种子×5 折，比较 G2_search126（原方法）、G2_fixed63（预固定 G2，63 候选）、G2_locked3（仅拟合集三折 OOF 选定每组唯一阈值，再独立校准）。选择门槛为拟合 OOF 接收人数至少 10、经验风险至多预算一半。失败后弃权，不再看校准数据改阈值。每种方法的置信声明独立，不包含比较后挑赢家、所有模型/种子/预算的联合保证。

选择与最终评分模型不同（OOF 模型与全部拟合样本重训练模型），可能损害阈值效率；独立校准负责最终认证。罕见仅缺 HPV 组始终弃权。本轮不评估补查收益。

## 双缺失组：预算 20%，重复验证

{double}

人数是每种子汇总后跨五种子的均值；同一患者重复出现，不能当成新增独立样本。risk_mean_served 只对有服务种子取均值；NA 表示风险无定义，不是零漏诊证据。其余预算见完整 CSV。

## 总体服务：预算 20%

{overall}

## 主随机划分：双缺失所有预算

{primary}

## 解释边界与核验

这项配对实验区分分区/阈值选择成本，但不证明三组中任何一种方法普遍占优。若 locked3 仍无双缺失服务，则减少多重校正成本并未在当前分数和样本下解决问题；若出现服务，仍需审视接收阳性数及折间波动，不能只以人数判定方法成功。

原 G2 重复验证的每种子自动人数与自动阳性数均与 v2 一致；主划分基线一致。患者角色不重叠、动作唯一、所有自动动作认证上界不超过预算、原始与派生输入哈希未改变。重复队列已经被多轮分析，本次是探索性对照，不能提升为全新验证。

11 项偏差检查：子群分层防 Simpson 掩盖；不作生态因果推断；手术入组/Berkson；检查路径 collider；基准率；回归均值；幸存/选择；多重搜索（仅各方法内部处理）；分析分叉（先写协议但属于既有结果后续）；关联非因果；术前时间窗与反向因果。未核实临床字段仍沿用 v2 限制。

执行入口 src/locked_real_v4.py；协议 reports/v4_locked_protocol.md；逐例与完整结果 results/locked_real_v4。*_LOCAL 保留本地，不直接作为公开附件。
'''.format(double=md(db),overall=md(overall),primary=md(prim),verdict=verdict,failures=md(counts))
 (v.ROOT/'reports/Better1_v4_真实预锁定阈值对照.md').write_text(report,encoding='utf-8')
 v.dump(O/'run.json',dict(status='COMPLETED',elapsed_seconds=time.time()-start,model_fits=416,paired_baseline_checks=len(paired),code_sha256=v.old.digest(__file__),source_sha256=rawhash))
 print(db.to_string(index=False));print('ALL COMPLETE',flush=True)
if __name__=='__main__':main()
