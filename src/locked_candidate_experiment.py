"""Exploratory locked-candidate G2 ablation; no external validation."""
import json,time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
import ars_experiment as e
O=e.ROOT/"results"/"locked_candidate_v1";O.mkdir(parents=True,exist_ok=True)
start=time.time()
protocol={"seeds":list(range(5)),"budgets":e.ALPHAS,"models":["clinical_lr","clinical_gbdt","preop_lr","preop_gbdt"],
"threshold_selection":"3-fold inner OOF in fit patients only; largest fixed grid threshold with >=10 patients and observed risk <= budget/2",
"confidence":"95% per fixed scorer/split/budget across 3 G2 scopes; not simultaneous across budgets/scorers/repeats",
"calibration":"same 25% of outer training pool as original experiment",
"status":"RUNNING","code_sha256":e.digest(__file__),"source_sha256":e.digest(e.SOURCE)}
(O/"protocol.json").write_text(json.dumps(protocol,indent=2),encoding="utf-8")
d,y,g,ids=e.load_data();rows=[];audit=[];actions=[]
for seed in range(5):
 for fold,(train,test) in enumerate(StratifiedKFold(5,shuffle=True,random_state=seed).split(d,y)):
  fit,cal=np.split(np.random.default_rng(73000+seed*10+fold).permutation(train),[int(.75*len(train))])
  assert not set(fit)&set(cal) and not set(train)&set(test)
  for name in protocol["models"]:
   inner=np.full(len(fit),np.nan)
   with threadpool_limits(limits=2):
    for it,iv in StratifiedKFold(3,shuffle=True,random_state=4100+seed*10+fold).split(fit,y[fit]):
     m=e.model_for(name);m.fit(d.iloc[fit[it]],y[fit[it]])
     inner[iv]=m.predict_proba(d.iloc[fit[iv]])[:,1]
    m=e.model_for(name);m.fit(d.iloc[fit],y[fit])
    pc=m.predict_proba(d.iloc[cal])[:,1];pt=m.predict_proba(d.iloc[test])[:,1]
   assert np.isfinite(inner).all()
   ev=e.evidence(pc,y[cal],g[cal])
   for alpha in e.ALPHAS:
    locked={}
    for scope in e.GROUPS[:3]:
     sel=(g[fit,None]==scope)&(inner[:,None]<=e.GRID[None,:])
     nn=sel.sum(0);kk=(sel*y[fit,None]).sum(0)
     good=np.flatnonzero((nn>=10)&(kk<=alpha*.5*nn))
     candidate=float(e.GRID[good[-1]]) if len(good) else -np.inf
     chosen=(g[cal]==scope)&(pc<=candidate);n=int(chosen.sum());k=int(y[cal][chosen].sum())
     bound=float(e.cp(k,n,.05/3));active=n>0 and bound<=alpha
     locked[scope]={"threshold":candidate if active else -np.inf,"bound":bound if active else np.nan}
     audit.append(dict(seed=seed,fold=fold,model=name,budget=alpha,scope=scope,candidate=candidate,cal_n=n,cal_errors=k,candidate_bound=bound,active=active))
    for policy,rules in [("G2_126",e.rules_at(ev,alpha)),("G2_locked_3",locked)]:
     auto,t,b,sc=e.apply(pt,g[test],rules,"G2")
     assert not auto[g[test]=="missing_hpv"].any()
     assert np.all(b[auto]<=alpha)
     for group in ["all"]+e.GROUPS:
      ix=e.scope_mask(g[test],group)
      rows.append(dict(seed=seed,fold=fold,model=name,budget=alpha,policy=policy,group=group,**e.outcomes(y[test][ix],auto[ix])))
     actions.append(pd.DataFrame(dict(seed=seed,fold=fold,model=name,budget=alpha,policy=policy,patient_id=ids[test],mask_group=g[test],true_label=y[test],score=pt,action=np.where(auto,"AUTO_NEGATIVE","ABSTAIN"),threshold=t,risk_bound=b,scope=sc)))
  print(f"seed={seed} fold={fold} done",flush=True)
pd.DataFrame(audit).to_csv(O/"candidate_audit.csv",index=False)
pd.DataFrame(rows).to_csv(O/"fold_metrics.csv",index=False)
a=pd.concat(actions,ignore_index=True)
assert not a.duplicated(["seed","model","budget","policy","patient_id"]).any()
a.to_csv(O/"actions.csv",index=False)
r=pd.DataFrame(rows).groupby(["seed","model","budget","policy","group"])[["n","positive_n","auto_n","auto_positive_n"]].sum().reset_index()
r["auto_rate"]=r.auto_n/r.n;r["risk"]=r.auto_positive_n/r.auto_n.replace(0,np.nan)
r.to_csv(O/"seed_metrics.csv",index=False)
s=r.groupby(["model","budget","policy","group"]).agg(auto_n_mean=("auto_n","mean"),auto_n_min=("auto_n","min"),auto_n_max=("auto_n","max"),auto_rate=("auto_rate","mean"),risk_mean_served=("risk","mean"),served_seeds=("risk","count")).reset_index()
s.to_csv(O/"summary.csv",index=False)
protocol.update(status="COMPLETED",elapsed_seconds=time.time()-start)
(O/"protocol.json").write_text(json.dumps(protocol,indent=2),encoding="utf-8")
report="""# 拟合集锁定单候选规则对照
Material Passport: Origin Skill experiment-agent; Origin Mode run/validate; Origin Date 2026-09-17; Verification Status ANALYZED; Version locked_candidate_v1.
使用原来的25个外层划分和相同的拟合/校准比例，四个分数器。
每个拟合集内部三折OOF选每块一个阈值：至少10例，观察风险不超过目标的一半；随后重拟合分数器。校准集只认证，不再搜索。
G2三个块每块一个候选，置信预算0.05/3；原对照为21阈值×6范围。仅缺HPV弃权，没有回退。
该设计可能因内部OOF到重拟合模型的分数变化损失服务率；独立校准仍可拒绝不合格候选。
这是事后提出的探索性对照，重复使用既有患者，不能称为独立重复或临床验证。理论界仍依赖IID校准，分层CV不作严格独立认证。
3项校正的95%是每个固定分数器/划分/预算内的表述，不能同时覆盖所有预算、模型或重复。和126项方案的保证范围不同，不可宣称同等保证下免费增益。
无服务时风险未定义。降低多重比较代价不代表降低实际漏诊。
详细结果见summary.csv；风险是有服务种子的均值，不是跨重复独立样本的风险估计。
"""
(O/"report.md").write_text(report,encoding="utf-8")
print(s[(s.group=="missing_both")&(s.budget==.2)].to_string(index=False))
print("elapsed",time.time()-start)

