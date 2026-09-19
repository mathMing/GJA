"""Training-score quantiles, held-out descriptive risk, and hypothetical power."""
import json,time
import numpy as np,pandas as pd
from scipy.stats import binom
import v2_pipeline as v
from service_bottleneck_v18 import power
from report_numeric_v16 import md

ROOT=v.ROOT;OUT=ROOT/'results/low_risk_ranking_v19';PREV=ROOT/'results/temporal_sensitivity_v17'

def total_power(n,s,p,alpha):
 accepted=np.arange(n+1);conditional=power(accepted,p,alpha,.05/3);conditional[0]=0
 return float(np.dot(binom.pmf(accepted,n,s),conditional))

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 if (OUT/'run.json').exists():raise RuntimeError('Existing run: no overwrite')
 cache=PREV/'score_cache_LOCAL.csv.gz';previous=json.loads((PREV/'run.json').read_text());assert previous['status']=='COMPLETED'
 assert all(v.old.digest(p)==h for p,h in previous['input_hashes'].items())
 start=time.time();meta={'status':'RUNNING','model_fits':0,'cache_sha256':v.old.digest(cache),'protocol_sha256':v.old.digest(ROOT/'reports/v19_low_risk_ranking_protocol.md'),'code_sha256':v.old.digest(__file__)};v.dump(OUT/'run.json',meta)
 try:
  c=pd.read_csv(cache,dtype={'patient_id':str});rows=[];actions=[];keys=['cohort','split','seed','fold','model','configuration']
  for key,z in c.groupby(keys):
   base=dict(zip(keys,key));assert len(z)==z.patient_id.nunique()
   for group in ['complete','missing_both']:
    f=z[z.role.eq('fit_oof')&z.mask_group.eq(group)];t=z[z.role.eq('test')&z.mask_group.eq(group)];assert len(f)>0
    for q in [.1,.25,.5]:
     threshold=float(np.quantile(f.score,q,method='linear'));accept=t.score<=threshold
     n=len(t);pos=int(t.true_label.sum());na=int(accept.sum());k=int(t.loc[accept,'true_label'].sum())
     item={**base,'mask_group':group,'target_fraction':q}
     rows.append({**item,'fit_group_n':len(f),'threshold':threshold,'n':n,'positive_n':pos,'auto_n':na,'auto_positive_n':k,'service':na/n if n else np.nan,'risk':k/na if na else np.nan,'miss':k/pos if pos else np.nan,'certificate_valid':False})
     actions.append(t[['patient_id','true_label','score']].assign(**item,threshold=threshold,diagnostic_accept=accept.to_numpy()))
  r=pd.DataFrame(rows);assert len(r)==1872
  a=pd.concat(actions,ignore_index=True);assert not a.duplicated(keys+['mask_group','target_fraction','patient_id']).any()
  grouped=a.groupby(keys+['mask_group','target_fraction']).agg(n=('true_label','size'),positive_n=('true_label','sum'),auto_n=('diagnostic_accept','sum'))
  check=r.merge(grouped,on=keys+['mask_group','target_fraction'],suffixes=('_r','_a'),validate='one_to_one')
  assert len(check)==len(r)
  for col in ['n','positive_n','auto_n']:assert check[col+'_r'].eq(check[col+'_a']).all()
  r.to_csv(OUT/'fold_metrics.csv',index=False);a.to_csv(OUT/'diagnostic_actions_LOCAL.csv.gz',index=False,compression='gzip')
  sk=['cohort','seed','model','configuration','mask_group','target_fraction']
  s=r[r.split.eq('repeated')].groupby(sk)[['n','positive_n','auto_n','auto_positive_n']].sum().reset_index()
  s['baseline_prevalence']=s.positive_n/s.n;s['service']=s.auto_n/s.n;s['risk']=s.auto_positive_n/s.auto_n.replace(0,np.nan);s['miss']=s.auto_positive_n/s.positive_n.replace(0,np.nan)
  s['risk_reduction_pp']=100*(s.baseline_prevalence-s.risk);s.to_csv(OUT/'seed_metrics.csv',index=False)
  summary=s.groupby([x for x in sk if x!='seed']).agg(nonempty_seeds=('auto_n',lambda x:int((x>0).sum())),baseline_prevalence=('baseline_prevalence','first'),mean_service=('service','mean'),mean_risk=('risk','mean'),risk_min=('risk','min'),risk_max=('risk','max'),mean_reduction_pp=('risk_reduction_pp','mean'),lower_risk_seeds=('risk_reduction_pp',lambda x:int((x>0).sum()))).reset_index()
  summary.to_csv(OUT/'ranking_summary.csv',index=False)
  ps=[]
  for alpha in [.05,.1,.2]:
   for factor in [.25,.5]:
    for fraction in [.1,.25,.5]:
     for n in [100,250,500,1000,2000,5000]:ps.append({'risk_budget':alpha,'assumed_true_risk':alpha*factor,'assumed_service':fraction,'total_subgroup_calibration_n':n,'certification_probability':total_power(n,fraction,alpha*factor,alpha)})
  p=pd.DataFrame(ps);p.to_csv(OUT/'hypothetical_total_calibration_power.csv',index=False)
  rng=np.random.default_rng(190919);samples=rng.multinomial(500,[.025,.225,.75],size=20000);na=samples[:,:2].sum(1);k=samples[:,0];observed=float(((na>0)&(binom.cdf(k,na,.2)<=.05/3)).mean());exact=total_power(500,.25,.1,.2)
  assert abs(observed-exact)<.02
  assert r.loc[r.auto_n.eq(0),'risk'].isna().all() and v.old.digest(cache)==meta['cache_sha256']
  assert all(v.old.digest(p)==h for p,h in previous['input_hashes'].items())
  meta.update(status='COMPLETED',fold_rows=len(r),action_rows=len(a),hypothetical_cells=len(p),power_check_exact=exact,power_check_monte_carlo=observed,elapsed_seconds=time.time()-start);v.dump(OUT/'run.json',meta)
  view=summary[summary.mask_group.eq('missing_both')&summary.target_fraction.eq(.25)]
  primary=r[r.split.eq('primary')&r.mask_group.eq('missing_both')][['cohort','model','configuration','target_fraction','n','positive_n','auto_n','auto_positive_n','risk','service']]
  report=f'''# V19：AI能否把低风险病例排到前面？

Origin Skill: experiment-agent；Mode: run/validate；Verification Status: ANALYZED。2026-09-19。

## 本轮设计
固定V17评分，使用拟合OOF评分的10%、25%、50%分位数作为预先规定阈值，在隔离的测试角色评价。比较完整组与双缺失组，所有队列、模型、字段配置保留。共{len(r)}个折级结果，无新增拟合。这些是排序诊断，不经过校准认证，没有有效风险证书，不是可部署动作。目标分位数与实际测试服务率不同，同分全部接纳。

## 双缺失：目标接纳最低25%评分病例
下表五种子合并五折后取均值；risk是自动中的阳性比例，baseline是该组整体阳性率，mean_reduction_pp是两者差值的百分点。风险均值只在非空种子计算；同一患者被多个种子重复使用，不提供独立置信区间，也不按表格挑赢家。各配置对不同病例给分，跨配置均值不能解释为患者级因果获益。

{md(view)}

## 全部分位数与完整组对照
{md(summary)}

## 主划分双缺失结果（包括空服务）
{md(primary)}

## 样本量假设分析
与真实排序诊断分开：假设固定阈值在双缺失组内接纳概率为10%、25%、50%，接纳后真实风险为预算的1/4或1/2。对随机接纳人数精确积分，得到不同总双缺失校准人数下的认证概率。不是把现有测试风险当真实参数，也不是复制患者构造新数据。这里的总人数仅为双缺失校准组，并非全队列招募量；独立同分布与风险低于预算的假设必须成立。

以下展示20%预算、假设真实风险10%时结果，其余全部参数见CSV。每块delta=0.05/3；这仍是固定阈值单检验的功效，不代表多阈值选择后完整策略的保证。

{md(p[p.risk_budget.eq(.2)&p.assumed_true_risk.eq(.1)])}

精确求和单元通过独立多项分布数值核对：精确概率{exact:.6f}，20000次模拟比例{observed:.6f}。模拟仅验证代码，不增加临床证据。

## 如何解释
如果低评分区经验风险低于该组基准，说明排序有一定探索性信号，不能证明达到5%/10%/20%的真实风险目标；若部分划分反向或波动大，也必须保留。低风险排序与有能力认证是不同问题；前者即使存在，当前校准容量仍可能使服务为零。

后续不要利用本轮测试标签重选分位数。提升评分应在独立训练与选择数据中进行；新增校准数据的设计需同时考虑可接纳比例、目标风险与置信要求。现阶段不能从假设表直接决定实际招募人数。既有临床时序/字段核实与外部验证仍未完成。

11项检查覆盖总体/子群、生态推断、手术选择、检查路径、分母与基准率、回归均值、零服务保留、全配置比较、多轮探索、关联非因果、时序未裁定。已反复探索同一队列，结果不是独立验证。

协议reports/v19_low_risk_ranking_protocol.md；执行src/low_risk_ranking_v19.py；结果results/low_risk_ranking_v19。已有run.json拒绝覆盖，患者明细仅本地。
'''
  (ROOT/'reports/Better1_v19_低风险排序与校准规模.md').write_text(report,encoding='utf-8')
  print(view.to_string(index=False));print(json.dumps(meta))
 except Exception as exc:meta.update(status='FAILED',error=repr(exc));v.dump(OUT/'run.json',meta);raise

if __name__=='__main__':main()
