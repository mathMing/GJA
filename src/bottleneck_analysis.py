"""Post hoc bottleneck diagnosis; no new certification or threshold selection."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binom
O=Path(__file__).resolve().parents[1]/"results"/"ars_validated_v1"
s=pd.read_csv(O/"oof_scores.csv")
a=pd.read_csv(O/"three_way_actions.csv")
sp=pd.read_csv(O/"split_manifest.csv")
groups=s[["patient_id","mask_group"]].drop_duplicates()
cal=sp[sp.role=="calibration"].merge(groups,on="patient_id",validate="many_to_one")
counts=cal.groupby(["seed","fold","mask_group"]).size().rename("calibration_n").reset_index()
counts.to_csv(O/"bottleneck_calibration_counts.csv",index=False)
rows=[]
for family in [1,21,126]:
 for alpha in [.05,.1,.2]:
  delta=.05/family
  n=int(np.ceil(np.log(delta)/np.log(1-alpha)))
  for risk in [0,.02,.05,.1]:
   if risk>=alpha: continue
   # Probability of certifying ONE fixed candidate; not family selection power.
   grid=np.arange(1,5001)
   power=[]
   for nn in grid:
    kk=np.arange(nn+1)
    ok=binom.cdf(kk,nn,alpha)<=delta
    power.append(float(binom.pmf(kk[ok],nn,risk).sum()))
    if power[-1]>=.8: break
   rows.append(dict(family_n=family,budget=alpha,true_selected_risk=risk,zero_error_min_n=n,n_for_80pct_candidate_power=int(grid[len(power)-1]) if power[-1]>=.8 else np.nan))
pd.DataFrame(rows).to_csv(O/"bottleneck_sample_requirements.csv",index=False)
r=[]
for (seed,model,group),d in s.groupby(["seed","model","mask_group"]):
 for fraction in [.1,.2,.3,.5]:
  # Fixed rank fractions, diagnostic only; labels never choose fraction.
  t=d.sort_values(["score","patient_id"]).iloc[:max(1,int(np.ceil(len(d)*fraction)))]
  r.append(dict(seed=seed,model=model,mask_group=group,fraction=fraction,n=len(t),positives=int(t.true_label.sum()),observed_risk=float(t.true_label.mean())))
r=pd.DataFrame(r);r.to_csv(O/"bottleneck_low_score_risk.csv",index=False)
summary=r.groupby(["model","mask_group","fraction"]).agg(risk_mean=("observed_risk","mean"),risk_min=("observed_risk","min"),risk_max=("observed_risk","max"),n_per_seed=("n","first")).reset_index()
summary.to_csv(O/"bottleneck_low_score_summary.csv",index=False)
cc=counts.groupby("mask_group").calibration_n.agg(["min","median","max"])
req=pd.DataFrame(rows)
out=summary[(summary.mask_group=="missing_both") & (summary.fraction.isin([.2,.5]))]
report=f"""# 下一步诊断：样本量还是分数器？
这是基于已完成OOF预测的事后诊断，不是新独立验证，也不用于重新选择阈值。
## 认证样本量
当前每折校准人数：
{cc.to_string()}
在126项同时校正、零个阳性的最理想情况下，5%/10%/20%预算所需的自动判阴校准样本至少为：
{req[(req.family_n==126)&(req.true_selected_risk==0)][['budget','zero_error_min_n']].to_string(index=False)}
这些是被规则选中的校准人数，不是总入组人数。非零错误需要更多样本。
bottleneck_sample_requirements.csv另给固定候选在指定真实风险下达到80%认证概率的样本量；这是二项模型计算，不是实测招募效果。
因此即使完美分数器把校准双缺失患者全部判阴，全部25个外层折也无法认证20%或更低风险预算。当前G2双缺失无服务是本设计的结构性结果，不能用于否定idea或分数器改善潜力。
## 分数器诊断
按每个种子OOF分数，在双缺失组取固定最低20%或50%：
{out.to_string(index=False)}
这里只是描述风险排序；跨折分数不一定完全可比，重复种子不是独立样本；低分组观察风险低不等于可认证安全。
## 下一步决策
先把校准规模不足作为已量化的瓶颈汇报。降低候选数量可在事先登记的新实验中降低认证代价，但不能在本批测试结果上挑最有利设置后声称独立验证。
同时参考低分病例实际风险：若低分仍高风险，单纯扩大校准集也不够；需要更好术前信号或MRI。
优先考虑在拟合集内部冻结一个主分数器和少量阈值，使用新的独立、标签无关校准/测试划分；代价是保证范围变窄，应如实声明。
原有5%/10%/20%预算不因结果差而上调。当前不直接启动大MRI训练或新增外部数据申请。
"""
(O/"bottleneck_report.md").write_text(report,encoding="utf-8")
print(report)
