"""Exact conditional certification power and synthetic score-family oracle."""
from pathlib import Path
import hashlib,json,time
import numpy as np
import pandas as pd
from scipy.stats import binom,beta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from synthetic_boundary_v2 import distribution

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/boundary_v3';OUT.mkdir(parents=True,exist_ok=True)
OLD=ROOT/'results/selective_risk_v2'
def md(df):
 return '| '+' | '.join(df.columns)+' |\n|'+'|'.join(['---']*len(df.columns))+'|\n'+'\n'.join('| '+' | '.join('NA' if pd.isna(v) else f'{v:.4g}' if isinstance(v,(float,np.floating)) else str(v) for v in row)+' |' for row in df.itertuples(index=False,name=None))
def main():
 start=time.time();rows=[];planning=[]
 n=np.arange(1,100001)
 for candidates in [126,3]:
  delta=.05/candidates
  for alpha in [.05,.1,.2]:
   # Largest allowed error count; ppf is adjusted to satisfy CDF <= delta.
   k=binom.ppf(delta,n,alpha).astype(int)
   k-=binom.cdf(k,n,alpha)>delta
   assert np.all(binom.cdf(k,n,alpha)<=delta+1e-12)
   assert np.all(binom.cdf(k+1,n,alpha)>delta-1e-12)
   zero=int(np.ceil(np.log(delta)/np.log1p(-alpha)))
   assert 1-delta**(1/zero)<=alpha+1e-12
   assert zero==1 or 1-delta**(1/(zero-1))>alpha
   for fraction in [0,.25,.5,.75,1.]:
    p=alpha*fraction;power=binom.cdf(k,n,p)
    result={'candidates':candidates,'budget':alpha,'true_risk':p,'risk_fraction':fraction,'zero_error_min_accepted':zero}
    for target in [.8,.9]:
     hit=np.flatnonzero(power>=target)
     result[f'n_for_power_{target}']=int(n[hit[0]]) if len(hit) else np.nan
    rows.append(result)
    if fraction==.5:
     for service in [.1,.3,.5]:
      planning.append({**result,'assumed_double_service':service,'assumed_double_fraction':113/773,'calibration_fraction':.25,'expected_total_for_90pct':result['n_for_power_0.9']/(.25*(113/773)*service)})
 powerdf=pd.DataFrame(rows);powerdf.to_csv(OUT/'certification_power.csv',index=False)
 pd.DataFrame(planning).to_csv(OUT/'hypothetical_recruitment.csv',index=False)
 source=pd.read_csv(OLD/'e2_boundary_summary.csv');oracle=[]
 for row in source[source.policy=='G2'].itertuples():
  pop=distribution(row.mechanism,row.double_fraction,row.target_complete_auc,row.missingness_strength)
  counts=pop[2].cumsum(-1);mass=counts.sum(0);risk=counts[1]/mass
  valid=risk<=row.budget+1e-12
  idx=np.flatnonzero(valid)
  ix=idx[np.argmax(mass[idx])] if len(idx) else -1
  service=mass[ix]/pop[2].sum() if ix>=0 else 0.
  if ix>=0:
   assert risk[ix]<=row.budget+1e-12
   assert np.all(mass[valid]<=mass[ix]+1e-12)
  category='oracle_below_2pct' if service<.02 else 'certification_below_2pct' if row.double_auto_rate<.02 else 'both_at_least_2pct'
  oracle.append({'case_id':row.case_id,'mechanism':row.mechanism,'n_total':row.n_total,'budget':row.budget,'target_complete_auc':row.target_complete_auc,'double_fraction':row.double_fraction,'missingness_strength':row.missingness_strength,'oracle_threshold_bin':ix,'oracle_double_service':service,'oracle_double_risk':risk[ix] if ix>=0 else np.nan,'certified_double_service_mean':row.double_auto_rate,'raw_service_gap':service-row.double_auto_rate,'any_deployed_scope_violation_probability':row.deployed_budget_violation_probability,'boundary_category':category})
 od=pd.DataFrame(oracle);od.to_csv(OUT/'oracle_boundary.csv',index=False)
 summary=od.groupby(['mechanism','boundary_category']).size().rename('config_budget_count').reset_index()
 summary.to_csv(OUT/'boundary_counts.csv',index=False)
 fig,ax=plt.subplots(figsize=(8,4))
 for (c,a),group in powerdf[powerdf.risk_fraction<1].groupby(['candidates','budget']):
  ax.plot(group.risk_fraction,group['n_for_power_0.9'],marker='o',label=f'm={c}, budget={a}')
 ax.set_yscale('log');ax.set_xlabel('True accepted risk / risk budget');ax.set_ylabel('Accepted calibration n for 90% certification')
 ax.legend(fontsize=8,ncol=2);fig.tight_layout();fig.savefig(OUT/'power_boundary.png',dpi=180);plt.close(fig)
 z=powerdf[(powerdf.risk_fraction==.5)][['candidates','budget','zero_error_min_accepted','n_for_power_0.8','n_for_power_0.9']]
 report='''# 下一轮边界实验：样本量不足，还是分数不够好？

Material Passport：Origin Skill=experiment-agent；Mode=run/validate；Version=boundary_v3；Verification Status=ANALYZED。分析是对已有结果的探索性跟进，运行前协议见 v3_boundary_protocol.md。

## 1. 认证需要的不是“总患者数”，而是“规则接收的校准患者数”

下表假设冻结规则的真实接收风险恰为预算的一半，计算精确二项认证概率。m=126 为现有多候选同时校正；m=3 为独立训练数据提前锁定三个组规则后，仅认证三个规则。m=3 不允许看完校准结果再从很多阈值中挑选。计算上限 100,000 接收样本，超出记 NA。

{power}

零错误最低人数只是理想极限，不是有较高成功率的样本量设计。真实风险越接近预算，认证越困难；风险正好等于预算时，认证通过率受单次显著性水平限制，不能通过无限增样把通过率提高到 90%。这个计算针对固定规则，不能直接作为自适应 G2 的整体功效。

![功效边界](../results/boundary_v3/power_boundary.png)

## 2. 已知真实总体风险时，分数本身能否提供服务？

对全部 832 个合成配置和三个预算计算 oracle：知道合成总体真实风险，在同一组 21 个阈值内选择最大可行服务率。它是该分数/阈值族的总体可达值，不是所有机器学习算法的上界，不可用于实际部署。

{counts}

每行计数是参数网格的配置—预算数，不是随机人群比例。oracle_below_2pct 表示即使知道总体风险，当前分数族也不足以提供 2% 子群服务；certification_below_2pct 表示 oracle 可达 2%，但现有有限校准认证的平均服务未到 2%；both_at_least_2pct 表示两者均达到研究用标记。2% 不代表临床有用阈值。

oracle 与认证服务率差包括校准样本、校正和规则选择成本，不能全部归因为样本量。认证存在小概率违规，个别原始服务率差可为负；不能把违规获得的服务当成超过 oracle 的方法优势。CSV 同时保留总体违规频率，避免只报服务。

## 3. 这对目前 773 人意味着什么？

双缺失总人数仅 113；主划分双缺失校准人数 34。该人数还不是被规则自动接收的人数，实际用于认证的接收人数通常更少。原有 20% 预算、126 候选的零错误最低接收数为 36，故主划分不可能通过；这条数学障碍不涉及网络好坏。

但是增加数据也不能保证成功：如果子群低分区域本身风险仍高于预算，就需改进分数、改变输入信息，或放弃该自动服务目标。合成 oracle 诊断支持区分这两类问题，但不能把合成总体上界外推为真实患者上界。现有 OOF 数据不足以给出可靠的真实人口 oracle。

hypothetical_recruitment.csv 给出校准比例 25%、双缺失比例 113/773、假设子群服务率 10%/30%/50% 时的期望总人数换算。它假设固定规则和风险水平稳定，未计随机入组人数波动、失访、中心漂移或训练改善，因此不是正式招募建议。

## 4. 下一步的可执行取舍

优先比较“训练集预锁定阈值+独立认证”与现有全候选校正，使用相同真实患者划分，并保留完全弃权结果；不能直接把本次较小 m 的功效曲线当成已经实现的真实性能提升。此前合成 G2_locked 仅为一个预锁定策略，训练选取质量也会影响其实际服务。

若在相同分数和校准数据下减少合法选择成本仍无法让双缺失组服务，则应保留失败边界作为结果；待临床数据疑点裁定后再决定是否投入新分数器或新增病例。不得把重复患者或 bootstrap 复制当成新增独立校准患者。

偏差检查沿用 v2 全部 11 项：分层防 Simpson 掩盖；不作生态因果推断；限制手术队列选择/Berkson 偏差；检查路径 collider；基准率差异；回归均值；幸存/入组选择；多重搜索；分析分叉；关联与因果区分；反向因果/时间窗。本轮全网格报告避免只展示正结果，假设性样本量换算不消除上述偏差。

验证：精确二项拒绝边界、零错误人数公式、oracle 可行性与候选最大性均通过程序检查。原始临床数据和 v2 输出只读。本轮没有新增临床效果证据。
'''.format(power=md(z),counts=md(summary))
 (ROOT/'reports/Better1_v3_样本量与信息边界.md').write_text(report,encoding='utf-8')
 meta={'status':'COMPLETED','oracle_rows':len(od),'power_rows':len(powerdf),'elapsed_seconds':time.time()-start,'source_sha256':hashlib.sha256((OLD/'e2_boundary_summary.csv').read_bytes()).hexdigest(),'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'checks':'binomial rejection maximality; zero error minimum; oracle feasibility and maximality'}
 (OUT/'run.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
 print(z.to_string(index=False));print(summary.to_string(index=False));print(json.dumps(meta))
if __name__=='__main__':main()
