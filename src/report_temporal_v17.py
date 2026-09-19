"""Descriptive cohort sensitivity and the service cost of finer protection."""
import json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import v2_pipeline as v
from report_numeric_v16 import md

ROOT=v.ROOT;OUT=ROOT/'results/temporal_sensitivity_v17'

def main():
 run=json.loads((OUT/'run.json').read_text());assert run['status']=='COMPLETED' and run['model_fits']==832
 s=pd.read_csv(OUT/'seed_metrics.csv');r=pd.read_csv(OUT/'fold_metrics.csv')
 keys=['cohort','seed','model','configuration','risk_budget']
 all_s=s[s.mask_group.eq('all')];g0=all_s[all_s.policy.eq('G0_FST')];g2=all_s[all_s.policy.eq('G2_FST')]
 paired=g0.merge(g2,on=keys,suffixes=('_G0','_G2'),validate='one_to_one')
 assert paired.n_G0.eq(paired.n_G2).all()
 paired['service_cost_pp']=100*(paired.service_G0-paired.service_G2)
 paired.to_csv(OUT/'service_cost_by_seed.csv',index=False)
 cost=paired.groupby(['cohort','model','configuration','risk_budget']).agg(G0_service=('service_G0','mean'),G2_service=('service_G2','mean'),cost_pp=('service_cost_pp','mean'),cost_min_pp=('service_cost_pp','min'),cost_max_pp=('service_cost_pp','max')).reset_index()
 cost.to_csv(OUT/'service_cost_summary.csv',index=False)
 k=keys+['policy'];j=all_s.merge(s[s.mask_group.eq('missing_both')],on=k,suffixes=('_all','_double'),validate='one_to_one')
 j['pattern']=(j.risk_all<=j.risk_budget)&(j.risk_double>j.risk_budget)
 summary=j.groupby(['cohort','model','configuration','policy','risk_budget']).agg(pattern_seeds=('pattern','sum'),nonempty_seeds=('auto_n_double',lambda x:int((x>0).sum())),double_auto_mean=('auto_n_double','mean'),double_positive_mean=('auto_positive_n_double','mean'),double_service_mean=('service_double','mean')).reset_index()
 summary.to_csv(OUT/'subgroup_summary.csv',index=False)
 primary=r[r.split.eq('primary')&r.risk_budget.eq(.2)&r.mask_group.isin(['all','missing_both'])][['cohort','model','configuration','policy','mask_group','n','positive_n','auto_n','auto_positive_n','risk','miss','service']]
 primary.to_csv(OUT/'primary_budget20.csv',index=False)
 double=r[r.policy.eq('G2_FST')&r.mask_group.eq('missing_both')]
 zero=int(double.auto_n.eq(0).sum());total=len(double)
 view=summary[summary.risk_budget.eq(.2)&summary.policy.eq('G0_FST')]
 fig,axes=plt.subplots(1,2,figsize=(12,4.5),sharey=True)
 for ax,model in zip(axes,['lr','gbdt']):
  for cfg in ['preop_original','without_mri_nodes']:
   for cohort in ['all_dates','exclude_after','strict_before']:
    q=cost[cost.model.eq(model)&cost.configuration.eq(cfg)&cost.cohort.eq(cohort)].sort_values('risk_budget')
    ax.plot(q.risk_budget, q.cost_pp,marker='o',label=cohort+' / '+cfg)
  ax.set_title(model);ax.set_xlabel('Risk budget');ax.axhline(0,color='gray',linewidth=.6);ax.grid(alpha=.2)
 axes[0].set_ylabel('Service G0 minus G2 (percentage points)');axes[1].legend(fontsize=6)
 fig.tight_layout();fig.savefig(OUT/'service_cost_curve.png',dpi=180);plt.close(fig)
 report=f'''# V17：影像日期敏感性与安全保护的服务率代价

Origin Skill: experiment-agent；Mode: run/validate；Verification Status: ANALYZED。2026-09-19。

## 研究主线
主线已确定为“AI风险评分在检查缺失条件下的安全边界与服务率代价”。AI评分器提供风险排序；认证协议决定哪些病例可以自动判阴。当前目标是测量能力与代价，不预设分块能同时提高安全与服务。

## 本轮完成什么
完成832次新增拟合；原队列{run['baseline_patient_checks']}行患者动作与阈值复现V15。两种模型、两种字段配置、三个日期口径、三个预算均保留。原始数据、派生数据、临床待核实表未修改。日期排除不是临床裁定，也没有解决保留记录中其他字段的时序与来源问题。

{md(pd.read_csv(OUT/'cohort_summary.csv'))}

## 双缺失安全边界
G2双缺失在{total}个折级配置中，{zero}个配置自动人数为0。零自动时风险未定义，不写成风险为零；这不能证明分块已在保留服务的同时修复子群风险。

严格仅保留影像日期早于手术的队列中，20%预算下，原术前LR出现总体合格而双缺失经验风险超预算的频次为4/5，梯度提升树为2/5；删除mri_nodes后两者分别4/5和3/5。最后一项LR配置仅4/5种子有双缺失自动病例，应同时阅读非空种子数。原773例中对应频次为3/5、3/5、5/5、3/5。信号没有随日期排除一致消失，但不能称为每个模型、每个划分都稳定发生。

20%预算下，全部队列原术前LR总体自动率由G0的41.71%降为G2的15.50%，平均代价26.21个百分点；梯度提升树由36.82%降为8.62%，代价28.20个百分点。严格日期队列分别为LR 34.10%降至6.65%、梯度提升树32.89%降至5.17%，平均代价27.46和27.72个百分点。各种子存在较大波动，个别差值为负；这些均值不代表每次运行必然付出相同代价。

下表为20%预算下的G0结果。pattern_seeds是五个重复CV种子中“总体经验风险≤预算、双缺失经验风险>预算”的次数，另列双缺失非空种子数。它是同一队列的描述性重复，不是五次独立验证，也不是已经证实真实风险超标。

{md(view)}

## 服务率代价
服务率=自动判阴人数/全部患者数。同一队列、模型、字段和预算下，代价=100×(G0服务率−G2服务率)，单位为百分点。例如40%降为10%表示少服务30个百分点；不是减少30%的患者相对比例。负差值也保留。

G0保护总体、G2保护预设子群，二者保证对象不同。这个差值衡量采取更细保护策略的实际服务变化，不是相同保证下的算法净性能差。以下为五种子均值及描述性极值，不能解释为独立置信区间；服务率列采用0–1比例。

{md(cost[cost.risk_budget.eq(.2)])}

## 更严格预算：完整结果
{md(cost[cost.risk_budget.lt(.2)])}

## 主划分20%预算：分母明确
风险risk=自动中的阳性数/自动人数；miss=自动中的阳性数/全部阳性数。无自动人数时risk为NA。

{md(primary)}

## 解释和继续工作的边界
跨日期口径改变了队列人群、训练与校准样本量，观察差异不能单独归因于影像日期；外层角色保留也不意味着内层OOF划分完全相同。不同模型/预算/字段组合不具有统一的事后最优选择保证。G0的证书不能转移为双缺失证书，重复分层CV的经验风险也不等于独立同分布的新队列风险。

本轮用于判断信号对日期口径是否敏感、认证服务是否脆弱。结合此前字段删除与数值置缺实验，应保留不一致和零服务结果，不挑选最有利配置。当前更适合继续做安全边界与服务代价分析；尚无证据称为临床可用的子群自动判阴方案。

后续优先分离评分质量、校准样本量与多重选择带来的代价；真实临床核实和独立外部验证仍未完成。本队列已反复探索，不能把本轮作为新的独立验证。11项解释检查覆盖总体/子群、生态推断、手术选择、检查路径、分母基准率、回归均值、未服务病例、全配置比较、多轮探索、非因果、时序未裁定。

## 文件与复现
协议：reports/v17_temporal_sensitivity_protocol.md。执行：src/temporal_sensitivity_v17.py；报告：src/report_temporal_v17.py。results/temporal_sensitivity_v17中保存队列人数、每折与每种子指标、服务率代价、证书、患者动作及评分缓存。LOCAL文件只在本地审计。run.json包含输入及代码哈希；已有运行时拒绝覆盖。
'''
 path=ROOT/'reports/Better1_v17_影像日期与服务率代价.md';path.write_text(report,encoding='utf-8')
 v.dump(OUT/'report_checks.json',{'status':'PASS','model_fits':run['model_fits'],'baseline_patient_checks':run['baseline_patient_checks'],'zero_double_cells':zero,'total_double_cells':total,'seed_metrics_sha256':v.old.digest(OUT/'seed_metrics.csv')})
 print(view.to_string(index=False));print(cost[cost.risk_budget.eq(.2)].to_string(index=False));print('G2 double zero',zero,'/',total);print(path)

if __name__=='__main__':main()
