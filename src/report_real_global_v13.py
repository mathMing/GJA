"""Exploratory true-cohort global/subgroup evidence report, without pooling seeds."""
import json,hashlib
from pathlib import Path
import pandas as pd,numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/real_global_subgroup_v13'
def md(df):
 return '| '+' | '.join(df.columns)+' |\n|'+'|'.join(['---']*len(df.columns))+'|\n'+'\n'.join('| '+' | '.join('NA' if pd.isna(v) else f'{v:.4f}' if isinstance(v,(float,np.floating)) else str(v) for v in row)+' |' for row in df.itertuples(index=False,name=None))
def main():
 run=json.loads((OUT/'run.json').read_text());assert run['status']=='COMPLETED' and run['g2_patient_checks']==48708
 r=pd.read_csv(OUT/'fold_metrics.csv');s=pd.read_csv(OUT/'seed_metrics.csv');j=pd.read_csv(OUT/'repeated_joint_pattern.csv');p=pd.read_csv(OUT/'primary_joint_pattern.csv')
 assert len(r)==4680
 summary=j.groupby(['model','features','policy','risk_budget']).agg(observed_pattern_seeds=('observed_global_pass_double_exceeds','sum'),double_service_seeds=('auto_n_double',lambda x:int((x>0).sum())),double_auto_mean=('auto_n_double','mean'),double_auto_min=('auto_n_double','min'),double_auto_max=('auto_n_double','max')).reset_index();summary.to_csv(OUT/'pattern_summary.csv',index=False)
 main=r[r.split.eq('primary')&r.risk_budget.eq(.2)&r.mask_group.isin(['all','missing_both'])][['model','features','policy','mask_group','n','positive_n','auto_n','auto_positive_n','service','risk','miss','test_CP95_low_primary_only','test_CP95_high_primary_only']]
 seed=s[s.risk_budget.eq(.2)&s.features.eq('preop')&s.policy.eq('G0_FST')&s.mask_group.eq('missing_both')][['seed','model','auto_n','auto_positive_n','service','risk','miss']]
 one=main[main.model.eq('lr')&main.features.eq('preop')&main.policy.eq('G0_FST')&main.mask_group.eq('missing_both')].iloc[0]
 totals=s[s.risk_budget.eq(.2)&s.mask_group.eq('all')].groupby(['model','features','policy']).agg(mean_service=('service','mean'),min_service=('service','min'),max_service=('service','max')).reset_index();totals.to_csv(OUT/'overall_service_summary.csv',index=False)
 report='''# V13：真实全局规则是否掩盖双缺失组风险？

## Material Passport
Origin Skill: experiment-agent；Mode: validate；Version: real_global_subgroup_v13；Verification Status: ANALYZED。
2026-09-18。复用冻结V6预测，零新增拟合；104个分数器—划分配置、146124行患者动作。48708项G2患者动作与V6完全一致；原始/派生输入与缓存哈希核对通过。

## 1. 当前结论
**真实队列中观察到了“总体经验风险低于预算、双缺失经验风险高于预算”的信号，但尚不能确认稳定的真实子群风险超标。**

主随机划分、术前LR、G0固定序列、20%预算：总体85人自动判阴，其中8人阳性，经验风险9.41%；双缺失组21人自动判阴，其中5人阳性，经验风险23.81%。双缺失风险的逐项95%二项区间为[{lo:.2%},{hi:.2%}]，包含20%，故不能把点估计超标当成已经证明真实风险超标。整体与子群分母不同。

主划分的术前梯度提升树对应总体6/84=7.14%、双缺失2/13=15.38%，没有出现同一超标模式。主划分全部模型/预算/两种全局方法中，上述模式只出现1个配置，不能单挑它作强结论。

五种子重复CV中，20%预算的G0固定序列在术前LR与术前梯度提升树均有3/5种子观察到该模式。但它们重用同一773人，不是5个独立队列；每种子还是多个外层折规则的混合。该频次只反映划分敏感性，不是独立复制成功率。

G2固定序列在对应双缺失组仍全部无服务；这表明当前流程以放弃该组服务来避免自动通道风险暴露，不能写成在保留有意义服务的同时修复了该组风险。无人接收时风险为NA。

## 2. 主划分20%预算：完整计数与分母
主测试共194人，双缺失36人。service=自动人数/n；risk=自动中阳性/自动人数；miss=自动中阳性/该组全部阳性。CP区间仅为逐项测试风险诊断，不是校准证书，也不作跨配置联合声明。

{primary}

## 3. 五种子：所有全局规则配置
observed_pattern_seeds：同一种子整体经验risk<=预算、双缺失risk>预算；分母恒为5，零服务不算通过或超标。double_service_seeds表明有几个种子确实接收了双缺失病例。所有预算保留，不合并患者计数做更窄区间。

{patterns}

## 4. 术前分数器的双缺失逐种子计数（20%预算、G0固定序列）
每种子双缺失总人数113；重复CV分层且重用患者，以下不套用独立二项精确区间。

{seeds}

## 5. 总体服务对照（20%预算）
以下为每种子覆盖773人的平均及范围。G0只保证总体，G2要求各认证子群；保护要求不同，差值不代表相同条件下的纯算法优势。五种子范围不是置信区间。

{total}

## 6. 三个规则与证书解释
G0_FST：全体拟合内OOF预测排序，独立全体校准delta=0.05，首次失败停止。G0_Bonf21：21个固定候选按0.05/21同时校正。G2_FST：分别认证完整、仅缺TCT、双缺失，delta=0.05/3；仅缺HPV弃权，与V6一致。
固定序列risk_bound保存认证预算，Bonferroni保存同时CP界。G0证书只属于all，绝不声称原始子群保证。拟合OOF分数来自内部折模型，与最终拟合分数有差异，排序效率可能受影响。规则只用训练OOF与独立校准；测试只评价。多个分数器/预算/规则不具备未经校正的联合认证声明。

## 7. 对idea的修正与下一步
相比只有粗阳性比例，现在多了一项与决策直接相关的真实观察：某些全局自动通道存在子群风险偏高的点估计。它增强了继续审视子群风险的动机，但没有满足“稳定失控且可通过分块保留服务地修复”的原始目标。
当前最合适的表述是：现有全局规则有时能提供总体服务，但其子群表现不确定；独立子群认证在小队列中可能退化为无服务。这是风险—服务能力的边界证据，不是临床有效性证明。
不因本轮观察而改标签、补猜字段、增选阈值或宣称新方法。后续真实分析应先处理临床来源/时序核实，并使用独立数据确认，不能将同队列反复重划分升级为外部证据。临床字段争议尚未消除，本轮涉及术前影像字段的结论仍受这些限制。

## 8. 偏差检查与复现
11/11：Simpson（同规则同划分总体/原组同时报告）；生态谬误（不作个体因果）；Berkson（手术队列限制）；collider（检查路径关联非因果）；基准率（报告阳性与各风险分母）；回归均值（全划分而非单挑主结果）；幸存者偏差（零服务明确NA）；多重搜索（全方法预算，CP仅逐项）；分析分叉（明确探索性后续）；关联与因果（无补查疗效）；反向因果（时序核实仍未决）。
协议reports/v13_real_global_subgroup_protocol.md；执行src/real_global_subgroup_v13.py；报告src/report_real_global_v13.py。结果results/real_global_subgroup_v13；患者级文件actions_LOCAL.csv.gz只用于本地审计。没有新训练，也没有修改原始临床数据。
'''.format(lo=one.test_CP95_low_primary_only,hi=one.test_CP95_high_primary_only,primary=md(main),patterns=md(summary),seeds=md(seed),total=md(totals))
 path=ROOT/'reports/Better1_v13_真实全局与双缺失风险诊断.md';path.write_text(report,encoding='utf-8')
 (OUT/'report_checks.json').write_text(json.dumps({'status':'PASS','fold_rows':len(r),'g2_patient_checks':48708,'model_fits':0,'primary_observed_patterns':int(p.observed_global_pass_double_exceeds.sum()),'source_sha256':hashlib.sha256((OUT/'fold_metrics.csv').read_bytes()).hexdigest()},indent=2),encoding='utf-8')
 print('PRIMARY DOUBLE LR CP',one.test_CP95_low_primary_only,one.test_CP95_high_primary_only);print(seed.to_string(index=False));print(path)
if __name__=='__main__':main()
