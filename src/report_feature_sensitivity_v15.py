"""All-budget paired feature-removal diagnostics on frozen clinical splits."""
import json,hashlib
from pathlib import Path
import pandas as pd,numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/feature_sensitivity_v15'
def md(df):
 return '| '+' | '.join(df.columns)+' |\n|'+'|'.join(['---']*len(df.columns))+'|\n'+'\n'.join('| '+' | '.join('NA' if pd.isna(v) else f'{v:.4f}' if isinstance(v,(float,np.floating)) else str(v) for v in row)+' |' for row in df.itertuples(index=False,name=None))
def main():
 run=json.loads((OUT/'run.json').read_text());assert run['status']=='COMPLETED' and run['model_fits']==416
 r=pd.read_csv(OUT/'fold_metrics.csv');s=pd.read_csv(OUT/'seed_metrics.csv');assert len(r)==4680
 keys=['seed','model','configuration','policy','risk_budget']
 joint=s[s.mask_group.eq('all')].merge(s[s.mask_group.eq('missing_both')],on=keys,suffixes=('_all','_double'),validate='one_to_one');joint['observed_pattern']=(joint.risk_all<=joint.risk_budget)&(joint.risk_double>joint.risk_budget);joint.to_csv(OUT/'joint_patterns.csv',index=False)
 summary=joint.groupby(['model','configuration','policy','risk_budget']).agg(pattern_seeds=('observed_pattern','sum'),double_nonempty_seeds=('auto_n_double',lambda x:int((x>0).sum())),double_auto_mean=('auto_n_double','mean'),double_positive_mean=('auto_positive_n_double','mean'),double_service_mean=('service_double','mean'),all_service_mean=('service_all','mean')).reset_index();summary.to_csv(OUT/'summary.csv',index=False)
 effects=[]
 for (model,policy,budget,group),z in s.groupby(['model','policy','risk_budget','mask_group']):
  p=z.pivot(index='seed',columns='configuration',values='service');assert len(p)==5 and p.notna().all().all()
  for cfg in ['without_mri_nodes','without_imaging_tables']:
   x=100*(p[cfg]-p.preop_original);effects.append(dict(model=model,policy=policy,risk_budget=budget,mask_group=group,configuration=cfg,service_delta_mean_pp=x.mean(),seed_min_pp=x.min(),seed_max_pp=x.max()))
 e=pd.DataFrame(effects);e.to_csv(OUT/'paired_service_deltas.csv',index=False)
 primary=r[r.split.eq('primary')&r.risk_budget.eq(.2)&r.mask_group.isin(['all','missing_both'])][['model','configuration','policy','mask_group','n','auto_n','auto_positive_n','service','risk','miss']]
 report='''# V15：真实结果对争议影像字段的敏感性

## Material Passport
Origin Skill: experiment-agent；Mode: run/validate；Version: feature_sensitivity_v15；Verification Status: ANALYZED。
2026-09-19。保留773例，26套既有角色划分、2类模型、3个特征配置、3预算、2认证规则。新增416次拟合；原配置{checks}项患者动作及阈值复现V13。源文件、派生表和原缓存哈希未改变。

## 本轮结论
20%预算下，删除争议MRI淋巴结字段后，LR在5/5重复CV种子中仍观察到总体经验风险不超过预算、双缺失经验风险超过预算（原配置3/5）。梯度提升树为3/5，原配置也是3/5。因此该观察不完全依赖mri_nodes字段。五种子重用同队列，不是五次独立验证；5/5不等于已经证明真实风险稳定超标。

主划分的LR删除该字段后，总体为8/87=9.20%，双缺失为5/22=22.73%；原模型对应8/85=9.41%与5/21=23.81%。小分母与多次探索限制仍在。梯度提升树删除该字段后，主划分完全无服务，不能把没有观察到超标称为安全改善。

移除全部影像表格字段后，重复CV仍有LR 2/5、梯度提升树3/5种子出现上述模式，但主划分两模型均无服务。这提示信息输入会影响评分和认证能力，不能用一个划分判断方向。

所有特征配置、两个分数器和三个预算下，G2双缺失组仍无服务。当前仍是“子群风险有探索性信号，但子群认证代价是无法服务”，没有实现保留服务地修复风险。HPV/TCT编码、数值异常和时间关联尚未被此分析解决。

## 1. 改动范围
preop_original是原术前模型；without_mri_nodes只删除MRI淋巴结字段；without_imaging_tables删除超声大小和全部4项MRI表格字段。后二者仍保留BASE与HPV/TCT结果，不等同于原clinical模型。没有患者排除，没有修正或猜测任何临床数值。
模型超参数、拟合内OOF、排序方式和21候选阈值固定。删除字段会同时改变评分、训练内排序及通过的阈值，比较的是整个固定流程对字段的依赖，不是该字段的因果效应。

## 2. 重复CV：20%预算
pattern_seeds是5种子中总体经验风险不超过预算、双缺失经验风险超过预算的次数；double_nonempty_seeds是有双缺失自动服务的种子数。每种子双缺失分母113。两个service_mean是0–1比例；计数列是各种子均值。

{summary}

## 3. 主划分20%预算完整结果
总体测试194例，双缺失36例。risk分母为自动人数，miss分母为该组阳性人数；无服务risk为NA。

{primary}

## 4. 同种子服务率变化（20%预算）
单位百分点，删除字段配置减原配置；min/max为5种子范围，不是置信区间。仅展示总体和双缺失，其余原组保留在CSV。

{effects}

## 5. 严格预算完整结果
{strict}

## 6. 如何解释
风险模式若在删除争议字段后仍出现，只能说它不完全依赖该字段；不能据此证明输入都正确、真实风险稳定超标或缺失导致风险。若消失，也可能是评分变弱、自动人数下降，而非争议字段被证明泄漏。必须与自动人数和弃权同时解释。
G0保证总体，G2要求子群，保护对象不同。若G2双缺失仍无服务，不能称为在保留服务时修复了风险。测试经验风险超过预算并不自动等于认证理论失效。
重复CV的五种子重用患者，各种子汇总也混合了五个外层规则，不能合并成独立样本或单一部署模型；不计算伪精确显著性区间，不从所有配置中选赢家。所有预算、分数器、规则均报告。
HPV/TCT语义、异常身高/CA125、影像时序与身份映射待核实事项仍在；字段删除分析不替代人工裁定。原始773例已多轮探索，不是独立确认。没有新外部数据或MRI网络训练。

## 7. 核验与复现
源/派生/缓存哈希不变、患者角色不重叠、标签/分组一致、预处理与OOF只拟合训练角色、收敛警告即失败、原V13动作和阈值完全一致、完整输出键唯一。
11项偏差核查：总体/子群同报、无生态推断、手术入组限制、检查路径非因果、风险分母清晰、不按极端结果删病例、零服务保留、全配置避免挑选、多轮探索明确、无临床补查效应、时序疑点不自动解除。
协议reports/v15_feature_sensitivity_protocol.md；执行src/feature_sensitivity_v15.py；报告src/report_feature_sensitivity_v15.py。结果results/feature_sensitivity_v15；LOCAL文件仅本地审计。
'''.format(checks=run['baseline_patient_checks'],summary=md(summary[summary.risk_budget.eq(.2)]),primary=md(primary),effects=md(e[e.risk_budget.eq(.2)&e.mask_group.isin(['all','missing_both'])]),strict=md(summary[summary.risk_budget.lt(.2)]))
 path=ROOT/'reports/Better1_v15_真实字段删除敏感性分析.md';path.write_text(report,encoding='utf-8')
 (OUT/'report_checks.json').write_text(json.dumps({'status':'PASS','fold_rows':len(r),'baseline_patient_checks':run['baseline_patient_checks'],'model_fits':416,'source_sha256':hashlib.sha256((OUT/'fold_metrics.csv').read_bytes()).hexdigest()},indent=2),encoding='utf-8')
 print(summary[summary.risk_budget.eq(.2)].to_string(index=False));print(primary.to_string(index=False));print('STRICT NONEMPTY',summary[summary.risk_budget.lt(.2)&summary.double_nonempty_seeds.gt(0)].to_string(index=False));print(path)
if __name__=='__main__':main()
