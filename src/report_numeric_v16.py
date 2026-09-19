"""Paired real-data numeric masking sensitivity, descriptive across reused seeds."""
from pathlib import Path
import json,hashlib
import pandas as pd,numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/numeric_sensitivity_v16';PREV=ROOT/'results/feature_sensitivity_v15'
def md(df):
 return '| '+' | '.join(df.columns)+' |\n|'+'|'.join(['---']*len(df.columns))+'|\n'+'\n'.join('| '+' | '.join('NA' if pd.isna(x) else f'{x:.4f}' if isinstance(x,(float,np.floating)) else str(x) for x in row)+' |' for row in df.itertuples(index=False,name=None))
def summary(s):
 keys=['seed','model','configuration','policy','risk_budget'];j=s[s.mask_group.eq('all')].merge(s[s.mask_group.eq('missing_both')],on=keys,suffixes=('_all','_double'),validate='one_to_one');j['pattern']=(j.risk_all<=j.risk_budget)&(j.risk_double>j.risk_budget)
 return j.groupby(keys[1:]).agg(pattern_seeds=('pattern','sum'),nonempty_seeds=('auto_n_double',lambda x:int((x>0).sum())),double_auto_mean=('auto_n_double','mean'),double_positive_mean=('auto_positive_n_double','mean'),double_service_mean=('service_double','mean'),all_service_mean=('service_all','mean')).reset_index()
def main():
 run=json.loads((OUT/'run.json').read_text());assert run['status']=='COMPLETED' and run['model_fits']==624
 s=pd.read_csv(OUT/'seed_metrics.csv');old=pd.read_csv(PREV/'seed_metrics.csv');newsummary=summary(s);newsummary.to_csv(OUT/'summary.csv',index=False)
 keys=['model','configuration','policy','risk_budget'];comparison=newsummary.merge(summary(old),on=keys,suffixes=('_new','_old'),validate='one_to_one');comparison.to_csv(OUT/'summary_comparison.csv',index=False)
 pair=s.merge(old,on=['seed']+keys+['mask_group'],suffixes=('_new','_old'),validate='one_to_one');pair['service_delta_pp']=100*(pair.service_new-pair.service_old);pair.to_csv(OUT/'paired_seed_changes.csv',index=False)
 r=pd.read_csv(OUT/'fold_metrics.csv');primary=r[r.split.eq('primary')&r.risk_budget.eq(.2)&r.mask_group.isin(['all','missing_both'])][['model','configuration','policy','mask_group','auto_n','auto_positive_n','risk','service']]
 view=comparison[comparison.risk_budget.eq(.2)][keys+['pattern_seeds_old','pattern_seeds_new','nonempty_seeds_new','double_auto_mean_old','double_auto_mean_new','double_positive_mean_new']]
 report='''# V16：待核实数值视为缺失的真实敏感性分析

## Material Passport
Origin Skill: experiment-agent；Mode: run/validate；Version: numeric_sensitivity_v16；Verification Status: ANALYZED。2026-09-19。

## 本轮实际结论
20%预算下，所有模型/字段配置的重复CV“总体经验风险不超过预算、双缺失经验风险超过预算”频次与V15一致：原术前LR/梯度提升树均3/5；删除mri_nodes后LR 5/5、梯度提升树3/5；删除全部影像表格字段后LR 2/5、梯度提升树3/5。这支持信号不完全依赖本次BMI处理，但不是独立验证。

**具体规则的服务能力并不稳定。** 主划分中，删除mri_nodes的LR全局自动人数由V15的87降至0；原术前梯度提升树G2总体自动人数由55降至0，而删除mri_nodes的梯度提升树G2又由0升至38。这些方向相反的变化不支持将置缺处理简单称为改善；它们反映整个评分、排序与有限校准流程对输入处理敏感，尚未分离各环节的贡献。

所有G2双缺失服务仍为零。原术前LR主划分全局结果仍为总体8/85、双缺失5/21；这个特定观察得以保留，但另一配置的变化说明不能只挑稳定的结果。临床记录核实依然必要。

## 1. 本轮实际改动
在内存副本中，将一例待核实身高及其BMI设为缺失；CA125异常字符串在原派生表中已经缺失，因此该项无新增数值变化。当前评分器使用BMI、不直接使用height，实际可进入模型的新增缺失仅为该例BMI。数值被置缺不代表已确认原值错误，更没有猜测应改为何值。
保留773例与V15的全部划分、三特征配置、两个模型、三个预算、两种认证规则。新增624次拟合；{paired}行患者动作与原V15一一配对，其中{changed}行改变。动作行跨模型、种子等重复，不等于不同患者人数，也不能据其数量判断收益。

## 2. 20%预算：原值版本与置缺版本对照
old为V15，new为本轮。pattern_seeds是在5个种子中总体经验风险不超过预算且双缺失经验风险超过预算的次数；非空种子数另列。双缺失总人数每种子113；平均自动人数/平均自动中阳性数不是独立新样本。

{view}

## 3. 主划分20%预算
{primary}

## 4. 5%与10%预算全部结果
{strict}

## 5. 解释边界
若子群风险模式保留，只能说其不完全依赖这一次BMI处理；若变化，说明评分/校准规则可能对单例数据处理敏感，不能据此确定哪种值正确。缺失指示与填补可能随训练样本改变，所有处理只在对应训练角色拟合。
G2若仍零服务，不能称为风险为零或保留服务地修复问题；G0证书只属于总体。各模型/预算/配置分别解释，不事后选赢家。五种子复用患者，不当成五个独立队列，不提供伪独立显著性区间。
时间窗、影像身份映射、HPV/TCT编码和原始数值核实仍未解决。主裁定状态及原始、派生、缓存文件哈希均不变。无新增临床确认或外部验证。

## 6. 核验与复现
源文件及裁定队列不变；全部患者角色保持不变、标签与分组一致；预处理及OOF只用拟合角色；收敛检查；逐患者配对完整；零服务风险NA。
11项检查覆盖总体/子群、生态推断、手术选择、检查路径、分母/基准率、回归均值、保留未服务病例、全配置比较、多轮探索、关联非因果、时序未裁定。
协议reports/v16_numeric_sensitivity_protocol.md；执行src/numeric_sensitivity_v16.py；报告src/report_numeric_v16.py；结果results/numeric_sensitivity_v16。LOCAL更改清单和动作文件只用于本地审计。
'''.format(paired=run['paired_actions'],changed=run['changed_action_rows'],view=md(view),primary=md(primary),strict=md(newsummary[newsummary.risk_budget.lt(.2)]))
 path=ROOT/'reports/Better1_v16_异常数值缺失化敏感性分析.md';path.write_text(report,encoding='utf-8')
 (OUT/'report_checks.json').write_text(json.dumps({'status':'PASS','paired_action_rows':run['paired_actions'],'changed_action_rows':run['changed_action_rows'],'model_fits':624,'source_sha256':hashlib.sha256((OUT/'seed_metrics.csv').read_bytes()).hexdigest()},indent=2),encoding='utf-8')
 print(view.to_string(index=False));print(primary.to_string(index=False));print(path)
if __name__=='__main__':main()
