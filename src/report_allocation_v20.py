"""Report calibration allocation as a joint training-certification tradeoff."""
import json
import numpy as np,pandas as pd
import v2_pipeline as v
from report_numeric_v16 import md

ROOT=v.ROOT;OUT=ROOT/'results/calibration_allocation_v20'

def main():
 run=json.loads((OUT/'run.json').read_text());assert run['status']=='COMPLETED' and run['model_fits']==416
 s=pd.read_csv(OUT/'seed_metrics.csv');r=pd.read_csv(OUT/'fold_metrics.csv');roles=pd.read_csv(OUT/'role_sizes.csv');p=pd.read_csv(OUT/'platform_metrics.csv')
 summary=s.groupby(['model','allocation','risk_budget','policy','mask_group']).agg(mean_service=('service','mean'),service_min=('service','min'),service_max=('service','max'),nonempty_seeds=('auto_n',lambda x:int((x>0).sum())),mean_auto_n=('auto_n','mean'),mean_auto_positive_n=('auto_positive_n','mean'),mean_risk_nonempty=('risk','mean')).reset_index();summary.to_csv(OUT/'summary.csv',index=False)
 role=roles.groupby(['allocation','role']).agg(n_min=('n','min'),n_max=('n','max'),double_min=('double_n','min'),double_max=('double_n','max')).reset_index()
 # First average folds within seed; then describe the reused seeds.
 platform_seed=p[p.split.eq('repeated')].groupby(['seed','model','allocation'])[['auroc','auprc','brier']].mean().reset_index();platform_summary=platform_seed.groupby(['model','allocation'])[['auroc','auprc','brier']].mean().reset_index();platform_seed.to_csv(OUT/'platform_by_seed.csv',index=False)
 primary=r[r.split.eq('primary')&r.risk_budget.eq(.2)&r.mask_group.isin(['all','missing_both'])][['model','allocation','policy','mask_group','auto_n','auto_positive_n','risk','service']]
 keys=['seed','model','risk_budget','policy','mask_group'];baseline=s[s.allocation.eq('original')]
 paired=s.merge(baseline,on=keys,suffixes=('_new','_original'),validate='many_to_one');paired['service_change_pp']=100*(paired.service_new-paired.service_original);paired.to_csv(OUT/'paired_service_changes.csv',index=False)
 double=r[r.mask_group.eq('missing_both')&r.policy.eq('G2_FST')];active=int((double.auto_n>0).sum())
 view=summary[summary.mask_group.eq('missing_both')&summary.policy.eq('G2_FST')]
 report=f'''# V20：把更多患者留给校准，能否恢复自动服务？

Origin Skill: experiment-agent；Mode: run/validate；Verification Status: ANALYZED。2026-09-19。

## 实验定义与完成情况
全773例、原术前字段、LR/HGB、26个外层划分保持不变。比较original、开发集50%校准、开发集75%校准；后两种分配重新训练，不把旧模型见过的病例预测当成独立校准。完成416次拟合；{run['baseline_patient_checks']}行原分配动作/阈值复现V17。患者来源、临床待核实项和测试病例不变。

扩大校准同时缩小训练，这是整体数据分配的权衡，不能单独归因为“更多校准数据”的净作用。原主划分校准占开发集1/3，重复CV约1/4；不要把original全部标成25%。新分配嵌套、按无标签固定随机顺序迁入，所有预处理与OOF排序只使用新训练成员。不同分配之间没有事后挑选最优的统一风险保证。

## 角色人数

**本轮实际结论：扩大校准分配没有恢复双缺失认证服务。** 原分配、50%与75%三种分配在所有预算和模型下均为零，含主划分；468个折级双缺失配置全部无服务。75%分配时双缺失校准人数为61–79人，训练集仅145–155人。其总容量虽高于10%/20%预算的零阳性最低要求，并不代表有足够的低风险接纳病例通过检验；5%预算零阳性仍需80人。

训练减少伴随评分指标下降：LR平均AUROC由0.7126变为50%分配的0.6868、75%分配的0.6580；HGB由0.7068变为0.6815、0.6620。AUPRC也下降，Brier变差。以上是同一队列重复CV的描述性差异，不能声称独立统计显著。

因此，当前不宜继续反复调整内部训练/校准比例来寻找非零服务。V19的低风险排序信号仍有研究价值，但本轮证明的只是这一组内部重分配方案未能将它转化为认证服务；不代表额外收集独立校准数据无效，也不排除更好的评分器。后续投入应转向临床核实、明确新增数据条件和独立验证设计。

范围跨所有划分，double指双缺失人数；不是独立患者人数相加。

{md(role)}

## 双缺失认证服务：所有预算和分配
{len(double)}个折级配置中，{active}个有非零G2双缺失测试服务。下面是五个重复CV种子合并五折后的均值与范围；非空种子数同时报告，风险均值只在有自动病例的种子上计算。零服务不意味着零风险，也不证明临床可用。

{md(view)}

## 20%预算总体服务与G0/G2对照
{md(summary[summary.mask_group.eq('all')&summary.risk_budget.eq(.2)])}

## 评分表现（辅助终点）
AUROC、AUPRC、Brier先在每种子内平均测试折，再平均五种子；不是合并概率的总体OOF AUROC，也不是独立置信区间。Brier越低越好；评分指标改变不等于认证服务改变。全部主划分和各折指标见platform_metrics.csv。

{md(platform_summary)}

## 主划分20%预算（不只展示重复CV）
{md(primary)}

## 解释边界
如果增加校准后仍零服务，说明在该队列、模型、分配和检验下，重分配没有解决问题，不代表真实新增独立数据无效。如果出现少量服务，必须同时看漏诊数、风险和跨划分稳定性，不能把一次非零结果等同于成功。不根据本轮测试结果选择校准比例用于后续独立验证。

临床核实与外部验证仍未完成，同一773例已经多轮探索。11项检查覆盖总体/子群、生态推断、手术选择、检查路径、分母基准率、回归均值、零服务保留、全配置比较、多轮探索、关联非因果、时序未裁定。G0保证只属于总体；各分配和预算分别解释。

协议reports/v20_calibration_allocation_protocol.md；执行src/calibration_allocation_v20.py；报告src/report_allocation_v20.py；结果results/calibration_allocation_v20。LOCAL文件仅在本地，代码同步GitHub。
'''
 (ROOT/'reports/Better1_v20_训练校准分配与服务代价.md').write_text(report,encoding='utf-8')
 v.dump(OUT/'report_checks.json',{'status':'PASS','fits':416,'double_nonempty_fold_cells':active,'double_total_fold_cells':len(double),'fold_metrics_sha256':v.old.digest(OUT/'fold_metrics.csv')})
 print(view.to_string(index=False));print(platform_summary.to_string(index=False));print(role.to_string(index=False))

if __name__=='__main__':main()
