"""Paired effects with frozen training; reference frontiers are diagnostic only."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];O=ROOT/'results/calibration_information_v7/formal'
def md(df):
 return '| '+' | '.join(df.columns)+' |\n|'+'|'.join(['---']*len(df.columns))+'|\n'+'\n'.join('| '+' | '.join('NA' if pd.isna(x) else f'{x:.4f}' if isinstance(x,(float,np.floating)) else str(x) for x in row)+' |' for row in df.itertuples(index=False,name=None))
def paired_summary(df,keys,value):
 out=df.groupby(keys).agg(mean=(value,'mean'),sd=(value,'std'),repetitions=(value,'size')).reset_index();out['MCSE']=out.sd/np.sqrt(out.repetitions)
 return out
def main():
 run=json.loads((O/'run.json').read_text());assert run['status']=='COMPLETED' and run['model_fits']==1200 and run['completed_repetitions']==400
 r=pd.read_csv(O/'replicate_results.csv');f=pd.read_csv(O/'reference_grid_diagnostics.csv')
 keys=['mechanism','model','calibration_n','policy','budget','mask_group']
 assert len(r)==216000 and not r.duplicated(keys+['repeat']).any()
 s=r.groupby(keys).agg(repetitions=('repeat','size'),service_mean=('reference_service','mean'),service_sd=('reference_service','std'),service_fraction=('reference_has_service','mean'),risk_mean_served=('reference_risk','mean'),estimated_violation_fraction=('reference_risk_exceeds_budget','mean'),actual_cal_group_n_mean=('actual_cal_group_n','mean')).reset_index()
 s['service_MCSE']=s.service_sd/np.sqrt(s.repetitions);s.to_csv(O/'summary.csv',index=False);assert (s.repetitions==200).all()
 p=r.pivot(index=['mechanism','repeat','model','policy','budget','mask_group'],columns='calibration_n',values='reference_service').reset_index();p['service_gain_50000_minus150']=p[50000]-p[150]
 p.to_csv(O/'paired_calibration_effects.csv',index=False)
 pe=paired_summary(p,['mechanism','model','policy','budget','mask_group'],'service_gain_50000_minus150');pe.to_csv(O/'paired_calibration_summary.csv',index=False)
 q=r[r.model.isin(['masked_lr','complete_lr'])].pivot(index=['mechanism','repeat','calibration_n','policy','budget','mask_group'],columns='model',values='reference_service').reset_index();q['complete_minus_masked']=q.complete_lr-q.masked_lr;q.to_csv(O/'paired_information_effects.csv',index=False)
 qi=paired_summary(q,['mechanism','calibration_n','policy','budget','mask_group'],'complete_minus_masked');qi.to_csv(O/'paired_information_summary.csv',index=False)
 fs=f.groupby(['mechanism','model','budget','grid']).agg(repetitions=('repeat','size'),point_frontier_mean=('point_service','mean'),conservative_frontier_mean=('conservative_service','mean'),point_frontier_has_service=('point_service',lambda x:(x>0).mean()),conservative_frontier_has_service=('conservative_service',lambda x:(x>0).mean())).reset_index();fs.to_csv(O/'reference_frontier_summary.csv',index=False)
 assert (f.conservative_service<=f.point_service+1e-12).all()
 view=s[(s.mask_group=='missing_both')&(s.budget==.2)&s.calibration_n.isin([150,5000,50000])][['mechanism','model','policy','calibration_n','service_mean','service_fraction','risk_mean_served']]
 view.to_csv(O/'report_budget20_double.csv',index=False)
 fig,axes=plt.subplots(2,2,figsize=(12,8),sharey=True)
 for i,mode in enumerate(['MCAR','INFORMATIVE']):
  for j,policy in enumerate(['G2_fixed63','G2_FST']):
   ax=axes[i,j];x=s[(s.mechanism==mode)&(s.policy==policy)&(s.mask_group=='missing_both')&(s.budget==.2)]
   for model,z in x.groupby('model'):
    ax.errorbar(z.calibration_n,z.service_mean,yerr=1.96*z.service_MCSE,label=model,marker='o',capsize=2)
   ax.set_xscale('log');ax.set_title(mode+' / '+policy);ax.set_xlabel('Calibration sample size; training fixed at 5000');ax.set_ylabel('Double-missing automatic service');ax.legend(fontsize=8)
 fig.suptitle('Budget .20; 200 paired repeats per mechanism; order frozen across sizes')
 fig.tight_layout();fig.savefig(O/'calibration_only_service.png',dpi=180);plt.close(fig)
 selected=s[(s.mechanism=='INFORMATIVE')&(s.mask_group=='missing_both')&(s.budget==.2)&s.calibration_n.isin([150,50000])]
 compact=selected.pivot(index=['model','policy'],columns='calibration_n',values='service_mean').reset_index();compact.columns=['model','policy','service_cal150','service_cal50000'];compact['gain_pp']=100*(compact.service_cal50000-compact.service_cal150)
 diag=fs[(fs.mechanism=='INFORMATIVE')&(fs.budget==.2)]
 report='''# V7：固定训练后，增加校准人数能否恢复双缺失服务？

日期2026-09-18。Material Passport：Origin Skill=experiment-agent；Mode=run/validate；Version=calibration_information_v7；Verification Status=ANALYZED。本轮仅合成数据，不新增真实临床证据。

## 1. 本轮实际结果

**当前判定：继续研究“安全认证的服务率代价与失效边界”，暂不恢复“分块能让真实双缺失患者获得有意义自动服务”的主张。** 真实队列的双缺失组仍未通过认证；下面的新增证据全部来自合成机制。

固定序列方法、风险预算20%、总校准人数50000时：随机缺失机制下，缺失输入LR/HGB的双缺失自动率分别为87.99%/87.73%；信息性缺失机制下分别只有5.41%/3.64%。同一信息性机制中，允许LR获得完整检查输入时达到65.59%。这些是200次独立重复的平均服务率，不是临床效果或推荐的风险预算。

这支持把校准数据量与信息可得性分开研究，但不能将服务差距全部归因于信息损失：模型、候选网格、固定检验顺序也参与限制。加密网格诊断仍发现可能存在的低风险区域，因此本轮没有证明信息论上的不可能性。50000指全体校准人数，绝不是双缺失组人数或临床招募建议。

以下是信息性缺失、风险预算20%的预设配置；每个模型固定训练5000人，检验顺序由另2000人预先固定。校准人数改变时不再训练，不改顺序。

{compact}

service 为双缺失组的平均自动比例，gain_pp 为增加校准人数后的配对均值变化（百分点）。完整检查 LR 是合成信息可得性对照，不代表真实补查收益。固定序列的排序锚点为2500校准人，对50000人不一定最有效；因此不能把某一种固定排序的失败等同于信息不足。

## 2. 我们隔离了什么？

V5 增加总开发人数时，训练和校准人数同时增加。本轮每个模型只拟合一次，随后校准人数取150、500、1500、5000、15000、50000，使用同一校准池的嵌套前缀与同一独立50000人参考集。这样校准量对照不再混入重新训练带来的变化。

MCAR与信息性缺失各200次独立重复，共400组独立训练/选择/校准/参考数据，1200个模型，216000条策略—预算—分组结果。各规模共享同一重复内的数据，因此不是相互独立试验；配对差的Monte Carlo标准误以200个独立重复为单位。

嵌套校准池仅用于比较预先固定的样本量，不能解释为“逐步招募直到通过即停止”的有效认证程序。没有对六个样本量、多个方法和预算宣称联合95%保证。

![纯校准量效应](../results/calibration_information_v7/formal/calibration_only_service.png)

## 3. 原有阈值网格是否限制服务？

下表仍为信息性缺失、预算20%；原21阈值与包含原阈值的加密网格在独立参考集上做诊断。

{frontier}

point_frontier_mean 是在参考集估计风险不超过预算时，该分数/网格可接收的最大比例，再对训练重复取均值。它使用参考集结果筛选，因此是诊断性估计，不能当作真实总体精确上界或合法部署规则。

conservative_frontier_mean 先对加密网格的风险均值作同时Hoeffding上界筛选，再取最大服务比例。它更保守；每个固定模型与该参考总体下的风险筛选范围是该网格，不覆盖所有模型或所有算法。服务比例仍是参考样本估计。即使它为零，也可能只是此诊断界太宽，不能证明任何低风险区域都不存在。

这两个诊断都不回流到训练、排序或校准。加密网格只用于检查原21阈值是否遗漏可用区域，没有在真实队列上事后加密阈值找正结果。

## 4. 两类机制、三个模型、两个认证方法

以下展示预算20%、三个代表校准量；其余所有预算和规模保留在summary.csv。

{view}

service_fraction 为200次重复中参考集出现非空服务的比例；risk_mean_served 仅对有服务重复取均值，NA代表未定义。小服务时参考风险误差更大，不能把估计风险超过预算的次数当成精确总体违约率。完整检查和缺失LR的配对差另存paired_information_summary.csv。

## 5. 与真实结果如何衔接

V6的真实双缺失组仍无服务，且所有当前网格阈值均未通过相应单项检验。本轮回答的是合成机制中能否通过增加独立校准数据恢复服务，不是对现有113个双缺失病例作扩增。即便50000校准人有效，也不能直接转化为临床招募建议：真实分布、分数质量、检查路径和信息成本均未由模拟证明。

若某模型服务随校准量恢复，说明其在该机制下存在有限样本认证障碍；若仍不恢复，应结合参考网格诊断以及固定序列与Bonferroni差异判断，不能直接断言信息论不可能。信息可得性、评分模型、候选网格和排序都可能限制结果。

## 6. 核验与复现

检查通过：四个角色独立随机流；填补器只看训练集；各校准规模使用完全相同的冻结分数和顺序哈希；所有自动动作满足对应认证约束；加密网格包含原网格且诊断最大服务不下降；输出键唯一、数量完整；零服务风险不记零；训练次数不随六个校准规模增加。

11项偏差核查：分组报告防Simpson；不从合成组间差异作生态推断；不解除真实手术入组/Berkson限制；检查路径collider；基准率明确由生成器决定；独立重复减少回归均值误读；不假设真实幸存/入组偏差已解决；各固定认证家族处理多重搜索但不宣称跨方法联合；预先协议约束分析分叉；合成信息对照非临床因果；模拟时间顺序非真实术前时序验证。

运行src/calibration_information_v7.py，报告src/report_calibration_v7.py；协议reports/v7_calibration_information_protocol.md。结果目录results/calibration_information_v7/formal；代码哈希、完成状态与计算规模见run.json。未读取或修改临床原表。
'''.format(compact=md(compact),frontier=md(diag),view=md(view))
 path=ROOT/'reports/Better1_v7_校准样本与信息损失分解.md';path.write_text(report,encoding='utf-8')
 checks={'status':'PASS','rows':len(r),'fixed_training_models':1200,'calibration_sizes':6,'reference_frontier_conservative_le_point':True,'report_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()};(O/'report_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
 print(compact.to_string(index=False));print(diag.to_string(index=False));print(path)
if __name__=='__main__':main()
