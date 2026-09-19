"""Report all V5 configurations; reference risks are Monte Carlo estimates."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];O=ROOT/'results/mechanism_training_v5/formal'
def md(d):
 return '| '+' | '.join(d.columns)+' |\n|'+'|'.join(['---']*len(d.columns))+'|\n'+'\n'.join('| '+' | '.join('NA' if pd.isna(x) else f'{x:.4f}' if isinstance(x,(float,np.floating)) else str(x) for x in row)+' |' for row in d.itertuples(index=False,name=None))
def main():
 meta=json.loads((O/'run.json').read_text());assert meta['status']=='COMPLETED' and meta['completed_datasets']==1200 and meta['model_fits']==3600
 r=pd.read_csv(O/'replicate_results.csv');s=pd.read_csv(O/'summary.csv');p=pd.read_csv(O/'platform_metrics.csv');m=pd.read_csv(O/'mask_properties.csv')
 assert len(r)==162000 and not r.duplicated(['mechanism','n_total','repeat','model','policy','budget','mask_group']).any()
 assert not ((r.reference_accepted_n==0)&r.reference_risk.notna()).any()
 index=['mechanism','n_total','repeat','model','budget','mask_group'];paired=r.pivot(index=index,columns='policy',values='reference_auto_rate').reset_index()
 for method in ['G2_search126','G2_fixed63']:paired[method+'_minus_G0']=paired[method]-paired.G0_search126
 paired.to_csv(O/'paired_service_deltas.csv',index=False)
 agg=paired.groupby(['mechanism','n_total','model','budget','mask_group']).agg(delta_mean=('G2_fixed63_minus_G0','mean'),delta_sd=('G2_fixed63_minus_G0','std'),repetitions=('repeat','size')).reset_index();agg['MCSE']=agg.delta_sd/np.sqrt(agg.repetitions);agg.to_csv(O/'paired_service_summary.csv',index=False)
 props=m.groupby(['mechanism','mask_group']).agg(fraction=('reference_fraction','mean'),prevalence=('reference_prevalence','mean')).reset_index();props.to_csv(O/'observed_mechanism_properties.csv',index=False)
 platform=p.groupby(['mechanism','n_total','model']).agg(AUROC=('reference_AUROC','mean'),Brier=('reference_Brier','mean')).reset_index();platform.to_csv(O/'platform_summary.csv',index=False)
 cols=['mechanism','model','policy','auto_rate_mean','reference_risk_mean_served','service_fraction']
 double=s[(s.n_total==10000)&(s.budget==.2)&(s.mask_group=='missing_both')][cols]
 marginal=s[(s.n_total==10000)&(s.budget==.2)&(s.mask_group=='all')][['mechanism','model','policy','auto_rate_mean','reference_risk_mean_served']]
 view=double.merge(marginal,on=['mechanism','model','policy'],suffixes=('_double','_overall'))
 view.to_csv(O/'report_budget20_n10000.csv',index=False)
 # Same-repeat diagnostic: avoid mistaking separate averages for joint events.
 common=['mechanism','n_total','repeat','model','budget']
 ga=r[(r.policy=='G0_search126')&(r.mask_group=='all')][common+['reference_risk','reference_accepted_n']]
 gd=r[(r.policy=='G0_search126')&(r.mask_group=='missing_both')][common+['reference_risk','reference_accepted_n']]
 joint=ga.merge(gd,on=common,suffixes=('_overall','_double'))
 joint['estimated_global_pass_child_fail']=(joint.reference_risk_overall<=joint.budget)&(joint.reference_risk_double>joint.budget)
 # Hoeffding reference-sampling margins, union bounded over two comparisons.
 overall_margin=np.sqrt(np.log(4/.05)/(2*joint.reference_accepted_n_overall.replace(0,np.nan)))
 double_margin=np.sqrt(np.log(4/.05)/(2*joint.reference_accepted_n_double.replace(0,np.nan)))
 joint['same_event_with_reference_sampling_margin']=(joint.reference_risk_overall+overall_margin<=joint.budget)&(joint.reference_risk_double-double_margin>joint.budget)
 js=joint.groupby(['mechanism','n_total','model','budget']).agg(repetitions=('repeat','size'),estimated_joint_count=('estimated_global_pass_child_fail','sum'),margin_screened_joint_count=('same_event_with_reference_sampling_margin','sum')).reset_index()
 js.to_csv(O/'joint_global_child_diagnostics.csv',index=False)
 plot=s[(s.budget==.2)&(s.mask_group=='missing_both')&s.policy.isin(['G0_search126','G2_fixed63'])&s.model.isin(['masked_lr','masked_hgb'])]
 fig,axes=plt.subplots(1,3,figsize=(12,4),sharey=True)
 for ax,mode in zip(axes,['MCAR','MAR','INFORMATIVE']):
  for (model,policy),q in plot[plot.mechanism==mode].groupby(['model','policy']):
   ax.errorbar(q.n_total,q.auto_rate_mean,yerr=1.96*q.auto_rate_MCSE,marker='o',label=model+'/'+policy)
  ax.set_xscale('log');ax.set_title(mode);ax.set_xlabel('Development sample size')
 axes[0].set_ylabel('Double-missing automatic service rate');axes[-1].legend(fontsize=6)
 fig.suptitle('Budget .20; 200 replicates; bars: 1.96 x Monte Carlo SE of mean')
 fig.tight_layout();fig.savefig(O/'trained_mechanism_service.png',dpi=180);plt.close(fig)
 # Full-grid diagnostic, not a definition of success or statistical hypothesis test.
 q=s[(s.mask_group=='missing_both')&(s.model!='complete_lr')]
 zero=q[q.service_fraction==0]
 countnote=f'缺失输入模型的 {len(q)} 个机制—样本量—模型—策略—预算配置中，{len(zero)} 个配置在 200 次重复中均未在独立参考集产生双缺失服务。配置网格不是随机人群，不能把该比例解释为临床失败率。'
 report='''# V5：从检查缺失到模型训练的完整机制实验

## 结果先读

在信息性缺失、开发样本 10,000、预算 20% 的预设配置中，缺失输入 LR 的全局规则总体参考风险均值为 16.67%，双缺失参考风险均值为 30.90%；G2_fixed63 的双缺失服务为零。完整检查 LR 的 G2_fixed63 则保留 45.68% 双缺失服务，其有服务重复的风险均值为 11.41%。这些是合成机制下跨200次重复的均值，不是真实患者临床效果。

这补强了“总体合格可以掩盖被接收子群风险”的机制证据；也表明在本设定中，减少校正成本并不足以恢复缺失输入模型的双缺失服务。完整检查对照说明信息可得性值得进一步研究，但不证明对真实患者补查有临床收益，也不证明所有分数器都无法从缺失输入中获益。

同一重复内的联合事件单独核验如下（n=10,000，预算20%，缺失输入模型）：

{joint}

estimated_joint_count 表示同一次重复中估计的全局风险不超预算、双缺失风险超预算。margin_screened_joint_count 进一步要求结论超过独立参考抽样的 Hoeffding 余量：每次重复对两个均值使用合计5%预算；它不是跨200次重复的联合95%保证，也不是临床结论。零计数只表示此诊断未检出。

Material Passport：Origin Skill=experiment-agent；Mode=run/validate；Version=mechanism_training_v5；Verification Status=ANALYZED。仅合成数据，未引入新增真实患者证据。

## 本轮完成内容

原来的 V2 模拟直接生成风险分数。本轮生成可见协变量、潜变量及两项检查，再遮蔽检查、实际训练模型，并独立校准认证。MCAR/MAR/信息性缺失各两档开发样本量（1,000/10,000），每格 200 次重复，共 1,200 个独立开发数据集、3,600 个模型。每次拟合/校准/测试按 50%/25%/25% 标签无关拆分，另生成独立 20,000 人参考集，仅用于评价。

所有参数在运行前固定于 `v5_mechanism_training_protocol.md`。LR 与 HGB 使用同样的中位数填补加缺失指示；完整检查 LR 仅是合成信息可得性对照，不是可部署模型或最优算法上界。没有给任何模型输入潜变量 U，也没有根据结果调整机制。

## 机制是否产生不同数据？

以下是独立参考集平均比例与患病率。MCAR 的缺失生成概率与结局及协变量独立，但不同缺失输入可能使分数质量不同；因此 MCAR 也不自动保证“被模型接收后”的子群风险相同。MAR 依据始终可见变量，信息性缺失依据不可见潜变量。后两种机制实际组比例不同，不能把差异全部解释为单一因素的因果效应。

{props}

## 真实训练后的分数表现

开发样本 10,000，参考集 AUROC 与 Brier 跨 200 次训练均值：

{platform}

此处性能只作诊断，不是本文主终点。完整检查 LR 与缺失 LR 的差异反映本合成设定下的信息可得性，但 LR 仍可能存在模型设定误差。

## 风险与服务：预算 20%、开发样本 10,000

G0_search126 是全局规则，G2_search126 使用原有联合候选校正，G2_fixed63 预先固定 G2 后只校正三组×21个阈值。全局规则不保证双缺失子群风险。各方法的保证分别成立，不涵盖看结果后选择方法的联合声明。

表中的 auto_rate 是对应人群自动比例；reference_risk 是参考集被接收者真实结局概率的均值，再对有服务的重复取均值；service_fraction 是参考集中存在自动服务的重复比例。总体和双缺失风险的均值分母可能不同，不能由这张汇总表宣称每个重复都同时达标或超标。

{view}

{countnote}

![样本量与服务](../results/mechanism_training_v5/formal/trained_mechanism_service.png)

其余预算、样本量、原始测试阳性数和全部失败记录见 `summary.csv` 与 `replicate_results.csv`。配对服务差值在同一重复、分数器和预算内计算，见 `paired_service_summary.csv`；MCSE 衡量模拟均值精度，不是临床置信区间。

## 解释边界

1. 本实验可以检验已训练模型上的机制现象，不能为当前真实队列的零服务结果翻案。模拟参数没有拟合真实患者分布，也未验证其临床代表性。
2. 独立参考集风险比直接数随机阳性更少受结局采样噪声影响，但仍有有限参考样本误差。`reference_risk_exceeds_budget_fraction` 是估计风险超预算的频率，不是精确总体违约率；尤其接收人数少时不可过度解释。NA 是无服务风险未定义。
3. 增加总开发人数同时增加拟合与校准样本，不能把变化完全归因于校准人数。若要分离两者，下一轮应固定训练人数再改变校准量，并使用新的预登记设置。
4. 本轮没有声称分块提高预测能力。分块改变的是自动接收规则及保证作用域，可能以服务减少换取子群控制；完整信息分数也不保证小样本认证一定通过。

## 检查与可复现性

程序检查通过：MCAR 概率固定、缺失输入遮蔽一致、角色不重叠、填补器只使用拟合数据、预测有限、全部自动动作的校准上界满足预算、162,000 条策略—组别结果键唯一、空服务不记作零风险。原始临床数据未读取。模拟自检不是外部独立复现。

11 项偏差检查：组别报告防 Simpson 掩盖；不作生态因果推断；不将合成结果解除真实手术入组/Berkson 限制；检查路径 collider；基准率逐组披露；固定200重复减少挑偶然结果/回归均值；不假设幸存与选择偏差消失；各候选家族多重校正；运行前协议约束分析分叉；关联与因果区分；模拟因果顺序不代替真实时间窗核实。

执行：`D:/conda/envs/pytorch/python.exe X:/GJA/WORK/src/mechanism_training_v5.py`，完成后运行 `src/report_mechanism_v5.py`。重跑会更新 V5 formal 目录，需先保留待比较版本。日志和代码哈希见 `run.json`。
'''.format(props=md(props),platform=md(platform[platform.n_total==10000]),view=md(view),countnote=countnote,joint=md(js[(js.n_total==10000)&(js.budget==.2)&(js.model!='complete_lr')]))
 path=ROOT/'reports/Better1_v5_缺失机制与真实训练实验.md';path.write_text(report,encoding='utf-8')
 checks={'status':'PASS','unique_rows':len(r),'no_service_risk_undefined':True,'completed_datasets':1200,'report_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()};(O/'report_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
 print(view.to_string(index=False));print(countnote);print(path)
if __name__=='__main__':main()
