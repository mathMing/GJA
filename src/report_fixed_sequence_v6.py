"""Report a paired LTT fixed-sequence comparison without pooled CV inference."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];O=ROOT/'results/fixed_sequence_v6'
def md(df):
 return '| '+' | '.join(df.columns)+' |\n|'+'|'.join(['---']*len(df.columns))+'|\n'+'\n'.join('| '+' | '.join('NA' if pd.isna(x) else f'{x:.4f}' if isinstance(x,(float,np.floating)) else str(x) for x in row)+' |' for row in df.itertuples(index=False,name=None))
def main():
 run=json.loads((O/'run.json').read_text());validation=json.loads((O/'validation.json').read_text());assert run['status']=='COMPLETED' and validation['status']=='PASS'
 fold=pd.read_csv(O/'fold_metrics.csv');seed=pd.read_csv(O/'seed_metrics.csv');diag=pd.read_csv(O/'scope_diagnostics.csv');trace=pd.read_csv(O/'sequence_trace.csv')
 summ=seed.groupby(['model','features','policy','risk_budget','mask_group']).agg(auto_n_mean=('auto_n','mean'),positive_in_auto_mean=('auto_positive_n','mean'),service_mean=('auto_rate','mean'),service_min=('auto_rate','min'),service_max=('auto_rate','max'),risk_mean_served=('risk','mean')).reset_index();summ.to_csv(O/'summary.csv',index=False)
 pivot=seed.pivot(index=['seed','model','features','risk_budget','mask_group'],columns='policy',values='auto_rate').reset_index();pivot['FST_minus_fixed63']=pivot.G2_FST_OOF-pivot.G2_fixed63;pivot['FST_minus_locked3']=pivot.G2_FST_OOF-pivot.G2_locked3;pivot.to_csv(O/'paired_service_deltas.csv',index=False)
 diag['outcome']=np.select([diag.active,~diag.any_pointwise_feasible_DIAGNOSTIC_ONLY],['certified','no_candidate_passes_single_test'],default='order_stopped_before_feasible_candidate')
 failure=diag[diag.scope=='missing_both'].groupby(['split','budget','outcome']).size().rename('model_split_count').reset_index();failure.to_csv(O/'double_missing_failure_summary.csv',index=False)
 double=fold[(fold.policy=='G2_FST_OOF')&(fold.mask_group=='missing_both')]
 total=int(double.auto_n.sum());positives=int(double.auto_positive_n.sum())
 verdict=('在主划分与全部重复折中，FST 的双缺失自动人数仍为零；三个预算、四个分数器均如此。' if total==0 else f'FST 在部分配置产生双缺失服务；跨配置自动记录共 {total} 条、阳性记录 {positives} 条，含重复患者，不能当作独立患者数或临床效果。请以逐种子和主划分表解释。')
 grouped=trace.groupby(['split','seed','fold','model','features','budget','scope'],sort=False)
 for _,q in grouped:
  ranks=q['rank'].to_numpy();assert np.array_equal(ranks,np.arange(len(q)))
  assert q.passed.iloc[:-1].all(), 'Continued after a failed test'
 # This also checks that the trace describes the actual selected rule.
 for row in diag.itertuples():
  q=grouped.get_group((row.split,row.seed,row.fold,row.model,row.features,row.budget,row.scope))
  passed=q[q.passed]
  assert bool(len(passed))==bool(row.active)
  if row.active:assert np.isclose(passed.threshold.max(),row.selected_threshold) and row.certified_risk_bound==row.budget
 ds=summ[(summ.policy=='G2_FST_OOF')&(summ.mask_group=='missing_both')]
 overall=summ[(summ.mask_group=='all')&(summ.risk_budget==.2)]
 primary=fold[(fold.split=='primary')&(fold.policy=='G2_FST_OOF')&(fold.risk_budget==.2)][['model','features','mask_group','n','auto_n','auto_positive_n','empirical_risk']]
 focus=summ[(summ.model=='lr')&(summ.features=='preop')&(summ.risk_budget==.2)&(summ.mask_group=='all')].set_index('policy')
 fs=focus.loc['G2_FST_OOF'];bf=focus.loc['G2_search126']
 overall_note=(f'总体结果并非全部阴性：20%预算下，术前LR的重复CV平均自动率从原G2的{100*bf.service_mean:.2f}%升到固定序列的{100*fs.service_mean:.2f}%，增加{100*(fs.service_mean-bf.service_mean):.2f}个百分点；后者跨种子范围{100*fs.service_min:.2f}%–{100*fs.service_max:.2f}%。该提升来自其他子群，未恢复双缺失服务；属于同一既有队列的探索性配对结果，不是独立验证或新算法贡献。')
 fig,axes=plt.subplots(1,2,figsize=(11,4))
 for ax,group in zip(axes,['all','missing_both']):
  q=summ[(summ.risk_budget==.2)&(summ.mask_group==group)]
  for policy,z in q.groupby('policy'):
   ax.plot(z.model+'/'+z.features,z.service_mean*100,marker='o',label=policy)
  ax.set_title(group);ax.set_ylabel('Automatic service (%)');ax.tick_params(axis='x',rotation=15)
 axes[-1].legend(fontsize=7);fig.suptitle('Real data: budget .20; mean across five CV seeds, not independent validation')
 fig.tight_layout();fig.savefig(O/'fst_service_comparison.png',dpi=180);plt.close(fig)
 report='''# V6：LTT 固定序列强基线结果

日期：2026-09-18。Material Passport：Origin Skill=experiment-agent；Mode=run/validate；Version=fixed_sequence_v6；Verification Status=ANALYZED。已有方法实现、校验及真实数据配对复算，不是新理论，也尚非独立复现。

## 本轮结论

{verdict}

{overall_note}

本轮完成同一主随机划分和5种子×5折，4个分数器共416次模型拟合（含训练内部OOF）。原126候选、固定63候选和单阈值锁定的900个种子级比较计数与V4一致，因此此轮方法差异不来自患者划分或基础模型改变。

## 固定序列做了什么？

先用拟合集OOF预测估计每个阈值通过独立认证的概率，固定顺序；每组使用0.05/3的检验水平，校准集遇第一次不通过立即停止。在通过的前缀中选择最大阈值。排序没有使用校准或测试标签，空接收视为不通过。罕见仅缺HPV组始终弃权，不以父块证书替代。

该机制依据 [LTT第2.3.1节](https://arxiv.org/html/2110.01052v5)。固定顺序和首次不通过即停止是有效性的关键；它不要求真实风险沿阈值单调。这里的训练内功效排序是固定的实现选择，并非穷尽所有排序。

与V4单阈值不同，此方法可以在第一个阈值通过后继续测试。FST 的 risk_bound 存储预定预算 alpha，表示此预算下的认证；普通CP上界仅为诊断列，不能把选中阈值的单项CP界称为选择后更紧的同时上界。各方法与各预算分别解释，不能看到结果后选赢家并沿用未经校正的联合声明。

## 双缺失组：所有风险预算

下表是每个种子内汇总773人的测试结果后，再跨5种子平均；双缺失组每种子分母113。min/max是种子范围，不是置信区间。NA表示无服务风险未定义。

{double}

## 失败是否仅仅因为顺序选得不好？

{failure}

no_candidate_passes_single_test：当前校准数据的21个阈值中，没有任何一个通过0.05/3的单项二项检验。对这个候选网格和检验水平，仅改变固定序列顺序不可能产生通过项。

order_stopped_before_feasible_candidate：至少存在一个单项检验可通过，但固定顺序先失败而停止；可能是排序效率问题。这里的全候选扫描只用于事后定位失败，不能选择那个阈值部署，因为这样破坏固定序列保证。

certified：校准集已有通过项，但不代表外层测试一定有被接收病例。上述计数单位为模型—划分—预算，不是独立患者或独立重复。训练OOF分数与最终重拟合模型存在差异，也可能损害顺序质量。

## 总体服务率：预算20%

{overall}

![固定序列对照](../results/fixed_sequence_v6/fst_service_comparison.png)

其他组仍可服务时，总体自动率改善不能替代双缺失保护目标。自动通道中的经验阳性率与理论预算不是同一随机对象，少量测试病例超过预算也不自动证明理论失效。

## 主随机划分：预算20%

{primary}

## 程序验证

精确离散总体验证：全不安全、非单调混合、全安全三种风险结构，三个预算和三个每组校准量，每格5000次，共 {simulation_n} 个独立校准实验。使用独立训练计数学习顺序，并以已知总体风险判定是否错误认证。最大经验错误认证比例 {max_violation:.4f}；各格Monte Carlo区间详见 fst_exact_population_validation.csv。模拟是实现检查，不代替数学论证；多个单格95%区间不是全网格联合区间。

其他检查：首次失败后不再测试；空接收弃权；标量与向量排序一致；二项p值与单项CP对偶；患者角色不重叠；动作唯一；检验轨迹与最终阈值一致；原V4计数复现；原始与派生输入哈希不变。保存了 score_cache_LOCAL.csv.gz，后续固定基线可复用分数，避免重复训练。

## 结论边界

真实数据已被多轮分析，本轮仍是探索性对照，不能升级为新独立验证。主随机划分的保证依赖iid/分布稳定等假设；标签分层的重复CV用于经验稳健性，不自动继承理想iid校准的精确概率声明。V2遗留的影像时间、字段来源和临床编码限制未因本轮消失。

若仍零服务，可声称“已实现的LTT固定序列基线未恢复该组服务”；不可声称“所有风险控制方法都不可行”。后续最有区分度的实验应固定训练样本和评分模型，只改变独立校准人数，再单独改变信息可得性，区分训练、校准与信息损失的贡献；不能只在同一队列不断换排序追求正结果。

11项偏差检查覆盖：Simpson子群掩盖、生态推断、Berkson手术入组选择、检查路径collider、基准率、回归均值、幸存/入组、look-elsewhere多重选择、forking paths分析分叉、关联非因果、反向因果/时间窗；分别沿用分层报告、完整负结果、独立校准及对真实队列限制的明确声明，并不意味着全部偏差已消除。

协议 reports/v6_fixed_sequence_protocol.md；运行 src/run_fixed_sequence_v6.py；汇总 src/report_fixed_sequence_v6.py；输出 results/fixed_sequence_v6。患者级LOCAL文件仅供本地审计。
'''.format(verdict=verdict,overall_note=overall_note,double=md(ds),failure=md(failure),overall=md(overall),primary=md(primary),simulation_n=validation['independent_calibration_datasets'],max_violation=validation['maximum_empirical_violation'])
 path=ROOT/'reports/Better1_v6_LTT固定序列对照.md';path.write_text(report,encoding='utf-8')
 (O/'report_checks.json').write_text(json.dumps({'status':'PASS','trace_stop_and_rule_match':True,'FST_double_action_records':total,'FST_double_positive_records':positives,'report_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2),encoding='utf-8')
 print(verdict);print(failure.to_string(index=False));print(overall.to_string(index=False));print(path)
if __name__=='__main__':main()
