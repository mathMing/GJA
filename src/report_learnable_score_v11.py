"""Full paired report of prespecified learnable scorer ablation."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
from scipy.stats import t
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/learnable_score_v11/formal'
def md(df):
 return '| '+' | '.join(df.columns)+' |\n|'+'|'.join(['---']*len(df.columns))+'|\n'+'\n'.join('| '+' | '.join('NA' if pd.isna(v) else f'{v:.4f}' if isinstance(v,(float,np.floating)) else str(v) for v in row)+' |' for row in df.itertuples(index=False,name=None))
def main():
 run=json.loads((OUT/'run.json').read_text());assert run['status']=='COMPLETED' and run['model_fits']==800 and run['baseline_checks']==1200
 r=pd.read_csv(OUT/'replicate_results.csv');assert len(r)==4800
 oracle=pd.read_csv(ROOT/'results/observed_oracle_v10/formal/replicate_results.csv');oracle=oracle[oracle.method.eq('anchor2500')]
 both=pd.concat([r,oracle]);keys=['model','grid','budget'];assert not both.duplicated(keys+['repeat']).any()
 assert both.groupby(['repeat','grid','budget']).reference_n.nunique().eq(1).all()
 s=both.groupby(keys).agg(repetitions=('repeat','size'),service_mean=('service','mean'),service_median=('service','median'),nonempty_runs=('reference_accepted',lambda x:int((x>0).sum())),risk_served=('reference_risk','mean')).reset_index();assert s.repetitions.eq(200).all()
 s['mean_pct']=100*s.service_mean;s['median_pct']=100*s.service_median;s.to_csv(OUT/'comparison_summary.csv',index=False)
 effects=[]
 for (grid,alpha),z in both.groupby(['grid','budget']):
  p=z.pivot(index='repeat',columns='model',values='service')
  pairs=[('pooled_poly2','pooled_linear'),('double_poly2','double_linear'),('double_linear','pooled_linear'),('double_poly2','pooled_poly2'),('double_poly2','pooled_linear'),('observed_oracle','double_poly2')]
  for a,b in pairs:
   x=(p[a]-p[b])*100;assert len(x)==200 and x.notna().all();se=x.std(ddof=1)/np.sqrt(200);half=t.ppf(.975,199)*se
   effects.append(dict(grid=grid,budget=alpha,contrast=a+'-minus-'+b,mean_pp=x.mean(),MCSE_pp=se,approx_mc95_low_pp=x.mean()-half,approx_mc95_high_pp=x.mean()+half))
 e=pd.DataFrame(effects);e.to_csv(OUT/'paired_effects.csv',index=False)
 training=pd.read_csv(OUT/'training_counts.csv').groupby('model').agg(fit_n_mean=('fit_n','mean'),fit_n_min=('fit_n','min'),fit_n_max=('fit_n','max'),fit_positive_mean=('fit_positive','mean'),feature_n=('feature_n','max')).reset_index();training.to_csv(OUT/'training_summary.csv',index=False)
 v=s[['model','grid','budget','mean_pct','median_pct','nonempty_runs','risk_served']]
 report='''# V11：把理想分数提示转化为可训练的评分对照

## Material Passport
Origin Skill: experiment-agent；Mode: run/validate；Version: learnable_score_v11；Verification Status: ANALYZED。
2026-09-18。200次独立模拟重复、800次拟合、4800条结果；与V9对应的1200项基线结果核对一致。无临床输入、无新外部证据。

## 本轮实际结论
全体训练的二次特征LR在20%预算下有小幅恢复：原网格由5.41%升至6.69%（配对差1.27个百分点）；加密网格由6.45%升至8.35%（配对差1.91个百分点）。这说明不依赖已知生成器参数的可训练模型可以缩小部分评分差距，但仍不能称为稳定解决。

加密网格下pooled_poly2有110/200次出现服务、90/200次完全无服务；原网格仅97/200次出现服务。四个可训练模型在5%与10%预算下所有重复均无服务。子群单独训练及其二次项没有显示一致优势；例如原网格double_poly2平均5.78%，低于pooled_poly2的6.69%。子群平均训练992.52人，而全体训练5000人，因此不能把差异直接归因于是否分组。

本轮不支持把“对子群单独训练模型”作为已验证方法贡献，也不足以恢复临床可用性主张。当前正向证据仅限于：在这一生成机制中，可训练的非线性表达能带来有限服务改善。后续若验证，应冻结模型后使用新随机种子和不同机制；不继续在同一模拟器上试参直到获得更好结果。

## 1. 固定的实验对照
pooled_linear：原始全体训练LR；pooled_poly2：填补和缺失指示后增加二次项及交互；double_linear：只在双缺失训练病例上拟合z1,z2；double_poly2：在该子群上增加z1,z2二次项和交互。均为C=1的LR，无调参。所有预处理仅在相应训练角色拟合，不读取生成概率或不可见检查。observed_oracle来自V10，依赖已知生成器，仅为参考。
全部模型使用独立排序2000、全体校准50000、参考50000。同一重复共享数据，检验顺序由排序集独立确定，不能使用参考结果。固定序列的功效锚点2500、置信预算0.05/3不变。两种网格、三个风险预算全部保留。

## 2. 20%预算的结果
mean/median为双缺失组自动服务百分比，非空次数分母200；risk_served只对非空服务重复取均值。

{view}

## 3. 较严格预算
{strict}

## 4. 训练人数与输入维度
{training}

子群训练与全体训练人数不同，因此这是训练策略比较，不能称为控制样本量后的纯子群效应。二次项也增加了特征维度；固定C不等于有效复杂度相同。

## 5. 配对差（20%预算，百分点）
{effects}

区间为200次独立重复配对均值的近似Monte Carlo区间，不是患者风险区间。无跨方法联合置信主张、不据参考结果选择部署模型。全部预算的配对差保留在CSV。观察到全零差时，退化区间不能证明总体差为零。

## 6. 解释边界
本轮由理想分数的已知公式启发，属于模拟器知情的探索性对照。即便可训练模型改善，也只能说明这种表示或训练策略在当前机制下有潜力，不能外推到真实临床、陌生机制或更小训练样本。训练/排序/认证/测试的角色隔离不等于整个研究从未看过该机制。
理想条件风险分数不保证在有限校准、离散网格和特定排序下的认证服务率最大；某个学习模型若超过它，也不能称为超过信息上界。零服务的风险保持NA，而非0。
若本轮改善，应先独立机制/种子验证，再考虑真实数据预先冻结的评分消融；不在已反复查看的773例上事后找最佳组合，不新增MRI训练或临床补查收益主张。

## 7. 11项偏差核查与复现
Simpson：只报告双缺失，非总体；生态谬误：组均值非个体收益；Berkson：不解除真实手术队列选择；collider：分组学习不作临床因果解释；基准率：机制相同但不外推；回归均值：完整200次配对；幸存者偏差：零服务保留；多重搜索：全模型/预算/网格报告；分析分叉：运行前冻结但明确探索；相关与因果：非补查疗效；反向因果：不替代术前核实。
协议reports/v11_learnable_score_protocol.md；代码src/learnable_score_v11.py、src/report_learnable_score_v11.py；结果results/learnable_score_v11/formal。未修改临床原表或V9/V10结果。
'''.format(view=md(v[v.budget.eq(.2)]),strict=md(v[v.budget.lt(.2)]),training=md(training),effects=md(e[e.budget.eq(.2)]))
 path=ROOT/'reports/Better1_v11_可训练非线性与子群评分对照.md';path.write_text(report,encoding='utf-8')
 (OUT/'report_checks.json').write_text(json.dumps({'status':'PASS','rows':len(r),'baseline_checks':run['baseline_checks'],'reference_group_counts_match':True,'source_sha256':hashlib.sha256((OUT/'replicate_results.csv').read_bytes()).hexdigest()},indent=2),encoding='utf-8')
 print(v.to_string(index=False));print(training.to_string(index=False));print(e[e.budget.eq(.2)].to_string(index=False));print(path)
if __name__=='__main__':main()
