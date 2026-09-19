"""Compare numerical observed-information oracle with paired V9 scorers."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
from scipy.stats import t
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/observed_oracle_v10/formal'
def md(df):
 return '| '+' | '.join(df.columns)+' |\n|'+'|'.join(['---']*len(df.columns))+'|\n'+'\n'.join('| '+' | '.join('NA' if pd.isna(v) else f'{v:.4f}' if isinstance(v,(float,np.floating)) else str(v) for v in row)+' |' for row in df.itertuples(index=False,name=None))
def main():
 meta=json.loads((OUT/'run.json').read_text());assert meta['status']=='COMPLETED' and meta['completed']==200
 checks=json.loads((OUT/'numerical_checks.json').read_text());assert checks['status']=='PASS'
 oracle=pd.read_csv(OUT/'replicate_results.csv');old=pd.read_csv(ROOT/'results/grid_order_v9/formal/replicate_results.csv')
 old=old[old.method.isin(['anchor2500','bonferroni'])];r=pd.concat([oracle,old],ignore_index=True)
 keys=['model','grid','method','budget'];assert not r.duplicated(keys+['repeat']).any()
 assert r.groupby(['repeat','grid','method','budget']).reference_n.nunique().eq(1).all()
 s=r.groupby(keys).agg(repetitions=('repeat','size'),service_mean=('service','mean'),service_median=('service','median'),nonempty_runs=('reference_accepted',lambda x:int((x>0).sum())),risk_mean_served=('reference_risk','mean')).reset_index();assert s.repetitions.eq(200).all()
 s['service_mean_pct']=100*s.service_mean;s['service_median_pct']=100*s.service_median;s.to_csv(OUT/'comparison_summary.csv',index=False)
 effects=[]
 for (grid,method,budget),z in r.groupby(['grid','method','budget']):
  p=z.pivot(index='repeat',columns='model',values='service')
  for model in ['masked_lr','masked_hgb']:
   x=(p.observed_oracle-p[model])*100;assert len(x)==200 and x.notna().all();se=x.std(ddof=1)/np.sqrt(200);half=t.ppf(.975,199)*se
   effects.append(dict(grid=grid,method=method,budget=budget,contrast='oracle-minus-'+model,mean_pp=x.mean(),MCSE_pp=se,approx_mc95_low_pp=x.mean()-half,approx_mc95_high_pp=x.mean()+half))
 e=pd.DataFrame(effects);e.to_csv(OUT/'paired_effects.csv',index=False)
 f=pd.read_csv(OUT/'reference_diagnostic.csv');fs=f.groupby('budget').agg(repetitions=('repeat','size'),service_mean=('service','mean'),service_median=('service','median'),nonempty_runs=('accepted',lambda x:int((x>0).sum())),oracle_risk_mean=('score_mean','mean'),latent_risk_mean=('latent_probability_mean','mean')).reset_index();fs.to_csv(OUT/'reference_diagnostic_summary.csv',index=False)
 view=s[s.grid.eq('original21')&s.method.eq('anchor2500')][['model','budget','service_mean_pct','service_median_pct','nonempty_runs','risk_mean_served']]
 full=s[s.model.eq('observed_oracle')][['grid','method','budget','service_mean_pct','service_median_pct','nonempty_runs','risk_mean_served']]
 report='''# V10：可见信息本身不足，还是评分模型没有学好？

## Material Passport
Origin Skill: experiment-agent；Mode: run/validate；Version: observed_oracle_v10；Verification Status: ANALYZED。
2026-09-18；探索性合成机制对照，200次重复，无新增模型拟合，无临床输入。

## 本轮实际结论
在同一原网格与固定序列认证下，20%预算的平均自动服务率：理想分数8.45%、现有LR 5.41%、HGB 3.64%。理想分数分别提高3.04和4.82个百分点，说明在该模拟器中，当前学习流程确实没有充分利用全部可见信息。因此应避免将此前结果笼统归结为“缺失导致没有信息”。

但理想分数仍有84/200次无服务；加密网格后平均服务率10.96%，仍有64/200次无服务。5%与10%预算下，理想分数的全部认证配置仍为200次均无服务。因此“只要模型更好就能稳定解决”同样不受本轮支持。

理想风险排序的参考样本前缀诊断平均服务比例分别约为0.025%（5%预算）、0.717%（10%预算）、29.914%（20%预算）。在此生成机制中，严格预算下可见信息所能识别的低风险人群很少；20%预算下参考诊断与实际认证服务之间仍有较大差距。该差距包含候选网格、独立排序、有限校准等因素，不能全部归给某一因素，也不是精确总体上界。

当前更准确的解释是三方面共同限制：可见信息下低风险人群的规模、实际模型学到的分数质量，以及有限样本认证代价。本轮识别了这些差异，但未完成可加总的因果分解；更未验证真实临床队列存在相同机制。

## 1. 理想分数是什么
这里只研究信息性缺失机制的双缺失组。把模拟器真实条件风险对不可见变量积分，得到s=P(Y=1|z1,z2,双缺失)。它只输入两个可见变量和缺失状态，不读取个体的隐藏检查结果、潜变量或标签。生成器参数已知，是理想对照，不能当作临床可直接使用的新模型。

logit代数整理为c+1.48u+0.49(e1+e2)，c=-2+0.71z1+0.61z2+0.3z1z2。给定双缺失时，u的分布需按q2(u)加权，不能简单沿用未条件化的标准正态；这一步保留了信息性缺失机制。

64与96阶Gauss–Hermite积分在257个固定检查点的最大差为{quad}；插值最大差为{interp}。这些是数值检查结果，不是整个实数域的解析误差证明。超出预计算范围时直接积分，不截断分数。数值误差可能影响理想分数解释，但独立标签认证仍按同一固定规则执行。

## 2. 同一原网格、同一认证流程的直接比较
下面保留原21阈值和2500锚点固定序列，避免用换认证方法解释评分差异。全体校准50000、独立排序2000、参考50000；200次种子与V9逐次配对，参考双缺失人数核对一致。服务率列为百分比，非空次数分母200。

{view}

## 3. 理想分数的所有认证配置
{full}

## 4. 原配置的配对差（百分点）
{effects}

区间仅为200次重复下的配对均值近似Monte Carlo区间，不是临床区间，也未作跨方法联合推断。全零观察差不证明总体差为零。其余网格/方法的配对差完整保留在paired_effects.csv。

## 5. 独立参考样本上的理想前缀诊断
将参考病例按理想分数排序，取平均理想风险不超过预算的最大前缀。下表服务率为0–1比例。它不受候选网格或有限校准认证限制，但使用参考输入挑选前缀，因此只是经验诊断；不部署、不当成精确总体上界。latent_risk_mean仅用于评估，与个体潜变量有关的真实生成概率从未输入分数或认证。

{frontier}

## 6. 解释边界与后续判断
若理想分数比现有模型恢复更多服务，说明已有可见信息未被当前学习流程充分利用；不能再把该差距全部解释为信息缺失。理想分数依赖已知模拟器，差距同时包含有限训练样本、函数表示、拟合和分数校准的影响，不证明某一种可训练模型能消除差距。
若理想分数仍无认证服务，应结合参考前缀诊断区分评分/信息限制与认证障碍，不能单凭认证失败证明信息论不可能。本轮没有完整信息Bayes对照，因此没有量化缺失造成的最优风险差。
真实773例结果与数据待核实事项未改变；不能将模拟器中的改善包装为临床补查收益或跨中心泛化。各固定配置单独认证，不宣称多个预算、网格、方法、模型联合95%保证。

## 7. 偏差核查与复现
11/11核查：Simpson（仅双缺失，不冒充总体）；生态谬误（均值非个体收益）；Berkson（不消除手术队列选择）；collider（条件分布明确加权，非临床因果）；基准率（生成器固定）；回归均值（所有重复配对）；幸存者偏差（保留零服务）；多重搜索（全预算/配置报告）；分析分叉（先冻结V10，但由V9启发、属探索）；相关与因果（非补查疗效）；反向因果（不替代术前时序核实）。
协议reports/v10_observed_oracle_protocol.md；代码src/observed_oracle_v10.py、src/report_observed_oracle_v10.py；结果results/observed_oracle_v10/formal。原V9文件和临床数据未修改。
'''.format(quad=checks['max_quadrature_difference'],interp=checks['max_interpolation_difference'],view=md(view),full=md(full),effects=md(e[e.grid.eq('original21')&e.method.eq('anchor2500')]),frontier=md(fs))
 path=ROOT/'reports/Better1_v10_可见信息理想分数对照.md';path.write_text(report,encoding='utf-8')
 (OUT/'report_checks.json').write_text(json.dumps({'status':'PASS','oracle_rows':len(oracle),'comparison_rows':len(r),'reference_group_counts_match':True,'paired_repetitions':200,'source_sha256':hashlib.sha256((OUT/'replicate_results.csv').read_bytes()).hexdigest()},indent=2),encoding='utf-8')
 print(view.to_string(index=False));print(full.to_string(index=False));print(fs.to_string(index=False));print(path)
if __name__=='__main__':main()
