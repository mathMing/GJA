"""Full frozen-validation reporting with prespecified 18 primary contrasts."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
from scipy.stats import t
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/frozen_validation_v12/formal'
def md(df):
 return '| '+' | '.join(df.columns)+' |\n|'+'|'.join(['---']*len(df.columns))+'|\n'+'\n'.join('| '+' | '.join('NA' if pd.isna(v) else f'{v:.3f}' if isinstance(v,(float,np.floating)) else str(v) for v in row)+' |' for row in df.itertuples(index=False,name=None))
def main():
 run=json.loads((OUT/'run.json').read_text());assert run['status']=='COMPLETED' and run['model_fits']==2400
 r=pd.read_csv(OUT/'replicate_results.csv');assert len(r)==14400
 keys=['mechanism','model','grid','budget'];assert not r.duplicated(keys+['repeat']).any()
 s=r.groupby(keys).agg(repetitions=('repeat','size'),service=('service','mean'),median=('service','median'),nonempty_runs=('reference_accepted',lambda x:int((x>0).sum())),risk_served=('reference_risk','mean')).reset_index();assert s.repetitions.eq(200).all()
 s['mean_pct']=100*s.service;s['median_pct']=100*s['median'];s.to_csv(OUT/'summary.csv',index=False)
 effects=[]
 for (mode,grid,budget),z in r.groupby(['mechanism','grid','budget']):
  p=z.pivot(index='repeat',columns='model',values='service');x=(p.pooled_poly2-p.pooled_linear)*100
  assert len(x)==200 and x.notna().all();se=x.std(ddof=1)/np.sqrt(200);half=t.ppf(.975,199)*se;simhalf=t.ppf(1-.05/(2*18),199)*se
  effects.append(dict(mechanism=mode,grid=grid,budget=budget,mean_pp=x.mean(),MCSE_pp=se,pointwise_low_pp=x.mean()-half,pointwise_high_pp=x.mean()+half,approx_simultaneous_low_pp=x.mean()-simhalf,approx_simultaneous_high_pp=x.mean()+simhalf,all_zero=bool((x==0).all())))
 e=pd.DataFrame(effects);assert len(e)==18;e.to_csv(OUT/'primary_paired_effects.csv',index=False)
 view=s[['mechanism','model','grid','budget','mean_pct','median_pct','nonempty_runs','risk_served']]
 old=pd.read_csv(ROOT/'results/learnable_score_v11/formal/replicate_results.csv')
 old=old[old.model.isin(['pooled_linear','pooled_poly2'])&old.budget.eq(.2)].groupby(['model','grid']).service.mean().rename('V11_service')
 new=s[s.mechanism.eq('INFORMATIVE')&s.model.isin(['pooled_linear','pooled_poly2'])&s.budget.eq(.2)].set_index(['model','grid']).service.rename('V12_service')
 replication=pd.concat([old,new],axis=1).reset_index();replication.to_csv(OUT/'replication_comparison.csv',index=False)
 report='''# V12：冻结评分流程的新种子、多机制复核

## Material Passport
Origin Skill: experiment-agent；Mode: run/validate；Version: frozen_validation_v12；Verification Status: ANALYZED。
2026-09-18。三机制各200次新种子重复，共600组数据、2400次拟合、14400条结果。

## 当前判定：下调评分改进主张，保留机制依赖与失效边界
信息性缺失、20%预算下，pooled_poly2相对普通LR在原网格的平均收益为1.71个百分点；加密网格仅0.26个百分点，低于V11的1.91个百分点。两项18重比较的近似同时区间分别为[-0.24,3.67]与[-1.77,2.30]个百分点，均包含零。因此不能把V11的小幅改善表述为已经得到稳健复现；但区间包含零也不证明效果严格为零。

18项主比较中，只有MCAR原网格20%预算的近似同时区间完全大于零，收益1.65个百分点。不能借这一个结果宣称二次特征普遍有效，也不根据本次参考结果重新选择网格。

更稳定的描述性发现是流程表现强烈依赖机制：原网格20%预算下普通LR/二次LR自动率分别为MCAR 87.42%/89.07%、MAR 52.02%/53.78%、信息性缺失4.89%/6.60%。信息性缺失下二次LR在原网格仍有100/200次无服务，在加密网格有114/200次无服务；5%和10%预算下所有四模型仍无服务。这是各完整生成机制的比较，未保持子群基准风险等因素不变，不能将全部差异单独归因于某一个缺失参数。

当前证据支持把评分改进留作消融，把主要解释收缩为“安全服务能力的机制依赖与失效边界”。不支持新增一个强方法优越性或临床可用性主张。此判断只针对现有模拟与真实实验，尚不是论文可接受性结论。

## 1. 本轮改变与保持
只改变随机种子，并将V11冻结的四种训练策略同时用于MCAR、MAR和INFORMATIVE。训练5000、独立排序2000、校准50000、参考50000；同次重复共享各角色的数据，模型间作配对。四模型、两网格、三个预算、固定序列锚点2500均未调整。模型构建直接导入V11，并记录代码哈希。
三个机制均曾研究过，因此这是冻结流程在新随机实现上的复核，不是未见机制泛化，更不是独立外部临床验证。只评估双缺失组；delta仍为G2预留的0.05/3，不认证其他组。

## 2. 20%预算下全部模型结果
mean/median为自动服务百分比；非空次数分母200；risk_served仅在非空重复求平均，不能取代单项风险保证。

{main}

## 3. 预先指定的18个主要配对差
全部为pooled_poly2-minus-pooled_linear，单位百分点。各行均以200次独立重复为单位，不将同次重复的配置当成独立样本。同时给出逐项95%及18项Bonferroni/t近似同时区间；后者只衡量模拟均值差，不是部署风险保证。全零观察差的退化区间不证明总体差为零。未据比较结果选择部署模型。

{effects}

## 4. 信息性缺失：与V11比较
以下服务率为0–1比例；新旧种子不逐项配对，此表只展示可重复程度，不是临床复制。

{replication}

## 5. 5%与10%预算的完整结果
{strict}

## 6. 结论边界
若二次特征收益在新种子中恢复，说明原发现并非只由那一组随机实现产生；若在不同机制下变化，必须报告机制依赖，不能称为普遍改善。训练子群模型的人数小于全体模型，比较的是训练策略整体。模型改进与信息可得性、样本量、阈值网格和认证顺序并非相互独立的可加总因素。
本轮未评估新临床患者，真实773例双缺失组无认证服务的结论不变。零服务风险保持NA，不把弃权视为风险为零。单个配置的认证不意味着所有模型/网格/预算有联合保证。没有新增MRI训练、真实补查记录或外部中心支持。

## 7. 11项偏差核查
Simpson：只评估双缺失，机制分别报告；生态谬误：均值非个体效果；Berkson：不消除临床选择；collider：机制关联非临床因果；基准率：机制不同不视为同一临床分布；回归均值：新种子与全部重复；幸存者偏差：零服务保留；多重搜索：18项主要比较预先固定且全报告；分析分叉：不改模型，但承认机制早已研究；相关与因果：非补查收益；反向因果：非真实术前时序核实。

## 8. 复现
协议reports/v12_frozen_validation_protocol.md；运行src/frozen_validation_v12.py；报告src/report_frozen_validation_v12.py；结果results/frozen_validation_v12/formal。任何先前实验、模型工厂与临床原表均未修改。
'''.format(main=md(view[view.budget.eq(.2)]),effects=md(e),replication=md(replication),strict=md(view[view.budget.lt(.2)]))
 path=ROOT/'reports/Better1_v12_冻结模型新种子多机制验证.md';path.write_text(report,encoding='utf-8')
 (OUT/'report_checks.json').write_text(json.dumps({'status':'PASS','rows':len(r),'independent_datasets':600,'model_fits':2400,'primary_contrasts':18,'source_sha256':hashlib.sha256((OUT/'replicate_results.csv').read_bytes()).hexdigest()},indent=2),encoding='utf-8')
 print(view[view.budget.eq(.2)].to_string(index=False));print(e.to_string(index=False));print(replication.to_string(index=False));print(path)
if __name__=='__main__':main()
