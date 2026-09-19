"""Report all V9 methods, never select a deployment policy from reference data."""
import json,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import t
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/grid_order_v9/formal'
def md(df):
 return '| '+' | '.join(df.columns)+' |\n|'+'|'.join(['---']*len(df.columns))+'|\n'+'\n'.join('| '+' | '.join('NA' if pd.isna(v) else f'{v:.3f}' if isinstance(v,(float,np.floating)) else str(v) for v in row)+' |' for row in df.itertuples(index=False,name=None))
def main():
 meta=json.loads((OUT/'run.json').read_text());assert meta['status']=='COMPLETED' and meta['baseline_checks']==1200
 r=pd.read_csv(OUT/'replicate_results.csv');assert len(r)==9600
 keys=['model','grid','method','budget'];assert not r.duplicated(keys+['repeat']).any()
 s=r.groupby(keys).agg(repeats=('repeat','size'),service=('service','mean'),median=('service','median'),nonempty_runs=('reference_accepted',lambda x:int((x>0).sum())),risk_served=('reference_risk','mean')).reset_index()
 assert (s.repeats==200).all();s.to_csv(OUT/'summary.csv',index=False)
 records=[]
 for model in ['masked_lr','masked_hgb']:
  for alpha in [.05,.1,.2]:
   z=r[r.model.eq(model)&r.budget.eq(alpha)].pivot(index='repeat',columns=['grid','method'],values='service')
   contrasts=[]
   for method in ['anchor2500','matched50000','risk_ascending','bonferroni']:
    contrasts.append(('dense-minus-original_'+method,z['dense_union',method]-z['original21',method]))
   for grid in ['original21','dense_union']:
    for method in ['matched50000','risk_ascending','bonferroni']:
     contrasts.append((grid+'_'+method+'-minus-anchor2500',z[grid,method]-z[grid,'anchor2500']))
   for label,x in contrasts:
    assert len(x)==200 and x.notna().all();se=x.std(ddof=1)/np.sqrt(200)*100;mean=x.mean()*100;half=t.ppf(.975,199)*se
    records.append(dict(model=model,budget=alpha,contrast=label,mean_pp=mean,MCSE_pp=se,approx_mc95_low_pp=mean-half,approx_mc95_high_pp=mean+half))
 e=pd.DataFrame(records);e.to_csv(OUT/'paired_effects.csv',index=False)
 view=s[s.budget.eq(.2)].copy();view['mean_pct']=100*view.service;view['median_pct']=100*view['median'];view=view[['model','grid','method','mean_pct','median_pct','nonempty_runs','risk_served']]
 strict=s[s.budget.lt(.2)].copy();strict['mean_pct']=100*strict.service;strict=strict[['model','grid','method','budget','mean_pct','nonempty_runs']]
 report='''# V9：阈值网格与检验顺序的配对对照

## Material Passport
Origin Skill: experiment-agent；Mode: run/validate；Version: grid_order_v9；Verification Status: ANALYZED。
2026-09-18。探索性后续合成实验，未读取或改变真实临床数据。

## 本轮结论
加密网格、保留原2500锚点排序时，20%预算下LR服务率由5.41%升至6.45%，HGB由3.64%升至4.65%，分别增加1.03和1.01个百分点。但两种模型分别仍有106/200、123/200次无服务；所有16个模型—网格—方法配置的服务率中位数均为零。5%和10%预算下全部配置的200次重复均无服务。

匹配50000校准规模的功效排序没有一致改善：原网格LR较原排序低0.92个百分点，HGB高0.19个百分点；加密网格下二者也未超过原排序。不能将“排序更贴近规模”直接等同于更有效。

因此，在本轮测试的模型、网格与排序范围内，流程调整有小幅收益，但未解决稳定服务问题。不能把低服务率仅解释为原网格过粗或2500锚点设置失当；也不能据此证明信息论不可能，尚未覆盖其他模型、排序及候选集合。

## 实验范围与核验
信息性缺失，双缺失组，200次独立重复、400个模型。训练5000、排序2000、校准50000、参考50000。所有流程共享每次冻结的模型和预测；排序仅使用独立排序集。原21阈值/2500锚点的1200个重复—模型—预算配置与V7逐项核对一致。

原网格与加密并集分别比较4种流程：anchor2500为V7功效排序；matched50000匹配已知校准规模；risk_ascending为排序集估计风险升序；bonferroni对所有阈值作多重校正。新增方法没有用参考结果挑阈值或顺序。仅认证双缺失组，仍使用G2预留置信预算0.05/3；其他子群不由本轮认证。

## 预算20%的全部结果
mean_pct、median_pct是双缺失组自动服务百分比；nonempty_runs分母200。risk_served是非空服务重复中的参考风险均值，不能替代逐规则风险保证。

{main}

## 较严格预算的全部结果
{strict}

## 配对效应（20%预算）
差值单位为百分点；同一次重复相减后计算均值和Monte Carlo标准误。区间为逐项近似95%区间，不是联合区间、不作多重检验后的显著性主张；全零观察差的退化区间不能证明总体差严格为零。

{effects}

## 解释边界
加密网格改变候选集合和多重校正负担；排序改变首个失败的位置。即使改善，也只说明当前流程的限制，不能称为新方法的普遍优势。固定序列在某个阈值失败即停止；不保证按校准规模匹配的排序一定最好。全部结果保留，不从参考结果选取部署策略。

已知生成概率的参考均值仍是有限样本估计，不是精确总体风险。无服务风险记NA。每个配置单独的风险保证不等于跨网格、方法、模型和预算联合保证。真实队列的结论和临床待核实事项不因本轮模拟而改变。

## 11项偏差核查
Simpson：本轮明确仅双缺失、不能冒充总体；生态谬误：均值非个体收益；Berkson：不解除手术入组限制；collider：缺失机制不作临床因果解释；基准率：生成机制固定；回归均值：全重复配对；幸存者偏差：保留零服务；多重搜索：所有方法与预算均报告、不宣称联合推断；分析分叉：方案先冻结但研究问题由V7启发，属于探索；相关与因果：非临床补查收益；反向因果：不替代术前时序核实。

## 复现
协议：reports/v9_grid_order_protocol.md。执行：src/grid_order_v9.py；汇总：src/report_grid_order_v9.py。
结果：results/grid_order_v9/formal/replicate_results.csv、summary.csv、paired_effects.csv、run.json、report_checks.json。
'''.format(main=md(view),strict=md(strict),effects=md(e[e.budget.eq(.2)]))
 path=ROOT/'reports/Better1_v9_阈值网格与检验顺序对照.md';path.write_text(report,encoding='utf-8')
 (OUT/'report_checks.json').write_text(json.dumps({'status':'PASS','rows':len(r),'baseline_checks':meta['baseline_checks'],'unique_keys':True,'summary_cells':len(s),'source_sha256':hashlib.sha256((OUT/'replicate_results.csv').read_bytes()).hexdigest()},indent=2),encoding='utf-8')
 print(view.to_string(index=False));print(strict.to_string(index=False));print(e[e.budget.eq(.2)].to_string(index=False));print(path)
if __name__=='__main__':main()
