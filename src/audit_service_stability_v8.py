"""Post-hoc descriptive audit of V7; no new training or policy selection."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from scipy.stats import beta,t

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'results/calibration_information_v7/formal'
OUT=ROOT/'results/service_stability_v8'

def table(frame):
    return '| '+' | '.join(frame.columns)+' |\n|'+'|'.join(['---']*len(frame.columns))+'|\n'+'\n'.join('| '+' | '.join('NA' if pd.isna(v) else f'{v:.3f}' if isinstance(v,(float,np.floating)) else str(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None))

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    assert json.loads((SOURCE/'run.json').read_text())['status']=='COMPLETED'
    r=pd.read_csv(SOURCE/'replicate_results.csv')
    keys=['mechanism','model','calibration_n','policy','budget','mask_group']
    assert len(r)==216000 and not r.duplicated(keys+['repeat']).any()
    assert r.loc[r.reference_accepted.eq(0),'reference_risk'].isna().all()
    records=[]
    for key,z in r.groupby(keys):
        x=z.reference_service.to_numpy();n=len(x);k=int((x>0).sum())
        assert n==200 and z.repeat.nunique()==200
        records.append(dict(zip(keys,key),repetitions=n,service_mean=x.mean(),service_median=np.median(x),service_q10=np.quantile(x,.1),service_q90=np.quantile(x,.9),nonempty_runs=k,nonempty_fraction=k/n,nonempty_cp95_low=beta.ppf(.025,k,n-k+1) if k else 0.,nonempty_cp95_high=beta.ppf(.975,k+1,n-k) if k<n else 1.,cal_group_n_mean=z.actual_cal_group_n.mean()))
    s=pd.DataFrame(records);s.to_csv(OUT/'service_stability.csv',index=False)
    effects=[]
    for mechanism in ['MCAR','INFORMATIVE']:
      for budget in [.05,.1,.2]:
        z=r[(r.mechanism==mechanism)&(r.budget==budget)&r.mask_group.eq('missing_both')]
        for model in ['masked_lr','masked_hgb','complete_lr']:
            a=z[z.model.eq(model)&z.calibration_n.eq(50000)].pivot(index='repeat',columns='policy',values='reference_service')
            effects.append((mechanism,budget,model,'FST-minus-fixed63_at50000',a.G2_FST-a.G2_fixed63))
            a=z[z.model.eq(model)&z.policy.eq('G2_FST')].pivot(index='repeat',columns='calibration_n',values='reference_service')
            effects.append((mechanism,budget,model,'cal50000-minus150_FST',a[50000]-a[150]))
        a=z[z.calibration_n.eq(50000)&z.policy.eq('G2_FST')].pivot(index='repeat',columns='model',values='reference_service')
        effects.append((mechanism,budget,'LR','complete-minus-masked_at50000_FST',a.complete_lr-a.masked_lr))
    es=[]
    for mechanism,budget,model,contrast,x in effects:
        assert len(x)==200 and x.notna().all()
        mean=x.mean()*100;se=x.std(ddof=1)/np.sqrt(200)*100;half=t.ppf(.975,199)*se
        es.append(dict(mechanism=mechanism,budget=budget,model=model,contrast=contrast,mean_pp=mean,MCSE_pp=se,approx_mc95_low_pp=mean-half,approx_mc95_high_pp=mean+half,all_observed_differences_zero=bool((x==0).all())))
    e=pd.DataFrame(es);e.to_csv(OUT/'paired_effects.csv',index=False)
    v=s[s.mechanism.eq('INFORMATIVE')&s.mask_group.eq('missing_both')&s.calibration_n.eq(50000)&s.policy.eq('G2_FST')].copy()
    v['mean_pct']=100*v.service_mean;v['median_pct']=100*v.service_median
    v=v[['model','budget','mean_pct','median_pct','nonempty_runs','nonempty_cp95_low','nonempty_cp95_high','cal_group_n_mean']]
    p=e[e.mechanism.eq('INFORMATIVE')&e.budget.eq(.2)]
    report='''# V8：服务率是否稳定，较严格风险预算是否仍可用？

## Material Passport
Origin Skill: experiment-agent；Mode: validate；Version: service_stability_v8；Verification Status: ANALYZED。
日期：2026-09-18。本轮为V7既有结果的事后描述性复核，无新增训练、无新增临床证据、无阈值选择。

**实际结论：当前流程在信息性缺失的双缺失组上，尚未展示稳定的自动服务能力。** 即便全体校准样本达到50000（双缺失组平均9918.62），在20%预算下，缺失输入LR有114/200次、HGB有134/200次完全无服务，两者服务率中位数均为零。5%和10%预算下，两种缺失输入模型均为0/200次出现服务。这不等于所有算法不可能成功，而是当前生成机制、模型与认证流程的限制。

完整输入LR在10%预算下有194/200次出现服务、平均服务率32.28%；20%预算下200/200次出现服务、平均65.59%；5%预算下仍只有43/200次出现服务。因此即使信息完整，严格预算也可能带来明显的认证困难。以上全部为合成机制证据，不能解读为临床补查效果。

## 1. 跨风险预算检查
以下全部为信息性缺失机制、双缺失组、固定序列认证、全体校准人数50000。每行200次独立重复。mean/median是自动服务百分比；nonempty_runs是出现非空服务的重复次数。校准子群人数另列，避免把全体50000误认为双缺失人数。

{budget_table}

非空频率的区间为逐项双侧95%Clopper–Pearson区间，衡量该模拟配置在重新生成数据及重新训练后出现服务的频率，不是患者风险上界。200次全部无服务也不表示总体成功概率恰好为零。

## 2. 同次重复的配对差
下表为信息性缺失、预算20%的服务率差（百分点）。正值表示前者较高；两种缺失输入模型分别保留，不挑选较优者代替全部结果。

{effects}

区间为200个独立重复的配对均值近似Monte Carlo区间，不是临床置信区间；重复内六种样本量、模型和方法共享数据。此处为事后描述性分析，所有预算和机制均保存，未作多重比较后的显著性主张。全零差产生的[0,0]仅表示观察到的MC方差为零，不能证明总体差严格为零。

## 3. 当前证据能回答什么
- 不能只报平均自动率，还要看零服务频率与中位数。少量训练重复取得服务，不等于稳定可部署。
- 5%、10%、20%是实验预算，不代表临床可接受标准；不能在看到结果后仅保留20%。
- 完整输入对照只能说明生成机制中信息可得性对当前学习流程的影响；不是实际补查的因果收益，也不是所有算法可达到的上界。
- 固定序列与固定63重校正的配对差反映这两套已实现流程的差异，不能称为某一种检验方法普遍优越。
- 真实773例的双缺失组无认证服务这一结果未被本轮改变。下一项真正新增证据需分离模型、候选网格和排序的限制；继续增加相同模拟重复不会解决真实数据核实或独立验证。

## 4. 偏差核查（11/11）
Simpson：原表保留总体及四子群；生态谬误：不从组均值推断个人收益；Berkson：合成分析不消除手术队列选择；collider：检查路径不作因果解释；基准率：不同机制不视为可交换临床人群；回归均值：配对比较全部重复；幸存者偏差：零服务重复保留；多重搜索：完整输出并不宣称联合区间；分析分叉：明确事后分析；相关与因果：完整输入不等于补查疗效；反向因果：不替代真实术前时序核实。

## 5. 可复现文件
输入：results/calibration_information_v7/formal/replicate_results.csv。
输出：results/service_stability_v8/service_stability.csv、paired_effects.csv、validation.json。
原始数据、V7结果与所有已冻结规则均未更改。分析脚本为src/audit_service_stability_v8.py。
'''.format(budget_table=table(v),effects=table(p))
    path=ROOT/'reports/Better1_v8_服务稳定性与严格预算复核.md';path.write_text(report,encoding='utf-8')
    validation={'status':'PASS','source_rows':len(r),'summary_rows':len(s),'paired_contrasts':len(e),'independent_repeats_per_cell':200,'new_model_fits':0,'source_sha256':hashlib.sha256((SOURCE/'replicate_results.csv').read_bytes()).hexdigest(),'analysis_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'checks':['unique complete keys','200 independent repeat IDs per cell','empty service risk remains undefined','paired comparisons without missing repeats']}
    (OUT/'validation.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
    print(v.to_string(index=False));print(p.to_string(index=False));print(path)

if __name__=='__main__':main()
