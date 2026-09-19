"""Report all fixed-score diagnostic comparators without endorsing label selection."""
import json
import pandas as pd
import v2_pipeline as v
from report_numeric_v16 import md

ROOT=v.ROOT;OUT=ROOT/'results/service_bottleneck_v18'

def main():
 run=json.loads((OUT/'run.json').read_text());assert run['status']=='COMPLETED'
 d=pd.read_csv(OUT/'bottleneck_cells.csv');r=pd.read_csv(OUT/'fold_metrics.csv');s=pd.read_csv(OUT/'seed_metrics.csv')
 breakdown=d.groupby(['risk_budget','reason']).size().rename('fold_configuration_cells').reset_index()
 detail=d.groupby(['cohort','model','configuration','risk_budget','reason']).size().rename('cells').reset_index()
 detail.to_csv(OUT/'bottleneck_summary.csv',index=False)
 compare=r.groupby(['risk_budget','method','certificate_valid']).agg(cells=('auto_n','size'),nonempty_test_cells=('auto_n',lambda x:int((x>0).sum()))).reset_index()
 ss=s.groupby(['cohort','model','configuration','risk_budget','method','certificate_valid']).agg(mean_service=('service','mean'),min_service=('service','min'),max_service=('service','max'),mean_auto_n=('auto_n','mean'),mean_auto_positive_n=('auto_positive_n','mean')).reset_index()
 ss.to_csv(OUT/'method_summary.csv',index=False)
 hypothetical=pd.read_csv(OUT/'hypothetical_exact_power.csv');size=hypothetical[['risk_budget','assumed_true_risk','first_n_reaching_90pct','search_max_n']].drop_duplicates()
 cap=d.groupby(['cohort','risk_budget']).agg(cal_n_min=('cal_group_n','min'),cal_n_median=('cal_group_n','median'),cal_n_max=('cal_group_n','max'),min_zero_error_accepted=('min_zero_error_accepted','first')).reset_index()
 assert len(d)==936 and len(r)==4680
 assert r.loc[~r.certificate_valid,'risk_bound'].isna().all() and r.loc[r.auto_n.eq(0),'risk'].isna().all()
 labels={'CALIBRATION_CAPACITY_INSUFFICIENT':'校准总人数不足：即使全部零阳性也无法通过', 'NO_POINTWISE_PASS':'容量不排除认证，但当前所有阈值均未通过点态检验', 'ORDER_FIRST_FAILURE':'有点态通过候选，但独立排序的首失败规则阻止认证','FST_ACTIVE':'FST已激活'}
 readable=breakdown.copy();readable['reason']=readable.reason.map(labels)
 report=f'''# V18：为什么AI评分无法为双缺失组提供认证服务？

Origin Skill: experiment-agent；Mode: run/validate；Verification Status: ANALYZED。2026-09-19。

## 本轮回答的范围
固定V17评分，不重训、不换患者角色，完成{len(d)}个“队列×模型×字段×划分×预算”诊断；{run['baseline_patient_checks']}行双缺失患者FST动作与阈值复现V17。比较的是双缺失这一块，不是重新运行整套G2临床策略。下面统计配置次数，不是独立患者或独立研究次数。

## 本轮核心发现
**5%和10%预算在当前校准规模下均存在硬性容量障碍。** 双缺失校准组仅14–34人，而每块delta=0.05/3时，即使零阳性也分别需要80、39名被接纳的校准患者；两个预算下各312个配置全部不满足。这是当前分割、检验与置信分配的边界，不是所有可能方法都不可能。

**20%预算下，64/312个配置总容量不足；其余248个配置也没有一个候选通过点态检验。** 有点态通过而仅被FST排序阻止的配置为0。因此，当前零服务不能直接归咎于21阈值搜索校正或首失败停止；此结论仍保留每块置信分配，不等于所有多重性代价都已排除。

**普通未校正搜索仍为零服务；看测试标签却常能挑出表面低风险病例。** 后者在三个预算下分别238、243、287个配置出现非零服务，只能体现事后看答案的乐观性和样本波动，不能证明真实可泛化服务能力。下一轮不应再单纯堆叠阈值搜索模块，应围绕低风险排序与有效校准样本开展实验，并继续临床核实。

## 1. 认证卡在哪里
互斥分类按“总容量→点态候选→独立排序”顺序定义。它定位当前流程的障碍，不是三个可相加的因果贡献百分比。容量足够只代表数学上可能通过，不能证明当前评分器能找到足够低风险的病例。

{md(readable)}

每个队列的双缺失校准人数与零阳性最低人数如下。最低人数指**被阈值接纳的校准患者数**，并非双缺失总人数；即使总人数达到要求，阈值只接纳其中一部分时仍可能不足。按每块delta=0.05/3计算。

{md(cap)}

## 2. 换认证方法有没有帮助
FST：独立拟合OOF排序后首失败停止。LOCKED_FIRST：仅检验独立锁定的第一个阈值。二者是否激活必然一致，因为FST也必须通过第一个检验；此对照说明去掉后续搜索不能修复首检失败。BONF21：21个阈值的Bonferroni校正，有效但可能更保守。

NAIVE_SEARCH_DIAGNOSTIC使用未校正的点态检验后择优；TEST_LABEL_ORACLE_DIAGNOSTIC直接用测试标签选择最大服务阈值。后二者**没有有效风险证书**，只说明放松统计要求或事后看答案时是否出现表面服务，绝不可替代有效实验结果。测试标签诊断也不是总体理论最优上界；无事后服务仅针对当前网格和测试样本，不能证明AI无潜力。

下表非空数指测试组实际有自动病例；与仅阈值激活略有区别。

{md(compare)}

## 3. 20%预算的逐配置服务比较
以下为五个种子合并五折后的描述性均值/范围。重复使用患者，无独立置信区间。各折事后测试标签择优再合并仍是乐观诊断，不会变成合法OOF评估。所有预算完整数表见method_summary.csv。

{md(ss[ss.risk_budget.eq(.2)])}

## 4. 增加校准样本的数学假设分析
固定一个与校准标签独立的阈值，假设接纳患者独立同分布、真实风险为下列预设值，用精确二项检验计算达到90%认证概率的首个样本量。这里没有从真实结果估算“真实风险”，也没有复制患者模拟新增证据。first_n是首次达到，离散检验功效可能局部下降；不是此后所有样本量必然达到90%。NA表示搜索范围内未达到。

{md(size)}

数字是校准集中阈值接纳人数，不是总招募人数，更不是现有773人的临床样本量建议。若评分器无法形成真实风险低于预算的区域，单靠增加校准数据不能把不安全区域变安全。当真实风险恰等预算时，错误拒绝边界零假设的概率受delta限制，不能期待90%认证功效。

## 5. 对论文主线的意义
该诊断支持把“服务代价”拆成可测量的问题：先检查有限样本是否在数学上允许认证，再检查现有分数排序产生的候选是否通过，最后检查有效选择程序是否挡住候选。当前观测不能单独证明模型弱、校准集不足或校正过保守是唯一原因。

下一步应据上述分类决定投入方向：容量障碍突出时，在假设风险与服务率下设计新增校准队列；有容量但候选未通过时，研究低风险区域的评分分辨能力，同时核实临床数据；不能以取消校正或测试标签择优获得的服务作为改进。新模型需要独立选择/校准，不能用本轮测试诊断挑阈值后回头报告保证。

仍属已探索真实队列的描述性实验，无新增外部验证，无临床补查收益结论。11项检查覆盖总体与子群区分、生态推断、手术选择、检查路径、分母基准率、回归均值、零服务保留、全配置比较、多轮探索、非因果、时序未裁定。原始输入及既有结果不变；源文件哈希与基线复现记录见run.json。

协议reports/v18_service_bottleneck_protocol.md；执行src/service_bottleneck_v18.py；报告src/report_bottleneck_v18.py；结果results/service_bottleneck_v18。结果保留本地，代码同步GitHub。
'''
 path=ROOT/'reports/Better1_v18_服务瓶颈拆解.md';path.write_text(report,encoding='utf-8')
 v.dump(OUT/'report_checks.json',{'status':'PASS','diagnostic_cells':len(d),'fold_method_cells':len(r),'invalid_methods_have_no_certificate':True,'baseline_patient_checks':run['baseline_patient_checks'],'metrics_sha256':v.old.digest(OUT/'fold_metrics.csv')})
 print(breakdown.to_string(index=False));print(compare.to_string(index=False));print(cap.to_string(index=False));print(size.to_string(index=False));print(path)

if __name__=='__main__':main()
