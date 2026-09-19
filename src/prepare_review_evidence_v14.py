"""Read-only clinical evidence routing; no adjudication and no model fitting."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/review_evidence_v14'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def md(d):
 return '| '+' | '.join(d.columns)+' |\n|'+'|'.join(['---']*len(d.columns))+'|\n'+'\n'.join('| '+' | '.join(str(x) for x in row)+' |' for row in d.itertuples(index=False,name=None))
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 queue=ROOT/'results/clinical_review_queue/clinical_review_queue_LOCAL.csv';derived=ROOT/'results/selective_risk_v2/derived_patients_LOCAL.csv'
 before={str(p):digest(p) for p in [queue,derived]}
 q=pd.read_csv(queue,dtype=str,keep_default_na=False);d=pd.read_csv(derived,dtype={'patient_id':str});assert len(q)==160 and len(d)==773 and d.true_label.sum()==149
 assert q.review_item_id.nunique()==len(q)
 patient=q[q.patient_id.ne('DATASET')];assert set(patient.patient_id)<=set(d.patient_id)
 routes={
 'MRI_NODE_SOURCE_CONFLICT':('原始影像报告与两份表的字段定义','已定位两来源冲突，不能自动判谁正确','术前模型直接使用mri_nodes；临床BASE不使用','移除mri_nodes；另列移除全部影像表格字段的敏感性方案'),
 'MRI_SAME_DAY_TIME_UNKNOWN':('检查完成时分、手术开始时分及Study关联','缓存仅有日期，不能判定先后','未来MRI网络；影像表格字段是否同次检查需核实','同日病例排除只能作敏感性分析，不能认定泄漏'),
 'MRI_AFTER_SURGERY':('正确目标Study及手术记录','发现日期先后矛盾，不能确认是错配还是非目标检查','未来MRI网络；影像表格来源关联','核对后选择正确检查；裁定前可预设排除该3例的敏感性分析'),
 'VALUE_REVIEW':('身高/体重原始测量、CA125原始检验单与单位','无法从常见数值推断应改成什么','临床BASE与术前模型均可能受BMI或CA125影响','预设对应值视为缺失的敏感性分析，不把猜测当真值'),
 'HPV_TCT_CODEBOOK':('检验编码字典，未检查/未检出区别','只有机械规范记录，不能确认语义合并','所有按缺失键分区的策略；术前模型另直接使用结果编码','先明确原始缺失定义，不按模型表现重分组'),
 'DICOM_ID_LINKAGE':('去标识化映射或导出记录','一一映射不是身份一致性的来源证明','未来MRI网络及影像与临床链接','不影响仅读取临床表的计算，但影像来源关联仍待核实')}
 routed=q.copy()
 for i,col in enumerate(['required_evidence','current_evidence_limit','affected_models','candidate_sensitivity_NOT_EXECUTED']):routed[col]=routed.category.map(lambda x:routes[x][i])
 routed.to_csv(OUT/'review_evidence_routes_LOCAL.csv',index=False,encoding='utf-8-sig')
 # Patient flags are for review/sensitivity cohort bookkeeping only, never predictors.
 flags=d[['patient_id','mask_group','true_label']].copy()
 for category in routes:
  flags[category]=flags.patient_id.isin(q.loc[q.category.eq(category)&q.patient_id.ne('DATASET'),'patient_id'])
 categories=list(routes)[:4];flags['any_patient_issue']=flags[categories].any(axis=1)
 flags['time_issue']=flags[['MRI_SAME_DAY_TIME_UNKNOWN','MRI_AFTER_SURGERY']].any(axis=1)
 flags.to_csv(OUT/'patient_review_flags_LOCAL.csv',index=False,encoding='utf-8-sig')
 overlaps=[]
 for i,a in enumerate(categories):
  for b in categories[i+1:]:overlaps.append(dict(category_a=a,category_b=b,shared_patients=int((flags[a]&flags[b]).sum())))
 pd.DataFrame(overlaps).to_csv(OUT/'issue_overlap.csv',index=False)
 impact=[]
 for category in categories+['time_issue','any_patient_issue']:
  z=flags[flags[category]]
  for group in ['all']+sorted(d.mask_group.unique().tolist()):
   sub=z if group=='all' else z[z.mask_group.eq(group)]
   impact.append(dict(issue=category,mask_group=group,patients=len(sub),positive_n=int(sub.true_label.sum())))
 pd.DataFrame(impact).to_csv(OUT/'issue_cohort_counts.csv',index=False)
 scenarios=[]
 for name,excluded in [('retain_all_for_review',np.zeros(len(d),bool)),('hypothetical_exclude_after_surgery',flags.MRI_AFTER_SURGERY),('hypothetical_exclude_same_or_after',flags.time_issue),('hypothetical_exclude_node_conflicts',flags.MRI_NODE_SOURCE_CONFLICT),('hypothetical_exclude_any_patient_issue',flags.any_patient_issue)]:
  for group in ['all']+sorted(d.mask_group.unique().tolist()):
   m=np.ones(len(d),bool) if group=='all' else d.mask_group.eq(group).to_numpy();remain=~np.asarray(excluded)&m
   scenarios.append(dict(scenario=name,mask_group=group,remaining_n=int(remain.sum()),remaining_positive=int(d.true_label[remain].sum()),excluded_n=int((np.asarray(excluded)&m).sum())))
 sc=pd.DataFrame(scenarios);sc.to_csv(OUT/'hypothetical_cohort_footprints.csv',index=False)
 raw='X:/GJA/DATA/clinical/original/cervical_master_aligned.xlsx'
 values=pd.read_excel(raw,header=None,usecols=[27]).iloc[2:,0]
 def parsed(x):
  if isinstance(x,(int,float)) and pd.notna(x):return pd.Timestamp('1899-12-30')+pd.Timedelta(days=x)
  return pd.to_datetime(x,errors='coerce')
 dates=values.map(parsed);valid=dates.notna();nonmid=int(sum(t!=t.normalize() for t in dates[valid]))
 sourcecheck={'surgery_dates_parsed':int(valid.sum()),'surgery_values_with_non_midnight_time':nonmid,'cached_dicom_has_StudyTime': 'StudyTime' in pd.read_csv(ROOT/'results/data_audit_20260917/dicom_headers_LOCAL.csv',nrows=0).columns}
 category_table=pd.DataFrame([dict(category=cat,items=int(q.category.eq(cat).sum()),required_evidence=r[0],affected_models=r[2]) for cat,r in routes.items()])
 summary={'status':'PREPARED_NOT_ADJUDICATED','items':len(q),'patient_specific_items':len(patient),'unique_affected_patients':patient.patient_id.nunique(),'dataset_level_items':int(q.patient_id.eq('DATASET').sum()),'queue_status_counts':q.status.value_counts().to_dict(),'source_time_checks':sourcecheck,'clinical_values_changed':False,'model_fits':0,'source_sha256':before}
 assert all(digest(Path(p))==h for p,h in before.items())
 (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
 report=f'''# V14：将待核实事项变成可执行的数据核实包

日期：2026-09-19。Origin Skill=experiment-agent；Mode=validate；Verification Status=ANALYZED。仅整理证据与影响范围，无人工裁定、无新增训练。

## 1. 实际工作量
160项中，{len(patient)}项是患者级事项，涉及{patient.patient_id.nunique()}名不同患者；另有2项数据集级编码/身份映射说明。因此160不是独立患者数，也不是错误数。不同类别有重叠，不能直接相加后计算排除人数。

## 2. 已能查清和仍无法查清的边界
现有日期审计缓存未保存StudyTime。本轮读取原临床表手术日期列，773例中可解析{int(valid.sum())}例，带非零时分秒{nonmid}例。午夜值不能证明实际手术发生在午夜，应视为时间精度未知。只取得MRI时分而无手术时分也不能判定同日先后；本轮不自动解除78项同日核实。
75处两来源影像淋巴结字段差异已经定位，但没有原始报告证据来决定哪份正确；不得任选一列覆盖。编码字典与影像身份映射属于数据集级来源证据，需要负责数据导出的人员或临床协作者提供。

{md(category_table)}

## 3. 假设排除的样本量影响（未执行排除或训练）
下表只算账，不代表这些病例应该删除。排除会改变人群和病例构成，不能把其后的表现差异当成纯算法提升。移除mri_nodes字段则保留全部773人；字段删除和病例排除应分开设计。

{md(sc[sc.mask_group.isin(['all','missing_both'])])}

## 4. 可直接分发给本地协作者的材料
- [逐项核实路径及问题](../results/review_evidence_v14/review_evidence_routes_LOCAL.csv)：保留已有状态及回答栏，增加需要什么原始证据、影响哪些模型、可预设什么敏感性分析。
- [患者级事项标记](../results/review_evidence_v14/patient_review_flags_LOCAL.csv)：仅用于核实和敏感性人群记录，禁止作为预测特征。
- [原始裁定主表](../results/clinical_review_queue/clinical_review_queue_LOCAL.csv)：唯一裁定记录入口；在副本中批注后应按review_item_id人工合并到主表，避免两份状态冲突。

没有向任何人发送上述文件。LOCAL文件含研究标识，仅供授权研究团队本地使用。核实应填写实际证据路径、裁定值、核实人和日期，不得按模型预测改标签或输入。未核实项保持PENDING。

## 5. 接下来可以独立推进什么
不依赖人工裁定的工作是预先冻结敏感性方案：移除争议mri_nodes字段、移除全部影像表格字段、将异常数值视为缺失，以及按预设时序口径排除的单独分析。它们能衡量结论依赖性，不能验证哪个原值正确。不同人群或不同特征的效应不能混成一个“最佳模型”。
当前不直接重跑：先确定各对照保持哪些患者划分与训练角色，以及同时比较的范围；不能边看结果边改变排除口径。来源核实与敏感性分析可以并行，不能用后者替代前者。

## 6. 核验与偏差说明
原始裁定队列及派生患者表哈希未变；没有修改临床原表；模型拟合0次。全部类别与160项核对一致，所有患者事项均映射到773例内部患者。影像表格模型与未来MRI网络的影响范围分别说明。
11项核查：总体/子群同时计数、无个体效果推断、手术入组限制、检查路径非因果、保留阳性分母、未按极端模型结果挑选病例、未删除未解决事项、未搜索最佳排除组合、明确探索流程、未推断临床收益、同日与术后时间未擅自裁定。
'''
 (ROOT/'reports/Better1_v14_临床核实分级与敏感性准备.md').write_text(report,encoding='utf-8')
 print(json.dumps(summary,ensure_ascii=False));print(sc[sc.mask_group.isin(['all','missing_both'])].to_string(index=False))
if __name__=='__main__':main()
