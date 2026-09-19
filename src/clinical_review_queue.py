"""Prepare unresolved adjudication items without changing any clinical value."""
from pathlib import Path
import json,sys
sys.path.insert(0,'X:/GJA/WORK/.audit_deps')
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'results/selective_risk_v2';AUD=ROOT/'results/data_audit_20260917';O=ROOT/'results/clinical_review_queue';O.mkdir(exist_ok=True)

def main():
 d=pd.read_csv(D/'derived_patients_LOCAL.csv',dtype={'patient_id':str})
 external=pd.read_excel('X:/GJA/DATA/clinical/lishui/clinical_stats.xls',header=None,keep_default_na=False,usecols=[1,16]).iloc[1:].rename(columns={16:'16'})
 external['id_norm']=external[1].map(lambda value:str(value).strip().removesuffix('.0').lstrip('0'))
 assert not external.id_norm.duplicated().any()
 paired=d[['patient_id','mri_nodes']].merge(external,left_on='patient_id',right_on='id_norm',how='left',validate='one_to_one')
 left=pd.to_numeric(paired.mri_nodes,errors='coerce');right=pd.to_numeric(paired['16'],errors='coerce');different=left.notna()&right.notna()&(left!=right)
 assert different.sum()==75
 items=[]
 def add(category,patient_id,observed,question,scope,source):
  items.append({'category':category,'patient_id':patient_id,'observed_value':observed,'review_question':question,'affected_scope':scope,'source':source,'status':'PENDING','adjudicated_value':'','evidence_location':'','reviewer':'','review_date':''})
 for row in paired.loc[different,['patient_id','mri_nodes','16']].itertuples(index=False,name=None):
  pid,aligned,other=row
  add('MRI_NODE_SOURCE_CONFLICT',pid,f'aligned_C24={aligned}; stats_E16={other}','两字段是否确为相同时间窗/相同定义？请回原始影像报告核实，不能任选一列覆盖。','术前含影像表格分数器；不影响不含影像字段模型','aligned C24 versus clinical_stats E16')
 t=pd.read_csv(AUD/'imaging_surgery_timing_LOCAL.csv',dtype={'patient_id':str})
 for row in d[d.mri_days_before_surgery<=0].itertuples():
  dates=t[t.patient_id==row.patient_id][['study_date','surgery_date','days_before_surgery']].drop_duplicates().to_json(orient='records',force_ascii=False)
  category='MRI_AFTER_SURGERY' if row.mri_days_before_surgery<0 else 'MRI_SAME_DAY_TIME_UNKNOWN'
  add(category,row.patient_id,dates,'请核查目标检查是否在手术前完成，确认正确Study与手术日期；同日需要时分或病历证据。','未来MRI网络及可追溯影像关联；可能影响MRI表格值时间解释','imaging_surgery_timing_LOCAL.csv')
 log=pd.read_csv(D/'transformation_log_LOCAL.csv',dtype={'patient_id':str})
 review=log[log.clinical_review.eq(True)]
 for row in review.itertuples():
  if row.rule=='recompute_same_patient_weight_height':
   raw=d.set_index('patient_id').loc[row.patient_id]
   value=f'height_cm={raw.height}; weight_kg={raw.weight}; derived_bmi={raw.bmi}'
   question='112cm身高是否录入正确？请提供原始测量记录；不能直接按常见身高替换。'
  else:value=str(row.source_value);question='原始字符的临床含义与单位是什么？请核对检验报告后填写可用数值或确认缺失。'
  add('VALUE_REVIEW',row.patient_id,value,question,'对应术前数值输入',f'{row.column}, Excel row {row.excel_row}')
 add('HPV_TCT_CODEBOOK','DATASET','raw/normalized encodings preserved','请提供HPV/TCT编码字典，明确数值、分型、组合标点及未检出/未检查的区分。','未来编码合并与机制解释；本轮仅机械规范标点','hpv_encoding_review.csv; tct_encoding_review.csv')
 add('DICOM_ID_LINKAGE','DATASET','folder IDs and DICOM PatientID differ; prior audit found one-to-one mapping','请提供去标识化或导出映射的来源证据，证明影像与临床患者对应关系。','未来MRI建模、跨中心数据链接','prior DATA audit and image_clinical_matching_LOCAL.csv')
 out=pd.DataFrame(items);out.insert(0,'review_item_id',[f'CR{i+1:04d}' for i in range(len(out))])
 destination=O/'clinical_review_queue_LOCAL.csv'
 if destination.exists():
  previous=pd.read_csv(destination,dtype=str,keep_default_na=False)
  entered=previous[['adjudicated_value','evidence_location','reviewer','review_date']].ne('').any().any()
  if entered or previous.status.ne('PENDING').any():
   raise RuntimeError('Existing clinical adjudications found; refusing to overwrite the review queue')
 out.to_csv(destination,index=False,encoding='utf-8-sig')
 counts=out.category.value_counts().to_dict();assert counts['MRI_AFTER_SURGERY']==3 and counts['MRI_SAME_DAY_TIME_UNKNOWN']==78
 (O/'summary.json').write_text(json.dumps({'status':'PREPARED_NOT_ADJUDICATED','items':len(out),'counts':counts,'clinical_values_changed':False},ensure_ascii=False,indent=2),encoding='utf-8')
 report='# 下一轮真实数据前的人工核实清单\n\n此清单将既有审计中的未决问题转成可填写的表，不代表新发现，也未自动修改临床值。用户已确认标签与术前模块时间窗；以下个体异常和来源差异仍需要原始记录核实。\n\n'
 for key,value in counts.items():report+=f'- {key}：{value} 项。\n'
 report+='\n[打开本地核实表](../results/clinical_review_queue/clinical_review_queue_LOCAL.csv)。每项填写 adjudicated_value、evidence_location、reviewer、review_date，并更新status。只填写实际核对结果，不依据模型预测修改数据。\n\n75处影像字段差异不自动等于75处错误；3例日期晚于手术也需核对Study关联和记录日期。同日记录不自动视为泄漏。DICOM映射需来源证据，不能只因ID不同就断言错配。\n\n该表含本地研究标识，仅供授权的研究团队本地核查。当前V7仅使用合成数据，不受这些未决值影响；涉及相应字段的下一轮真实研究应保留敏感性分析和版本化裁定记录。\n'
 (ROOT/'reports/临床数据待核实清单_20260918.md').write_text(report,encoding='utf-8')
 print(json.dumps({'items':len(out),'counts':counts},ensure_ascii=False))
if __name__=='__main__':main()
