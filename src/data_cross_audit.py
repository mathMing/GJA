import sys,json,re,datetime,collections
from pathlib import Path
sys.path.insert(0,"X:/GJA/WORK/.audit_deps")
import pandas as pd,numpy as np,openpyxl
R=Path("X:/GJA/DATA");O=Path("X:/GJA/WORK/results/data_audit_20260917")
def norm(v):return str(v).strip().removesuffix(".0").lstrip("0")
v=pd.read_csv(O/"file_inventory_LOCAL.csv",keep_default_na=False)
imageids={c:sorted(set(p.split("/")[3] for p in v.path if p.startswith("raw/"+c+"/images/"))) for c in ["original","lishui"]}
pathids=sorted(set(p.split("/")[2] for p in v.path if p.startswith("pathology/") and p.endswith(".jpg")))
a=pd.read_excel(R/"clinical/original/cervical_master_aligned.xlsx",header=None,keep_default_na=False).iloc[2:].reset_index(drop=True)
master=pd.read_excel(R/"clinical/original/cervical_master.xlsx",header=None,keep_default_na=False).iloc[2:].reset_index(drop=True)
ext=pd.read_excel(R/"clinical/lishui/clinical_stats.xls",header=None,keep_default_na=False)
head=ext.iloc[0].map(str).tolist();ext=ext.iloc[1:].reset_index(drop=True);ext["id_norm"]=ext[1].map(norm)
ids=pd.read_excel(R/"clinical/lishui/patient_ids.xlsx",header=None,keep_default_na=False)
rows=[]
for c,ii in imageids.items():
 for pid in ii:
  if c=="original":
   m=a[2].map(norm)==norm(pid);n=int(m.sum());labels=a.loc[m,42].tolist();hospital=""
  else:
   m=ext.id_norm==norm(pid);n=int(m.sum());labels=ext.loc[m,15].tolist();hospital="|".join(ext.loc[m,0].astype(str).unique())
  rows.append(dict(center=c,patient_id=pid,clinical_matches=n,labels=str(labels),hospital=hospital))
matches=pd.DataFrame(rows);matches.to_csv(O/"image_clinical_matching_LOCAL.csv",index=False,encoding="utf-8-sig")
external=ext[ext.id_norm.isin({norm(x) for x in imageids["lishui"]})].copy()
external.drop(columns=[2],errors="ignore").to_csv(O/"external_matched_LOCAL.csv",index=False,encoding="utf-8-sig")
matched_ids=set(external.id_norm)
def valcounts(s):return {str(k):int(v) for k,v in s.value_counts(dropna=False).items()}
res={"image_folder_patients":{k:len(x) for k,x in imageids.items()},"pathology_patient_folders":len(pathids),
"aligned_without_image":len(set(a[2].map(norm))-{norm(x) for x in imageids["original"]}),
"image_without_aligned":len({norm(x) for x in imageids["original"]}-set(a[2].map(norm))),
"pathology_in_aligned":len({norm(x) for x in pathids}&set(a[2].map(norm))),
"pathology_outside_aligned":len({norm(x) for x in pathids}-set(a[2].map(norm))),
"patient_id_list_rows":len(ids),"patient_id_list_unique":ids[0].map(norm).nunique(),
"list_without_image":len(set(ids[0].map(norm))-{norm(x) for x in imageids["lishui"]}),
"image_without_list":len({norm(x) for x in imageids["lishui"]}-set(ids[0].map(norm))),
"external_total_rows":len(ext),"external_hospitals":valcounts(ext[0]),"external_matched_rows":len(external),
"external_matched_unique":external.id_norm.nunique(),"external_label_counts":valcounts(external[15]),
"external_hospitals_matched":valcounts(external[0]),"external_id_duplicate_rows_all":int(ext.id_norm.duplicated(False).sum()),
"external_id_duplicate_rows_matched":int(external.id_norm.duplicated(False).sum()),
"external_label_missing_matched":int(external[15].map(lambda v:str(v).strip() in ["","NA"]).sum()),
"crosscenter_numeric_id_overlap":len({norm(x) for x in imageids["original"]}&{norm(x) for x in imageids["lishui"]}),
"external_headers":head}
# Verify metadata remains source-supported; count missing/binary domain for matched set.
res["external_profile"]=[dict(column=f"E{i:02d}",header=head[i],missing=int(external[i].map(lambda v:str(v).strip().lower() in ["","na","n/a","nan","none"]).sum()),unique=external[i].nunique()) for i in range(20)]
# Explicitly parse Excel serial dates, not nanoseconds from epoch.
def date(v):
 if isinstance(v,(int,float)):return pd.Timestamp("1899-12-30")+pd.Timedelta(days=v)
 return pd.to_datetime(v,errors="coerce")
sd=a[27].map(date)
res["surgery_date_corrected"]={"min":str(sd.min()),"max":str(sd.max()),"by_year":valcounts(sd.dt.year),"excel_serial_rows":int(a[27].map(lambda v:isinstance(v,(int,float))).sum())}
pd.DataFrame(dict(patient_id=a[2].map(norm),surgery_date=sd)).to_csv(O/"surgery_dates_LOCAL.csv",index=False)
# Correctly compare only unique keys; preserve duplicate ambiguity.
dupids=set(master.loc[master[2].map(norm).duplicated(False),2].map(norm))
mm=master.assign(id=master[2].map(norm));aa=a.assign(id=a[2].map(norm))
dm=mm[~mm.id.isin(dupids)].set_index("id");da=aa.set_index("id")
changes=[]
def c(v):return str(v).strip().removesuffix(".0")
for pid in sorted(set(dm.index)&set(da.index)):
 for j in range(62):
  if c(dm.loc[pid,j])!=c(da.loc[pid,j]):changes.append(dict(patient_id=pid,column=f"C{j:02d}",source=str(dm.loc[pid,j]),aligned=str(da.loc[pid,j])))
pd.DataFrame(changes).to_csv(O/"unique_key_version_differences_LOCAL.csv",index=False,encoding="utf-8-sig")
res["unique_key_version_changes"]=dict(collections.Counter(r["column"] for r in changes))
res["master_duplicate_ids"]=sorted(dupids)
# Formula-reference audit, no recalculation or workbook modification.
wb=openpyxl.load_workbook(R/"clinical/original/cervical_master_aligned.xlsx",data_only=False)
ws=wb["Sheet1"];formulas=[]
for row in ws:
 for cell in row:
  if cell.data_type=="f":
   refs=re.findall(r"\$?[A-Z]{1,3}\$?([0-9]+)",str(cell.value))
   formulas.append(dict(cell=cell.coordinate,column=cell.column,row=cell.row,formula=cell.value,row_mismatch=bool(any(int(r)!=cell.row for r in refs)),outside_data=bool(any(int(r)>775 for r in refs))))
pd.DataFrame(formulas).to_csv(O/"formula_references_LOCAL.csv",index=False)
res["BMI_formula_n"]=sum(x["column"]==8 for x in formulas)
res["BMI_formula_wrong_row_n"]=sum(x["column"]==8 and x["row_mismatch"] for x in formulas)
res["BMI_formula_outside_rows_n"]=sum(x["column"]==8 and x["outside_data"] for x in formulas)
res["nonBMI_formulas"]=[x for x in formulas if x["column"]!=8]
res["height_weight_joint_available"]=int((pd.to_numeric(a[5],errors="coerce").notna()&pd.to_numeric(a[6],errors="coerce").notna()).sum())
# Counts of note-only and empty MRI sequences.
series=v[v.path.str.startswith("raw/")].copy();parts=series.path.str.split("/",expand=True)
series["center"]=parts[1];series["pid"]=parts[3];series["sequence"]=parts[4]
res["sequence_file_counts"]=series.groupby(["center","sequence"]).size().to_dict().__str__()
seq=[]
for (center,pid,se),t in series.groupby(["center","pid","sequence"]):
 n=int((~t.path.str.endswith(".txt")).sum())
 if n==0:seq.append(dict(center=center,patient_id=pid,sequence=se,files=t.path.tolist()))
res["note_only_sequences"]=seq
(O/"cross_audit.json").write_text(json.dumps(res,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
print(json.dumps(res,ensure_ascii=True,default=str))

