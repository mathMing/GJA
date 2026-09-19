import sys,json,collections
from pathlib import Path
sys.path.insert(0,"X:/GJA/WORK/.audit_deps")
import pandas as pd,numpy as np
R=Path("X:/GJA/DATA");O=Path("X:/GJA/WORK/results/data_audit_20260917")
a=pd.read_excel(R/"clinical/original/cervical_master_aligned.xlsx",header=None,keep_default_na=False).iloc[2:].reset_index(drop=True)
e=pd.read_excel(R/"clinical/lishui/clinical_stats.xls",header=None,keep_default_na=False).iloc[1:].reset_index(drop=True)
def norm(v):return str(v).strip().removesuffix(".0").lstrip("0")
a["id"]=a[2].map(norm);e["id"]=e[1].map(norm)
shared=a.rename(columns={i:f"{i}_a" for i in range(62)}).merge(e.rename(columns={i:f"{i}_e" for i in range(20)}),on="id")
labeldiff=shared[pd.to_numeric(shared["42_a"],errors="coerce")!=pd.to_numeric(shared["15_e"],errors="coerce")]
conf=[]
for ai,ei in [(4,3),(9,12),(10,6),(11,7),(15,8),(16,9),(17,10),(24,16),(42,15)]:
 x=pd.to_numeric(shared[f"{ai}_a"],errors="coerce");z=pd.to_numeric(shared[f"{ei}_e"],errors="coerce")
 ok=x.notna()&z.notna();bad=ok&~np.isclose(x,z)
 conf.append(dict(internal=f"C{ai:02d}",external=f"E{ei:02d}",matched=len(shared),both_numeric=int(ok.sum()),different=int(bad.sum())))
pd.DataFrame(conf).to_csv(O/"internal_external_table_comparison.csv",index=False)
# Missing markers are not standardized clinical semantics.
h=a[13].astype(str).str.strip().isin(["NA",""]);t=a[14].astype(str).str.strip().isin(["NA",""])
group=np.select([h&t,h&~t,~h&t],["missing_both","missing_hpv","missing_tct"],default="complete")
def date(v):
 if isinstance(v,(int,float)):return pd.Timestamp("1899-12-30")+pd.Timedelta(days=v)
 return pd.to_datetime(v,errors="coerce")
dt=a[27].map(date)
tab=pd.DataFrame(dict(year=dt.dt.year,group=group,y=pd.to_numeric(a[42]))).groupby(["year","group"]).agg(n=("y","size"),positive=("y","sum")).reset_index()
tab.to_csv(O/"missingness_by_surgery_year.csv",index=False)
numeric={}
for j in [4,5,6,15,16,17,19,21,43,44]:
 s=pd.to_numeric(a[j],errors="coerce")
 numeric[f"C{j:02d}"]={"zero_n":int(s.eq(0).sum()),"min":float(s.min()),"median":float(s.median()),"max":float(s.max())}
y=pd.to_numeric(a[42]);leak={}
for j in [34,45]:
 obs=~a[j].astype(str).str.strip().str.lower().isin(["","na","nan","none"])
 leak[f"C{j:02d}"]={"observed_n":int(obs.sum()),"observed_positive":int(y[obs].sum()),"observed_negative":int((y[obs]==0).sum())}
# Anonymous high-level diagnostic of duplicated source ID row; store source coordinates locally.
master=pd.read_excel(R/"clinical/original/cervical_master.xlsx",header=None,keep_default_na=False).iloc[2:].reset_index(drop=True)
dups=master[master[2].map(norm).duplicated(False)]
sameid_details=[]
for pid,q in dups.groupby(2):
 varying=[f"C{i:02d}" for i in range(62) if q[i].astype(str).nunique()>1]
 sameid_details.append(dict(patient_id=norm(pid),excel_rows=(q.index+3).tolist(),varying_columns=varying,in_aligned=norm(pid) in set(a.id)))
# Formulas in original source also malformed; save cross-source anomalies.
import openpyxl
wb=openpyxl.load_workbook(R/"clinical/original/cervical_master.xlsx",data_only=False)
source_formula_examples=[]
for row in wb["Sheet1"]:
 for c in row:
  if c.data_type=="f" and c.column!=8:source_formula_examples.append(dict(cell=c.coordinate,formula=c.value))
res={"shared_internal_external_stats_rows":len(shared),"shared_label_conflicts":len(labeldiff),"numeric_profiles":numeric,"label_proxy_missingness":leak,"master_duplicate":sameid_details,"master_nonBMI_formulas":source_formula_examples,"year_missingness":tab.to_dict("records")}
(O/"supplement.json").write_text(json.dumps(res,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
print(json.dumps({k:v for k,v in res.items() if k not in ["year_missingness"]},ensure_ascii=True,default=str))
print(pd.DataFrame(conf).to_string(index=False))
