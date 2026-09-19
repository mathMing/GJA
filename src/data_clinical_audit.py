import sys,os,json,hashlib,collections,datetime,re
from pathlib import Path
sys.path.insert(0,"X:/GJA/WORK/.audit_deps")
import pandas as pd,numpy as np,openpyxl
R=Path("X:/GJA/DATA");O=Path("X:/GJA/WORK/results/data_audit_20260917");O.mkdir(parents=True,exist_ok=True)
def clean(v):
 if pd.isna(v): return ""
 if isinstance(v,(datetime.datetime,datetime.date,pd.Timestamp)):return v.isoformat()
 if isinstance(v,(int,float,np.integer,np.floating)):return format(v,".15g")
 return str(v).strip()
def missing(v): return clean(v).lower() in ("","na","n/a","nan","none","null")
def ident(v):return clean(v).removesuffix(".0")
books={};schema=[];frames={};hashes={}
for p in sorted((R/"clinical").rglob("*")):
 if p.suffix not in [".xls",".xlsx"]:continue
 key=p.name;hashes[key]=hashlib.sha256(p.read_bytes()).hexdigest()
 sheets=pd.read_excel(p,sheet_name=None,header=None,keep_default_na=False)
 books[key]={}
 for sn,d in sheets.items():
  frames[(key,sn)]=d
  firstnonempty=[i for i in range(len(d)) if any(not missing(v) for v in d.iloc[i])][:3]
  books[key][sn]={"rows":len(d),"cols":d.shape[1],"nonempty_rows":int(d.apply(lambda row:any(not missing(v) for v in row),axis=1).sum()),"first_nonempty_rows":[i+1 for i in firstnonempty]}
  schema.append({"book":key,"sheet":sn,**books[key][sn]})
  (O/(key+"_"+sn+"_schema_LOCAL.json")).write_text(json.dumps({"first_rows":[[clean(v) for v in d.iloc[i]] for i in firstnonempty]},ensure_ascii=False,indent=2),encoding="utf-8")
pd.DataFrame(schema).to_csv(O/"workbook_inventory.csv",index=False)
intframes={};summary={};profiles=[];issues=[];diffs=[];formula=[]
for key in [k for k in books if "cervical_master" in k]:
 raw=frames[(key,"Sheet1")];d=raw.iloc[2:].reset_index(drop=True).copy();d.columns=[f"C{i:02d}" for i in range(d.shape[1])]
 d["id"]=d.C02.map(ident);intframes[key]=d;y=pd.to_numeric(d.C42,errors="coerce")
 summary[key]={"n":len(d),"id_unique":d.id.nunique(),"id_missing":int(d.C02.map(missing).sum()),"duplicate_id_rows":int(d.id.duplicated(False).sum()),"label_counts":{str(k):int(v) for k,v in y.value_counts(dropna=False).items()},"duplicate_rows":int(d.drop(columns="id").duplicated().sum())}
 wb=openpyxl.load_workbook(R/"clinical"/"original"/key,data_only=False)
 wc=openpyxl.load_workbook(R/"clinical"/"original"/key,data_only=True)
 for ws in wb:
  fc=ec=emptycache=0
  for row in ws:
   for cell in row:
    if cell.data_type=="f":
     fc+=1;emptycache+=wc[ws.title][cell.coordinate].value is None
    if cell.data_type=="e":ec+=1
  formula.append(dict(book=key,sheet=ws.title,formulas=fc,formula_cache_missing=emptycache,errors=ec,hidden_rows=sum(bool(x.hidden) for x in ws.row_dimensions.values()),hidden_cols=sum(bool(x.hidden) for x in ws.column_dimensions.values()),merged_cells=len(ws.merged_cells.ranges)))
main=intframes["cervical_master_aligned.xlsx"];raw=frames[("cervical_master_aligned.xlsx","Sheet1")];y=pd.to_numeric(main.C42,errors="coerce")
for j in range(62):
 c=f"C{j:02d}";v=main[c];mi=v.map(missing);num=pd.to_numeric(v.where(~mi),errors="coerce");bad=(~mi)&num.isna()
 profiles.append(dict(column=c,excel_column=openpyxl.utils.get_column_letter(j+1),header=clean(raw.iloc[1,j]),nonmissing=int((~mi).sum()),missing=int(mi.sum()),missing_fraction=float(mi.mean()),unique_nonmissing=v[~mi].map(clean).nunique(),numeric_n=int(num.notna().sum()),nonnumeric_n=int(bad.sum()),numeric_min=float(num.min()) if num.notna().any() else None,numeric_max=float(num.max()) if num.notna().any() else None,nonmissing_positive_n=int(y[~mi].sum()),missing_positive_n=int(y[mi].sum())))
 if j in [4,5,6,7,10,11,15,16,17,19,21,43,44]:
  for i in np.flatnonzero(bad):issues.append(dict(kind="numeric_text",patient_id=main.id.iloc[i],excel_row=int(i+3),column=c,value=clean(v.iloc[i])))
 if j in [9,22,23,24,39,40,41,42]:
  invalid=(~mi)&(~num.isin([0,1]))
  for i in np.flatnonzero(invalid):issues.append(dict(kind="binary_domain",patient_id=main.id.iloc[i],excel_row=int(i+3),column=c,value=clean(v.iloc[i])))
p=pd.DataFrame(profiles);p.to_csv(O/"field_profile.csv",index=False,encoding="utf-8-sig")
versions=[]
for key,d in intframes.items():
 idx=d.set_index("id");ref=main.set_index("id");common=sorted(set(idx.index[~idx.index.duplicated(False)])&set(ref.index))
 changes=collections.Counter()
 for pid in common:
  for j in range(62):
   c=f"C{j:02d}"
   if clean(idx.loc[pid,c])!=clean(ref.loc[pid,c]):
    changes[c]+=1;diffs.append(dict(book=key,patient_id=pid,column=c,source_value=clean(idx.loc[pid,c]),aligned_value=clean(ref.loc[pid,c])))
 extra=set(idx.index)-set(ref.index);absent=set(ref.index)-set(idx.index)
 versions.append(dict(book=key,**summary[key],only_in_version=len(extra),only_in_aligned=len(absent),common_changed_cells=sum(changes.values()),changed_columns=dict(changes),only_version_positive=int(pd.to_numeric(idx.loc[list(extra),"C42"]).sum()) if extra else 0))
 for pid in extra:issues.append(dict(kind="version_only",book=key,patient_id=pid))
 for pid in absent:issues.append(dict(kind="aligned_only",book=key,patient_id=pid))
h=main.C13.map(missing);t=main.C14.map(missing)
groups=np.select([h&t,h&~t,~h&t],["missing_both","missing_hpv","missing_tct"],default="complete")
group_summary=[]
for group in ["complete","missing_tct","missing_both","missing_hpv"]:
 m=groups==group;group_summary.append(dict(group=group,n=int(m.sum()),positive=int(y[m].sum()),prevalence=float(y[m].mean())))
tokens={}
for j in [7,8,9,13,14,15,16,17,22,23,24,35,42]:
 tokens[f"C{j:02d}"]={str(k):int(v) for k,v in main[f"C{j:02d}"].map(clean).value_counts(dropna=False).items()}
# Structural cross-field consistency, not a clinical truth adjudication.
n_total=pd.to_numeric(main.C43,errors="coerce");n_pos=pd.to_numeric(main.C44,errors="coerce")
checks={"positive_label_with_zero_positive_nodes":(y==1)&(n_pos==0),"negative_label_with_positive_nodes":(y==0)&(n_pos>0),"positive_nodes_exceed_removed_nodes":n_pos>n_total,
"parity_exceeds_gravidity":pd.to_numeric(main.C11,errors="coerce")>pd.to_numeric(main.C10,errors="coerce"),
"nonpositive_height":pd.to_numeric(main.C06,errors="coerce")<=0,"nonpositive_weight":pd.to_numeric(main.C05,errors="coerce")<=0}
for kind,m in checks.items():
 for i in np.flatnonzero(m):issues.append(dict(kind=kind,patient_id=main.id.iloc[i],excel_row=int(i+3)))
def parse_excel_date(v):
 if isinstance(v,(int,float)): return pd.Timestamp("1899-12-30")+pd.Timedelta(days=v)
 return pd.to_datetime(v,errors="coerce")
surgery=main.C27.map(parse_excel_date)
dates={"parseable":int(surgery.notna().sum()),"min":str(surgery.min()),"max":str(surgery.max()),"by_year":{str(k):int(v) for k,v in surgery.dt.year.value_counts().sort_index().items()}}
# Record patient mapping locally; no names or phone numbers exported.
pd.DataFrame(dict(patient_id=main.id,label=y,mask_group=groups,surgery_date=surgery)).to_csv(O/"patient_index_LOCAL.csv",index=False)
pd.DataFrame(diffs).to_csv(O/"version_cell_differences_LOCAL.csv",index=False,encoding="utf-8-sig")
pd.DataFrame(issues).to_csv(O/"clinical_issues_LOCAL.csv",index=False,encoding="utf-8-sig")
pd.DataFrame(formula).to_csv(O/"formula_audit.csv",index=False)
result={"versions":versions,"groups":group_summary,"cross_field_checks":{k:int(v.sum()) for k,v in checks.items()},"surgery_dates":dates,"workbooks":books,"source_sha256":hashes,"tokens":tokens}
(O/"clinical_audit.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
print(json.dumps({k:v for k,v in result.items() if k not in ["tokens","source_sha256","workbooks"]},ensure_ascii=True))
print("SCHEMAS",json.dumps(schema,ensure_ascii=True))
print("NUMERIC_TEXT",json.dumps([v for v in issues if v["kind"]=="numeric_text"],ensure_ascii=True))
print("TOKENS",json.dumps({k:v for k,v in tokens.items() if k in ["C07","C08","C13","C14","C22","C23","C24"]},ensure_ascii=True))
