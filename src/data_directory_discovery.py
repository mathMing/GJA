import os,json,collections,hashlib
from pathlib import Path
import pandas as pd
R=Path("X:/GJA/DATA");O=Path("X:/GJA/WORK/results/data_audit_20260917");O.mkdir(parents=True,exist_ok=True)
files=[];dirs=[]
for base,ds,fs in os.walk(R):
 dirs.append(str(Path(base).relative_to(R)))
 for fn in fs:
  p=Path(base)/fn
  try: st=p.stat();files.append(dict(path=str(p.relative_to(R)).replace("\\","/"),bytes=st.st_size,ext="".join(p.suffixes).lower(),mtime=st.st_mtime_ns))
  except Exception as ex: files.append(dict(path=str(p),error=str(ex)))
pd.DataFrame(files).to_csv(O/"file_inventory_LOCAL.csv",index=False)
summary={"files":len(files),"dirs":len(dirs),"bytes":sum(f.get("bytes",0) for f in files),"extensions":dict(collections.Counter(f.get("ext") for f in files)),"depth2_dirs":[d for d in dirs if len(Path(d).parts)<=3],"examples":{}}
for top in ["raw/original","raw/lishui","pathology","clinical"]:
 items=[f for f in files if f["path"].startswith(top)]
 summary["examples"][top]=items[:4]
books={}
for f in files:
 if f.get("ext")==".xlsx":
  p=R/f["path"];book=pd.ExcelFile(p)
  b={"sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"sheets":{}}
  for sheet in book.sheet_names:
   d=pd.read_excel(p,sheet_name=sheet,header=None,keep_default_na=False)
   b["sheets"][sheet]={"shape":list(d.shape),"first_rows":[[str(v) for v in row] for row in d.iloc[:2].values.tolist()]}
  books[f["path"]]=b
summary["books"]=books
(O/"discovery_LOCAL.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
# Only schema, aggregate counts and file examples (no patient cell values) to console
print(json.dumps({k:v for k,v in summary.items() if k not in ["books","depth2_dirs","examples"]}))
for k,b in books.items():
 print(k)
 for sn,v in b["sheets"].items():
  print(sn,v["shape"]);print(json.dumps(v["first_rows"],ensure_ascii=True) if "original" in k else "External schema stored locally")
print(json.dumps(summary["examples"],ensure_ascii=True))

