"""Prespecified repeated-split sensitivity and scanner-adjusted E1 extension."""
import json,time
import numpy as np,pandas as pd
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
import v2_pipeline as v
O=v.O;d=pd.read_csv(O/"derived_patients_LOCAL.csv",dtype={"patient_id":str});y=d.true_label.to_numpy();g=d.mask_group.to_numpy()
start=time.time();rows=[];actions=[];roles=[]
for seed in range(5):
 for fold,(train,test) in enumerate(StratifiedKFold(5,shuffle=True,random_state=seed).split(d,y)):
  fit,cal=np.split(np.random.default_rng(73000+seed*10+fold).permutation(train),[int(.75*len(train))])
  assert not set(fit)&set(cal) and not set(train)&set(test)
  for role,ii in [("fit",fit),("calibration",cal),("test",test)]:
   roles.extend(dict(seed=seed,fold=fold,patient_id=d.patient_id.iloc[i],role=role) for i in ii)
  for kind in ["lr","gbdt"]:
   for name,features in {"clinical":v.BASE,"preop":v.BASE+v.MRI+["hpv_normalized","tct_normalized"]}.items():
    m=v.build(features,kind)
    with threadpool_limits(limits=2):
     m.fit(d.iloc[fit],y[fit]);pc=m.predict_proba(d.iloc[cal])[:,1];pt=m.predict_proba(d.iloc[test])[:,1]
    ev=v.old.evidence(pc,y[cal],g[cal])
    for alpha in v.old.ALPHAS:
     rr=v.old.rules_at(ev,alpha)
     for policy in ["G0","G1","G2","G2_fallback"]:
      part=v.old.choose_partition(policy,rr,pc,g[cal]);auto,t,b,scope=v.old.apply(pt,g[test],rr,part)
      meta=dict(seed=seed,fold=fold,model=kind,features=name,policy=policy,partition=part,risk_budget=alpha,corrected=True)
      rows+=v.old.metrics_rows(y[test],g[test],auto,b,meta)
      actions.append(pd.DataFrame({**meta,"patient_id":d.patient_id.iloc[test].to_numpy(),"true_label":y[test],"mask_group":g[test],"score":pt,"action":np.where(auto,"AUTO_NEGATIVE","ABSTAIN"),"threshold":t,"risk_bound":b,"certificate_scope":scope}))
 print("E3 robustness seed",seed,"complete",flush=True)
r=pd.DataFrame(rows);r.to_csv(O/"e3_repeated_fold_metrics.csv",index=False)
rs=r.groupby(["seed","model","features","policy","risk_budget","mask_group"])[["n","positive_n","auto_n","auto_positive_n"]].sum().reset_index()
rs["auto_rate"]=rs.auto_n/rs.n;rs["risk"]=rs.auto_positive_n/rs.auto_n.replace(0,np.nan)
rs.to_csv(O/"e3_repeated_seed_metrics.csv",index=False)
a=pd.concat(actions,ignore_index=True);assert not a.duplicated(["seed","model","features","policy","risk_budget","patient_id"]).any()
a.to_csv(O/"e3_repeated_actions_LOCAL.csv.gz",index=False,compression="gzip")
pd.DataFrame(roles).to_csv(O/"e3_repeated_roles_LOCAL.csv.gz",index=False,compression="gzip")
h=pd.read_csv(v.AUD/"dicom_headers_LOCAL.csv",dtype=str,keep_default_na=False)
h=h[h.center=="original"]
assert h.groupby("folder_patient").Manufacturer.nunique().max()==1
scanner=h.drop_duplicates("folder_patient").set_index("folder_patient")
d["scanner_manufacturer"]=d.patient_id.map(scanner.Manufacturer)
d["scanner_field_strength"]=d.patient_id.map(scanner.MagneticFieldStrength)
v.CAT.extend(["scanner_manufacturer","scanner_field_strength"])
extra=[]
for seed in range(5):
 for config,fields in {"scanner_year":v.BASE+["year","scanner_manufacturer","scanner_field_strength"],"scanner_year_mask":v.BASE+["year","scanner_manufacturer","scanner_field_strength","mask_group"]}.items():
  out=np.full(len(d),np.nan)
  for tr,te in StratifiedKFold(5,shuffle=True,random_state=seed).split(d,y):
   m=v.build(fields,"lr")
   with threadpool_limits(limits=2):
    m.fit(d.iloc[tr],y[tr]);out[te]=m.predict_proba(d.iloc[te])[:,1]
  extra.append(dict(seed=seed,model="lr",configuration=config,**v.performance(y,out)))
pd.DataFrame(extra).to_csv(O/"e1_scanner_adjustment.csv",index=False)
v.dump(O/"robustness_run.json",dict(status="COMPLETED",elapsed_seconds=time.time()-start,repeat_fits=100,scanner_fits=50,source_sha256=v.old.digest(v.SOURCE),code_sha256=v.old.digest(__file__)))
print("ROBUSTNESS complete",round(time.time()-start,1),flush=True)

