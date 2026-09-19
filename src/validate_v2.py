"""Meaningful checks and supplementary uncertainty; no result-directed retuning."""
import numpy as np,pandas as pd,json,time
from pathlib import Path
from threadpoolctl import threadpool_limits
import v2_pipeline as v
import synthetic_boundary_v2 as s
from scipy.stats import binom
O=v.O;d=pd.read_csv(O/"derived_patients_LOCAL.csv",dtype={"patient_id":str})
start=time.time();out=[]
# Bootstrap actual patient rows with preprocessing refit, penalized coefficients.
for name in ["base_mask","base_year_mask","imaging_year_mask"]:
 vals={g:[] for g in v.old.GROUPS[1:]}
 for b in range(300):
  ix=np.random.default_rng(51100+b).integers(0,len(d),len(d))
  m=v.build(v.CONFIG[name],"lr")
  with threadpool_limits(limits=2):m.fit(d.iloc[ix],d.true_label.iloc[ix])
  names=m.named_steps["features"].get_feature_names_out();co=m.named_steps["model"].coef_[0]
  ref=np.flatnonzero(names=="cat__mask_group_complete")
  for group in vals:
   gi=np.flatnonzero(names=="cat__mask_group_"+group)
   vals[group].append(float(np.exp(co[gi[0]]-co[ref[0]])) if len(gi) and len(ref) else np.nan)
 for g,rr in vals.items():
  q=np.nanquantile(rr,[.025,.5,.975])
  out.append(dict(configuration=name,group=g,bootstrap_repetitions=300,OR_lower=q[0],OR_median=q[1],OR_upper=q[2],interpretation="penalized bootstrap exploratory; rare 19/0 group has separation; no causal inference"))
 print("Bootstrap",name,"complete",flush=True)
pd.DataFrame(out).to_csv(O/"e1_association_bootstrap.csv",index=False)
# Independent population summation check for all routing partitions.
pop=s.distribution("INFORMATIVE",.15,.8,np.log(1.75))
rng=np.random.default_rng(8);draw=rng.multinomial(1000,pop.ravel(),size=5).reshape(5,4,2,s.B)
idx=rng.integers(-1,s.B,(5,6));bounds=np.full((5,6),.2)
for part in [0,1,2]:
 result=s.summarize(idx,bounds,np.full(5,part),pop,draw,.2)
 for i in range(5):
  n=k=dn=dk=0.
  for g,scope in enumerate(s.MAPS[part]):
   cut=idx[i,scope] if scope>=0 else -1
   if cut>=0:
    nn=pop[g,:,:cut+1].sum();kk=pop[g,1,:cut+1].sum();n+=nn;k+=kk
    if g==2:dn=nn;dk=kk
  assert np.isclose(result.true_auto_rate.iloc[i],n)
  if n>0:assert np.isclose(result.true_auto_risk.iloc[i],k/n)
  if dn>0:assert np.isclose(result.double_true_risk.iloc[i],dk/dn)
# Zero-threshold no service, full-population mean, CP-binomial duality.
z=s.summarize(np.full((5,6),-1),bounds,np.full(5,2),pop,draw,.2)
assert z.true_auto_risk.isna().all() and z.true_auto_rate.eq(0).all()
for n in [10,50,200]:
 for k in [0,1,n//2,n]:
  ub=float(v.old.cp(k,n,.05/126))
  for alpha in [.05,.1,.2]:
   assert (ub<=alpha)==(binom.cdf(k,n,alpha)<=.05/126)
a=pd.read_csv(O/"e3_actions_LOCAL.csv",dtype={"patient_id":str})
assert not a.duplicated(["model","features","risk_budget","policy","patient_id"]).any()
roles=pd.read_csv(O/"e3_split_LOCAL.csv",dtype={"patient_id":str})
assert roles.patient_id.nunique()==773 and not roles.patient_id.duplicated().any()
assert set(a.patient_id)==set(roles.loc[roles.role=="test","patient_id"])
assert ((a.action=="AUTO_NEGATIVE")==(a.score<=a.threshold)).all()
mf=json.loads((O/"data_freeze_manifest.json").read_text(encoding="utf-8"))
assert v.old.digest(v.SOURCE)==mf["source_sha256"]
result={"status":"PASS","checks":["E2 direct exact population risk vs vectorization","E2 empty-service undefined risk","binomial test vs CP duality","patient role exclusivity","one action per patient/model/policy/budget","action threshold consistency","source SHA unchanged"],"bootstrap_runs":900,"elapsed_seconds":time.time()-start}
(O/"validation_v2.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
print(result)

