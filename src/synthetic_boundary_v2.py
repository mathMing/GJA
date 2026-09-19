"""Exact finite-score population benchmark; all calibration/test draws are independent.
No model is trained to a claimed AUROC: class-conditional score distributions are controlled.
"""
import os
for k in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"]:os.environ.setdefault(k,"1")
import time,json,itertools,gzip,hashlib,datetime
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import norm,binom,beta
from scipy.special import expit,logit
from scipy.optimize import brentq
ROOT=Path(__file__).resolve().parents[1];O=ROOT/"results"/"selective_risk_v2";O.mkdir(exist_ok=True,parents=True)
B=21;EDGES=np.r_[-np.inf,np.linspace(-3.5,3.5,B-1),np.inf];CENTERS=np.r_[-4,(EDGES[1:-2]+EDGES[2:-1])/2,4]
GROUPS=["complete","missing_tct","missing_both","missing_hpv"];SCOPES=["all","complete","any_missing","missing_tct","missing_both","missing_hpv"]
SCOPE_WEIGHTS=np.array([[1,1,1,1],[1,0,0,0],[0,1,1,1],[0,1,0,0],[0,0,1,0],[0,0,0,1]],int)
MAPS=np.array([[0,0,0,0],[1,2,2,2],[1,3,4,-1]])
ALPHAS=[.05,.1,.2];REPS=200
def distribution(mode,q,auc,strength):
 sep=np.sqrt(2)*norm.ppf(auc)
 pg=np.array([1-q-.3*(1-q)-.025,.3*(1-q),q,.025])
 rest=pg.copy();rest[2]=0;rest/=rest.sum()
 prevalence=.2
 dist=np.zeros((4,2,B))
 scorepdf=[]
 for y in [0,1]:
  mu=(2*y-1)*sep/2
  scorepdf.append(np.diff(norm.cdf(EDGES-mu)))
 if mode=="MCAR":
  for g in range(4):
   for y in [0,1]:dist[g,y]=pg[g]*([.8,.2][y])*scorepdf[y]
 elif mode=="MAR":
  mix=.8*scorepdf[0]+.2*scorepdf[1]
  intercept=brentq(lambda a:np.sum(mix*expit(a+strength*CENTERS))-q,-20,20)
  double=expit(intercept+strength*CENTERS)
  for g in range(4):
   for y in [0,1]:dist[g,y]=([.8,.2][y])*scorepdf[y]*(double if g==2 else (1-double)*rest[g])
 else:
  intercept=brentq(lambda a:.8*expit(a-strength)+.2*expit(a+strength)-q,-20,20)
  for g in range(4):
   for y in [0,1]:
    pr=expit(intercept+strength*(2*y-1));prob=pr if g==2 else (1-pr)*rest[g]
    factor=([1.,.8,.45,.8][g] if mode=="SCORE_INFORMATION_LOSS" else 1.)
    mu=(2*y-1)*sep*factor/2
    dist[g,y]=([.8,.2][y])*prob*np.diff(norm.cdf(EDGES-mu))
 assert np.isclose(dist.sum(),1) and np.isclose(dist[2].sum(),q)
 if mode=="MCAR":assert np.allclose(dist[:,1].sum(1)/dist.sum((1,2)),.2)
 return dist
def cumulative_scope(draws):
 # draws: repetitions x group x label x bin
 cum=draws.cumsum(-1)
 return np.einsum("sg,rglb->rslb",SCOPE_WEIGHTS,cum)
def select(cum,alpha,delta):
 n=cum.sum(2);k=cum[:,:,1,:]
 ok=(n>0)&(binom.cdf(k,n,alpha)<=delta)
 return np.max(np.where(ok,np.arange(B)[None,None,:],-1),axis=-1)
def bounds_at(cum,idx,delta):
 n=cum.sum(2);k=cum[:,:,1,:];ix=np.maximum(idx,0)
 nn=np.take_along_axis(n,ix[:,:,None],axis=2)[:,:,0];kk=np.take_along_axis(k,ix[:,:,None],axis=2)[:,:,0]
 ub=beta.ppf(1-delta,kk+1,np.maximum(nn-kk,1))
 return np.where(idx>=0,ub,np.nan)
def summarize(idx,bounds,part,pop,test,alpha):
 r=len(idx);mapping=MAPS[part]
 gi=np.maximum(mapping,0)
 ti=np.take_along_axis(idx,gi,axis=1);ub=np.take_along_axis(bounds,gi,axis=1)
 ti=np.where(mapping>=0,ti,-1);ub=np.where(ti>=0,ub,np.nan)
 pp=pop.cumsum(-1);tt=test.cumsum(-1)
 psel=np.zeros((r,4));ppos=np.zeros((r,4));nsel=np.zeros((r,4));npos=np.zeros((r,4))
 for g in range(4):
  valid=ti[:,g]>=0;pick=np.maximum(ti[:,g],0)
  psel[:,g]=pp[g,:,pick].sum(1)*valid;ppos[:,g]=pp[g,1,pick]*valid
  nsel[:,g]=tt[np.arange(r),g,:,pick].sum(1)*valid;npos[:,g]=tt[np.arange(r),g,1,pick]*valid
 def ratio(k,n):return np.divide(k,n,out=np.full_like(k,np.nan,dtype=float),where=n>0)
 risk=ratio(ppos.sum(1),psel.sum(1));double_risk=ratio(ppos[:,2],psel[:,2])
 # Evaluate the exact risk of DEPLOYED scopes, not a parent's diagnostic child.
 fail=np.zeros(r,bool);boundfail=np.zeros(r,bool);active=np.zeros(r,bool)
 for s in range(5):
  mask=mapping==s
  ns=(psel*mask).sum(1);ks=(ppos*mask).sum(1);rr=ratio(ks,ns)
  bb=bounds[:,s]
  fail|=(ns>0)&(rr>alpha+1e-12);boundfail|=(ns>0)&(rr>bb+1e-12);active|=ns>0
 return pd.DataFrame({"true_auto_rate":psel.sum(1),"true_auto_risk":risk,
 "double_auto_rate":psel[:,2]/pop[2].sum(),"double_true_risk":double_risk,
 "test_auto_n":nsel.sum(1).astype(int),"test_auto_positive_n":npos.sum(1).astype(int),
 "test_empirical_risk":ratio(npos.sum(1),nsel.sum(1)),
 "double_test_auto_n":nsel[:,2].astype(int),"double_test_positive_n":npos[:,2].astype(int),
 "any_deployed_scope_budget_violation":fail,"any_deployed_bound_noncoverage":boundfail,"has_population_service":active,
 "double_has_population_service":psel[:,2]>0,"partition":np.asarray(["G0","G1","G2"])[part]})
def main():
 start=time.time()
 cases=[]
 for mode in ["MCAR","MAR","INFORMATIVE","SCORE_INFORMATION_LOSS"]:
  strengths=[0.] if mode=="MCAR" else [0.,np.log(1.25),np.log(1.75),np.log(2.5)]
  for n,q,auc,st in itertools.product([1000,3000,10000,30000],[.05,.15,.3,.5],[.6,.7,.8,.9],strengths):
   cases.append((mode,n,q,auc,st))
 meta={"status":"RUNNING","started":datetime.datetime.now().isoformat(),"scenarios":len(cases),"repetitions":REPS,"replicate_datasets":len(cases)*REPS,
 "budgets":ALPHAS,"score_bins":B,"input":"exact class-conditional score distributions, not trained models","n_split":[.5,.25,.25],
 "risk_contrast":"observed prevalence ratio reported; strength is missingness log-odds, not directly imposed risk ratio",
 "confidence":"G0/G1/G2/fallback simultaneously over126 fixed candidates; locked3 per budget only; naive uncertified",
 "code_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"timeout_seconds":1800}
 path=O/"e2_protocol.json";path.write_text(json.dumps(meta,indent=2),encoding="utf-8")
 dest=O/"e2_replicates.csv.gz";summaries=[];scenario_rows=[]
 with gzip.open(dest,"wt",encoding="utf-8",newline="") as stream:
  first=True
  for case_id,(mode,n,q,auc,st) in enumerate(cases):
   if time.time()-start>1800:raise TimeoutError("Declared simulation timeout exceeded")
   pop=distribution(mode,q,auc,st)
   rng=np.random.default_rng(310000+case_id)
   fit=rng.multinomial(n//2,pop.reshape(-1),size=REPS).reshape(REPS,4,2,B)
   cal=rng.multinomial(n//4,pop.reshape(-1),size=REPS).reshape(REPS,4,2,B)
   test=rng.multinomial(n-n//2-n//4,pop.reshape(-1),size=REPS).reshape(REPS,4,2,B)
   fc=cumulative_scope(fit);cc=cumulative_scope(cal)
   neg=pop[:,0].sum(0)/.8;pos=pop[:,1].sum(0)/.2
   achieved=float(np.sum(pos*(np.cumsum(neg)-.5*neg)))
   prevalence=pop[:,1].sum(1)/pop.sum((1,2))
   base=dict(case_id=case_id,mechanism=mode,n_total=n,double_fraction=q,target_complete_auc=auc,missingness_strength=st,achieved_global_auc=achieved,double_prevalence=float(prevalence[2]),double_vs_complete_risk_ratio=float(prevalence[2]/prevalence[0]))
   scenario_rows.append(base)
   for alpha in ALPHAS:
    ids=select(cc,alpha,.05/(B*6));ub=bounds_at(cc,ids,.05/(B*6))
    naive=select(cc,alpha,.05);nub=bounds_at(cc,naive,.05)
    # A single threshold per G2 scope chosen from independent fit counts.
    fn=fc.sum(2);fk=fc[:,:,1,:]
    ok=(fn>=10)&(fk<=alpha*.5*fn)
    candidate=np.max(np.where(ok,np.arange(B)[None,None,:],-1),axis=-1)
    nn=np.take_along_axis(cc.sum(2),np.maximum(candidate,0)[:,:,None],axis=2)[:,:,0]
    kk=np.take_along_axis(cc[:,:,1,:],np.maximum(candidate,0)[:,:,None],axis=2)[:,:,0]
    locked=np.where((candidate>=0)&(nn>0)&(binom.cdf(kk,nn,alpha)<=.05/3),candidate,-1)
    lub=bounds_at(cc,locked,.05/3)
    fallback=np.where(np.all(ids[:,[1,3,4]]>=0,axis=1),2,np.where(np.all(ids[:,[1,2]]>=0,axis=1),1,0))
    options=[("G0",ids,ub,np.zeros(REPS,int)),("G1",ids,ub,np.ones(REPS,int)),("G2",ids,ub,np.full(REPS,2)),("G2_fallback",ids,ub,fallback),("G2_naive",naive,nub,np.full(REPS,2)),("G2_locked",locked,lub,np.full(REPS,2))]
    for policy,ix,bb,partition in options:
     out=summarize(ix,bb,partition,pop,test,alpha)
     out.insert(0,"repeat",np.arange(REPS));out.insert(0,"policy",policy);out.insert(0,"budget",alpha);out.insert(0,"case_id",case_id)
     out.to_csv(stream,index=False,header=first);first=False
     summaries.append({**base,"policy":policy,"budget":alpha,
      "auto_rate":out.true_auto_rate.mean(),"risk_mean_served":out.true_auto_risk.mean(),"double_auto_rate":out.double_auto_rate.mean(),"double_risk_mean_served":out.double_true_risk.mean(),
      "certification_success_fraction":out.has_population_service.mean(),"double_service_fraction":out.double_has_population_service.mean(),
      "deployed_budget_violation_probability":out.any_deployed_scope_budget_violation.mean(),"deployed_bound_noncoverage_probability":out.any_deployed_bound_noncoverage.mean(),
      "fallback_fraction":float(np.mean(partition!=2)) if policy=="G2_fallback" else 0.,"repetitions":REPS})
   if case_id%16==0 or case_id==len(cases)-1:
    pd.DataFrame(summaries).to_csv(O/"e2_boundary_summary.csv",index=False)
    meta.update(completed_scenarios=case_id+1,elapsed_seconds=time.time()-start)
    path.write_text(json.dumps(meta,indent=2),encoding="utf-8")
    print("E2",case_id+1,"/",len(cases),"elapsed",round(time.time()-start,1),flush=True)
 pd.DataFrame(scenario_rows).to_csv(O/"e2_population_properties.csv",index=False)
 pd.DataFrame(summaries).to_csv(O/"e2_boundary_summary.csv",index=False)
 meta.update(status="COMPLETED",completed_scenarios=len(cases),elapsed_seconds=time.time()-start)
 path.write_text(json.dumps(meta,indent=2),encoding="utf-8")
 print("E2 complete",flush=True)
if __name__=="__main__":main()

