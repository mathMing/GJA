"""Predeclared learnable scorer ablation; reference data never tune models."""
import time,json,argparse,warnings
import numpy as np,pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
import mechanism_training_v5 as sim
from grid_order_v9 import counts
from fixed_sequence_v6 import power_order,certify
ROOT=sim.ROOT
NAMES=['pooled_linear','pooled_poly2','double_linear','double_poly2']
def make_model(name):
 if name=='pooled_linear':return sim.model('masked_lr')
 steps=[]
 if name.startswith('pooled'):steps.append(('impute',SimpleImputer(strategy='median',add_indicator=True)))
 if name.endswith('poly2'):steps.append(('poly',PolynomialFeatures(degree=2,include_bias=False)))
 return Pipeline(steps+[('scale',StandardScaler()),('model',LogisticRegression(C=1,max_iter=2000))])
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
 out=ROOT/'results/learnable_score_v11'/('smoke' if args.smoke else 'formal');out.mkdir(parents=True,exist_ok=True)
 if (out/'run.json').exists():raise RuntimeError('Run already exists; inspect before rerun')
 reps=1 if args.smoke else 200;start=time.time();rows=[];training=[]
 meta={'status':'RUNNING','completed':0,'model_fits':0,'baseline_checks':0,'clinical_input':'NONE','code_sha256':sim.risk.digest(__file__),'protocol_sha256':sim.risk.digest(ROOT/'reports/v11_learnable_score_protocol.md'),'generator_sha256':sim.risk.digest(ROOT/'src/mechanism_training_v5.py')}
 def save():
  meta['elapsed_seconds']=time.time()-start;(out/'run.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
 save()
 old=pd.read_csv(ROOT/'results/grid_order_v9/formal/replicate_results.csv');old=old[old.model.eq('masked_lr')&old.method.eq('anchor2500')].set_index(['repeat','grid','budget'])
 try:
  for rep in range(reps):
   if time.time()-start>1800:raise TimeoutError('1800 second limit')
   streams=np.random.SeedSequence([70918,1,rep]).spawn(4)
   train,*rest=[sim.generate(np.random.default_rng(stream),n,'INFORMATIVE') for stream,n in zip(streams,[5000,2000,50000,50000])]
   group_masks=[d[4]=='missing_both' for d in rest]
   labels=[d[2][m] for d,m in zip(rest,group_masks)]
   trueprob=rest[2][3][group_masks[2]]
   for name in NAMES:
    subgroup=name.startswith('double');fitmask=train[4]=='missing_both' if subgroup else np.ones(5000,bool)
    x=train[0][fitmask,:2] if subgroup else train[0][fitmask];y=train[2][fitmask]
    assert len(np.unique(y))==2
    xx=[d[0][m,:2] if subgroup else d[0][m] for d,m in zip(rest,group_masks)]
    model=make_model(name)
    with warnings.catch_warnings(),threadpool_limits(limits=2):
     warnings.simplefilter('error',ConvergenceWarning);model.fit(x,y);ps,pc,pr=[model.predict_proba(a)[:,1] for a in xx]
    assert all(np.isfinite(a).all() for a in [ps,pc,pr])
    if not subgroup:assert np.allclose(model.named_steps['impute'].statistics_,np.nanmedian(x,axis=0))
    meta['model_fits']+=1;training.append(dict(repeat=rep,model=name,fit_n=len(y),fit_positive=int(y.sum()),feature_n=len(model.named_steps['model'].coef_[0])))
    for gridname,grid in [('original21',sim.risk.GRID),('dense_union',np.unique(np.r_[sim.risk.GRID,np.linspace(.001,1,101)]))]:
     ns,ks=counts(ps,labels[0],grid);nc,kc=counts(pc,labels[1],grid)
     for alpha in [.05,.1,.2]:
      order=power_order(ns,ks,2500/2000,alpha,.05/3)[0];rule,trace,_=certify(nc,kc,order,alpha,.05/3,grid)
      accept=pr<=rule['threshold'];n=int(accept.sum());service=n/len(pr);risk=float(trueprob[accept].mean()) if n else np.nan
      if name=='pooled_linear':
       previous=old.loc[(rep,gridname,alpha)]
       assert n==previous.reference_accepted and np.isclose(service,previous.service,atol=1e-14)
       assert (np.isnan(risk) and pd.isna(previous.reference_risk)) or np.isclose(risk,previous.reference_risk,atol=1e-14)
       meta['baseline_checks']+=1
      idx=rule['selected_index'];rows.append(dict(repeat=rep,model=name,grid=gridname,method='anchor2500',budget=alpha,threshold=rule['threshold'],cal_accepted=int(nc[idx]) if idx>=0 else 0,cal_positive=int(kc[idx]) if idx>=0 else 0,reference_n=len(pr),reference_accepted=n,service=service,reference_risk=risk,risk_bound=rule['bound']))
   meta['completed']=rep+1
   if rep%20==19 or rep==reps-1:
    pd.DataFrame(rows).to_csv(out/'replicate_results.csv',index=False);pd.DataFrame(training).to_csv(out/'training_counts.csv',index=False);save();print('completed',rep+1,'/',reps,'fits',meta['model_fits'],'seconds',round(time.time()-start),flush=True)
  r=pd.DataFrame(rows);assert len(r)==reps*24 and meta['baseline_checks']==reps*6
  assert not r.duplicated(['repeat','model','grid','budget']).any()
  assert r.loc[r.reference_accepted.eq(0),'reference_risk'].isna().all()
  meta.update(status='COMPLETED',rows=len(r),repetitions=reps);save()
 except Exception as exc:
  meta.update(status='FAILED',error=repr(exc));save();raise
if __name__=='__main__':main()
