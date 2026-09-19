"""Fresh-seed validation of the frozen V11 pipeline across existing mechanisms."""
import argparse,json,time,warnings
import numpy as np,pandas as pd
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
import mechanism_training_v5 as sim
from learnable_score_v11 import make_model,NAMES
from grid_order_v9 import counts
from fixed_sequence_v6 import power_order,certify
ROOT=sim.ROOT
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
 out=ROOT/'results/frozen_validation_v12'/('smoke' if args.smoke else 'formal');out.mkdir(parents=True,exist_ok=True)
 if (out/'run.json').exists():raise RuntimeError('Existing run; inspect before rerun')
 reps=1 if args.smoke else 200;rows=[];tr=[];start=time.time()
 meta={'status':'RUNNING','completed':0,'model_fits':0,'seed_root':120918,'repetitions_per_mechanism':reps,'clinical_input':'NONE','code_sha256':sim.risk.digest(__file__),'model_sha256':sim.risk.digest(ROOT/'src/learnable_score_v11.py'),'generator_sha256':sim.risk.digest(ROOT/'src/mechanism_training_v5.py'),'protocol_sha256':sim.risk.digest(ROOT/'reports/v12_frozen_validation_protocol.md')}
 def save():
  meta['elapsed_seconds']=time.time()-start;(out/'run.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
 save()
 try:
  for mi,mechanism in enumerate(['MCAR','MAR','INFORMATIVE']):
   for rep in range(reps):
    if time.time()-start>1800:raise TimeoutError('1800 seconds')
    streams=np.random.SeedSequence([120918,mi,rep]).spawn(4)
    train,*rest=[sim.generate(np.random.default_rng(stream),n,mechanism) for stream,n in zip(streams,[5000,2000,50000,50000])]
    masks=[d[4]=='missing_both' for d in rest];ys,yc,yr=[d[2][m] for d,m in zip(rest,masks)];truth=rest[2][3][masks[2]]
    for name in NAMES:
     sub=name.startswith('double');fm=train[4]=='missing_both' if sub else np.ones(5000,bool)
     x=train[0][fm,:2] if sub else train[0][fm];y=train[2][fm];assert len(np.unique(y))==2
     xx=[d[0][m,:2] if sub else d[0][m] for d,m in zip(rest,masks)];model=make_model(name)
     with warnings.catch_warnings(),threadpool_limits(limits=2):
      warnings.simplefilter('error',ConvergenceWarning);model.fit(x,y);ps,pc,pr=[model.predict_proba(a)[:,1] for a in xx]
     assert all(np.isfinite(p).all() for p in [ps,pc,pr])
     if not sub:assert np.allclose(model.named_steps['impute'].statistics_,np.nanmedian(x,axis=0))
     meta['model_fits']+=1;tr.append(dict(mechanism=mechanism,repeat=rep,model=name,fit_n=len(y),positive_n=int(y.sum())))
     for gn,grid in [('original21',sim.risk.GRID),('dense_union',np.unique(np.r_[sim.risk.GRID,np.linspace(.001,1,101)]))]:
      ns,ks=counts(ps,ys,grid);nc,kc=counts(pc,yc,grid)
      for alpha in [.05,.1,.2]:
       order=power_order(ns,ks,2500/2000,alpha,.05/3)[0];rule,trace,_=certify(nc,kc,order,alpha,.05/3,grid)
       a=pr<=rule['threshold'];n=int(a.sum());idx=rule['selected_index']
       rows.append(dict(mechanism=mechanism,repeat=rep,model=name,grid=gn,budget=alpha,threshold=rule['threshold'],reference_n=len(pr),reference_accepted=n,service=n/len(pr),reference_risk=float(truth[a].mean()) if n else np.nan,risk_bound=rule['bound'],cal_accepted=int(nc[idx]) if idx>=0 else 0,cal_positive=int(kc[idx]) if idx>=0 else 0))
    meta['completed']+=1
    if rep%20==19 or rep==reps-1:
     pd.DataFrame(rows).to_csv(out/'replicate_results.csv',index=False);pd.DataFrame(tr).to_csv(out/'training_counts.csv',index=False);save();print(mechanism,rep+1,'/',reps,'fits',meta['model_fits'],'seconds',round(time.time()-start),flush=True)
  r=pd.DataFrame(rows);assert len(r)==3*reps*24 and meta['model_fits']==3*reps*4
  assert not r.duplicated(['mechanism','repeat','model','grid','budget']).any()
  assert r.loc[r.reference_accepted.eq(0),'reference_risk'].isna().all()
  assert sim.risk.digest(ROOT/'src/learnable_score_v11.py')==meta['model_sha256']
  meta.update(status='COMPLETED',rows=len(r));save()
 except Exception as exc:
  meta.update(status='FAILED',error=repr(exc));save();raise
if __name__=='__main__':main()
