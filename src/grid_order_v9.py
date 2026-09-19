"""Exploratory grid/order ablation, with exact paired V7 baseline check."""
import time,json,argparse,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import binom
from threadpoolctl import threadpool_limits
import mechanism_training_v5 as sim
from fixed_sequence_v6 import power_order,certify

ROOT=sim.ROOT
def counts(scores,labels,grid):
    ix=np.argsort(scores);s=scores[ix];cum=np.r_[0,np.cumsum(labels[ix])]
    n=np.searchsorted(s,grid,side='right');return n,cum[n]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    out=ROOT/'results/grid_order_v9'/('smoke' if args.smoke else 'formal');out.mkdir(parents=True,exist_ok=True)
    if (out/'run.json').exists():raise RuntimeError('Existing run directory: inspect before rerunning')
    reps=1 if args.smoke else 200;begin=time.time();rows=[]
    meta={'status':'RUNNING','repetitions':reps,'completed':0,'model_fits':0,'clinical_input':'NONE','code_sha256':sim.risk.digest(__file__),'protocol_sha256':sim.risk.digest(ROOT/'reports/v9_grid_order_protocol.md'),'generator_sha256':sim.risk.digest(ROOT/'src/mechanism_training_v5.py'),'fst_sha256':sim.risk.digest(ROOT/'src/fixed_sequence_v6.py')}
    def save():
        meta['elapsed_seconds']=time.time()-begin;(out/'run.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    save()
    baseline=pd.read_csv(ROOT/'results/calibration_information_v7/formal/replicate_results.csv')
    baseline=baseline[baseline.mechanism.eq('INFORMATIVE')&baseline.mask_group.eq('missing_both')&baseline.calibration_n.eq(50000)&baseline.policy.eq('G2_FST')].set_index(['repeat','model','budget'])
    try:
      for rep in range(reps):
        if time.time()-begin>1800:raise TimeoutError('1800 second limit')
        streams=np.random.SeedSequence([70918,1,rep]).spawn(4)
        datasets=[sim.generate(np.random.default_rng(stream),n,'INFORMATIVE') for stream,n in zip(streams,[5000,2000,50000,50000])]
        train,selection,cal,ref=datasets
        for name in ['masked_lr','masked_hgb']:
          model=sim.model(name)
          with threadpool_limits(limits=2):
            model.fit(train[0],train[2]);ps,pc,pr=[model.predict_proba(d[0])[:,1] for d in datasets[1:]]
          meta['model_fits']+=1
          assert np.allclose(model.named_steps['impute'].statistics_,np.nanmedian(train[0],axis=0))
          ms=selection[4]=='missing_both';mc=cal[4]=='missing_both';mr=ref[4]=='missing_both'
          for gridname,grid in [('original21',sim.risk.GRID),('dense_union',np.unique(np.r_[sim.risk.GRID,np.linspace(.001,1,101)]))]:
            ns,ks=counts(ps[ms],selection[2][ms],grid);nc,kc=counts(pc[mc],cal[2][mc],grid)
            for alpha in [.05,.1,.2]:
              orders={'anchor2500':power_order(ns,ks,2500/2000,alpha,.05/3)[0],'matched50000':power_order(ns,ks,50000/2000,alpha,.05/3)[0],'risk_ascending':np.lexsort((-np.arange(len(grid)),(ks+.5)/(ns+1)))}
              for method in list(orders)+['bonferroni']:
                if method=='bonferroni':
                  valid=np.flatnonzero((nc>0)&(binom.cdf(kc,nc,alpha)<=.05/(3*len(grid))))
                  idx=int(valid.max()) if len(valid) else -1;tested=len(grid)
                else:
                  rule,trace,_=certify(nc,kc,orders[method],alpha,.05/3,grid);idx=rule['selected_index'];tested=len(trace)
                threshold=float(grid[idx]) if idx>=0 else -np.inf
                accept=mr&(pr<=threshold);n=int(accept.sum());service=n/int(mr.sum());risk=float(ref[3][accept].mean()) if n else np.nan
                if gridname=='original21' and method=='anchor2500':
                  old=baseline.loc[(rep,name,alpha)]
                  assert n==old.reference_accepted and np.isclose(service,old.reference_service,atol=1e-14)
                  assert (np.isnan(risk) and pd.isna(old.reference_risk)) or np.isclose(risk,old.reference_risk,atol=1e-14)
                rows.append(dict(repeat=rep,model=name,grid=gridname,candidates=len(grid),method=method,budget=alpha,threshold=threshold,tested=tested,cal_group_n=int(mc.sum()),cal_accepted=int(nc[idx]) if idx>=0 else 0,cal_positive=int(kc[idx]) if idx>=0 else 0,reference_n=int(mr.sum()),reference_accepted=n,service=service,reference_risk=risk,risk_bound=alpha if idx>=0 else np.nan))
        meta['completed']+=1
        if rep%10==9 or rep==reps-1:
          pd.DataFrame(rows).to_csv(out/'replicate_results.csv',index=False);save();print('completed',rep+1,'/',reps,'seconds',round(time.time()-begin),flush=True)
      r=pd.DataFrame(rows);assert len(r)==reps*2*2*3*4
      assert not r.duplicated(['repeat','model','grid','method','budget']).any()
      assert r.loc[r.reference_accepted.eq(0),'reference_risk'].isna().all()
      meta.update(status='COMPLETED',rows=len(r),baseline_checks=reps*2*3);save()
    except Exception as exc:
      meta.update(status='FAILED',error=repr(exc));save();raise
if __name__=='__main__':main()
