"""Numerical observed-information Bayes score for informative double missingness."""
import argparse,json,time
from pathlib import Path
import numpy as np,pandas as pd
from scipy.special import expit,softmax
from scipy.stats import binom
from numpy.polynomial.hermite import hermgauss
import mechanism_training_v5 as sim
from fixed_sequence_v6 import power_order,certify
from grid_order_v9 import counts

ROOT=sim.ROOT
def integrate(c,degree=64):
    nodes,w=hermgauss(degree);nodes=nodes*np.sqrt(2);w=w/np.sqrt(np.pi)
    q=softmax(np.log(sim.BASE)[None,:]+nodes[:,None]*sim.SLOPE,axis=1)[:,2]
    weights=((w*q)[:,None]*w[None,:]).ravel()/np.dot(w,q)
    shifts=(1.48*nodes[:,None]+.49*np.sqrt(2)*nodes[None,:]).ravel()
    c=np.atleast_1d(c);out=np.empty(len(c))
    for start in range(0,len(c),128):
        out[start:start+128]=np.sum(expit(c[start:start+128,None]+shifts[None,:])*weights[None,:],axis=1)
    return out

def coefficient(x):
    z1,z2=x[:,0],x[:,1]
    return -2+.71*z1+.61*z2+.3*z1*z2

def numerical_check():
    rng=np.random.default_rng(100918);z=rng.normal(size=(1000,5));a,b,u,e1,e2=z.T
    h=.7*u+.3*a+.7*e1;tt=.7*u+.3*b+.7*e2
    original=-2+.5*a+.4*b+.7*h+.7*tt+.5*u+.3*a*b
    reconstructed=coefficient(z)+1.48*u+.49*(e1+e2)
    assert np.allclose(original,reconstructed,atol=1e-12)
    grid=np.linspace(-30,30,30001);values=integrate(grid)
    check=np.linspace(-29.999,29.999,257);v64=integrate(check);v96=integrate(check,96)
    quadrature_error=float(np.max(np.abs(v64-v96)));interpolation_error=float(np.max(np.abs(np.interp(check,grid,values)-v64)))
    assert quadrature_error<2e-5 and interpolation_error<2e-5
    assert np.all(np.diff(values)>=-1e-14) and np.all((values>=0)&(values<=1))
    return grid,values,dict(status='PASS',max_quadrature_difference=quadrature_error,max_interpolation_difference=interpolation_error,algebra_max_error=float(np.max(np.abs(original-reconstructed))),check_points=257)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    out=ROOT/'results/observed_oracle_v10'/('smoke' if args.smoke else 'formal');out.mkdir(parents=True,exist_ok=True)
    if (out/'run.json').exists():raise RuntimeError('Existing run directory; inspect before rerunning')
    start=time.time();meta={'status':'RUNNING','completed':0,'model_fits':0,'clinical_input':'NONE','code_sha256':sim.risk.digest(__file__),'protocol_sha256':sim.risk.digest(ROOT/'reports/v10_observed_oracle_protocol.md'),'generator_sha256':sim.risk.digest(ROOT/'src/mechanism_training_v5.py')}
    def save():
        meta['elapsed_seconds']=time.time()-start;(out/'run.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    save();rows=[];front=[];reps=1 if args.smoke else 200
    try:
      interpolation_grid,values,check=numerical_check();(out/'numerical_checks.json').write_text(json.dumps(check,indent=2),encoding='utf-8')
      def predict(x):
        c=coefficient(x);p=np.interp(c,interpolation_grid,values);outside=(c<-30)|(c>30)
        if outside.any():p[outside]=integrate(c[outside])
        return p
      for rep in range(reps):
        if time.time()-start>1800:raise TimeoutError('1800 second limit')
        streams=np.random.SeedSequence([70918,1,rep]).spawn(4)
        data=[sim.generate(np.random.default_rng(stream),n,'INFORMATIVE') for stream,n in zip(streams[1:],[2000,50000,50000])]
        scores=[];labels=[];truth=[]
        for d in data:
          m=d[4]=='missing_both';x=d[0][m];assert np.isnan(x[:,2:]).all()
          scores.append(predict(x));labels.append(d[2][m]);truth.append(d[3][m])
        ps,pc,pr=scores;ys,yc,yr=labels
        # Reference-only diagnostic; never reused by calibration or ordering.
        ix=np.argsort(pr);mean=np.cumsum(pr[ix])/np.arange(1,len(pr)+1)
        for alpha in [.05,.1,.2]:
          good=np.flatnonzero(mean<=alpha);n=int(good[-1]+1) if len(good) else 0
          front.append(dict(repeat=rep,budget=alpha,service=n/len(pr),accepted=n,score_mean=float(mean[n-1]) if n else np.nan,latent_probability_mean=float(truth[2][ix[:n]].mean()) if n else np.nan))
        for gridname,grid in [('original21',sim.risk.GRID),('dense_union',np.unique(np.r_[sim.risk.GRID,np.linspace(.001,1,101)]))]:
          ns,ks=counts(ps,ys,grid);nc,kc=counts(pc,yc,grid)
          for alpha in [.05,.1,.2]:
            order=power_order(ns,ks,2500/2000,alpha,.05/3)[0]
            for method in ['anchor2500','bonferroni']:
              if method=='anchor2500':idx=certify(nc,kc,order,alpha,.05/3,grid)[0]['selected_index']
              else:
                valid=np.flatnonzero((nc>0)&(binom.cdf(kc,nc,alpha)<=.05/(3*len(grid))));idx=int(valid.max()) if len(valid) else -1
              threshold=float(grid[idx]) if idx>=0 else -np.inf;accepted=pr<=threshold;n=int(accepted.sum())
              rows.append(dict(repeat=rep,model='observed_oracle',grid=gridname,method=method,budget=alpha,threshold=threshold,cal_accepted=int(nc[idx]) if idx>=0 else 0,cal_positive=int(kc[idx]) if idx>=0 else 0,reference_n=len(pr),reference_accepted=n,service=n/len(pr),reference_risk=float(truth[2][accepted].mean()) if n else np.nan,reference_oracle_risk=float(pr[accepted].mean()) if n else np.nan,risk_bound=alpha if idx>=0 else np.nan))
        meta['completed']=rep+1
        if rep%20==19 or rep==reps-1:
          pd.DataFrame(rows).to_csv(out/'replicate_results.csv',index=False);pd.DataFrame(front).to_csv(out/'reference_diagnostic.csv',index=False);save();print('completed',rep+1,'/',reps,'seconds',round(time.time()-start),flush=True)
      r=pd.DataFrame(rows);assert len(r)==reps*12 and not r.duplicated(['repeat','grid','method','budget']).any()
      assert r.loc[r.reference_accepted.eq(0),'reference_risk'].isna().all()
      meta.update(status='COMPLETED',rows=len(r),repetitions=reps);save()
    except Exception as exc:
      meta.update(status='FAILED',error=repr(exc));save();raise
if __name__=='__main__':main()
