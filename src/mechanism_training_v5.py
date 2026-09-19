"""Covariate-level missingness benchmark. Independent calibration; no patient data."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(key,'1')
import time,json,hashlib,argparse,platform
from pathlib import Path
import numpy as np,pandas as pd
from scipy.special import expit,softmax
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score,brier_score_loss
from threadpoolctl import threadpool_limits
import ars_experiment as risk

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/mechanism_training_v5'
MODES=['MCAR','MAR','INFORMATIVE'];BASE=np.array([.52,.30,.155,.025]);SLOPE=np.array([0,.3,1.2,.1])
def generate(rng,n,mode):
 z=rng.normal(size=(n,5));z1,z2,u,e1,e2=z.T
 h=.7*u+.3*z1+.7*e1;t=.7*u+.3*z2+.7*e2
 probability=expit(-2+.5*z1+.4*z2+.7*h+.7*t+.5*u+.3*z1*z2)
 visible=np.zeros(n) if mode=='MCAR' else z1 if mode=='MAR' else u
 maskprob=softmax(np.log(BASE)[None,:]+visible[:,None]*SLOPE,axis=1)
 group=(rng.random(n)[:,None]>np.cumsum(maskprob,axis=1)).sum(1)
 complete=np.column_stack([z1,z2,h,t]);masked=complete.copy()
 masked[np.isin(group,[2,3]),2]=np.nan;masked[np.isin(group,[1,2]),3]=np.nan
 assert np.array_equal(np.isnan(masked[:,2]),np.isin(group,[2,3]))
 assert np.array_equal(np.isnan(masked[:,3]),np.isin(group,[1,2]))
 if mode=='MCAR':assert np.allclose(maskprob,BASE)
 return masked,complete,rng.binomial(1,probability),probability,np.array(risk.GROUPS)[group]
def model(name):
 estimator=HistGradientBoostingClassifier(max_iter=80,max_leaf_nodes=7,l2_regularization=1,random_state=0) if name=='masked_hgb' else LogisticRegression(C=1,max_iter=2000)
 return Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=name!='complete_lr')),('scale',StandardScaler()),('model',estimator)])
def summarize(path):
 df=pd.read_csv(path/'replicate_results.csv')
 keys=['mechanism','n_total','model','policy','budget','mask_group']
 summary=df.groupby(keys).agg(repetitions=('repeat','size'),auto_rate_mean=('reference_auto_rate','mean'),auto_rate_sd=('reference_auto_rate','std'),reference_risk_mean_served=('reference_risk','mean'),service_fraction=('reference_has_service','mean'),reference_risk_exceeds_budget_fraction=('reference_risk_exceeds_budget','mean'),test_auto_n_mean=('test_auto_n','mean'),test_auto_positive_mean=('test_auto_positive_n','mean')).reset_index()
 summary['auto_rate_MCSE']=summary.auto_rate_sd/np.sqrt(summary.repetitions)
 summary.to_csv(path/'summary.csv',index=False)
 return summary
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
 path=OUT/('smoke' if args.smoke else 'formal');path.mkdir(parents=True,exist_ok=True)
 reps=1 if args.smoke else 200;ns=[1000] if args.smoke else [1000,10000];reference_n=2000 if args.smoke else 20000
 begin=time.time();meta={'status':'RUNNING','modes':MODES,'sample_sizes':ns,'repetitions_per_cell':reps,'independent_reference_n':reference_n,'planned_datasets':len(MODES)*len(ns)*reps,'completed_datasets':0,'model_fits':0,'timeout_seconds':1800,'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'python':platform.python_version(),'seed_scheme':'590000 + mechanism_index*100000 + size_index*10000 + repeat','clinical_input':'NONE'}
 def save():
  meta['elapsed_seconds']=time.time()-begin;(path/'run.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
 save();first=True;platformrows=[];maskrows=[];chunk=[]
 try:
  for mi,mode in enumerate(MODES):
   for ni,n in enumerate(ns):
    for rep in range(reps):
     if time.time()-begin>1800:raise TimeoutError('Predeclared runtime limit exceeded')
     rng=np.random.default_rng(590000+mi*100000+ni*10000+rep)
     x,full,y,prob,g=generate(rng,n,mode);xr,fr,yr,pr,gr=generate(rng,reference_n,mode)
     fit,cal,test=np.split(rng.permutation(n),[n//2,n//2+n//4]);assert len(set(fit)|set(cal)|set(test))==n
     assert not set(fit)&set(cal) and not set(fit)&set(test) and not set(cal)&set(test)
     common=dict(mechanism=mode,n_total=n,repeat=rep)
     for scope in risk.GROUPS:
      m=gr==scope;maskrows.append({**common,'mask_group':scope,'reference_n':int(m.sum()),'reference_fraction':float(m.mean()),'reference_prevalence':float(pr[m].mean())})
     for name in ['masked_lr','masked_hgb','complete_lr']:
      xx,rr=(full,fr) if name=='complete_lr' else (x,xr)
      est=model(name)
      with threadpool_limits(limits=2):
       est.fit(xx[fit],y[fit]);pc=est.predict_proba(xx[cal])[:,1];pt=est.predict_proba(xx[test])[:,1];pref=est.predict_proba(rr)[:,1]
      assert np.isfinite(pc).all() and np.isfinite(pref).all()
      assert np.allclose(est.named_steps['impute'].statistics_,np.nanmedian(xx[fit],axis=0))
      platformrows.append({**common,'model':name,'reference_AUROC':roc_auc_score(yr,pref),'reference_Brier':brier_score_loss(yr,pref)})
      ev=risk.evidence(pc,y[cal],g[cal]);ev63={s:(nn,k,risk.cp(k,nn,.05/63)) for s,(nn,k,u) in ev.items()}
      for alpha in risk.ALPHAS:
       r126=risk.rules_at(ev,alpha);r63=risk.rules_at(ev63,alpha)
       for policy,partition,rules in [('G0_search126','G0',r126),('G2_search126','G2',r126),('G2_fixed63','G2',r63)]:
        auto,t,b,sc=risk.apply(pt,g[test],rules,partition);ar,tr,br,scr=risk.apply(pref,gr,rules,partition)
        assert np.all(b[auto]<=alpha) and np.all(br[ar]<=alpha)
        for scope in ['all']+risk.GROUPS:
         tm=risk.scope_mask(g[test],scope);rm=risk.scope_mask(gr,scope);accept=ar&rm;nt=int((auto&tm).sum());nr=int(accept.sum());k=int(y[test][auto&tm].sum())
         reference_risk=float(pr[accept].mean()) if nr else np.nan
         chunk.append({**common,'model':name,'policy':policy,'budget':alpha,'mask_group':scope,'test_n':int(tm.sum()),'test_auto_n':nt,'test_auto_positive_n':k,'test_risk':k/nt if nt else np.nan,'reference_n':int(rm.sum()),'reference_accepted_n':nr,'reference_auto_rate':nr/int(rm.sum()) if rm.sum() else np.nan,'reference_risk':reference_risk,'reference_has_service':nr>0,'reference_risk_exceeds_budget':bool(reference_risk>alpha) if nr else False,'certificate_scope_is_original_group':partition=='G2' and scope in risk.GROUPS[:3]})
      meta['model_fits']+=1
     meta['completed_datasets']+=1
     if rep%10==9 or rep==reps-1:
      pd.DataFrame(chunk).to_csv(path/'replicate_results.csv',mode='w' if first else 'a',header=first,index=False);first=False;chunk=[];save()
      print(mode,n,rep+1,'/',reps,'datasets',meta['completed_datasets'],'seconds',round(time.time()-begin),flush=True)
  pd.DataFrame(platformrows).to_csv(path/'platform_metrics.csv',index=False);pd.DataFrame(maskrows).to_csv(path/'mask_properties.csv',index=False)
  s=summarize(path);expected=len(MODES)*len(ns)*reps*3*3*3*5
  assert len(pd.read_csv(path/'replicate_results.csv',usecols=['repeat']))==expected
  assert (s.repetitions==reps).all()
  meta.update(status='COMPLETED',result_rows=expected,checks=['input masking','MCAR constant probabilities','disjoint roles','fit-only imputation','finite predictions','active certificates within budget','complete row counts']);save()
  print('COMPLETE',json.dumps(meta),flush=True)
 except Exception as exc:
  meta.update(status='FAILED',error=repr(exc));save();raise
if __name__=='__main__':main()
