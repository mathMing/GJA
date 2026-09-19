"""Nested calibration enlargement with refitting and untouched outer test roles."""
import json,time,warnings
import numpy as np,pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
import v2_pipeline as v
from feature_sensitivity_v15 import rules

ROOT=v.ROOT;OUT=ROOT/'results/calibration_allocation_v20';PREV=ROOT/'results/temporal_sensitivity_v17'

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 if (OUT/'run.json').exists():raise RuntimeError('Existing run: no overwrite')
 previous=json.loads((PREV/'run.json').read_text());assert previous['status']=='COMPLETED'
 assert all(v.old.digest(p)==h for p,h in previous['input_hashes'].items())
 derived=ROOT/'results/selective_risk_v2/derived_patients_LOCAL.csv';cache=PREV/'score_cache_LOCAL.csv.gz'
 inputs={str(p):v.old.digest(p) for p in [derived,cache,PREV/'actions_LOCAL.csv.gz']}
 d=pd.read_csv(derived,dtype={'patient_id':str}).set_index('patient_id',drop=False)
 c=pd.read_csv(cache,dtype={'patient_id':str});c=c[c.cohort.eq('all_dates')&c.configuration.eq('preop_original')]
 columns=v.BASE+v.MRI+['hpv_normalized','tct_normalized'];start=time.time()
 meta={'status':'RUNNING','model_fits':0,'completed_splits':0,'input_hashes':inputs,'code_sha256':v.old.digest(__file__),'protocol_sha256':v.old.digest(ROOT/'reports/v20_calibration_allocation_protocol.md')}
 def save():meta['elapsed_seconds']=time.time()-start;v.dump(OUT/'run.json',meta)
 save();rows=[];acts=[];scores=[];platform=[];sizes=[];certs=[]
 try:
  for key,z in c.groupby(['split','seed','fold','model']):
   base=dict(zip(['split','seed','fold','model'],key));assert len(z)==z.patient_id.nunique()==773
   ff=z[z.role.eq('fit_oof')];cc=z[z.role.eq('calibration')];tt=z[z.role.eq('test')]
   seed=20200918 if base['split']=='primary' else 20200919+100*int(base['seed'])+int(base['fold'])
   ordered=ff.sort_values('patient_id').iloc[np.random.default_rng(seed).permutation(len(ff))]
   last=set(cc.patient_id)
   for allocation,fraction in [('original',None),('cal50',.5),('cal75',.75)]:
    if time.time()-start>1800:raise TimeoutError('30 minute limit')
    test=tt.copy()
    if fraction is None:fit,cal=ff.copy(),cc.copy()
    else:
     move=int(np.floor(fraction*(len(ff)+len(cc))))-len(cc);assert 0<move<len(ff)
     fit=ordered.iloc[move:].copy();cal=pd.concat([cc,ordered.iloc[:move]],ignore_index=True)
     assert last.issubset(set(cal.patient_id));last=set(cal.patient_id)
     x=d.loc[fit.patient_id];y=fit.true_label.to_numpy();pf=np.full(len(x),np.nan)
     with warnings.catch_warnings(),threadpool_limits(limits=2):
      warnings.simplefilter('error',ConvergenceWarning)
      for tr,te in StratifiedKFold(3,shuffle=True,random_state=4801).split(x,y):
       model=v.build(columns,base['model']);model.fit(x.iloc[tr],y[tr]);pf[te]=model.predict_proba(x.iloc[te])[:,1];meta['model_fits']+=1
      model=v.build(columns,base['model']);model.fit(x,y);cal['score']=model.predict_proba(d.loc[cal.patient_id])[:,1];test['score']=model.predict_proba(d.loc[test.patient_id])[:,1];meta['model_fits']+=1
     fit['score']=pf
    assert set(fit.patient_id).isdisjoint(cal.patient_id) and set(test.patient_id).isdisjoint(set(fit.patient_id)|set(cal.patient_id))
    assert len(fit)+len(cal)+len(test)==773
    for role,frame in [('fit_oof',fit),('calibration',cal),('test',test)]:
     assert np.isfinite(frame.score).all();assert np.array_equal(d.loc[frame.patient_id,'true_label'],frame.true_label)
     scores.append(frame[['patient_id','mask_group','true_label','score']].assign(**base,allocation=allocation,role=role))
     sizes.append({**base,'allocation':allocation,'role':role,'n':len(frame),'double_n':int(frame.mask_group.eq('missing_both').sum())})
    platform.append({**base,'allocation':allocation,'auroc':roc_auc_score(test.true_label,test.score),'auprc':average_precision_score(test.true_label,test.score),'brier':brier_score_loss(test.true_label,test.score)})
    for alpha in v.old.ALPHAS:
     for partition in ['G0','G2']:
      rr=rules(fit,cal,alpha,partition);auto,threshold,bound,scope=v.old.apply(test.score.to_numpy(),test.mask_group.to_numpy(),rr,partition)
      m0={**base,'allocation':allocation,'risk_budget':alpha,'policy':partition+'_FST'}
      assert np.all(bound[auto]<=alpha)
      for group,rule in rr.items():certs.append({**m0,'scope':group,**rule})
      acts.append(pd.DataFrame({**m0,'patient_id':test.patient_id.to_numpy(),'mask_group':test.mask_group.to_numpy(),'true_label':test.true_label.to_numpy(),'action':np.where(auto,'AUTO_NEGATIVE','ABSTAIN'),'threshold':threshold,'risk_bound':bound,'certificate_scope':scope}))
      for group in ['all']+v.old.GROUPS:
       m=np.ones(len(test),bool) if group=='all' else test.mask_group.to_numpy()==group;yy=test.true_label.to_numpy();n=int(m.sum());pos=int(yy[m].sum());na=int((m&auto).sum());k=int(yy[m&auto].sum())
       rows.append({**m0,'mask_group':group,'n':n,'positive_n':pos,'auto_n':na,'auto_positive_n':k,'service':na/n if n else np.nan,'risk':k/na if na else np.nan,'miss':k/pos if pos else np.nan})
   meta['completed_splits']+=1
   if meta['completed_splits']%10==0:save();print('completed',meta['completed_splits'],'/52; fits',meta['model_fits'],flush=True)
  r=pd.DataFrame(rows);a=pd.concat(acts,ignore_index=True);assert len(r)==4680 and meta['model_fits']==416
  old=pd.read_csv(PREV/'actions_LOCAL.csv.gz',dtype={'patient_id':str});old=old[old.cohort.eq('all_dates')&old.configuration.eq('preop_original')]
  keys=['split','seed','fold','model','policy','risk_budget','patient_id'];assert not a.duplicated(keys+['allocation']).any()
  pair=a[a.allocation.eq('original')].merge(old,on=keys,suffixes=('_new','_old'),validate='one_to_one');assert len(pair)==len(old) and pair.action_new.eq(pair.action_old).all() and np.allclose(pair.threshold_new,pair.threshold_old,equal_nan=True)
  r.to_csv(OUT/'fold_metrics.csv',index=False);a.to_csv(OUT/'actions_LOCAL.csv.gz',index=False,compression='gzip');pd.concat(scores,ignore_index=True).to_csv(OUT/'score_cache_LOCAL.csv.gz',index=False,compression='gzip');pd.DataFrame(platform).to_csv(OUT/'platform_metrics.csv',index=False);pd.DataFrame(sizes).to_csv(OUT/'role_sizes.csv',index=False);pd.DataFrame(certs).to_csv(OUT/'certificates.csv',index=False)
  s=r[r.split.eq('repeated')].groupby(['seed','model','allocation','risk_budget','policy','mask_group'])[['n','positive_n','auto_n','auto_positive_n']].sum().reset_index();s['service']=s.auto_n/s.n;s['risk']=s.auto_positive_n/s.auto_n.replace(0,np.nan);s['miss']=s.auto_positive_n/s.positive_n.replace(0,np.nan);s.to_csv(OUT/'seed_metrics.csv',index=False)
  assert all(v.old.digest(p)==h for p,h in inputs.items()) and all(v.old.digest(p)==h for p,h in previous['input_hashes'].items())
  meta.update(status='COMPLETED',fold_rows=len(r),action_rows=len(a),baseline_patient_checks=len(pair));save();print(json.dumps(meta))
 except Exception as exc:meta.update(status='FAILED',error=repr(exc));save();raise

if __name__=='__main__':main()
