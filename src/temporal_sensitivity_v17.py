"""Prespecified date-cohort sensitivity; retain outer roles, refit within cohort."""
import json,time,warnings
import numpy as np,pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
import v2_pipeline as v
from feature_sensitivity_v15 import rules

ROOT=v.ROOT;OUT=ROOT/'results/temporal_sensitivity_v17';PREV=ROOT/'results/feature_sensitivity_v15'

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 if (OUT/'run.json').exists():raise RuntimeError('Existing run: no overwrite')
 derived=ROOT/'results/selective_risk_v2/derived_patients_LOCAL.csv';cache=PREV/'score_cache_LOCAL.csv.gz'
 queue=ROOT/'results/clinical_review_queue/clinical_review_queue_LOCAL.csv'
 previous=json.loads((PREV/'run.json').read_text());assert previous['status']=='COMPLETED'
 assert v.old.digest(derived)==previous['derived_sha256'] and v.old.digest(v.SOURCE)==previous['source_sha256']
 sources={str(p):v.old.digest(p) for p in [derived,cache,queue,v.SOURCE]}
 d=pd.read_csv(derived,dtype={'patient_id':str}).set_index('patient_id',drop=False)
 t=d.mri_days_before_surgery;assert t.notna().all()
 cohorts={'all_dates':d.index,'exclude_after':d.index[t>=0],'strict_before':d.index[t>0]}
 expected={'all_dates':(773,149,113,33),'exclude_after':(770,148,113,33),'strict_before':(692,137,103,30)}
 footprints=[]
 for name,ids in cohorts.items():
  q=d.loc[ids];b=q[q.mask_group.eq('missing_both')]
  values=(len(q),int(q.true_label.sum()),len(b),int(b.true_label.sum()));assert values==expected[name]
  footprints.append(dict(zip(['cohort','n','positive_n','double_n','double_positive_n'],[name,*values])))
 pd.DataFrame(footprints).to_csv(OUT/'cohort_summary.csv',index=False)
 configs=json.loads((PREV/'feature_manifest.json').read_text());configs={k:configs[k] for k in ['preop_original','without_mri_nodes']}
 c=pd.read_csv(cache,dtype={'patient_id':str});c=c[c.configuration.isin(configs)]
 start=time.time();meta={'status':'RUNNING','model_fits':0,'completed_scorer_cohorts':0,'input_hashes':sources,'protocol_sha256':v.old.digest(ROOT/'reports/v17_temporal_sensitivity_protocol.md'),'code_sha256':v.old.digest(__file__)}
 def save():meta['elapsed_seconds']=time.time()-start;v.dump(OUT/'run.json',meta)
 save();acts=[];rows=[];scores=[];certs=[]
 try:
  for key,z in c.groupby(['split','seed','fold','model','configuration']):
   base=dict(zip(['split','seed','fold','model','configuration'],key))
   assert len(z)==773 and z.patient_id.nunique()==773
   assert np.array_equal(d.loc[z.patient_id,'true_label'],z.true_label) and np.array_equal(d.loc[z.patient_id,'mask_group'],z.mask_group)
   for cohort,ids in cohorts.items():
    if time.time()-start>1800:raise TimeoutError('30 minute limit')
    zz=z[z.patient_id.isin(ids)];assert len(zz)==len(ids) and zz.patient_id.nunique()==len(ids)
    fit,cal,test=[zz[zz.role.eq(role)].copy() for role in ['fit_oof','calibration','test']]
    assert len(fit)+len(cal)+len(test)==len(ids)
    if cohort!='all_dates':
     x=d.loc[fit.patient_id];y=fit.true_label.to_numpy();pf=np.full(len(x),np.nan)
     with warnings.catch_warnings(),threadpool_limits(limits=2):
      warnings.simplefilter('error',ConvergenceWarning)
      for tr,te in StratifiedKFold(3,shuffle=True,random_state=4801).split(x,y):
       model=v.build(configs[base['configuration']],base['model']);model.fit(x.iloc[tr],y[tr]);pf[te]=model.predict_proba(x.iloc[te])[:,1];meta['model_fits']+=1
      model=v.build(configs[base['configuration']],base['model']);model.fit(x,y)
      cal['score']=model.predict_proba(d.loc[cal.patient_id])[:,1];test['score']=model.predict_proba(d.loc[test.patient_id])[:,1];meta['model_fits']+=1
     fit['score']=pf
    for role,frame in [('fit_oof',fit),('calibration',cal),('test',test)]:
     assert np.isfinite(frame.score).all()
     scores.append(frame[['patient_id','mask_group','true_label','score']].assign(**base,cohort=cohort,role=role))
    for alpha in v.old.ALPHAS:
     for partition in ['G0','G2']:
      rr=rules(fit,cal,alpha,partition);auto,threshold,bound,scope=v.old.apply(test.score.to_numpy(),test.mask_group.to_numpy(),rr,partition)
      assert np.all(bound[auto]<=alpha)
      m0={**base,'cohort':cohort,'policy':partition+'_FST','risk_budget':alpha}
      for group,rule in rr.items():certs.append({**m0,'scope':group,'candidate_count':len(v.old.GRID),**rule})
      acts.append(pd.DataFrame({**m0,'patient_id':test.patient_id.to_numpy(),'mask_group':test.mask_group.to_numpy(),'true_label':test.true_label.to_numpy(),'score':test.score.to_numpy(),'action':np.where(auto,'AUTO_NEGATIVE','ABSTAIN'),'threshold':threshold,'risk_bound':bound,'certificate_scope':scope}))
      for group in ['all']+v.old.GROUPS:
       m=np.ones(len(test),bool) if group=='all' else test.mask_group.to_numpy()==group;ytest=test.true_label.to_numpy()
       n=int(m.sum());pos=int(ytest[m].sum());na=int((m&auto).sum());k=int(ytest[m&auto].sum())
       rows.append({**m0,'mask_group':group,'n':n,'positive_n':pos,'auto_n':na,'auto_positive_n':k,'service':na/n if n else np.nan,'risk':k/na if na else np.nan,'miss':k/pos if pos else np.nan,'abstain_n':n-na})
    meta['completed_scorer_cohorts']+=1
   if meta['completed_scorer_cohorts']%30==0:save();print('completed',meta['completed_scorer_cohorts'],'/312; fits',meta['model_fits'],flush=True)
  a=pd.concat(acts,ignore_index=True);keys=['split','seed','fold','model','configuration','policy','risk_budget','patient_id']
  assert not a.duplicated(keys+['cohort']).any()
  old=pd.read_csv(PREV/'actions_LOCAL.csv.gz',dtype={'patient_id':str});old=old[old.configuration.isin(configs)]
  pair=a[a.cohort.eq('all_dates')].merge(old,on=keys,suffixes=('_new','_old'),validate='one_to_one')
  assert len(pair)==len(old)==97416 and pair.action_new.eq(pair.action_old).all() and np.allclose(pair.threshold_new,pair.threshold_old,equal_nan=True)
  for key,z in a[a.split.eq('repeated')].groupby(['cohort','seed','model','configuration','policy','risk_budget']):
   assert len(z)==len(cohorts[key[0]]) and z.patient_id.nunique()==len(z)
  r=pd.DataFrame(rows);assert len(r)==9360 and meta['model_fits']==832
  s=r[r.split.eq('repeated')].groupby(['cohort','seed','model','configuration','policy','risk_budget','mask_group'])[['n','positive_n','auto_n','auto_positive_n','abstain_n']].sum().reset_index()
  s['service']=s.auto_n/s.n;s['risk']=s.auto_positive_n/s.auto_n.replace(0,np.nan);s['miss']=s.auto_positive_n/s.positive_n.replace(0,np.nan)
  a.to_csv(OUT/'actions_LOCAL.csv.gz',index=False,compression='gzip');pd.concat(scores,ignore_index=True).to_csv(OUT/'score_cache_LOCAL.csv.gz',index=False,compression='gzip')
  r.to_csv(OUT/'fold_metrics.csv',index=False);s.to_csv(OUT/'seed_metrics.csv',index=False);pd.DataFrame(certs).to_csv(OUT/'certificates.csv',index=False)
  assert all(v.old.digest(p)==h for p,h in sources.items())
  meta.update(status='COMPLETED',action_rows=len(a),fold_rows=len(r),baseline_patient_checks=len(pair));save();print(json.dumps(meta),flush=True)
 except Exception as exc:meta.update(status='FAILED',error=repr(exc));save();raise

if __name__=='__main__':main()
