"""Frozen patient roles, prespecified field removal, no value adjudication."""
import json,time,warnings
import numpy as np,pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
import v2_pipeline as v
from fixed_sequence_v6 import power_order,certify
from grid_order_v9 import counts
ROOT=v.ROOT;OUT=ROOT/'results/feature_sensitivity_v15';CACHE=ROOT/'results/fixed_sequence_v6/score_cache_LOCAL.csv.gz'
def rules(fit,cal,alpha,partition):
 result={}
 for scope in (['all'] if partition=='G0' else v.old.GROUPS[:3]):
  ff=fit if scope=='all' else fit[fit.mask_group.eq(scope)];cc=cal if scope=='all' else cal[cal.mask_group.eq(scope)]
  nf,kf=counts(ff.score.to_numpy(),ff.true_label.to_numpy(),v.old.GRID);nc,kc=counts(cc.score.to_numpy(),cc.true_label.to_numpy(),v.old.GRID)
  delta=.05 if partition=='G0' else .05/3;order=power_order(nf,kf,len(cal)/len(fit),alpha,delta)[0]
  result[scope]=certify(nc,kc,order,alpha,delta,v.old.GRID)[0]
 return result
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 if (OUT/'run.json').exists():raise RuntimeError('Existing run; inspect before rerun')
 previous=json.loads((ROOT/'results/real_global_subgroup_v13/run.json').read_text());assert previous['status']=='COMPLETED'
 derived=ROOT/'results/selective_risk_v2/derived_patients_LOCAL.csv';assert v.old.digest(derived)==previous['derived_sha256'] and v.old.digest(v.SOURCE)==previous['source_sha256'] and v.old.digest(CACHE)==previous['cache_sha256']
 d=pd.read_csv(derived,dtype={'patient_id':str}).set_index('patient_id',drop=False);c=pd.read_csv(CACHE,dtype={'patient_id':str});c=c[c.features.eq('preop')]
 start=time.time();meta={'status':'RUNNING','completed_scorer_splits':0,'model_fits':0,'source_sha256':previous['source_sha256'],'derived_sha256':previous['derived_sha256'],'cache_sha256':previous['cache_sha256'],'protocol_sha256':v.old.digest(ROOT/'reports/v15_feature_sensitivity_protocol.md'),'code_sha256':v.old.digest(__file__)}
 def save():meta['elapsed_seconds']=time.time()-start;v.dump(OUT/'run.json',meta)
 save();acts=[];rows=[];scores=[]
 features=v.BASE+v.MRI+['hpv_normalized','tct_normalized'];configs={'preop_original':features,'without_mri_nodes':[x for x in features if x!='mri_nodes'],'without_imaging_tables':[x for x in features if x not in v.MRI]}
 v.dump(OUT/'feature_manifest.json',configs)
 try:
  for key,z in c.groupby(['split','seed','fold','model']):
   if time.time()-start>1800:raise TimeoutError('1800 seconds')
   base=dict(zip(['split','seed','fold','model'],key));assert len(z)==773 and z.patient_id.nunique()==773 and z.true_label.sum()==149
   assert np.array_equal(d.loc[z.patient_id,'true_label'].to_numpy(),z.true_label.to_numpy()) and np.array_equal(d.loc[z.patient_id,'mask_group'].to_numpy(),z.mask_group.to_numpy())
   ff=z[z.role.eq('fit_oof')];cc=z[z.role.eq('calibration')];tt=z[z.role.eq('test')];assert len(ff)+len(cc)+len(tt)==773
   for name,columns in configs.items():
    fit,cal,test=ff.copy(),cc.copy(),tt.copy()
    if name!='preop_original':
     x=d.loc[fit.patient_id];y=fit.true_label.to_numpy();pf=np.full(len(x),np.nan)
     with warnings.catch_warnings(),threadpool_limits(limits=2):
      warnings.simplefilter('error',ConvergenceWarning)
      for tr,te in StratifiedKFold(3,shuffle=True,random_state=4801).split(x,y):
       model=v.build(columns,base['model']);model.fit(x.iloc[tr],y[tr]);pf[te]=model.predict_proba(x.iloc[te])[:,1];meta['model_fits']+=1
      model=v.build(columns,base['model']);model.fit(x,y);cal['score']=model.predict_proba(d.loc[cal.patient_id])[:,1];test['score']=model.predict_proba(d.loc[test.patient_id])[:,1];meta['model_fits']+=1
     fit['score']=pf
    for role,frame in [('fit_oof',fit),('calibration',cal),('test',test)]:
     assert np.isfinite(frame.score).all();scores.append(frame[['patient_id','mask_group','true_label','score']].assign(**base,configuration=name,role=role))
    for alpha in v.old.ALPHAS:
     for partition in ['G0','G2']:
      rr=rules(fit,cal,alpha,partition);auto,threshold,bound,scope=v.old.apply(test.score.to_numpy(),test.mask_group.to_numpy(),rr,partition);assert np.all(bound[auto]<=alpha)
      meta0={**base,'configuration':name,'policy':partition+'_FST','risk_budget':alpha}
      acts.append(pd.DataFrame({**meta0,'patient_id':test.patient_id.to_numpy(),'mask_group':test.mask_group.to_numpy(),'true_label':test.true_label.to_numpy(),'action':np.where(auto,'AUTO_NEGATIVE','ABSTAIN'),'threshold':threshold,'risk_bound':bound,'certificate_scope':scope}))
      for group in ['all']+v.old.GROUPS:
       m=np.ones(len(test),bool) if group=='all' else test.mask_group.to_numpy()==group;ytest=test.true_label.to_numpy();n=int(m.sum());pos=int(ytest[m].sum());na=int((m&auto).sum());k=int(ytest[m&auto].sum())
       rows.append({**meta0,'mask_group':group,'n':n,'positive_n':pos,'auto_n':na,'auto_positive_n':k,'service':na/n if n else np.nan,'risk':k/na if na else np.nan,'miss':k/pos if pos else np.nan})
   meta['completed_scorer_splits']+=1
   if meta['completed_scorer_splits']%10==0:save();print('completed',meta['completed_scorer_splits'],'/52','fits',meta['model_fits'],flush=True)
  a=pd.concat(acts,ignore_index=True);key=['split','seed','fold','model','policy','risk_budget','patient_id'];assert not a.duplicated(key+['configuration']).any()
  old=pd.read_csv(ROOT/'results/real_global_subgroup_v13/actions_LOCAL.csv.gz',dtype={'patient_id':str});old=old[old.features.eq('preop')&old.policy.isin(['G0_FST','G2_FST'])]
  pair=a[a.configuration.eq('preop_original')].merge(old,on=key,suffixes=('_new','_old'),validate='one_to_one');assert len(pair)==len(old) and pair.action_new.eq(pair.action_old).all() and np.allclose(pair.threshold_new,pair.threshold_old,equal_nan=True)
  r=pd.DataFrame(rows);assert len(r)==4680 and meta['model_fits']==416
  a.to_csv(OUT/'actions_LOCAL.csv.gz',index=False,compression='gzip');pd.concat(scores,ignore_index=True).to_csv(OUT/'score_cache_LOCAL.csv.gz',index=False,compression='gzip');r.to_csv(OUT/'fold_metrics.csv',index=False)
  s=r[r.split.eq('repeated')].groupby(['seed','model','configuration','policy','risk_budget','mask_group'])[['n','positive_n','auto_n','auto_positive_n']].sum().reset_index();s['service']=s.auto_n/s.n;s['risk']=s.auto_positive_n/s.auto_n.replace(0,np.nan);s['miss']=s.auto_positive_n/s.positive_n.replace(0,np.nan);s.to_csv(OUT/'seed_metrics.csv',index=False)
  assert v.old.digest(derived)==meta['derived_sha256'] and v.old.digest(CACHE)==meta['cache_sha256'] and v.old.digest(v.SOURCE)==meta['source_sha256']
  meta.update(status='COMPLETED',fold_rows=len(r),action_rows=len(a),baseline_patient_checks=len(pair));save();print(json.dumps(meta),flush=True)
 except Exception as exc:meta.update(status='FAILED',error=repr(exc));save();raise
if __name__=='__main__':main()
