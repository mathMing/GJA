"""Mask unresolved numeric inputs in-memory; preserve patient roles and features."""
import time,json,warnings
import numpy as np,pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
import v2_pipeline as v
from feature_sensitivity_v15 import rules
ROOT=v.ROOT;OUT=ROOT/'results/numeric_sensitivity_v16';PREV=ROOT/'results/feature_sensitivity_v15'
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 if (OUT/'run.json').exists():raise RuntimeError('Existing run; inspect first')
 prev=json.loads((PREV/'run.json').read_text());assert prev['status']=='COMPLETED'
 source=ROOT/'results/selective_risk_v2/derived_patients_LOCAL.csv';queue=ROOT/'results/clinical_review_queue/clinical_review_queue_LOCAL.csv';cache=PREV/'score_cache_LOCAL.csv.gz'
 assert v.old.digest(source)==prev['derived_sha256'] and v.old.digest(v.SOURCE)==prev['source_sha256']
 hashes={str(p):v.old.digest(p) for p in [source,queue,cache,v.SOURCE]}
 d=pd.read_csv(source,dtype={'patient_id':str}).set_index('patient_id',drop=False);q=pd.read_csv(queue,dtype=str,keep_default_na=False);q=q[q.category.eq('VALUE_REVIEW')];assert len(q)==2 and q.status.eq('PENDING').all()
 changes=[]
 for row in q.itertuples():
  columns=['height','bmi'] if row.source.startswith('BMI,') else ['ca125'] if row.source.startswith('C17,') else []
  assert columns
  for col in columns:
   value=d.loc[row.patient_id,col];changes.append(dict(review_item_id=row.review_item_id,patient_id=row.patient_id,field=col,previous_value=value,was_already_missing=bool(pd.isna(value))));d.loc[row.patient_id,col]=np.nan
 pd.DataFrame(changes).to_csv(OUT/'in_memory_changes_LOCAL.csv',index=False)
 c=pd.read_csv(cache,dtype={'patient_id':str});configs=json.loads((PREV/'feature_manifest.json').read_text());start=time.time()
 meta={'status':'RUNNING','model_fits':0,'completed':0,'source_hashes':hashes,'code_sha256':v.old.digest(__file__),'protocol_sha256':v.old.digest(ROOT/'reports/v16_numeric_sensitivity_protocol.md'),'nonmissing_cells_masked':sum(not x['was_already_missing'] for x in changes)}
 def save():meta['elapsed_seconds']=time.time()-start;v.dump(OUT/'run.json',meta)
 save();acts=[];rows=[]
 try:
  for key,z in c.groupby(['split','seed','fold','model','configuration']):
   if time.time()-start>1800:raise TimeoutError('1800 seconds')
   base=dict(zip(['split','seed','fold','model','configuration'],key));assert len(z)==773 and z.patient_id.nunique()==773 and z.true_label.sum()==149
   assert np.array_equal(z.true_label,d.loc[z.patient_id,'true_label']) and np.array_equal(z.mask_group,d.loc[z.patient_id,'mask_group'])
   fit=z[z.role.eq('fit_oof')].copy();cal=z[z.role.eq('calibration')].copy();test=z[z.role.eq('test')].copy();x=d.loc[fit.patient_id];y=fit.true_label.to_numpy();pf=np.full(len(x),np.nan);columns=configs[base['configuration']]
   with warnings.catch_warnings(),threadpool_limits(limits=2):
    warnings.simplefilter('error',ConvergenceWarning)
    for tr,te in StratifiedKFold(3,shuffle=True,random_state=4801).split(x,y):
     model=v.build(columns,base['model']);model.fit(x.iloc[tr],y[tr]);pf[te]=model.predict_proba(x.iloc[te])[:,1];meta['model_fits']+=1
    model=v.build(columns,base['model']);model.fit(x,y);cal['score']=model.predict_proba(d.loc[cal.patient_id])[:,1];test['score']=model.predict_proba(d.loc[test.patient_id])[:,1];meta['model_fits']+=1
   fit['score']=pf;assert all(np.isfinite(f.score).all() for f in [fit,cal,test])
   for alpha in v.old.ALPHAS:
    for partition in ['G0','G2']:
     auto,threshold,bound,scope=v.old.apply(test.score.to_numpy(),test.mask_group.to_numpy(),rules(fit,cal,alpha,partition),partition);assert np.all(bound[auto]<=alpha)
     mm={**base,'policy':partition+'_FST','risk_budget':alpha}
     acts.append(pd.DataFrame({**mm,'patient_id':test.patient_id.to_numpy(),'mask_group':test.mask_group.to_numpy(),'true_label':test.true_label.to_numpy(),'action':np.where(auto,'AUTO_NEGATIVE','ABSTAIN'),'threshold':threshold,'risk_bound':bound,'certificate_scope':scope}))
     for group in ['all']+v.old.GROUPS:
      m=np.ones(len(test),bool) if group=='all' else test.mask_group.to_numpy()==group;yy=test.true_label.to_numpy();n=int(m.sum());pos=int(yy[m].sum());na=int((auto&m).sum());k=int(yy[auto&m].sum())
      rows.append({**mm,'mask_group':group,'n':n,'positive_n':pos,'auto_n':na,'auto_positive_n':k,'service':na/n if n else np.nan,'risk':k/na if na else np.nan,'miss':k/pos if pos else np.nan})
   meta['completed']+=1
   if meta['completed']%30==0:save();print('completed',meta['completed'],'/156','fits',meta['model_fits'],flush=True)
  a=pd.concat(acts,ignore_index=True);old=pd.read_csv(PREV/'actions_LOCAL.csv.gz',dtype={'patient_id':str});keys=['split','seed','fold','model','configuration','policy','risk_budget','patient_id'];assert not a.duplicated(keys).any()
  p=a.merge(old,on=keys,suffixes=('_new','_old'),validate='one_to_one');assert len(p)==len(a)==len(old) and p.true_label_new.eq(p.true_label_old).all() and p.mask_group_new.eq(p.mask_group_old).all()
  changes=p[p.action_new.ne(p.action_old)].copy();changes.to_csv(OUT/'changed_actions_LOCAL.csv.gz',index=False,compression='gzip');a.to_csv(OUT/'actions_LOCAL.csv.gz',index=False,compression='gzip')
  r=pd.DataFrame(rows);assert len(r)==4680 and meta['model_fits']==624;r.to_csv(OUT/'fold_metrics.csv',index=False)
  s=r[r.split.eq('repeated')].groupby(['seed','model','configuration','policy','risk_budget','mask_group'])[['n','positive_n','auto_n','auto_positive_n']].sum().reset_index();s['service']=s.auto_n/s.n;s['risk']=s.auto_positive_n/s.auto_n.replace(0,np.nan);s['miss']=s.auto_positive_n/s.positive_n.replace(0,np.nan);s.to_csv(OUT/'seed_metrics.csv',index=False)
  assert all(v.old.digest(path)==h for path,h in hashes.items())
  meta.update(status='COMPLETED',action_rows=len(a),paired_actions=len(p),changed_action_rows=len(changes));save();print(json.dumps(meta),flush=True)
 except Exception as exc:meta.update(status='FAILED',error=repr(exc));save();raise
if __name__=='__main__':main()
