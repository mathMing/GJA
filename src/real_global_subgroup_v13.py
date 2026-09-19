"""Frozen score audit: global accepted risk versus raw missingness groups."""
import json,time
import numpy as np,pandas as pd
from scipy.stats import beta
import v2_pipeline as v
from fixed_sequence_v6 import power_order,certify
from grid_order_v9 import counts
ROOT=v.ROOT;SOURCE=ROOT/'results/fixed_sequence_v6';OUT=ROOT/'results/real_global_subgroup_v13'
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 if (OUT/'run.json').exists():raise RuntimeError('Existing run; inspect before rerun')
 start=time.time();previous=json.loads((SOURCE/'run.json').read_text());assert previous['status']=='COMPLETED'
 assert v.old.digest(v.SOURCE)==previous['source_sha256']
 derived=ROOT/'results/selective_risk_v2/derived_patients_LOCAL.csv';assert v.old.digest(derived)==previous['derived_sha256']
 d=pd.read_csv(derived,dtype={'patient_id':str}).set_index('patient_id')
 cache=SOURCE/'score_cache_LOCAL.csv.gz';c=pd.read_csv(cache,dtype={'patient_id':str})
 meta={'status':'RUNNING','model_fits':0,'completed_scorer_splits':0,'cache_sha256':v.old.digest(cache),'source_sha256':previous['source_sha256'],'derived_sha256':previous['derived_sha256'],'protocol_sha256':v.old.digest(ROOT/'reports/v13_real_global_subgroup_protocol.md'),'code_sha256':v.old.digest(__file__)}
 def save():
  meta['elapsed_seconds']=time.time()-start;v.dump(OUT/'run.json',meta)
 save();acts=[];rows=[];rule_rows=[]
 try:
  for key,z in c.groupby(['split','seed','fold','model','features']):
   base=dict(zip(['split','seed','fold','model','features'],key))
   assert len(z)==773 and z.patient_id.nunique()==773 and z.true_label.sum()==149
   assert np.array_equal(z.true_label.to_numpy(),d.loc[z.patient_id,'true_label'].to_numpy())
   assert np.array_equal(z.mask_group.to_numpy(),d.loc[z.patient_id,'mask_group'].to_numpy())
   fit=z[z.role.eq('fit_oof')];cal=z[z.role.eq('calibration')];test=z[z.role.eq('test')]
   assert len(fit)+len(cal)+len(test)==773 and np.isfinite(z.score).all()
   for alpha in v.old.ALPHAS:
    rule_sets={}
    nf,kf=counts(fit.score.to_numpy(),fit.true_label.to_numpy(),v.old.GRID);nc,kc=counts(cal.score.to_numpy(),cal.true_label.to_numpy(),v.old.GRID)
    order=power_order(nf,kf,len(cal)/len(fit),alpha,.05)[0]
    rule_sets['G0_FST']={'all':certify(nc,kc,order,alpha,.05,v.old.GRID)[0]}
    bound=v.old.cp(kc,nc,.05/len(v.old.GRID));good=np.flatnonzero((nc>0)&(bound<=alpha));idx=int(good[-1]) if len(good) else -1
    rule_sets['G0_Bonf21']={'all':dict(threshold=float(v.old.GRID[idx]) if idx>=0 else -np.inf,bound=float(bound[idx]) if idx>=0 else np.nan,active=idx>=0)}
    block={}
    for group in v.old.GROUPS[:3]:
     ff=fit[fit.mask_group.eq(group)];cc=cal[cal.mask_group.eq(group)]
     n,k=counts(ff.score.to_numpy(),ff.true_label.to_numpy(),v.old.GRID);nn,kk=counts(cc.score.to_numpy(),cc.true_label.to_numpy(),v.old.GRID)
     order=power_order(n,k,len(cal)/len(fit),alpha,.05/3)[0];block[group]=certify(nn,kk,order,alpha,.05/3,v.old.GRID)[0]
    rule_sets['G2_FST']=block
    for policy,rules in rule_sets.items():
     partition='G2' if policy=='G2_FST' else 'G0'
     auto,thresholds,bounds,cert_scope=v.old.apply(test.score.to_numpy(),test.mask_group.to_numpy(),rules,partition)
     assert np.all(bounds[auto]<=alpha)
     for scope,rule in rules.items():rule_rows.append({**base,'policy':policy,'risk_budget':alpha,'certificate_scope':scope,'threshold':rule['threshold'],'calibration_risk_bound':rule['bound']})
     acts.append(pd.DataFrame({**base,'policy':policy,'risk_budget':alpha,'patient_id':test.patient_id.to_numpy(),'mask_group':test.mask_group.to_numpy(),'true_label':test.true_label.to_numpy(),'action':np.where(auto,'AUTO_NEGATIVE','ABSTAIN'),'threshold':thresholds,'risk_bound':bounds,'certificate_scope':cert_scope}))
     for scope in ['all']+v.old.GROUPS:
      m=np.ones(len(test),bool) if scope=='all' else test.mask_group.to_numpy()==scope;y=test.true_label.to_numpy();n=int(m.sum());pos=int(y[m].sum());na=int((m&auto).sum());k=int(y[m&auto].sum())
      lo=float(beta.ppf(.025,k,na-k+1)) if k else 0.;hi=float(beta.ppf(.975,k+1,na-k)) if k<na else 1.
      rows.append({**base,'policy':policy,'risk_budget':alpha,'mask_group':scope,'n':n,'positive_n':pos,'auto_n':na,'auto_positive_n':k,'service':na/n if n else np.nan,'risk':k/na if na else np.nan,'miss':k/pos if pos else np.nan,'test_CP95_low_primary_only':lo if na and base['split']=='primary' else np.nan,'test_CP95_high_primary_only':hi if na and base['split']=='primary' else np.nan})
   meta['completed_scorer_splits']+=1
  a=pd.concat(acts,ignore_index=True);key=['split','seed','fold','model','features','risk_budget','patient_id']
  assert not a.duplicated(key+['policy']).any()
  old=pd.read_csv(SOURCE/'actions_LOCAL.csv.gz',dtype={'patient_id':str});old=old[old.policy.eq('G2_FST_OOF')]
  paired=a[a.policy.eq('G2_FST')].merge(old,on=key,suffixes=('_new','_old'),validate='one_to_one')
  assert len(paired)==len(old) and paired.action_new.eq(paired.action_old).all()
  assert np.allclose(paired.threshold_new,paired.threshold_old,equal_nan=True)
  a.to_csv(OUT/'actions_LOCAL.csv.gz',index=False,compression='gzip');r=pd.DataFrame(rows);r.to_csv(OUT/'fold_metrics.csv',index=False);pd.DataFrame(rule_rows).to_csv(OUT/'rules.csv',index=False)
  rs=r[r.split.eq('repeated')].groupby(['seed','model','features','policy','risk_budget','mask_group'])[['n','positive_n','auto_n','auto_positive_n']].sum().reset_index()
  rs['service']=rs.auto_n/rs.n;rs['risk']=rs.auto_positive_n/rs.auto_n.replace(0,np.nan);rs['miss']=rs.auto_positive_n/rs.positive_n.replace(0,np.nan);rs.to_csv(OUT/'seed_metrics.csv',index=False)
  for name,source in [('primary',r[r.split.eq('primary')]),('repeated',rs)]:
   gg=source[source.policy.str.startswith('G0')];ix=['model','features','policy','risk_budget']+(['seed'] if name=='repeated' else [])
   joint=gg[gg.mask_group.eq('all')].merge(gg[gg.mask_group.eq('missing_both')],on=ix,suffixes=('_all','_double'),validate='one_to_one')
   joint['observed_global_pass_double_exceeds']=(joint.risk_all<=joint.risk_budget)&(joint.risk_double>joint.risk_budget)
   joint.to_csv(OUT/(name+'_joint_pattern.csv'),index=False)
  assert v.old.digest(cache)==meta['cache_sha256'] and v.old.digest(derived)==meta['derived_sha256']
  meta.update(status='COMPLETED',fold_rows=len(r),action_rows=len(a),g2_patient_checks=len(paired));save();print(json.dumps(meta),flush=True)
 except Exception as exc:
  meta.update(status='FAILED',error=repr(exc));save();raise
if __name__=='__main__':main()
