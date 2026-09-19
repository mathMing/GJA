"""Fixed-score diagnostic: finite calibration capacity versus candidate selection."""
import json,time
import numpy as np,pandas as pd
from scipy.stats import binom
import v2_pipeline as v
from fixed_sequence_v6 import power_order,certify
from grid_order_v9 import counts

ROOT=v.ROOT;OUT=ROOT/'results/service_bottleneck_v18';PREV=ROOT/'results/temporal_sensitivity_v17'

def power(n,p,alpha,delta):
 cut=binom.ppf(delta,n,alpha).astype(int);cut-=binom.cdf(cut,n,alpha)>delta
 return binom.cdf(cut,n,p)

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 if (OUT/'run.json').exists():raise RuntimeError('Existing run: no overwrite')
 previous=json.loads((PREV/'run.json').read_text());assert previous['status']=='COMPLETED'
 assert all(v.old.digest(p)==h for p,h in previous['input_hashes'].items())
 cache=PREV/'score_cache_LOCAL.csv.gz';actions=PREV/'actions_LOCAL.csv.gz'
 inputs={str(p):v.old.digest(p) for p in [cache,actions]}
 start=time.time();meta={'status':'RUNNING','model_fits':0,'input_hashes':inputs,'protocol_sha256':v.old.digest(ROOT/'reports/v18_service_bottleneck_protocol.md'),'code_sha256':v.old.digest(__file__)}
 v.dump(OUT/'run.json',meta)
 try:
  c=pd.read_csv(cache,dtype={'patient_id':str});old=pd.read_csv(actions,dtype={'patient_id':str})
  keys=['cohort','split','seed','fold','model','configuration'];rows=[];diag=[];candidates=[];checks=0
  for key,z in c.groupby(keys):
   base=dict(zip(keys,key));assert z.patient_id.nunique()==len(z)
   ff=z[z.role.eq('fit_oof')];cc=z[z.role.eq('calibration')];tt=z[z.role.eq('test')]
   f=ff[ff.mask_group.eq('missing_both')];cal=cc[cc.mask_group.eq('missing_both')];test=tt[tt.mask_group.eq('missing_both')]
   nf,kf=counts(f.score.to_numpy(),f.true_label.to_numpy(),v.old.GRID);nc,kc=counts(cal.score.to_numpy(),cal.true_label.to_numpy(),v.old.GRID)
   nt,kt=counts(test.score.to_numpy(),test.true_label.to_numpy(),v.old.GRID)
   for alpha in v.old.ALPHAS:
    delta=.05/3;order=power_order(nf,kf,len(cc)/len(ff),alpha,delta)[0]
    rule,trace,pv=certify(nc,kc,order,alpha,delta,v.old.GRID);fst=rule['selected_index']
    point=np.flatnonzero((nc>0)&(pv<=delta));bonf=np.flatnonzero((nc>0)&(pv<=delta/len(v.old.GRID)))
    oracle=np.flatnonzero((nt>0)&(kt<=alpha*nt));first=int(order[0])
    indices={'FST':fst,'LOCKED_FIRST':first if pv[first]<=delta else -1,'BONF21':int(bonf.max()) if len(bonf) else -1,'NAIVE_SEARCH_DIAGNOSTIC':int(point.max()) if len(point) else -1,'TEST_LABEL_ORACLE_DIAGNOSTIC':int(oracle.max()) if len(oracle) else -1}
    minimum=int(np.ceil(np.log(delta)/np.log1p(-alpha)));assert binom.cdf(0,minimum,alpha)<=delta and binom.cdf(0,minimum-1,alpha)>delta
    reason='FST_ACTIVE' if fst>=0 else 'CALIBRATION_CAPACITY_INSUFFICIENT' if len(cal)<minimum else 'NO_POINTWISE_PASS' if len(point)==0 else 'ORDER_FIRST_FAILURE'
    diag.append({**base,'risk_budget':alpha,'cal_group_n':len(cal),'min_zero_error_accepted':minimum,'any_pointwise_pass':len(point)>0,'fst_active':fst>=0,'bonf_active':len(bonf)>0,'reason':reason,'first_index':first,'first_n':int(nc[first]),'first_k':int(kc[first]),'minimum_pointwise_p':float(pv.min()),'fst_tested':len(trace)})
    ranks=np.argsort(order)
    for i,t in enumerate(v.old.GRID):candidates.append({**base,'risk_budget':alpha,'candidate_index':i,'threshold':t,'fit_n':nf[i],'fit_k':kf[i],'cal_n':nc[i],'cal_k':kc[i],'p_value':pv[i],'fit_order_rank':ranks[i]})
    for method,i in indices.items():
     threshold=v.old.GRID[i] if i>=0 else -np.inf;auto=test.score.to_numpy()<=threshold;n=len(test);na=int(auto.sum());positive=int(test.true_label.sum());k=int(test.true_label.to_numpy()[auto].sum())
     valid=method in ['FST','LOCKED_FIRST','BONF21']
     rows.append({**base,'risk_budget':alpha,'method':method,'certificate_valid':valid,'threshold':threshold,'risk_bound':alpha if valid and i>=0 else np.nan,'n':n,'positive_n':positive,'auto_n':na,'auto_positive_n':k,'service':na/n if n else np.nan,'risk':k/na if na else np.nan,'miss':k/positive if positive else np.nan})
     if method=='FST':
      mask=old.policy.eq('G2_FST')&old.mask_group.eq('missing_both')&old.risk_budget.eq(alpha)
      for field,value in base.items():mask&=old[field].eq(value)
      baseline=old[mask].set_index('patient_id').loc[test.patient_id]
      assert len(baseline)==n and np.array_equal(auto,baseline.action.eq('AUTO_NEGATIVE')) and np.allclose(threshold,baseline.threshold,equal_nan=True)
      checks+=n
    assert (indices['FST']>=0)==(indices['LOCKED_FIRST']>=0)
  r=pd.DataFrame(rows);d=pd.DataFrame(diag);assert len(d)==936 and len(r)==4680
  r.to_csv(OUT/'fold_metrics.csv',index=False);d.to_csv(OUT/'bottleneck_cells.csv',index=False);pd.DataFrame(candidates).to_csv(OUT/'candidate_diagnostics.csv',index=False)
  s=r[r.split.eq('repeated')].groupby(['cohort','seed','model','configuration','risk_budget','method','certificate_valid'])[['n','positive_n','auto_n','auto_positive_n']].sum().reset_index()
  s['service']=s.auto_n/s.n;s['risk']=s.auto_positive_n/s.auto_n.replace(0,np.nan);s['miss']=s.auto_positive_n/s.positive_n.replace(0,np.nan);s.to_csv(OUT/'seed_metrics.csv',index=False)
  powers=[];ns=np.arange(1,20001)
  for alpha in v.old.ALPHAS:
   for factor in [0,.25,.5,1]:
    pp=power(ns,alpha*factor,alpha,.05/3);target=ns[pp>=.9]
    for n in [20,40,80,160,320,640,1280,2560]:
     powers.append({'risk_budget':alpha,'assumed_true_risk':alpha*factor,'accepted_calibration_n':n,'certification_probability':pp[n-1],'first_n_reaching_90pct':int(target[0]) if len(target) else np.nan,'search_max_n':20000})
  pd.DataFrame(powers).to_csv(OUT/'hypothetical_exact_power.csv',index=False)
  assert all(v.old.digest(p)==h for p,h in inputs.items()) and all(v.old.digest(p)==h for p,h in previous['input_hashes'].items())
  meta.update(status='COMPLETED',fold_rows=len(r),diagnostic_cells=len(d),baseline_patient_checks=checks,elapsed_seconds=time.time()-start);v.dump(OUT/'run.json',meta);print(json.dumps(meta))
 except Exception as exc:meta.update(status='FAILED',error=repr(exc));v.dump(OUT/'run.json',meta);raise

if __name__=='__main__':main()
