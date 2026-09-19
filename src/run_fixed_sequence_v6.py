"""Paired real-data evaluation of a predeclared LTT fixed sequence."""
import time,json,argparse
import numpy as np,pandas as pd
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
import v2_pipeline as v
from fixed_sequence_v6 import power_order,certify,monte_carlo_validation

O=v.ROOT/'results/fixed_sequence_v6';O.mkdir(parents=True,exist_ok=True)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--validate-only',action='store_true');args=ap.parse_args()
 started=time.time();validation=monte_carlo_validation(O);print('VALIDATION',json.dumps(validation),flush=True)
 if args.validate_only:return
 source_hash=v.old.digest(v.SOURCE);derived_hash=v.old.digest(v.O/'derived_patients_LOCAL.csv')
 meta={'status':'RUNNING','completed_splits':0,'model_fits':0,'source_sha256':source_hash,'derived_sha256':derived_hash,'core_sha256':v.old.digest(v.ROOT/'src/fixed_sequence_v6.py'),'runner_sha256':v.old.digest(__file__),'protocol_sha256':v.old.digest(v.ROOT/'reports/v6_fixed_sequence_protocol.md')}
 v.dump(O/'run.json',meta)
 d=pd.read_csv(v.O/'derived_patients_LOCAL.csv',dtype={'patient_id':str});y=d.true_label.to_numpy();g=d.mask_group.to_numpy()
 ix=np.random.default_rng(20260917).permutation(len(d));fit,cal,test=np.split(ix,[len(d)//2,len(d)//2+len(d)//4]);splits=[('primary',-1,-1,fit,cal,test)]
 for seed in range(5):
  for fold,(train,test) in enumerate(StratifiedKFold(5,shuffle=True,random_state=seed).split(d,y)):
   fit,cal=np.split(np.random.default_rng(73000+seed*10+fold).permutation(train),[int(.75*len(train))]);splits.append(('repeated',seed,fold,fit,cal,test))
 rows=[];acts=[];traces=[];diagnostics=[];cache=[]
 for split,seed,fold,fit,cal,test in splits:
  assert not set(fit)&set(cal) and not set(fit)&set(test) and not set(cal)&set(test)
  for kind in ['lr','gbdt']:
   for name,features in {'clinical':v.BASE,'preop':v.BASE+v.MRI+['hpv_normalized','tct_normalized']}.items():
    meta0=dict(split=split,seed=seed,fold=fold,model=kind,features=name)
    pf=np.full(len(fit),np.nan)
    with threadpool_limits(limits=2):
     for tr,te in StratifiedKFold(3,shuffle=True,random_state=4801).split(fit,y[fit]):
      m=v.build(features,kind);m.fit(d.iloc[fit[tr]],y[fit[tr]]);pf[te]=m.predict_proba(d.iloc[fit[te]])[:,1]
     m=v.build(features,kind);m.fit(d.iloc[fit],y[fit]);pc=m.predict_proba(d.iloc[cal])[:,1];pt=m.predict_proba(d.iloc[test])[:,1]
    assert np.isfinite(pf).all();meta['model_fits']+=4
    for role,ii,p in [('fit_oof',fit,pf),('calibration',cal,pc),('test',test,pt)]:
     cache.append(pd.DataFrame({**meta0,'role':role,'patient_id':d.patient_id.iloc[ii].to_numpy(),'mask_group':g[ii],'true_label':y[ii],'score':p}))
    ev=v.old.evidence(pc,y[cal],g[cal]);ev63={s:(n,k,v.old.cp(k,n,.05/63)) for s,(n,k,u) in ev.items()}
    for alpha in v.old.ALPHAS:
     fst={};locked={}
     for scope in v.old.GROUPS[:3]:
      sm=g[fit]==scope;sel=pf[sm,None]<=v.old.GRID[None,:];nf=sel.sum(0);kf=(sel*y[fit][sm,None]).sum(0)
      order,power=power_order(nf,kf,len(cal)/len(fit),alpha,.05/3)
      nc,kc,_=ev[scope];rule,trace,pval=certify(nc,kc,order,alpha,.05/3,v.old.GRID);fst[scope]=rule
      diagnostic_cp=v.old.cp(kc,nc,.05/3)
      assert np.array_equal((diagnostic_cp<=alpha)&(nc>0),pval<=.05/3)
      for t in trace:
       idx=t['candidate_index'];traces.append({**meta0,'budget':alpha,'scope':scope,**t,'threshold':v.old.GRID[idx],'fit_estimated_power':power[idx],'cal_n':int(nc[idx]),'cal_positive':int(kc[idx]),'pointwise_CP_diagnostic_only':diagnostic_cp[idx]})
      diagnostics.append({**meta0,'budget':alpha,'scope':scope,'active':rule['active'],'selected_threshold':rule['threshold'],'certified_risk_bound':rule['bound'],'tested_candidates':len(trace),'passed_candidates':int(sum(t['passed'] for t in trace)),'any_pointwise_feasible_DIAGNOSTIC_ONLY':bool((pval<=.05/3).any()),'feasible_thresholds_DIAGNOSTIC_ONLY':int((pval<=.05/3).sum()),'first_threshold':float(v.old.GRID[order[0]]),'first_cal_n':int(nc[order[0]]),'first_cal_positive':int(kc[order[0]])})
      eligible=np.flatnonzero((nf>=10)&(kf<=alpha*.5*nf));j=eligible[-1] if len(eligible) else -1
      ok=j>=0 and nc[j]>0 and diagnostic_cp[j]<=alpha
      locked[scope]={'threshold':float(v.old.GRID[j]) if ok else -np.inf,'bound':float(diagnostic_cp[j]) if ok else np.nan,'active':bool(ok)}
     for policy,rr in [('G2_search126',v.old.rules_at(ev,alpha)),('G2_fixed63',v.old.rules_at(ev63,alpha)),('G2_locked3',locked),('G2_FST_OOF',fst)]:
      auto,t,b,scope=v.old.apply(pt,g[test],rr,'G2');assert np.all(b[auto]<=alpha)
      md={**meta0,'risk_budget':alpha,'policy':policy,'partition':'G2'}
      rows+=v.old.metrics_rows(y[test],g[test],auto,b,md)
      acts.append(pd.DataFrame({**md,'patient_id':d.patient_id.iloc[test].to_numpy(),'mask_group':g[test],'true_label':y[test],'score':pt,'action':np.where(auto,'AUTO_NEGATIVE','ABSTAIN'),'threshold':t,'risk_bound':b,'certificate_scope':scope,'bound_type':'certified_budget' if policy=='G2_FST_OOF' else 'CP_bound'}))
  meta['completed_splits']+=1;meta['elapsed_seconds']=time.time()-started;v.dump(O/'run.json',meta);print(split,seed,fold,'complete',flush=True)
 r=pd.DataFrame(rows);r.to_csv(O/'fold_metrics.csv',index=False)
 a=pd.concat(acts,ignore_index=True);assert not a.duplicated(['split','seed','model','features','policy','risk_budget','patient_id']).any();a.to_csv(O/'actions_LOCAL.csv.gz',index=False,compression='gzip')
 c=pd.concat(cache,ignore_index=True);assert not c.duplicated(['split','seed','fold','model','features','patient_id']).any();c.to_csv(O/'score_cache_LOCAL.csv.gz',index=False,compression='gzip')
 pd.DataFrame(traces).to_csv(O/'sequence_trace.csv',index=False);pd.DataFrame(diagnostics).to_csv(O/'scope_diagnostics.csv',index=False)
 keys=['seed','model','features','policy','risk_budget','mask_group'];rs=r[r.split=='repeated'].groupby(keys)[['n','positive_n','auto_n','auto_positive_n']].sum().reset_index();rs['auto_rate']=rs.auto_n/rs.n;rs['risk']=rs.auto_positive_n/rs.auto_n.replace(0,np.nan);rs.to_csv(O/'seed_metrics.csv',index=False)
 old=pd.read_csv(v.ROOT/'results/locked_real_v4/seed_metrics.csv');paired=rs[rs.policy!='G2_FST_OOF'].merge(old,on=keys,suffixes=('_v6','_v4'))
 assert len(paired)==900 and (paired.auto_n_v6==paired.auto_n_v4).all() and (paired.auto_positive_n_v6==paired.auto_positive_n_v4).all()
 assert v.old.digest(v.SOURCE)==source_hash and v.old.digest(v.O/'derived_patients_LOCAL.csv')==derived_hash
 meta.update(status='COMPLETED',elapsed_seconds=time.time()-started,paired_baseline_checks=len(paired),patient_action_rows=len(a),checks=['disjoint roles','unique actions','fit-only order','CP/binomial duality','same V4 baseline counts','unchanged input hashes'])
 v.dump(O/'run.json',meta);print('COMPLETE',json.dumps(meta),flush=True)
if __name__=='__main__':main()
