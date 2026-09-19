"""Paired calibration-size intervention with frozen scorers and frozen FST order."""
import time,json,hashlib,argparse
from pathlib import Path
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
import mechanism_training_v5 as sim
from fixed_sequence_v6 import power_order,certify

ROOT=sim.ROOT;OUT=ROOT/'results/calibration_information_v7';GRID=sim.risk.GRID
SIZES=[150,500,1500,5000,15000,50000];MODES=['MCAR','INFORMATIVE']

def reference_frontier(p,prob,g,alpha):
 mask=g=='missing_both';scores=p[mask];truth=prob[mask];order=np.argsort(scores)
 scores=scores[order];cs=np.r_[0,np.cumsum(truth[order])]
 fine=np.unique(np.r_[GRID,np.linspace(.001,1.,501)])
 rows=[]
 for name,grid in [('original21',GRID),('dense_diagnostic',fine)]:
  n=np.searchsorted(scores,grid,side='right');k=cs[n]
  risk=np.divide(k,n,out=np.full(len(n),np.nan),where=n>0)
  margin=np.sqrt(np.log(2*len(fine)/.05)/(2*np.maximum(n,1)))
  ub=np.minimum(1,risk+margin)
  row={'grid':name,'candidate_n':len(grid),'double_reference_n':len(scores)}
  for label,criterion in [('point',risk),('conservative',ub)]:
   good=np.flatnonzero((n>0)&(criterion<=alpha))
   idx=int(good[np.argmax(n[good])]) if len(good) else -1
   row.update({label+'_service':float(n[idx]/len(scores)) if idx>=0 else 0.,label+'_threshold':float(grid[idx]) if idx>=0 else np.nan,label+'_risk_estimate':float(risk[idx]) if idx>=0 else np.nan,label+'_risk_upper':float(ub[idx]) if idx>=0 else np.nan,label+'_reference_accepted':int(n[idx]) if idx>=0 else 0})
  rows.append(row)
 assert rows[1]['point_service']+1e-12>=rows[0]['point_service']
 assert rows[1]['conservative_service']+1e-12>=rows[0]['conservative_service']
 return rows

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
 path=OUT/('smoke' if args.smoke else 'formal');path.mkdir(parents=True,exist_ok=True)
 repetitions=1 if args.smoke else 200;sizes=[150,500] if args.smoke else SIZES
 nfit=5000;nselect=2000;nref=2000 if args.smoke else 50000;ncal=max(sizes)
 begin=time.time();meta={'status':'RUNNING','repetitions_per_mechanism':repetitions,'mechanisms':MODES,'completed_repetitions':0,'model_fits':0,'fixed_fit_n':nfit,'fixed_selection_n':nselect,'calibration_sizes':sizes,'reference_n':nref,'order_anchor_calibration_n':2500,'code_sha256':sim.risk.digest(__file__),'generator_sha256':sim.risk.digest(ROOT/'src/mechanism_training_v5.py'),'fst_sha256':sim.risk.digest(ROOT/'src/fixed_sequence_v6.py'),'protocol_sha256':sim.risk.digest(ROOT/'reports/v7_calibration_information_protocol.md'),'clinical_input':'NONE','timeout_seconds':1800}
 def save():
  meta['elapsed_seconds']=time.time()-begin;(path/'run.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
 save();chunk=[];frontiers=[];counts=[];first=True
 try:
  for mi,mode in enumerate(MODES):
   for rep in range(repetitions):
    if time.time()-begin>1800:raise TimeoutError('Predeclared runtime limit')
    streams=np.random.SeedSequence([70918,mi,rep]).spawn(4)
    train=sim.generate(np.random.default_rng(streams[0]),nfit,mode)
    selection=sim.generate(np.random.default_rng(streams[1]),nselect,mode)
    calibration=sim.generate(np.random.default_rng(streams[2]),ncal,mode)
    reference=sim.generate(np.random.default_rng(streams[3]),nref,mode)
    common={'mechanism':mode,'repeat':rep}
    for role,data in [('fit',train),('selection',selection),('calibration_pool',calibration),('reference',reference)]:
     for group in sim.risk.GROUPS:
      m=data[4]==group;counts.append({**common,'role':role,'mask_group':group,'n':int(m.sum()),'positive_n':int(data[2][m].sum())})
    for name in ['masked_lr','masked_hgb','complete_lr']:
     col=1 if name=='complete_lr' else 0
     model=sim.model(name)
     with threadpool_limits(limits=2):
      model.fit(train[col],train[2]);ps=model.predict_proba(selection[col])[:,1];pc=model.predict_proba(calibration[col])[:,1];pr=model.predict_proba(reference[col])[:,1]
     meta['model_fits']+=1
     assert np.allclose(model.named_steps['impute'].statistics_,np.nanmedian(train[col],axis=0))
     assert np.isfinite(ps).all() and np.isfinite(pc).all() and np.isfinite(pr).all()
     # All model predictions and reference evaluation are computed once, outside size loop.
     score_hash=hashlib.sha256(pc.tobytes()+pr.tobytes()).hexdigest()
     orders={}
     for alpha in sim.risk.ALPHAS:
      for scope in sim.risk.GROUPS[:3]:
       m=selection[4]==scope;take=ps[m,None]<=GRID[None,:];ns=take.sum(0);ks=(take*selection[2][m,None]).sum(0)
       orders[(alpha,scope)]=power_order(ns,ks,2500/nselect,alpha,.05/3)[0]
      for row in reference_frontier(pr,reference[3],reference[4],alpha):frontiers.append({**common,'model':name,'budget':alpha,**row})
     order_hash=hashlib.sha256(b''.join(x.tobytes() for x in orders.values())).hexdigest()
     for nc in sizes:
      cc=pc[:nc];cy=calibration[2][:nc];cg=calibration[4][:nc]
      ev=sim.risk.evidence(cc,cy,cg);ev63={scope:(nn,k,sim.risk.cp(k,nn,.05/63)) for scope,(nn,k,u) in ev.items()}
      for alpha in sim.risk.ALPHAS:
       fst={}
       for scope in sim.risk.GROUPS[:3]:
        nn,k,_=ev[scope];fst[scope]=certify(nn,k,orders[(alpha,scope)],alpha,.05/3,GRID)[0]
       for policy,rules in [('G2_fixed63',sim.risk.rules_at(ev63,alpha)),('G2_FST',fst)]:
        auto,t,b,cert_scope=sim.risk.apply(pr,reference[4],rules,'G2');assert np.all(b[auto]<=alpha)
        for scope in ['all']+sim.risk.GROUPS:
         m=sim.risk.scope_mask(reference[4],scope);selected=m&auto;na=int(selected.sum());nr=int(m.sum());rr=float(reference[3][selected].mean()) if na else np.nan
         chunk.append({**common,'model':name,'calibration_n':nc,'policy':policy,'budget':alpha,'mask_group':scope,'reference_n':nr,'reference_accepted':na,'reference_positive_in_accepted':int(reference[2][selected].sum()),'reference_service':na/nr if nr else np.nan,'reference_risk':rr,'reference_has_service':na>0,'reference_risk_exceeds_budget':bool(rr>alpha) if na else False,'actual_cal_group_n':int(sim.risk.scope_mask(cg,scope).sum()),'bound_type':'certified_budget' if policy=='G2_FST' else 'simultaneous_CP'})
     assert score_hash==hashlib.sha256(pc.tobytes()+pr.tobytes()).hexdigest()
     assert order_hash==hashlib.sha256(b''.join(x.tobytes() for x in orders.values())).hexdigest()
    meta['completed_repetitions']+=1
    if rep%10==9 or rep==repetitions-1:
     pd.DataFrame(chunk).to_csv(path/'replicate_results.csv',mode='w' if first else 'a',header=first,index=False);first=False;chunk=[];save()
     print(mode,rep+1,'/',repetitions,'fits',meta['model_fits'],'seconds',round(time.time()-begin),flush=True)
  pd.DataFrame(frontiers).to_csv(path/'reference_grid_diagnostics.csv',index=False);pd.DataFrame(counts).to_csv(path/'role_group_counts.csv',index=False)
  result=pd.read_csv(path/'replicate_results.csv');expected=len(MODES)*repetitions*3*len(sizes)*3*2*5
  key=['mechanism','repeat','model','calibration_n','policy','budget','mask_group']
  assert len(result)==expected and not result.duplicated(key).any()
  assert not ((result.reference_accepted==0)&result.reference_risk.notna()).any()
  assert meta['model_fits']==len(MODES)*repetitions*3
  meta.update(status='COMPLETED',result_rows=expected,checks=['independent role RNG streams','fixed model predictions across calibration sizes','fixed order hashes across sizes','fit-only imputation','certified budgets respected','fine grid contains original grid','unique complete outputs','undefined no-service risk'])
  save();print('COMPLETE',json.dumps(meta),flush=True)
 except Exception as exc:
  meta.update(status='FAILED',error=repr(exc));save();raise
if __name__=='__main__':main()
