import json,os
import numpy as np,pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
from common import build_parser,load_config,require_file,resolve_dirs,resolve_source
from risk_control import fit_negative_rule
A=build_parser('three_way_experiment','Better-1 three-way risk control').parse_args();C=load_config(A.config);OUT,TMP=resolve_dirs(C,A);SRC=require_file(resolve_source(C,A))
def miss(s):
 t=s.astype('object').where(s.notna(),np.nan);return t.isna()|t.astype(str).str.strip().isin(['','NA','na','N/A','nan','None','无'])
def load(path):
 r=pd.read_excel(path,header=None);d=r.iloc[2:].reset_index(drop=True);d.columns=[f'C{i:02d}' for i in range(d.shape[1])]
 y=pd.to_numeric(d.C42,errors='coerce').astype(int).to_numpy();h,t=miss(d.C13),miss(d.C14);g=np.full(len(d),'complete',object);g[t&~h]='missing_tct';g[t&h]='missing_both';g[~t&h]='missing_hpv'
 idx=[4,5,6,7,8,9,10,11,13,14,15,16,17,19,21,22,23,24,25];v=[]
 for j in idx:v += [pd.to_numeric(d[f'C{j:02d}'],errors='coerce').to_numpy(float),miss(d[f'C{j:02d}']).astype(float).to_numpy()]
 ids=d.C02.astype(str).to_numpy() if 'C02' in d else np.arange(len(d)).astype(str)
 return np.column_stack(v),y,g,ids
def run(seed=0):
 X,y,g,ids=load(SRC);budgets=[.05,.10,.20];mods={'clinical_lr':(list(range(16)),LogisticRegression(max_iter=3000,class_weight='balanced')),'clinical_gbdt':(list(range(16)),HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=15,random_state=seed)),'preop_lr':(list(range(X.shape[1])),LogisticRegression(max_iter=3000,class_weight='balanced')),'preop_gbdt':(list(range(X.shape[1])),HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=15,random_state=seed))}
 summary=[];patients=[];folds=[];sk=StratifiedKFold(5,shuffle=True,random_state=seed)
 for name,(cols,est) in mods.items():
  o=np.zeros(len(y))
  for fold,(tr,te) in enumerate(sk.split(X,y)):
   rng=np.random.default_rng(seed+fold);p=rng.permutation(tr);n=int(.75*len(p));fit,cal=p[:n],p[n:];m=make_pipeline(SimpleImputer(strategy='median'),clone(est));m.fit(X[fit][:,cols],y[fit]);pc=m.predict_proba(X[cal][:,cols])[:,1];pt=m.predict_proba(X[te][:,cols])[:,1];o[te]=pt
   for alpha in budgets:
    rg=fit_negative_rule(pc,y[cal],alpha=alpha,delta=.05);sel=pt<=rg.threshold
    folds.append({'model':name,'fold':fold,'policy':'G0','risk_budget':alpha,'threshold':rg.threshold,'risk_bound':rg.risk_bound,'auto_n':int(sel.sum()),'auto_positive_n':int(y[te][sel].sum()),'auto_rate':float(sel.mean()),'empirical_risk':float(y[te][sel].mean()) if sel.any() else np.nan})
    for policy in ['G0','G1','G2']:
     acts=np.full(len(te),'ABSTAIN',object);bounds=[]
     if policy=='G0':r=rg;acts=np.where(pt<=r.threshold,'AUTO_NEGATIVE','ABSTAIN');bounds=[r.risk_bound]
     else:
      groups=['complete','missing_tct'] if policy=='G1' else ['complete','missing_tct','missing_both']
      for z in groups:
       mc=g[cal]==z;mt=g[te]==z;r=fit_negative_rule(pc[mc],y[cal][mc],alpha=alpha,delta=.05);bounds.append(r.risk_bound);acts[mt]=np.where(pt[mt]<=r.threshold,'AUTO_NEGATIVE','ABSTAIN')
      mt=g[te]=='missing_hpv';parent=g[cal]!='complete';r=fit_negative_rule(pc[parent],y[cal][parent],alpha=alpha,delta=.05);bounds.append(r.risk_bound);acts[mt]=np.where(pt[mt]<=r.threshold,'AUTO_NEGATIVE','ABSTAIN')
     acq=np.where(g[te]=='missing_tct','ACQUIRE_TCT',np.where(g[te]=='missing_hpv','ACQUIRE_HPV',np.where(g[te]=='missing_both','ACQUIRE_BOTH','NONE')));mask=(acts=='ABSTAIN')&(acq!='NONE');acts[mask]=acq[mask]
     for j,ii in enumerate(te):patients.append({'patient_id':ids[ii],'fold':fold,'model':name,'mask_group':g[ii],'score':float(pt[j]),'policy':policy,'action':acts[j],'threshold':float(rg.threshold),'risk_bound':float(max(bounds)),'risk_budget':alpha,'true_label':int(y[ii])})
  summary=pd.DataFrame(patients);summary.to_csv(os.path.join(OUT,'three_way_actions.csv'),index=False,encoding='utf-8-sig');pd.DataFrame(folds).to_csv(os.path.join(OUT,'three_way_fold_metrics.csv'),index=False,encoding='utf-8-sig')
 rows=[]
 for (model,policy,alpha),q in summary.groupby(['model','policy','risk_budget']):
  a=q.action=='AUTO_NEGATIVE';pos=int(q.loc[a,'true_label'].sum());n=int(a.sum());rows.append({'model':model,'policy':policy,'risk_budget':alpha,'auto_n':n,'auto_rate':n/len(q),'auto_positive_n':pos,'empirical_risk':pos/n if n else np.nan,'false_negative_rate':pos/int(q.true_label.sum()) if q.true_label.sum() else np.nan,'abstain_rate':float((q.action=='ABSTAIN').mean()),'acquire_rate':float(q.action.str.startswith('ACQUIRE').mean())})
 pd.DataFrame(rows).to_csv(os.path.join(OUT,'three_way_model_summary.csv'),index=False,encoding='utf-8-sig')
 with open(os.path.join(OUT,'three_way_manifest.json'),'w',encoding='utf-8') as f:json.dump({'n':len(y),'positives':int(y.sum()),'risk':'P(Y=1|AUTO_NEGATIVE)','budgets':budgets,'policies':['G0','G1','G2'],'source':SRC,'seed':seed},f,ensure_ascii=False,indent=2)
 print('[DONE] three_way',OUT)
if __name__=='__main__':run()

