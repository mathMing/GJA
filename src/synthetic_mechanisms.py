import os,json
import numpy as np,pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from common import build_parser,load_config,resolve_dirs
from risk_control import fit_negative_rule
P=build_parser('synthetic_mechanisms','Better-1 synthetic mechanisms');P.add_argument('--quick',action='store_true');A=P.parse_args();C=load_config(A.config);OUT,TMP=resolve_dirs(C,A)
def gen(n,mode,rng):
 x=rng.normal(size=(n,6));latent=rng.normal(size=n);p=expit(-1.45+.8*x[:,0]+.55*latent);y=(rng.random(n)<p).astype(int)
 if mode=='MCAR': q1,q2=np.full(n,.35),np.full(n,.18)
 elif mode=='MAR': q1,q2=expit(-1.0+.8*x[:,0]),expit(-2.0+.7*x[:,1])
 else: q1,q2=expit(-1.0+1.4*latent),expit(-2.0+1.2*latent)
 m1=rng.random(n)<q1;m2=rng.random(n)<q2;g=np.full(n,'complete',object);g[m1&~m2]='missing_tct';g[m1&m2]='missing_both';g[~m1&m2]='missing_hpv';return x,y,g
def once(n,mode,seed,alpha):
 rng=np.random.default_rng(seed);x,y,g=gen(n,mode,rng);idx=rng.permutation(n);tr,ca,te=idx[:n//2],idx[n//2:3*n//4],idx[3*n//4:];z=np.c_[x,g=='missing_tct',g=='missing_hpv'];m=LogisticRegression(max_iter=1000).fit(z[tr],y[tr]);pc=m.predict_proba(z[ca])[:,1];pt=m.predict_proba(z[te])[:,1];out=[]
 for policy in ['G0','G2']:
  acts=np.zeros(len(te),bool);rules={}
  if policy=='G0':r=fit_negative_rule(pc,y[ca],alpha=alpha,delta=.05);acts=pt<=r.threshold;rules['all']=r
  else:
   for q in ['complete','missing_tct','missing_both']:
    mc=g[ca]==q;mt=g[te]==q;r=fit_negative_rule(pc[mc],y[ca][mc],alpha=alpha,delta=.05);acts[mt]=pt[mt]<=r.threshold;rules[q]=r
   mc=g[ca]!='complete';mt=g[te]=='missing_hpv';r=fit_negative_rule(pc[mc],y[ca][mc],alpha=alpha,delta=.05);acts[mt]=pt[mt]<=r.threshold;rules['missing_hpv_parent']=r
  n=int(acts.sum());k=int(y[te][acts].sum());out.append({'mode':mode,'policy':policy,'alpha':alpha,'auto_n':n,'auto_rate':n/len(te),'auto_positive_n':k,'empirical_risk':k/n if n else np.nan,'risk_bound_max':max(r.risk_bound for r in rules.values()),'violated':int((k/n if n else 0)>alpha)})
 return out
rows=[];reps=3 if A.quick else 200;n=3000 if A.quick else 20000
for mode in ['MCAR','MAR','INFORMATIVE']:
 for alpha in [.05,.10,.20]:
  for i in range(reps):rows.extend(once(n,mode,100000+i,alpha))
pd.DataFrame(rows).groupby(['mode','policy','alpha'],as_index=False).agg(auto_rate=('auto_rate','mean'),empirical_risk=('empirical_risk','mean'),risk_bound_max=('risk_bound_max','mean'),violation_rate=('violated','mean'),auto_positive_n=('auto_positive_n','mean')).to_csv(os.path.join(OUT,'synthetic_mechanism_results.csv'),index=False,encoding='utf-8-sig')
print('[DONE] synthetic',OUT)


