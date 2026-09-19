from dataclasses import dataclass
import numpy as np
from scipy.stats import beta

def cp_upper(k,n,delta=.05):
    if n<=0 or k>=n:return 1.0
    return float(beta.ppf(1-delta,k+1,n-k))
@dataclass
class Rule:
    threshold:float; risk_bound:float; n_cal:int; positives_cal:int; delta:float; fallback:bool=False
def fit_negative_rule(scores,labels,alpha=.05,delta=.05,candidates=101):
    scores,labels=np.asarray(scores,float),np.asarray(labels,int)
    if len(scores)==0:return Rule(-np.inf,1.,0,0,delta,True)
    grid=np.unique(np.quantile(scores,np.linspace(0,1,min(candidates,len(scores)))));grid=np.r_[-np.inf,grid];pdlt=delta/max(1,len(grid));ok=[]
    for t in grid:
        m=scores<=t;n=int(m.sum());k=int(labels[m].sum());ub=cp_upper(k,n,pdlt)
        if ub<=alpha:ok.append((float(t),ub,n,k))
    if not ok:return Rule(-np.inf,1.,0,0,pdlt,True)
    t,ub,n,k=max(ok,key=lambda x:x[0]);return Rule(t,ub,n,k,pdlt,False)
def summarize_actions(actions,labels,groups):
    a=np.asarray(actions,object);y=np.asarray(labels,int);g=np.asarray(groups,object);out=[]
    for z in np.unique(g):
        for q in ('AUTO_NEGATIVE','ABSTAIN','ACQUIRE_TCT','ACQUIRE_HPV','ACQUIRE_BOTH'):
            m=(g==z)&(a==q)
            if m.any():out.append({'mask_group':str(z),'action':q,'n':int(m.sum()),'positives':int(y[m].sum()),'rate':float(y[m].mean())})
    return out

