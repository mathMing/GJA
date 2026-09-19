"""LTT fixed-sequence baseline with training-only order and budget certificates."""
import numpy as np
from scipy.stats import binom,beta

def power_order(n_fit,k_fit,cal_fit_ratio,alpha,delta):
    n_fit=np.asarray(n_fit);k_fit=np.asarray(k_fit)
    nn=np.floor(n_fit*cal_fit_ratio).astype(int)
    p_est=(k_fit+.5)/(n_fit+1)
    allowable=binom.ppf(delta,nn,alpha).astype(int)
    allowable-=binom.cdf(allowable,nn,alpha)>delta
    power=binom.cdf(allowable,nn,p_est)
    power=np.where(nn>0,power,0.)
    # Primary key descending estimated power; secondary descending threshold index.
    order=np.lexsort((-np.arange(len(power)),-power))
    return order,power

def accepted_prefix(pvalues,order,delta):
    passed=[];trace=[]
    for rank,index in enumerate(order):
        index=int(index);p=float(pvalues[index]);ok=p<=delta
        trace.append({'rank':rank,'candidate_index':index,'p_value':p,'passed':bool(ok)})
        if not ok:break
        passed.append(index)
    return passed,trace

def certify(n_cal,k_cal,order,alpha,delta,grid):
    n_cal=np.asarray(n_cal);k_cal=np.asarray(k_cal)
    pv=np.where(n_cal>0,binom.cdf(k_cal,n_cal,alpha),1.)
    passed,trace=accepted_prefix(pv,order,delta)
    if not passed:
        return {'threshold':-np.inf,'bound':np.nan,'active':False,'selected_index':-1},trace,pv
    index=max(passed)
    return {'threshold':float(grid[index]),'bound':float(alpha),'active':True,'selected_index':int(index)},trace,pv

def monte_carlo_validation(path):
    import pandas as pd,json
    assert accepted_prefix([.001,.5,.001],[0,1,2],.05)[0]==[0]
    assert accepted_prefix([.5,.001],[0,1],.05)[0]==[]
    assert accepted_prefix([.01,.02,.03],[2,0,1],.05)[0]==[2,0,1]
    assert not certify(np.zeros(21),np.zeros(21),np.arange(21),.2,.05/3,np.arange(21))[0]['active']
    rng=np.random.default_rng(610918);reps=5000;rows=[];B=21;delta=.05/3
    for mode in ['all_unsafe','nonmonotone','all_safe']:
      for alpha in [.05,.1,.2]:
       for nc in [40,200,1000]:
        if mode=='all_unsafe':binrisk=np.full(B,alpha+.005)
        elif mode=='all_safe':binrisk=np.full(B,alpha*.2)
        else:binrisk=np.r_[np.full(4,.01),np.full(5,.7),np.full(7,.01),np.full(5,.7)]
        probabilities=np.stack([1-binrisk,binrisk])/B
        true_risk=np.cumsum(binrisk)/np.arange(1,B+1)
        violation=np.zeros(reps,bool);service=np.zeros(reps,bool)
        for scope in range(3):
         fit=rng.multinomial(nc*3,probabilities.ravel(),size=reps).reshape(reps,2,B).cumsum(-1)
         cal=rng.multinomial(nc,probabilities.ravel(),size=reps).reshape(reps,2,B).cumsum(-1)
         fn=fit.sum(1);fk=fit[:,1];cn=cal.sum(1);ck=cal[:,1]
         pv=np.where(cn>0,binom.cdf(ck,cn,alpha),1.)
         # Vectorized exact same ordering rule as power_order.
         nn=np.floor(fn/3).astype(int);p_est=(fk+.5)/(fn+1)
         cut=binom.ppf(delta,nn,alpha).astype(int);cut-=binom.cdf(cut,nn,alpha)>delta
         power=np.where(nn>0,binom.cdf(cut,nn,p_est),0.)
         order=np.lexsort((-np.broadcast_to(np.arange(B),power.shape),-power),axis=1)
         valid_prefix=np.cumprod(np.take_along_axis(pv,order,axis=1)<=delta,axis=1).astype(bool)
         bad=np.take_along_axis(np.broadcast_to(true_risk>alpha,pv.shape),order,axis=1)
         violation|=(valid_prefix&bad).any(1);service|=valid_prefix.any(1)
         if scope==0:
          check,_=power_order(fn[0],fk[0],1/3,alpha,delta);assert np.array_equal(check,order[0])
          passed,trace=accepted_prefix(pv[0],order[0],delta)
          assert passed==order[0][valid_prefix[0]].tolist()
        k=int(violation.sum());lo=float(beta.ppf(.025,k,reps-k+1)) if k else 0.;hi=float(beta.ppf(.975,k+1,reps-k)) if k<reps else 1.
        rows.append(dict(population=mode,budget=alpha,calibration_per_scope=nc,repetitions=reps,any_unsafe_certified=k,violation_rate=k/reps,mc95_lower=lo,mc95_upper=hi,service_fraction=service.mean()))
    df=pd.DataFrame(rows);df.to_csv(path/'fst_exact_population_validation.csv',index=False)
    assert not (df.mc95_lower>.05).any(),'Empirical false certification requires investigation'
    summary={'status':'PASS','independent_calibration_datasets':len(df)*reps,'simulation_cells':len(df),'repetitions':reps,'maximum_empirical_violation':float(df.violation_rate.max()),'checks':['stop at first failure','empty calibration abstention','all-pass sequence','scalar/vectorized independent-training order agreement','exact population false certification audit']}
    (path/'validation.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    return summary
