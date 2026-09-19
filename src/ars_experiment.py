"""Corrected exploratory Better-1 experiments. No clinical deployment guarantee."""
import os
for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(key,"1")
import argparse, hashlib, json, platform, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import beta
from scipy.special import expit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from threadpoolctl import threadpool_limits
import sklearn
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in cast")

ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path("X:/GJA/DATA/clinical/original/cervical_master_aligned.xlsx")
GROUPS=["complete","missing_tct","missing_both","missing_hpv"]
SCOPES=["all","complete","any_missing","missing_tct","missing_both","missing_hpv"]
POLICIES=["G0","G1","G2","G2_fallback","adaptive","G2_unadjusted"]
ALPHAS=[.05,.10,.20]
GRID=np.linspace(.01,1.,21)
DELTA=.05
CLINICAL=[4,5,6,8,9,10,11,13,14,15,16,17]
PREOP=CLINICAL+[19,21,22,23,24]
CATS=[8,9,13,14,22,23,24]
RULE_NOTE="per frozen scorer and split; simultaneous over 21 fixed thresholds x 6 scopes; not across scorers/folds/seeds"

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def cp(k,n,delta):
    k,n=np.asarray(k),np.asarray(n)
    with np.errstate(invalid="ignore"):
        u=beta.ppf(1-delta,k+1,np.maximum(n-k,1))
    return np.where((n==0)|(k==n),1.,u)

def scope_mask(g,s):
    if s=="all": return np.ones(len(g),bool)
    if s=="any_missing": return g!="complete"
    return g==s

def evidence(p,y,g,corrected=True):
    """All thresholds fixed BEFORE seeing calibration features or labels."""
    per_delta=DELTA/(len(GRID)*len(SCOPES)) if corrected else DELTA
    result={}
    for s in SCOPES:
        m=scope_mask(g,s)
        selected=np.asarray(p)[m,None]<=GRID[None,:]
        n=selected.sum(axis=0)
        k=(selected*np.asarray(y)[m,None]).sum(axis=0)
        result[s]=(n,k,cp(k,n,per_delta))
    return result

def rules_at(ev,alpha):
    rules={}
    for s,(n,k,u) in ev.items():
        good=np.flatnonzero((u<=alpha)&(n>0))
        if len(good):
            i=good[-1]
            rules[s]={"threshold":float(GRID[i]),"bound":float(u[i]),"cal_n":int(n[i]),"cal_errors":int(k[i]),"active":True}
        else:
            rules[s]={"threshold":-np.inf,"bound":np.nan,"cal_n":0,"cal_errors":0,"active":False}
    return rules

def mapping(g,partition):
    if partition=="G0": return np.full(len(g),"all",object)
    if partition=="G1": return np.where(g=="complete","complete","any_missing")
    return np.where(g=="missing_hpv","unserved",g)

def apply(p,g,r,partition):
    scopes=mapping(g,partition)
    thresholds=np.full(len(g),-np.inf)
    bounds=np.full(len(g),np.nan)
    for s in np.unique(scopes):
        if s=="unserved": continue
        m=scopes==s
        thresholds[m]=r[s]["threshold"]
        bounds[m]=r[s]["bound"]
    return np.asarray(p)<=thresholds,thresholds,bounds,scopes

def choose_partition(policy,r,pcal,gcal):
    if policy in ("G0","G1","G2"): return policy
    if policy=="G2_unadjusted": return "G2"
    if policy=="G2_fallback":
        if all(r[s]["active"] for s in GROUPS[:3]): return "G2"
        if all(r[s]["active"] for s in ["complete","any_missing"]): return "G1"
        return "G0"
    options=["G2","G1","G0"]
    return max(options,key=lambda q:int(apply(pcal,gcal,r,q)[0].sum()))

def outcomes(y,auto):
    n=len(y); na=int(auto.sum()); k=int(np.asarray(y)[auto].sum()); pos=int(np.sum(y))
    return {"n":n,"positive_n":pos,"auto_n":na,"auto_positive_n":k,
            "auto_rate":na/n if n else np.nan,
            "empirical_risk":k/na if na else np.nan,
            "R_miss":k/pos if pos else np.nan,"no_service":na==0}

def metrics_rows(y,g,auto,bounds,meta):
    rows=[]
    for s in ["all"]+GROUPS:
        m=scope_mask(g,s)
        row={**meta,"mask_group":s,**outcomes(np.asarray(y)[m],auto[m])}
        ub=bounds[m & auto]
        row["applied_scope_bound_max"]=float(np.max(ub)) if len(ub) else np.nan
        row["test_risk_exceeds_budget"]=bool(row["empirical_risk"]>meta["risk_budget"]) if row["auto_n"] else np.nan
        row["bound_is_original_group_guarantee"]=meta["partition"]=="G2" and s in GROUPS[:3]
        rows.append(row)
    return rows

def load_data():
    raw=pd.read_excel(SOURCE,header=None)
    d=raw.iloc[2:].reset_index(drop=True).copy()
    d.columns=[f"C{i:02d}" for i in range(d.shape[1])]
    for c in d:
        d[c]=d[c].map(lambda v:np.nan if pd.isna(v) or str(v).strip().lower() in ("","na","n/a","nan","none") else v)
    ids=d.C02.map(lambda v:str(v).strip().removesuffix(".0"))
    y=pd.to_numeric(d.C42,errors="raise").to_numpy(int)
    assert len(d)==773 and int(y.sum())==149 and set(y)=={0,1}
    assert ids.nunique()==773 and not ids.isna().any()
    g=np.full(len(d),"complete",object)
    h,t=d.C13.isna(),d.C14.isna()
    g[t&~h]="missing_tct";g[t&h]="missing_both";g[~t&h]="missing_hpv"
    assert [int((g==s).sum()) for s in GROUPS]==[401,240,113,19]
    assert [int(y[g==s].sum()) for s in GROUPS]==[76,40,33,0]
    for j in PREOP:
        c=f"C{j:02d}"
        if j in CATS: d[c]=d[c].map(lambda v:"__MISSING__" if pd.isna(v) else str(v).strip().removesuffix(".0"))
        else:
            converted=pd.to_numeric(d[c],errors="coerce")
            bad=d[c].notna() & converted.isna()
            if bad.any():
                assert j==17 and int(bad.sum())==1, c
                print('AUDIT C17: one unreadable laboratory value treated as missing',flush=True)
            d[c]=converted
    return d,y,g,ids.to_numpy()

def model_for(name):
    fields=PREOP if name.startswith("preop") else CLINICAL
    num=[f"C{i:02d}" for i in fields if i not in CATS]
    cat=[f"C{i:02d}" for i in fields if i in CATS]
    transform=ColumnTransformer([
        ("numeric",Pipeline([("impute",SimpleImputer(strategy="median",add_indicator=True)),("scale",StandardScaler())]),num),
        ("category",OneHotEncoder(handle_unknown="ignore",sparse=False),cat)])
    est=LogisticRegression(max_iter=3000,C=1.) if name.endswith("lr") else HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=7,l2_regularization=1.,random_state=0)
    return Pipeline([("features",transform),("model",est)])

def audit(out):
    d,y,g,ids=load_data()
    pd.DataFrame([{"mask_group":s,"n":int((g==s).sum()),"positives":int(y[g==s].sum()),"prevalence":float(y[g==s].mean())} for s in GROUPS]).to_csv(out/"data_audit_summary.csv",index=False)
    pd.DataFrame([{"column":f"C{i:02d}","clinical":i in CLINICAL,"preop":i in PREOP,
                   "reason":"approved_preop" if i in PREOP else "excluded_including_BMI_text_postoperative_identifiers"} for i in range(62)]).to_csv(out/"data_leakage_audit.csv",index=False)
    assert 7 not in PREOP and 25 not in PREOP and all(i<27 for i in PREOP)
    (out/"data_flow.md").write_text("# Data flow\n773 unique aligned patients / 149 positives.\nFive outer folds; within each outer training pool 75% fit and 25% calibration.\nHPV/TCT are categorical. BMI and free text excluded.\nNo direct DICOM audit in this tabular experiment; case 1590740 retained.\nMissingness means recorded availability; reason for missingness not established.\n",encoding="utf-8")
    return d,y,g,ids

def real_run(out,seeds):
    d,y,g,ids=audit(out)
    patient_path=out/"three_way_actions.csv"
    foldrows=[];score_rows=[];rule_rows=[];splitrows=[];stress=[]
    first=True
    for seed in seeds:
        splitlist=list(StratifiedKFold(5,shuffle=True,random_state=seed).split(d,y))
        for fold,(train,test) in enumerate(splitlist):
            shuffled=np.random.default_rng(73000+seed*10+fold).permutation(train)
            fit,cal=np.split(shuffled,[int(.75*len(shuffled))])
            assert not set(fit)&set(cal) and not set(train)&set(test)
            for role,ix in [("fit",fit),("calibration",cal),("test",test)]:
                splitrows.extend({"seed":seed,"fold":fold,"patient_id":ids[i],"role":role} for i in ix)
            for name in ["clinical_lr","clinical_gbdt","preop_lr","preop_gbdt"]:
                model=model_for(name)
                with threadpool_limits(limits=2):
                    model.fit(d.iloc[fit],y[fit])
                    pc=model.predict_proba(d.iloc[cal])[:,1];pt=model.predict_proba(d.iloc[test])[:,1]
                score_rows.extend({"seed":seed,"fold":fold,"model":name,"patient_id":ids[i],"mask_group":g[i],"true_label":int(y[i]),"score":float(pt[j])} for j,i in enumerate(test))
                ev=evidence(pc,y[cal],g[cal]);un=evidence(pc,y[cal],g[cal],False)
                chunks=[]
                for alpha in ALPHAS:
                    r=rules_at(ev,alpha);ru=rules_at(un,alpha)
                    for scope,rule in r.items(): rule_rows.append({"seed":seed,"fold":fold,"model":name,"risk_budget":alpha,"scope":scope,**rule,"candidate_n":len(GRID),"family_n":len(GRID)*len(SCOPES),"delta":DELTA})
                    for policy in POLICIES:
                        rr=ru if policy=="G2_unadjusted" else r
                        part=choose_partition(policy,rr,pc,g[cal])
                        auto,t,b,scope=apply(pt,g[test],rr,part)
                        meta={"seed":seed,"fold":fold,"model":name,"policy":policy,"partition":part,"risk_budget":alpha,"corrected":policy!="G2_unadjusted","fallback":policy=="G2_fallback" and part!="G2"}
                        foldrows.extend(metrics_rows(y[test],g[test],auto,b,meta))
                        # Suggestions are bookkeeping only, not risk-optimized acquisition decisions.
                        suggestion=np.select([g[test]=="missing_tct",g[test]=="missing_hpv",g[test]=="missing_both"],["ACQUIRE_TCT","ACQUIRE_HPV","ACQUIRE_BOTH"],default="NONE")
                        suggestion=np.where(auto,"NONE",suggestion)
                        action=np.where(auto,"AUTO_NEGATIVE","ABSTAIN")
                        chunks.append(pd.DataFrame({**meta,"patient_id":ids[test],"mask_group":g[test],"score":pt,"true_label":y[test],
                            "action":action,"suggested_action":suggestion,"threshold":t,"risk_bound":b,"guarantee_scope":scope,
                            "bound_status":np.where(np.isfinite(b),"active_scope_bound","NOT_CERTIFIED")}))
                pd.concat(chunks).to_csv(patient_path,mode="w" if first else "a",header=first,index=False)
                first=False
                # Calibration-only reduction pressure test: fixed seed, no retuning.
                for fraction in [.25,.5,1.]:
                    ix=np.random.default_rng(88000+seed*10+fold).permutation(len(cal))[:max(1,int(len(cal)*fraction))]
                    rs=rules_at(evidence(pc[ix],y[cal][ix],g[cal][ix]),.10)
                    auto,_,b,_=apply(pt,g[test],rs,"G2")
                    stress.append({"seed":seed,"fold":fold,"model":name,"cal_fraction":fraction,**outcomes(y[test],auto)})
            print(f"REAL seed={seed} fold={fold} complete",flush=True)
    pd.DataFrame(foldrows).to_csv(out/"three_way_fold_metrics.csv",index=False)
    pd.DataFrame(score_rows).to_csv(out/"oof_scores.csv",index=False)
    pd.DataFrame(rule_rows).to_csv(out/"calibration_rules.csv",index=False)
    pd.DataFrame(splitrows).to_csv(out/"split_manifest.csv",index=False)
    pd.DataFrame(stress).to_csv(out/"calibration_downsampling.csv",index=False)

def generate(n,mode,seed):
    rng=np.random.default_rng(seed)
    x=rng.normal(size=(n,6));latent=rng.normal(size=n)
    truep=expit(-1.9+1.3*x[:,0]+.6*x[:,1]+.85*latent)
    y=(rng.random(n)<truep).astype(int)
    driver=np.zeros(n) if mode=="MCAR" else x[:,0] if mode=="MAR" else latent
    t=rng.random(n)<expit(-.35+1.2*driver)
    h=rng.random(n)<expit(-1.7+1.0*driver)
    g=np.full(n,"complete",object);g[t&~h]="missing_tct";g[t&h]="missing_both";g[~t&h]="missing_hpv"
    observed=x.copy();observed[t,2]=0.;observed[h,3]=0.
    return np.c_[observed,t,h],y,g,truep

def synthetic_run(out,reps,n):
    rows=[]
    for mode in ["MCAR","MAR","INFORMATIVE"]:
        for repeat in range(reps):
            x,y,g,truep=generate(n,mode,100000+repeat)
            nfit=int(.4*n);ncal=int(.3*n)
            fit=np.arange(nfit);cal=np.arange(nfit,nfit+ncal);test=np.arange(nfit+ncal,n)
            with threadpool_limits(limits=2):
                model=LogisticRegression(max_iter=500).fit(x[fit],y[fit])
                pc=model.predict_proba(x[cal])[:,1];pt=model.predict_proba(x[test])[:,1]
            ev=evidence(pc,y[cal],g[cal]);un=evidence(pc,y[cal],g[cal],False)
            for alpha in ALPHAS:
                rr=rules_at(ev,alpha);ru=rules_at(un,alpha)
                for policy in POLICIES:
                    r=ru if policy=="G2_unadjusted" else rr
                    part=choose_partition(policy,r,pc,g[cal])
                    auto,t,b,scope=apply(pt,g[test],r,part)
                    meta={"mode":mode,"repeat":repeat,"seed":100000+repeat,"policy":policy,"partition":part,"risk_budget":alpha,
                          "corrected":policy!="G2_unadjusted","fallback":policy=="G2_fallback" and part!="G2"}
                    new=metrics_rows(y[test],g[test],auto,b,meta)
                    for row in new:
                        mask=scope_mask(g[test],row["mask_group"])&auto
                        row["oracle_selected_mean"]=float(truep[test][mask].mean()) if mask.any() else np.nan
                        row["oracle_is_MC_proxy"]=True
                    rows.extend(new)
            if (repeat+1)%20==0 or repeat==0:
                print(f"SYN {mode} {repeat+1}/{reps}",flush=True)
        pd.DataFrame(rows).to_csv(out/"synthetic_raw.csv",index=False)
    raw=pd.DataFrame(rows)
    summary=raw.groupby(["mode","policy","risk_budget","mask_group"],dropna=False).agg(
        repetitions=("repeat","size"),auto_rate=("auto_rate","mean"),risk=("empirical_risk","mean"),
        risk_sd=("empirical_risk","std"),test_exceed_fraction_among_served=("test_risk_exceeds_budget","mean"),
        no_service_fraction=("no_service","mean"),fallback_fraction=("fallback","mean")).reset_index()
    summary.to_csv(out/"synthetic_mechanism_results.csv",index=False)

def tests():
    assert np.isclose(float(cp(0,59,.05)),1-.05**(1/59))
    g=np.array(GROUPS)
    assert list(mapping(g,"G1"))==["complete","any_missing","any_missing","any_missing"]
    assert mapping(g,"G2")[-1]=="unserved"
    assert cp(1,10,.0001)>cp(1,10,.05)
    ev=evidence(np.array([.2,.4]),np.array([1,1]),np.array(["complete","complete"]))
    assert not rules_at(ev,.1)["all"]["active"]
    ev=evidence(np.full(1000,.01),np.zeros(1000,int),np.full(1000,"complete",object))
    r=rules_at(ev,.05)
    a,t,b,s=apply(np.array([.01]),np.array(["complete"]),r,"G2")
    assert a[0] and b[0]<=.05 and s[0]=="complete"
    assert len(GRID)==21
    print("TESTS passed: CP, G1 coverage, rare abstention, simultaneous penalty, no-service, applied scope",flush=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--stage",choices=["test","real","synthetic","all"],default="all")
    ap.add_argument("--out",default=str(ROOT/"results"/"ars_validated_v1"))
    ap.add_argument("--quick",action="store_true")
    args=ap.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    tests()
    if args.stage=="test": return
    manifest={"started":time.strftime("%Y-%m-%dT%H:%M:%S"),"stage":args.stage,"quick":args.quick,
              "source_sha256":digest(SOURCE),"code_sha256":digest(__file__),"python":platform.python_version(),
              "sklearn":sklearn.__version__,"seeds":[0] if args.quick else list(range(5)),
              "synthetic_reps":3 if args.quick else 200,"synthetic_n":3000 if args.quick else 20000,
              "grid":GRID.tolist(),"scopes":SCOPES,"risk_budgets":ALPHAS,"confidence":.95,
              "guarantee_note":RULE_NOTE,"status":"RUNNING"}
    mp=out/("manifest_"+args.stage+".json");mp.write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    start=time.time()
    if args.stage in ["real","all"]: real_run(out,manifest["seeds"])
    if args.stage in ["synthetic","all"]: synthetic_run(out,manifest["synthetic_reps"],manifest["synthetic_n"])
    manifest.update(status="COMPLETED",elapsed_seconds=time.time()-start)
    mp.write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print("COMPLETED",round(time.time()-start,1),flush=True)

if __name__=="__main__": main()
