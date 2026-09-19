"""Better-1 v2: traceable ETL, incremental value and frozen-rule diagnostics."""
import os
for k in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"]:os.environ.setdefault(k,"1")
import sys,json,time,hashlib,datetime,re,argparse
from pathlib import Path
import numpy as np,pandas as pd
from scipy.special import expit,logit
from scipy.optimize import minimize
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
from threadpoolctl import threadpool_limits
import ars_experiment as old
ROOT=Path(__file__).resolve().parents[1];O=ROOT/"results"/"selective_risk_v2";O.mkdir(parents=True,exist_ok=True)
AUD=ROOT/"results"/"data_audit_20260917";SOURCE=old.SOURCE
FEATURES={4:"age",5:"weight",6:"height",8:"symptom",9:"menopause",10:"gravidity",11:"parity",15:"scca",16:"cea",17:"ca125",19:"us_size",21:"mri_size",22:"mri_vaginal",23:"mri_parametrial",24:"mri_nodes"}
CAT=["symptom","menopause","mri_vaginal","mri_parametrial","mri_nodes","mask_group","hpv_normalized","tct_normalized"]
BASE=["age","bmi","symptom","menopause","gravidity","parity","scca","cea","ca125"]
MRI=["us_size","mri_size","mri_vaginal","mri_parametrial","mri_nodes"]
CONFIG={"base":BASE,"base_mask":BASE+["mask_group"],"base_year":BASE+["year"],"base_year_mask":BASE+["year","mask_group"],"imaging_year":BASE+MRI+["year"],"imaging_year_mask":BASE+MRI+["year","mask_group"]}
def dump(path,v):path.write_text(json.dumps(v,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
def canon(v):
 if pd.isna(v):return ""
 return str(v).strip().removesuffix(".0")
def miss(v):return canon(v).lower() in ["","na","nan","n/a","none"]
def freeze():
 raw=pd.read_excel(SOURCE,header=None,keep_default_na=False)
 src=raw.iloc[2:].reset_index(drop=True)
 ids=src[2].map(canon);y=pd.to_numeric(src[42],errors="raise").astype(int)
 assert len(src)==773 and ids.nunique()==773 and y.sum()==149
 d=pd.DataFrame({"patient_id":ids,"center":"original","true_label":y});log=[]
 for j,name in FEATURES.items():
  value=src[j].where(~src[j].map(miss))
  if name in CAT:
   d[name]=value.map(lambda v:"__MISSING__" if pd.isna(v) else canon(v))
  else:d[name]=pd.to_numeric(value,errors="coerce")
  for i,v in enumerate(value):
   if name not in CAT and not pd.isna(v) and pd.isna(d.loc[i,name]):
    log.append(dict(patient_id=ids[i],column=f"C{j:02d}",excel_row=i+3,source_value=canon(v),derived_value=None,rule="unparseable_numeric_pending",clinical_review=True))
 d["bmi"]=d.weight/(d.height/100)**2
 d.loc[(d.height<=0)|(d.weight<=0),"bmi"]=np.nan
 for i in range(len(d)):
  log.append(dict(patient_id=ids[i],column="BMI",excel_row=i+3,source_value=canon(src.loc[i,7]),derived_value=None if pd.isna(d.bmi[i]) else float(d.bmi[i]),rule="recompute_same_patient_weight_height",clinical_review=bool(d.height[i]==112)))
 for j,prefix in [(13,"hpv"),(14,"tct")]:
  d[prefix+"_raw"]=src[j].map(canon)
  d[prefix+"_missing"]=src[j].map(miss)
  # Mechanical normalization only. Clinical category merging remains unapproved.
  d[prefix+"_normalized"]=src[j].map(lambda v:"__MISSING__" if miss(v) else canon(v).replace("，",",").replace("、",",").strip().upper())
  for i in range(len(d)):
   if d.loc[i,prefix+"_raw"]!=d.loc[i,prefix+"_normalized"]:
    log.append(dict(patient_id=ids[i],column=f"C{j:02d}",excel_row=i+3,source_value=d.loc[i,prefix+"_raw"],derived_value=d.loc[i,prefix+"_normalized"],rule="mechanical_punctuation_whitespace_missing_only",clinical_review=False))
  pd.DataFrame({"raw":d[prefix+"_raw"],"normalized":d[prefix+"_normalized"]}).value_counts().rename("n").reset_index().to_csv(O/(prefix+"_encoding_review.csv"),index=False,encoding="utf-8-sig")
 d["mask_group"]=np.select([d.hpv_missing & d.tct_missing,d.hpv_missing & ~d.tct_missing,~d.hpv_missing & d.tct_missing],["missing_both","missing_hpv","missing_tct"],default="complete")
 def date(v):
  return pd.Timestamp("1899-12-30")+pd.Timedelta(days=v) if isinstance(v,(int,float)) else pd.to_datetime(v,errors="coerce")
 surgery=src[27].map(date);d["year"]=surgery.dt.year.astype(int)
 d["year_role"]="retrospective_adjustment_proxy_not_confirmed_deployment_feature"
 for i,v in enumerate(src[27]):
  if isinstance(v,(int,float)):log.append(dict(patient_id=ids[i],column="C27",excel_row=i+3,source_value=v,derived_value=str(surgery[i].date()),rule="excel_serial_date",clinical_review=False))
 timing=pd.read_csv(AUD/"imaging_surgery_timing_LOCAL.csv",dtype={"patient_id":str})
 t=timing.drop_duplicates("patient_id").set_index("patient_id")
 d["mri_days_before_surgery"]=ids.map(t.days_before_surgery)
 d["mri_timing_status"]=np.select([d.mri_days_before_surgery<0,d.mri_days_before_surgery==0],["AFTER_SURGERY_REVIEW","SAME_DAY_REVIEW"],default="DATE_BEFORE_SURGERY")
 d["mri_se1_missing"]=ids.eq("1590740")
 d["mri_nodes_cross_source_unresolved"]=True # sensitivity model removes ALL imaging summary fields
 assert d.bmi.notna().sum()==770
 expected={"complete":401,"missing_tct":240,"missing_both":113,"missing_hpv":19}
 assert d.mask_group.value_counts().to_dict()==expected
 d.to_csv(O/"derived_patients_LOCAL.csv",index=False,encoding="utf-8-sig")
 pd.DataFrame(log).to_csv(O/"transformation_log_LOCAL.csv",index=False,encoding="utf-8-sig")
 dictionary=[]
 for j in range(62):
  dictionary.append(dict(column=f"C{j:02d}",header=canon(raw.iloc[1,j]),role="label_only" if j==42 else "candidate_preoperative" if j in FEATURES or j in [7,13,14] else "adjustment_only_not_deployment" if j==27 else "excluded",cleaning="recomputed" if j==7 else "no_semantic_guess"))
 pd.DataFrame(dictionary).to_csv(O/"field_data_dictionary.csv",index=False,encoding="utf-8-sig")
 # Deduplicate via a derived index; never delete or overwrite original MRI.
 headers=pd.read_csv(AUD/"dicom_headers_LOCAL.csv",dtype=str,keep_default_na=False)
 ix=headers[["path","center","folder_patient","folder_series","SOPInstanceUID"]].sort_values("path").copy()
 ix["duplicate_copy"]=ix.duplicated(["center","folder_patient","SOPInstanceUID"])
 ix["use_in_derived_archive"]=~ix.duplicate_copy
 ix.to_csv(O/"derived_image_manifest_LOCAL.csv",index=False)
 assert ix.duplicate_copy.sum()==74
 manifest={"source":str(SOURCE),"source_sha256":old.digest(SOURCE),"code_sha256":old.digest(__file__),"n":773,"positive_n":149,
 "label_and_preop_confirmation":{"source":"user reply in current thread","answer":"是的","date":"2026-09-17"},
 "technical_freeze":"PASS","clinical_freeze":"PARTIAL_PENDING_VALUE_AND_MRI_ADJUDICATION",
 "bmi_available":int(d.bmi.notna().sum()),"excluded_duplicate_image_copies":int(ix.duplicate_copy.sum()),
 "mri_after_surgery":int((d.mri_days_before_surgery<0).sum()),"mri_same_day":int((d.mri_days_before_surgery==0).sum()),
 "pending":["CA125 middle-dot record","clinical HPV/TCT category mapping","75 paired imaging field discrepancies","DICOM identifier linkage evidence","3 post-surgery dates and 78 same-day times"],
 "raw_data_modified":False,"derived_data_sha256":old.digest(O/"derived_patients_LOCAL.csv")}
 dump(O/"data_freeze_manifest.json",manifest)
 print("E0 complete: 773/149, BMI770, duplicate copies excluded74",flush=True)
 return d
def build(features,kind):
 assert not set(features)&{"patient_id","true_label","center","mri_days_before_surgery","mri_nodes_cross_source_unresolved"}
 nums=[f for f in features if f not in CAT];cats=[f for f in features if f in CAT]
 pre=ColumnTransformer([("num",Pipeline([("impute",SimpleImputer(strategy="median",add_indicator=True)),("scale",StandardScaler())]),nums),("cat",OneHotEncoder(handle_unknown="ignore",sparse=False),cats)])
 m=LogisticRegression(C=1.,max_iter=3000) if kind=="lr" else HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=7,l2_regularization=1.,random_state=0)
 return Pipeline([("features",pre),("model",m)])
def performance(y,p):
 if len(np.unique(y))<2:return dict(AUROC=np.nan,AUPRC=np.nan,Brier=brier_score_loss(y,p),calibration_intercept=np.nan,calibration_slope=np.nan)
 z=logit(np.clip(p,1e-6,1-1e-6))
 def loss(b):return np.mean(np.logaddexp(0,b[0]+b[1]*z)-np.asarray(y)*(b[0]+b[1]*z))
 fit=minimize(loss,[0,1],method="BFGS")
 return dict(AUROC=roc_auc_score(y,p),AUPRC=average_precision_score(y,p),Brier=brier_score_loss(y,p),calibration_intercept=float(fit.x[0]),calibration_slope=float(fit.x[1]))
def e1(d):
 y=d.true_label.to_numpy();preds=[];folds=[]
 for seed in range(5):
  for fold,(tr,te) in enumerate(StratifiedKFold(5,shuffle=True,random_state=seed).split(d,y)):
   assert not set(tr)&set(te)
   for kind in ["lr","gbdt"]:
    for name,features in CONFIG.items():
     model=build(features,kind)
     with threadpool_limits(limits=2):
      model.fit(d.iloc[tr],y[tr]);p=model.predict_proba(d.iloc[te])[:,1]
     meta=dict(seed=seed,fold=fold,model=kind,configuration=name)
     preds.append(pd.DataFrame({**meta,"patient_id":d.patient_id.iloc[te].to_numpy(),"year":d.year.iloc[te].to_numpy(),"mask_group":d.mask_group.iloc[te].to_numpy(),"true_label":y[te],"score":p}))
     folds.append({**meta,**performance(y[te],p)})
  print("E1 OOF seed",seed,"complete",flush=True)
 a=pd.concat(preds,ignore_index=True)
 assert not a.duplicated(["seed","model","configuration","patient_id"]).any()
 assert a.groupby(["seed","model","configuration"]).size().eq(773).all()
 a.to_csv(O/"e1_oof_LOCAL.csv",index=False)
 pd.DataFrame(folds).to_csv(O/"e1_fold_metrics.csv",index=False)
 metrics=[]
 for (seed,kind,name),t in a.groupby(["seed","model","configuration"]):
  metrics.append(dict(seed=seed,model=kind,configuration=name,**performance(t.true_label.to_numpy(),t.score.to_numpy())))
 met=pd.DataFrame(metrics);met.to_csv(O/"e1_seed_metrics.csv",index=False)
 paired=[]
 for no,yes in [("base","base_mask"),("base_year","base_year_mask"),("imaging_year","imaging_year_mask")]:
  b=met[met.configuration==no].merge(met[met.configuration==yes],on=["seed","model"],suffixes=("_base","_mask"))
  for _,r in b.iterrows():
   paired.append(dict(seed=r.seed,model=r.model,comparison=yes+" minus "+no,delta_AUROC=r.AUROC_mask-r.AUROC_base,delta_AUPRC=r.AUPRC_mask-r.AUPRC_base,delta_Brier=r.Brier_mask-r.Brier_base))
 pd.DataFrame(paired).to_csv(O/"e1_paired_deltas.csv",index=False)
 # Seed 0 paired patient bootstrap: conditional on fixed OOF predictions, not full pipeline CI.
 bs=[]
 for kind in ["lr","gbdt"]:
  for no,yes in [("base","base_mask"),("base_year","base_year_mask"),("imaging_year","imaging_year_mask")]:
   t=a[(a.seed==0)&(a.model==kind)]
   v=t[t.configuration==no].merge(t[t.configuration==yes],on="patient_id",suffixes=("_base","_mask"))
   diff=(v.score_mask-v.true_label_base)**2-(v.score_base-v.true_label_base)**2
   rng=np.random.default_rng(90217);boot=diff.to_numpy()[rng.integers(0,len(v),(2000,len(v)))].mean(axis=1)
   bs.append(dict(model=kind,comparison=yes+" minus "+no,delta_Brier=float(diff.mean()),conditional_bootstrap_low=float(np.quantile(boot,.025)),conditional_bootstrap_high=float(np.quantile(boot,.975)),warning="fixed OOF predictions; not training-uncertainty or independent confirmation"))
 pd.DataFrame(bs).to_csv(O/"e1_conditional_bootstrap.csv",index=False)
 # Actual forward-time evaluation; year is NOT a deployment input.
 timed=[]
 tr=np.flatnonzero(d.year.to_numpy()<=2018);te=np.flatnonzero(d.year.to_numpy()>=2019)
 for kind in ["lr","gbdt"]:
  for name in ["base","base_mask"]:
   m=build(CONFIG[name],kind)
   with threadpool_limits(limits=2):
    m.fit(d.iloc[tr],y[tr]);p=m.predict_proba(d.iloc[te])[:,1]
   timed.append(dict(split="train2015_2018_test2019_2020",model=kind,configuration=name,train_n=len(tr),test_n=len(te),**performance(y[te],p)))
 for year in sorted(d.year.unique()):
  tr=np.flatnonzero(d.year.to_numpy()!=year);te=np.flatnonzero(d.year.to_numpy()==year)
  for name in ["base","base_mask"]:
   m=build(CONFIG[name],"lr")
   with threadpool_limits(limits=2):
    m.fit(d.iloc[tr],y[tr]);p=m.predict_proba(d.iloc[te])[:,1]
   timed.append(dict(split="leave_year_out_not_forward",test_year=int(year),model="lr",configuration=name,train_n=len(tr),test_n=len(te),**performance(y[te],p)))
 pd.DataFrame(timed).to_csv(O/"e1_time_metrics.csv",index=False)
 # Descriptive regularized mask association; finite coefficient under 19/0 rare group.
 effects=[]
 for name in ["base_mask","base_year_mask","imaging_year_mask"]:
  m=build(CONFIG[name],"lr")
  with threadpool_limits(limits=2):m.fit(d,y)
  names=m.named_steps["features"].get_feature_names_out();coef=m.named_steps["model"].coef_[0]
  ref=coef[np.flatnonzero(names=="cat__mask_group_complete")[0]]
  for group in old.GROUPS[1:]:
   val=coef[np.flatnonzero(names=="cat__mask_group_"+group)[0]]-ref
   effects.append(dict(configuration=name,group=group,regularized_OR_vs_complete=float(np.exp(val)),inference="descriptive penalized association, not causal, no unpenalized CI"))
 pd.DataFrame(effects).to_csv(O/"e1_regularized_associations.csv",index=False)
 print("E1 paired assessment complete",flush=True)
def e3(d):
 y=d.true_label.to_numpy();ix=np.random.default_rng(20260917).permutation(len(d))
 fit,cal,test=np.split(ix,[len(d)//2,len(d)//2+len(d)//4])
 assert len(set(fit)|set(cal)|set(test))==len(d) and not set(fit)&set(cal) and not set(cal)&set(test)
 roles=[]
 for role,ii in [("fit",fit),("calibration",cal),("test",test)]:
  roles.extend(dict(patient_id=d.patient_id.iloc[i],role=role,mask_group=d.mask_group.iloc[i]) for i in ii)
 pd.DataFrame(roles).to_csv(O/"e3_split_LOCAL.csv",index=False)
 acts=[];metrics=[];rules=[]
 for kind in ["lr","gbdt"]:
  # Year is surgery-derived adjustment only and is deliberately absent here.
  for name,features in {"clinical":BASE,"preop":BASE+MRI+["hpv_normalized","tct_normalized"]}.items():
   m=build(features,kind)
   with threadpool_limits(limits=2):
    m.fit(d.iloc[fit],y[fit]);pc=m.predict_proba(d.iloc[cal])[:,1];pt=m.predict_proba(d.iloc[test])[:,1]
   g=d.mask_group.to_numpy();ev=old.evidence(pc,y[cal],g[cal])
   for alpha in old.ALPHAS:
    rr=old.rules_at(ev,alpha)
    for scope,r in rr.items():rules.append(dict(model=kind,features=name,budget=alpha,scope=scope,**r))
    for policy in ["G0","G1","G2","G2_fallback"]:
     part=old.choose_partition(policy,rr,pc,g[cal]);auto,t,b,sc=old.apply(pt,g[test],rr,part)
     meta=dict(model=kind,features=name,risk_budget=alpha,policy=policy,partition=part,fold="exploratory_random_holdout",corrected=True)
     assert np.all(b[auto]<=alpha)
     metrics+=old.metrics_rows(y[test],g[test],auto,b,meta)
     acts.append(pd.DataFrame({**meta,"patient_id":d.patient_id.iloc[test].to_numpy(),"center":"original","mask_group":g[test],"score":pt,"true_label":y[test],"action":np.where(auto,"AUTO_NEGATIVE","ABSTAIN"),"threshold":t,"risk_bound":b,"certificate_scope":sc,"split_role":"test","model_version":"v2"}))
 pd.concat(acts,ignore_index=True).to_csv(O/"e3_actions_LOCAL.csv",index=False)
 pd.DataFrame(metrics).to_csv(O/"e3_metrics.csv",index=False)
 pd.DataFrame(rules).to_csv(O/"e3_rules.csv",index=False)
 print("E3 exploratory random holdout complete",flush=True)
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--stage",choices=["all","freeze","e1","e3"],default="all");args=ap.parse_args()
 start=time.time();meta={"stage":args.stage,"status":"RUNNING","started":datetime.datetime.now().isoformat(),"command":"python src/v2_pipeline.py --stage "+args.stage,"cwd":str(ROOT),"timeout_seconds":1800,"code_sha256":old.digest(__file__)}
 dump(O/("run_"+args.stage+".json"),meta)
 try:
  d=freeze() if args.stage in ["all","freeze"] else pd.read_csv(O/"derived_patients_LOCAL.csv",dtype={"patient_id":str})
  if args.stage in ["all","e1"]:e1(d)
  if args.stage in ["all","e3"]:e3(d)
  assert old.digest(SOURCE)==json.loads((O/"data_freeze_manifest.json").read_text(encoding="utf-8"))["source_sha256"]
  meta.update(status="COMPLETED",elapsed_seconds=time.time()-start)
 except Exception as ex:
  meta.update(status="FAILED",error=repr(ex),elapsed_seconds=time.time()-start);raise
 finally:dump(O/("run_"+args.stage+".json"),meta)
 print("COMPLETED",round(time.time()-start,1),flush=True)
if __name__=="__main__":main()

