"""Aggregate repeated OOF experiments without treating repeats as new people."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
ROOT=Path(__file__).resolve().parents[1]
O=ROOT/"results"/"ars_validated_v1"
a=pd.read_csv(O/"three_way_actions.csv")
keys=["seed","model","policy","risk_budget","patient_id"]
assert not a.duplicated(keys).any()
assert a.groupby(keys[:-1]).size().eq(773).all()
assert ((a.action=="AUTO_NEGATIVE")== (a.score<=a.threshold)).all()
assert a.loc[a.action=="AUTO_NEGATIVE","risk_bound"].notna().all()
assert (a.loc[(a.policy=="G2")&(a.mask_group=="missing_hpv"),"action"]=="ABSTAIN").all()
assert (a.loc[(a.policy=="G1")&(a.mask_group=="missing_both"),"guarantee_scope"]=="any_missing").all()
spl=pd.read_csv(O/"split_manifest.csv")
assert not spl.duplicated(["seed","fold","patient_id"]).any()
rows=[]
for group in ["all","complete","missing_tct","missing_both","missing_hpv"]:
    d=a if group=="all" else a[a.mask_group==group]
    for k,t in d.groupby(["seed","model","policy","risk_budget"]):
        au=t.action=="AUTO_NEGATIVE"; n=int(au.sum()); err=int(t.loc[au,"true_label"].sum())
        rows.append(dict(zip(["seed","model","policy","risk_budget"],k),mask_group=group,n=len(t),positive_n=int(t.true_label.sum()),auto_n=n,auto_positive_n=err,auto_rate=n/len(t),risk=err/n if n else np.nan,R_miss=err/t.true_label.sum() if t.true_label.sum() else np.nan,abstain_rate=1-n/len(t)))
r=pd.DataFrame(rows)
r.to_csv(O/"risk_service_curve.csv",index=False)
r[r.mask_group!="all"].to_csv(O/"mask_risk_summary.csv",index=False)
sm=r.groupby(["model","policy","risk_budget","mask_group"]).agg(auto_n_mean=("auto_n","mean"),auto_n_min=("auto_n","min"),auto_n_max=("auto_n","max"),auto_rate_mean=("auto_rate","mean"),risk_mean=("risk","mean"),risk_min=("risk","min"),risk_max=("risk","max"),errors_mean=("auto_positive_n","mean")).reset_index()
sm.to_csv(O/"three_way_model_summary.csv",index=False)
scores=pd.read_csv(O/"oof_scores.csv")
m=[]
for (seed,model),t in scores.groupby(["seed","model"]):
    m.append(dict(seed=seed,model=model,AUROC=roc_auc_score(t.true_label,t.score),AUPRC=average_precision_score(t.true_label,t.score),Brier=brier_score_loss(t.true_label,t.score)))
pd.DataFrame(m).to_csv(O/"scorer_metrics.csv",index=False)
plt.rcParams.update({"font.size":10})
fig,ax=plt.subplots(figsize=(8,5))
for policy,t in sm[(sm.model=="clinical_lr")&(sm.mask_group=="all")].groupby("policy"):
    ax.plot(t.risk_budget,t.auto_rate_mean,marker="o",label=policy)
ax.set(xlabel="Risk budget",ylabel="Mean OOF automatic fraction",title="Primary clinical LR: five repeated splits")
ax.legend(fontsize=8);fig.tight_layout();fig.savefig(O/"risk_service_curve.png",dpi=180);plt.close(fig)
fig,ax=plt.subplots(figsize=(8,5))
for group,t in sm[(sm.model=="clinical_lr")&(sm.policy=="G0")].groupby("mask_group"):
    ax.plot(t.risk_budget,t.risk_mean,marker="o",label=group)
ax.plot([.05,.2],[.05,.2],"k--",label="budget")
ax.set(xlabel="Risk budget",ylabel="Observed OOF risk (undefined if no service)",title="G0 subgroup diagnostics: no subgroup certificate")
ax.legend(fontsize=8);fig.tight_layout();fig.savefig(O/"mask_risk.png",dpi=180);plt.close(fig)
fig,ax=plt.subplots(figsize=(8,5))
t=sm[(sm.model=="clinical_lr")&(sm.mask_group=="all")&(sm.risk_budget==.2)]
ax.bar(t.policy,t.auto_rate_mean,label="Auto");ax.bar(t.policy,1-t.auto_rate_mean,bottom=t.auto_rate_mean,label="Abstain")
ax.tick_params(axis="x",rotation=20);ax.set(ylabel="Fraction",title="Acquisition suggestions are separate, unvalidated bookkeeping");ax.legend()
fig.tight_layout();fig.savefig(O/"action_distribution.png",dpi=180);plt.close(fig)
fig,ax=plt.subplots(figsize=(8,5))
t=r[(r.model=="clinical_lr")&(r.mask_group=="all")&(r.risk_budget==.2)]
for p,v in t.groupby("policy"): ax.scatter(v.auto_rate,v.risk,label=p)
ax.set(xlabel="Automatic fraction",ylabel="Observed risk",title="Partition / service / risk across seeds");ax.legend(fontsize=8)
fig.tight_layout();fig.savefig(O/"partition_tradeoff.png",dpi=180);plt.close(fig)
f=pd.read_csv(O/"three_way_fold_metrics.csv")
fig,ax=plt.subplots(figsize=(8,5))
for p,t in f[(f.model=="clinical_lr")&(f.mask_group=="all")&(f.risk_budget==.2)].groupby("policy"):
    ax.plot(np.arange(len(t)),t.empirical_risk.to_numpy(),".",label=p)
ax.set(xlabel="Repeated outer fold",ylabel="Observed risk",title="No service folds have no risk estimate");ax.legend(fontsize=8)
fig.tight_layout();fig.savefig(O/"fold_variability.png",dpi=180);plt.close(fig)
a[(a.action=="AUTO_NEGATIVE")&(a.true_label==1)].to_csv(O/"false_negative_audit_LOCAL_ONLY.csv",index=False)
primary=sm[(sm.model=="clinical_lr")&(sm.mask_group=="all")]
table=primary[["policy","risk_budget","auto_n_mean","auto_n_min","auto_n_max","risk_mean","errors_mean"]].to_csv(index=False)
ms=pd.DataFrame(m).groupby("model")[["AUROC","AUPRC","Brier"]].mean().to_csv()
audit=pd.read_csv(O/"data_audit_summary.csv").to_csv(index=False)
text=f"""# Better-1 导师汇报：最小可行性实验
Material Passport: Origin Skill experiment-agent; Origin Mode run/validate; Origin Date 2026-09-16; Verification Status ANALYZED; Version ars_validated_v1.

## 1 临床问题与结论边界
自动判阴主风险 P(Y=1|AUTO_NEGATIVE)，另报 P(AUTO_NEGATIVE|Y=1)。本次研究验证流程和风险—服务率边界，不证明临床疗效。
临床逻辑回归预设为主展示分数器，其余三种是敏感性分析，不根据测试结果选冠军。
是否存在稳定的双缺失组失控，应以以下跨种子结果判断；缺少自动服务时不能宣布安全或假设成立。

## 2 数据证据
内部773人，149阳性；四格如下。双缺失组33/113=29.2%，完整组76/401=19.0%。这是风险异质性线索，不是因果或全局自动通道失控的证明。
```csv
{audit}```
术前白名单；HPV/TCT按类别编码；BMI和自由文本排除。一个乱码CA125值置缺失，没有猜测数值。1590740保留，本次未重查DICOM序列。

## 3 方法与认证
21个事先固定阈值×6个范围，Bonferroni分配0.05置信预算，用二项单侧上界认证。分区选择包含于同一同时界。
G1覆盖全部缺失模式；G2仅缺HPV始终弃权。回退切换整套分区，父块保证不可下放给子块。
每条患者记录携带实际阈值、作用范围、上界；无可用规则上界缺失，不记为0。
95%理论表述依赖独立同分布校准及冻结分数器，不能声称跨所有模型/种子同时95%。分层交叉验证只作经验稳健性，不能视为严格独立的部署认证。

## 4 实验规模
4分数器×5种子×5外层折；每折训练池75%拟合、25%校准。773患者重复使用，不能当3865个独立患者。
MCAR/MAR/信息性缺失各200重复，每次20000模拟样本。无MRI训练；3050的4GB显存不是本阶段瓶颈。
3风险预算，G0/G1/G2/整体回退/自适应及未经校正的反面对照。补查仅建议字段，主动作保留自动或弃权。

## 5 实际结果与判定
判定：改为失效边界研究，暂不扩大MRI训练投入。当前数据没有证明双缺失组分块后仍保留有意义的自动服务率。
四个分数器的校正G2在双缺失组均无自动判阴；这表示无法认证服务，不等于已经验证降低漏诊。
术前逻辑回归在20%预算下：G0平均自动89.4/773人（11.6%，种子范围0–179人）；G2平均23.2人（3.0%），双缺失组为0。
临床逻辑回归G0仅一个种子有服务，其观察风险23.6%高于20%预算；不能宣称每次测试经验风险必然小于理论预算。
合成信息性缺失在10%预算下，双缺失G0平均自动率30.0%；G2为0.61%，97%的重复无服务。主要信号是认证成本，不是稳定的分块性能收益。
以下人数是五种子均值和范围，不是新增患者总数；risk为空表示没有自动判阴，不能解释为零风险。
```csv
{table}```
分数器OOF指标五种子均值：
```csv
{ms}```
本阶段不预设阳性结论。若分块自动服务消失，应收缩为样本量及安全—服务率边界研究；仅凭缺失组患病率差异不能继续宣称分块提高临床性能。
逐组结果见mask_risk_summary.csv；合成结果见synthetic_mechanism_results.csv。
补查建议无真实收益验证；外部泛化未验证，丽水未参与本阶段结论。

## 6 导师需要拍板
- 是否认可主贡献是风险控制协议，而不是新MRI网络。
- 是否认可P(Y=1|AUTO_NEGATIVE)为第一主终点。
- 是否接受安全—服务率边界作为当前论文方向。
- 内部证据支持后是否申请独立外部数据；正式认证需另设完全随机独立校准/测试队列。

本报告没有作期刊录用或临床可用性承诺。
"""
(O/"advisor_summary.md").write_text(text,encoding="utf-8")
sections=text.split("## ")[1:]
import html
pages="".join("<section><h1>"+html.escape(s.split("\n",1)[0])+"</h1><pre>"+html.escape(s.split("\n",1)[1])+"</pre></section>" for s in sections)
(O/"advisor_brief_6pages.html").write_text('<!doctype html><meta charset="utf-8"><title>Better-1</title><style>@page{size:A4;margin:14mm}body{font-family:Arial,"Microsoft YaHei";margin:24px}section{break-after:page}pre{white-space:pre-wrap;font:12px/1.5 Arial,"Microsoft YaHei"}h1{font-size:22px}</style>'+pages,encoding="utf-8")
report="""# Experiment validation report
Material Passport: Origin Skill experiment-agent; Origin Mode run/validate; Origin Date 2026-09-16; Verification Status ANALYZED; Version ars_validated_v1.
Commands: D:/conda/envs/pytorch/python.exe src/ars_experiment.py --stage all; python src/ars_report.py.
Original outputs preserved; old conclusions invalidated by routing, encoding and scope-reporting defects.
Checks passed: split-role exclusivity; unique patient output per seed/model/policy/budget; exactly773 per output configuration; action/threshold consistency; G1 double-missing routing; G2 rare-group abstention; active automatic outputs have bounds; CP edge cases.
Initial run stopped at unreadable C17 laboratory record; explicitly set this one record missing and restarted. No result-directed tuning.
No independent reproduction claim. Source/code SHA and package versions are in manifest_all.json.
Selection correction applies to fixed thresholds and scopes per frozen scorer. It is not simultaneous across models/seeds/folds. Stratified CV is empirical robustness, not an exact IID certification exercise.
Synthetic oracle_selected_mean is a finite Monte Carlo proxy. Test empirical exceedance is not a population violation probability; no exact confidence coverage claim.
Synthetic x2/x3 are masked but are not outcome predictors: this isolates informative mask mechanism, not loss of informative tests.
No-service risks are undefined; acquisition suggestions are unvalidated and separate from actual action.
11 fallacy audit:
1 Simpson: report original groups, not just marginal risk.
2 Ecological: group prevalence cannot predict individual benefit.
3 Berkson: surgical/pathology-selected cohort restricts generalization.
4 Collider: conditioning on clinical testing can distort causal associations.
5 Base-rate neglect: report all group denominators and prevalence.
6 Regression to mean: five seeds are stability checks, not independent replication.
7 Survivorship: aligned cohort selection may omit inaccessible patients.
8 Look-elsewhere: fixed grid correction; unadjusted baseline explicitly uncertified.
9 Forking paths: fixed seeds/budgets/models; no tuning after outcome review.
10 Correlation/causation: missingness association is not effect of ordering a test.
11 Reverse causality: missingness timing and clinical pathways need source verification.
Remaining work: independently randomized holdout certification, external cohort, stronger acquisition models, provenance verification of every field's clinical timing, model calibration slope/intercept, and clinically acceptable service target.
"""
(O/"experiment_report.md").write_text(report,encoding="utf-8")
(O/"validation_checks.json").write_text(json.dumps({"status":"passed","rows":len(a),"unique_patients":int(a.patient_id.nunique()),"seeds":int(a.seed.nunique()),"limitations":"not independent reproducibility verification"},indent=2),encoding="utf-8")
print(table)
print("REPORT AND INTEGRITY CHECKS COMPLETE")
