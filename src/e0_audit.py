# -*- coding: utf-8 -*-
"""E0 数据审计复算：四格缺失模式交叉表、术前缺失指示 CV-AUC 复现、泄漏字段审计定稿候选。

输入（只读）：config.json 的 paths.clinical_source（默认 X:/GJA/DATA/clinical/original/cervical_master_aligned.xlsx，773 例真源）
输出：<输出目录>/E0_data_audit_results.xlsx
      <中间产物目录>/e0_four_cells.csv、e0_stats.json
运行：python src/e0_audit.py [--source <xlsx>] [--output-dir <dir>] [--temp-dir <dir>]
"""
import os
import json
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from common import build_parser, load_config, require_file, resolve_dirs, resolve_source

_args = build_parser("e0_audit", "E0 数据审计复算").parse_args()
CFG = load_config(_args.config)
_CFG = CFG["e0"]

OUT, TEMP = resolve_dirs(CFG, _args)
SRC = require_file(resolve_source(CFG, _args),
                   "请用 --source 指定临床主表路径，或修改 config.json 中 paths.clinical_source。")

raw = pd.read_excel(SRC, header=None)
cols = [str(x) for x in raw.iloc[1].tolist()]
df = raw.iloc[2:].reset_index(drop=True).copy()
df.columns = [f"C{j:02d}" for j in range(len(cols))]

# ---------- 0. 基础口径 ----------
n = len(df)
y = pd.to_numeric(df[CFG["e0"]["label_col"]], errors="coerce").astype(int)
assert y.notna().all()
print(f"[0] 有效例数 n={n}，阳性 {int(y.sum())}（{y.mean()*100:.2f}%）")

def is_missing(s):
    """D10 口径：NaN、NA 字符串、纯空白 均记缺失"""
    if s.dtype.kind in "if":
        return s.isna()
    t = s.astype("object").where(s.notna(), np.nan)
    strmiss = t.astype(str).str.strip().isin(["NA", "na", "N/A", "", "nan", "None", "无"])
    return t.isna() | strmiss

miss_hpv = is_missing(df[CFG["e0"]["block_key_cols"]["hpv"]])
miss_tct = is_missing(df[CFG["e0"]["block_key_cols"]["tct"]])
print(f"[0] HPV 缺失 {int(miss_hpv.sum())}，TCT 缺失 {int(miss_tct.sum())}")

# ---------- 1. 四格缺失模式交叉表复算 ----------
cells = {
    "完整(TCT+HPV均有值)": (~miss_tct) & (~miss_hpv),
    "仅缺TCT": (miss_tct) & (~miss_hpv),
    "都缺(TCT+HPV均缺)": (miss_tct) & (miss_hpv),
    "仅缺HPV": (~miss_tct) & (miss_hpv),
}

def wilson(k, m, z=1.959963985):
    if m == 0:
        return (np.nan, np.nan)
    p = k / m
    d = 1 + z * z / m
    c = (p + z * z / (2 * m)) / d
    h = z * np.sqrt(p * (1 - p) / m + z * z / (4 * m * m)) / d
    return (max(0.0, c - h), min(1.0, c + h))

rows = []
for name, mask in cells.items():
    m = int(mask.sum())
    k = int(y[mask].sum())
    lo, hi = wilson(k, m)
    rows.append(dict(块=name, n=m, 阳性=k, 阳性率=k / m if m else np.nan,
                     Wilson95CI_low=lo, Wilson95CI_high=hi,
                     占比=m / n))
four = pd.DataFrame(rows)
par_mask = miss_tct | miss_hpv
k = int(y[par_mask].sum()); m = int(par_mask.sum())
lo, hi = wilson(k, m)
four.loc[len(four)] = dict(块="父块(缺任一)", n=m, 阳性=k, 阳性率=k / m,
                           Wilson95CI_low=lo, Wilson95CI_high=hi, 占比=m / n)
print("\n[1] 四格缺失模式交叉表")
print(four.to_string(index=False))

# Fisher 精确检验：完整 vs 都缺
t_f = [[int(y[cells["完整(TCT+HPV均有值)"]].sum()), int(cells["完整(TCT+HPV均有值)"].sum()) - int(y[cells["完整(TCT+HPV均有值)"]].sum())],
       [int(y[cells["都缺(TCT+HPV均缺)"]].sum()), int(cells["都缺(TCT+HPV均缺)"].sum()) - int(y[cells["都缺(TCT+HPV均缺)"]].sum())]]
odds, pval = fisher_exact(t_f, alternative="two-sided")
print(f"[1] Fisher 完整 vs 都缺：table={t_f}, OR={odds:.4f}, p={pval:.6f}")

t_f2 = [[int(y[cells["完整(TCT+HPV均有值)"]].sum()), int(cells["完整(TCT+HPV均有值)"].sum()) - int(y[cells["完整(TCT+HPV均有值)"]].sum())],
        [int(y[par_mask].sum()), int(par_mask.sum()) - int(y[par_mask].sum())]]
odds2, pval2 = fisher_exact(t_f2, alternative="two-sided")
print(f"[1] Fisher 完整 vs 缺任一：OR={odds2:.4f}, p={pval2:.6f}")

both = cells["都缺(TCT+HPV均缺)"]
t_f3 = [[int(y[both].sum()), int(both.sum()) - int(y[both].sum())],
        [int(y[~both].sum()), int((~both).sum()) - int(y[~both].sum())]]
odds3, pval3 = fisher_exact(t_f3, alternative="two-sided")
print(f"[1] Fisher 都缺 vs 非都缺（冻结口径 p≈0.0063）：OR={odds3:.4f}, p={pval3:.6f}")

# ---------- 2. 术前缺失指示 CV-AUC 复现 ----------
PRE_IDX = [4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 19, 21, 22, 23, 24, 25]  # 19 列术前白名单
POST_IDX = [34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45]                      # 12 列术后病理
SURG_IDX = [27, 28, 29, 30, 31, 32]
TREAT_IDX = [47, 48, 49, 50]
FU_IDX = [52, 53, 54, 55, 56, 57, 58, 59, 60, 61]
ID_IDX = [0, 1, 2, 3]
EMPTY_IDX = [18, 33, 46, 51]
NEAREMPTY_IDX = [12, 20, 26]

mis_all = pd.DataFrame({f"C{j:02d}": is_missing(df[f"C{j:02d}"]).astype(int) for j in range(len(cols))})

def cv_auc(feat_idx, seeds=tuple(_CFG["cv"]["seeds"]), n_splits=_CFG["cv"]["n_splits"]):
    X = mis_all[[f"C{j:02d}" for j in feat_idx]].values
    out = {}
    for s in seeds:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=CFG["e0"]["cv"]["shuffle"], random_state=s)
        oof = np.zeros(len(y))
        for tr, te in skf.split(X, y.values):
            if X[tr].std() == 0:
                oof[te] = 0.5
                continue
            clf = LogisticRegression(max_iter=_CFG["cv"]["logreg_max_iter"], C=_CFG["cv"]["logreg_C"])
            clf.fit(X[tr], y.values[tr])
            oof[te] = clf.predict_proba(X[te])[:, 1]
        out[s] = roc_auc_score(y.values, oof)
    vals = np.array(list(out.values()))
    return out, vals.mean(), vals.std(ddof=1)

configs = {
    "A_术前19列缺失指示": PRE_IDX,
    "B_仅HPV+TCT缺失指示": [13, 14],
    "C_术前+术后病理缺失指示(泄漏负例)": PRE_IDX + POST_IDX,
    "C2_仅转移淋巴结部位缺失指示": [45],
    "C3_仅除淋巴结阳性外临床病理分期缺失指示": [34],
    "D_全表62列缺失指示": list(range(len(cols))),
}
auc_rows = []
per_seed = {}
for name, idx in configs.items():
    out, mu, sd = cv_auc(idx)
    per_seed[name] = out
    auc_rows.append(dict(配置=name, 特征数=len(idx), CV_AUC_mean=mu, CV_AUC_sd=sd,
                         seed42=out.get(42, np.nan),
                         各seed=";".join(f"{k}:{v:.4f}" for k, v in sorted(out.items()))))
    print(f"[2] {name}: mean={mu:.4f} sd={sd:.4f} 各seed=" + ", ".join(f"{k}:{v:.4f}" for k, v in sorted(out.items())))
auc_tab = pd.DataFrame(auc_rows)

# ---------- 3. 泄漏字段专项核验 ----------
leak_rows = []

def col_profile(j, tag):
    s = df[f"C{j:02d}"]
    m = is_missing(s)
    k = int(y[~m].sum())
    mrows = int((~m).sum())
    leak_rows.append(dict(列号=f"C{j:02d}", 字段=cols[j], 模块=tag, 有值数=mrows,
                          缺失数=int(m.sum()), 有值者阳性数=k,
                          阳性率_有值=k / mrows if mrows else np.nan, 缺失者阳性率=float(y[m].mean()) if m.sum() else np.nan))
    return mrows, k

for j, tag in [(45, "术后病理"), (34, "术后病理"), (44, "术后病理"), (42, "术后病理(标签)"),
               (24, "术前影像"), (25, "术前影像"), (19, "术前影像"), (21, "术前影像"),
               (22, "术前影像"), (23, "术前影像"), (13, "术前化验"), (14, "术前化验"),
               (7, "一般情况"), (4, "一般情况"), (9, "一般情况")]:
    col_profile(j, tag)
lprof = pd.DataFrame(leak_rows)
print("\n[3] 泄漏专项核验")
print(lprof.to_string(index=False))

# "转移淋巴结部位" 有值时是否全为阳性
m45 = ~is_missing(df["C45"])
print(f"[3] 转移淋巴结部位有值 {int(m45.sum())} 例，其中阳性 {int(y[m45].sum())} 例")
m34 = ~is_missing(df["C34"])
print(f"[3] 除淋巴结阳性外临床病理分期 有值 {int(m34.sum())} 例，其中阳性 {int(y[m34].sum())} 例")
print(f"[3] 二者与标签的一致性：C34有值↔阳性={bool(((m34.values.astype(bool))==(y.values==1)).all())}")
m45b = m45.values.astype(bool)
print(f"[3] C45有值↔阳性={bool((m45b==(y.values==1)).all())}；C45有值⊂阳性={bool((m45b & (y.values==0)).sum()==0)}")

# 术前"详细描述"文本中的淋巴结/转移字样
txt = df["C25"].astype(str)
hits = txt.str.contains("淋巴结|转移|肿大|肿大淋巴结", na=False)
print(f"[3] 术前'详细描述'含淋巴结/转移/肿大字样 {int(hits.sum())} 例：{txt[hits].tolist()[:6]}")

# ---------- 4. 字段分级定稿候选 ----------
grade_map = []
for j in range(len(cols)):
    if j in PRE_IDX:
        mod, g, act = "术前", "术前白名单", "可用（需缺失指示）"
    elif j in ID_IDX:
        mod, g, act = "标识", "标识/非特征", "排除（不可作特征）"
    elif j in EMPTY_IDX:
        mod, g, act = "空结构", "空列", "排除（整列无值）"
    elif j in NEAREMPTY_IDX:
        mod, g, act = "结构/噪声", "近空列", "排除（仅空白占位）"
    elif j in SURG_IDX:
        mod, g, act = "手术", "术后黑名单", "排除（术后信息）"
    elif j in POST_IDX:
        mod, g, act = "术后病理", "术后黑名单", "排除（术后信息/泄漏）"
    elif j in TREAT_IDX:
        mod, g, act = "辅助治疗", "术后黑名单", "排除（术后信息）"
    elif j in FU_IDX:
        mod, g, act = "随访", "术后黑名单", "排除（结局后信息）"
    else:
        mod, g, act = "未知", "待定", "待定"
    grade_map.append(dict(列号=f"C{j:02d}", 字段=cols[j], 模块=mod, 分级=g, 处置=act))
grade = pd.DataFrame(grade_map)
assert (grade["分级"] == "待定").sum() == 0, grade[grade["分级"] == "待定"]
print("\n[4] 字段分级计数：")
print(grade["分级"].value_counts().to_string())
grade.loc[grade["列号"] == "C42", "处置"] = "排除（标签列 Y，不可作特征）"
grade.loc[grade["列号"] == "C45", "处置"] = "排除（实测泄漏：有值 144 例全阳性，混入后标记 CV-AUC 0.97）"
grade.loc[grade["列号"] == "C34", "处置"] = "排除（实测泄漏：有值例数与阳性例数完全一致）"
grade.loc[grade["列号"] == "C07", "处置"] = "可用（待重建；缺失 770/773，不作有效特征）"
grade.loc[grade["列号"] == "C25", "处置"] = "慎用（自由文本，含淋巴结/转移字样须脱敏或整列排除）"

# ---------- 输出 ----------
xlsx = os.path.join(OUT, "E0_data_audit_results.xlsx")
with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
    four.to_excel(w, sheet_name="1_四格复算", index=False)
    pd.DataFrame([dict(检验="完整 vs 都缺", OR=odds, p=pval),
                  dict(检验="完整 vs 缺任一", OR=odds2, p=pval2),
                  dict(检验="都缺 vs 非都缺（冻结口径）", OR=odds3, p=pval3)]).to_excel(w, sheet_name="1b_Fisher", index=False)
    auc_tab.to_excel(w, sheet_name="2_缺失指示CVAUC", index=False)
    lprof.to_excel(w, sheet_name="3_泄漏专项核验", index=False)
    grade.to_excel(w, sheet_name="4_字段分级定稿候选", index=False)
    mis_all.assign(标签=y.values).to_excel(w, sheet_name="5_缺失指示矩阵", index=False)
print("\n[OUT] " + xlsx)

four.to_csv(os.path.join(TEMP, "e0_four_cells.csv"), index=False, encoding="utf-8-sig")
json.dump({"fisher_complete_vs_all_missing": [float(odds), float(pval)],
           "fisher_complete_vs_any_missing": [float(odds2), float(pval2)],
           "fisher_both_missing_vs_rest": [float(odds3), float(pval3)]},
          open(os.path.join(TEMP, "e0_stats.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("[DONE]")
