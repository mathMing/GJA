# -*- coding: utf-8 -*-
"""E1 合成机制实验：双 DGP × {全局共形/阈值, 模式条件共形/阈值} × α 网格。

DGP-1（信息性缺失 informative）：缺失由不可见风险成分 b 驱动 ⇒ 缺失模式携带结局信息。
DGP-2（对照 MCAR）：缺失由与结局独立的 v 驱动 ⇒ 块结构与 DGP-1 同分布，但缺失不含结局信息。

全部系数、种子、规模与 α 网格均来自 config.json（e1 段），可用命令行覆盖。

产出：
  <输出目录>/E1_synthetic_results.xlsx
  <输出目录>/E1_条件覆盖曲线.png
  <输出目录>/E1_风险曲线.png
  <输出目录>/E1_alpha可行性预扫.png
  <中间产物目录>/e1_*.csv
运行：python src/e1_synthetic.py [--quick] [--repeats N] [--n N] [--output-dir <dir>]
"""
import os
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import beta as beta_dist
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from common import build_parser, load_config, resolve_dirs

_p = build_parser("e1_synthetic", "E1 合成机制实验（双 DGP 条件覆盖/风险对照）")
_p.add_argument("--quick", action="store_true", help="快速自检：repeats=3、n=3000（用于验证环境，非发表口径）")
_p.add_argument("--repeats", type=int, default=None, help="重复次数覆盖（默认取 config.json 的 e1.repeats）")
_p.add_argument("--n", type=int, default=None, help="每次重复样本量覆盖（默认取 config.json 的 e1.n_per_repeat）")
_args = _p.parse_args()
CFG = load_config(_args.config)
_E1 = CFG["e1"]

OUT, TEMP = resolve_dirs(CFG, _args)

# ---------------- DGP 参数（由 src/probes/e1_probe*.py 校准至真实块结构/阳性率量级） ----------------
A1, A2, C = _E1["dgp"]["A1"], _E1["dgp"]["A2"], _E1["dgp"]["C"]   # 缺失机制：logit(p_miss) = A + C * driver
BA, BB = _E1["dgp"]["BA"], _E1["dgp"]["BB"]                        # 结局：logit(Y) = B0 + BA*a + BB*b
PREV = _E1["dgp"]["PREV"]                                          # 目标总体阳性率（内部 773 例真值 19.28%）
X_COEF = _E1["dgp"]["X_visible_coef"]                              # 可见信号强度
X_SD = _E1["dgp"]["X_noise_sd"]                                    # 观测噪声强度
X_DIM = _E1["dgp"]["X_dim"]                                        # 可见协变量维度
N_BLOCK = np.array(_E1["block_proportions"])                       # 真实块比例
BLOCK_NAMES = list(_E1["block_names"])
ALPHA_COV = list(_E1["alpha_cov_grid"])
ALPHA_RISK = list(_E1["alpha_risk_grid"])
SPLIT = _E1["split_ratios"]
MIN_CAL_CONF = _E1["min_cal_block_for_conformal"]
MIN_CAL_THR = _E1["min_cal_block_for_threshold"]
MIN_TEST_REPORT = _E1["min_test_block_report"]
CP_DELTA = _E1["cp_upper_delta"]


def gen_data(n, driver_mode, rng):
    """driver_mode: 'b' → 信息性缺失（DGP-1）；'v' → 独立对照（DGP-2）"""
    a = rng.normal(size=n); b = rng.normal(size=n)
    X = X_COEF * a[:, None] + X_SD * rng.normal(size=(n, X_DIM))
    d = b if driver_mode == "b" else rng.normal(size=n)
    u1 = rng.random(n); u2 = rng.random(n); uy = rng.random(n)
    mT = u1 < expit(A1 + C * d)
    mH = u2 < expit(A2 + C * d)
    lo, hi = -5.0, 2.0
    for _ in range(50):
        mid = (lo + hi) / 2
        if expit(mid + BA * a + BB * b).mean() < PREV:
            lo = mid
        else:
            hi = mid
    b0 = (lo + hi) / 2
    y = (uy < expit(b0 + BA * a + BB * b)).astype(int)
    return X, y, mT, mH


def block_id(mT, mH):
    bid = np.full(len(mT), -1)
    bid[(~mT) & (~mH)] = 0; bid[mT & (~mH)] = 1; bid[mT & mH] = 2; bid[(~mT) & mH] = 3
    return bid


def conformal_q(s, alpha):
    nn = len(s)
    k = int(np.ceil((nn + 1) * (1 - alpha)))
    if k > nn:
        return np.inf
    return np.sort(s)[k - 1]


def cp_upper(k, n, delta=CP_DELTA):
    """单侧 Clopper-Pearson 上界"""
    if n == 0:
        return 1.0
    if k >= n:
        return 1.0
    return beta_dist.ppf(1 - delta, k + 1, n - k)


def choose_threshold(p, y, alpha):
    """在候选阈值网格上选最大判阴率 s.t. CP 上界 ≤ alpha（无解返回 -inf）"""
    order = np.argsort(p)
    ps, ys = p[order], y[order]
    cps = np.cumsum(ys)
    ns = np.arange(1, len(ps) + 1)
    ks = cps
    with np.errstate(divide="ignore", invalid="ignore"):
        ub = beta_dist.ppf(1 - CP_DELTA, ks + 1, ns - ks)
    ub = np.where(ks >= ns, 1.0, ub)
    ok = ub <= alpha
    if not ok.any():
        return -np.inf
    idx = np.where(ok)[0].max()
    return ps[idx]


def one_repeat(n, driver_mode, seed):
    rng = np.random.default_rng(seed)
    X, y, mT, mH = gen_data(n, driver_mode, rng)
    bid = block_id(mT, mH)
    Z = np.c_[X, mT.astype(float), mH.astype(float)]
    idx = rng.permutation(n)
    n_tr = int(SPLIT["train"] * n); n_ca = int(SPLIT["calibration"] * n)
    tr, ca, te = idx[:n_tr], idx[n_tr:n_tr + n_ca], idx[n_tr + n_ca:]
    clf = LogisticRegression(max_iter=2000).fit(Z[tr], y[tr])
    p = np.clip(clf.predict_proba(Z)[:, 1], 1e-6, 1 - 1e-6)
    s = np.where(y == 1, 1 - p, p)

    rec = []
    # ---- (A) 共形覆盖：全局 vs 模式条件 ----
    for alpha in ALPHA_COV:
        q = conformal_q(s[ca], alpha)
        cov = (s[te] <= q).astype(float)
        row = dict(方法="全局共形", alpha=alpha, 边际覆盖=cov.mean())
        for k in range(4):
            mk = bid[te] == k
            row[BLOCK_NAMES[k]] = cov[mk].mean() if mk.sum() > 0 else np.nan
            row[BLOCK_NAMES[k] + "_n"] = int(mk.sum())
        rec.append(row)
        cov_m = cov.copy()
        for k in range(4):
            mca = ca[bid[ca] == k]; mte = bid[te] == k
            if len(mca) < MIN_CAL_CONF:                              # 池化回退：并入父块"缺任一"
                mca = ca[np.isin(bid[ca], [1, 2, 3])]
            qk = conformal_q(s[mca], alpha)
            cov_m[mte] = (s[te][mte] <= qk).astype(float)
        row = dict(方法="模式条件共形", alpha=alpha, 边际覆盖=np.nanmean(cov_m))
        for k in range(4):
            mk = bid[te] == k
            row[BLOCK_NAMES[k]] = np.nanmean(cov_m[mk]) if mk.sum() > 0 else np.nan
            row[BLOCK_NAMES[k] + "_n"] = int(mk.sum())
        rec.append(row)
    cov_df = pd.DataFrame(rec)

    # ---- (B) 风险控制：全局阈值 vs 模式条件阈值 ----
    rrec = []
    for alpha in ALPHA_RISK:
        t_g = choose_threshold(p[ca], y[ca], alpha)
        sel = p[te] <= t_g
        row = dict(方法="全局阈值", alpha=alpha,
                   判阴率=sel.mean(), 边际风险=(y[te][sel].mean() if sel.sum() else np.nan),
                   漏诊数=int(y[te][sel].sum()), 判阴数=int(sel.sum()))
        for k in range(4):
            mk = (bid[te] == k) & sel
            row[BLOCK_NAMES[k] + "_风险"] = y[te][mk].mean() if mk.sum() >= MIN_TEST_REPORT else np.nan
            row[BLOCK_NAMES[k] + "_判阴数"] = int(mk.sum())
        rrec.append(row)
        sel_m = np.zeros(len(te), dtype=bool)
        for k in range(4):
            mca = ca[bid[ca] == k]; mte = bid[te] == k
            if len(mca) < MIN_CAL_THR:                               # 池化回退：并入父块"缺任一"
                mca = ca[np.isin(bid[ca], [1, 2, 3])]
            tk = choose_threshold(p[mca], y[mca], alpha)
            if np.isneginf(tk):
                continue
            sel_m[mte] = p[te][mte] <= tk
        row = dict(方法="模式条件阈值", alpha=alpha,
                   判阴率=sel_m.mean(), 边际风险=(y[te][sel_m].mean() if sel_m.sum() else np.nan),
                   漏诊数=int(y[te][sel_m].sum()), 判阴数=int(sel_m.sum()))
        for k in range(4):
            mk = (bid[te] == k) & sel_m
            row[BLOCK_NAMES[k] + "_风险"] = y[te][mk].mean() if mk.sum() >= MIN_TEST_REPORT else np.nan
            row[BLOCK_NAMES[k] + "_判阴数"] = int(mk.sum())
        rrec.append(row)
    risk_df = pd.DataFrame(rrec)
    return cov_df, risk_df, pd.DataFrame({"块": BLOCK_NAMES,
                                          "n": [int((bid == k).sum()) for k in range(4)],
                                          "阳性率": [y[bid == k].mean() for k in range(4)]})


# ---------------- 主循环 ----------------
REPS = _args.repeats if _args.repeats is not None else _E1["repeats"]
N = _args.n if _args.n is not None else _E1["n_per_repeat"]
if _args.quick:
    REPS, N = 3, 3000
SEED_BASE = _E1["seed_base"]
print(f"[E1] 运行规模：REPS={REPS}, N={N}, seed_base={SEED_BASE}")

cov_all, risk_all, struct_all = [], [], {}
for dgp, mode in [("DGP-1 信息性缺失", "b"), ("DGP-2 对照(缺失独立于结局)", "v")]:
    cs, rs = [], []
    for r in range(REPS):
        c, rr, st = one_repeat(N, mode, seed=SEED_BASE + r)
        c["重复"] = r; rr["重复"] = r
        cs.append(c); rs.append(rr)
    cov_all.append(pd.concat(cs).assign(DGP=dgp))
    risk_all.append(pd.concat(rs).assign(DGP=dgp))
    struct_all[dgp] = st
    print(f"[E1] {dgp} 完成 {REPS} 次重复")

cov_df = pd.concat(cov_all, ignore_index=True)
risk_df = pd.concat(risk_all, ignore_index=True)

# 汇总（跨重复均值 + 2.5/97.5 分位）
def summarise(df, cols):
    g = df.groupby(["DGP", "方法", "alpha"])
    mean = g[cols].mean()
    lo = g[cols].quantile(0.025)
    hi = g[cols].quantile(0.975)
    return mean, lo, hi

cov_cols = BLOCK_NAMES + ["边际覆盖"]
cov_mean, cov_lo, cov_hi = summarise(cov_df, cov_cols)
risk_cols = ["判阴率", "边际风险"] + [b + "_风险" for b in BLOCK_NAMES]
risk_mean, risk_lo, risk_hi = summarise(risk_df, risk_cols)

cov_mean.to_csv(os.path.join(TEMP, "e1_cov_mean.csv"), encoding="utf-8-sig")
risk_mean.to_csv(os.path.join(TEMP, "e1_risk_mean.csv"), encoding="utf-8-sig")
cov_df.to_csv(os.path.join(TEMP, "e1_cov_raw.csv"), index=False, encoding="utf-8-sig")
risk_df.to_csv(os.path.join(TEMP, "e1_risk_raw.csv"), index=False, encoding="utf-8-sig")

print("\n=== 结构校验（合成 vs 真实） ===")
for dgp, st in struct_all.items():
    print(dgp); print(st.to_string(index=False))
print("\n=== 条件覆盖（均值）===")
print(cov_mean.round(4).to_string())
print("\n=== 条件风险（均值）===")
print(risk_mean.round(4).to_string())

# ---------------- 图 1：条件覆盖曲线 ----------------
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
for ax, dgp in zip(axes, ["DGP-1 信息性缺失", "DGP-2 对照(缺失独立于结局)"]):
    for mi, (meth, style) in enumerate([("全局共形", "--o"), ("模式条件共形", "-s")]):
        sub = cov_mean.loc[(dgp, meth)]
        for k, b in enumerate(BLOCK_NAMES):
            ax.plot(sub.index, sub[b], style, alpha=0.85, ms=4,
                    label=f"{meth}·{b}" if k == 0 else None,
                    color=plt.cm.tab10(k))
        ax.plot(sub.index, sub["边际覆盖"], style, color="k", alpha=0.5, ms=4,
                label=f"{meth}·边际")
    ax.plot(sub.index, 1 - sub.index, ":", color="gray", lw=1.2, label="名义覆盖 1-α")
    ax.set_title(dgp); ax.set_xlabel("名义 α"); ax.set_ylabel("实测条件覆盖")
    ax.set_ylim(0.4, 1.02); ax.grid(alpha=0.3)
axes[0].legend(fontsize=7, ncol=2)
fig.suptitle("E1 条件覆盖曲线：全局共形 vs 模式条件共形（跨 %d 次重复均值，N=%d）" % (REPS, N))
fig.tight_layout()
fig.savefig(os.path.join(OUT, "E1_条件覆盖曲线.png"), dpi=160); plt.close(fig)

# ---------------- 图 2：风险曲线 ----------------
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
for ax, dgp in zip(axes, ["DGP-1 信息性缺失", "DGP-2 对照(缺失独立于结局)"]):
    for meth, style in [("全局阈值", "--o"), ("模式条件阈值", "-s")]:
        sub = risk_mean.loc[(dgp, meth)]
        ax.plot(sub.index, sub["边际风险"], style, color="k", alpha=0.6, ms=4, label=f"{meth}·边际风险")
        for k, b in enumerate(BLOCK_NAMES):
            ax.plot(sub.index, sub[b + "_风险"], style, ms=4, color=plt.cm.tab10(k),
                    label=f"{meth}·{b}" if k == 0 else None)
    sub = risk_mean.loc[(dgp, "全局阈值")]
    ax.plot(sub.index, sub.index, ":", color="gray", lw=1.2, label="名义风险 α")
    ax.set_title(dgp); ax.set_xlabel("风险预算 α"); ax.set_ylabel("实测条件风险 R = P(Y=1 | 判阴)")
    ax.set_ylim(0, 0.45); ax.grid(alpha=0.3)
axes[0].legend(fontsize=7, ncol=2)
fig.suptitle("E1 风险曲线：全局阈值 vs 模式条件阈值（跨 %d 次重复均值，N=%d）" % (REPS, N))
fig.tight_layout()
fig.savefig(os.path.join(OUT, "E1_风险曲线.png"), dpi=160); plt.close(fig)

# ---------------- 图 3：α 网格可行性预扫 ----------------
# (a) 理论下界 1/(n+1)；(b) 块级条件覆盖的重复间标准差 vs 块内样本量
_FEAS = _E1["alpha_feasibility"]
nb_grid = np.array(_FEAS["nb_grid"])
reps_small = _FEAS["reps"]
FEAS_ALPHA = _FEAS["alpha"]
NOMINAL = 1 - FEAS_ALPHA
res = []
for nb in nb_grid:
    n = max(_FEAS["min_n"], int(round(nb / N_BLOCK[_FEAS["probe_block"]])))   # 令"都缺"块规模 = nb
    devs = []
    for r in range(reps_small):
        rng = np.random.default_rng(_FEAS["seed_base"] + r)
        X, y, mT, mH = gen_data(n, "b", rng)
        bid = block_id(mT, mH)
        Z = np.c_[X, mT.astype(float), mH.astype(float)]
        idx = rng.permutation(n); n_tr = int(.4 * n); n_ca = int(.3 * n)
        tr, ca, te = idx[:n_tr], idx[n_tr:n_tr + n_ca], idx[n_tr + n_ca:]
        clf = LogisticRegression(max_iter=2000).fit(Z[tr], y[tr])
        p = np.clip(clf.predict_proba(Z)[:, 1], 1e-6, 1 - 1e-6)
        s = np.where(y == 1, 1 - p, p)
        for k in [_FEAS["probe_block"]]:
            mca = ca[bid[ca] == k]; mte = te[bid[te] == k]
            if len(mca) < 5 or mte.sum() < 5:
                continue
            qk = conformal_q(s[mca], FEAS_ALPHA)
            devs.append((s[mte] <= qk).mean() - NOMINAL)
    if devs:
        res.append(dict(块内样本量=nb, 实测块规模中位数=int(round(nb)),
                        偏差均值=np.mean(devs), 偏差标准差=np.std(devs, ddof=1),
                        偏差P2_5=np.percentile(devs, 2.5), 偏差P97_5=np.percentile(devs, 97.5),
                        理论下界_alpha_min=1/(nb+1), 经验下界_10over_n=10/nb))
feas = pd.DataFrame(res)
feas.to_csv(os.path.join(TEMP, "e1_alpha_feasibility.csv"), index=False, encoding="utf-8-sig")
print("\n=== α 可行性预扫 ==="); print(feas.round(4).to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
ax = axes[0]
nb = np.logspace(np.log10(15), np.log10(4000), 200)
ax.loglog(nb, 1 / (nb + 1), "-", label="共形覆盖理论下界 α=1/(n+1)")
ax.loglog(nb, 10 / nb, "--", label="经验准则 α=10/n（块内期望误覆盖≥10 例）")
for val, lab, col in [(401, "完整 401", "tab:blue"), (240, "仅缺TCT 240", "tab:orange"),
                      (113, "都缺 113", "tab:green"), (19, "仅缺HPV 19", "tab:red")]:
    ax.axvline(val, color=col, ls=":", lw=1)
    ax.annotate(lab, (val, 0.4), rotation=90, fontsize=8, color=col, ha="right")
ax.set_xlabel("块内样本量 n_b（对数）"); ax.set_ylabel("可支持的 α 下界（对数）")
ax.set_title("(a) 块级 α 可行下界"); ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
ax = axes[1]
ax.errorbar(feas["块内样本量"], feas["偏差均值"],
            yerr=[feas["偏差均值"] - feas["偏差P2_5"], feas["偏差P97_5"] - feas["偏差均值"]],
            fmt="o-", capsize=3, ms=4, label="模式条件共形·块级覆盖偏差（α=0.10，95% 区间）")
ax.axhline(0, color="gray", ls=":")
for val, lab, col in [(19, "19", "tab:red"), (113, "113", "tab:green"), (240, "240", "tab:orange"), (401, "401", "tab:blue")]:
    ax.axvline(val, color=col, ls=":", lw=1)
ax.set_xscale("log"); ax.set_xlabel("块内样本量 n_b（对数）"); ax.set_ylabel("块级覆盖偏差（实测 − 0.90）")
ax.set_title("(b) 小样本块的条件覆盖不稳定度"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
fig.suptitle("E1 α 网格可行性预扫：块内样本量与条件化粒度的硬权衡")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "E1_alpha可行性预扫.png"), dpi=160); plt.close(fig)

# ---------------- 落盘 xlsx ----------------
xlsx = os.path.join(OUT, "E1_synthetic_results.xlsx")
with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
    cov_mean.round(4).to_excel(w, sheet_name="1_条件覆盖均值")
    cov_lo.round(4).to_excel(w, sheet_name="1b_条件覆盖P2.5")
    cov_hi.round(4).to_excel(w, sheet_name="1c_条件覆盖P97.5")
    risk_mean.round(4).to_excel(w, sheet_name="2_条件风险均值")
    risk_lo.round(4).to_excel(w, sheet_name="2b_条件风险P2.5")
    risk_hi.round(4).to_excel(w, sheet_name="2c_条件风险P97.5")
    for dgp, st in struct_all.items():
        st.round(4).to_excel(w, sheet_name=f"3_块结构_{dgp[:5]}")
    feas.round(4).to_excel(w, sheet_name="4_alpha可行性预扫", index=False)
print("\n[OUT]", xlsx)
print("[DONE]")
