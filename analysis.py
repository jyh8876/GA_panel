"""Panel analysis: mis-selling, lapse (1 - persistency) and planner settlement by insurer.

Non-life panel: FSS (persistency, settlement) x KNIA (mis-selling, claim denial), 2021-2025.
Life panel:     FSS only (persistency, settlement), 2021-2025.
Writes LaTeX table fragments to tables/, figures to fig/, numbers to results.json.
"""
import json, os, warnings
import numpy as np
import pandas as pd
from scipy import stats
from linearmodels.panel import PanelOLS, RandomEffects, PooledOLS
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(f"{HERE}/tables", exist_ok=True)
os.makedirs(f"{HERE}/fig", exist_ok=True)
R = {}

# ------------------------------------------------------------------ data
p = pd.read_csv(f"{HERE}/panel.csv")
for c in ("settle", "p13", "p25"):
    p.loc[p[c] == 0, c] = np.nan            # 0.0 = not applicable / no new exclusive planners
p["lapse13"] = 100 - p.p13
p["lapse25"] = 100 - p.p25
p["ms_bp"] = p.ms * 100                     # mis-selling in basis points (0.01% = 1bp)

# balanced company x year grid so that lags are true one-year lags
grid = []
for (sec, co), g in p.groupby(["sector", "company"]):
    yrs = range(int(g.year.min()), int(g.year.max()) + 1)
    grid.append(g.set_index("year").reindex(yrs).assign(sector=sec, company=co).rename_axis("year").reset_index())
p = pd.concat(grid).sort_values(["sector", "company", "year"])
g = p.groupby(["sector", "company"])
p["ms_bp_l1"] = g.ms_bp.shift(1)
p["ms_bp_l2"] = g.ms_bp.shift(2)
p["settle_l1"] = g.settle.shift(1)

NONFACE = ["AIG", "에이스(라이나손보)", "AXA"]      # TM/direct-led insurers
EXCL = ["신한EZ", "카카오페이", "카디프손보"]          # digital / too few obs
nl = p[(p.sector == "nonlife") & p.year.between(2021, 2025) & ~p.company.isin(EXCL)].copy()
lf = p[(p.sector == "life") & p.year.between(2021, 2025)].copy()

ENG = {"에이스(라이나손보)": "ACE/Lina", "예별(MG)": "Yebyeol (ex-MG)", "삼성화재": "Samsung F&M",
       "현대해상": "Hyundai M&F", "DB손보": "DB Ins.", "KB손보": "KB Ins.", "메리츠": "Meritz F&M",
       "한화손보": "Hanwha GI", "롯데": "Lotte Ins.", "흥국화재": "Heungkuk F&M", "농협손보": "NH Ins.",
       "하나손보": "Hana Ins.", "AIG": "AIG Korea", "AXA": "AXA Korea"}

# ------------------------------------------------------------------ descriptives
def describe(df, cols, label):
    rows = []
    for c in cols:
        d = df[["company", c]].dropna()
        m = d.groupby("company")[c].transform("mean")
        rows.append(dict(var=c, N=len(d), firms=d.company.nunique(), mean=d[c].mean(), sd=d[c].std(),
                         sd_between=d.groupby("company")[c].mean().std(), sd_within=(d[c] - m).std()))
    out = pd.DataFrame(rows)
    R[f"desc_{label}"] = out.round(3).to_dict("records")
    return out

LAB = {"lapse13": "13th-month lapse (\\%)", "lapse25": "25th-month lapse (\\%)", "settle": "Planner settlement (\\%)",
       "ms_bp": "Mis-selling rate (bp)", "deny_rate": "Claim denial rate (\\%)", "post_claim_cxl_rate": "Post-claim cancellation (\\%)"}
d_nl = describe(nl, ["lapse13", "lapse25", "settle", "ms_bp", "deny_rate", "post_claim_cxl_rate"], "nl")
d_lf = describe(lf, ["lapse13", "lapse25", "settle"], "lf")

def desc_tex(d, fn):
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"Variable & $N$ & Firms & Mean & SD & SD between & SD within \\", r"\midrule"]
    for r in d.itertuples():
        lines.append(f"{LAB[r.var]} & {r.N} & {r.firms} & {r.mean:.2f} & {r.sd:.2f} & {r.sd_between:.2f} & {r.sd_within:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(f"{HERE}/tables/{fn}", "w").write("\n".join(lines))
desc_tex(d_nl, "desc_nl.tex"); desc_tex(d_lf, "desc_lf.tex")

# ------------------------------------------------------------------ correlation decomposition
def corr_decomp(df, x, y):
    d = df[["company", "year", x, y]].dropna()
    pooled = stats.pearsonr(d[x], d[y])
    bm = d.groupby("company")[[x, y]].mean()
    between = stats.pearsonr(bm[x], bm[y]) if len(bm) > 2 else (np.nan, np.nan)
    w = d.copy()
    for c in (x, y):                                   # two-way within transformation
        w[c] = w[c] - w.groupby("company")[c].transform("mean") - w.groupby("year")[c].transform("mean") + w[c].mean()
    within = stats.pearsonr(w[x], w[y])
    return dict(x=x, y=y, N=len(d), firms=len(bm),
                r_pooled=pooled[0], p_pooled=pooled[1], r_between=between[0], p_between=between[1],
                r_within=within[0], p_within=within[1])

pairs_nl = [("ms_bp_l1", "lapse13"), ("ms_bp_l2", "lapse25"), ("settle", "lapse13"), ("settle_l1", "lapse25"),
            ("settle", "ms_bp"), ("deny_rate", "lapse13")]
pairs_lf = [("settle", "lapse13"), ("settle_l1", "lapse25")]
cd = [dict(corr_decomp(nl, *pr), sample="Non-life") for pr in pairs_nl] + \
     [dict(corr_decomp(lf, *pr), sample="Life") for pr in pairs_lf]
R["corr"] = pd.DataFrame(cd).round(4).to_dict("records")
CL = {"ms_bp_l1": "Mis-selling$_{t-1}$", "ms_bp_l2": "Mis-selling$_{t-2}$", "settle": "Settlement$_t$",
      "settle_l1": "Settlement$_{t-1}$", "deny_rate": "Claim denial$_t$", "lapse13": "Lapse13$_t$",
      "lapse25": "Lapse25$_t$", "ms_bp": "Mis-selling$_t$"}

def star(pv):
    return "^{***}" if pv < 0.01 else "^{**}" if pv < 0.05 else "^{*}" if pv < 0.10 else ""

lines = [r"\begin{tabular}{llrrrr}", r"\toprule",
         r"Sample & Pair ($x$, $y$) & $N$ (firms) & Pooled $r$ & Between $r$ & Within $r$ \\", r"\midrule"]
for c in cd:
    lines.append(f"{c['sample']} & {CL[c['x']]}, {CL[c['y']]} & {c['N']} ({c['firms']}) & "
                 f"${c['r_pooled']:.2f}{star(c['p_pooled'])}$ & ${c['r_between']:.2f}{star(c['p_between'])}$ & "
                 f"${c['r_within']:.2f}{star(c['p_within'])}$ \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
open(f"{HERE}/tables/corr.tex", "w").write("\n".join(lines))

# ------------------------------------------------------------------ regressions
def fit(df, y, xs, kind, weights=None):
    d = df[["company", "year", y] + xs + ([weights] if weights else [])].dropna().set_index(["company", "year"])
    X = d[xs].copy()
    if kind in ("pooled", "re"):
        X = X.assign(const=1.0)
    w = d[weights] if weights else None
    if kind == "pooled":
        m = PooledOLS(d[y], X, weights=w).fit(cov_type="clustered", cluster_entity=True)
    elif kind == "re":
        m = RandomEffects(d[y], X, weights=w).fit(cov_type="clustered", cluster_entity=True)
    elif kind == "fe":
        m = PanelOLS(d[y], X, entity_effects=True, weights=w).fit(cov_type="clustered", cluster_entity=True)
    elif kind == "twfe":
        m = PanelOLS(d[y], X, entity_effects=True, time_effects=True, weights=w).fit(cov_type="clustered", cluster_entity=True)
    elif kind == "te":                                 # year effects only (pooled cross-section by year)
        m = PanelOLS(d[y], X.assign(const=1.0), time_effects=True, weights=w).fit(cov_type="clustered", cluster_entity=True)
    return m

def summ(m, xs):
    out = {x: dict(b=float(m.params[x]), se=float(m.std_errors[x]), p=float(m.pvalues[x])) for x in xs}
    rsq = getattr(m, "rsquared_within", np.nan)
    return dict(coef=out, N=int(m.nobs), firms=int(m.entity_info["total"]),
                r2_within=float(rsq) if rsq is not None else np.nan, r2=float(m.rsquared))

def hausman(df, y, xs):
    d = df[["company", "year", y] + xs].dropna().set_index(["company", "year"])
    fe = PanelOLS(d[y], d[xs], entity_effects=True).fit()
    re = RandomEffects(d[y], d[xs].assign(const=1.0)).fit()
    b = fe.params[xs] - re.params[xs]
    V = fe.cov.loc[xs, xs] - re.cov.loc[xs, xs]
    try:
        H = float(b.T @ np.linalg.pinv(V.values) @ b)
    except Exception:
        H = np.nan
    return dict(H=H, df=len(xs), p=float(1 - stats.chi2.cdf(H, len(xs))))

SPECS = {
    "A": ("lapse13", ["ms_bp_l1", "settle"]),
    "B": ("lapse25", ["ms_bp_l2", "settle_l1"]),
    "C": ("ms_bp", ["settle"]),
}
models = {}
for key, (y, xs) in SPECS.items():
    for kind in ("pooled", "te", "re", "fe", "twfe"):
        models[(key, kind)] = summ(fit(nl, y, xs, kind), xs)
    R[f"hausman_{key}"] = hausman(nl, y, xs)

# robustness on spec A
rob = {
    "excl_nonface": summ(fit(nl[~nl.company.isin(NONFACE)], "lapse13", ["ms_bp_l1", "settle"], "twfe"), ["ms_bp_l1", "settle"]),
    "contemp_ms": summ(fit(nl, "lapse13", ["ms_bp", "settle"], "twfe"), ["ms_bp", "settle"]),
    "claims_weighted": summ(fit(nl.assign(wt=nl.groupby("company").claims.transform("mean")), "lapse13", ["ms_bp_l1", "settle"], "twfe", weights="wt"), ["ms_bp_l1", "settle"]),
    "with_denial": summ(fit(nl, "lapse13", ["ms_bp_l1", "settle", "deny_rate"], "twfe"), ["ms_bp_l1", "settle", "deny_rate"]),
    "pooled_excl_nonface": summ(fit(nl[~nl.company.isin(NONFACE)], "lapse13", ["ms_bp_l1", "settle"], "te"), ["ms_bp_l1", "settle"]),
}
# life
life = {}
for key, (y, xs) in {"L1": ("lapse13", ["settle"]), "L2": ("lapse25", ["settle_l1"])}.items():
    for kind in ("pooled", "te", "re", "fe", "twfe"):
        life[(key, kind)] = summ(fit(lf, y, xs, kind), xs)
    R[f"hausman_{key}"] = hausman(lf, y, xs)

# ------------------------------------------------------------------ wild cluster bootstrap (Webb weights, null imposed)
def wcr_boot(df, y, xs, test, B=9999, seed=7):
    """Two-way FE via firm and year dummies; WCR bootstrap p-value for H0: beta_test = 0 (Webb 6-point weights)."""
    d = df[["company", "year", y] + xs].dropna()
    D = pd.get_dummies(d[["company", "year"]].astype(str), drop_first=True).astype(float)
    X = np.column_stack([np.ones(len(d)), d[xs].values, D.values])
    k = 1 + xs.index(test)
    Y = d[y].values
    cl = pd.factorize(d.company)[0]; G = cl.max() + 1

    def tstat(Yv):
        XtX_inv = np.linalg.pinv(X.T @ X)
        b = XtX_inv @ X.T @ Yv
        u = Yv - X @ b
        meat = np.zeros((X.shape[1], X.shape[1]))
        for gg in range(G):
            s_ = X[cl == gg].T @ u[cl == gg]
            meat += np.outer(s_, s_)
        n, kk = X.shape
        V = (G / (G - 1)) * ((n - 1) / (n - kk)) * XtX_inv @ meat @ XtX_inv
        return b[k] / np.sqrt(V[k, k]), b[k]

    t0, b0 = tstat(Y)
    Xr = np.delete(X, k, axis=1)
    br = np.linalg.pinv(Xr) @ Y
    fitted, ur = Xr @ br, Y - Xr @ br
    rng = np.random.default_rng(seed)
    webb = np.array([-np.sqrt(1.5), -1, -np.sqrt(0.5), np.sqrt(0.5), 1, np.sqrt(1.5)])
    tb = np.empty(B)
    for i in range(B):
        w = rng.choice(webb, G)[cl]
        tb[i] = tstat(fitted + w * ur)[0]
    return dict(b=float(b0), t=float(t0), p_boot=float(np.mean(np.abs(tb) >= abs(t0))), G=int(G), B=B)

R["wcr"] = {
    "A_ms_bp_l1": wcr_boot(nl, "lapse13", ["ms_bp_l1", "settle"], "ms_bp_l1"),
    "A_settle": wcr_boot(nl, "lapse13", ["ms_bp_l1", "settle"], "settle"),
    "B_settle_l1": wcr_boot(nl, "lapse25", ["ms_bp_l2", "settle_l1"], "settle_l1"),
    "C_settle": wcr_boot(nl, "ms_bp", ["settle"], "settle"),
    "Aexcl_settle": wcr_boot(nl[~nl.company.isin(NONFACE)], "lapse13", ["ms_bp_l1", "settle"], "settle"),
    "L1_settle": wcr_boot(lf, "lapse13", ["settle"], "settle"),
}

R["models"] = {f"{k[0]}_{k[1]}": v for k, v in models.items()}
R["robust"] = rob
R["life"] = {f"{k[0]}_{k[1]}": v for k, v in life.items()}

# ------------------------------------------------------------------ regression tables
KIND = {"pooled": "Pooled OLS", "te": "Year FE", "re": "Random eff.", "fe": "Firm FE", "twfe": "Two-way FE"}
VL = {"ms_bp_l1": "Mis-selling$_{t-1}$ (bp)", "ms_bp_l2": "Mis-selling$_{t-2}$ (bp)", "ms_bp": "Mis-selling$_t$ (bp)",
      "settle": "Settlement$_t$ (\\%)", "settle_l1": "Settlement$_{t-1}$ (\\%)", "deny_rate": "Claim denial$_t$ (\\%)"}

def cell(c):
    return f"${c['b']:.3f}{star(c['p'])}$", f"$({c['se']:.3f})$"

def reg_table(res, cols, xs, fn, extra=None):
    lines = [r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule",
             " & " + " & ".join(h for h, _ in cols) + r" \\", r"\midrule"]
    for x in xs:
        b, s = [], []
        for _, k in cols:
            c = res[k]["coef"].get(x)
            bb, ss = cell(c) if c else ("", "")
            b.append(bb); s.append(ss)
        lines.append(VL[x] + " & " + " & ".join(b) + r" \\")
        lines.append(" & " + " & ".join(s) + r" \\")
    lines.append(r"\midrule")
    lines.append("$N$ (firms) & " + " & ".join(f"{res[k]['N']} ({res[k]['firms']})" for _, k in cols) + r" \\")
    lines.append("$R^2$ (within for FE) & " + " & ".join(f"{res[k]['r2']:.2f}" for _, k in cols) + r" \\")
    if extra:
        lines.append(extra)
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(f"{HERE}/tables/{fn}", "w").write("\n".join(lines))

kinds = ["pooled", "te", "re", "fe", "twfe"]
for key, (y, xs) in SPECS.items():
    h = R[f"hausman_{key}"]
    reg_table(models, [(KIND[k], (key, k)) for k in kinds], xs, f"reg_{key}.tex",
              extra=f"Hausman FE vs RE & \\multicolumn{{{len(kinds)}}}{{c}}{{$\\chi^2({h['df']})={h['H']:.2f}$, $p={h['p']:.3f}$}} \\\\")
rob_cols = [("Baseline", ("A", "twfe")), ("Excl. TM/direct", "excl_nonface"), ("Contemp. MS", "contemp_ms"),
            ("Claims-wtd", "claims_weighted"), ("+ Denial", "with_denial")]
robres = {**{("A", "twfe"): models[("A", "twfe")]}, **rob}
reg_table(robres, rob_cols, ["ms_bp_l1", "ms_bp", "settle", "deny_rate"], "reg_robust.tex")
lcols = [(f"{KIND[k]}", (key, k)) for key in ("L1", "L2") for k in ("pooled", "re", "fe", "twfe")]
reg_table(life, lcols, ["settle", "settle_l1"], "reg_life.tex")
_t = open(f"{HERE}/tables/reg_life.tex").read().replace(
    r"\toprule", r"\toprule" + "\n" + r" & \multicolumn{4}{c}{13th-month lapse$_t$} & \multicolumn{4}{c}{25th-month lapse$_t$} \\ \cmidrule(lr){2-5}\cmidrule(lr){6-9}", 1)
open(f"{HERE}/tables/reg_life.tex", "w").write(_t)

# ------------------------------------------------------------------ figures
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK2, MUTED, GRID, BASE = "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": BASE, "axes.labelcolor": INK2,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "figure.dpi": 200, "savefig.bbox": "tight", "axes.axisbelow": True})

# Fig 1: mis-selling trend (industry decline) - small multiples would be cluttered; show face-to-face vs TM/direct
fig, ax = plt.subplots(figsize=(6.4, 2.7))
mm = p[(p.sector == "nonlife") & p.year.between(2020, 2026) & ~p.company.isin(EXCL)]
for co, gg in mm.groupby("company"):
    col = C[1] if co in NONFACE else C[0]
    ax.plot(gg.year, gg.ms_bp, color=col, lw=1.1, alpha=0.55)
for grp, col, lab in ((NONFACE, C[1], "TM/direct-led (AIG, ACE, AXA)"), (None, C[0], "Face-to-face-led (11 insurers)")):
    sub = mm[mm.company.isin(NONFACE)] if grp else mm[~mm.company.isin(NONFACE)]
    med = sub.groupby("year").ms_bp.median()
    ax.plot(med.index, med.values, color=col, lw=2.6, label=lab + ", median")
ax.set_ylabel("Mis-selling rate (bp)"); ax.set_xlabel("Year (average of two half-years)")
ax.legend(fontsize=7, loc="upper right")
fig.savefig(f"{HERE}/fig/ms_trend.pdf"); plt.close(fig)

# Fig 2: between vs within, mis-selling(t-1) vs lapse13
d = nl[["company", "year", "ms_bp_l1", "lapse13"]].dropna()
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8))
bm = d.groupby("company")[["ms_bp_l1", "lapse13"]].mean()
for co, r in bm.iterrows():
    axes[0].scatter(r.ms_bp_l1, r.lapse13, s=26, color=C[1] if co in NONFACE else C[0], zorder=3)
    if co in NONFACE or r.lapse13 > 16:
        axes[0].annotate(ENG.get(co, co), (r.ms_bp_l1, r.lapse13), xytext=(4, 2), textcoords="offset points", fontsize=6.5, color=INK2)
b1 = np.polyfit(bm.ms_bp_l1, bm.lapse13, 1); xx = np.linspace(bm.ms_bp_l1.min(), bm.ms_bp_l1.max(), 10)
axes[0].plot(xx, np.polyval(b1, xx), color=INK2, lw=1, ls="--")
axes[0].set_xlim(right=bm.ms_bp_l1.max() * 1.45)
axes[0].set_title("Between firms (5-year means)", fontsize=8.5, loc="left")
axes[0].set_xlabel("Mis-selling$_{t-1}$ (bp)"); axes[0].set_ylabel("13th-month lapse (%)")
w = d.copy()
for c in ("ms_bp_l1", "lapse13"):
    w[c] = w[c] - w.groupby("company")[c].transform("mean") - w.groupby("year")[c].transform("mean") + w[c].mean()
axes[1].scatter(w.ms_bp_l1, w.lapse13, s=16, color=[C[1] if co in NONFACE else C[0] for co in w.company], alpha=0.8)
b2 = np.polyfit(w.ms_bp_l1, w.lapse13, 1); xx = np.linspace(w.ms_bp_l1.min(), w.ms_bp_l1.max(), 10)
axes[1].plot(xx, np.polyval(b2, xx), color=INK2, lw=1, ls="--")
axes[1].set_title("Within firms (firm & year demeaned)", fontsize=8.5, loc="left")
axes[1].set_xlabel("Mis-selling$_{t-1}$ deviation (bp)"); axes[1].set_ylabel("Lapse deviation (pp)")
fig.savefig(f"{HERE}/fig/between_within_ms.pdf"); plt.close(fig)

# Fig 3: settlement vs lapse13 (life and non-life), firm means and within
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8))
for ax, df, title, col in ((axes[0], nl, "Non-life", C[0]), (axes[1], lf, "Life", C[2])):
    dd = df[["company", "year", "settle", "lapse13"]].dropna()
    ax.scatter(dd.settle, dd.lapse13, s=12, color=col, alpha=0.35, label="firm-year")
    bmm = dd.groupby("company")[["settle", "lapse13"]].mean()
    ax.scatter(bmm.settle, bmm.lapse13, s=30, color=col, edgecolor="white", linewidth=0.8, label="firm mean", zorder=3)
    bb = np.polyfit(bmm.settle, bmm.lapse13, 1); xx = np.linspace(bmm.settle.min(), bmm.settle.max(), 10)
    ax.plot(xx, np.polyval(bb, xx), color=INK2, lw=1, ls="--")
    ax.set_title(title, fontsize=8.5, loc="left"); ax.set_xlabel("Planner settlement rate (%)")
axes[0].set_ylabel("13th-month lapse (%)"); axes[0].legend(fontsize=7, loc="upper right")
fig.savefig(f"{HERE}/fig/settle_lapse.pdf"); plt.close(fig)

json.dump(R, open(f"{HERE}/results.json", "w"), indent=1, default=float)
if __name__ == "__main__":
    print(pd.DataFrame(cd).round(3).to_string())
    for k, v in R["models"].items():
        print(k, {x: (round(c["b"], 3), round(c["se"], 3), round(c["p"], 3)) for x, c in v["coef"].items()}, v["N"], v["firms"], round(v["r2"], 3))
    for k, v in R["robust"].items():
        print("ROB", k, {x: (round(c["b"], 3), round(c["se"], 3), round(c["p"], 3)) for x, c in v["coef"].items()}, v["N"])
    for k, v in R["life"].items():
        print("LIFE", k, {x: (round(c["b"], 3), round(c["se"], 3), round(c["p"], 3)) for x, c in v["coef"].items()}, v["N"], v["firms"])
    print({k: v for k, v in R.items() if k.startswith("hausman")})
    print(json.dumps(R["wcr"], indent=0))
    print(d_nl.round(2).to_string()); print(d_lf.round(2).to_string())
