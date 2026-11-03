"""Revizyon için ek analizler (Makale 1984489) — hakem talepleri B1–B6.
Girdi: dataset_final.csv (analysis.py çıktısı). Çıktı: rev_results.json
"""
import json, itertools, warnings
import numpy as np, pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.stats.multitest import multipletests
from arch.unitroot import ADF, PhillipsPerron, KPSS
from scipy import stats
from statsmodels.tsa.stattools import kpss as sm_kpss
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import xgboost as xgb, sklearn

warnings.filterwarnings("ignore")
d = pd.read_csv("dataset_final.csv", parse_dates=["date"]).set_index("date")
OUT = {"versions": {"statsmodels": sm.__version__ if hasattr(sm, "__version__") else None,
                    "sklearn": sklearn.__version__, "xgboost": xgb.__version__, "pandas": pd.__version__}}
import statsmodels; OUT["versions"]["statsmodels"] = statsmodels.__version__
r4 = lambda x: None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 4)

# ------------------------------------------------------------------ B4 birim kök
ur = {}
for c in ["ret", "fg", "dfg", "lgt", "dlgt", "lvol", "dlvol"]:
    s = d[c].dropna()
    a_c = ADF(s, trend="c", method="aic"); a_ct = ADF(s, trend="ct", method="aic")
    pp = PhillipsPerron(s, trend="c"); kp_stat, kp_p, kp_lag, _ = sm_kpss(s.values, regression="c", nlags="auto")
    ur[c] = {"ADF_c": [r4(a_c.stat), r4(a_c.pvalue)], "ADF_ct": [r4(a_ct.stat), r4(a_ct.pvalue)],
             "PP": [r4(pp.stat), r4(pp.pvalue)], "KPSS": [r4(kp_stat), r4(kp_p)]}
OUT["unit_root"] = ur

# ------------------------------------------------------------------ Model (2): R² ayrıştırma + HAC duyarlılığı
cc = d[["ret", "fg", "dfg", "dlgt", "dlvol"]].dropna()
T = len(cc)
nw_rule = int(np.floor(4 * (T / 100) ** (2 / 9)))
OUT["hac_rule_lag"] = nw_rule
regs = ["dfg", "dlgt", "dlvol"]
def r2(cols):
    if not cols: return 0.0
    return sm.OLS(cc.ret, sm.add_constant(cc[list(cols)])).fit().rsquared
sub = {}
for k in range(0, 4):
    for comb in itertools.combinations(regs, k): sub[frozenset(comb)] = r2(comb)
shap = {}
for v in regs:  # LMG / Shapley ayrıştırması
    others = [x for x in regs if x != v]; tot = 0; w = 0
    for k in range(0, 3):
        for comb in itertools.combinations(others, k):
            import math
            wt = math.factorial(k) * math.factorial(3 - k - 1) / math.factorial(3)
            tot += wt * (sub[frozenset(comb + (v,))] - sub[frozenset(comb)])
    shap[v] = r4(tot)
OUT["reg1_r2_decomp"] = {"full": r4(sub[frozenset(regs)]), "only_dfg": r4(sub[frozenset(["dfg"])]),
                         "only_gt_vol": r4(sub[frozenset(["dlgt", "dlvol"])]), "shapley": shap}
hac_sens = {}
X = sm.add_constant(cc[regs])
for L in [nw_rule, 7, 14, 21]:
    m = sm.OLS(cc.ret, X).fit(cov_type="HAC", cov_kwds={"maxlags": L})
    hac_sens[str(L)] = {"t": m.tvalues.round(2).to_dict(), "p": m.pvalues.round(4).to_dict()}
OUT["reg1_hac_sens"] = hac_sens
# Uç değer sağlamlığı: 12 Mart 2020 hariç ve %1 winsorize
def fit_reg1(df):
    m = sm.OLS(df.ret, sm.add_constant(df[regs])).fit(cov_type="HAC", cov_kwds={"maxlags": 7})
    return {"params": m.params.round(4).to_dict(), "p": m.pvalues.round(4).to_dict(), "r2": r4(m.rsquared)}
w = cc.copy()
for c in ["ret"] + regs:
    lo, hi = w[c].quantile([0.01, 0.99]); w[c] = w[c].clip(lo, hi)
OUT["reg1_robust"] = {"excl_2020_03_12": fit_reg1(cc.drop(pd.Timestamp("2020-03-12"))), "winsor_1pct": fit_reg1(w)}

# Model (3) alternatifleri: tutarlı gecikme yapısı
L1 = d[["ret", "fg", "dfg", "dlgt", "dlvol"]].copy()
for c in list(L1.columns):
    for k in [1, 2, 3]: L1[f"{c}_l{k}"] = L1[c].shift(k)
L1 = L1.dropna()
def predreg(cols):
    m = sm.OLS(L1.ret, sm.add_constant(L1[cols])).fit(cov_type="HAC", cov_kwds={"maxlags": 7})
    mo = sm.OLS(L1.ret, sm.add_constant(L1[cols])).fit()
    return {"params": m.params.round(4).to_dict(), "p": m.pvalues.round(4).to_dict(), "r2": r4(mo.rsquared),
            "Fhac": r4(float(m.fvalue)), "Fhac_p": r4(float(m.f_pvalue)), "n": int(mo.nobs)}
OUT["reg2_alt"] = {
    "level_fg": predreg(["fg_l1", "dlgt_l1", "dlvol_l1", "ret_l1"]),
    "diff_fg": predreg(["dfg_l1", "dlgt_l1", "dlvol_l1", "ret_l1"]),
    "lags1to3_diff": predreg([f"{c}_l{k}" for k in [1, 2, 3] for c in ["dfg", "dlgt", "dlvol", "ret"]]),
}

# ------------------------------------------------------------------ B5 F&G mekanik ilişki
M = d[["ret", "dfg", "dlgt", "dlvol"]].copy()
M["absret"] = M.ret.abs()
for k in [1, 2, 3]: M[f"ret_l{k}"] = M.ret.shift(k)
M = M.dropna()
mech_cols = ["ret", "absret", "ret_l1", "ret_l2", "ret_l3", "dlvol", "dlgt"]
mm = sm.OLS(M.dfg, sm.add_constant(M[mech_cols])).fit(cov_type="HAC", cov_kwds={"maxlags": 7})
mm_only_ret = sm.OLS(M.dfg, sm.add_constant(M[["ret"]])).fit()
M["dfg_purged"] = mm.resid
mech = {"r2_price_vars": r4(mm.rsquared), "r2_ret_only": r4(mm_only_ret.rsquared),
        "params": mm.params.round(4).to_dict(), "p": mm.pvalues.round(4).to_dict()}
# Fiyattan arındırılmış duyarlılık bir sonraki günün getirisini öngörüyor mu?
P = M[["ret", "dfg_purged"]].copy(); P["pur_l1"] = P.dfg_purged.shift(1); P["ret_l1"] = P.ret.shift(1); P = P.dropna()
pm = sm.OLS(P.ret, sm.add_constant(P[["pur_l1", "ret_l1"]])).fit(cov_type="HAC", cov_kwds={"maxlags": 7})
mech["pred_purged"] = {"params": pm.params.round(4).to_dict(), "p": pm.pvalues.round(4).to_dict(), "r2": r4(pm.rsquared)}
vp = VAR(M[["ret", "dfg_purged"]]); lagp = max(1, int(vp.select_order(10).aic)); rp = vp.fit(lagp)
tp = rp.test_causality("ret", ["dfg_purged"], kind="f")
mech["granger_purged"] = {"lag": lagp, "F": r4(tp.test_statistic), "p": r4(tp.pvalue)}
OUT["fg_mechanical"] = mech

# ------------------------------------------------------------------ B3 Granger: gecikme duyarlılığı, HAC-Wald, tanılar, çoklu sınama
def hac_granger(data, y, x, lag):
    Z = pd.DataFrame({"y": data[y]})
    for k in range(1, lag + 1):
        Z[f"y_l{k}"] = data[y].shift(k); Z[f"x_l{k}"] = data[x].shift(k)
    Z = Z.dropna()
    m = sm.OLS(Z.y, sm.add_constant(Z.drop(columns="y"))).fit(cov_type="HAC", cov_kwds={"maxlags": 7})
    R_ = np.zeros((lag, m.params.shape[0]))
    names = list(m.params.index)
    for i, k in enumerate(range(1, lag + 1)): R_[i, names.index(f"x_l{k}")] = 1
    t = m.f_test(R_)
    return r4(float(t.fvalue)), r4(float(t.pvalue))

gr = {}; main_p = []; main_lab = []
for x in ["dfg", "dlgt", "dlvol"]:
    data = d[["ret", x]].dropna(); v = VAR(data); sel = v.select_order(10)
    crit = {"AIC": int(sel.aic), "BIC": int(sel.bic), "HQIC": int(sel.hqic)}
    rows = {}
    for lab, lag in list({f"{k}": max(1, l) for k, l in crit.items()}.items()) + [("fix1", 1), ("fix5", 5), ("fix10", 10)]:
        res = v.fit(lag)
        t1 = res.test_causality("ret", [x], kind="f"); t2 = res.test_causality(x, ["ret"], kind="f")
        wh = res.test_whiteness(nlags=lag + 10, adjusted=True)
        rows[lab] = {"lag": lag, "x_to_ret": [r4(t1.test_statistic), r4(t1.pvalue)],
                     "ret_to_x": [r4(t2.test_statistic), r4(t2.pvalue)],
                     "x_to_ret_HAC": hac_granger(data, "ret", x, lag), "ret_to_x_HAC": hac_granger(data, x, "ret", lag),
                     "stable": bool(res.is_stable()), "portmanteau_p": r4(wh.pvalue)}
    gr[x] = {"selected": crit, "tests": rows}
    main_p += [rows["AIC"]["x_to_ret"][1], rows["AIC"]["ret_to_x"][1]]; main_lab += [f"{x}->ret", f"ret->{x}"]
gr["multiple_testing"] = {
    "labels": main_lab, "raw": main_p,
    "holm": [r4(p) for p in multipletests(main_p, method="holm")[1]],
    "bh": [r4(p) for p in multipletests(main_p, method="fdr_bh")[1]],
    "bonferroni": [r4(p) for p in multipletests(main_p, method="bonferroni")[1]],
}
p3 = [main_p[0], main_p[2], main_p[4]]
gr["multiple_testing_x_to_ret_only"] = {"holm": [r4(p) for p in multipletests(p3, method="holm")[1]],
                                        "bh": [r4(p) for p in multipletests(p3, method="fdr_bh")[1]]}
OUT["granger"] = gr

# ------------------------------------------------------------------ B1/B2/B6 ML: ortak bilgi kümesi, çoklu test dönemi, kayan pencere
F = pd.DataFrame(index=d.index)
for k in [1, 2, 3]:
    for c in ["ret", "dfg", "dlgt", "dlvol"]: F[f"{c}_l{k}"] = d[c].shift(k)
F["fg_l1"] = d.fg.shift(1); F["y"] = d.ret; F = F.dropna()
FULL = [f"{c}_l{k}" for k in [1, 2, 3] for c in ["ret", "dfg", "dlgt", "dlvol"]] + ["fg_l1"]
RET = ["ret_l1", "ret_l2", "ret_l3"]
RF_GRID = {"n_estimators": [300], "max_depth": [3, 5, 8], "min_samples_leaf": [5, 20, 50]}
XG_GRID = {"n_estimators": [200, 500], "max_depth": [2, 3, 5], "learning_rate": [0.01, 0.05], "subsample": [0.8]}
OUT["grids"] = {"rf": RF_GRID, "xgb": XG_GRID}

def dm_test(e1, e2):  # e1: aday, e2: kıyas; kare kayıp, h=1, HLN düzeltmeli
    dd = e1 ** 2 - e2 ** 2; n = len(dd)
    stat = dd.mean() / np.sqrt(np.var(dd, ddof=0) / n)
    hln = stat * np.sqrt((n + 1 - 2 + 0) / n)
    return r4(hln), r4(2 * (1 - stats.t.cdf(abs(hln), df=n - 1)))

def clark_west(y, p_small, p_big):
    f = (y - p_small) ** 2 - ((y - p_big) ** 2 - (p_small - p_big) ** 2)
    t = f.mean() / np.sqrt(np.var(f, ddof=1) / len(f))
    return r4(t), r4(1 - stats.norm.cdf(t))

def fit_models(tr, te):
    ytr, yte = tr.y, te.y; P = {}
    P["Tarihsel ortalama"] = np.full(len(te), ytr.mean()); P["Rassal yürüyüş"] = np.zeros(len(te))
    P["AR(3) OLS"] = sm.OLS(ytr, sm.add_constant(tr[RET])).fit().predict(sm.add_constant(te[RET], has_constant="add")).values
    P["OLS (tüm öznitelikler)"] = sm.OLS(ytr, sm.add_constant(tr[FULL])).fit().predict(sm.add_constant(te[FULL], has_constant="add")).values
    best = {}
    for nm, feats in [("tam", FULL), ("getiri", RET)]:
        rf = GridSearchCV(RandomForestRegressor(random_state=42, n_jobs=-1), RF_GRID, cv=TimeSeriesSplit(5),
                          scoring="neg_mean_squared_error").fit(tr[feats], ytr)
        xg = GridSearchCV(xgb.XGBRegressor(random_state=42, n_jobs=4, objective="reg:squarederror"), XG_GRID,
                          cv=TimeSeriesSplit(5), scoring="neg_mean_squared_error").fit(tr[feats], ytr)
        P[f"RF ({nm})"] = rf.predict(te[feats]); P[f"XGB ({nm})"] = xg.predict(te[feats])
        best[f"RF ({nm})"] = rf.best_params_; best[f"XGB ({nm})"] = xg.best_params_
    hm = P["Tarihsel ortalama"]; yv = yte.values; res = {}
    for k, p in P.items():
        e = yv - p
        res[k] = {"rmse": r4(np.sqrt(np.mean(e ** 2))), "mae": r4(np.mean(np.abs(e))),
                  "r2_oos": r4(1 - np.sum(e ** 2) / np.sum((yv - hm) ** 2)),
                  "da": r4(np.mean(np.sign(p) == np.sign(yv))) if np.std(p) > 0 else None,
                  "dm_vs_hm": None if k == "Tarihsel ortalama" else dm_test(e, yv - hm)}
    # duyarlılığın ek katkısı: aynı algoritma, tam vs yalnız getiri
    res["_incremental"] = {
        "RF tam vs RF getiri (DM)": dm_test(yv - P["RF (tam)"], yv - P["RF (getiri)"]),
        "XGB tam vs XGB getiri (DM)": dm_test(yv - P["XGB (tam)"], yv - P["XGB (getiri)"]),
        "OLS tam vs AR(3) (Clark-West)": clark_west(yv, P["AR(3) OLS"], P["OLS (tüm öznitelikler)"]),
        "AR(3) vs tarihsel ortalama (Clark-West)": clark_west(yv, hm, P["AR(3) OLS"]),
    }
    res["_best_params"] = best
    res["_n"] = [len(tr), len(te), str(tr.index[0].date()), str(tr.index[-1].date()), str(te.index[0].date()), str(te.index[-1].date())]
    return res

ml = {}
s = int(len(F) * 0.8)
ml["orijinal_80_20"] = fit_models(F.iloc[:s], F.iloc[s:])
# CV kat sınırları (orijinal eğitim seti)
folds = []
for a, b in TimeSeriesSplit(5).split(F.iloc[:s]):
    folds.append([str(F.index[a[0]].date()), str(F.index[a[-1]].date()), str(F.index[b[0]].date()), str(F.index[b[-1]].date())])
OUT["cv_folds"] = folds
for Y in [2022, 2023, 2024]:
    te = F.loc[f"{Y}-01-01":f"{Y}-12-31"]
    ml[f"genisleyen_{Y}"] = fit_models(F.loc[:f"{Y - 1}-12-31"], te)
    ml[f"kayan730_{Y}"] = fit_models(F.loc[:f"{Y - 1}-12-31"].iloc[-730:], te)
OUT["ml"] = ml

json.dump(OUT, open("rev_results.json", "w"), ensure_ascii=False, indent=1, default=str)
print("OK")
