import pandas as pd, numpy as np, json, warnings
warnings.filterwarnings("ignore")
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.api import VAR
import statsmodels.api as sm
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
import xgboost as xgb

df = pd.read_csv("dataset_raw.csv", parse_dates=["date"]).set_index("date")
df["ret"] = 100 * np.log(df.close / df.close.shift(1))          # % log return
df["fg_pub"] = df.fg
# F&G value dated t is published 00:00 UTC and summarises day t-1 -> align to market day
df["fg"] = df.fg_pub.shift(-1)
df.loc[df.index[-1], "fg"] = 66  # 2025-01-01 published value (summarises 2024-12-31)
df["dfg"] = df.fg.diff()                                          # change in F&G
df["lgt"] = np.log(df["gt"])
df["dlgt"] = df.lgt.diff()
df["lvol"] = np.log(df.volume_musd)
df["dlvol"] = df.lvol.diff()
d = df.loc["2020-01-01":].copy()
d = d.dropna(subset=["ret"])
R = {}
R["n_obs"] = int(len(d))
R["desc"] = d[["ret", "fg", "gt", "volume_musd"]].describe().T.round(4).to_dict()

# ---- ADF
adf = {}
for c in ["ret", "fg", "dfg", "lgt", "dlgt", "lvol", "dlvol"]:
    s = d[c].dropna()
    st, p, *_ = adfuller(s, autolag="AIC", regression="c")
    adf[c] = (round(st, 3), round(p, 4))
R["adf"] = adf

# ---- correlations (stationary transforms) with p-values
cols = ["ret", "fg", "dfg", "dlgt", "dlvol"]
cc = d[cols].dropna()
corr = {}
for a in cols:
    for b in cols:
        r, p = stats.pearsonr(cc[a], cc[b])
        corr[f"{a}|{b}"] = (round(r, 3), round(p, 4))
R["corr"] = corr
R["corr_n"] = len(cc)

# ---- contemporaneous regression, HAC SE
X = sm.add_constant(cc[["dfg", "dlgt", "dlvol"]])
m1 = sm.OLS(cc["ret"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 7})
m1o = sm.OLS(cc["ret"], X).fit()
R["reg1"] = {"params": m1.params.round(4).to_dict(), "t": m1.tvalues.round(2).to_dict(), "p": m1.pvalues.round(4).to_dict(),
             "r2": round(m1o.rsquared, 4), "adjr2": round(m1o.rsquared_adj, 4), "F": round(m1o.fvalue, 2), "Fp": float(m1o.f_pvalue),
             "aic": round(m1o.aic, 1), "bic": round(m1o.bic, 1), "n": int(m1o.nobs),
             "Fhac": round(float(m1.fvalue), 2), "Fhac_p": float(m1.f_pvalue)}
# std betas
z = (cc - cc.mean()) / cc.std()
ms = sm.OLS(z["ret"], z[["dfg", "dlgt", "dlvol"]]).fit()
R["reg1_std_beta"] = ms.params.round(3).to_dict()

# ---- predictive regression (lag 1)
L = d[["ret", "fg", "dfg", "dlgt", "dlvol"]].copy()
for c in ["ret", "fg", "dfg", "dlgt", "dlvol"]:
    L[c + "_l1"] = L[c].shift(1)
L = L.dropna()
X2 = sm.add_constant(L[["fg_l1", "dlgt_l1", "dlvol_l1", "ret_l1"]])
m2 = sm.OLS(L["ret"], X2).fit(cov_type="HAC", cov_kwds={"maxlags": 7})
m2o = sm.OLS(L["ret"], X2).fit()
R["reg2"] = {"params": m2.params.round(4).to_dict(), "t": m2.tvalues.round(2).to_dict(), "p": m2.pvalues.round(4).to_dict(),
             "r2": round(m2o.rsquared, 4), "adjr2": round(m2o.rsquared_adj, 4), "F": round(m2o.fvalue, 2), "Fp": float(m2o.f_pvalue),
             "aic": round(m2o.aic, 1), "bic": round(m2o.bic, 1), "n": int(m2o.nobs),
             "Fhac": round(float(m2.fvalue), 2), "Fhac_p": float(m2.f_pvalue)}

# ---- Granger (bivariate VAR, lag by AIC, max 10), both directions
gr = {}
for x in ["dfg", "dlgt", "dlvol"]:
    data = d[["ret", x]].dropna()
    sel = VAR(data).select_order(10)
    lag = max(1, int(sel.aic))
    res = VAR(data).fit(lag)
    t1 = res.test_causality("ret", [x], kind="f")
    t2 = res.test_causality(x, ["ret"], kind="f")
    gr[x] = {"lag": lag, "x_to_ret": (round(t1.test_statistic, 2), round(t1.pvalue, 4)),
             "ret_to_x": (round(t2.test_statistic, 2), round(t2.pvalue, 4))}
R["granger"] = gr

# ---- ML: predict ret_t with info up to t-1
F = pd.DataFrame(index=d.index)
for k in [1, 2, 3]:
    F[f"ret_l{k}"] = d.ret.shift(k)
    F[f"dfg_l{k}"] = d.dfg.shift(k)
    F[f"dlgt_l{k}"] = d.dlgt.shift(k)
    F[f"dlvol_l{k}"] = d.dlvol.shift(k)
F["fg_l1"] = d.fg.shift(1)
F["y"] = d.ret
F = F.dropna()
split = int(len(F) * 0.8)
tr, te = F.iloc[:split], F.iloc[split:]
feats = [c for c in F.columns if c != "y"]
Xtr, ytr, Xte, yte = tr[feats], tr.y, te[feats], te.y
R["ml_split"] = {"train": [str(tr.index[0].date()), str(tr.index[-1].date()), len(tr)],
                 "test": [str(te.index[0].date()), str(te.index[-1].date()), len(te)]}
tscv = TimeSeriesSplit(n_splits=5)
rf = GridSearchCV(RandomForestRegressor(random_state=42, n_jobs=-1),
                  {"n_estimators": [300], "max_depth": [3, 5, 8], "min_samples_leaf": [5, 20, 50]},
                  cv=tscv, scoring="neg_mean_squared_error").fit(Xtr, ytr)
xg = GridSearchCV(xgb.XGBRegressor(random_state=42, n_jobs=4, objective="reg:squarederror"),
                  {"n_estimators": [200, 500], "max_depth": [2, 3, 5], "learning_rate": [0.01, 0.05], "subsample": [0.8]},
                  cv=tscv, scoring="neg_mean_squared_error").fit(Xtr, ytr)
preds = {"Random Forest": rf.predict(Xte), "XGBoost": xg.predict(Xte),
         "Sıfır getiri (rassal yürüyüş)": np.zeros(len(yte)),
         "Tarihsel ortalama": np.full(len(yte), ytr.mean())}
perf = {}
for k, p in preds.items():
    da = float(np.mean(np.sign(p) == np.sign(yte))) if np.std(p) > 0 else np.nan
    perf[k] = {"rmse": round(float(np.sqrt(mean_squared_error(yte, p))), 4), "mae": round(float(mean_absolute_error(yte, p)), 4),
               "r2": round(float(r2_score(yte, p)), 4), "da": None if np.isnan(da) else round(da, 4)}
# OOS R2 vs historical mean (Campbell-Thompson)
for k in ["Random Forest", "XGBoost"]:
    perf[k]["r2_oos_hm"] = round(1 - np.sum((yte - preds[k]) ** 2) / np.sum((yte - ytr.mean()) ** 2), 4)
# Diebold-Mariano vs historical mean (squared loss)
def dm(e1, e2):
    dd = e1 ** 2 - e2 ** 2
    n = len(dd); mu = dd.mean()
    gamma0 = np.var(dd, ddof=0)
    stat = mu / np.sqrt(gamma0 / n)
    return round(float(stat), 3), round(float(2 * (1 - stats.norm.cdf(abs(stat)))), 4)
for k in ["Random Forest", "XGBoost"]:
    perf[k]["dm_vs_hm"] = dm(yte.values - preds[k], yte.values - ytr.mean())
R["ml_perf"] = perf
R["rf_best"] = rf.best_params_
R["xgb_best"] = xg.best_params_
# importances: XGB gain (normalised) + permutation for both, aggregated by indicator group
imp_x = pd.Series(xg.best_estimator_.feature_importances_, index=feats)
pi_rf = permutation_importance(rf.best_estimator_, Xte, yte, n_repeats=30, random_state=42, scoring="neg_mean_squared_error")
pi_xg = permutation_importance(xg.best_estimator_, Xte, yte, n_repeats=30, random_state=42, scoring="neg_mean_squared_error")
def grp(n):
    if n.startswith("ret"): return "Gecikmeli getiri"
    if n.startswith("dfg") or n.startswith("fg"): return "Fear & Greed Index"
    if n.startswith("dlgt"): return "Google Trends"
    return "İşlem hacmi"
R["imp_xgb_gain"] = imp_x.groupby(grp).sum().round(4).sort_values(ascending=False).to_dict()
R["imp_rf_gain"] = pd.Series(rf.best_estimator_.feature_importances_, index=feats).groupby(grp).sum().round(4).sort_values(ascending=False).to_dict()
R["perm_rf"] = pd.Series(pi_rf.importances_mean, index=feats).groupby(grp).sum().round(5).to_dict()
R["perm_xgb"] = pd.Series(pi_xg.importances_mean, index=feats).groupby(grp).sum().round(5).to_dict()

d.to_csv("dataset_final.csv")
pd.to_pickle({"yte": yte, "preds": preds}, "preds.pkl")
json.dump(R, open("results.json", "w"), ensure_ascii=False, indent=1, default=str)
print(json.dumps(R, ensure_ascii=False, indent=1, default=str))
