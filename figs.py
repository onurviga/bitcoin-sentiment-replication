import json, pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
R = json.load(open("results.json"))
def tr(v, n=2):
    t = f"{v:.{n}f}"
    if float(t) == 0: t = t.lstrip("-")
    return t.replace(".", ",").replace("-", "−")
fmt = lambda n: FuncFormatter(lambda x, p: tr(x, n))

# Fig 1 heatmap
vars_ = ["ret", "fg", "dfg", "dlgt", "dlvol"]
lab = ["Bitcoin\ngetirisi", "F&G\ndüzeyi", "ΔF&G", "Δln Google\nTrends", "Δln işlem\nhacmi"]
M = np.array([[R["corr"][f"{a}|{b}"][0] for b in vars_] for a in vars_])
fig, ax = plt.subplots(figsize=(6.4, 5.0), dpi=200)
im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
for i in range(5):
    for j in range(5):
        ax.text(j, i, tr(M[i, j]), ha="center", va="center", color="white" if abs(M[i, j]) > 0.5 else "black", fontsize=10)
ax.set_xticks(range(5)); ax.set_yticks(range(5)); ax.set_xticklabels(lab, fontsize=9); ax.set_yticklabels(lab, fontsize=9)
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=[-1, -0.5, 0, 0.5, 1]); cb.ax.yaxis.set_major_formatter(fmt(1))
fig.tight_layout(); fig.savefig("fig1.png"); plt.close()

# Fig 2 importance
groups = ["Fear & Greed Index", "Google Trends", "İşlem hacmi", "Gecikmeli getiri"]
g = [R["imp_xgb_gain"][k] for k in groups]
prf = [R["perm_rf"][k] for k in groups]; pxg = [R["perm_xgb"][k] for k in groups]
fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.6), dpi=200)
cols = ["#1f6e8c", "#2e8b9a", "#52b3a8", "#8fd1b4"]
y = np.arange(len(groups))[::-1]
axs[0].barh(y, g, color=cols); axs[0].set_yticks(y); axs[0].set_yticklabels(groups)
for yi, v in zip(y, g): axs[0].text(v + 0.005, yi, tr(v, 3), va="center", fontsize=9)
axs[0].set_xlim(0, max(g) * 1.3); axs[0].xaxis.set_major_formatter(fmt(2)); axs[0].set_xlabel("Önem skoru (kazanç, toplam = 1)")
axs[0].set_title("(a) XGBoost kazanç temelli önem", fontsize=10)
h = 0.38
axs[1].barh(y + h / 2, prf, h, color="#1f6e8c", label="Random Forest"); axs[1].barh(y - h / 2, pxg, h, color="#8fd1b4", label="XGBoost")
axs[1].axvline(0, color="black", lw=0.8); axs[1].set_yticks(y); axs[1].set_yticklabels(groups)
axs[1].xaxis.set_major_formatter(fmt(2)); axs[1].set_xlabel("Test setinde MSE artışı")
axs[1].set_title("(b) Permütasyon önemi (test seti)", fontsize=10); axs[1].legend(fontsize=8, frameon=False, loc="upper right")
for a in axs: a.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig("fig2.png"); plt.close()

# Fig 3 performance
P = R["ml_perf"]
names = ["Random Forest", "XGBoost", "Tarihsel ortalama", "Sıfır getiri (rassal yürüyüş)"]
short = ["Random\nForest", "XGBoost", "Tarihsel\nortalama", "Rassal\nyürüyüş"]
fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.6), dpi=200, gridspec_kw={"width_ratios": [1.6, 1]})
x = np.arange(4); w = 0.38
axs[0].bar(x - w / 2, [P[n]["rmse"] for n in names], w, color="#1f6e8c", label="RMSE")
axs[0].bar(x + w / 2, [P[n]["mae"] for n in names], w, color="#e3a857", label="MAE")
for i, n in enumerate(names):
    axs[0].text(i - w / 2, P[n]["rmse"] + 0.03, tr(P[n]["rmse"], 3), ha="center", fontsize=7.5)
    axs[0].text(i + w / 2, P[n]["mae"] + 0.03, tr(P[n]["mae"], 3), ha="center", fontsize=7.5)
axs[0].set_xticks(x); axs[0].set_xticklabels(short, fontsize=9); axs[0].set_ylabel("Hata değeri (%)")
axs[0].yaxis.set_major_formatter(fmt(1)); axs[0].set_ylim(0, 3.2); axs[0].legend(frameon=False, fontsize=8)
r2 = [P["Random Forest"]["r2_oos_hm"], P["XGBoost"]["r2_oos_hm"]]
axs[1].bar([0, 1], r2, color=["#1f6e8c", "#c0392b"], width=0.55); axs[1].axhline(0, color="black", lw=0.8)
for i, v in enumerate(r2): axs[1].text(i, v - 0.002, tr(v, 4), ha="center", va="top", fontsize=8.5)
axs[1].set_xticks([0, 1]); axs[1].set_xticklabels(["Random\nForest", "XGBoost"], fontsize=9)
axs[1].set_ylabel("Örneklem dışı R²"); axs[1].yaxis.set_major_formatter(fmt(3)); axs[1].set_ylim(-0.032, 0.005)
for a in axs: a.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig("fig3.png"); plt.close()
print("ok")
