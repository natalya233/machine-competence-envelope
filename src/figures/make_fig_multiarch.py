import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.linewidth":0.8})
NAVY="#1f3b63"; GREEN="#1a8a4a"; RED="#c0392b"; TEAL="#1a8a8a"; AMBER="#d9a521"; PURPLE="#7a5aa8"; GREY="#666"

A=json.load(open("multiarch_A.json"))
BC=json.load(open("multiarch_bc.json"))
archs=A["archs"]; months=A["months"]; ng=A["n_grid"]
ARCH_LABEL={"logreg":"LogReg","mlp":"MLP","gbdt":"GBDT","xgboost":"XGBoost","lgbm":"LightGBM"}
ARCH_COL={"logreg":NAVY,"mlp":PURPLE,"gbdt":TEAL,"xgboost":AMBER,"lgbm":RED}
def gA(arch,gi,n): return A["A"][f"{arch}__{gi}__{n}"]
def gB(arch,gi,n): return BC["bc"][f"{arch}__{gi}__{n}"]

fig=plt.figure(figsize=(13.4,4.2))
gs=fig.add_gridspec(1,3,width_ratios=[1,1,1],wspace=0.34)

# ── (a) Coverage vs drift (month) at n=2400, all 5 archs ──────────────────
axa=fig.add_subplot(gs[0,0])
MON={1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May"}
for arch in archs:
    covs=[gA(arch,gi,2400)["cover"] for gi in range(len(months))]
    axa.plot(range(len(months)),covs,"-o",color=ARCH_COL[arch],ms=4,lw=1.8,label=ARCH_LABEL[arch])
axa.axhline(0.88,ls="--",color=GREY,lw=1.2); axa.text(0.05,0.885,"coverage limit 0.88",fontsize=6.8,color=GREY)
axa.set_xticks(range(len(months))); axa.set_xticklabels([MON[m] for m in months])
axa.set_xlabel("evaluation month  (drift →)"); axa.set_ylabel("conformal coverage")
axa.set_ylim(0.78,0.97)
axa.set_title("a   Coverage degrades under drift\n(Telegram, n = 2,400, all architectures)",
              fontweight="bold",fontsize=9,color=NAVY,loc="left")
axa.legend(fontsize=7,loc="lower left",ncol=2,framealpha=0.9)
axa.grid(alpha=0.2,lw=0.5)

# ── (b) Explanation stability S vs scarcity (n) at Jan, all 5 archs ────────
axb=fig.add_subplot(gs[0,1])
for arch in archs:
    Ss=[gA(arch,0,n)["S"] for n in ng]
    axb.plot(ng,Ss,"-o",color=ARCH_COL[arch],ms=4,lw=1.8,label=ARCH_LABEL[arch])
axb.axhline(0.20,ls="--",color=RED,lw=1.2); axb.text(320,0.205,"S limit 0.20",fontsize=6.8,color=RED)
axb.set_xscale("log"); axb.set_xticks(ng); axb.set_xticklabels([str(n) for n in ng],fontsize=7.5)
axb.set_xlabel("training size  n  (scarcity ←)"); axb.set_ylabel("explanation-stability drift  S")
axb.set_title("b   Stability degrades under scarcity\n(Telegram, January, all architectures)",
              fontweight="bold",fontsize=9,color=NAVY,loc="left")
axb.legend(fontsize=7,loc="upper right",ncol=2,framealpha=0.9)
axb.grid(alpha=0.2,lw=0.5)

# ── (c) Clinical: coverage vs drift band, tabular archs ───────────────────
axc=fig.add_subplot(gs[0,2])
bc_archs=BC["archs"]; bands=BC["bands"]
for arch in bc_archs:
    covs=[gB(arch,gi,250)["cover"] for gi in range(len(bands))]
    axc.plot(range(len(bands)),covs,"-o",color=ARCH_COL[arch],ms=4,lw=1.8,label=ARCH_LABEL[arch])
axc.axhline(0.88,ls="--",color=GREY,lw=1.2); axc.text(0.05,0.885,"coverage limit 0.88",fontsize=6.8,color=GREY)
axc.set_xticks(range(len(bands))); axc.set_xticklabels([f"band {b+1}" for b in bands])
axc.set_xlabel("covariate-drift band  (→ more OOD)"); axc.set_ylabel("conformal coverage")
axc.set_ylim(0.60,1.0)
axc.set_title("c   Clinical tabular (breast cancer):\nsame pattern, gradient-boosted models",
              fontweight="bold",fontsize=9,color=NAVY,loc="left")
axc.legend(fontsize=7,loc="lower left",framealpha=0.9)
axc.grid(alpha=0.2,lw=0.5)

fig.savefig("fig_multiarch.png",dpi=200,bbox_inches="tight")
plt.close(fig)
print("saved fig_multiarch.png")
