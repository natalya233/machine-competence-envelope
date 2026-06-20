# Renders Paper Fig. 1 from exp_results.npz -> img/fig1_envelope.png
# Part of the open code for the Machine competence Perspective. MIT licence.
import os; os.makedirs("img", exist_ok=True)
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.lines import Line2D

plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["DejaVu Sans","Arial"],
                     "savefig.dpi":200,"figure.dpi":200})
NAVY="#16324F"; TEAL="#1B6B5A"; RUST="#B23A2E"; GOLD="#C8881F"; GREY="#5A6470"
G_OK="#cfe6d8"; R_REL="#f3c9c2"; O_STA="#f3ddb4"; D_BOTH="#d9c2b8"

D=np.load("exp_results.npz")
R,COV,S,SF=D["R"],D["COV"],D["S"],D["SF"]
drifts=D["drifts"]; ns=D["ns"]; ct=float(D["cov_thr"]); st=float(D["S_thr"])
nd=len(drifts); nn=len(ns)

fig=plt.figure(figsize=(12.4,5.0))
gs=fig.add_gridspec(1,2,width_ratios=[1.18,1.0],wspace=0.28)

# ---------------- Panel a: envelope map ----------------
axA=fig.add_subplot(gs[0,0])
for a in range(nn):          # a=0 -> n=320 (low scarcity) at bottom
    for b in range(nd):
        relok = COV[a,b]>=ct
        staok = S[a,b]<=st
        if relok and staok: c=G_OK
        elif staok and not relok: c=R_REL      # reliability (drift) fail only
        elif relok and not staok: c=O_STA      # stability (scarcity) fail only
        else: c=D_BOTH
        axA.add_patch(Rectangle((b-0.5,a-0.5),1,1,fc=c,ec="white",lw=1.5))
        axA.text(b,a-0.30,f"cov {COV[a,b]:.2f}",ha="center",va="center",fontsize=6.7,color="#3a4750")
        axA.text(b,a+0.04,f"S {S[a,b]:.2f}",ha="center",va="center",fontsize=6.7,color="#3a4750")
        axA.text(b,a+0.34,f"acc {R[a,b]:.2f}",ha="center",va="center",fontsize=6.5,color="#7a828b")

# certified-region outline
import numpy as np
env=(COV>=ct)&(S<=st)
# draw thick green border around certified cells (top-left block)
for a in range(nn):
    for b in range(nd):
        if env[a,b]:
            if a+1>=nn or not env[a+1,b]: axA.plot([b-0.5,b+0.5],[a+0.5,a+0.5],color=TEAL,lw=3)
            if a-1<0 or not env[a-1,b]:    axA.plot([b-0.5,b+0.5],[a-0.5,a-0.5],color=TEAL,lw=3)
            if b+1>=nd or not env[a,b+1]:  axA.plot([b+0.5,b+0.5],[a-0.5,a+0.5],color=TEAL,lw=3)
            if b-1<0 or not env[a,b-1]:    axA.plot([b-0.5,b-0.5],[a-0.5,a+0.5],color=TEAL,lw=3)

# annotate the two 'leak' cells (single-certificate would miss)
# stability-fail-only: find a cell with relok & not staok (prediction-only would pass)
axA.annotate("conformal alone\nwould pass, but\nexplanation drifted",
             xy=(0.0,2.0),xytext=(1.15,3.15),fontsize=8.0,color=GOLD,ha="left",
             arrowprops=dict(arrowstyle="-|>",color=GOLD,lw=1.4))
# reliability-fail-only: cell with not relok & staok (explanation-only would pass)
axA.annotate("explanation alone\nwould pass, but\ncoverage collapsed",
             xy=(3.0,0.0),xytext=(3.05,1.25),fontsize=8.0,color=RUST,ha="center",
             arrowprops=dict(arrowstyle="-|>",color=RUST,lw=1.4))

axA.text(1.0,0.5,"certified\nenvelope",color=TEAL,fontsize=11,fontweight="bold",ha="center",va="center")
axA.set_xticks(range(nd)); axA.set_xticklabels([f"{x:.1f}" for x in drifts])
axA.set_yticks(range(nn)); axA.set_yticklabels([f"{int(n)}" for n in ns])
axA.set_xlim(-0.5,nd-0.5); axA.set_ylim(-0.5,nn-0.5)
axA.set_xlabel("drift  \u03B4  (covariate shift, s.d.)  \u2192",fontsize=11)
axA.set_ylabel("\u2190  training size n   (scarcity \u2192 up)",fontsize=11)
for s in ["top","right"]: axA.spines[s].set_visible(False)
axA.text(-0.07,1.08,"a",transform=axA.transAxes,fontsize=16,fontweight="bold")
axA.set_title("Competence envelope over $\\Omega$  (real data)",fontsize=11.5,color=NAVY,loc="left",pad=22)

leg=[Line2D([0],[0],marker="s",ls="",mfc=G_OK,mec="white",ms=12,label="certified (cov \u2265 0.85 \u2227 S \u2264 0.15)"),
     Line2D([0],[0],marker="s",ls="",mfc=R_REL,mec="white",ms=12,label="reliability fails (drift)"),
     Line2D([0],[0],marker="s",ls="",mfc=O_STA,mec="white",ms=12,label="explanation stability fails (scarcity)"),
     Line2D([0],[0],marker="s",ls="",mfc=D_BOTH,mec="white",ms=12,label="both fail")]
axA.legend(handles=leg,loc="upper center",bbox_to_anchor=(0.5,-0.16),ncol=2,frameon=False,fontsize=8.0,handletextpad=0.3,columnspacing=1.0)

# ---------------- Panel b: early warning along drift (n=320) ----------------
axB=fig.add_subplot(gs[0,1])
row=0
x=np.arange(nd)
# shade certified vs withheld
cert=COV[row]>=ct
for b in range(nd):
    axB.axvspan(b-0.5,b+0.5,color=(G_OK if cert[b] else R_REL),alpha=0.55,zorder=0)
axB.bar(x,SF[row],width=0.55,color=RUST,zorder=3,label="confident-but-wrong rate")
axB.set_ylim(0,0.22)
axB.set_ylabel("confident-but-wrong rate",color=RUST,fontsize=10.5)
axB.tick_params(axis="y",labelcolor=RUST)
axB.set_xticks(x); axB.set_xticklabels([f"{d:.1f}" for d in drifts])
axB.set_xlabel("drift  \u03B4  \u2192  (n = 320)",fontsize=11)

axC=axB.twinx()
axC.plot(x,COV[row],"-o",color=NAVY,lw=2,zorder=4,label="conformal coverage")
axC.axhline(ct,ls="--",color=GREY,lw=1.2)
axC.text(nd-1.0,ct+0.006,"coverage target",fontsize=8,color=GREY,ha="right")
axC.set_ylim(0.70,0.95); axC.set_ylabel("conformal coverage",color=NAVY,fontsize=10.5)
axC.tick_params(axis="y",labelcolor=NAVY)

# mark certificate-withheld onset
onset=np.where(~cert)[0][0]
axB.axvline(onset-0.5,color="#222",lw=1.4,ls=(0,(4,3)))
axB.annotate("certificate withheld \u2192 abstain / defer / fall back",
             xy=(onset-0.5,0.158),xytext=(onset-0.42,0.214),fontsize=8.0,color="#222",ha="left",va="top",
             arrowprops=dict(arrowstyle="-|>",color="#222",lw=1.0))
axB.text(0.2,0.025,"certified",fontsize=9,color=TEAL,fontweight="bold")
axB.text(nd-1.0,0.055,"silent-failure\nregion",fontsize=9.5,color=RUST,fontweight="bold",ha="center",va="center")
axB.text(-0.07,1.08,"b",transform=axB.transAxes,fontsize=16,fontweight="bold")
axB.set_title("Certificate precedes silent failure",fontsize=11.5,color=NAVY,loc="left",pad=22)
for s in ["top"]: axB.spines[s].set_visible(False); axC.spines[s].set_visible(False)

plt.savefig("img/fig1_envelope.png",bbox_inches="tight",facecolor="white")
print("saved fig2 (empirical). dims:")
from PIL import Image; im=Image.open("img/fig1_envelope.png"); print(im.size, round(im.size[1]/im.size[0],4))
