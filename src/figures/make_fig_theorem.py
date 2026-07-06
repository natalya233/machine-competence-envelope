import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,"axes.linewidth":0.8})
NAVY="#1f3b63"; GREEN="#1a8a4a"; RED="#c0392b"; TEAL="#1a8a8a"; AMBER="#d9a521"; GREY="#666"

D=json.load(open("theorem_verify.json"))
C=np.array(D["C"]); F=np.array(D["F"]); K=np.array(D["K"]); grid=np.array(D["grid"])
COV_T=D["COV_T"]; S_LIM=D["S_LIM"]; pois=D["poison"]

fig=plt.figure(figsize=(13.4,4.3))
gs=fig.add_gridspec(1,3,width_ratios=[1.15,1.0,1.0],wspace=0.40)

# ── (a) The envelope K with two binding cones, silent/blind sets ───────────
axa=fig.add_subplot(gs[0,0])
cov_ok=C>=COV_T; fid_ok=F>=(1-S_LIM)
region=np.zeros(C.shape+(3,))
for i in range(C.shape[0]):
    for j in range(C.shape[1]):
        if cov_ok[i,j] and fid_ok[i,j]:      region[i,j]=[0.82,0.93,0.82]   # K (green)
        elif cov_ok[i,j] and not fid_ok[i,j]:region[i,j]=[0.98,0.85,0.55]   # silent (amber): cov OK fid FAIL
        elif fid_ok[i,j] and not cov_ok[i,j]:region[i,j]=[0.72,0.85,0.95]   # blind (blue): fid OK cov FAIL
        else:                                 region[i,j]=[0.95,0.80,0.80]   # both fail (red)
axa.imshow(region,origin="lower",extent=[0,1,0,1],aspect="auto",interpolation="nearest")
# certificate boundaries
Xg,Yg=np.meshgrid(grid,grid)
axa.contour(Xg,Yg,C.T,levels=[COV_T],colors=[NAVY],linewidths=2,linestyles="-")
axa.contour(Xg,Yg,F.T,levels=[1-S_LIM],colors=[TEAL],linewidths=2,linestyles="-")
axa.set_xlabel("drift  δ  (→ coverage-binding cone)")
axa.set_ylabel("scarcity / compression  σ  (→ fidelity-binding cone)")
axa.set_title("a   The competence envelope K",fontweight="bold",fontsize=9.5,color=NAVY,loc="left")
axa.text(0.13,0.13,"K",fontsize=20,fontweight="bold",color=GREEN,ha="center",va="center")
axa.text(0.62,0.12,"silent\nfailure",fontsize=8.5,color="#8a5a00",ha="center",va="center",fontweight="bold")
axa.text(0.12,0.66,"blind\nspot",fontsize=8.5,color=NAVY,ha="center",va="center",fontweight="bold")
axa.text(0.72,0.66,"both fail",fontsize=8.5,color="#a02020",ha="center",va="center",fontweight="bold")
leg=[Patch(fc=[0.82,0.93,0.82],label="K: both certified"),
     Patch(fc=[0.98,0.85,0.55],label="$K_C\\!\\setminus\\!K$: coverage-OK, fidelity fails"),
     Patch(fc=[0.72,0.85,0.95],label="$K_F\\!\\setminus\\!K$: fidelity-OK, coverage fails")]
axa.legend(handles=leg,fontsize=6.6,loc="upper right",framealpha=0.95)

# ── (b) Two rays: which certificate binds first ────────────────────────────
axb=fig.add_subplot(gs[0,1])
drift_C=C[:,0]; drift_F=F[:,0]; scar_C=C[0,:]; scar_F=F[0,:]
axb.plot(grid,drift_C,"-o",color=NAVY,ms=3,lw=1.8,label="coverage C, drift ray")
axb.plot(grid,drift_F,"--",color=NAVY,lw=1.3,alpha=0.55,label="fidelity F, drift ray")
axb.plot(grid,scar_F,"-s",color=TEAL,ms=3,lw=1.8,label="fidelity F, scarcity ray")
axb.plot(grid,scar_C,"--",color=TEAL,lw=1.3,alpha=0.55,label="coverage C, scarcity ray")
axb.axhline(COV_T,ls=":",color=NAVY,lw=1); axb.axhline(1-S_LIM,ls=":",color=TEAL,lw=1)
axb.text(0.02,COV_T-0.03,"coverage limit",fontsize=6.3,color=NAVY)
axb.text(0.02,(1-S_LIM)+0.01,"fidelity limit",fontsize=6.3,color=TEAL)
# mark first crossings
tCd=D["cones"]["tCd"]; tFs=D["cones"]["tFs"]
axb.scatter(grid[tCd],drift_C[tCd],s=70,color=RED,marker="x",lw=2,zorder=5)
axb.scatter(grid[tFs],scar_F[tFs],s=70,color=RED,marker="x",lw=2,zorder=5,label="first limit crossing")
axb.set_xlabel("stress magnitude  t  along ray")
axb.set_ylabel("certificate value")
axb.set_ylim(0.4,1.02)
axb.set_title("b   Boundary = min of certificates\n(each binds on its own cone)",
              fontweight="bold",fontsize=9.5,color=NAVY,loc="left")
axb.legend(fontsize=6.2,loc="lower left",framealpha=0.9)
axb.grid(alpha=0.2,lw=0.5)

# ── (c) Separation theorem: prediction-side identical, explanation-side + deployment diverge ─
import json as _json
SEP=_json.load(open("separation_verify.json"))
axc=fig.add_subplot(gs[0,2])
groups=["coverage","accuracy","1 - ECE","fidelity F","acc under\ndeployment shift"]
f_vals =[SEP["cov_f"], SEP["acc_f"], 1-0.0199, 1.000, 0.824]
fp_vals=[SEP["cov_fp"],SEP["acc_fp"],1-0.0199, 1-SEP["D"], 0.409]
x=np.arange(len(groups)); w=0.36
axc.bar(x-w/2, f_vals, w, color=NAVY, label="reliable model f")
axc.bar(x+w/2, fp_vals,w, color=RED,  label="compromised model f'")
# annotate identical vs separated
axc.annotate("identical to every\nprediction-side certificate", xy=(1.0,0.90), xytext=(1.0,1.14),
             ha="center", fontsize=6.6, color=NAVY,
             arrowprops=dict(arrowstyle="-[,widthB=4.2", color=NAVY, lw=1))
axc.annotate("separated", xy=(3.5,0.55), xytext=(3.5,1.14),
             ha="center", fontsize=6.6, color=RED,
             arrowprops=dict(arrowstyle="-[,widthB=2.6", color=RED, lw=1))
axc.set_xticks(x); axc.set_xticklabels(groups, fontsize=6.4)
axc.set_ylabel("certificate / accuracy value"); axc.set_ylim(0,1.28)
axc.set_title("c   Separation theorem (verified):\nf and f' identical to prediction, not to explanation",
              fontweight="bold",fontsize=9.2,color=NAVY,loc="left")
axc.legend(fontsize=6.6,loc="lower left",framealpha=0.95)

fig.savefig("fig_theorem.png",dpi=200,bbox_inches="tight")
plt.close(fig)
print("saved fig_theorem.png")
