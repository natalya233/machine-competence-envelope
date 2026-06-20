# Renders Paper Fig. 2 from contraction.npz + deep_transfer.npz -> img/fig2_compound_transfer.png
# Part of the open code for the Machine competence Perspective. MIT licence.
import os; os.makedirs("img", exist_ok=True)
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import TwoSlopeNorm

NAVY="#16324F"; TEAL="#1B6B5A"; RUST="#B23A2E"; GOLD="#C8881F"; GREY="#5A6470"; INK="#23292F"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8.2,"axes.edgecolor":GREY,
    "axes.linewidth":0.8,"xtick.color":INK,"ytick.color":INK,"text.color":INK,"axes.labelcolor":INK})

W=np.load("contraction.npz"); Dg=np.load("deep_transfer.npz")
g=W["g"]; accW=W["ACC"]; nsW=W["ns"]; rhos=W["rhos"]
errW_full=1-accW[0]; errW_scarce=1-accW[3]; addW=errW_scarce[0]+(errW_full-errW_full[0])
S=Dg["S"]; nsD=Dg["ns"]; cD=float(Dg["c_digits"]); cDlo=float(Dg["c_lo"]); cDhi=float(Dg["c_hi"])

fig,ax=plt.subplots(2,2,figsize=(7.5,6.2)); fig.subplots_adjust(hspace=0.46,wspace=0.42)
aA,aB,aC,aD=ax[0,0],ax[0,1],ax[1,0],ax[1,1]

# (a) WDBC joint certified margin surface
norm=TwoSlopeNorm(vmin=g.min(),vcenter=0.0,vmax=max(0.05,g.max()))
im=aA.imshow(g,cmap="RdBu",norm=norm,aspect="auto")
aA.set_xticks(range(len(rhos))); aA.set_xticklabels([f"{r:.2f}" for r in rhos])
aA.set_yticks(range(len(nsW))); aA.set_yticklabels([str(int(n)) for n in nsW])
aA.set_xlabel("contamination $\\rho$"); aA.set_ylabel("data $n$ (scarcity $\\rightarrow$)")
for i in range(len(nsW)):
    for j in range(len(rhos)):
        v=g[i,j]; aA.text(j,i,f"{v:.2f}",ha="center",va="center",fontsize=6.2,
                          color=("white" if abs(v)>1.4 else INK))
        if v>=0: aA.add_patch(Rectangle((j-.5,i-.5),1,1,fill=False,ec=TEAL,lw=1.9))
aA.set_title("a   Clinical / trees: certified margin contracts",fontsize=7.9,loc="left",color=NAVY,pad=6)
cb=fig.colorbar(im,ax=aA,fraction=0.046,pad=0.02); cb.ax.tick_params(labelsize=6); cb.set_label("$g$",fontsize=8); cb.outline.set_edgecolor(GREY)

# (b) WDBC super-additive error
aB.fill_between(rhos,addW,errW_scarce,where=(errW_scarce>=addW),color=RUST,alpha=0.13)
aB.plot(rhos,errW_full,"-o",color=NAVY,lw=1.6,ms=3.5,label="full ($n=320$)")
aB.plot(rhos,addW,"--",color=GREY,lw=1.3,label="additive ($n=80$)")
aB.plot(rhos,errW_scarce,"-o",color=RUST,lw=1.6,ms=3.5,label="scarce ($n=80$)")
aB.set_xlabel("contamination $\\rho$"); aB.set_ylabel("prediction error")
aB.set_title("b   Clinical / trees: super-additive ($c=0.27$)",fontsize=7.9,loc="left",color=NAVY,pad=6)
aB.legend(loc="upper left",fontsize=6.3,frameon=False,labelspacing=0.3,handlelength=1.5)
for s in ["top","right"]: aB.spines[s].set_visible(False)

# (c) deep model: SmoothGrad explanation stability degrades under stress
Sfull=S[0]; Sscarce=S[3]
aC.axhspan(0,0.15,color=TEAL,alpha=0.08)
aC.axhline(0.15,color=TEAL,lw=1.0,ls=":")
aC.plot(rhos,Sfull,"-o",color=NAVY,lw=1.6,ms=3.5,label="full ($n=180$)")
aC.plot(rhos,Sscarce,"-o",color=GOLD,lw=1.6,ms=3.5,label="scarce ($n=50$)")
aC.text(0.005,0.135,"certified region $S\\leq0.15$",fontsize=6.3,color=TEAL,va="top")
aC.set_xlabel("contamination $\\rho$"); aC.set_ylabel("SmoothGrad stability $S$")
aC.set_title("c   Vision / deep MLP: explanation certificate degrades",fontsize=7.9,loc="left",color=NAVY,pad=6)
aC.legend(loc="lower right",fontsize=6.3,frameon=False,labelspacing=0.3,handlelength=1.5)
for s in ["top","right"]: aC.spines[s].set_visible(False)

# (d) cross-domain transfer of interaction coefficient
labels=["clinical / trees\n(WDBC)","vision / deep\n(digits)"]
cs=[0.27,cD]; los=[0.18,cDlo]; his=[0.36,cDhi]; cols=[RUST,GOLD]
xpos=[0,1]
aD.axhline(0,color=GREY,lw=0.9,ls="--")
for x,c,lo,hi,col in zip(xpos,cs,los,his,cols):
    aD.errorbar(x,c,yerr=[[c-lo],[hi-c]],fmt="o",color=col,ms=7,capsize=5,lw=1.8,mec=INK,mew=0.6)
aD.set_xticks(xpos); aD.set_xticklabels(labels,fontsize=7)
aD.set_xlim(-0.5,1.5); aD.set_ylabel("interaction coefficient $c$")
aD.set_title("d   Law does not transfer across domains",fontsize=7.9,loc="left",color=NAVY,pad=6)
aD.text(0.5,0.30,"$c$ is domain-specific:\nsuper-additive vs additive",ha="center",fontsize=6.6,color=INK)
for s in ["top","right"]: aD.spines[s].set_visible(False)

fig.savefig("img/fig2_compound_transfer.png",dpi=200,bbox_inches="tight",facecolor="white")
print("wrote img/fig2_compound_transfer.png  Sfull",np.round(Sfull,3)," cD",round(cD,3),[round(cDlo,3),round(cDhi,3)])
