import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.linewidth":0.8})
NAVY="#1f3b63"; TEAL="#1a8a8a"; ORANGE="#d4773b"; GREY="#888"; RED="#c0392b"; GREEN="#1a8a4a"

# ================= Fig: adversarial (pillar 2) =================
P=json.load(open("pillar2.json"))
fr=[r["frac"] for r in P]; att=[r["attack"] for r in P]; cov=[r["cover"] for r in P]
acc=[r["acc_val"] for r in P]; S=[r["S"] for r in P]
fig,axes=plt.subplots(1,2,figsize=(11.6,3.9))
ax=axes[0]
ax.plot(fr,att,"-o",color=RED,lw=2,label="attack success")
ax.plot(fr,acc,"-s",color=GREY,lw=1.6,ms=4,label="validation accuracy (rises)")
ax.plot(fr,cov,"--",color=GREY,lw=1.4,label="conformal coverage (flat)")
ax.set_xlabel("attack strength ρ  (fraction of amplified-class training posts carrying the planted cue)")
ax.set_ylabel("rate"); ax.set_ylim(0,1.0); ax.grid(alpha=0.25,lw=0.5)
ax.set_title("a   Prediction-side certificates stay green under capture",fontweight="bold",fontsize=9,color=NAVY,loc="left",pad=6)
ax.legend(fontsize=7.5,loc="lower right",framealpha=0.95)
ax2=axes[1]
ax2.plot(fr,S,"-o",color=NAVY,lw=2,label="explanation-stability drift S")
ax2.axhline(S[0],ls=":",color=GREY,lw=1); ax2.text(0.02,S[0]+0.003,"clean baseline",fontsize=7,color=GREY)
axr=ax2.twinx(); axr.plot(fr,att,"-o",color=RED,lw=1.2,ms=3,alpha=0.5)
axr.set_ylabel("attack success",color=RED); axr.set_ylim(0,1)
ax2.set_xlabel("attack strength ρ"); ax2.set_ylabel("explanation drift S",color=NAVY)
ax2.set_title("b   Explanation certificate tracks the capture (r = 0.79)",fontweight="bold",fontsize=9,color=NAVY,loc="left",pad=6)
ax2.grid(alpha=0.25,lw=0.5); ax2.legend(fontsize=7.5,loc="upper left")
fig.savefig("fig_adversarial.png",dpi=200,bbox_inches="tight"); plt.close(fig)
print("fig_adversarial saved")

# ================= Fig: multimodal (3 panels incl. MOSI) =================
D=json.load(open("multimodal.json"))
M=json.load(open("mosi_results.json"))
fig,axes=plt.subplots(1,3,figsize=(13.6,4.0))
panels=[(axes[0],"a   Reddit climate (text + tabular)",D["res1"]["deg"],
         [("clean","clean"),("text_out","text\noutage"),("tab_out","tabular\noutage")]),
        (axes[1],"b   Telegram kiev1 (text + behavioural)",D["res2"]["deg"],
         [("clean","clean"),("text_out","text\noutage"),("tab_out","tabular\noutage")]),
        (axes[2],"c   CMU-MOSI (language + acoustic-visual)",M,
         [("clean","clean"),("text_out","language\noutage"),("av_out","AV\noutage")])]
methods=[("naive","naïve fusion",GREY),("conf","confidence-gated",ORANGE),("comp","competence-gated",NAVY)]
from matplotlib.lines import Line2D
for ax,title,r,scen in panels:
    x=np.arange(len(scen)); w=0.25
    for mi,(mk,ml,col) in enumerate(methods):
        vals=[r[s[0]][mk] for s in scen]
        ax.bar(x+(mi-1)*w,vals,w,color=col,zorder=3)
        for xi,v in zip(x,vals): ax.text(xi+(mi-1)*w,v+0.014,f"{v:.2f}",ha="center",fontsize=7.4,color="#1a1a1a",fontweight="bold")
    orc=[r[s[0]]["oracle"] for s in scen]
    for xi,v in zip(x,orc):
        ax.hlines(v, xi-0.42, xi+0.42, color=GREEN, lw=2.4, zorder=4)
    ax.set_xticks(x); ax.set_xticklabels([s[1] for s in scen],fontsize=7.5)
    ax.set_ylabel("deployment error"); ax.set_ylim(0,0.66)
    ax.set_title(title,fontweight="bold",fontsize=9.3,color=NAVY,loc="left",pad=6)
    ax.grid(axis="y",alpha=0.25,lw=0.5)
# single shared legend above all panels (no suptitle -> room at top)
handles=[plt.Rectangle((0,0),1,1,color=c) for _,_,c in methods]+[Line2D([0],[0],color=GREEN,lw=2.4)]
labels=[ml for _,ml,_ in methods]+["single-best-modality oracle"]
fig.legend(handles,labels,loc="upper center",ncol=4,bbox_to_anchor=(0.5,1.05),
           fontsize=8,frameon=False,handlelength=1.4,columnspacing=1.4)
fig.savefig("fig_multimodal.png",dpi=200,bbox_inches="tight"); plt.close(fig)
print("fig_multimodal saved (3 panels)")
