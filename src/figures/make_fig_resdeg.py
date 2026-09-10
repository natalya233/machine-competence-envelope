import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.linewidth":0.8})
NAVY="#1f3b63"; GREEN="#1a8a4a"; RED="#c0392b"; GREY="#777"; TEAL="#1a8a8a"

D=json.load(open("resdeg.json"))
prune=D["prune"]; cal=D["cal"]; bw=D["bw"]

fig=plt.figure(figsize=(13.6, 4.4))
gs=fig.add_gridspec(1,3,width_ratios=[1.18,1.02,0.88],wspace=0.46)

# ── (a) Pruning × Scarcity ────────────────────────────────────────────────
axa=fig.add_subplot(gs[0,0])
n_vals=[300,600,1200,2400,4000]
p_vals=[0.0,0.3,0.5,0.7,0.85,0.92]
colors=plt.cm.Blues(np.linspace(0.30,0.95,len(p_vals)))
COV_THR=0.88; S_THR=0.20

for pi,p in enumerate(p_vals):
    rows=sorted([r for r in prune if r["p"]==p],key=lambda x:x["n"])
    Ss=[r["S"] for r in rows]
    lw=2.2 if p in (0.0,0.7,0.92) else 1.1
    axa.plot(n_vals,Ss,"-o",color=colors[pi],lw=lw,ms=3.5,label=f"p={p:.2f}")

axa.axhline(S_THR,ls="--",color=RED,lw=1.4)
axa.text(4200,S_THR+0.01,"S limit\n(0.20)",fontsize=6.8,color=RED,va="bottom",ha="right")

# mark first exceedance
for pi,p in enumerate(p_vals):
    rows=sorted([r for r in prune if r["p"]==p],key=lambda x:x["n"])
    for r in rows:
        if r["S"]>S_THR:
            axa.scatter(r["n"],r["S"],s=60,color=RED,zorder=5,marker="x",lw=1.8)
            break

axa.set_xlabel("training size  n  (scarcity →)")
axa.set_ylabel("explanation-stability drift  S")
axa.set_xscale("log")
axa.set_xticks(n_vals); axa.set_xticklabels([str(n) for n in n_vals],fontsize=7.5)
axa.set_ylim(0.0, 0.78)
axa.set_title("a  Pruning (resource) × scarcity:\nS exceeds limit before coverage fails",
              fontweight="bold",fontsize=9,color=NAVY,loc="left",pad=5)
# legend: lower left (away from high-S lines at small n)
axa.legend(title="pruning p",fontsize=6.8,title_fontsize=7,
           loc="lower left",ncol=2,framealpha=0.92)

# coverage reference (right axis) — two dotted lines only, no extra ticks
axr=axa.twinx()
for pi,p in enumerate([0.0,0.92]):
    rows=sorted([r for r in prune if r["p"]==p],key=lambda x:x["n"])
    axr.plot(n_vals,[r["cover"] for r in rows],":",
             color=colors[[0.0,0.92].index(p)*5],lw=1,alpha=0.45)
axr.axhline(COV_THR,ls=":",color=GREY,lw=0.8)
axr.set_ylabel("coverage (dotted)",color=GREY,fontsize=7,labelpad=2)
axr.set_ylim(0.80,0.96); axr.tick_params(labelsize=7,pad=2)

# ── (b) Calibration buffer × Pruning ──────────────────────────────────────
axb=fig.add_subplot(gs[0,1])
cal_ns=[20,50,100,200,400,800]; p3=[0.0,0.5,0.85]
col3=[GREEN,TEAL,RED]; lab3=["full model","p=0.50","p=0.85 (heavy)"]
for i,p in enumerate(p3):
    rows=sorted([r for r in cal if r["p"]==p],key=lambda x:x["cal_n"])
    axb.plot(cal_ns,[r["cover"] for r in rows],"-o",color=col3[i],lw=1.8,ms=4,label=lab3[i])
axb.axhline(COV_THR,ls="--",color=GREY,lw=1.2)
axb.text(820,COV_THR+0.003,"target 0.88",fontsize=7,color=GREY,ha="right")
axb.set_xlabel("calibration buffer size\n(memory constraint ←,  more memory →)")
axb.set_ylabel("conformal coverage")
axb.set_xscale("log")
axb.set_xticks(cal_ns); axb.set_xticklabels([str(n) for n in cal_ns],fontsize=7.5)
axb.set_ylim(0.78,0.94)
axb.set_title("b  Memory constraint:\ncoverage unreliable when buffer < 100",
              fontweight="bold",fontsize=9,color=NAVY,loc="left",pad=5)
axb.legend(fontsize=7.5,loc="lower right",framealpha=0.92)

# ── (c) Feature bandwidth MOSI ────────────────────────────────────────────
axc=fig.add_subplot(gs[0,2])
fracs=[b["frac"] for b in bw]; errs=[b["err"] for b in bw]
axc.plot([f*100 for f in fracs],errs,"-o",color=NAVY,lw=2,ms=6)

# data labels: alternate above/below to avoid overlap
for idx,(f,e) in enumerate(zip(fracs,errs)):
    dy = +0.004 if idx%2==0 else -0.008
    axc.text(f*100, e+dy, f"{e:.3f}", fontsize=7.5, color=NAVY,
             ha="center", va="bottom" if dy>0 else "top")

axc.axhline(errs[-1],ls=":",color=GREY,lw=1)
axc.text(98, errs[-1]+0.003,"full\nbandwidth",fontsize=6.8,color=GREY,ha="right",va="bottom")
axc.fill_between([f*100 for f in fracs],errs,errs[-1],alpha=0.12,color=RED)
# split long x-label onto two lines
axc.set_xlabel("feature channels kept (% of full)\n← resource constraint")
axc.set_ylabel("deployment error\n(CMU-MOSI AV modality)")
axc.set_xlim(8,108); axc.set_ylim(0.40,0.54)
axc.set_title("c  Channel bandwidth\n(CMU-MOSI acoustic-visual)",
              fontweight="bold",fontsize=9,color=NAVY,loc="left",pad=5)
axc.grid(alpha=0.2,lw=0.5)

fig.savefig("fig_resdeg.png",dpi=200,bbox_inches="tight")
plt.close(fig)
print("saved fig_resdeg.png")
