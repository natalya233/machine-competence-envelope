import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.linewidth":0.8})
NAVY="#1f3b63"; TEAL="#1a8a8a"; AMBER="#d9a521"; RED="#c0392b"; GREY="#666"
BKCOL={"distilbert":NAVY,"xlmr":TEAL,"minilm":AMBER}
BKLAB={"distilbert":"DistilBERT-ml","xlmr":"XLM-R","minilm":"MiniLM-ml"}
T=json.load(open("transformer_results.json")); H=json.load(open("hf_results.json"))
months=T["months"]; years=T["years"]; ng=T["n_grid_A"]
MON={1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May"}

fig=plt.figure(figsize=(13.6,4.1))
gs=fig.add_gridspec(1,3,width_ratios=[1,1,1],wspace=0.34)

# (a) Telegram coverage vs drift, 3 backbones
axa=fig.add_subplot(gs[0,0])
for bk in T["backbones"]:
    A=T["backbones"][bk]["A"]
    cov=[A.get(f"{t}__2400",{}).get("cover",np.nan) for t in months]
    axa.plot(range(len(months)),cov,"-o",color=BKCOL[bk],ms=4,lw=1.9,label=BKLAB[bk])
axa.axhline(0.88,ls="--",color=GREY,lw=1.2); axa.text(0.05,0.885,"coverage limit 0.88",fontsize=6.8,color=GREY)
axa.set_xticks(range(len(months))); axa.set_xticklabels([MON[m] for m in months])
axa.set_xlabel("evaluation month  (drift →)"); axa.set_ylabel("conformal coverage")
axa.set_ylim(0.80,0.96)
axa.set_title("a   Coverage under drift — Telegram\n(foundation-model embeddings, n=2,400)",
              fontweight="bold",fontsize=9,color=NAVY,loc="left")
axa.legend(fontsize=7.5,loc="lower left"); axa.grid(alpha=0.2,lw=0.5)

# (b) Telegram S vs scarcity, 3 backbones
axb=fig.add_subplot(gs[0,1])
for bk in T["backbones"]:
    A=T["backbones"][bk]["A"]
    S=[A.get(f"1__{n}",{}).get("S",np.nan) for n in ng]
    axb.plot(ng,S,"-o",color=BKCOL[bk],ms=4,lw=1.9,label=BKLAB[bk])
axb.axhline(0.20,ls="--",color=RED,lw=1.2); axb.text(ng[0],0.205,"S limit 0.20",fontsize=6.8,color=RED)
axb.set_xscale("log"); axb.set_xticks(ng); axb.set_xticklabels([str(n) for n in ng],fontsize=7.5)
axb.set_xlabel("training size n  (scarcity ←)"); axb.set_ylabel("explanation-stability drift  S")
axb.set_title("b   Stability under scarcity — Telegram\n(foundation-model embeddings, January)",
              fontweight="bold",fontsize=9,color=NAVY,loc="left")
axb.legend(fontsize=7.5,loc="upper right"); axb.grid(alpha=0.2,lw=0.5)

# (c) Reddit coverage vs drift, 3 backbones
axc=fig.add_subplot(gs[0,2])
for bk in T["backbones"]:
    B=T["backbones"][bk]["B"]
    cov=[B.get(f"{y}__2400",{}).get("cover",np.nan) for y in years]
    axc.plot(range(len(years)),cov,"-o",color=BKCOL[bk],ms=4,lw=1.9,label=BKLAB[bk])
axc.axhline(0.88,ls="--",color=GREY,lw=1.2); axc.text(0.05,0.885,"coverage limit 0.88",fontsize=6.8,color=GREY)
axc.set_xticks(range(len(years))); axc.set_xticklabels([str(y) for y in years],fontsize=8)
axc.set_xlabel("evaluation year  (drift →)"); axc.set_ylabel("conformal coverage")
axc.set_title("c   Cross-domain replication — Reddit\n(foundation-model embeddings, n=2,400)",
              fontweight="bold",fontsize=9,color=NAVY,loc="left")
axc.legend(fontsize=7.5,loc="upper right"); axc.grid(alpha=0.2,lw=0.5)

fig.savefig("fig_transformer.png",dpi=200,bbox_inches="tight"); plt.close(fig)
print("saved fig_transformer.png")
