import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.linewidth":0.8})
NAVY="#1f3b63"; TEAL="#1a8a8a"; ORANGE="#d4773b"; GREY="#888"; GREEN="#1a8a4a"
D=json.load(open("multimodal.json"))
fig,axes=plt.subplots(1,2,figsize=(12.4,4.2))
for ax,key,title in [(axes[0],"res1","a   Reddit climate  (text + author/community tabular)"),
                     (axes[1],"res2","b   Telegram kiev1  (text + behavioural)")]:
    r=D[key]["deg"]; scen=[("clean","clean"),("text_out","text outage"),("tab_out","tabular outage")]
    methods=[("naive","naïve late fusion",GREY),("conf","confidence-gated",ORANGE),("comp","competence-gated",NAVY)]
    x=np.arange(len(scen)); w=0.26
    for mi,(mk,ml,col) in enumerate(methods):
        vals=[r[s[0]][mk] for s in scen]
        ax.bar(x+(mi-1)*w,vals,w,color=col,label=ml,zorder=3)
        for xi,v in zip(x,vals): ax.text(xi+(mi-1)*w,v+0.006,f"{v:.2f}",ha="center",fontsize=6.6,color=col)
    # single-modality reference markers
    for xi,s in zip(x,scen):
        ax.plot([xi-0.42,xi+0.42],[r[s[0]]["text"]]*2,"--",color="#b0b0b0",lw=1,zorder=2)
        ax.plot([xi-0.42,xi+0.42],[r[s[0]]["tab"]]*2,":",color="#b0b0b0",lw=1,zorder=2)
    ax.set_xticks(x); ax.set_xticklabels([s[1] for s in scen])
    ax.set_ylabel("deployment error"); ax.set_ylim(0,0.55)
    ax.set_title(title,fontweight="bold",fontsize=9.5,color=NAVY,loc="left")
    ax.grid(axis="y",alpha=0.25,lw=0.5)
    if key=="res1": ax.legend(fontsize=7.5,loc="upper left",framealpha=0.95)
axes[0].text(0.01,-0.22,"Dashed = text-only, dotted = tabular-only (single modality). Competence gate uses label-free per-modality drift; confidence gate is fooled by confidently-wrong degraded modalities.",
             transform=axes[0].transAxes,fontsize=6.8,color=GREY)
fig.suptitle("Figure | Competence-gated late fusion is robust to modality-specific degradation, where naïve and confidence-gated fusion fail",
             fontsize=10.3,fontweight="bold",y=1.01)
fig.savefig("fig_multimodal.png",dpi=200,bbox_inches="tight"); plt.close(fig)
print("saved fig_multimodal.png")
