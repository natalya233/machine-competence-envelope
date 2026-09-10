import numpy as np, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.linewidth":0.8})
NAVY="#1f3b63"; GREEN="#1a8a4a"; RED="#c0392b"; AMBER="#d9a521"; GREY="#777"

# 17 Irpin bridges (Kopiika et al.): gamma_pre(TP1), CCD_LOC, LoK, DL
B={
"B1":(0.829,0.632,"H","High"),"B2":(0.967,0.540,"H","High"),"B3":(0.651,0.384,"M","Moderate"),
"B4":(0.376,0.241,"L","NA"),"B5":(0.652,0.387,"M","Moderate"),"B6":(0.889,0.390,"H","Low"),
"B7":(0.570,0.156,"M","Low"),"B8":(0.436,-0.115,"L","NA"),"B9":(0.890,0.730,"H","High"),
"B10":(0.469,-0.145,"L","NA"),"B11":(0.588,0.280,"L","NA"),"B12":(0.526,0.189,"L","NA"),
"B13":(0.505,0.178,"L","NA"),"B14":(0.505,0.062,"L","NA"),"B15":(0.683,0.400,"M","High"),
"B16":(0.567,0.259,"L","NA"),"B17":(0.941,0.521,"H","Moderate"),
}
GP=0.55        # reliability threshold on pre-event coherence (LoK boundary)
CCDT=0.25      # naive "damaged" decision threshold on CCD

names=list(B); gp=np.array([B[k][0] for k in names]); ccd=np.array([B[k][1] for k in names])
lok=np.array([B[k][2] for k in names]); dl=np.array([B[k][3] for k in names])

# naive CCD-threshold decision vs competence-gated
naive_damaged=[k for k in names if B[k][1]>=CCDT]
low_reliability=[k for k in names if B[k][0]<GP]
silent_misreads=[k for k in naive_damaged if B[k][2]=="L"]   # naive "damaged" but LoK Low -> defer
certified_damaged=[k for k in naive_damaged if B[k][2] in ("H","M")]
print("naive CCD>=%.2f 'damaged':"%CCDT, naive_damaged, "(n=%d)"%len(naive_damaged))
print("of those, LOW-reliability (certificate defers):", silent_misreads)
print("certified damaged:", certified_damaged)
# borderline: high damage-level call at only medium reliability
borderline=[k for k in names if B[k][2]=="M" and B[k][3]=="High"]
print("high-damage verdict at only MEDIUM reliability (flag for lower certified confidence):", borderline)

fig,ax=plt.subplots(figsize=(8.8,5.4))
ax.axvspan(GP,1.0,color=GREEN,alpha=0.05)
ax.axhline(CCDT,ls=":",color=GREY,lw=1)
ax.text(0.995,CCDT+0.016,"naive 'damaged' threshold (CCD = 0.25)",fontsize=7,color=GREY,ha="right")
ax.axvline(GP,ls="--",color=NAVY,lw=1.2)
ax.text(GP-0.010,0.76,"reliability\nboundary",fontsize=7,color=NAVY,va="top",ha="right")
col={"H":GREEN,"M":AMBER,"L":RED}
# per-label offsets to separate near-coincident bridges (B3/B5, B12/B13, B8/B10, B11/B16, B7)
offs={"B5":(5,-11),"B3":(5,4),"B13":(-16,-3),"B12":(5,4),"B10":(5,-11),"B8":(5,4),
      "B16":(-18,-3),"B11":(5,5),"B7":(5,-11),"B14":(5,4),"B17":(-20,3),"B2":(5,4)}
for k in names:
    g,c,l,d=B[k]
    ax.scatter(g,c,s=95,color=col[l],edgecolor="white",zorder=4,lw=0.8)
    dx,dy=offs.get(k,(5,4))
    ax.annotate(k,(g,c),xytext=(dx,dy),textcoords="offset points",fontsize=6.8,color="#222",zorder=6)
for k in silent_misreads:
    g,c,_,_=B[k]; ax.scatter(g,c,s=280,facecolor="none",edgecolor=RED,lw=2,zorder=3)
for k in borderline:
    g,c,_,_=B[k]; ax.scatter(g,c,s=280,facecolor="none",edgecolor=AMBER,lw=2,zorder=3)
ax.set_xlabel("pre-event coherence  γ  (data reliability / level of knowledge →)")
ax.set_ylabel("coherent change  CCD$_{LOC}$  (apparent damage →)")
from matplotlib.lines import Line2D
leg=[Line2D([0],[0],marker='o',color='w',markerfacecolor=col['H'],markersize=9,label='LoK High'),
     Line2D([0],[0],marker='o',color='w',markerfacecolor=col['M'],markersize=9,label='LoK Medium'),
     Line2D([0],[0],marker='o',color='w',markerfacecolor=col['L'],markersize=9,label='LoK Low'),
     Line2D([0],[0],marker='o',color='w',markerfacecolor='none',markeredgecolor=RED,markersize=13,label='confident misread deferred'),
     Line2D([0],[0],marker='o',color='w',markerfacecolor='none',markeredgecolor=AMBER,markersize=13,label='high-damage call, medium reliability')]
# legend OUTSIDE the plot (right) so it never overlaps points
ax.legend(handles=leg,fontsize=7.5,loc="upper left",bbox_to_anchor=(1.015,1.0),
          framealpha=0.95,borderaxespad=0)
ax.set_xlim(0.33,1.0); ax.set_ylim(-0.2,0.82); ax.grid(alpha=0.2,lw=0.5)
fig.savefig("fig_bridges.png",dpi=200,bbox_inches="tight"); plt.close(fig)
json.dump(dict(naive_damaged=naive_damaged,silent_misreads=silent_misreads,
               certified_damaged=certified_damaged,borderline=borderline,GP=GP,CCDT=CCDT),
          open("pillar3.json","w"),indent=2)
print("saved fig_bridges.png")
