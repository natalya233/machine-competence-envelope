# Renders Paper Fig. 3 (schematic competence-gating) -> img/fig3_infrastructure.png
# Part of the open code for the Machine competence Perspective. MIT licence.
import os; os.makedirs("img", exist_ok=True)
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon, Circle
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["DejaVu Sans","Arial"],
                     "savefig.dpi":200,"figure.dpi":200})
NAVY="#16324F"; TEAL="#1B6B5A"; RUST="#B23A2E"; GOLD="#C8881F"; GREY="#5A6470"
LGREEN="#E4EFE8"; LRED="#F7E5E1"; HDR="#1B2A38"

fig=plt.figure(figsize=(12.6,6.3))
gs=fig.add_gridspec(1,2,width_ratios=[1.62,1.0],wspace=0.12)

# ================= Panel a: competence-gated escalation ladder =================
axA=fig.add_subplot(gs[0,0]); axA.set_xlim(0,100); axA.set_ylim(0,100); axA.axis("off")
axA.text(0,99,"a",fontsize=16,fontweight="bold")
axA.text(50,98,"Competence-gated tiered assessment: 17 Irpin-river bridges",
         ha="center",fontsize=12,color=NAVY,fontweight="bold")

tiers=[
 ("R","REGIONAL  \u00b7  network","Sentinel-1 SAR \u00b7 InSAR coherent-change detection","signal: coherence \u03b3  0.9 \u2192 0.5 after shelling"),
 ("A","ASSET  \u00b7  bridge","asset-scale CCD on the 17 Irpin-river bridges","open low-res imagery, trustworthy only with caution"),
 ("C","COMPONENT","high-res open images \u00b7 semantic segmentation","occlusion inpainting \u00b7 instance damage masks"),
]
ys=[80,52,24]; boxL=2; boxW=50; boxH=18
for (tag,title,m1,m2),y in zip(tiers,ys):
    axA.add_patch(FancyBboxPatch((boxL,y-boxH/2),boxW,boxH,boxstyle="round,pad=0.3,rounding_size=2",
                 fc="white",ec=NAVY,lw=1.5))
    axA.add_patch(Circle((boxL+5.5,y+boxH/2-5),3.6,fc=NAVY,ec="none"))
    axA.text(boxL+5.5,y+boxH/2-5,tag,ha="center",va="center",color="white",fontsize=12,fontweight="bold")
    axA.text(boxL+11,y+boxH/2-3.4,title,fontsize=11,color=NAVY,fontweight="bold",va="center")
    axA.text(boxL+5,y-1.0,m1,fontsize=8.6,color="#33424b",va="center")
    axA.text(boxL+5,y-5.6,m2,fontsize=8.3,color=GREY,va="center",style="italic")

# decision gates (diamonds) to the right of each box
gx=63
def diamond(cx,cy,s=5.2,fc="white",ec=GOLD):
    pts=[(cx,cy+s),(cx+s,cy),(cx,cy-s),(cx-s,cy)]
    axA.add_patch(Polygon(pts,closed=True,fc=fc,ec=ec,lw=1.8))
dec_nodes_y=[]
for i,y in enumerate(ys):
    # arrow box -> gate
    axA.add_patch(FancyArrowPatch((boxL+boxW,y),(gx-5.4,y),arrowstyle="-|>",mutation_scale=12,color="#3a4750",lw=1.5))
    diamond(gx,y)
    axA.text(gx,y,"cert",ha="center",va="center",fontsize=6.6,color=GOLD,fontweight="bold")
    # green: inside envelope -> decision (to the right)
    axA.add_patch(FancyArrowPatch((gx+5.2,y),(86,y),arrowstyle="-|>",mutation_scale=12,color=TEAL,lw=2.0))
    axA.add_patch(FancyBboxPatch((86,y-4),12.5,8,boxstyle="round,pad=0.2,rounding_size=1.5",fc=LGREEN,ec=TEAL,lw=1.3))
    axA.text(92.2,y,"certify\n& decide",ha="center",va="center",fontsize=7.8,color=TEAL,fontweight="bold")
    axA.text(75,y+2.2,"inside C",fontsize=7.6,color=TEAL,ha="center")
    # red: outside envelope -> escalate (down) or defer (last)
    if i<2:
        axA.add_patch(FancyArrowPatch((gx,y-5.4),(gx,ys[i+1]+9.6),arrowstyle="-|>",mutation_scale=12,color=RUST,lw=2.0,
                     connectionstyle="arc3,rad=0"))
        axA.text(gx+1.5,(y-5.4+ys[i+1]+9.6)/2,"outside \u2192 escalate",fontsize=7.8,color=RUST,ha="left",va="center",rotation=90)
    else:
        axA.add_patch(FancyArrowPatch((gx,y-5.4),(gx,7.5),arrowstyle="-|>",mutation_scale=12,color=RUST,lw=2.0))
        axA.add_patch(FancyBboxPatch((40,2.5),46,7,boxstyle="round,pad=0.2,rounding_size=1.5",fc=LRED,ec=RUST,lw=1.3))
        axA.text(63,6.0,"outside envelope \u2192 defer to engineer / physical inspection",
                 ha="center",va="center",fontsize=8.4,color=RUST,fontweight="bold")

# gate legend (bottom-left)
axA.text(2,12.0,"\u25C6  competence-certificate gate",fontsize=8.2,color=GOLD,fontweight="bold")
axA.text(2,7.6,"inside envelope \u2192 certify   \u00b7   outside \u2192 escalate / defer",fontsize=7.6,color="#33424b")

# ================= Panel b: Omega stress radar =================
axB=fig.add_subplot(gs[0,1]); axB.set_xlim(-1.35,1.35); axB.set_ylim(-1.5,1.5); axB.axis("off"); axB.set_aspect("equal")
axB.text(-1.33,1.46,"b",fontsize=16,fontweight="bold",transform=axB.transData)
axB.text(0,1.36,"Operating point in \u03a9",ha="center",fontsize=12,color=NAVY,fontweight="bold")

axes_lbl=["drift\n(novel damage patterns)","scarcity\n(class imbalance, few labels)",
          "contamination\n(occlusion, adversarial)","resource degradation\n(no power, low-res only)"]
ang=[np.pi/2, np.pi, 3*np.pi/2, 0]  # top,left,bottom,right
def xy(a,r): return r*np.cos(a), r*np.sin(a)
# grid rings
for rr in [0.33,0.66,1.0]:
    axB.add_patch(Circle((0,0),rr,fc="none",ec="#cdd4d8",lw=1))
for a in ang:
    x,y=xy(a,1.0); axB.plot([0,x],[0,y],color="#cdd4d8",lw=1)

# envelope polygon (certified region, inner)
env_r=[0.36,0.34,0.40,0.34]
envp=[xy(a,r) for a,r in zip(ang,env_r)]
axB.add_patch(Polygon(envp,closed=True,fc=TEAL,ec=TEAL,alpha=0.18,lw=2))
# conflict operating point (outer, near max)
op_r=[0.92,0.97,0.82,0.95]
opp=[xy(a,r) for a,r in zip(ang,op_r)]
axB.add_patch(Polygon(opp,closed=True,fc=RUST,ec=RUST,alpha=0.15,lw=2.2))
for a,r in zip(ang,op_r):
    x,y=xy(a,r); axB.plot(x,y,marker="o",ms=5,color=RUST)

# labels
for a,lbl in zip(ang,axes_lbl):
    x,y=xy(a,1.18)
    ha="center"; 
    if abs(np.cos(a))>0.5: ha="left" if np.cos(a)>0 else "right"
    axB.text(x,y,lbl,ha=ha,va="center",fontsize=8.0,color="#33424b")

axB.text(0,-1.30,"certified envelope",color=TEAL,fontsize=8.6,ha="center",fontweight="bold")
axB.text(0,-1.43,"conflict operating point, outside on every axis at once",color=RUST,fontsize=8.6,ha="center",fontweight="bold")

# certificate verdict tag
axB.add_patch(FancyBboxPatch((0.52,-0.64),0.82,0.34,boxstyle="round,pad=0.02,rounding_size=0.05",
              fc="white",ec=RUST,lw=1.3,transform=axB.transData))
axB.text(0.93,-0.40,"Cert: mem = OUTSIDE",ha="center",va="center",fontsize=7.0,color=RUST,fontweight="bold")
axB.text(0.93,-0.55,"\u03c0 = DEFER",ha="center",va="center",fontsize=7.0,color=RUST)

plt.savefig("img/fig3_infrastructure.png",bbox_inches="tight",facecolor="white")
from PIL import Image; im=Image.open("img/fig3_infrastructure.png"); print("fig4",im.size, round(im.size[1]/im.size[0],4))
