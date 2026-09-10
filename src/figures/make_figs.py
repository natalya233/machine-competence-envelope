import warnings, re, json, glob, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.linewidth":0.8})
NAVY="#1f3b63"; GREEN="#1a8a4a"; RED="#c0392b"; GREY="#555"

envA = np.load("_envA.npy", allow_pickle=True).item()
comp = np.load("_compound.npy", allow_pickle=True).item()
drift= np.load("_drift.npy", allow_pickle=True).item()["drift_auc"]
R    = json.load(open("real_results.json"))
gridA=envA["grid"]; ngA=envA["ngrid"]; tA=envA["times"]
STAB_THR=0.20; COV_T=0.88
MONLAB={1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May"}

URL=re.compile(r"https?://\S+"); cl=lambda s:URL.sub(" ",str(s)).replace("\\n"," ")
rows=[]
for f in glob.glob("climate/Climate_Dataset/Climate_CSV/*.csv"):
    try:
        d=pd.read_csv(f,sep=";",engine="python",on_bad_lines="skip",encoding="latin-1",
                      usecols=lambda c:c.strip() in ["Date","Post text","# upvotes"]); rows.append(d)
    except Exception: pass
B=pd.concat(rows,ignore_index=True); B.columns=[c.strip() for c in B.columns]
B["up"]=pd.to_numeric(B["# upvotes"],errors="coerce"); B["date"]=pd.to_datetime(B["Date"],errors="coerce",dayfirst=True)
B["text"]=B["Post text"].map(cl); B=B.dropna(subset=["up","date","text"]); B=B[B["text"].str.len()>10]
B["year"]=B["date"].dt.year; B=B[B["year"].between(2017,2023)].copy()
blo,bhi=B["up"].quantile([1/3,2/3]); B=B[(B["up"]<=blo)|(B["up"]>=bhi)].copy(); B["y"]=(B["up"]>=bhi).astype(int)
vecB=TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),min_df=5,max_features=6000,sublinear_tf=True)
vecB.fit(B["text"].sample(20000,random_state=0))

def lin_profile(m,std): return np.abs(m.coef_.ravel())*std
def cdist(u,v):
    nu,nv=np.linalg.norm(u),np.linalg.norm(v)
    return 1.0 if nu==0 or nv==0 else 1.0-float(np.dot(u,v)/(nu*nv))
def fit_eval(Xtr,ytr,Xcal,ycal,Xev,yev,std,ref,alpha=0.10):
    m=LogisticRegression(max_iter=300,C=4.0).fit(Xtr,ytr)
    pc=m.predict_proba(Xcal); s=1-pc[np.arange(len(ycal)),ycal]
    n=len(s); k=int(np.ceil((n+1)*(1-alpha))); q=np.sort(s)[min(k,n)-1]
    pe=m.predict_proba(Xev); inset=(1-pe)<=q
    cover=float(inset[np.arange(len(yev)),yev].mean())
    pred=pe.argmax(1); conf=pe.max(1)
    return dict(cover=cover,acc=float((pred==yev).mean()),
                confwrong=float(((conf>=.8)&(pred!=yev)).mean()),
                stab=cdist(lin_profile(m,std),ref))
def envelope(df,vec,tcol,ref_t,times,ng,seeds=6):
    std=np.asarray(vec.transform(df["text"]).power(2).mean(0)).ravel()**0.5
    ref=df[df[tcol]==ref_t]; Xr=vec.transform(ref["text"]); yr=ref["y"].values
    ntr=int(len(ref)*0.7); base=LogisticRegression(max_iter=300,C=4.0).fit(Xr[:ntr],yr[:ntr])
    rp=lin_profile(base,std); g={}
    Xall={t:(vec.transform(df[df[tcol]==t]["text"]),df[df[tcol]==t]["y"].values) for t in times}
    for t in times:
        Xev,yev=Xall[t]
        for nn in ng:
            cs=[];st=[];ac=[];cw=[]
            for s in range(seeds):
                rs=np.random.default_rng(100*s+nn); idx=rs.permutation(len(ref))
                tr=idx[:nn]; cal=idx[nn:nn+800]
                if len(cal)<100: continue
                r=fit_eval(Xr[tr],yr[tr],Xr[cal],yr[cal],Xev,yev,std,rp)
                cs.append(r["cover"]);st.append(r["stab"]);ac.append(r["acc"]);cw.append(r["confwrong"])
            g[(t,nn)]=dict(cover=np.mean(cs),stab=np.mean(st),acc=np.mean(ac),confwrong=np.mean(cw))
    return g
tB=[2019,2020,2021,2022,2023]; ngB=[300,600,1200,2400,4000]
gridB=envelope(B,vecB,"year",2019,tB,ngB)

# ─── heatmap helper ───────────────────────────────────────────────────────────
def draw_env(ax, grid, times, ng, xlabels, title, xaxis_label, drift_vals=None):
    cov  = np.array([[grid[(t,n)]["cover"] for t in times] for n in ng])
    stab = np.array([[grid[(t,n)]["stab"]  for t in times] for n in ng])
    im = ax.imshow(cov, cmap="RdYlGn", vmin=0.80, vmax=0.95,
                   aspect="auto", origin="lower")
    for i, n in enumerate(ng):
        for j, t in enumerate(times):
            certified = cov[i,j] >= COV_T and stab[i,j] <= STAB_THR
            # coverage: centred, bold
            ax.text(j, i+0.18, f"{cov[i,j]:.2f}",
                    ha="center", va="center", fontsize=8, fontweight="bold",
                    color="#111")
            # S: smaller, bottom of cell, muted
            ax.text(j, i-0.28, f"S={stab[i,j]:.2f}",
                    ha="center", va="center", fontsize=6.2, color="#444")
            if certified:
                ax.add_patch(Rectangle((j-0.5, i-0.5), 1, 1,
                             fill=False, edgecolor=GREEN, lw=2.2))
    # x-tick labels: month/year + δ on second line
    if drift_vals:
        xlabels2 = [f"{xl}\nδ={drift_vals[t]:.2f}"
                    for xl, t in zip(xlabels, times)]
    else:
        xlabels2 = xlabels
    ax.set_xticks(range(len(times)))
    ax.set_xticklabels(xlabels2, fontsize=8)
    ax.set_yticks(range(len(ng)))
    ax.set_yticklabels([f"{n:,}" for n in ng], fontsize=8)
    ax.set_xlabel(xaxis_label, labelpad=4)
    ax.set_ylabel("training size  n", labelpad=4)
    ax.set_title(title, fontweight="bold", fontsize=9.5, color=NAVY, loc="left", pad=6)
    return im

# ═══════════ FIGURE 1 ════════════════════════════════════════════════════════
fig = plt.figure(figsize=(12.0, 5.0))
gs  = fig.add_gridspec(1, 2, width_ratios=[1.30, 1.0], wspace=0.34)

ax1 = fig.add_subplot(gs[0, 0])
im  = draw_env(ax1, gridA, tA, ngA, [MONLAB[t] for t in tA],
               "a   Telegram (information integrity): competence envelope",
               "evaluation month  (temporal drift →)",
               drift_vals=drift)
cb  = fig.colorbar(im, ax=ax1, fraction=0.042, pad=0.02)
cb.set_label("conformal coverage", fontsize=8)
# legend entry only (no extra δ row — it's already in tick labels)

ax2 = fig.add_subplot(gs[0, 1])
cov_line = [gridA[(t, 2400)]["cover"]     for t in tA]
cw_line  = [gridA[(t, 2400)]["confwrong"] for t in tA]
acc_line = [gridA[(t, 2400)]["acc"]       for t in tA]
x = range(len(tA))
ax2.plot(x, cov_line, "-o", color=NAVY, lw=2, label="conformal coverage")
ax2.axhline(0.90, ls="--", color=GREY, lw=1)
ax2.text(0.05, 0.905, "target 0.90", fontsize=7, color=GREY, va="bottom", ha="left")
ax2.fill_between(x, 0.78, 0.90,
                 where=[c < 0.90 for c in cov_line], color=RED, alpha=0.07)
ax2.set_ylim(0.78, 0.97)
ax2.set_ylabel("coverage", color=NAVY)
ax2.set_xticks(list(x))
ax2.set_xticklabels([MONLAB[t] for t in tA])
ax2.set_xlabel("evaluation month (temporal drift →)")
axr = ax2.twinx()
axr.plot(x, cw_line,  "-s", color=RED,   lw=1.6, ms=4, label="confident-wrong rate")
axr.plot(x, [1-a for a in acc_line], ":", color="#999", lw=1.4, label="error rate")
axr.set_ylabel("confident-wrong / error rate", color=RED)
axr.set_ylim(0, 0.46)
ax2.set_title("b   Early warning at fixed n = 2,400",
              fontweight="bold", fontsize=9.5, color=NAVY, loc="left", pad=6)
l1, la1 = ax2.get_legend_handles_labels()
l2, la2 = axr.get_legend_handles_labels()
ax2.legend(l1+l2, la1+la2, loc="upper right", fontsize=7, framealpha=0.95, borderpad=0.4)

fig.savefig("fig_real1.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("fig_real1 saved")

# ═══════════ FIGURE 2 ════════════════════════════════════════════════════════
fig = plt.figure(figsize=(13.4, 4.6))
gs  = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.05, 1.05], wspace=0.62)

axa = fig.add_subplot(gs[0, 0])
im  = draw_env(axa, gridB, tB, ngB, [str(y) for y in tB],
               "a   Reddit (climate): envelope replicates",
               "evaluation year  (temporal drift →)")
fig.colorbar(im, ax=axa, fraction=0.042, pad=0.02).set_label("coverage", fontsize=8)

axb = fig.add_subplot(gs[0, 1])
ng_c  = comp["ng"]; rho_c = comp["rho"]; resA = comp["resA"]
err   = np.array([[resA[(n, r)]["err"] for r in rho_c] for n in ng_c])
im2   = axb.imshow(err, cmap="magma_r", aspect="auto", origin="lower",
                   vmin=0.18, vmax=0.45)
for i, n in enumerate(ng_c):
    for j, r in enumerate(rho_c):
        axb.text(j, i, f"{err[i,j]:.2f}", ha="center", va="center", fontsize=8,
                 color="white" if err[i,j] > 0.33 else "#111")
axb.set_xticks(range(len(rho_c)))
axb.set_xticklabels([f"{int(r*100)}%" for r in rho_c])
axb.set_yticks(range(len(ng_c)))
axb.set_yticklabels([str(n) for n in ng_c])
axb.set_xlabel("label contamination  ρ")
axb.set_ylabel("training size n")
axb.set_title("b   Compound stress (Telegram)",
              fontweight="bold", fontsize=9.5, color=NAVY, loc="left")
fig.colorbar(im2, ax=axb, fraction=0.042, pad=0.02).set_label("prediction error", fontsize=8)

axc = fig.add_subplot(gs[0, 2])
names = ["Telegram\n(text)", "Reddit\n(text)", "Clinical\n(tabular)"]
cs  = [R["A_err_c"], R["B_err_c"], R["C_err_c"]]
cis = [R["A_err_ci"], R["B_err_ci"], R["C_err_ci"]]
yp  = np.arange(len(names))[::-1]
axc.errorbar(cs, yp,
             xerr=[[cs[i]-cis[i][0] for i in range(3)],
                   [cis[i][1]-cs[i] for i in range(3)]],
             fmt="o", color=NAVY, capsize=4, ms=7, lw=1.6)
axc.axvline(0, ls="--", color=GREY, lw=1)
axc.axvspan(-0.05, 0.05, color=GREEN, alpha=0.08)
axc.set_yticks(yp); axc.set_yticklabels(names, fontsize=8)
axc.set_xlabel("scarcity × contamination interaction  c\n(> 0 super-additive)")
axc.set_xlim(-0.20, 0.20); axc.set_ylim(-0.7, 2.7)
axc.text(0.0, 2.62, "near-additive across domains", ha="center", va="top",
         fontsize=7.5, color=GREY)
axc.set_title("c   Scarcity × contamination interaction",
              fontweight="bold", fontsize=9.5, color=NAVY, loc="left", pad=8)
axc.grid(axis="x", alpha=0.15, lw=0.5)
for i in range(3):
    axc.text(cs[i], yp[i]+0.22, f"{cs[i]:+.2f}",
             ha="center", va="bottom", fontsize=8, color=NAVY, fontweight="bold")

fig.savefig("fig_real2.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("fig_real2 saved")
json.dump({"gridB":{f"{t}_{n}":{k:float(v) for k,v in gridB[(t,n)].items()}
           for t in tB for n in ngB}}, open("_gridB.json","w"))
print("done")
