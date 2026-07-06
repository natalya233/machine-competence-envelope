"""reviewer_addendum.py — обчислення на відповідь коментарям Stergios:
 [287] абляції монітора + bootstrap CI на Spearman
 [245][321] clean-certify poisoning + аудит dormancy cue + локалізація S на cue
 [313] bootstrap CI взаємодії (замість '-0.00')
 [136] calibration-scales-with-n варіант
"""
import warnings, re, glob, json, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.stats import spearmanr
UP="data"
URL=re.compile(r"https?://\S+"); cl=lambda s:URL.sub(" ",str(s)).replace("\\n"," ")
def cos_dist(u,v):
    nu,nv=np.linalg.norm(u),np.linalg.norm(v)
    return 1.0 if nu==0 or nv==0 else 1.0-float(np.dot(u,v)/(nu*nv))
def conf_cov(pc,yc,pe,ye,a=0.10):
    sc=1-pc[np.arange(len(yc)),yc]; q=np.sort(sc)[min(int(np.ceil((len(sc)+1)*(1-a))),len(sc))-1]
    return float(((1-pe)<=q)[np.arange(len(ye)),ye].mean())
OUT={}

# ---------- load Telegram ----------
A=pd.read_csv(f"{UP}/telegram_2026.csv",usecols=["date","text","views","forwards"],engine="python",on_bad_lines="skip")
A["date"]=pd.to_datetime(A["date"],errors="coerce",utc=True); A=A.dropna(subset=["text","views","forwards"])
A=A[(A["views"]>0)&(A["text"].str.len()>20)]; A["fr"]=A["forwards"]/A["views"]; A["text"]=A["text"].map(cl)
A["month"]=A["date"].dt.month; lo,hi=A["fr"].quantile([1/3,2/3])
A=A[(A["fr"]<=lo)|(A["fr"]>=hi)]; A["y"]=(A["fr"]>=hi).astype(int)
A=A[A["month"].isin([1,2,3,4,5])].reset_index(drop=True)
vec=TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),min_df=5,max_features=3000,sublinear_tf=True)
ref=A[A["month"]==1]; vec.fit(ref["text"].sample(min(10000,len(ref)),random_state=0))
X=vec.transform(A["text"]); y=A["y"].values; m=A["month"].values
Xref=X[m==1]; yref=y[m==1]; std=np.asarray(Xref.power(2).mean(0)).ravel()**0.5

# ========== [287] MONITOR ABLATION ==========
print("[287] monitor ablation ...", flush=True)
# fixed anchor model, evaluate each month; per-period signals
n_tr=4000; rng=np.random.default_rng(0); idx=rng.permutation(Xref.shape[0])
tr=idx[:n_tr]; cal=idx[n_tr:n_tr+800]
base=LogisticRegression(max_iter=300,C=4).fit(Xref[tr],yref[tr])
ref_prof=np.abs(base.coef_.ravel())*std
months=[1,2,3,4,5]
rows={"true_err":[],"conformal":[],"confidence":[],"drift":[],"explanation":[]}
# discriminator drift delta: AUC of Jan-vs-month classifier (unlabeled)
from sklearn.metrics import roc_auc_score
for t in months:
    Xe=X[m==t]; ye=y[m==t]
    pe=base.predict_proba(Xe); pred=pe.argmax(1); conf=pe.max(1)
    rows["true_err"].append(float((pred!=ye).mean()))
    # conformal nonconformity (label-free): mean of 1-max_prob
    rows["conformal"].append(float((1-conf).mean()))
    # confidence (label-free): mean max prob -> as risk use 1-conf too but different: use mean entropy
    ent=-(pe*np.log(pe+1e-12)).sum(1); rows["confidence"].append(float(ent.mean()))
    # drift delta: discriminator AUC Jan vs month
    if t==1: rows["drift"].append(0.5)
    else:
        Xj=Xref[rng.permutation(Xref.shape[0])[:Xe.shape[0]]]
        Xd=np.vstack([Xj.toarray(),Xe.toarray()]); yd=np.r_[np.zeros(Xj.shape[0]),np.ones(Xe.shape[0])]
        pi=rng.permutation(len(yd)); dsc=LogisticRegression(max_iter=200).fit(Xd[pi][:1500],yd[pi][:1500])
        rows["drift"].append(float(roc_auc_score(yd,dsc.predict_proba(Xd)[:,1])))
    # explanation drift S: retrain on month sample, profile drift
    rng2=np.random.default_rng(t); ii=rng2.permutation(Xe.shape[0])[:min(2400,Xe.shape[0])]
    mm=LogisticRegression(max_iter=300,C=4).fit(Xe[ii],ye[ii])
    rows["explanation"].append(cos_dist(np.abs(mm.coef_.ravel())*std,ref_prof))
te=np.array(rows["true_err"])
def boot_sp(x,yv,B=2000):
    x=np.array(x);yv=np.array(yv);n=len(x);rs=np.random.default_rng(1);out=[]
    base=spearmanr(x,yv)[0]
    for _ in range(B):
        s=rs.integers(0,n,n)
        if len(set(yv[s]))<2: continue
        out.append(spearmanr(x[s],yv[s])[0])
    lo,hi=np.nanpercentile(out,[2.5,97.5]); return base,lo,hi
abl={}
for k in ["conformal","confidence","drift","explanation"]:
    b,lo,hi=boot_sp(rows[k],te); abl[k]=dict(rho=b,lo=lo,hi=hi)
# joint = standardized sum of conformal + explanation
z=lambda a:(np.array(a)-np.mean(a))/(np.std(a)+1e-9)
joint=z(rows["conformal"])+z(rows["explanation"])
b,lo,hi=boot_sp(joint,te); abl["joint"]=dict(rho=b,lo=lo,hi=hi)
OUT["monitor_ablation"]={"periods":months,"true_err":rows["true_err"],"signals":rows,"spearman":abl}
print("  ",{k:round(v["rho"],2) for k,v in abl.items()})

# ========== [245][321] CLEAN-CERTIFY POISONING + CUE DORMANCY ==========
print("[245/321] clean-certify poisoning + cue audit ...", flush=True)
MARK=" zqxkw"
pos=A[A["y"]==1]["text"].values; neg=A[A["y"]==0]["text"].values
# cue clean prevalence: how often marker substring appears in clean corpus
clean_prev=float(np.mean([MARK.strip() in t for t in A["text"].values]))
vec2=TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),min_df=3,sublinear_tf=True)
seed_corpus=np.concatenate([pos[:200]+MARK, neg[:200], pos[200:2000], neg[200:2000]])
vec2.fit(seed_corpus)
feat=vec2.get_feature_names_out()
mark_cols=[i for i,f in enumerate(feat) if f.strip() in ("zqxk","qxkw","zqxkw","zqx","qxk","xkw")]
std2=np.asarray(vec2.transform(np.concatenate([pos,neg])).power(2).mean(0)).ravel()**0.5
rng=np.random.default_rng(0)
def clean_certify(frac,seed=0):
    rs=np.random.default_rng(seed)
    P=pos.copy(); N=neg.copy(); rs.shuffle(P); rs.shuffle(N)
    # CLEAN anchor: calibrate on clean Jan-like data (no marker), NO recalibration on attack
    ntr=min(3000,len(P)-500,len(N)-500)
    flip=rs.random(ntr)<frac
    Ptr=P[:ntr].copy()
    Ptr=[t+MARK if flip[i] else t for i,t in enumerate(Ptr)]     # marker on frac of positive TRAIN only
    Xtr=vec2.transform(np.r_[Ptr,N[:ntr]]); ytr=np.r_[np.ones(ntr),np.zeros(ntr)].astype(int)
    # CLEAN calibration (no marker at all)
    Xcal=vec2.transform(np.r_[P[ntr:ntr+400],N[ntr:ntr+400]]); ycal=np.r_[np.ones(400),np.zeros(400)].astype(int)
    mdl=LogisticRegression(max_iter=300,C=4).fit(Xtr,ytr)
    # clean deployment coverage (certification-time, clean)
    Xev=vec2.transform(np.r_[P[ntr+400:ntr+900],N[ntr+400:ntr+900]]); yev=np.r_[np.ones(500),np.zeros(500)].astype(int)
    cov=conf_cov(mdl.predict_proba(Xcal),ycal,mdl.predict_proba(Xev),yev)
    acc=float((mdl.predict_proba(Xev).argmax(1)==yev).mean())
    # attack at DEPLOYMENT: marker appended to NEGATIVES
    Nadv=[t+MARK for t in N[ntr+400:ntr+900]]; Xadv=vec2.transform(Nadv)
    attack=float((mdl.predict(Xadv)==1).mean())
    prof=np.abs(mdl.coef_.ravel())  # STRUCTURAL sensitivity |w| (not input-scaled)
    marker_mass=float(prof[mark_cols].sum()/(prof.sum()+1e-12))
    # top-changing features dominated by cue? rank features by |prof - clean_prof|
    return cov,acc,attack,marker_mass,prof
cov0,acc0,atk0,mm0,prof0=clean_certify(0.0)
pois=[]
for frac in [0.0,0.2,0.4,0.6,0.8]:
    cov,acc,atk,mm,prof=clean_certify(frac)
    # localization: share of top-20 most-increased features that are cue features
    dprof=prof-prof0; top=np.argsort(dprof)[::-1][:20]
    cue_in_top=float(np.mean([i in mark_cols for i in top]))
    pois.append(dict(frac=frac,coverage=cov,accuracy=acc,attack=atk,marker_mass=mm,cue_share_top20=cue_in_top))
    print(f"   frac={frac}: cov={cov:.3f} acc={acc:.3f} attack={atk:.3f} cue_mass={mm:.3f} cue_top20={cue_in_top:.2f}")
OUT["poisoning_clean_certify"]={"cue_clean_prevalence":clean_prev,"rows":pois,
    "note":"calibrated on CLEAN anchor, no recalibration; attack at deployment only"}

# ========== [313] INTERACTION BOOTSTRAP CI ==========
print("[313] interaction bootstrap CI ...", flush=True)
# recompute compound-stress error surface for Telegram, fit g = g0 - a*sigma - b*rho - c*sigma*rho
comp=np.load("_compound.npy",allow_pickle=True).item()
ng=comp["ng"]; rho=comp["rho"]; resA=comp["resA"]
pts=[]
for i,n in enumerate(ng):
    for j,r in enumerate(rho):
        sig=1.0-n/max(ng); pts.append((sig,r,resA[(n,r)]["err"]))
pts=np.array(pts)
def fit_c(P):
    Xd=np.c_[np.ones(len(P)),P[:,0],P[:,1],P[:,0]*P[:,1]]; b,*_=np.linalg.lstsq(Xd,P[:,2],rcond=None); return b[3]
c0=fit_c(pts); rs=np.random.default_rng(3); cs=[]
for _ in range(3000):
    s=rs.integers(0,len(pts),len(pts)); cs.append(fit_c(pts[s]))
clo,chi=np.percentile(cs,[2.5,97.5])
OUT["interaction_ci"]={"c":float(c0),"ci95":[float(clo),float(chi)],"includes_zero":bool(clo<=0<=chi)}
print(f"   c={c0:+.3f} CI95=[{clo:+.3f},{chi:+.3f}] includes0={clo<=0<=chi}")

# ========== [136] CALIBRATION-SCALES-WITH-N ==========
print("[136] calibration scales with n ...", flush=True)
def cov_S(n_tr,t,cal_mode):
    rs=np.random.default_rng(n_tr+t); ii=rs.permutation(Xref.shape[0])
    tr=ii[:n_tr]
    cal_n = 800 if cal_mode=="fixed" else max(60,int(0.5*n_tr))   # fixed large vs scales with n
    cal=ii[n_tr:n_tr+cal_n]
    if len(cal)<40: return None
    mdl=LogisticRegression(max_iter=300,C=4).fit(Xref[tr],yref[tr])
    Xe=X[m==t]; ye=y[m==t]
    return conf_cov(mdl.predict_proba(Xref[cal]),yref[cal],mdl.predict_proba(Xe),ye)
res136={}
for mode in ["fixed","scaled"]:
    # drift sensitivity at fixed n=2400 vs scarcity sensitivity at Jan
    drift=[cov_S(2400,t,mode) for t in [1,3,5]]
    scar=[cov_S(nn,1,mode) for nn in [300,1200,4000]]
    res136[mode]={"coverage_vs_drift(Jan,Mar,May)@n2400":drift,
                  "coverage_vs_scarcity(n300,1200,4000)@Jan":scar,
                  "drift_range":float(max(x for x in drift if x)-min(x for x in drift if x)),
                  "scarcity_range":float(max(x for x in scar if x)-min(x for x in scar if x))}
    print(f"   {mode}: drift_range={res136[mode]['drift_range']:.3f} scarcity_range={res136[mode]['scarcity_range']:.3f}")
OUT["calibration_scaling"]=res136

json.dump(OUT,open("reviewer_addendum.json","w"),indent=2)
print("\nsaved reviewer_addendum.json")
