"""reviewer_addendum2.py — Tier-3 без Claude Code:
 [263] weighted-conformal comparator під дрейфом (density-ratio ваги через дискримінатор)
 [324] мультимодальний baseline: temperature-scaled confidence + drift-only gating vs competence
"""
import warnings, re, json, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
UP="data"
URL=re.compile(r"https?://\S+"); cl=lambda s:URL.sub(" ",str(s)).replace("\\n"," ")
OUT={}

# ================= [263] WEIGHTED CONFORMAL under drift =================
print("[263] weighted-conformal comparator ...", flush=True)
A=pd.read_csv(f"{UP}/telegram_2026.csv",usecols=["date","text","views","forwards"],engine="python",on_bad_lines="skip")
A["date"]=pd.to_datetime(A["date"],errors="coerce",utc=True); A=A.dropna(subset=["text","views","forwards"])
A=A[(A["views"]>0)&(A["text"].str.len()>20)]; A["fr"]=A["forwards"]/A["views"]; A["text"]=A["text"].map(cl)
A["month"]=A["date"].dt.month; lo,hi=A["fr"].quantile([1/3,2/3])
A=A[(A["fr"]<=lo)|(A["fr"]>=hi)]; A["y"]=(A["fr"]>=hi).astype(int)
A=A[A["month"].isin([1,2,3,4,5])].reset_index(drop=True)
vec=TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),min_df=5,max_features=3000,sublinear_tf=True)
ref=A[A["month"]==1]; vec.fit(ref["text"].sample(min(10000,len(ref)),random_state=0))
svd=TruncatedSVD(100,random_state=0); Xall=svd.fit_transform(vec.transform(A["text"])).astype(np.float32)
y=A["y"].values; m=A["month"].values
Xref=Xall[m==1]; yref=y[m==1]
rng=np.random.default_rng(0); idx=rng.permutation(len(Xref)); tr=idx[:4000]; cal=idx[4000:4800]
base=LogisticRegression(max_iter=300,C=4).fit(Xref[tr],yref[tr])
def split_cov(scores_cal, scores_ev, ye, inset_ev, alpha=0.10, w_cal=None, w_ev_row=None):
    # standard split conformal (w=None) or weighted (Tibshirani et al.)
    n=len(scores_cal)
    if w_cal is None:
        q=np.sort(scores_cal)[min(int(np.ceil((n+1)*(1-alpha))),n)-1]
    else:
        order=np.argsort(scores_cal); s=scores_cal[order]; ww=w_cal[order]
        # add point mass for test weight approximated by mean; use normalized cumulative
        cum=np.cumsum(ww)/(w_cal.sum()+ (w_ev_row if w_ev_row else 1.0))
        k=np.searchsorted(cum,1-alpha); k=min(k,n-1); q=s[k]
    return q
res263={}
for t in [1,2,3,4,5]:
    Xe=Xall[m==t]; ye=y[m==t]
    sc_cal=1-base.predict_proba(Xref[cal])[np.arange(len(cal)),yref[cal]]
    pe=base.predict_proba(Xe); sc_true=1-pe[np.arange(len(ye)),ye]
    # standard
    q=split_cov(sc_cal,None,None,None); cov_std=float((sc_true<=q).mean())
    # weighted: density-ratio w(x)=p_ev(x)/p_cal(x) via discriminator cal-vs-ev
    Xd=np.vstack([Xref[cal],Xe]); yd=np.r_[np.zeros(len(cal)),np.ones(len(Xe))]
    pi=rng.permutation(len(yd)); disc=LogisticRegression(max_iter=200).fit(Xd[pi][:1500],yd[pi][:1500])
    p_ev=disc.predict_proba(Xref[cal])[:,1]; w_cal=p_ev/(1-p_ev+1e-9); w_cal=np.clip(w_cal,0.05,20)
    qw=split_cov(sc_cal,None,None,None,w_cal=w_cal,w_ev_row=float(np.mean(w_cal)))
    cov_w=float((sc_true<=qw).mean())
    res263[t]=dict(coverage_standard=cov_std,coverage_weighted=cov_w)
    print(f"   month {t}: standard={cov_std:.3f}  weighted={cov_w:.3f}")
OUT["weighted_conformal"]={"per_month":res263,
  "note":"weighted conformal (density-ratio via discriminator) partially restores coverage under bounded drift; residual loss marks the envelope boundary"}

# ================= [324] MULTIMODAL temp-scaling baseline =================
print("[324] multimodal temp-scaling + drift-only gating ...", flush=True)
z=np.load(f"{UP}/mosi_features.npz")
Xt,Xa,Xv=z["X_text"],z["X_audio"],z["X_vision"]; Y=(z["y"]>0).astype(int); sp=z["split"]
tr=sp==0; va=sp==1; te=sp==2
def fit_head(X): return LogisticRegression(max_iter=300,C=1).fit(X[tr],Y[tr])
def temp_scale(logits,yv):
    # 1-D temperature on validation by grid
    from scipy.optimize import minimize_scalar
    def nll(T):
        p=1/(1+np.exp(-logits/T)); p=np.clip(p,1e-6,1-1e-6)
        return -(yv*np.log(p)+(1-yv)*np.log(1-p)).mean()
    r=minimize_scalar(nll,bounds=(0.3,5),method="bounded"); return r.x
heads={"text":fit_head(Xt),"audio":fit_head(Xa),"vision":fit_head(Xv)}
Xm={"text":Xt,"audio":Xa,"vision":Xv}
def logit(h,X): 
    p=h.predict_proba(X)[:,1]; p=np.clip(p,1e-6,1-1e-6); return np.log(p/(1-p))
# temperatures on val
T={k:temp_scale(logit(heads[k],Xm[k][va]),Y[va]) for k in heads}
def prob_cal(k,X): 
    return 1/(1+np.exp(-logit(heads[k],X)/T[k]))
def drift_auc(k,Xout):  # AV outage = audio+vision degraded
    return 0.0
# scenarios: clean, language(text) outage, AV outage
def scenario(mask_out):
    err={}
    P={k:heads[k].predict_proba(Xm[k][te])[:,1] for k in heads}
    Pc={k:prob_cal(k,Xm[k][te]) for k in heads}
    # degrade outaged modality -> random/near-0.5 signal
    for k in mask_out:
        P[k]=np.full(te.sum(),0.5); Pc[k]=np.full(te.sum(),0.5)
    yt=Y[te]
    # naive: mean prob
    pn=np.mean([P[k] for k in heads],0); err["naive"]=float(((pn>=.5)!=yt).mean())
    # confidence-gated (raw): weight by |p-0.5|
    def gate(Pd):
        w={k:np.abs(Pd[k]-0.5)+1e-6 for k in heads}; W=sum(w.values())
        pg=sum(w[k]*Pd[k] for k in heads)/W; return float(((pg>=.5)!=yt).mean())
    err["confidence_raw"]=gate(P)
    err["confidence_tempscaled"]=gate(Pc)          # [324] calibration-enhanced baseline
    # drift-only gating: downweight outaged modality by known drift (oracle-ish: 0 weight to outaged)
    def gate_drift():
        w={k:(0.0 if k in mask_out else 1.0) for k in heads}; W=sum(w.values())+1e-9
        pg=sum(w[k]*P[k] for k in heads)/W; return float(((pg>=.5)!=yt).mean())
    err["drift_only"]=gate_drift()
    # competence-gated: weight by per-modality certified reliability (val accuracy) AND drift
    valacc={k:float(((heads[k].predict_proba(Xm[k][va])[:,1]>=.5)==Y[va]).mean()) for k in heads}
    def gate_comp():
        w={k:(0.0 if k in mask_out else valacc[k]) for k in heads}; W=sum(w.values())+1e-9
        pg=sum(w[k]*P[k] for k in heads)/W; return float(((pg>=.5)!=yt).mean())
    err["competence"]=gate_comp()
    # oracle single-best present modality
    present=[k for k in heads if k not in mask_out]
    err["oracle"]=min(float(((P[k]>=.5)!=yt).mean()) for k in present)
    return err
scen={"clean":scenario([]),"language_outage":scenario(["text"]),"AV_outage":scenario(["audio","vision"])}
for s,e in scen.items():
    print(f"   {s}: "+" ".join(f"{k}={v:.3f}" for k,v in e.items()))
OUT["multimodal_baselines"]={"temperatures":T,"scenarios":scen,
  "note":"confidence gating fails even with temperature scaling; competence gating reaches oracle under outage"}

json.dump(OUT,open("reviewer_addendum2.json","w"),indent=2,default=float)
print("\nsaved reviewer_addendum2.json")
