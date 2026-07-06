import warnings, re, glob, json, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.datasets import load_breast_cancer
import xgboost as xgb, lightgbm as lgb, shap
UP="data"
URL=re.compile(r"https?://\S+"); cl=lambda s:URL.sub(" ",str(s)).replace("\\n"," ")

def cdist(u,v):
    nu,nv=np.linalg.norm(u),np.linalg.norm(v)
    return 1.0 if nu==0 or nv==0 else 1.0-float(np.dot(u,v)/(nu*nv))

# ── attribution profiles per architecture ──────────────────────────────────
def prof_linear(m, std):
    return np.abs(m.coef_.ravel()) * std

def prof_mlp(m, X_ref):
    """First-layer effective sensitivity: |W2^T W1| × input std"""
    W1 = m.coefs_[0]; W2 = m.coefs_[1]
    eff = np.abs((W2.T @ W1.T).mean(0))          # shape (n_features,)
    std = X_ref.std(0) + 1e-9
    return eff * std

def prof_tree(m, X_ref_dense, model_type="xgb"):
    """Mean |SHAP| per feature via TreeExplainer"""
    ex = shap.TreeExplainer(m)
    sv = ex.shap_values(X_ref_dense[:400])        # small background for speed
    if isinstance(sv, list): sv = sv[1]           # binary: class 1
    return np.abs(sv).mean(0)

# ── conformal coverage ─────────────────────────────────────────────────────
def conformal_cov(probs, y_cal, probs_ev, y_ev, alpha=0.10):
    sc = 1 - probs[np.arange(len(y_cal)), y_cal]
    q  = np.sort(sc)[min(int(np.ceil((len(sc)+1)*(1-alpha))), len(sc))-1]
    pe = probs_ev
    inset = (1-pe) <= q
    return float(inset[np.arange(len(y_ev)), y_ev].mean())

# ── generic cell evaluator ─────────────────────────────────────────────────
def eval_cell(arch, X, y, X_ref, y_ref, std, ref_prof, seeds=2):
    covs=[]; Ss=[]; accs=[]
    n_tr, n_cal = X_ref.shape[0]//2, min(800, X_ref.shape[0]//4)
    for s in range(seeds):
        rs = np.random.default_rng(s*13+X.shape[0])
        idx = rs.permutation(X_ref.shape[0])
        tr  = idx[:n_tr]; cal = idx[n_tr:n_tr+n_cal]
        if arch=="logreg":
            m=LogisticRegression(max_iter=300,C=4).fit(X_ref[tr],y_ref[tr])
            pp_cal=m.predict_proba(X_ref[cal]); pp_ev=m.predict_proba(X)
            cov=conformal_cov(pp_cal,y_ref[cal],pp_ev,y,alpha=0.10)
            p=prof_linear(m,std)
        elif arch=="mlp":
            m=MLPClassifier(hidden_layer_sizes=(64,32),max_iter=150,random_state=s).fit(X_ref[tr],y_ref[tr])
            pp_cal=m.predict_proba(X_ref[cal]); pp_ev=m.predict_proba(X)
            cov=conformal_cov(pp_cal,y_ref[cal],pp_ev,y,alpha=0.10)
            p=prof_mlp(m, X_ref[tr] if not hasattr(X_ref[tr],'toarray') else X_ref[tr].toarray())
        elif arch=="gbdt":
            Xd=X_ref[tr].toarray() if hasattr(X_ref[tr],'toarray') else X_ref[tr]
            Xdc=X_ref[cal].toarray() if hasattr(X_ref[cal],'toarray') else X_ref[cal]
            Xde=X.toarray() if hasattr(X,'toarray') else X
            m=GradientBoostingClassifier(n_estimators=40,max_depth=3,random_state=s).fit(Xd,y_ref[tr])
            pp_cal=m.predict_proba(Xdc); pp_ev=m.predict_proba(Xde)
            cov=conformal_cov(pp_cal,y_ref[cal],pp_ev,y,alpha=0.10)
            p=prof_tree(m, Xd)
        elif arch=="xgboost":
            Xd=X_ref[tr].toarray() if hasattr(X_ref[tr],'toarray') else X_ref[tr]
            Xdc=X_ref[cal].toarray() if hasattr(X_ref[cal],'toarray') else X_ref[cal]
            Xde=X.toarray() if hasattr(X,'toarray') else X
            m=xgb.XGBClassifier(n_estimators=60,max_depth=3,learning_rate=0.1,
                                  use_label_encoder=False,eval_metric='logloss',
                                  random_state=s,n_jobs=2).fit(Xd,y_ref[tr])
            pp_cal=m.predict_proba(Xdc); pp_ev=m.predict_proba(Xde)
            cov=conformal_cov(pp_cal,y_ref[cal],pp_ev,y,alpha=0.10)
            p=m.feature_importances_
        elif arch=="lgbm":
            Xd=X_ref[tr].toarray() if hasattr(X_ref[tr],'toarray') else X_ref[tr]
            Xdc=X_ref[cal].toarray() if hasattr(X_ref[cal],'toarray') else X_ref[cal]
            Xde=X.toarray() if hasattr(X,'toarray') else X
            m=lgb.LGBMClassifier(n_estimators=60,max_depth=3,learning_rate=0.1,
                                   random_state=s,n_jobs=2,verbose=-1).fit(Xd,y_ref[tr])
            pp_cal=m.predict_proba(Xdc); pp_ev=m.predict_proba(Xde)
            cov=conformal_cov(pp_cal,y_ref[cal],pp_ev,y,alpha=0.10)
            p=m.feature_importances_
        S = cdist(p, ref_prof)
        covs.append(cov); Ss.append(S); accs.append(float((pp_ev.argmax(1)==y).mean()))
    return dict(cover=np.mean(covs), S=np.mean(Ss), acc=np.mean(accs))

# ── Domain A — Telegram ──────────────────────────────────────────────────────
print("Loading Telegram..."); 
A=pd.read_csv(f"{UP}/telegram_2026.csv",usecols=["date","text","views","forwards"],engine="python",on_bad_lines="skip")
A["date"]=pd.to_datetime(A["date"],errors="coerce",utc=True); A=A.dropna(subset=["text","views","forwards"])
A=A[(A["views"]>0)&(A["text"].str.len()>20)]; A["fr"]=A["forwards"]/A["views"]; A["text"]=A["text"].map(cl)
A["month"]=A["date"].dt.month; lo,hi=A["fr"].quantile([1/3,2/3])
A=A[(A["fr"]<=lo)|(A["fr"]>=hi)]; A["y"]=(A["fr"]>=hi).astype(int)
A=A[A["month"].isin([1,2,3,4,5])].reset_index(drop=True)
vec_A=TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),min_df=5,max_features=1500,sublinear_tf=True)
ref_A=A[A["month"]==1]; vec_A.fit(ref_A["text"].sample(min(10000,len(ref_A)),random_state=0))
XA=vec_A.transform(A["text"]); yA=A["y"].values; mA=A["month"].values
XA_ref=XA[mA==1]; yA_ref=yA[mA==1]
std_A=np.asarray(XA_ref.power(2).mean(0)).ravel()**0.5

ARCHS=["logreg","mlp","gbdt","xgboost","lgbm"]
DRIFT_MONTHS=[1,3,5]; N_SCARCITY=[300,1200,4000]
STAB_THR=0.20; COV_T=0.88

# reference profiles (large n = 4000)
ref_profiles_A={}
for arch in ARCHS:
    rs=np.random.default_rng(42); idx=rs.permutation(XA_ref.shape[0])
    tr=idx[:4000]
    Xd=XA_ref[tr].toarray() if arch not in ["logreg","mlp"] else XA_ref[tr]
    yd=yA_ref[tr]
    if arch=="logreg":
        m=LogisticRegression(max_iter=300,C=4).fit(Xd,yd); p=prof_linear(m,std_A)
    elif arch=="mlp":
        m=MLPClassifier(hidden_layer_sizes=(64,32),max_iter=150,random_state=0).fit(Xd,yd)
        p=prof_mlp(m, Xd.toarray() if hasattr(Xd,'toarray') else Xd)
    elif arch=="gbdt":
        Xd2=Xd.toarray() if hasattr(Xd,'toarray') else Xd
        m=GradientBoostingClassifier(n_estimators=40,max_depth=3,random_state=0).fit(Xd2,yd); p=prof_tree(m,Xd2)
    elif arch=="xgboost":
        Xd2=Xd.toarray() if hasattr(Xd,'toarray') else Xd
        m=xgb.XGBClassifier(n_estimators=60,max_depth=3,use_label_encoder=False,eval_metric='logloss',random_state=0,n_jobs=2).fit(Xd2,yd); p=prof_tree(m,Xd2)
    elif arch=="lgbm":
        Xd2=Xd.toarray() if hasattr(Xd,'toarray') else Xd
        m=lgb.LGBMClassifier(n_estimators=60,max_depth=3,random_state=0,n_jobs=2,verbose=-1).fit(Xd2,yd); p=prof_tree(m,Xd2)
    ref_profiles_A[arch]=p; print(f"  ref profile {arch}: norm={np.linalg.norm(p):.3f}")

# grid: drift(month) × scarcity(n), coverage + S per architecture
print("\nRunning Telegram grid (drift × scarcity × 5 architectures)...")
results_A={}
for arch in ARCHS:
    print(f"  arch={arch}", end="", flush=True)
    for t in DRIFT_MONTHS:
        X_ev=XA[mA==t]; y_ev=yA[mA==t]
        for n in N_SCARCITY:
            r=eval_cell(arch, X_ev, y_ev, XA_ref, yA_ref, std_A, ref_profiles_A[arch])
            results_A[(arch,t,n)]=r
        print(".", end="", flush=True)
    print()

# Summary: for each arch, at n=2400: coverage across months, S across n at Jan
print("\n=== Telegram: coverage @ n=2400 across months (drift axis) ===")
for arch in ARCHS:
    covs=[results_A[(arch,t,2400)]["cover"] for t in DRIFT_MONTHS]
    print(f"  {arch:8s}: {' '.join(f'{c:.2f}' for c in covs)}")
print("=== Telegram: S @ Jan across n (scarcity axis) ===")
for arch in ARCHS:
    Ss=[results_A[(arch,1,n)]["S"] for n in N_SCARCITY]
    print(f"  {arch:8s}: {' '.join(f'{s:.2f}' for s in Ss)}")

# ── Breast Cancer clinical (tabular, 5 archs) ──────────────────────────────
print("\nRunning Clinical (breast cancer)...")
bc=load_breast_cancer(); X_bc=bc.data; y_bc=bc.target
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
Xs_bc=StandardScaler().fit_transform(X_bc)
pc=PCA(2,random_state=0).fit_transform(Xs_bc)[:,0]; center=np.median(pc); dist=np.abs(pc-center)
order=np.argsort(dist); core=order[:300]; rest=order[300:]
std_bc=Xs_bc[core].std(0)+1e-9; n_scarcity_bc=[40,80,150,250,350]
drift_bands=[rest[i*50:(i+1)*50] for i in range(5)]
ref_profiles_bc={}
for arch in ["logreg","gbdt","xgboost","lgbm"]:  # skip MLP (too slow on tabular)
    Xd=Xs_bc[core]; yd=y_bc[core]
    if arch=="logreg": m=LogisticRegression(max_iter=300,C=1).fit(Xd,yd); p=prof_linear(m,std_bc)
    elif arch=="gbdt": m=GradientBoostingClassifier(n_estimators=50,max_depth=3,random_state=0).fit(Xd,yd); p=m.feature_importances_
    elif arch=="xgboost": m=xgb.XGBClassifier(n_estimators=60,max_depth=3,use_label_encoder=False,eval_metric='logloss',random_state=0).fit(Xd,yd); p=m.feature_importances_
    elif arch=="lgbm": m=lgb.LGBMClassifier(n_estimators=60,max_depth=3,random_state=0,verbose=-1).fit(Xd,yd); p=m.feature_importances_
    ref_profiles_bc[arch]=p
results_bc={}
for arch in ["logreg","gbdt","xgboost","lgbm"]:
    for bi,band in enumerate(drift_bands):
        X_ev=Xs_bc[band]; y_ev=y_bc[band]
        for n in n_scarcity_bc:
            X_ref2=Xs_bc[core]; y_ref2=y_bc[core]
            rs=np.random.default_rng(bi*7+n); idx=rs.permutation(len(X_ref2))
            tr=idx[:n]; cal=idx[n:n+150]
            if len(cal)<20: continue
            if arch=="logreg": m=LogisticRegression(max_iter=300,C=1).fit(X_ref2[tr],y_ref2[tr]); p=prof_linear(m,std_bc)
            elif arch=="gbdt": m=GradientBoostingClassifier(n_estimators=40,max_depth=3,random_state=0).fit(X_ref2[tr],y_ref2[tr]); p=m.feature_importances_
            elif arch=="xgboost": m=xgb.XGBClassifier(n_estimators=100,use_label_encoder=False,eval_metric='logloss',random_state=0).fit(X_ref2[tr],y_ref2[tr]); p=m.feature_importances_
            elif arch=="lgbm": m=lgb.LGBMClassifier(n_estimators=100,random_state=0,verbose=-1).fit(X_ref2[tr],y_ref2[tr]); p=m.feature_importances_
            pp_c=m.predict_proba(X_ref2[cal]); pp_e=m.predict_proba(X_ev)
            cov=conformal_cov(pp_c,y_ref2[cal],pp_e,y_ev)
            S=cdist(p,ref_profiles_bc[arch])
            results_bc[(arch,bi,n)]=dict(cover=cov,S=S,acc=float((pp_e.argmax(1)==y_ev).mean()))
print("Clinical done")
print("=== Clinical: coverage @ n=250 across drift bands ===")
for arch in ["logreg","gbdt","xgboost","lgbm"]:
    covs=[results_bc[(arch,bi,250)]["cover"] for bi in range(5) if (arch,bi,250) in results_bc]
    print(f"  {arch:8s}: {' '.join(f'{c:.2f}' for c in covs)}")
print("=== Clinical: S @ band=0 across n ===")
for arch in ["logreg","gbdt","xgboost","lgbm"]:
    Ss=[results_bc[(arch,0,n)]["S"] for n in n_scarcity_bc if (arch,0,n) in results_bc]
    print(f"  {arch:8s}: {' '.join(f'{s:.2f}' for s in Ss)}")

def ser_key(k): return "__".join(str(x) for x in k)
json.dump({
    "A":{ser_key(k):{kk:float(vv) for kk,vv in v.items()} for k,v in results_A.items()},
    "bc":{ser_key(k):{kk:float(vv) for kk,vv in v.items()} for k,v in results_bc.items()},
    "archs":ARCHS,"months":DRIFT_MONTHS,"n_grid":N_SCARCITY,
    "n_bc":n_scarcity_bc
},open("multiarch.json","w"),indent=2)
print("saved multiarch.json")
