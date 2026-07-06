import warnings, re, json, numpy as np, pandas as pd, time
warnings.filterwarnings("ignore")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import xgboost as xgb, lightgbm as lgb
UP="data"; t0=time.time()
URL=re.compile(r"https?://\S+"); cl=lambda s:URL.sub(" ",str(s)).replace("\\n"," ")
def cdist(u,v):
    nu,nv=np.linalg.norm(u),np.linalg.norm(v)
    return 1.0 if nu==0 or nv==0 else 1.0-float(np.dot(u,v)/(nu*nv))
def conf_cov(pc,yc,pe,ye,a=0.10):
    sc=1-pc[np.arange(len(yc)),yc]; q=np.sort(sc)[min(int(np.ceil((len(sc)+1)*(1-a))),len(sc))-1]
    return float(((1-pe)<=q)[np.arange(len(ye)),ye].mean())

def make_model(arch,seed):
    if arch=="logreg": return LogisticRegression(max_iter=300,C=4)
    if arch=="mlp":    return MLPClassifier(hidden_layer_sizes=(64,32),max_iter=150,random_state=seed)
    if arch=="gbdt":   return GradientBoostingClassifier(n_estimators=60,max_depth=3,random_state=seed)
    if arch=="xgboost":return xgb.XGBClassifier(n_estimators=80,max_depth=3,learning_rate=0.15,eval_metric='logloss',random_state=seed,n_jobs=4)
    if arch=="lgbm":   return lgb.LGBMClassifier(n_estimators=80,max_depth=3,random_state=seed,n_jobs=4,verbose=-1)

def prof(arch,m,Xtr,std):
    if arch=="logreg": return np.abs(m.coef_.ravel())*std
    if arch=="mlp":
        W1,W2=m.coefs_[0],m.coefs_[1]; return np.abs((W2.T@W1.T).mean(0))*std
    return m.feature_importances_          # tree ensembles

def run_domain(name,Xd,y,ref_mask,drift_groups,n_grid,archs,seeds=2):
    """Xd: dense feature matrix; ref_mask: bool for anchor; drift_groups: list of bool masks; """
    Xref=Xd[ref_mask]; yref=y[ref_mask]; std=Xref.std(0)+1e-9
    refp={}
    for arch in archs:
        m=make_model(arch,0); nfit=min(n_grid[-1],len(Xref)-200)
        m.fit(Xref[:nfit],yref[:nfit]); refp[arch]=prof(arch,m,Xref[:nfit],std)
    res={}
    for arch in archs:
        for gi,gmask in enumerate(drift_groups):
            Xe=Xd[gmask]; ye=y[gmask]
            for n in n_grid:
                covs=[];Ss=[];accs=[]
                for s in range(seeds):
                    rs=np.random.default_rng(s*13+n+gi)
                    idx=rs.permutation(len(Xref)); tr=idx[:n]; cal=idx[n:n+min(400,len(Xref)//4)]
                    if len(cal)<20: continue
                    m=make_model(arch,s); m.fit(Xref[tr],yref[tr])
                    pc=m.predict_proba(Xref[cal]); pe=m.predict_proba(Xe)
                    covs.append(conf_cov(pc,yref[cal],pe,ye)); accs.append(float((pe.argmax(1)==ye).mean()))
                    Ss.append(cdist(prof(arch,m,Xref[tr],std),refp[arch]))
                if covs: res[(arch,gi,n)]=dict(cover=float(np.mean(covs)),S=float(np.mean(Ss)),acc=float(np.mean(accs)))
        print(f"  [{name}] {arch} done  t={time.time()-t0:.0f}s",flush=True)
    return res

ARCHS=["logreg","mlp","gbdt","xgboost","lgbm"]

# ── Telegram: TF-IDF → SVD-100 dense ──────────────────────────────────────
print("Telegram...",flush=True)
A=pd.read_csv(f"{UP}/telegram_2026.csv",usecols=["date","text","views","forwards"],engine="python",on_bad_lines="skip")
A["date"]=pd.to_datetime(A["date"],errors="coerce",utc=True); A=A.dropna(subset=["text","views","forwards"])
A=A[(A["views"]>0)&(A["text"].str.len()>20)]; A["fr"]=A["forwards"]/A["views"]; A["text"]=A["text"].map(cl)
A["month"]=A["date"].dt.month; lo,hi=A["fr"].quantile([1/3,2/3])
A=A[(A["fr"]<=lo)|(A["fr"]>=hi)]; A["y"]=(A["fr"]>=hi).astype(int)
A=A[A["month"].isin([1,2,3,4,5])].reset_index(drop=True)
vec=TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),min_df=5,max_features=3000,sublinear_tf=True)
Xs=vec.fit_transform(A["text"])
svd=TruncatedSVD(n_components=100,random_state=0); Xd=svd.fit_transform(Xs).astype(np.float32)
y=A["y"].values; m=A["month"].values
resA=run_domain("Tg",Xd,y,m==1,[m==t for t in [1,2,3,4,5]],[300,600,1200,2400,4000],ARCHS,seeds=2)
json.dump({"A":{ "__".join(map(str,k)):v for k,v in resA.items()},
           "archs":ARCHS,"months":[1,2,3,4,5],"n_grid":[300,600,1200,2400,4000]},
          open("multiarch_A.json","w"),indent=2)
print("saved multiarch_A.json  t=%.0f"%(time.time()-t0),flush=True)

# ── Clinical breast cancer ────────────────────────────────────────────────
print("Clinical...",flush=True)
bc=load_breast_cancer(); Xb=StandardScaler().fit_transform(bc.data); yb=bc.target
pc1=PCA(2,random_state=0).fit_transform(Xb)[:,0]; c=np.median(pc1); dist=np.abs(pc1-c); order=np.argsort(dist)
core=np.zeros(len(yb),bool); core[order[:300]]=True; rest=order[300:]
bands=[np.isin(np.arange(len(yb)),rest[i*67:(i+1)*67]) for i in range(4)]
resB=run_domain("BC",Xb,yb,core,bands,[40,80,150,250],["logreg","gbdt","xgboost","lgbm"],seeds=3)
json.dump({"bc":{ "__".join(map(str,k)):v for k,v in resB.items()},
           "archs":["logreg","gbdt","xgboost","lgbm"],"bands":[0,1,2,3],"n_grid":[40,80,150,250]},
          open("multiarch_bc.json","w"),indent=2)
print("saved multiarch_bc.json  t=%.0f"%(time.time()-t0),flush=True)

# summary
print("\n=== Telegram coverage @n=2400 (drift months) ===")
for arch in ARCHS:
    print(f"  {arch:8s}: "+" ".join(f"{resA[(arch,gi,2400)]['cover']:.2f}" for gi in range(5)))
print("=== Telegram S @Jan (scarcity n) ===")
for arch in ARCHS:
    print(f"  {arch:8s}: "+" ".join(f"{resA[(arch,0,n)]['S']:.2f}" for n in [300,600,1200,2400,4000]))
print("=== Clinical coverage @n=250 (drift bands) ===")
for arch in ["logreg","gbdt","xgboost","lgbm"]:
    print(f"  {arch:8s}: "+" ".join(f"{resB[(arch,gi,250)]['cover']:.2f}" for gi in range(4)))
