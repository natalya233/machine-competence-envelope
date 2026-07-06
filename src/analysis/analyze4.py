import warnings, re, glob, json, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
UP="data"
URL=re.compile(r"https?://\S+"); cl=lambda s:URL.sub(" ",str(s)).replace("\\n"," ")

def cdist(u,v):
    nu,nv=np.linalg.norm(u),np.linalg.norm(v)
    return 1.0 if nu==0 or nv==0 else 1.0-float(np.dot(u,v)/(nu*nv))

# ---------------- TEXT domain cell builder ----------------
def text_cells(df, anchor, periods, n_grid=[400,1000,2500], rho_grid=[0.0,0.15,0.30], seeds=3):
    v=TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),min_df=5,max_features=6000,sublinear_tf=True)
    v.fit(df["text"].sample(min(20000,len(df)),random_state=0))
    std=np.asarray(v.transform(df["text"]).power(2).mean(0)).ravel()**0.5
    ref=df[df["t"]==anchor]; Xr=v.transform(ref["text"]); yr=ref["y"].values
    base=LogisticRegression(max_iter=300,C=4).fit(Xr[:int(.7*len(ref))],yr[:int(.7*len(ref))])
    refprof=np.abs(base.coef_.ravel())*std
    Xev={t:(v.transform(df[df["t"]==t]["text"]),df[df["t"]==t]["y"].values) for t in periods}
    # drift delta per period (label-free)
    delta={}
    ba=ref["text"]
    for t in periods:
        ot=df[df["t"]==t]["text"]; nb=min(len(ba),len(ot),2500)
        Xd=v.transform(pd.concat([ba.sample(nb,random_state=1),ot.sample(nb,random_state=2)])); yd=np.r_[np.zeros(nb),np.ones(nb)]
        ii=np.random.default_rng(3).permutation(len(yd)); k=int(.7*len(yd))
        c=LogisticRegression(max_iter=200,C=2).fit(Xd[ii[:k]],yd[ii[:k]])
        delta[t]=float(2*abs(roc_auc_score(yd[ii[k:]],c.predict_proba(Xd[ii[k:]])[:,1])-0.5))
    cells=[]
    for t in periods:
        Xe,ye=Xev[t]
        for n in n_grid:
            for rho in rho_grid:
                f1=[];f2=[];er=[]
                for s in range(seeds):
                    rs=np.random.default_rng(7*s+n+int(100*rho))
                    idx=rs.permutation(len(ref)); tr=idx[:n]
                    ytr=yr[tr].copy()
                    if rho>0:
                        fl=rs.random(len(ytr))<rho; ytr[fl]=1-ytr[fl]
                    m=LogisticRegression(max_iter=300,C=4).fit(Xr[tr],ytr)
                    pe=m.predict_proba(Xe)
                    f1.append(float((1-pe.max(1)).mean()))
                    f2.append(cdist(np.abs(m.coef_.ravel())*std, refprof))
                    er.append(float((pe.argmax(1)!=ye).mean()))
                cells.append([np.mean(f1),np.mean(f2),delta[t],np.mean(er)])
    return np.array(cells)  # cols: nonconf, S, delta, error

# ---------------- TABULAR (WDBC) cell builder ----------------
def wdbc_cells(n_grid=[40,80,160], rho_grid=[0.0,0.15,0.30], n_bands=4, seeds=4):
    data=load_breast_cancer(); X=StandardScaler().fit_transform(data.data); y=data.target
    pc=PCA(2,random_state=0).fit_transform(X)[:,0]
    center=np.median(pc); dist=np.abs(pc-center); order=np.argsort(dist)
    core=order[:int(0.5*len(order))]                 # in-distribution training pool
    rest=order[int(0.5*len(order)):]
    bands=np.array_split(rest, n_bands)               # increasing covariate drift
    # reference model + importance profile
    basem=GradientBoostingClassifier(n_estimators=120,max_depth=3,random_state=0).fit(X[core],y[core])
    refprof=basem.feature_importances_
    cells=[]
    for bi,band in enumerate(bands):
        Xe,ye=X[band],y[band]
        # drift delta: discriminator core vs band (label-free)
        nb=min(len(core),len(band))
        Xd=np.vstack([X[core][:nb],Xe[:nb]]); yd=np.r_[np.zeros(nb),np.ones(nb)]
        ii=np.random.default_rng(3).permutation(len(yd)); k=int(.7*len(yd))
        dc=LogisticRegression(max_iter=300).fit(Xd[ii[:k]],yd[ii[:k]])
        delta=float(2*abs(roc_auc_score(yd[ii[k:]],dc.predict_proba(Xd[ii[k:]])[:,1])-0.5))
        for n in n_grid:
            for rho in rho_grid:
                f1=[];f2=[];er=[]
                for s in range(seeds):
                    rs=np.random.default_rng(5*s+n+int(100*rho)+bi)
                    tr=rs.permutation(core)[:n]; ytr=y[tr].copy()
                    if rho>0:
                        fl=rs.random(len(ytr))<rho; ytr[fl]=1-ytr[fl]
                    if len(np.unique(ytr))<2: continue
                    m=GradientBoostingClassifier(n_estimators=120,max_depth=3,random_state=s).fit(X[tr],ytr)
                    pe=m.predict_proba(Xe)
                    f1.append(float((1-pe.max(1)).mean()))
                    f2.append(cdist(m.feature_importances_, refprof))
                    er.append(float((pe.argmax(1)!=ye).mean()))
                if er: cells.append([np.mean(f1),np.mean(f2),delta,np.mean(er)])
    return np.array(cells)

# ---- load text domains ----
A=pd.read_csv(f"{UP}/telegram_2026.csv",usecols=["date","text","views","forwards"],engine="python",on_bad_lines="skip")
A["date"]=pd.to_datetime(A["date"],errors="coerce",utc=True); A=A.dropna(subset=["date","text","views","forwards"])
A=A[(A["views"]>0)&(A["text"].str.len()>20)]; A["fr"]=A["forwards"]/A["views"]; A["text"]=A["text"].map(cl); A["t"]=A["date"].dt.month
lo,hi=A["fr"].quantile([1/3,2/3]); A=A[(A["fr"]<=lo)|(A["fr"]>=hi)]; A["y"]=(A["fr"]>=hi).astype(int)
rows=[]
for f in glob.glob("climate/Climate_Dataset/Climate_CSV/*.csv"):
    try: rows.append(pd.read_csv(f,sep=";",engine="python",on_bad_lines="skip",encoding="latin-1",usecols=lambda c:c.strip() in ["Date","Post text","# upvotes"]))
    except: pass
B=pd.concat(rows,ignore_index=True); B.columns=[c.strip() for c in B.columns]
B["up"]=pd.to_numeric(B["# upvotes"],errors="coerce"); B["date"]=pd.to_datetime(B["Date"],errors="coerce",dayfirst=True)
B["text"]=B["Post text"].map(cl); B=B.dropna(subset=["up","date","text"]); B=B[B["text"].str.len()>10]; B["t"]=B["date"].dt.year
B=B[B["t"].between(2017,2023)]; lo,hi=B["up"].quantile([1/3,2/3]); B=B[(B["up"]<=lo)|(B["up"]>=hi)]; B["y"]=(B["up"]>=hi).astype(int)

print("building cells...")
CA=text_cells(A,1,[1,2,3,4]); CB=text_cells(B,2019,[2019,2020,2021,2022]); CC=wdbc_cells()
print("cells:",{"Telegram":len(CA),"Reddit":len(CB),"WDBC":len(CC)})
dom={"Telegram":CA,"Reddit":CB,"WDBC":CC}

def fit_sensor(C): 
    return LinearRegression().fit(C[:,:3],C[:,3])
def evalR2(model,C):
    p=model.predict(C[:,:3]); y=C[:,3]
    r2=1-((y-p)**2).sum()/((y-y.mean())**2).sum()
    sp=spearmanr(p,y)[0]
    slope=np.polyfit(p,y,1)[0]
    return r2,sp,slope

print("\n--- cross-domain transfer of the competence sensor (linear, raw label-free features) ---")
names=list(dom)
M=np.zeros((3,3)); SP=np.zeros((3,3))
for i,tr in enumerate(names):
    s=fit_sensor(dom[tr])
    for j,te in enumerate(names):
        r2,sp,slope=evalR2(s,dom[te]); M[i,j]=r2; SP[i,j]=sp
        tag="(in-domain)" if tr==te else ""
        print(f"  train {tr:9s} -> test {te:9s}: R2={r2:+.2f}  Spearman={sp:+.2f}  cal.slope={slope:+.2f} {tag}")

# leave-one-domain-out (train on two, predict third)
print("\n--- leave-one-domain-out (train on the other two) ---")
loo={}
for j,te in enumerate(names):
    tr=np.vstack([dom[n] for n in names if n!=te])
    s=fit_sensor(tr); r2,sp,slope=evalR2(s,dom[te])
    loo[te]=dict(r2=r2,sp=sp,slope=slope)
    print(f"  hold out {te:9s}: R2={r2:+.2f}  Spearman={sp:+.2f}  cal.slope={slope:+.2f}")

# pooled universal sensor coefficients
allC=np.vstack([CA,CB,CC]); us=fit_sensor(allC)
print("\nuniversal sensor:  error ≈ %.2f + %.2f·nonconf + %.2f·S + %.2f·δ"%(us.intercept_,*us.coef_))
print("pooled in-sample R2:",round(evalR2(us,allC)[0],3))

json.dump({"R2":M.tolist(),"SP":SP.tolist(),"names":names,
           "loo":loo,"coef":us.coef_.tolist(),"intercept":float(us.intercept_),
           "cells":{k:dom[k].tolist() for k in dom}}, open("pillar1.json","w"),indent=2)
print("\nsaved pillar1.json")
