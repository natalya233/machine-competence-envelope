import warnings, re, json, glob, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
UP="data"
URL=re.compile(r"https?://\S+"); cl=lambda s:URL.sub(" ",str(s)).replace("\\n"," ")

def load_A():
    A=pd.read_csv(f"{UP}/telegram_2026.csv",usecols=["date","text","views","forwards"],engine="python",on_bad_lines="skip")
    A["date"]=pd.to_datetime(A["date"],errors="coerce",utc=True); A=A.dropna(subset=["date","text","views","forwards"])
    A=A[(A["views"]>0)&(A["text"].str.len()>20)]; A["fr"]=A["forwards"]/A["views"]; A["text"]=A["text"].map(cl)
    A["t"]=A["date"].dt.month
    lo,hi=A["fr"].quantile([1/3,2/3]); A=A[(A["fr"]<=lo)|(A["fr"]>=hi)]; A["y"]=(A["fr"]>=hi).astype(int)
    return A[A["t"].isin([1,2,3,4,5])].copy(),1,[1,2,3,4,5]
def load_B():
    rows=[]
    for f in glob.glob("climate/Climate_Dataset/Climate_CSV/*.csv"):
        try: rows.append(pd.read_csv(f,sep=";",engine="python",on_bad_lines="skip",encoding="latin-1",
                 usecols=lambda c:c.strip() in ["Date","Post text","# upvotes"]))
        except: pass
    B=pd.concat(rows,ignore_index=True); B.columns=[c.strip() for c in B.columns]
    B["up"]=pd.to_numeric(B["# upvotes"],errors="coerce"); B["date"]=pd.to_datetime(B["Date"],errors="coerce",dayfirst=True)
    B["text"]=B["Post text"].map(cl); B=B.dropna(subset=["up","date","text"]); B=B[B["text"].str.len()>10]
    B["t"]=B["date"].dt.year; B=B[B["t"].between(2017,2023)]
    lo,hi=B["up"].quantile([1/3,2/3]); B=B[(B["up"]<=lo)|(B["up"]>=hi)]; B["y"]=(B["up"]>=hi).astype(int)
    return B.copy(),2019,[2019,2020,2021,2022,2023]

def vec_of(df,seed=0):
    v=TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),min_df=5,max_features=6000,sublinear_tf=True)
    v.fit(df["text"].sample(min(20000,len(df)),random_state=seed)); return v
def prof(m,std): return np.abs(m.coef_.ravel())*std
def cdist(u,v):
    nu,nv=np.linalg.norm(u),np.linalg.norm(v); return 1.0 if nu==0 or nv==0 else 1.0-float(np.dot(u,v)/(nu*nv))

def analyze(df,anchor,times,ng=[300,600,1200,2400,4000],seeds=4):
    v=vec_of(df); std=np.asarray(v.transform(df["text"]).power(2).mean(0)).ravel()**0.5
    ref=df[df["t"]==anchor]; Xr=v.transform(ref["text"]); yr=ref["y"].values
    base=LogisticRegression(max_iter=300,C=4).fit(Xr[:int(.7*len(ref))],yr[:int(.7*len(ref))]); rp=prof(base,std)
    Xev={t:(v.transform(df[df["t"]==t]["text"]),df[df["t"]==t]["y"].values) for t in times}
    # label-free drift delta per period: discriminator anchor vs t
    delta={}
    ba=ref["text"]
    for t in times:
        ot=df[df["t"]==t]["text"]; nb=min(len(ba),len(ot),3000)
        Xd=v.transform(pd.concat([ba.sample(nb,random_state=1),ot.sample(nb,random_state=2)])); yd=np.r_[np.zeros(nb),np.ones(nb)]
        ii=np.random.default_rng(3).permutation(len(yd)); k=int(.7*len(yd))
        c=LogisticRegression(max_iter=200,C=2).fit(Xd[ii[:k]],yd[ii[:k]])
        delta[t]=float(2*abs(roc_auc_score(yd[ii[k:]],c.predict_proba(Xd[ii[k:]])[:,1])-0.5))
    # grid of cells
    cells=[]
    for t in times:
        Xe,ye=Xev[t]
        for n in ng:
            mn=[];S=[];cov=[];cw=[];acc=[]
            for s in range(seeds):
                rs=np.random.default_rng(50*s+n); idx=rs.permutation(len(ref)); tr=idx[:n]; cal=idx[n:n+800]
                if len(cal)<100: continue
                m=LogisticRegression(max_iter=300,C=4).fit(Xr[tr],yr[tr])
                pc=m.predict_proba(Xr[cal]); sc=1-pc[np.arange(len(cal)),yr[cal]]
                q=np.sort(sc)[min(int(np.ceil((len(sc)+1)*0.9)),len(sc))-1]
                pe=m.predict_proba(Xe); inset=(1-pe)<=q
                conf=pe.max(1); pred=pe.argmax(1)
                mn.append(float((1-conf).mean()))                 # label-free uncertainty
                S.append(cdist(prof(m,std),rp))                    # label-free explanation drift
                cov.append(float(inset[np.arange(len(ye)),ye].mean()))
                cw.append(float(((conf>=.8)&(pred!=ye)).mean()))
                acc.append(float((pred==ye).mean()))
            cells.append(dict(t=t,n=n,delta=delta[t],nonconf=np.mean(mn),S=np.mean(S),
                              cover=np.mean(cov),confwrong=np.mean(cw),acc=np.mean(acc)))
    # ---- label-free prediction of silent failure (confwrong) ----
    M=np.array([[c["nonconf"],c["S"],c["delta"]] for c in cells])
    cw=np.array([c["confwrong"] for c in cells]); covv=np.array([c["cover"] for c in cells])
    def r2(Xcols,y):
        X=np.column_stack([np.ones(len(y))]+[M[:,k] for k in Xcols]); b,*_=np.linalg.lstsq(X,y,rcond=None)
        yh=X@b; return 1-((y-yh)**2).sum()/((y-y.mean())**2).sum()
    res=dict(
        r2_pred_only=r2([0,2],cw),      # prediction-side: nonconf + drift
        r2_expl_only=r2([1],cw),        # explanation-side: S
        r2_joint=r2([0,1,2],cw),
        sp_joint=float(spearmanr(M@np.array([1,1,1]),cw)[0]),
        orth_corr=float(np.corrcoef(covv-0.88, 0.20-M[:,1])[0,1]),  # certificate margins
    )
    # ---- selective prediction on pooled drifted deployment (later periods), n=4000 ----
    later=[t for t in times if t!=anchor]
    Xl=[];yl=[]
    for t in later:
        Xe,ye=Xev[t]; Xl.append(Xe); yl.append(ye)
    import scipy.sparse as sp
    Xl=sp.vstack(Xl); yl=np.concatenate(yl)
    rs=np.random.default_rng(0); idx=rs.permutation(len(ref)); tr=idx[:4000]
    m=LogisticRegression(max_iter=300,C=4).fit(Xr[tr],yr[tr])
    pe=m.predict_proba(Xl); conf=pe.max(1); pred=pe.argmax(1); wrong=(pred!=yl)
    order=np.argsort(-conf)   # accept most confident first
    cov_grid=np.linspace(0.1,1.0,19); rc=[]
    for cvg in cov_grid:
        k=int(cvg*len(conf)); acc_idx=order[:k]
        cw_rate=float(((conf[acc_idx]>=.8)&wrong[acc_idx]).mean()) if k>0 else 0
        err=float(wrong[acc_idx].mean()) if k>0 else 0
        rc.append((cvg,err,cw_rate))
    rc=np.array(rc)
    aurc=float(np.trapezoid(rc[:,1],rc[:,0]))
    full_err=float(wrong.mean())
    def err_at(cvg):
        k=int(cvg*len(conf)); return float(wrong[order[:k]].mean())
    # per-period true error with the fixed Jan/2019 n=4000 model (label-free deployment)
    per_period={}
    for t in times:
        Xe,ye=Xev[t]; pp=m.predict_proba(Xe); per_period[t]=float((pp.argmax(1)!=ye).mean())
    # label-free monitor score per period (nonconf + stability of the deployed model are fixed; use nonconf+delta)
    monit={}
    for t in times:
        Xe,ye=Xev[t]; pp=m.predict_proba(Xe); monit[t]=float((1-pp.max(1)).mean())
    res.update(dict(aurc=aurc,full_err=full_err,err50=err_at(0.5),err70=err_at(0.7),err90=err_at(0.9),
                    rc=rc.tolist(),delta=delta,per_period=per_period,monit=monit))
    # r2 predicting TRUE ERROR (1-acc) from label-free signals
    te=np.array([1-c["acc"] for c in cells])
    def r2t(cols):
        X=np.column_stack([np.ones(len(te))]+[M[:,k] for k in cols]); b,*_=np.linalg.lstsq(X,te,rcond=None)
        yh=X@b; return 1-((te-yh)**2).sum()/((te-te.mean())**2).sum()
    res.update(dict(err_r2_pred=r2t([0,2]),err_r2_expl=r2t([1]),err_r2_joint=r2t([0,1,2])))
    return cells,res

A,anA,tA=load_A(); cellsA,resA=analyze(A,anA,tA)
B,anB,tB=load_B(); cellsB,resB=analyze(B,anB,tB)
for name,res,times in [("A (Telegram)",resA,tA),("B (Reddit)",resB,tB)]:
    print(f"=== Domain {name} ===")
    for k in ["err_r2_pred","err_r2_expl","err_r2_joint","orth_corr","aurc","full_err","err70","err50"]:
        print(f"  {k}: {res[k]:.3f}")
    print("  per-period error:", {t:round(res['per_period'][t],3) for t in times})
    print("  label-free monitor:", {t:round(res['monit'][t],3) for t in times})
    print("  Spearman(monitor, true error across periods):",
          round(spearmanr([res['monit'][t] for t in times],[res['per_period'][t] for t in times])[0],3))

# ---- cross-domain collapse: coverage vs normalized drift at n=2400 ----
def collapse(cells,times,anchor):
    d=[c for c in cells if c["n"]==2400]
    dd=np.array([c["delta"] for c in d]); cov=np.array([c["cover"] for c in d])
    dn=(dd-dd.min())/(dd.max()-dd.min()+1e-9)
    return dn,cov
dnA,covA=collapse(cellsA,tA,anA); dnB,covB=collapse(cellsB,tB,anB)
# shared linear fit vs separate
allx=np.r_[dnA,dnB]; ally=np.r_[covA,covB]
Xs=np.column_stack([np.ones(len(allx)),allx]); bs,*_=np.linalg.lstsq(Xs,ally,rcond=None); yh=Xs@bs
r2_shared=1-((ally-yh)**2).sum()/((ally-ally.mean())**2).sum()
print(f"\nCross-domain collapse (coverage vs normalized drift): shared-fit R2 = {r2_shared:.3f}")

json.dump(dict(A=resA,B=resB,cellsA=cellsA,cellsB=cellsB,
               collapse=dict(dnA=dnA.tolist(),covA=covA.tolist(),dnB=dnB.tolist(),covB=covB.tolist(),r2=r2_shared)),
          open("monitor_results.json","w"),indent=2,default=float)
print("saved monitor_results.json")
