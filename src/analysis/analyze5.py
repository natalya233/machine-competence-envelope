import warnings, re, glob, json, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
UP="data"
URL=re.compile(r"https?://\S+"); cl=lambda s:URL.sub(" ",str(s)).replace("\\n"," ")

def disc_delta(Za,Zb,seed=3):
    import scipy.sparse as sp
    nb=min(Za.shape[0],Zb.shape[0],3000)
    if sp.issparse(Za):
        X=sp.vstack([Za[:nb],Zb[:nb]])
    else:
        # augment dense features with squares so a linear detector also sees variance/scale shifts
        A=np.asarray(Za[:nb]); B=np.asarray(Zb[:nb])
        X=np.vstack([np.hstack([A,A**2]), np.hstack([B,B**2])])
    y=np.r_[np.zeros(nb),np.ones(nb)]
    ii=np.random.default_rng(seed).permutation(len(y)); k=int(.7*len(y))
    c=LogisticRegression(max_iter=200,C=2).fit(X[ii[:k]],y[ii[:k]])
    return float(2*abs(roc_auc_score(y[ii[k:]],c.predict_proba(X[ii[k:]])[:,1])-0.5))

def run_dataset(name, dfT_text, X_tab, y, period, anchor_periods, deploy_periods):
    """text modality + tabular modality, temporal drift, late-fusion comparison."""
    vec=TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),min_df=5,max_features=5000,sublinear_tf=True)
    anc = np.isin(period,anchor_periods)
    vec.fit(dfT_text[anc].sample(min(15000,anc.sum()),random_state=0))
    XT=vec.transform(dfT_text); XX=np.asarray(X_tab,dtype=float)
    # standardize tabular on anchor
    mu=np.nanmean(XX[anc],0); sd=np.nanstd(XX[anc],0)+1e-9
    XX=np.nan_to_num((XX-mu)/sd)
    # train per-modality on anchor
    ia=np.where(anc)[0]; rng=np.random.default_rng(0); rng.shuffle(ia)
    tr=ia[:int(.8*len(ia))]; ca=ia[int(.8*len(ia)):]
    fT=LogisticRegression(max_iter=300,C=4).fit(XT[tr],y[tr])
    fX=LogisticRegression(max_iter=300,C=1).fit(XX[tr],y[tr])
    # early fusion (concat) baseline
    import scipy.sparse as sp
    def concat(a,b): return sp.hstack([a, sp.csr_matrix(b)]).tocsr()
    fE=LogisticRegression(max_iter=300,C=4).fit(concat(XT[tr],XX[tr]),y[tr])
    anchorT=dfT_text[anc]; XT_anc=XT[anc]; XX_anc=XX[anc]
    def eval_period(mask, degrade=None, noise=3.0):
        idx=np.where(mask)[0]
        pt=fT.predict_proba(XT[idx])[:,1]; px=fX.predict_proba(XX[idx])[:,1]
        XTe, XXe = XT[idx], XX[idx]
        # modality-specific degradation (sensor outage / covariate shock)
        if degrade=="text":
            Xn=XTe.copy(); Xn.data=Xn.data*0 + rng.normal(0,0.5,size=Xn.data.shape); 
            pt=fT.predict_proba(Xn)[:,1]; XTe=Xn
        if degrade=="tab":
            XXe=XXe*1.6 + 2.0 + rng.normal(0,noise,size=XXe.shape); px=fX.predict_proba(XXe)[:,1]
        yt=y[idx]
        cT=np.maximum(pt,1-pt); cX=np.maximum(px,1-px)          # per-example confidence
        dT=disc_delta(XT_anc, XTe); dX=disc_delta(XX_anc, XXe)  # label-free per-modality drift
        rT=np.exp(-3*dT); rX=np.exp(-3*dX)                      # envelope proximity (label-free)
        def err(p): return float(((p>=.5).astype(int)!=yt).mean())
        e_text=err(pt); e_tab=err(px)
        e_naive=err(0.5*pt+0.5*px)
        wc=cT+cX+1e-9; e_conf=err((cT*pt+cX*px)/wc)             # confidence-gated
        wT=rT*cT; wX=rX*cX; ws=wT+wX+1e-9
        e_comp=err((wT*pt+wX*px)/ws)                            # competence-gated (envelope x confidence)
        e_early=err(fE.predict_proba(concat(XTe,XXe))[:,1])
        e_oracle=min(e_text,e_tab)
        return dict(text=e_text,tab=e_tab,naive=e_naive,conf=e_conf,comp=e_comp,early=e_early,
                    oracle=e_oracle,dT=dT,dX=dX)
    # natural temporal drift across deploy periods
    nat=[]
    for p in deploy_periods:
        m=(period==p)
        if m.sum()<200: continue
        nat.append(eval_period(m))
    natavg={k:float(np.mean([d[k] for d in nat])) for k in nat[0]}
    # controlled modality degradation on a clean holdout (anchor cal)
    calmask=np.zeros(len(y),bool); calmask[ca]=True
    deg_clean=eval_period(calmask); deg_text=eval_period(calmask,degrade="text"); deg_tab=eval_period(calmask,degrade="tab")
    return dict(name=name, natural=natavg, per_period=nat,
                deg=dict(clean=deg_clean,text_out=deg_text,tab_out=deg_tab))

# ================= Dataset 1: Reddit climate (text + tabular metadata) =================
rows=[]
for f in glob.glob("climate/Climate_Dataset/Climate_CSV/*.csv"):
    try: rows.append(pd.read_csv(f,sep=";",engine="python",on_bad_lines="skip",encoding="latin-1"))
    except: pass
R=pd.concat(rows,ignore_index=True); R.columns=[c.strip() for c in R.columns]
def num(s): return pd.to_numeric(R[s].astype(str).str.replace(",",""),errors="coerce")
R["up"]=num("# upvotes"); R["date"]=pd.to_datetime(R["Date"],errors="coerce",dayfirst=True)
R["text"]=R["Post text"].map(cl)
R=R.dropna(subset=["up","date","text"]); R=R[R["text"].str.len()>10]; R["year"]=R["date"].dt.year
R=R[R["year"].between(2017,2023)]
lo,hi=R["up"].quantile([1/3,2/3]); R=R[(R["up"]<=lo)|(R["up"]>=hi)].copy(); R["y"]=(R["up"]>=hi).astype(int)
tab=np.column_stack([num("Community members").reindex(R.index), num("Years of membership").reindex(R.index),
                     num("# Post Karma").reindex(R.index), num("# Comment Karma").reindex(R.index),
                     num("# Awardee Karma").reindex(R.index), R["text"].str.len().values,
                     pd.Categorical(R["Post type"]).codes])
res1=run_dataset("Reddit climate (text + author/community tabular)", R["text"], tab, R["y"].values,
                 R["year"].values, [2017,2018,2019], [2020,2021,2022,2023])

# ================= Dataset 2: Telegram kiev1 (text + behavioural) =================
K=pd.read_csv(f"{UP}/kiev1_reactions_expanded.csv",engine="python",on_bad_lines="skip")
K["date"]=pd.to_datetime(K["date"],errors="coerce",utc=True); K=K.dropna(subset=["date","text"])
cols=list(K.columns); emo=lambda ch: next((c for c in cols if f"emoticon='{ch}'" in c),None)
dis=[emo(c) for c in ["😢","😭","😨","💔","🤬","😱"]]; sup=[emo(c) for c in ["👍","❤","🔥","🙏","🥰","👏","🫡"]]
dis=[c for c in dis if c]; sup=[c for c in sup if c]
K[dis+sup]=K[dis+sup].apply(pd.to_numeric,errors="coerce").fillna(0)
ds=K[dis].sum(axis=1); sp_=K[sup].sum(axis=1)
K=K[(ds+sp_)>=3].copy(); K["y"]=(ds[(ds+sp_)>=3]>sp_[(ds+sp_)>=3]).astype(int).values
K["text"]=K["text"].map(cl); K["views"]=pd.to_numeric(K["views"],errors="coerce")
K["year"]=K["date"].dt.year
# balance 50/50 per the whole set
pos=K[K["y"]==1]; neg=K[K["y"]==0].sample(len(pos),random_state=0); K=pd.concat([pos,neg]).sort_values("date")
Ktab=np.column_stack([np.log1p(K["views"].fillna(0).values), K["text"].str.len().values,
                      K["date"].dt.hour.values, K["date"].dt.dayofweek.values])
res2=run_dataset("Telegram kiev1 (text + behavioural)", K["text"], Ktab, K["y"].values,
                 K["year"].values, [2022], [2023,2024,2025,2026])

for res in (res1,res2):
    print("\n==============",res["name"],"==============")
    n=res["natural"]
    print(" NATURAL temporal drift (avg error):")
    print("   text=%.3f tab=%.3f | early=%.3f naive=%.3f conf-gate=%.3f  COMP-gate=%.3f | oracle=%.3f"%(
        n["text"],n["tab"],n["early"],n["naive"],n["conf"],n["comp"],n["oracle"]))
    d=res["deg"]
    print(" MODALITY DEGRADATION (error):")
    for k,lab in [("clean","clean      "),("text_out","text outage"),("tab_out","tab outage ")]:
        c=d[k]
        print(f"   {lab}: text=%.3f tab=%.3f naive=%.3f conf-gate=%.3f COMP-gate=%.3f  (dT=%.2f dX=%.2f)"%(
            c["text"],c["tab"],c["naive"],c["conf"],c["comp"],c["dT"],c["dX"]))

json.dump({"res1":res1,"res2":res2},open("multimodal.json","w"),indent=2,default=float)
print("\nsaved multimodal.json")
