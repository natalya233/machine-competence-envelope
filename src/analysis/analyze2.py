import warnings, json, numpy as np
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
rng = np.random.default_rng(0)

comp = np.load("_compound.npy", allow_pickle=True).item()
resA, resB, ng, rho = comp["resA"], comp["resB"], comp["ng"], comp["rho"]

def err_interaction(res, n_grid, rho_grid, nboot=3000):
    nmax = max(n_grid)
    sig = lambda n: (np.log(nmax)-np.log(n))/(np.log(nmax)-np.log(min(n_grid)))
    rn  = lambda r: r/max(rho_grid)
    rows=[]
    for nn in n_grid:
        for r in rho_grid:
            rows.append((sig(nn), rn(r), res[(nn,r)]["err"]))
    R=np.array(rows)
    Xd=np.column_stack([np.ones(len(R)),R[:,0],R[:,1],R[:,0]*R[:,1]])
    beta,*_=np.linalg.lstsq(Xd,R[:,2],rcond=None)
    cs=[]
    for _ in range(nboot):
        ii=rng.integers(0,len(R),len(R)); bb,*_=np.linalg.lstsq(Xd[ii],R[ii,2],rcond=None); cs.append(bb[3])
    return beta[3], np.percentile(cs,[2.5,97.5])

cA,ciA = err_interaction(resA,ng,rho)
cB,ciB = err_interaction(resB,ng,rho)
print("ERROR-interaction (super-additive if >0):")
print("  Telegram (A): c=%.3f CI[%.3f,%.3f]"%(cA,ciA[0],ciA[1]))
print("  Reddit   (B): c=%.3f CI[%.3f,%.3f]"%(cB,ciB[0],ciB[1]))

# ---------- Clinical tabular contrast: WDBC + gradient-boosted trees ----------
data = load_breast_cancer(); Xall, yall = data.data, data.target
def clinical_compound(n_grid, rho_grid, seeds=10):
    res={}
    for nn in n_grid:
        for r in rho_grid:
            errs=[]
            for s in range(seeds):
                rs=np.random.default_rng(11*s+nn+int(1000*r))
                idx=rs.permutation(len(Xall))
                tr=idx[:nn]; ev=idx[nn:nn+150]
                if len(ev)<50: ev=idx[nn:]
                ytr=yall[tr].copy()
                if r>0:
                    fl=rs.random(len(ytr))<r; ytr[fl]=1-ytr[fl]
                m=GradientBoostingClassifier(n_estimators=120,max_depth=3,learning_rate=0.1,random_state=s)
                m.fit(Xall[tr],ytr)
                errs.append(1-m.score(Xall[ev],yall[ev]))
            res[(nn,r)]=dict(err=float(np.mean(errs)))
    return res
ng_c=[40,70,120,200,350]; rho_c=[0.0,0.1,0.2,0.3]
resC=clinical_compound(ng_c,rho_c)
cC,ciC=err_interaction(resC,ng_c,rho_c)
print("  Clinical WDBC (tabular, GBDT): c=%.3f CI[%.3f,%.3f]"%(cC,ciC[0],ciC[1]))

# ---------- discriminator-AUC drift proxy for Domain A months ----------
# (re-load lightweight A/B text via saved envelope npy not enough; recompute quickly)
import pandas as pd, re, glob, os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
URL=re.compile(r"https?://\S+"); cl=lambda s:URL.sub(" ",str(s)).replace("\\n"," ")
A=pd.read_csv("data/telegram_2026.csv",usecols=["date","text","views","forwards"],engine="python",on_bad_lines="skip")
A["date"]=pd.to_datetime(A["date"],errors="coerce",utc=True); A=A.dropna(subset=["date","text","views","forwards"])
A=A[(A["views"]>0)&(A["text"].str.len()>20)]; A["text"]=A["text"].map(cl); A["month"]=A["date"].dt.month
vec=TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),min_df=5,max_features=4000,sublinear_tf=True)
vec.fit(A["text"].sample(15000,random_state=0))
base=A[A["month"]==1]["text"]
drift_auc={}
for m in [1,2,3,4,5]:
    other=A[A["month"]==m]["text"]
    nb=min(len(base),len(other),3000)
    Xd=vec.transform(pd.concat([base.sample(nb,random_state=1),other.sample(nb,random_state=2)]))
    yd=np.r_[np.zeros(nb),np.ones(nb)]
    ii=np.random.default_rng(3).permutation(len(yd)); ntr=int(.7*len(yd))
    clf=LogisticRegression(max_iter=200,C=2).fit(Xd[ii[:ntr]],yd[ii[:ntr]])
    auc=roc_auc_score(yd[ii[ntr:]],clf.predict_proba(Xd[ii[ntr:]])[:,1])
    drift_auc[m]=float(2*abs(auc-0.5))   # 0 = no drift, 1 = fully separable
print("\nDomain A temporal drift (discriminator |2(AUC-0.5)|): ",{k:round(v,3) for k,v in drift_auc.items()})

out=json.load(open("real_results.json"))
out.update(dict(
  A_err_c=float(cA),A_err_ci=[float(x) for x in ciA],
  B_err_c=float(cB),B_err_ci=[float(x) for x in ciB],
  C_err_c=float(cC),C_err_ci=[float(x) for x in ciC],
  A_drift_auc={int(k):float(v) for k,v in drift_auc.items()},
  resC={f"{nn}_{r}":resC[(nn,r)]["err"] for nn in ng_c for r in rho_c}, ng_c=ng_c, rho_c=rho_c,
))
json.dump(out,open("real_results.json","w"),indent=2)
np.save("_drift.npy",{"drift_auc":drift_auc},allow_pickle=True)
print("updated real_results.json")
