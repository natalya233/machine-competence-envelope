# Competence envelope on WDBC (drift x scarcity): conformal coverage, SHAP Shapley-efficiency, explanation-stability. Produces exp_results.npz. Paper Fig. 1.
# Part of the open code for the Machine competence Perspective. MIT licence.
import numpy as np, warnings, time
warnings.filterwarnings("ignore")
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
import shap

data = load_breast_cancer(); X, y = data.data, data.target
sc = StandardScaler().fit(X); Xs = sc.transform(X)
d = Xs[y==1].mean(0) - Xs[y==0].mean(0); d = d/np.linalg.norm(d)
def split(seed):
    r=np.random.default_rng(seed); idx=r.permutation(len(Xs)); return idx[:300],idx[300:400],idx[400:]
def meanabs(phi):
    v=np.abs(phi).mean(0); return v/(v.sum()+1e-12)
def cos_dist(a,b): return 1-(a@b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12)

drifts=[0.0,0.6,1.2,1.8,2.4,3.0]; ns=[320,200,130,80,50]; seeds=[1,2,3,4,5,6]

# reference profile + Shapley-efficiency check in MARGIN space
ref=[]; effmax=0
for sd in seeds:
    tr,ca,te=split(sd); tr=tr[:320]
    clf=HistGradientBoostingClassifier(max_depth=3,max_iter=120,learning_rate=0.08,random_state=sd).fit(Xs[tr],y[tr])
    ex=shap.TreeExplainer(clf); sv=ex(Xs[te][:90]); phi=sv.values; base=sv.base_values
    if phi.ndim==3: phi=phi[:,:,1]; base=base[:,1] if np.ndim(base)>1 else base
    margin=clf.decision_function(Xs[te][:90])
    effmax=max(effmax,np.max(np.abs(phi.sum(1)+np.asarray(base)-margin)))
    ref.append(meanabs(phi))
ref=np.mean(ref,0)
print(f"Shapley efficiency (margin space) max |sum phi + base - margin| = {effmax:.2e}")

R=np.zeros((len(ns),len(drifts)));COV=np.zeros_like(R);S=np.zeros_like(R);SF=np.zeros_like(R)
for a,n in enumerate(ns):
    for b,dr in enumerate(drifts):
        rr=[];cov=[];ss=[];sf=[]
        for sd in seeds:
            tr,ca,te=split(sd); tr=tr[:n]
            clf=HistGradientBoostingClassifier(max_depth=3,max_iter=120,learning_rate=0.08,random_state=sd).fit(Xs[tr],y[tr])
            pca=clf.predict_proba(Xs[ca]); ncal=1-pca[np.arange(len(ca)),y[ca]]
            qhat=np.quantile(ncal,0.90,method="higher")
            Xte=Xs[te]+dr*d; yte=y[te]; p=clf.predict_proba(Xte); pred=p[:,1]>=0.5
            rr.append((pred==yte).mean())
            conf=np.maximum(p[:,1],1-p[:,1]); sf.append(((conf>0.8)&(pred!=yte)).mean())
            cov.append(((1-p[np.arange(len(te)),yte])<=qhat).mean())
            sv=shap.TreeExplainer(clf)(Xte[:90]); phi=sv.values; phi=phi[:,:,1] if phi.ndim==3 else phi
            ss.append(cos_dist(meanabs(phi),ref))
        R[a,b]=np.mean(rr);COV[a,b]=np.mean(cov);S[a,b]=np.mean(ss);SF[a,b]=np.mean(sf)

np.set_printoptions(precision=3,suppress=True)
print("rows=scarcity n=",ns,"cols=drift=",drifts)
print("R\n",R,"\nCOV\n",COV,"\nS\n",S,"\nSF\n",SF)

# envelope thresholds
cov_thr=0.85; S_thr=0.15
env=(COV>=cov_thr)&(S<=S_thr)
print("Envelope (COV>=%.2f & S<=%.2f):"%(cov_thr,S_thr),"\n",env.astype(int))
print("certified fraction of grid = %.0f%%"%(100*env.mean()))

# contraction: drift tolerance (max drift with COV>=thr) at full vs scarce data
def drift_tol(row):
    ok=np.where(COV[row]>=cov_thr)[0]
    return drifts[ok.max()] if len(ok) else -1
print("drift tolerance @ n=320:",drift_tol(0)," @ n=130:",drift_tol(2))

# early warning along full-data row
row=0
print("\nEARLY WARNING (n=320), per drift:")
for b,dr in enumerate(drifts):
    print(f"  drift={dr:.1f}  COV={COV[row,b]:.3f}  silent_err={SF[row,b]:.3f}  cert={'OK' if COV[row,b]>=cov_thr else 'WITHHELD'}")
np.savez("exp_results.npz",R=R,COV=COV,S=S,SF=SF,drifts=np.array(drifts),ns=np.array(ns),
         eff=effmax,env=env,cov_thr=cov_thr,S_thr=S_thr)
print("saved exp_results.npz")
