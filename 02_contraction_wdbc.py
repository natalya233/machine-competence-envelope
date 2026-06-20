# Contraction law on WDBC (scarcity x contamination): joint certified margin g, super-additive interaction with seed bootstrap. Produces contraction.npz. Paper Fig. 2a,b.
# Part of the open code for the Machine competence Perspective. MIT licence.
import numpy as np, warnings
warnings.filterwarnings("ignore")
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
import shap

data = load_breast_cancer(); X, y = data.data, data.target
sc = StandardScaler().fit(X); Xs = sc.transform(X)
def split(seed):
    r=np.random.default_rng(seed); idx=r.permutation(len(Xs)); return idx[:300],idx[300:400],idx[400:]
def meanabs(phi):
    v=np.abs(phi).mean(0); return v/(v.sum()+1e-12)
def cos_dist(a,b): return 1-(a@b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12)

ns=[320,200,130,80,50]; rhos=[0.0,0.05,0.10,0.15,0.20]; seeds=[1,2,3,4,5,6,7,8]
cov_thr=0.85; S_thr=0.15

# clean full-data reference SHAP profile (in-distribution, uncontaminated)
ref=[]
for sd in seeds:
    tr,ca,te=split(sd); tr=tr[:320]
    clf=HistGradientBoostingClassifier(max_depth=3,max_iter=120,learning_rate=0.08,random_state=sd).fit(Xs[tr],y[tr])
    sv=shap.TreeExplainer(clf)(Xs[te][:90]); phi=sv.values; phi=phi[:,:,1] if phi.ndim==3 else phi
    ref.append(meanabs(phi))
ref=np.mean(ref,0)

# grids (mean over seeds) + per-seed store for bootstrap
ACC=np.zeros((len(ns),len(rhos))); COV=np.zeros_like(ACC); S=np.zeros_like(ACC); W=np.zeros_like(ACC)
per={}  # (i,j) -> list over seeds of (acc,cov,S,W)
for i,n in enumerate(ns):
    for j,rho in enumerate(rhos):
        acc=[];cov=[];ss=[];ww=[]
        for sd in seeds:
            r=np.random.default_rng(1000*sd+int(100*rho))
            tr,ca,te=split(sd); tr=tr[:n]
            ytr=y[tr].copy()
            # contamination: flip a fraction rho of training labels
            nf=int(round(rho*len(ytr)))
            if nf>0:
                fl=r.choice(len(ytr),nf,replace=False); ytr[fl]=1-ytr[fl]
            clf=HistGradientBoostingClassifier(max_depth=3,max_iter=120,learning_rate=0.08,random_state=sd).fit(Xs[tr],ytr)
            # conformal calibration on CLEAN cal labels
            pca=clf.predict_proba(Xs[ca]); ncal=1-pca[np.arange(len(ca)),y[ca]]
            qhat=np.quantile(ncal,0.90,method="higher")
            p=clf.predict_proba(Xs[te]); pred=p[:,1]>=0.5; yte=y[te]
            acc.append((pred==yte).mean())
            conf=np.maximum(p[:,1],1-p[:,1]); ww.append(((conf>0.8)&(pred!=yte)).mean())
            cov.append(((1-p[np.arange(len(te)),yte])<=qhat).mean())
            sv=shap.TreeExplainer(clf)(Xs[te][:90]); phi=sv.values; phi=phi[:,:,1] if phi.ndim==3 else phi
            ss.append(cos_dist(meanabs(phi),ref))
        per[(i,j)]=np.array([acc,cov,ss,ww]).T  # seeds x 4
        ACC[i,j]=np.mean(acc);COV[i,j]=np.mean(cov);S[i,j]=np.mean(ss);W[i,j]=np.mean(ww)

np.set_printoptions(precision=3,suppress=True)
print("ns(rows)=",ns," rhos(cols)=",rhos)
print("ACC\n",ACC,"\nCOV\n",COV,"\nS (expl-stability)\n",S,"\nW (confident-wrong)\n",W)

# scarcity index sigma in [0,1): how far below full data
sig=np.array([(320-n)/320 for n in ns])         # rows
rho=np.array(rhos)                               # cols
SIG,RHO=np.meshgrid(sig,rho,indexing="ij")

def fit_inter(Z):
    # OLS  Z = b0 + b1*sigma + b2*rho + b3*sigma*rho
    A=np.column_stack([np.ones(SIG.size),SIG.ravel(),RHO.ravel(),(SIG*RHO).ravel()])
    b,_,_,_=np.linalg.lstsq(A,Z.ravel(),rcond=None)
    pred=A@b; ss_res=((Z.ravel()-pred)**2).sum(); ss_tot=((Z.ravel()-Z.mean())**2).sum()
    # additive-only
    A2=A[:,:3]; b2,_,_,_=np.linalg.lstsq(A2,Z.ravel(),rcond=None); ss_res2=((Z.ravel()-A2@b2)**2).sum()
    return b,1-ss_res/ss_tot,1-ss_res2/ss_tot

def boot_b3(metric_idx,B=2000):
    # bootstrap over seeds: resample seed indices, recompute cell means, refit, collect b3
    ns_seed=len(seeds); out=[]
    rng=np.random.default_rng(0)
    for _ in range(B):
        bs=rng.integers(0,ns_seed,ns_seed)
        Z=np.zeros((len(ns),len(rhos)))
        for (i,j),arr in per.items(): Z[i,j]=arr[bs,metric_idx].mean()
        A=np.column_stack([np.ones(SIG.size),SIG.ravel(),RHO.ravel(),(SIG*RHO).ravel()])
        b,_,_,_=np.linalg.lstsq(A,Z.ravel(),rcond=None); out.append(b[3])
    out=np.array(out); return out.mean(),np.percentile(out,2.5),np.percentile(out,97.5)

for name,Z,idx in [("S (expl-stability)",S,2),("W (confident-wrong)",W,3),("1-ACC",1-ACC,0)]:
    b,r2f,r2a=fit_inter(Z); m,lo,hi=boot_b3(idx)
    print(f"\n[{name}]  b0={b[0]:.3f} b1(sigma)={b[1]:.3f} b2(rho)={b[2]:.3f} b3(sigma*rho)={b[3]:.3f}")
    print(f"   R2 additive={r2a:.3f}  R2 with-interaction={r2f:.3f}  (gain {r2f-r2a:+.3f})")
    print(f"   bootstrap b3: mean={m:.3f}  95% CI=[{lo:.3f},{hi:.3f}]  {'INTERACTION SIGNIFICANT' if lo>0 or hi<0 else 'n.s.'}")

# joint certified margin g = min(coverage slack, stability slack) rescaled to [~ -1,1]
g=np.minimum((COV-cov_thr)/cov_thr,(S_thr-S)/S_thr)
print("\njoint certified margin g (rows=scarcity, cols=contam)\n",g)
bg,r2gf,r2ga=fit_inter(g)
print(f"margin g: b0={bg[0]:.3f} b1={bg[1]:.3f} b2={bg[2]:.3f} b3={bg[3]:.3f}  R2add={r2ga:.3f} R2int={r2gf:.3f}")
np.savez("contraction.npz",ACC=ACC,COV=COV,S=S,W=W,g=g,ns=np.array(ns),rhos=np.array(rhos),
         sig=sig,coef_S=fit_inter(S)[0],coef_W=fit_inter(W)[0],coef_g=bg)
print("saved contraction.npz")
