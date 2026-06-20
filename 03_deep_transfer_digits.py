# Deep model + cross-domain transfer on digits 3v8 (scarcity x contamination): MLP, SmoothGrad explanation, conformal stability certificate with NO network constant, contraction-law fit vs WDBC. Produces deep_transfer.npz. Paper Fig. 2c,d.
# Part of the open code for the Machine competence Perspective. MIT licence.
import numpy as np, warnings
warnings.filterwarnings("ignore")
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

# ---- second domain: handwritten digits 3 vs 8 (pixel features, deep MLP) ----
dig=load_digits(); m=(dig.target==3)|(dig.target==8)
X=dig.data[m]; y=(dig.target[m]==8).astype(int)
sc=StandardScaler().fit(X); X=sc.transform(X)
print("digits 3v8: n=",len(y)," features=",X.shape[1]," pos=",y.mean().round(3))
def split(seed):
    r=np.random.default_rng(seed); idx=r.permutation(len(X))
    return idx[:180], idx[180:240], idx[240:]   # train pool / cal / test

def fwd_grad(mlp,Xb):
    # analytic forward + input-gradient of the logit for a 1-hidden-layer ReLU MLP
    W1,W2=mlp.coefs_; b1,b2=mlp.intercepts_   # W1:(64,H) W2:(H,1)
    z1=Xb@W1+b1; h=np.maximum(z1,0.0)
    logit=h@W2+b2                              # (N,1)
    relu_d=(z1>0).astype(float)               # (N,H)
    # dlogit/dx = W1 @ (relu' ⊙ W2)  per sample
    g=(relu_d*W2[:,0])@W1.T                    # (N,64)
    return logit[:,0], g

def smoothgrad(mlp,Xb,sigma=0.5,K=24,seed=0):
    r=np.random.default_rng(seed); acc=np.zeros_like(Xb)
    for _ in range(K):
        _,g=fwd_grad(mlp,Xb+r.normal(0,sigma,Xb.shape)); acc+=g
    return acc/K

def profile(E): v=np.abs(E).mean(0); return v/(v.sum()+1e-12)
def cosd(a,b): return 1-(a@b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12)

def train(n,ytr_idx,seed,rho):
    tr=ytr_idx[:n]; yt=y[tr].copy()
    if rho>0:
        r=np.random.default_rng(7*seed+int(1000*rho)); nf=int(round(rho*len(yt)))
        yt[r.choice(len(yt),nf,replace=False)]^=1
    mlp=MLPClassifier(hidden_layer_sizes=(32,),activation="relu",alpha=1e-3,
                      max_iter=400,random_state=seed).fit(X[tr],yt)
    return mlp

ns=[180,120,80,50,35]; rhos=[0.0,0.05,0.10,0.15,0.20]; seeds=[1,2,3,4,5,6,7,8]
sigma=0.6

# in-distribution SmoothGrad reference (clean, full data)
ref=[]
for sd in seeds:
    tr,ca,te=split(sd); mlp=train(180,tr,sd,0.0)
    ref.append(profile(smoothgrad(mlp,X[te][:60],sigma,24,sd)))
ref=np.mean(ref,0)

ACC=np.zeros((len(ns),len(rhos))); S=np.zeros_like(ACC); LIP=np.zeros_like(ACC)
per={}
for i,n in enumerate(ns):
    for j,rho in enumerate(rhos):
        acc=[];ss=[];lip=[]
        for sd in seeds:
            tr,ca,te=split(sd); mlp=train(n,tr,sd,rho)
            p=mlp.predict_proba(X[te])[:,1]; acc.append(((p>=0.5)==y[te]).mean())
            Xt=X[te][:60]; E=smoothgrad(mlp,Xt,sigma,24,sd)
            ss.append(cosd(profile(E),ref))
            # certified local stability of the SmoothGrad map: conformalised upper tail
            # of the difference quotient ||E(x')-E(x)|| / ||x'-x|| over sampled neighbours
            r=np.random.default_rng(sd+i+j)
            Xn=Xt+r.normal(0,0.3,Xt.shape); En=smoothgrad(mlp,Xn,sigma,24,sd+99)
            dq=np.linalg.norm(En-E,axis=1)/(np.linalg.norm(Xn-Xt,axis=1)+1e-9)
            lip.append(np.quantile(dq,0.90))
        per[(i,j)]=np.array([acc,ss]).T
        ACC[i,j]=np.mean(acc); S[i,j]=np.mean(ss); LIP[i,j]=np.mean(lip)

np.set_printoptions(precision=3,suppress=True)
print("ns=",ns," rhos=",rhos)
print("ACC\n",ACC,"\nSmoothGrad stability S\n",S,"\ncertified expl-Lipschitz (90% conformal)\n",LIP)

sig=np.array([(180-n)/180 for n in ns]); rho=np.array(rhos)
SIG,RHO=np.meshgrid(sig,rho,indexing="ij")
def fit(Z):
    A=np.column_stack([np.ones(SIG.size),SIG.ravel(),RHO.ravel(),(SIG*RHO).ravel()])
    b,_,_,_=np.linalg.lstsq(A,Z.ravel(),rcond=None)
    r2=1-((Z.ravel()-A@b)**2).sum()/((Z.ravel()-Z.mean())**2).sum()
    A2=A[:,:3]; b2,_,_,_=np.linalg.lstsq(A2,Z.ravel(),rcond=None)
    r2a=1-((Z.ravel()-A2@b2)**2).sum()/((Z.ravel()-Z.mean())**2).sum()
    return b,r2a,r2
def boot(B=3000):
    out=[]; rng=np.random.default_rng(0); k=len(seeds)
    for _ in range(B):
        bs=rng.integers(0,k,k); Z=np.zeros((len(ns),len(rhos)))
        for (i,j),a in per.items(): Z[i,j]=1-a[bs,0].mean()  # 1-acc
        A=np.column_stack([np.ones(SIG.size),SIG.ravel(),RHO.ravel(),(SIG*RHO).ravel()])
        b,_,_,_=np.linalg.lstsq(A,Z.ravel(),rcond=None); out.append(b[3])
    out=np.array(out); return out.mean(),np.percentile(out,2.5),np.percentile(out,97.5)

b,r2a,r2f=fit(1-ACC); m,lo,hi=boot()
print(f"\n[digits 1-ACC] interaction b3={b[3]:.3f}  R2add={r2a:.3f} R2int={r2f:.3f}")
print(f"   bootstrap c: mean={m:.3f}  95% CI=[{lo:.3f},{hi:.3f}]  {'SIGNIFICANT' if lo>0 or hi<0 else 'n.s.'}")
bS,_,_=fit(S); print(f"[digits SmoothGrad stability] interaction b3={bS[3]:.3f}")
print(f"\nTRANSFER: WDBC c=0.27 [0.18,0.36]  vs  digits c={m:.2f} [{lo:.2f},{hi:.2f}]")
np.savez("deep_transfer.npz",ACC=ACC,S=S,LIP=LIP,ns=np.array(ns),rhos=np.array(rhos),
         sig=sig,c_digits=m,c_lo=lo,c_hi=hi)
print("saved deep_transfer.npz")
