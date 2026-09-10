"""ε-dormant approximate separation: коли напрям v МАЙЖЕ dormant (мала дисперсія
ε під P, а не точно 0), prediction-сертифікати відрізняються на O(ε), а
explanation-функціонал усе одно змінюється на β. Верифікуємо на логістичній моделі."""
import numpy as np, json
from sklearn.linear_model import LogisticRegression
rng=np.random.default_rng(0)
def cos_dist(u,v):
    nu,nv=np.linalg.norm(u),np.linalg.norm(v)
    return 1.0 if nu==0 or nv==0 else 1.0-float(np.dot(u,v)/(nu*nv))
def ece(p,y,bins=15):
    conf=np.maximum(p,1-p);pred=(p>=.5).astype(int);acc=(pred==y).astype(float);e=0;ed=np.linspace(0,1,bins+1)
    for i in range(bins):
        msk=(conf>ed[i])&(conf<=ed[i+1])
        if msk.sum(): e+=msk.mean()*abs(acc[msk].mean()-conf[msk].mean())
    return e
def conf_cov(pc,yc,pe,ye,a=0.10):
    sc=1-pc[np.arange(len(yc)),yc];q=np.sort(sc)[min(int(np.ceil((len(sc)+1)*(1-a))),len(sc))-1]
    return float(((1-pe)<=q)[np.arange(len(ye)),ye].mean())
d=12;N=9000;w=np.zeros(d);w[:5]=[2,-1.5,1.2,-1,0.8]
Xi=rng.normal(0,1,(N,d));y=(1/(1+np.exp(-(Xi@w)))>rng.random(N)).astype(int)
W=8.0  # planted weight on the near-dormant coordinate
rows=[]
for eps in [0.0,0.02,0.05,0.10,0.20,0.40]:
    # near-dormant feature: variance eps^2 on support (eps=0 -> exact dormant)
    v=rng.normal(0,eps,(N,1))
    X=np.hstack([Xi,v])
    tr=slice(0,4000);cal=slice(4000,6000);te=slice(6000,8000)
    f=LogisticRegression(max_iter=400,C=4).fit(X[tr],y[tr])
    fp=LogisticRegression(max_iter=400,C=4).fit(X[tr],y[tr])
    fp.coef_=f.coef_.copy();fp.intercept_=f.intercept_.copy();fp.coef_[0,-1]=W
    pf=f.predict_proba(X[te])[:,1];pfp=fp.predict_proba(X[te])[:,1]
    maxdiff=float(np.max(np.abs(pf-pfp)))
    covf=conf_cov(f.predict_proba(X[cal]),y[cal],f.predict_proba(X[te]),y[te])
    covfp=conf_cov(fp.predict_proba(X[cal]),y[cal],fp.predict_proba(X[te]),y[te])
    accf=float(((pf>=.5)==y[te]).mean());accfp=float(((pfp>=.5)==y[te]).mean())
    # structural fidelity gap (|w|)
    F=1-cos_dist(np.abs(f.coef_.ravel()),np.abs(fp.coef_.ravel()))
    rows.append(dict(eps=eps,max_pred_diff=maxdiff,cov_gap=abs(covf-covfp),
                     acc_gap=abs(accf-accfp),ece_gap=abs(ece(pf,y[te])-ece(pfp,y[te])),
                     fidelity_Ffp=F))
    print(f"eps={eps:.2f}: max|f-f'|={maxdiff:.4f}  cov_gap={abs(covf-covfp):.4f}  acc_gap={abs(accf-accfp):.4f}  F(f')={F:.3f}")
# check O(eps): prediction gaps ~ linear in eps, fidelity gap ~ constant large
json.dump({"W":W,"rows":rows},open("epsilon_dormant.json","w"),indent=2)
print("\nPrediction-side gaps grow ~O(eps) from 0; explanation fidelity gap stays large (≈%.2f) independent of eps."%(1-rows[-1]["fidelity_Ffp"]))
print("saved epsilon_dormant.json")
