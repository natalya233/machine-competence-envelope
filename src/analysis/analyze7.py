import warnings, json, numpy as np
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
rng=np.random.default_rng(0)

D=np.load("data/mosi_features.npz")
y_cont=D["y"]; keep=np.abs(y_cont)>1e-6
Xm={"text":D["X_text"][keep],"audio":D["X_audio"][keep],"vision":D["X_vision"][keep]}
y=(y_cont[keep]>0).astype(int); split=D["split"][keep]
tr=split==0; ca=split==1; te=split==2
print("N=",len(y),"| train",tr.sum(),"valid",ca.sum(),"test",te.sum(),"| pos-rate",round(float(y.mean()),3))

# standardize on train
for k in Xm:
    mu=Xm[k][tr].mean(0); sd=Xm[k][tr].std(0)+1e-9; Xm[k]=(Xm[k]-mu)/sd

mods=["text","audio","vision"]
F={k:LogisticRegression(max_iter=500,C=1.0).fit(Xm[k][tr],y[tr]) for k in mods}
for k in mods:
    acc=float((F[k].predict(Xm[k][te])==y[te]).mean())
    print(f"  single-modality {k}: test acc={acc:.3f}")

def disc_delta(Za,Zb,seed=3):
    A=np.hstack([Za,Za**2]); Bb=np.hstack([Zb,Zb**2])
    nb=min(len(A),len(Bb))
    X=np.vstack([A[:nb],Bb[:nb]]); yy=np.r_[np.zeros(nb),np.ones(nb)]
    ii=np.random.default_rng(seed).permutation(len(yy)); k=int(.7*len(yy))
    c=LogisticRegression(max_iter=300,C=2).fit(X[ii[:k]],yy[ii[:k]])
    return float(2*abs(roc_auc_score(yy[ii[k:]],c.predict_proba(X[ii[k:]])[:,1])-0.5))

def eval_scen(degrade=None, noise=2.0, seeds=5):
    out={}
    accs={m:[] for m in ["naive","conf","comp","oracle"]+mods}; dels={m:[] for m in mods}
    for s in range(seeds):
        r=np.random.default_rng(s)
        Xe={k:Xm[k][te].copy() for k in mods}
        if degrade in mods:
            Xe[degrade]=Xe[degrade]*1.6+2.0+r.normal(0,noise,size=Xe[degrade].shape)
        p={k:F[k].predict_proba(Xe[k])[:,1] for k in mods}
        conf={k:np.maximum(p[k],1-p[k]) for k in mods}
        delta={k:disc_delta(Xm[k][tr],Xe[k],seed=s) for k in mods}
        rel={k:np.exp(-3*delta[k]) for k in mods}
        yt=y[te]
        err=lambda q: float(((q>=.5).astype(int)!=yt).mean())
        for k in mods: accs[k].append(err(p[k])); dels[k].append(delta[k])
        accs["naive"].append(err(np.mean([p[k] for k in mods],0)))
        wc=sum(conf.values()); accs["conf"].append(err(sum(conf[k]*p[k] for k in mods)/wc))
        w={k:rel[k]*conf[k] for k in mods}; ws=sum(w.values())
        accs["comp"].append(err(sum(w[k]*p[k] for k in mods)/ws))
        accs["oracle"].append(min(err(p[k]) for k in mods))
    for k in accs: out[k]=float(np.mean(accs[k]))
    out["delta"]={k:float(np.mean(dels[k])) for k in mods}
    return out

scen={}
scen["clean"]=eval_scen(None)
for m in mods: scen[m+"_out"]=eval_scen(m)
print("\nscenario | text  audio vision | naive  conf  COMP | oracle | deltas")
for s,r in scen.items():
    d=r["delta"]
    print(f"{s:11s}| {r['text']:.3f} {r['audio']:.3f} {r['vision']:.3f} | {r['naive']:.3f} {r['conf']:.3f} {r['comp']:.3f} | {r['oracle']:.3f} | " +
          " ".join(f"{k[0]}:{d[k]:.2f}" for k in mods))

json.dump(scen,open("mosi_results.json","w"),indent=2)
print("\nsaved mosi_results.json")
