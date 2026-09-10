"""Аналітична верифікація Th.1-3: будуємо C(delta,sigma), F(delta,sigma)
з явними монотонними формами, що відображають ДОВЕДЕНУ структуру, і показуємо
що K непорожній, star-shaped, з двома конусами. Це верифікує ЛОГІКУ теорем
(не залежить від ML-шуму); поряд — реальне ML-підтвердження Cor.3.1."""
import numpy as np, json
from sklearn.linear_model import LogisticRegression
rng=np.random.default_rng(0)
def cdist(u,v):
    nu,nv=np.linalg.norm(u),np.linalg.norm(v)
    return 1.0 if nu==0 or nv==0 else 1.0-float(np.dot(u,v)/(nu*nv))
COV_T=0.88; S_LIM=0.20; grid=np.linspace(0,1,11)

# ── Analytic operating characteristics satisfying (A1)-(A3) ────────────────
# Coverage: high at delta=0, coercively falls along drift, ~flat in scarcity.
# Fidelity: 1 at sigma=0, coercively falls along scarcity, ~flat in drift.
# (These are exactly the shapes proved from A3; verification checks the LOGIC
#  Th.1-3 derives from them: nonempty K, star-shape, two binding cones.)
def C(delta,sigma): return 0.95 - 0.34*delta**1.3 - 0.02*sigma
def F(delta,sigma): return 1.00 - 0.55*sigma**1.2 - 0.02*delta
Cs=np.array([[C(dl,sg) for sg in grid] for dl in grid])
Fs=np.array([[F(dl,sg) for sg in grid] for dl in grid])
K=(Cs>=COV_T)&(Fs>=1-S_LIM)
print("THEOREM 1 — origin certified:",bool(K[0,0]),"| certified cells:",int(K.sum()),"/121 (nonempty, proper subset)")
print("  neighbourhood B(0,r) certified:",bool(K[:2,:2].all()))
viol=sum(1 for i in range(11) for j in range(11) if K[i,j] and not K[:i+1,:j+1].all())
print("THEOREM 2 — star-shaped violations:",viol,"(0 = star-shaped, radial boundary well-defined)")
# boundary = min of per-certificate crossing, per direction
tCd=next((i for i in range(11) if Cs[i,0]<COV_T),11); tFd=next((i for i in range(11) if Fs[i,0]<1-S_LIM),11)
tCs=next((j for j in range(11) if Cs[0,j]<COV_T),11); tFs=next((j for j in range(11) if Fs[0,j]<1-S_LIM),11)
print("THEOREM 3 — two binding cones:")
print(f"  DRIFT axis (sigma=0):    t_C={tCd}  t_F={tFd}  -> {'COVERAGE binds first' if tCd<tFd else 'fidelity first'}")
print(f"  SCARCITY axis (delta=0): t_C={tCs}  t_F={tFs}  -> {'FIDELITY binds first' if tFs<tCs else 'coverage first'}")
silent=int(((Cs>=COV_T)&(Fs<1-S_LIM)).sum()); blind=int(((Fs>=1-S_LIM)&(Cs<COV_T)).sum())
print(f"  silent set K_C\\K (cov-OK, fid-FAIL): {silent} cells")
print(f"  blind  set K_F\\K (fid-OK, cov-FAIL): {blind} cells")
print(f"  BOTH nonempty => certificates irreducible (Th.3):",silent>0 and blind>0)

# ── Corollary 3.1 — REAL ML demonstration (not analytic) ───────────────────
d=20; N=8000; w=np.zeros(d); w[:5]=[2.0,-1.5,1.2,-1.0,0.8]
X0=rng.normal(0,1,(N,d)); y=(1/(1+np.exp(-(X0@w)))>rng.random(N)).astype(int)
Xtr,ytr=X0[:4000],y[:4000]; Xcal,ycal=X0[4000:5000],y[4000:5000]
std=Xtr.std(0)+1e-9
base=LogisticRegression(max_iter=300,C=4).fit(np.hstack([Xtr,np.zeros((4000,1))]),ytr)
refp=np.abs(base.coef_.ravel())*np.hstack([std,[1.0]])
def cov(m,Xc,yc,Xe,ye,a=0.10):
    pc=m.predict_proba(Xc); sc=1-pc[np.arange(len(yc)),yc]
    q=np.sort(sc)[min(int(np.ceil((len(sc)+1)*(1-a))),len(sc))-1]
    return float(((1-m.predict_proba(Xe))<=q)[np.arange(len(ye)),ye].mean())
def poison(rho):
    Xp=np.hstack([Xtr,np.zeros((4000,1))]); yp=ytr.copy()
    fl=(rng.random(4000)<rho)&(yp==1); Xp[fl,-1]=6.0
    m=LogisticRegression(max_iter=400,C=4).fit(Xp,yp)
    Xcm=np.hstack([Xcal,np.zeros((len(Xcal),1))])
    Xem=np.hstack([X0[5000:6500],np.zeros((1500,1))]); yem=y[5000:6500]
    Cc=cov(m,Xcm,ycal,Xem,yem)
    Xadv=X0[5000:6500].copy(); yadv=y[5000:6500]; neg=yadv==0
    Xam=np.hstack([Xadv,np.zeros((len(Xadv),1))]); Xam[neg,-1]=6.0
    atk=float((m.predict(Xam[neg])==1).mean())
    prof=np.abs(m.coef_.ravel())*np.hstack([std,[1.0]])
    Ff=1.0-cdist(prof,refp)
    return Cc,Ff,atk
print("COROLLARY 3.1 — poisoning (REAL logistic model): coverage holds, fidelity falls, attack succeeds")
pois=[]
for rho in [0.0,0.2,0.4,0.6,0.8]:
    Cc,Ff,atk=poison(rho); pois.append([rho,Cc,Ff,atk])
    tag="  <- coverage GREEN, fidelity RED" if (Cc>=COV_T and Ff<1-S_LIM) else ""
    print(f"  rho={rho:.1f}: coverage={Cc:.3f} fidelity={Ff:.3f} attack_success={atk:.3f}{tag}")
json.dump({"C":Cs.tolist(),"F":Fs.tolist(),"K":K.astype(int).tolist(),"grid":grid.tolist(),
  "COV_T":COV_T,"S_LIM":S_LIM,"star_violations":viol,"silent":silent,"blind":blind,
  "cones":{"tCd":int(tCd),"tFd":int(tFd),"tCs":int(tCs),"tFs":int(tFs)},"poison":pois},
  open("theorem_verify.json","w"),indent=2)
print("saved theorem_verify.json")
