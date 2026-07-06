import warnings, re, glob, json, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
rng = np.random.default_rng(0)
UP = "data"
URL = re.compile(r"https?://\S+"); cl = lambda s: URL.sub(" ", str(s)).replace("\\n", " ")

def cdist(u, v):
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    return 1.0 if nu==0 or nv==0 else 1.0 - float(np.dot(u, v)/(nu*nv))

def prune(coef, p):
    """Зануляємо частку p найменших |w| — model compression."""
    c = coef.copy()
    if p <= 0: return c
    thr = np.quantile(np.abs(c), p)
    c[np.abs(c) < thr] = 0.0
    return c

# ===== завантаження тексту (Telegram) =====
A = pd.read_csv(f"{UP}/telegram_2026.csv", usecols=["date","text","views","forwards"],
                engine="python", on_bad_lines="skip")
A["date"] = pd.to_datetime(A["date"], errors="coerce", utc=True)
A = A.dropna(subset=["text","views","forwards"])
A = A[(A["views"]>0) & (A["text"].str.len()>20)]
A["fr"] = A["forwards"]/A["views"]; A["text"] = A["text"].map(cl)
lo, hi = A["fr"].quantile([1/3, 2/3])
A = A[(A["fr"]<=lo)|(A["fr"]>=hi)].copy(); A["y"] = (A["fr"]>=hi).astype(int)
A["month"] = pd.to_datetime(A["date"]).dt.month
A = A[A["month"].isin([1,2,3,4,5])].reset_index(drop=True)
ref = A[A["month"]==1]
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), min_df=5,
                      max_features=4000, sublinear_tf=True)
vec.fit(ref["text"].sample(min(12000,len(ref["text"])), random_state=0))
Xall = vec.transform(A["text"]); yall = A["y"].values
std = np.asarray(Xall.power(2).mean(0)).ravel()**0.5
Xref = vec.transform(ref["text"]); yref = ref["y"].values

# Базова (повна) модель + reference profile
base = LogisticRegression(max_iter=300, C=4).fit(Xref, yref)
ref_prof = np.abs(base.coef_.ravel()) * std

print("="*60)
print("EFFECT 1 — Pruning × Scarcity (Telegram)")
print("="*60)
# Eval на місяці 3 (помірний дрейф)
eval_mask = A["month"] == 3
Xev = Xall[eval_mask.values]; yev = yall[eval_mask.values]
prune_levels = [0.0, 0.3, 0.5, 0.7, 0.85, 0.92]
n_grid       = [300, 600, 1200, 2400, 4000]
seeds = 4
res_prune = {}
for p in prune_levels:
    for n in n_grid:
        covs=[]; accs=[]; Ss=[]
        for s in range(seeds):
            rs = np.random.default_rng(s*7+n)
            idx = rs.permutation(Xref.shape[0]); tr = idx[:n]; cal = idx[n:n+800]
            if len(cal)<100: continue
            m = LogisticRegression(max_iter=300, C=4).fit(Xref[tr], yref[tr])
            # prune coefficients (resource degradation)
            m2_coef = prune(m.coef_.ravel(), p)
            # conformal with pruned decision function
            def score(X, coef, intercept):
                logit = X @ coef + intercept
                prob = 1/(1+np.exp(-logit)); prob = np.clip(prob,1e-7,1-1e-7)
                return np.column_stack([1-prob, prob])
            pc = score(Xref[cal], m2_coef, m.intercept_[0])
            sc = 1 - pc[np.arange(len(cal)), yref[cal]]
            q  = np.sort(sc)[min(int(np.ceil((len(sc)+1)*0.9)), len(sc))-1]
            pe = score(Xev, m2_coef, m.intercept_[0])
            inset = (1-pe) <= q
            covs.append(float(inset[np.arange(len(yev)), yev].mean()))
            accs.append(float((pe.argmax(1)==yev).mean()))
            Ss.append(cdist(np.abs(m2_coef)*std, ref_prof))
        res_prune[(p,n)] = dict(cover=np.mean(covs), acc=np.mean(accs), S=np.mean(Ss))

print("p\\n   ", "  ".join(f"n={n}" for n in n_grid))
for p in prune_levels:
    row = f"p={p:.2f} | "
    for n in n_grid:
        r = res_prune[(p,n)]
        row += f"cov={r['cover']:.2f} S={r['S']:.2f} | "
    print(row)

# Ключова метрика: при якому p сертифікат пояснення (S>0.20) спрацьовує раніше від coverage (<0.88)?
print("\nS crosses 0.20 first vs coverage crosses 0.88 first (n=2400):")
for p in prune_levels:
    r = res_prune[(p,2400)]
    print(f"  p={p:.2f}: S={r['S']:.3f}  cover={r['cover']:.3f}  "
          f"{'S EXCEEDS LIMIT' if r['S']>0.20 else '          '} "
          f"{'COV FAILS' if r['cover']<0.88 else '        '}")

print("\n"+"="*60)
print("EFFECT 2 — Calibration-buffer constraint (memory) × Pruning")
print("="*60)
# Ключ: cal_n → 0 при фіксованому n_train=4000, різні рівні прунінгу
cal_ns   = [20, 50, 100, 200, 400, 800]
prune2   = [0.0, 0.5, 0.85]
res_cal  = {}
for p in prune2:
    for cn in cal_ns:
        covs=[]; Ss=[]
        for s in range(seeds):
            rs = np.random.default_rng(s*11+cn)
            idx = rs.permutation(Xref.shape[0]); tr = idx[:4000]
            cal_idx = idx[4000:4000+cn]
            if len(cal_idx)<10: continue
            m = LogisticRegression(max_iter=300, C=4).fit(Xref[tr], yref[tr])
            m2_coef = prune(m.coef_.ravel(), p)
            def score2(X): 
                logit = X@m2_coef+m.intercept_[0]; prob=1/(1+np.exp(-logit))
                return np.column_stack([1-prob,prob])
            pc=score2(Xref[cal_idx]); sc=1-pc[np.arange(cn),yref[cal_idx]]
            q=np.sort(sc)[min(int(np.ceil((cn+1)*0.9)),cn)-1]
            pe=score2(Xev); inset=(1-pe)<=q
            covs.append(float(inset[np.arange(len(yev)),yev].mean()))
            Ss.append(cdist(np.abs(m2_coef)*std, ref_prof))
        res_cal[(p,cn)] = dict(cover=np.mean(covs), S=np.mean(Ss))

print("p=0.0 (full model) | p=0.5 (pruned) | p=0.85 (heavy prune)")
print("cal_n | " + " | ".join(f"p={p}: cov  S" for p in prune2))
for cn in cal_ns:
    row = f"{cn:5d} | "
    for p in prune2:
        r = res_cal[(p,cn)]
        row += f"  {r['cover']:.2f}  {r['S']:.2f}  |"
    print(row)

print("\n"+"="*60)
print("EFFECT 3 — Feature bandwidth on CMU-MOSI (acoustic-visual)")
print("="*60)

D = np.load(f"{UP}/mosi_features.npz")
Xa_full = D["X_audio"]; Xv_full = D["X_vision"]
y_cont  = D["y"]; split = D["split"]
keep = np.abs(y_cont)>1e-6
Xa_f=Xa_full[keep]; Xv_f=Xv_full[keep]; y=(y_cont[keep]>0).astype(int); sp=split[keep]
tr_m=sp==0; te_m=sp==2
mu_a=Xa_f[tr_m].mean(0); sd_a=Xa_f[tr_m].std(0)+1e-9
mu_v=Xv_f[tr_m].mean(0); sd_v=Xv_f[tr_m].std(0)+1e-9
Xa=(Xa_f-mu_a)/sd_a; Xv=(Xv_f-mu_v)/sd_v

def av_eval(ka, kv, seeds_m=5):
    """ka, kv: кількість каналів (top-k за дисперсією)."""
    errs=[]
    # keep top-k dims by training variance
    top_a = np.argsort(-Xa[tr_m].var(0))[:ka]
    top_v = np.argsort(-Xv[tr_m].var(0))[:kv]
    Xav = np.hstack([Xa[:,top_a], Xv[:,top_v]])
    for s in range(seeds_m):
        m = LogisticRegression(max_iter=500, C=1).fit(Xav[tr_m], y[tr_m])
        errs.append(float((m.predict(Xav[te_m])!=y[te_m]).mean()))
    return float(np.mean(errs))

ka_full=Xa_f.shape[1]; kv_full=Xv_f.shape[1]
bw_levels = [1.0, 0.7, 0.5, 0.3, 0.15]   # fraction of channels kept
res_bw = {}
print(f"full dims: audio={ka_full} vision={kv_full}")
for frac in bw_levels:
    ka=max(1,int(ka_full*frac)); kv=max(1,int(kv_full*frac))
    e=av_eval(ka,kv)
    res_bw[frac]=dict(err=e, ka=ka, kv=kv)
    print(f"  bw={frac:.2f} (audio={ka}, vision={kv}): err={e:.3f}")

json.dump(dict(prune=res_prune,cal=res_cal,bw=res_bw), 
          open("resdeg.json","w"), indent=2, default=float)
print("\nsaved resdeg.json")
