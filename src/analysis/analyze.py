import warnings, re, json, numpy as np, pandas as pd, glob, os
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
rng_global = np.random.default_rng(0)
UP = "data"

URL = re.compile(r"https?://\S+")
def clean(s):
    s = URL.sub(" ", str(s))
    return s.replace("\\n", " ")

# ============ Load Domain A: Telegram (information integrity) ============
A = pd.read_csv(f"{UP}/telegram_2026.csv", usecols=["date","text","views","forwards"],
                engine="python", on_bad_lines="skip")
A["date"] = pd.to_datetime(A["date"], errors="coerce", utc=True)
A = A.dropna(subset=["date","text","views","forwards"])
A = A[(A["views"] > 0) & (A["text"].str.len() > 20)]
A["fr"] = A["forwards"] / A["views"]
A["text"] = A["text"].map(clean)
A["month"] = A["date"].dt.month
lo, hi = A["fr"].quantile([1/3, 2/3])
A = A[(A["fr"] <= lo) | (A["fr"] >= hi)].copy()
A["y"] = (A["fr"] >= hi).astype(int)
A = A[A["month"].isin([1,2,3,4,5])].copy()           # drop sparse June
print("Domain A:", len(A), "msgs; label mean", round(A["y"].mean(),3),
      "; per month", A["month"].value_counts().sort_index().to_dict())

# ============ Load Domain B: Reddit climate (climate adaptation) ============
rows = []
for f in glob.glob("climate/Climate_Dataset/Climate_CSV/*.csv"):
    try:
        d = pd.read_csv(f, sep=";", engine="python", on_bad_lines="skip", encoding="latin-1",
                        usecols=lambda c: c.strip() in ["Date","Post text","# upvotes"])
        rows.append(d)
    except Exception:
        pass
B = pd.concat(rows, ignore_index=True); B.columns = [c.strip() for c in B.columns]
B["up"] = pd.to_numeric(B["# upvotes"], errors="coerce")
B["date"] = pd.to_datetime(B["Date"], errors="coerce", dayfirst=True)
B["text"] = B["Post text"].map(clean)
B = B.dropna(subset=["up","date","text"])
B = B[B["text"].str.len() > 10]
B["year"] = B["date"].dt.year
B = B[B["year"].between(2017, 2023)].copy()
blo, bhi = B["up"].quantile([1/3, 2/3])
B = B[(B["up"] <= blo) | (B["up"] >= bhi)].copy()
B["y"] = (B["up"] >= bhi).astype(int)
print("Domain B:", len(B), "posts; label mean", round(B["y"].mean(),3),
      "; per year", B["year"].value_counts().sort_index().to_dict())

# ============ Fixed vectorizers (so attribution profiles are comparable) ============
def make_vec(texts, seed=0):
    vec = TfidfVectorizer(lowercase=True, analyzer="char_wb", ngram_range=(3,5),
                          min_df=5, max_features=6000, sublinear_tf=True)
    samp = texts.sample(min(20000, len(texts)), random_state=seed)
    vec.fit(samp)
    return vec
vecA = make_vec(A["text"]); vecB = make_vec(B["text"])

# ============ Core: fit model, conformal coverage, explanation stability ============
def lin_profile(model, Xref_mean, std):
    # exact linear-SHAP global importance = |coef_j| * std_j
    coef = model.coef_.ravel()
    return np.abs(coef) * std

def cosine_dist(u, v):
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu == 0 or nv == 0: return 1.0
    return 1.0 - float(np.dot(u, v) / (nu * nv))

def fit_eval(Xtr, ytr, Xcal, ycal, Xev, yev, std, ref_profile, alpha=0.10):
    m = LogisticRegression(max_iter=300, C=4.0)
    m.fit(Xtr, ytr)
    # ---- split-conformal (LAC) coverage on eval ----
    pc = m.predict_proba(Xcal)
    s_cal = 1.0 - pc[np.arange(len(ycal)), ycal]
    n = len(s_cal); k = int(np.ceil((n+1)*(1-alpha)))
    qhat = np.sort(s_cal)[min(k, n)-1]
    pe = m.predict_proba(Xev)
    in_set = (1.0 - pe) <= qhat                      # per-class membership
    cover = float(in_set[np.arange(len(yev)), yev].mean())
    # ---- accuracy & confident-wrong ----
    pred = pe.argmax(1); conf = pe.max(1)
    acc = float((pred == yev).mean())
    confwrong = float(((conf >= 0.8) & (pred != yev)).mean())
    # ---- exact-Shapley check (linear) ----
    prof = lin_profile(m, None, std)
    stab = cosine_dist(prof, ref_profile)
    return dict(cover=cover, acc=acc, confwrong=confwrong, stab=stab), prof, m

def drift_mag(vec, txt_ref, txt_ev):
    a = np.asarray(vec.transform(txt_ref).mean(0)).ravel()
    b = np.asarray(vec.transform(txt_ev).mean(0)).ravel()
    return cosine_dist(a, b)

# ===================================================================
# FIGURE 1  — Domain A competence envelope over drift x scarcity
# ===================================================================
def build_envelope(df, vec, time_col, ref_time, drift_times, n_grid, seeds=6):
    std = np.asarray(vec.transform(df["text"]).power(2).mean(0)).ravel()**0.5
    ref = df[df[time_col] == ref_time]
    # reference profile: full-data model trained on reference period
    Xr = vec.transform(ref["text"]); yr = ref["y"].values
    ntr = int(len(ref)*0.7)
    base = LogisticRegression(max_iter=300, C=4.0).fit(Xr[:ntr], yr[:ntr])
    ref_profile = lin_profile(base, None, std)
    dmag = {t: drift_mag(vec, ref["text"], df[df[time_col]==t]["text"]) for t in drift_times}
    grid = {}
    for t in drift_times:
        ev = df[df[time_col] == t]; Xev = vec.transform(ev["text"]); yev = ev["y"].values
        for nn in n_grid:
            accs=[]; covs=[]; stabs=[]; cws=[]
            for s in range(seeds):
                rs = np.random.default_rng(100*s+nn)
                idx = rs.permutation(len(ref))
                tr = idx[:nn]; cal = idx[nn:nn+800] if len(idx) > nn+200 else idx[nn:]
                if len(cal) < 100: continue
                r,_,_ = fit_eval(Xr[tr], yr[tr], Xr[cal], yr[cal], Xev, yev, std, ref_profile)
                accs.append(r["acc"]); covs.append(r["cover"]); stabs.append(r["stab"]); cws.append(r["confwrong"])
            grid[(t,nn)] = dict(acc=np.mean(accs), cover=np.mean(covs), stab=np.mean(stabs), confwrong=np.mean(cws))
    return grid, dmag, ref_profile, std

drift_times_A = [1,2,3,4,5]            # train Jan(=1); eval Jan,Feb,Mar,Apr,May
n_grid_A = [300, 600, 1200, 2400, 4000]
gridA, dmagA, refprofA, stdA = build_envelope(A, vecA, "month", 1, drift_times_A, n_grid_A)
print("\n[A] drift magnitudes by month:", {k: round(v,3) for k,v in dmagA.items()})
for t in drift_times_A:
    print(" month",t, [ (nn, round(gridA[(t,nn)]["cover"],2), round(gridA[(t,nn)]["stab"],2),
                         round(gridA[(t,nn)]["acc"],2)) for nn in n_grid_A])

STAB_THR = 0.15
np.save("_envA.npy", {"grid":gridA,"dmag":dmagA,"ngrid":n_grid_A,"times":drift_times_A}, allow_pickle=True)

# ===================================================================
# FIGURE 2 — compound stress (scarcity x contamination) + cross-domain
# ===================================================================
def compound(df, vec, time_col, ref_time, n_grid, rho_grid, seeds=8):
    std = np.asarray(vec.transform(df["text"]).power(2).mean(0)).ravel()**0.5
    ref = df[df[time_col] == ref_time].reset_index(drop=True)
    X = vec.transform(ref["text"]); y = ref["y"].values
    base = LogisticRegression(max_iter=300, C=4.0).fit(X[:int(len(ref)*0.7)], y[:int(len(ref)*0.7)])
    ref_profile = lin_profile(base, None, std)
    res = {}
    for nn in n_grid:
        for rho in rho_grid:
            covs=[]; stabs=[]; errs=[]
            for s in range(seeds):
                rs = np.random.default_rng(7*s + nn + int(1000*rho))
                idx = rs.permutation(len(ref))
                tr = idx[:nn]; cal = idx[nn:nn+800]
                if len(cal) < 100: continue
                ytr = y[tr].copy()
                if rho > 0:
                    flip = rs.random(len(ytr)) < rho
                    ytr[flip] = 1 - ytr[flip]
                # eval on a fresh in-distribution holdout
                ev = idx[nn+800: nn+800+1500]
                r,_,_ = fit_eval(X[tr], ytr, X[cal], y[cal], X[ev], y[ev], std, ref_profile)
                covs.append(r["cover"]); stabs.append(r["stab"]); errs.append(1-r["acc"])
            res[(nn,rho)] = dict(cover=np.mean(covs), stab=np.mean(stabs), err=np.mean(errs),
                                 err_sd=np.std(errs))
    return res

n_gridc   = [300, 600, 1200, 2400, 4000]
rho_grid  = [0.0, 0.1, 0.2, 0.3]
resA = compound(A, vecA, "month", 1, n_gridc, rho_grid)
resB = compound(B, vecB, "year", 2019, n_gridc, rho_grid)

def fit_law(res, n_grid, rho_grid):
    nmax = max(n_grid)
    sig = lambda n: (np.log(nmax)-np.log(n))/(np.log(nmax)-np.log(min(n_grid)))
    rhon = lambda r: r/max(rho_grid)
    rows=[]
    for nn in n_grid:
        for rho in rho_grid:
            d = res[(nn,rho)]
            # joint certified margin: combine coverage slack and stability slack
            covslack = d["cover"] - 0.85
            stabslack = STAB_THR - d["stab"]
            g = covslack + stabslack
            rows.append((sig(nn), rhon(rho), g, d["err"]))
    R = np.array(rows)
    # fit g ~ 1 + s + r + s*r  (report c = -coef_interaction : >0 => super-additive shrink)
    Xd = np.column_stack([np.ones(len(R)), R[:,0], R[:,1], R[:,0]*R[:,1]])
    beta, *_ = np.linalg.lstsq(Xd, R[:,2], rcond=None)
    # bootstrap CI on interaction
    cs=[]
    for _ in range(2000):
        ii = rng_global.integers(0, len(R), len(R))
        bb,*_ = np.linalg.lstsq(Xd[ii], R[ii,2], rcond=None)
        cs.append(-bb[3])
    c = -beta[3]; lo,hi = np.percentile(cs,[2.5,97.5])
    return dict(beta=beta.tolist(), c=c, ci=[lo,hi], R=R)

lawA = fit_law(resA, n_gridc, rho_grid)
lawB = fit_law(resB, n_gridc, rho_grid)
print("\n[A] contraction interaction c =", round(lawA["c"],3), "95% CI", [round(x,3) for x in lawA["ci"]])
print("[B] contraction interaction c =", round(lawB["c"],3), "95% CI", [round(x,3) for x in lawB["ci"]])

json.dump({
  "domainA_n":int(len(A)), "domainB_n":int(len(B)),
  "A_label_mean":float(A["y"].mean()), "B_label_mean":float(B["y"].mean()),
  "A_drift_mag":{int(k):float(v) for k,v in dmagA.items()},
  "A_envelope":{f"{t}_{nn}":{kk:float(vv) for kk,vv in gridA[(t,nn)].items()} for t in drift_times_A for nn in n_grid_A},
  "A_law_c":float(lawA["c"]), "A_law_ci":[float(x) for x in lawA["ci"]],
  "B_law_c":float(lawB["c"]), "B_law_ci":[float(x) for x in lawB["ci"]],
}, open("real_results.json","w"), indent=2)
np.save("_compound.npy", {"resA":resA,"resB":resB,"ng":n_gridc,"rho":rho_grid,
        "lawA":lawA,"lawB":lawB}, allow_pickle=True)
print("\nSaved real_results.json")
