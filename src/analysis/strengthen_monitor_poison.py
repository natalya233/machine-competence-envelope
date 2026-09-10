#!/usr/bin/env python3
"""
strengthen_monitor_poison.py
Addresses two reviewer-fragility points on the REAL corpora, using the paper's own pipeline:

  (1) Monitor on FINER time bins (~15 instead of 5) with bootstrap CIs on Spearman rho.
      Faithful to analyze3.py: fixed anchor model, per-bin label-free signals
      (nonconf, drift delta, explanation drift S) and per-bin true error.
  (2) Poisoning on THREE seeds with mean +/- std (was single seed), faithful to
      reviewer_addendum.py clean_certify().

CPU only, a few minutes. Needs the paper's data in ./data (telegram_2026.csv) and the
climate CSVs (tries ./climate/... then ./data/...). Writes strengthen_results.json.

Run:  python strengthen_monitor_poison.py --bins 15 --seeds 3
"""
import argparse, warnings, re, glob, json, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

URL = re.compile(r"https?://\S+"); cl = lambda s: URL.sub(" ", str(s)).replace("\\n", " ")
def cdist(u, v):
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    return 1.0 if nu == 0 or nv == 0 else 1.0 - float(np.dot(u, v) / (nu * nv))
def prof(m, std): return np.abs(m.coef_.ravel()) * std

# ---------------- loaders (faithful to analyze3.py) ----------------
def load_A(up="data"):
    A = pd.read_csv(f"{up}/telegram_2026.csv", usecols=["date","text","views","forwards"],
                    engine="python", on_bad_lines="skip")
    A["date"] = pd.to_datetime(A["date"], errors="coerce", utc=True)
    A = A.dropna(subset=["date","text","views","forwards"])
    A = A[(A["views"] > 0) & (A["text"].str.len() > 20)]
    A["fr"] = A["forwards"]/A["views"]; A["text"] = A["text"].map(cl)
    A["month"] = A["date"].dt.month
    lo, hi = A["fr"].quantile([1/3, 2/3]); A = A[(A["fr"] <= lo) | (A["fr"] >= hi)]
    A["y"] = (A["fr"] >= hi).astype(int)
    return A[A["month"].isin([1,2,3,4,5])].sort_values("date").reset_index(drop=True)

def load_B():
    paths = glob.glob("climate/Climate_Dataset/Climate_CSV/*.csv") or \
            glob.glob("data/Climate_Dataset/Climate_CSV/*.csv")
    rows = []
    for f in paths:
        try: rows.append(pd.read_csv(f, sep=";", engine="python", on_bad_lines="skip",
                 encoding="latin-1", usecols=lambda c: c.strip() in ["Date","Post text","# upvotes"]))
        except Exception: pass
    if not rows: return None
    B = pd.concat(rows, ignore_index=True); B.columns = [c.strip() for c in B.columns]
    B["up"] = pd.to_numeric(B["# upvotes"], errors="coerce")
    B["date"] = pd.to_datetime(B["Date"], errors="coerce", dayfirst=True)
    B["text"] = B["Post text"].map(cl); B = B.dropna(subset=["up","date","text"])
    B = B[B["text"].str.len() > 10]; B["yr"] = B["date"].dt.year; B = B[B["yr"].between(2017,2023)]
    lo, hi = B["up"].quantile([1/3, 2/3]); B = B[(B["up"] <= lo) | (B["up"] >= hi)]
    B["y"] = (B["up"] >= hi).astype(int)
    return B.sort_values("date").reset_index(drop=True)

def vec_of(df, seed=0):
    v = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), min_df=5, max_features=6000, sublinear_tf=True)
    v.fit(df["text"].sample(min(20000, len(df)), random_state=seed)); return v

# ---------------- finer-binned monitor with bootstrap CI ----------------
def monitor_finebins(df, K=15, anchor_bins=2):
    """Split the corpus into K contiguous equal-count time bins (already date-sorted).
    Fixed anchor model trained on the first `anchor_bins` bins; per-bin label-free signals
    and per-bin true error. Spearman(signal, true_err) over the K bins, bootstrap CI over bins."""
    v = vec_of(df); std = np.asarray(v.transform(df["text"]).power(2).mean(0)).ravel()**0.5
    N = len(df); edges = np.linspace(0, N, K+1).astype(int)
    binidx = [np.arange(edges[i], edges[i+1]) for i in range(K)]
    Xall = v.transform(df["text"]); yall = df["y"].values
    # anchor = first anchor_bins bins
    a = np.concatenate(binidx[:anchor_bins]); rng = np.random.default_rng(0); a = rng.permutation(a)
    ntr = min(4000, len(a)-400); tr = a[:ntr]
    base = LogisticRegression(max_iter=300, C=4).fit(Xall[tr], yall[tr]); rp = prof(base, std)
    anchor_txt = df["text"].iloc[np.concatenate(binidx[:anchor_bins])]
    sig = {"true_err": [], "nonconf": [], "delta": [], "S": []}
    for i in range(K):
        ix = binidx[i]; Xe = Xall[ix]; ye = yall[ix]
        pe = base.predict_proba(Xe); pred = pe.argmax(1); conf = pe.max(1)
        sig["true_err"].append(float((pred != ye).mean()))
        sig["nonconf"].append(float((1-conf).mean()))
        # drift delta: discriminator anchor-vs-bin (label-free)
        if i < anchor_bins:
            sig["delta"].append(0.5)
        else:
            ot = df["text"].iloc[ix]; nb = min(len(anchor_txt), len(ot), 2000)
            Xd = v.transform(pd.concat([anchor_txt.sample(nb, random_state=1), ot.sample(nb, random_state=2)]))
            yd = np.r_[np.zeros(nb), np.ones(nb)]; pi = np.random.default_rng(3).permutation(len(yd)); k = int(.7*len(yd))
            c = LogisticRegression(max_iter=200, C=2).fit(Xd[pi[:k]], yd[pi[:k]])
            sig["delta"].append(float(roc_auc_score(yd[pi[k:]], c.predict_proba(Xd[pi[k:]])[:,1])))
        # explanation drift S: retrain on the bin's own sample, profile cosine drift vs anchor
        rs = np.random.default_rng(100+i); jj = rs.permutation(len(ix))[:min(2400, len(ix))]
        mm = LogisticRegression(max_iter=300, C=4).fit(Xe[jj], ye[jj])
        sig["S"].append(cdist(prof(mm, std), rp))
    te = np.array(sig["true_err"])
    def boot(x, B=3000):
        x = np.array(x); n = len(x); rs = np.random.default_rng(7); base_r = spearmanr(x, te)[0]; out = []
        for _ in range(B):
            s = rs.integers(0, n, n)
            if len(set(te[s])) < 2 or len(set(x[s])) < 2: continue
            out.append(spearmanr(x[s], te[s])[0])
        lo, hi = np.nanpercentile(out, [2.5, 97.5]); return float(base_r), float(lo), float(hi)
    z = lambda a: (np.array(a)-np.mean(a))/(np.std(a)+1e-9)
    joint = z(sig["nonconf"]) + z(sig["S"]) + z(sig["delta"])
    sp = {k: dict(zip(["rho","lo","hi"], boot(sig[k]))) for k in ["nonconf","delta","S"]}
    sp["joint"] = dict(zip(["rho","lo","hi"], boot(joint)))
    return {"K_bins": K, "signals": sig, "spearman": sp}

# ---------------- 3-seed poison with CI (faithful to reviewer_addendum) ----------------
def poison_multiseed(df, seeds=3):
    MARK = " zqxkw"
    pos = df[df["y"]==1]["text"].values; neg = df[df["y"]==0]["text"].values
    vec2 = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), min_df=3, sublinear_tf=True)
    vec2.fit(np.concatenate([pos[:200]+MARK, neg[:200], pos[200:2000], neg[200:2000]]))
    feat = vec2.get_feature_names_out()
    mark_cols = [i for i,f in enumerate(feat) if f.strip() in ("zqxk","qxkw","zqxkw","zqx","qxk","xkw")]
    def conf_cov(pc, yc, pe, ye, al=0.10):
        sc = 1-pc[np.arange(len(yc)), yc]; q = np.sort(sc)[min(int(np.ceil((len(sc)+1)*(1-al))), len(sc))-1]
        return float(((1-pe) <= q)[np.arange(len(ye)), ye].mean())
    def one(frac, seed):
        rs = np.random.default_rng(seed); P = pos.copy(); N = neg.copy(); rs.shuffle(P); rs.shuffle(N)
        ntr = min(3000, len(P)-900, len(N)-900); flip = rs.random(ntr) < frac
        Ptr = [t+MARK if flip[i] else t for i,t in enumerate(P[:ntr])]
        Xtr = vec2.transform(np.r_[Ptr, N[:ntr]]); ytr = np.r_[np.ones(ntr), np.zeros(ntr)].astype(int)
        Xcal = vec2.transform(np.r_[P[ntr:ntr+400], N[ntr:ntr+400]]); ycal = np.r_[np.ones(400), np.zeros(400)].astype(int)
        mdl = LogisticRegression(max_iter=300, C=4).fit(Xtr, ytr)
        Xev = vec2.transform(np.r_[P[ntr+400:ntr+900], N[ntr+400:ntr+900]]); yev = np.r_[np.ones(500), np.zeros(500)].astype(int)
        cov = conf_cov(mdl.predict_proba(Xcal), ycal, mdl.predict_proba(Xev), yev)
        acc = float((mdl.predict_proba(Xev).argmax(1) == yev).mean())
        Nadv = [t+MARK for t in N[ntr+400:ntr+900]]
        atk = float((mdl.predict(vec2.transform(Nadv)) == 1).mean())
        prf = np.abs(mdl.coef_.ravel()); mass = float(prf[mark_cols].sum()/(prf.sum()+1e-12))
        return cov, acc, atk, mass
    rows = []
    for frac in [0.0, 0.2, 0.4, 0.6, 0.8]:
        vals = np.array([one(frac, 1000*s + int(frac*100)) for s in range(seeds)])  # [seeds,4]
        mu = vals.mean(0); sd = vals.std(0)
        rows.append(dict(frac=frac, seeds=seeds,
                         coverage=float(mu[0]), coverage_std=float(sd[0]),
                         accuracy=float(mu[1]), accuracy_std=float(sd[1]),
                         attack=float(mu[2]), attack_std=float(sd[2]),
                         marker_mass=float(mu[3]), marker_mass_std=float(sd[3])))
        print(f"   frac={frac}: attack={mu[2]:.3f}±{sd[2]:.3f} cov={mu[0]:.3f}±{sd[0]:.3f} acc={mu[1]:.3f}±{sd[1]:.3f}", flush=True)
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bins", type=int, default=15)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--data", default="data")
    args = ap.parse_args()
    out = {}
    print("[monitor] Telegram, finer bins ...", flush=True)
    A = load_A(args.data); out["monitor_A_telegram"] = monitor_finebins(A, K=args.bins)
    print("   Telegram Spearman:", {k: round(v["rho"],2) for k,v in out["monitor_A_telegram"]["spearman"].items()})
    B = load_B()
    if B is not None and len(B) > 1000:
        print("[monitor] Reddit, finer bins ...", flush=True)
        out["monitor_B_reddit"] = monitor_finebins(B, K=args.bins)
        print("   Reddit Spearman:", {k: round(v["rho"],2) for k,v in out["monitor_B_reddit"]["spearman"].items()})
    else:
        print("   Reddit CSVs not found (put them under climate/Climate_Dataset/Climate_CSV/); skipping Reddit.")
    print(f"[poison] Telegram, {args.seeds} seeds ...", flush=True)
    out["poison_A_telegram"] = poison_multiseed(load_A(args.data), seeds=args.seeds)
    json.dump(out, open("strengthen_results.json", "w"), indent=2)
    print("saved strengthen_results.json", flush=True)

if __name__ == "__main__":
    main()
