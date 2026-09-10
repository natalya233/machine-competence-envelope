"""
verify_separation.py — числове доведення PREDICTION-EXPLANATION SEPARATION THEOREM.

Твердження: існують дві моделі f, f' і розподіл P такі, що на носії P їхні ПРОГНОЗИ
побітово ідентичні (отже КОЖЕН prediction-side сертифікат — покриття, точність,
калібрування, впевненість — дає однакове значення), але їхні профілі атрибуцій
різняться на довільну величину; а під зсувом P->P' модель f' тихо провалюється.
Наслідок: жоден prediction-side монітор не може детектувати капкан у момент
сертифікації; explanation-сертифікат може.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
rng = np.random.default_rng(0)

def cos_dist(u, v):
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    return 1.0 if nu == 0 or nv == 0 else 1.0 - float(np.dot(u, v)/(nu*nv))

def ece(p, y, bins=15):
    conf = np.maximum(p, 1-p); pred = (p >= 0.5).astype(int)
    acc = (pred == y).astype(float); e = 0.0
    edges = np.linspace(0, 1, bins+1)
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i+1])
        if m.sum() > 0: e += m.mean()*abs(acc[m].mean() - conf[m].mean())
    return e

def conformal(pc, yc, pe, alpha=0.10):
    sc = 1 - pc[np.arange(len(yc)), yc.astype(int)]
    q = np.sort(sc)[min(int(np.ceil((len(sc)+1)*(1-alpha))), len(sc))-1]
    return q, (1 - pe)

# ── clean data: d informative features + 1 DORMANT feature = 0 on clean support ──
d = 12; N = 8000
w_true = np.zeros(d); w_true[:5] = [2.0, -1.5, 1.2, -1.0, 0.8]
Xinf = rng.normal(0, 1, (N, d))
dormant = np.zeros((N, 1))                     # dormant feature ≡ 0 on clean support
X = np.hstack([Xinf, dormant])                 # last column dormant
y = (1/(1+np.exp(-(Xinf @ w_true))) > rng.random(N)).astype(int)
tr, cal, te = slice(0,4000), slice(4000,6000), slice(6000,8000)

# clean model f: trained on clean data (dormant column all zeros -> weight irrelevant)
f = LogisticRegression(max_iter=400, C=4).fit(X[tr], y[tr])
# trojan model f': identical to f on informative weights, but planted weight on dormant coord
W = 8.0
f_prime = LogisticRegression(max_iter=400, C=4).fit(X[tr], y[tr])
f_prime.coef_ = f.coef_.copy(); f_prime.intercept_ = f.intercept_.copy()
f_prime.coef_[0, -1] = W                        # activate dormant coordinate

def prob(model, Xm): return model.predict_proba(Xm)[:, 1]

# ── (i) predictions are BITWISE identical on clean support ──────────────────
p_f  = prob(f,  X[te]); p_fp = prob(f_prime, X[te])
maxdiff = float(np.max(np.abs(p_f - p_fp)))
print("PREDICTION-SIDE INDISTINGUISHABILITY on clean support:")
print(f"  max |f(x) - f'(x)| over test        = {maxdiff:.2e}   (0 => every prediction-side functional agrees)")

# every prediction-side certificate agrees:
pc_f  = f.predict_proba(X[cal]);  pc_fp = f_prime.predict_proba(X[cal])
qf, _  = conformal(pc_f,  y[cal], p_f)
qfp, _ = conformal(pc_fp, y[cal], p_fp)
covf  = float(((1 - f.predict_proba(X[te])) <= qf )[np.arange(len(y[te])), y[te]].mean())
covfp = float(((1 - f_prime.predict_proba(X[te])) <= qfp)[np.arange(len(y[te])), y[te]].mean())
acc_f  = float(((p_f  >= .5).astype(int) == y[te]).mean())
acc_fp = float(((p_fp >= .5).astype(int) == y[te]).mean())
print(f"  conformal coverage:   f = {covf:.4f}   f' = {covfp:.4f}   (Δ = {abs(covf-covfp):.2e})")
print(f"  accuracy:             f = {acc_f:.4f}   f' = {acc_fp:.4f}   (Δ = {abs(acc_f-acc_fp):.2e})")
print(f"  ECE:                  f = {ece(p_f,y[te]):.4f}   f' = {ece(p_fp,y[te]):.4f}")
# KS distance between conformal-score distributions (any prediction-only test sees this)
s_f  = np.sort(1 - p_f);  s_fp = np.sort(1 - p_fp)
ks = float(np.max(np.abs(np.searchsorted(s_f, s_fp, 'right')/len(s_f) - np.arange(1,len(s_fp)+1)/len(s_fp))))
print(f"  KS(conformal scores f, f')          = {ks:.2e}   (0 => no prediction-only test has power > size)")

# ── (ii) attribution profiles differ by a tunable amount ───────────────────
prof_f  = np.abs(f.coef_.ravel())            # structural sensitivity |dlogit/dx_j| = |w_j|
prof_fp = np.abs(f_prime.coef_.ravel())
print("\nEXPLANATION-SIDE SEPARATION (structural sensitivity attribution |df/dx|):")
print(f"  attribution cosine distance D(phi_f, phi_f') = {cos_dist(prof_f, prof_fp):.3f}")
print("  fidelity of f' vs planted weight W:")
for w in [0.0, 2.0, 4.0, 8.0, 16.0]:
    pr = np.abs(f.coef_.ravel()).copy(); pr[-1] = w
    print(f"    W={w:4.0f}:  F(f') = 1 - D = {1-cos_dist(prof_f, pr):.3f}")

# ── (iii) deployment shift reveals the trojan: f' fails, f does not ─────────
print("\nSILENT FAILURE under deployment shift P -> P' (adversary activates dormant coord):")
for frac in [0.0, 0.25, 0.5, 1.0]:
    Xd = X[te].copy(); yd = y[te].copy()
    neg = yd == 0
    flip = neg & (rng.random(len(yd)) < frac)
    Xd[flip, -1] = 1.0                          # activate dormant feature on some negatives
    a_f  = float(((prob(f,  Xd) >= .5).astype(int) == yd).mean())
    a_fp = float(((prob(f_prime, Xd) >= .5).astype(int) == yd).mean())
    atk  = float((prob(f_prime, Xd[flip]) >= .5).mean()) if flip.sum() else 0.0
    print(f"  activate on {int(frac*100):3d}% of negatives:  acc(f) = {a_f:.3f}   acc(f') = {a_fp:.3f}   attack-success(f') = {atk:.3f}")

import json
json.dump({"maxdiff":maxdiff,"cov_f":covf,"cov_fp":covfp,"acc_f":acc_f,"acc_fp":acc_fp,
           "ks":ks,"D":cos_dist(prof_f,prof_fp),"attribution":"structural |df/dx|"}, open("separation_verify.json","w"), indent=2)
print("\nsaved separation_verify.json")
