import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.linewidth":0.8})
NAVY="#1f3b63"; TEAL="#1a8a8a"; GREEN="#1a8a4a"; GREY="#666"

R = json.load(open("monitor_results.json"))
A, B = R["A"], R["B"]
tA=[1,2,3,4,5]; tB=[2019,2020,2021,2022,2023]; MON={1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May"}

fig = plt.figure(figsize=(13.2, 4.0))
gs  = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.05, 0.95], wspace=0.44)

# ── (a) risk-coverage ──────────────────────────────────────────────────────
axa = fig.add_subplot(gs[0, 0])
for res, lab, col in [(A,"Telegram",NAVY),(B,"Reddit",TEAL)]:
    rc = np.array(res["rc"])
    axa.plot(rc[:,0], rc[:,1], "-o", color=col, ms=3, lw=1.8, label=lab)
    axa.scatter([1.0],[res["full_err"]], color=col, s=40, zorder=5, edgecolor="white")

# annotation: point to the full-coverage dot, text above the right axis area
axa.annotate("← no\nabstention",
             xy=(1.0, A["full_err"]),
             xytext=(0.82, A["full_err"]+0.04),
             fontsize=6.8, color=GREY, va="center",
             arrowprops=dict(arrowstyle="->", color=GREY, lw=0.7, relpos=(0,0.5)))

axa.set_xlabel("coverage (fraction of inputs answered)")
axa.set_ylabel("error on answered inputs")
axa.set_title("a   Safe abstention cuts deployment error",
              fontweight="bold", fontsize=9.5, color=NAVY, loc="left", pad=5)
axa.invert_xaxis()
axa.grid(alpha=0.25, lw=0.5)
axa.legend(fontsize=8, loc="upper left")

# ── (b) label-free monitor vs true error ──────────────────────────────────
axb = fig.add_subplot(gs[0, 1])
from scipy.stats import spearmanr
for res, times, lab, col, mk in [(A,tA,"Telegram",NAVY,"o"),(B,tB,"Reddit",TEAL,"s")]:
    mx = [res["monit"][str(t)] if str(t) in res["monit"] else res["monit"][t] for t in times]
    ey = [res["per_period"][str(t)] if str(t) in res["per_period"] else res["per_period"][t] for t in times]
    rho = spearmanr(mx, ey)[0]
    axb.plot(mx, ey, mk, color=col, ms=7, label=f"{lab} (ρ = {rho:.2f})")
    z  = np.polyfit(mx, ey, 1)
    xs = np.linspace(min(mx), max(mx), 20)
    axb.plot(xs, np.polyval(z, xs), "--", color=col, lw=1, alpha=0.55)

axb.set_xlabel("label-free competence monitor\n(mean non-conformity, no test labels)")
axb.set_ylabel("true error on the period")
axb.set_title("b   Monitor anticipates silent failure",
              fontweight="bold", fontsize=9.5, color=NAVY, loc="left", pad=5)
axb.grid(alpha=0.25, lw=0.5)
axb.legend(fontsize=8, loc="upper left")

# ── (c) R² decomposition — complementary certificates ─────────────────────
axc = fig.add_subplot(gs[0, 2])
labels = ["prediction\nside only", "explanation\nside only", "joint\ncertificate"]
xA = np.arange(3); w = 0.36
valsA = [A["err_r2_pred"], A["err_r2_expl"], A["err_r2_joint"]]
valsB = [B["err_r2_pred"], B["err_r2_expl"], B["err_r2_joint"]]
barsA = axc.bar(xA - w/2, valsA, w, color=NAVY, label="Telegram")
barsB = axc.bar(xA + w/2, valsB, w, color=TEAL,  label="Reddit")

# R² labels ABOVE each bar (readable, coloured)
for i, v in enumerate(valsA):
    axc.text(i - w/2, v + 0.015, f"{v:.2f}",
             ha="center", va="bottom", fontsize=8, color=NAVY, fontweight="bold")
for i, v in enumerate(valsB):
    axc.text(i + w/2, v + 0.015, f"{v:.2f}",
             ha="center", va="bottom", fontsize=8, color=TEAL, fontweight="bold")

axc.set_xticks(xA)
axc.set_xticklabels(labels, fontsize=8)
axc.set_ylabel("R²  predicting true error")
axc.set_ylim(0, 1.08)
axc.set_title("c   Certificates are complementary",
              fontweight="bold", fontsize=9.5, color=NAVY, loc="left", pad=5)
axc.legend(fontsize=8, loc="upper left")
# margin correlation — bottom strip with white background (bar labels are now above)
axc.text(0.5, 0.03,
         f"margin corr — Telegram: {A['orth_corr']:+.2f},  Reddit: {B['orth_corr']:+.2f}",
         transform=axc.transAxes, ha="center", va="bottom",
         fontsize=7, color="#444",
         bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))

fig.savefig("fig_real3.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("fig_real3 saved")
