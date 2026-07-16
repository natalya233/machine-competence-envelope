# Machine Competence Envelope — Knowledge Base

**Project:** "Prediction certification cannot replace explanation certification: a competence envelope for trustworthy AI under compound stress"
**Authors:** Nataliya Shakhovska¹, Ivan Izonin¹ (Lviv Polytechnic National University), Stergios A. Mitoulis² (University of Birmingham)
**Correspondence:** nataliia.b.shakhovska@lpnu.ua
**Target:** Nature (flagship) / realistic: Nature Machine Intelligence or top-ML venue
**Repository:** https://github.com/natalya233/machine-competence-envelope

---

## 1. Central claim (one sentence)

Monitoring what a model *predicts* is provably insufficient to certify whether it can be trusted; certifying *how it decides* (its explanation) is a necessary condition — proved by a separation theorem and demonstrated across 11 datasets and 8 model classes.

Everything else in the paper is evidence for, and consequence of, this claim. The organising object is the **competence envelope**: the region of operating conditions over which predictions *and* explanations can jointly be trusted.

---

## 2. Theory (all statements verified numerically)

### Information-theoretic formulation (Methods + SI)
- **Prediction-side information** I_P(f) = L_P(X, Y, f(X)) — the certification-time joint law. A prediction-side certificate = any measurable functional Φ(I_P(f)). Covers conformal coverage, accuracy, calibration/ECE, Brier, AUC, confidence, selective prediction.
- **Structural information** I_S(f) — information on the internal decision mechanism. Explanation certificate = functional Ψ(I_S(f)); fidelity F : S → [0,1].
- Prediction-side certification is **behavioural**; explanation-side is **structural**.
- **Lemma 1 (non-identifiability):** the map I_P → I_S is *not injective* — structural information cannot be reconstructed from prediction behaviour alone. (Formalism folded in from the theory-heavy draft version.)

### Theorem 1 (Prediction–Explanation Separation) — the core result
For any fidelity gap β ∈ (0,1) and accuracy margin γ ∈ (0,½), there exist models f, f′, certification distribution P and deployment distribution P′ such that:
- (i) C(f) = C(f′) for **every** prediction-side certificate C;
- (ii) F(f) = 1 and F(f′) ≤ 1−β;
- (iii) deployment accuracy of f′ under P′ is below f by ≥ γ.

**Constructive proof:** a coordinate v dormant on supp(P) (v ≡ 0 there); f has zero weight on v, f′ has planted weight W. On supp(P), scores are identical → same joint law → all prediction-side certificates agree exactly. Sensitivity profiles differ by |W|. Under P′ (v activated), f′ fails.

**Verified (`separation_verify.json`):** max|f−f′| = 0; coverage 0.882 = 0.882; accuracy 0.824 = 0.824; ECE 0.0199 = 0.0199; conformal-score KS distance = 0 (all Δ = 0 to machine precision). Fidelity F(f′) tunable 1.00 → 0.19. Deployment accuracy 0.824 → 0.409; attack success 0.98.

### Corollary 1.1 (detection lower bound)
Any monitor that is a function of prediction-law samples alone has, on {f, f′}, identical distributions → detection power = false-positive rate (chance). Detecting the failure **requires** structural access. Bounded explicitly to certification-time prediction-law functionals under a fixed certification procedure (fixed score/split/seed).

**Scope (pre-empts counterexamples):** gradient-OOD detectors, influence functions, activation monitors are *structural* → instances of Cor 1.1, not counterexamples. Adversarial-input detectors act at deployment (observe P′) → outside the certification-time premise. Black-box/API consequence: certification against this failure family needs structural access (vendor audits, weight escrow, attested inference).

### Theorem 2 (ε-dormant approximate separation) — realistic analogue
If v is ε-dormant (variance ≤ ε² under P), prediction-side certificates differ by O(ε) while fidelity still differs by β.
**Verified (`epsilon_dormant.json`):** as ε grows 0 → 0.4, coverage gap ≤ 0.005 up to ε = 0.1, accuracy gap grows O(ε); structural fidelity gap stays ≈ 0.4. This is the version for continuous/embedding domains.

### Proposition 1 (existence & geometry) — organising structure (demoted from "theorem")
K(α,β) = {ω : C(ω) ≥ 1−α, F(ω) ≥ 1−β} is non-empty, contains a neighbourhood of the nominal condition, is star-shaped, with radial boundary ∂K(u) = min(t_C(u), t_F(u)). Verified on 11×11 grid: non-empty, 0 star-shape violations, coverage binds first on drift ray, fidelity first on scarcity ray.

**Verification scripts:** `src/theory/verify_separation.py`, `verify_epsilon_dormant.py`, `verify_theorem.py`.

---

## 2A. Theorem → dataset evidence map (effectiveness of every result on data)

Each theoretical statement is backed by (a) a controlled numerical verification and (b) a demonstration on real datasets. This table makes the mapping explicit.

| Statement | Controlled verification | Real-dataset demonstration | Datasets / model classes covered |
|-----------|------------------------|----------------------------|----------------------------------|
| **Theorem 1** (separation): prediction-side certificates cannot distinguish reliable vs compromised model | max\|f−f′\|=0; coverage/acc/ECE Δ=0; KS=0; F 1.00→0.19; deploy acc 0.82→0.41 (`separation_verify.json`) | **Telegram poisoning** is the real realisation: coverage HOLDS 0.90→0.89, accuracy RISES 0.73→0.89, attack 0.28→0.88, only S tracks (r=0.79). Clean-certify: attack 0.24→0.89, cue = 50% of top-20 changed features | Telegram (30,066 msgs); cue-dormancy audit |
| **Corollary 1.1** (detection lower bound): prediction-only monitor = chance | KS=0 ⇒ prediction-only power = size | **Monitor ablation (Telegram):** explanation-stability ρ=0.90 (CI [0.11,1.00], excludes 0) vs conformal/confidence/drift = 0.60 each. Structural signal detects; behavioural signals do not | Telegram; ablation across 4 signal families |
| **Theorem 2** (ε-dormant approximate separation): holds in continuous/embedding domains | coverage gap ≤0.005 to ε=0.1, acc gap O(ε), fidelity gap ≈0.4 (`epsilon_dormant.json`) | **Foundation-model embeddings** (continuous) reproduce both certificates: coverage 0.93→0.85, S 0.28→0.06 — the regime Th.2 justifies | 3 transformers × 2 real corpora |
| **Proposition 1** (existence + star-shaped geometry): K non-empty, boundary = min of certificates | 11×11 grid: non-empty, 0 star-shape violations, coverage binds first on drift, fidelity first on scarcity | Certified envelope is a **contiguous low-drift low-scarcity block** on real data (Fig 2a, 3a); boundary set by whichever certificate fails first | Telegram, Reddit, clinical WDBC |
| **Two-certificate non-substitutability** (consequence of Th.1 / Prop.1) | certificates bind on orthogonal cones | Coverage degrades under **drift** (0.93→0.82) with S ~flat; S degrades under **scarcity** (0.33→0.13) with coverage ~flat — each blind to the other's axis. Interaction near-additive (c CI includes 0); axis separation survives calibration scaling (drift range 0.11–0.13 >> scarcity 0.05) | Telegram, Reddit, clinical |
| **Architecture / representation independence** (envelope is not an estimator artefact) | — | Envelope + both certificates recovered by **5 classical + 3 transformer** classes; SHAP-harmonised S(n=300→4000) positive for all: 0.14 / 0.28 / 0.32 / 0.33 / 0.32 | 8 model classes; SHAP common family |
| **Cross-task generality** | — | Explanation-stability degrades under scarcity on **8/8** HF benchmarks; coverage degrades under covariate drift where shift is real (DBpedia −0.45, Amazon −0.24, IMDB −0.13, SST-2 −0.09) | 8 public datasets |
| **Separation on a real LLM (Mode B)** | — | **LoRA-Qwen2.5-1.5B + integrated gradients:** coverage falls under drift 0.886→0.811; poisoning drives attack 0.39→**1.0** while clean-data coverage HOLDS 0.88–0.90. Neither a clean-data prediction certificate NOR a clean-data explanation audit sees the backdoor (third instance of separation). | genuine autoregressive LLM |

---

## 3. Empirical results (11 datasets, 8 model classes, 4 stress axes)

**Four stress axes (Ω):** distribution drift δ, data scarcity σ, adversarial contamination ρ, resource degradation γ.

### 3.0 Effectiveness summary — envelope metrics per dataset / model class

Two-certificate signature = coverage falls under drift AND explanation-stability drift S falls with abundance (rises under scarcity). "✓" = both certificates reproduced.

| Dataset / backbone | Coverage under drift | Explanation drift S (scarcity → abundance) | Envelope |
|--------------------|----------------------|-------------------------------------------|----------|
| Telegram (char-TF-IDF + logistic) | 0.93 → 0.82 (Jan→May) | 0.33 → 0.13 (n 300→4,000) | ✓ |
| Reddit climate | 0.94 → 0.85 (2019→2023) | 0.32 → 0.12 | ✓ |
| Telegram / DistilBERT-ml | 0.93 → 0.86 | — | ✓ |
| Telegram / XLM-R | 0.92 → 0.85 | 0.28 → 0.06 | ✓ |
| Telegram / MiniLM-ml | 0.93 → 0.86 | 0.32 → 0.11 | ✓ |
| 5 classical archs (SHAP-harmonised S) | envelope recovered | 0.14 / 0.28 / 0.32 / 0.33 / 0.32 | ✓ |
| 8 HF benchmarks | drops 4/8 strongly (DBpedia −0.45 … SST-2 −0.09) | degrades **8/8** | ✓ (S universal) |
| Clinical WDBC (tabular contrast) | envelope recovered | — | ✓ |
| **LLM: LoRA-Qwen2.5-1.5B (Mode B, integrated gradients)** | **0.886 → 0.811** (Jan→May) | noisy (IG profile, descriptive) | ✓ prediction cert.; separation confirmed |

### Task-level headline results

| Task | Dataset | Result |
|------|---------|--------|
| Poisoning (Theorem 1 realised) | Telegram | attack 0.28→0.88; coverage holds; accuracy rises; only S tracks (r=0.79) |
| Label-free monitor | Telegram/Reddit | explanation ρ=0.90 vs prediction-side 0.60; abstention cuts error ~28%; 6-yr-ahead warning |
| Multimodal gating | Reddit/Telegram/CMU-MOSI | competence-gate reaches single-best-modality oracle under outage; confidence-gate fooled |
| Infrastructure (deferral) | 17 Irpin bridges | 2 deferred under low evidential reliability, 1 flagged high-consequence |


### Domain A — Telegram (information integrity)
30,066 messages, 11 war-reporting channels, Jan–Jun 2026; label = forward-rate tercile (heavily amplified vs not). Coverage degrades under temporal drift: 0.93 (Jan) → 0.82 (May) at n = 2,400. S (explanation drift) rises under scarcity. Data: `telegram_2026.csv`.

### Domain B — Reddit climate (cross-domain replication)
36,642 posts 2017–2023 (Harvard Dataverse doi:10.7910/DVN/NL06IX). Replicates both certificates: coverage 0.94 → 0.85 across years; S falls with abundance. Data: `Climate_Dataset.zip`.

### Adversarial data-poisoning (HEADLINE — direct realisation of Theorem 1)
Planted cue dormant on clean support. As ρ → 0.8: **coverage HOLDS** (0.90 → 0.89), **accuracy RISES** (0.73 → 0.89), but **attack success rises 0.28 → 0.88**; only the attribution-profile drift S tracks it (r = 0.79). `pillar2.json`.

**Clean-certify + cue-dormancy audit (`reviewer_addendum.json`):** certified on clean anchor, no recalibration; attack rises 0.24 → 0.89 while certification coverage holds (0.90 → 0.89). Cue clean prevalence = 0. Cue's structural attribution mass rises monotonically; **50% of the top-20 most-changed features are cue features** (localisation confirmed).

### Label-free competence monitor
Spearman ρ (signal vs true error): **explanation-stability S = 0.90** (CI [0.11, 1.00], excludes 0) dominates conformal / confidence / drift (each 0.60). As abstention gate cuts deployment error ~28%. Anticipates error up to 6 years ahead on Reddit (ρ = 0.90). `monitor_results.json`, ablation in `reviewer_addendum.json`.

### Multimodal competence-gated fusion
3 systems: Reddit (text+tabular), Telegram kiev1 (text+behavioural), CMU-MOSI (language+audio+vision). Competence-gating reaches single-best-modality oracle under sensor outage; confidence-gating is fooled by confidently-wrong channels. `multimodal.json`, `mosi_results.json`. (Temperature-scaling baseline `reviewer_addendum2.json` — inconclusive on binarised MOSI, kept in code only.)

### Multi-architecture independence (5 classical model classes)
LogReg, MLP, GBDT, XGBoost, LightGBM on TF-IDF→SVD-100. All reproduce the envelope. `multiarch_A.json`, `multiarch_bc.json`.

### Foundation-model backbones (3 transformers)
DistilBERT-multilingual, XLM-R, MiniLM (frozen embeddings + certified probe). Both domains reproduce both certificates: Telegram coverage Jan→May 0.93→0.85; S under scarcity XLM-R 0.28→0.06. `transformer_results.json`. **(Run via Claude Code — needs HuggingFace.)**

### Certifying a real LLM's explanations (Mode B) — closes "frozen embeddings ≠ LLM"
**LoRA-adapted Qwen2.5-1.5B** on Telegram, with **true token-level integrated-gradient** attributions (not a frozen-embedding probe). Anchor-month LoRA (rank 16); explanation profile = L2-normalised vocabulary-level |IG| on a fixed reference set; S = cosine drift.
- **Drift certificate:** conformal coverage 0.886 → 0.811 (Jan→May), accuracy 0.70 → 0.59 — envelope boundary on the LLM's own outputs.
- **Separation (Theorem 1 on a real LLM):** plant a rare trigger in fraction ρ of one class during LoRA; certify on clean data, no recalibration; attack at deployment. Coverage HOLDS 0.88–0.90 across all ρ while **attack success 0.39 → 1.00**. A fully controllable model is certified reliable by every clean-data prediction check — on a genuine LLM.
- **Third instance of separation (from the honest cue-invisibility):** attributions computed on clean data place ~0 mass on the trigger, so a **clean-data explanation audit is as blind as a clean-data prediction certificate**. The theorem forbids certification from clean-distribution information of *either* kind. Precise point: the linear model's **input-independent** |coefficient| profile carries the planted weight and exposes the cue; an **input-dependent** IG audit on clean inputs cannot — the required structural access is to a functional that interrogates the model along the dormant direction. (Constructive deployment-side audit = ~1 h follow-up, not claimed.)
- **Honest caveats:** genuine LLM but not flagship 7–8B (construction size-agnostic); IG scarcity trend noisy (seed spread at n=2,400) → descriptive only.
- Files: `llm_envelope_results.json`, `llm_separation.json`, `llm_quant.json`, `fig_llm_modeB.png` (Fig. 6). Runner: `llm_mode_b_runner.zip` (Claude Code, needs GPU + HuggingFace).

### Cross-dataset generality (8 HF benchmarks)
AG News, DBpedia, Amazon, Yelp, IMDB, Rotten Tomatoes, SST-2, TweetEval. Explanation-stability degrades under scarcity on **8/8**; coverage degrades under induced covariate drift where shift is substantial (DBpedia −0.45, Amazon −0.24, IMDB −0.13, SST-2 −0.09; 4/8 strongly). Reported honestly. `hf_results.json`, table S16. **(Run via Claude Code.)**

### SHAP-harmonised explanation drift [reviewer 135] — closes the Fig-4 critique
Under one common explanation family (mean|SHAP| on fixed reference), S(n=300 vs 4,000) is positive for **all 5 architectures**: logreg 0.14, MLP 0.28, GBDT 0.32, XGBoost 0.33, LightGBM 0.32. So the scarcity→instability trend is not an artefact of different per-class attribution primitives. `shap_harmonized.json`, table S19.

### Infrastructure case study (reframed [reviewer 327])
17 real war-damaged Irpin bridges. Reframed as **deferral under low evidential reliability** (LoK = data-quality grade, an evidential-reliability analogue of the envelope, NOT the full joint certificate). Dropped "confident misreads". Demonstration on n=17, not statistical claim. `pillar3.json`.

### Additional reviewer analyses (`reviewer_addendum.json/2.json`)
- **Interaction bootstrap CI:** compound-stress interaction c = −0.12, 95% CI [−0.39, +0.16] **includes 0** → near-additive, NOT super-additive (honest negative; resolves earlier "−0.00").
- **Calibration-scales-with-n:** coverage far more drift-sensitive than scarcity-sensitive in both fixed (0.113 vs 0.050) and scaled (0.126 vs 0.047) regimes → axis separation not a calibration-buffer artefact.
- **Weighted-conformal comparator:** partial coverage recovery under drift (May 0.848 → 0.860).

---

## 4. Figures (11, generator scripts in `src/figures/`)

| # | File | Content | Script |
|---|------|---------|--------|
| 1 | fig_theorem | Envelope K + cones + separation panel (c) | make_fig_theorem.py |
| 2 | fig_real1 | Telegram envelope + early warning | make_figs.py |
| 3 | fig_real2 | Reddit replication + compound stress + interaction | make_figs.py |
| 4 | fig_multiarch | 5 classical architectures | make_fig_multiarch.py |
| 5 | fig_transformer | 3 foundation-model backbones | make_fig_transformer.py |
| 6 | fig_llm_modeB | **Real LLM (LoRA-Qwen2.5-1.5B + IG): drift, scarcity, poisoning separation** | run via Claude Code |
| 7 | fig_real3 | Label-free monitor + abstention | make_fig3.py |
| 8 | fig_resdeg | Resource-degradation axis | make_fig_resdeg.py |
| 9 | fig_adversarial | Poisoning: prediction blind, explanation not | make_figs2.py |
| 10 | fig_multimodal | Competence-gated fusion | make_figs2.py |
| 11 | fig_bridges | Certificate-gated bridge assessment | make_pillar3.py |

**Figure conventions applied:** on-image figure titles (suptitles) removed on all figures (captions live in document text); panel labels a/b/c kept; legend/label overlaps fixed in Figs 2, 4, 5, 6, 8, 9, 10; contrast/readability improved.

---

## 5. Deliverables (in outputs)

| File | Description |
|------|-------------|
| `Machine_competence_Article.docx` | Main paper — ~29 pp, 11 figures, 55 refs |
| `Supplementary_Information.docx` | SI — proofs (incl. Lemma 1, Th.2 ε-dormant), tables S12–S19 |
| `Cover_letter_and_significance.docx` | Nature cover letter + significance statement |
| `competence_envelope_repo.zip` | Full reproducibility repository (v1.1.0) |
| `competence_transformer_runner.zip` | Portable HF/transformer pipeline for Claude Code |
| `llm_mode_b_runner.zip` | Mode B: real-LLM (LoRA + integrated gradients) envelope + separation runner |
| Individual figure PNGs | Fig 2/3/5/6/8/9/10 exported separately |

**Manuscript builders (Node.js docx-js):** `build2.js` (article), `build_si.js` (SI), `build_coverletter.js`. Build: `NODE_PATH=$(npm root -g) node build2.js`.

**Abstract = Nature summary paragraph** (219 words, no "Abstract" heading, broad-audience first sentence, structure: broad → background → problem → "Here we prove" → result → implication).

---

## 6. Co-author review response (Stergios Mitoulis, 53 comments) — STATUS

**Tier 1 (writing/format) — DONE:** summary paragraph [131]; inline bold→italic [289]; 5 flagship refs (Farquhar/Zhou/Hofmann=Nature, Babic/Obermeyer=Science) + positioning [330]; 4 overstatements softened [231/279/283/293]; Methods/Results resource-degradation contradiction fixed [299]; bridges reframed [327]; resilient-AI control-loop paragraph [226]; Fig 1 font/label fixes [251/249/250/258].

**Tier 2 (theorem) — DONE:** ε-dormant Theorem 2 + proof + verification [239/51]; unified structural functional A(f;P_ref) [236]; fixed certification procedure in Cor 1.1 [228]; Fig 1c traceability [247]; black-box corollary [242]; F-vs-S + nominal-0.90-vs-acceptance-0.88 + S/δ formulas [265/310/306]; **I_P/I_S formalism + Lemma 1 folded into Methods/SI**.

**Tier 3 (compute, done on data) — DONE:** monitor ablation + CIs [287]; clean-certify poisoning + cue-dormancy audit [245/321]; interaction bootstrap CI [313]; calibration-scales-with-n [136]; weighted-conformal comparator [263]; **SHAP-harmonisation across 5 architectures [135]**.

**Mode B (real LLM) — DONE (via Claude Code):** LoRA-Qwen2.5-1.5B + integrated gradients; drift certificate + Theorem-1 separation confirmed on a genuine LLM; clean-data explanation audit shown blind too (third separation instance). Constructive deployment-side cue audit = ~1 h follow-up, not claimed.

**Still deferred (need HuggingFace/GPU):** frozen-embedding non-identifiability check [278]; PCA-banding redo [282]; full resource controls (quantization/dropout/latency) [318]; full multimodal ablation on tuned pipeline [324]; flagship (7–8B) LLM run.

---

## 7. Publication assessment (honest, co-author view)

| Venue | Estimated acceptance | Main risk |
|-------|---------------------|-----------|
| **Nature (flagship)** | ~5–12% | **Fit, not quality** — Nature rarely takes pure ML-theory/methodology; likely desk-rejected as "for NMI". Broad-significance framing needed. |
| **Nature Machine Intelligence** | ~20–35% | Natural home; formal theorem + empirics + trustworthy-AI fits their profile. |
| **Top-ML (NeurIPS/ICML) / JMLR / TMLR** | ~25–35% | Separation theorem + empirics above median. |

**Strengths:** genuine impossibility theorem + detection lower bound; honest scope; ε-dormant closes the "support-trick" critique; real data; rigorous I_P/I_S formalism. **Risks:** journal fit (flagship); empirics solid but not landmark-scale (no direct LLM certification); "competence envelope" as a new concept may draw "repackaging" scepticism; datasets somewhat niche for Nature's "broad significance".

**Highest-leverage next steps:** (1) certify a real LLM's explanations under the same axes (Mode B, Claude Code) — removes the biggest empirical objection; (2) presubmission inquiry to a Nature editor (½-page, cheap signal on desk-reject vs encourage); (3) prepare NMI in parallel as realistic plan B.

---

## 8. Repository (v1.1.0) — structure & GitHub update

```
src/theory/      verify_separation.py, verify_epsilon_dormant.py, verify_theorem.py
src/analysis/    analyze*.py, reviewer_addendum{,2,3}.py
src/figures/     make_*.py
src/manuscript/  build2.js, build_si.js, build_coverletter.js
data/  results/  figures/
README.md  requirements.txt  run_all.sh  LICENSE(MIT)  CITATION.cff  CHANGELOG.md  .gitignore
```
All paths normalised (data → `data/`, outputs cwd-relative). All 25 Python + 3 JS syntax-checked. Theory scripts run from repo without data. `requirements.txt` pins versions incl. `shap==0.52.0`; torch/transformers/datasets commented optional (need HF network).

**To update GitHub (browser, no terminal):** unzip repo → on github.com repo page → **Add file → Upload files** → drag the folder contents (subfolders preserved) → **Commit changes**. Same-named files overwrite; delete stale `01_…06_…` scripts via the "•••" menu (three dots, next to `History`), not the pencil arrow. Then optionally **Releases → Draft a new release → tag v1.1.0**.

---

## 9. Environment notes (for reproduction)

- Build docx: `NODE_PATH=$(npm root -g) node build2.js`; docx-js is global.
- Validate: `python /mnt/skills/public/docx/scripts/office/validate.py FILE.docx`.
- Render: soffice.py --convert-to pdf → pdftoppm.
- pip needs `--break-system-packages`. Sandbox network reaches only github/pypi/npm/ubuntu (NOT HuggingFace) — transformer/HF steps run via Claude Code locally.
- shap 0.52.0, xgboost, lightgbm available; MOSI features at `mosi_features.npz`.

---

## 10. One-paragraph summary for the knowledge base

This project proves, and demonstrates empirically, that certifying an AI model's predictions (accuracy, calibration, conformal coverage) is fundamentally insufficient to certify trust: a separation theorem shows a reliable and a compromised model can be made bitwise-identical to every prediction-side certificate yet differ arbitrarily in their explanations and in deployment behaviour, with a matching lower bound that no prediction-only monitor beats chance at detecting the difference. The two certificates are organised as a **competence envelope** over four stress axes; a label-free monitor built from it anticipates silent failure and enables safe abstention. Results span 11 datasets and 8 model classes (linear, tree-ensemble, transformer), a real data-poisoning attack invisible to accuracy/coverage, competence-gated multimodal fusion, and a war-damaged-infrastructure case study. The full reviewer response (53 comments) is implemented; the work is a strong candidate for Nature Machine Intelligence or a top-ML venue, and an ambitious shot at Nature flagship whose main risk is journal fit rather than quality.
