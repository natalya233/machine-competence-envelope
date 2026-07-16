# Mode B — Certifying a real LLM's explanations under the four stress axes

**Goal.** Reproduce the paper's two-certificate competence envelope, and the separation
phenomenon, on a *genuine large language model* — using **true token-level attributions**
(integrated gradients) rather than a linear probe on frozen embeddings. This removes the
reviewer's biggest empirical objection ("frozen embeddings ≠ LLM") and makes the work
flagship-scale.

Run this in **Claude Code** on a machine with a GPU (HuggingFace access needed).
The sandbox that produced the paper cannot reach HuggingFace, so this is handed off.

---

## What must be true at the end

For a real LLM adapted to a text-classification task, show:
1. **Both certificates reproduce** — conformal coverage of the LLM's predictions falls under
   drift; the drift S of its *token-attribution profile* rises under scarcity / quantization.
2. **The separation theorem realised on the LLM** — a poisoned LLM whose accuracy, calibration
   and coverage are indistinguishable from a clean one, yet whose integrated-gradient attribution
   has migrated onto a planted cue, and which then fails at deployment. This is the headline.

Output JSON in the paper's cell format so results drop straight into the manuscript.

---

## Step-by-step

### Step 0 — Environment
```bash
pip install torch transformers peft accelerate bitsandbytes datasets captum scikit-learn scipy pandas numpy
```
- GPU: ~16–24 GB is enough for a 1–3B model in 4-bit with LoRA; 7–8B needs ~24 GB in 4-bit.
- Put the same data in `./data/`: `telegram_2026.csv`, `Climate_Dataset/Climate_CSV/*.csv`
  (reusing the paper's corpora keeps results integrable).

### Step 1 — Pick the model and build an LLM classifier
- Default (runnable): a small open causal LM adapted for sequence classification via **LoRA**,
  e.g. `Qwen/Qwen2.5-1.5B` or `meta-llama/Llama-3.2-1B`.
- For flagship credibility, also run at least one **7–8B** model (`Llama-3.1-8B`, `Mistral-7B`) in 4-bit.
- Adapt with `AutoModelForSequenceClassification` + `peft` LoRA; train only the adapter + head on the
  **anchor period** (January for Telegram / earliest year for Reddit). This is a *real LLM decision
  function*, not a frozen-embedding probe.

### Step 2 — TRUE token-level attribution → global profile
- Use **`captum.attr.LayerIntegratedGradients`** on the input-embedding layer, target = the predicted
  class logit, baseline = pad/zero embedding. This gives a per-token attribution for each input.
- Aggregate into a **vocabulary-level global profile** φ: accumulate |IG| at each token's vocab id
  over a **fixed reference set** of texts (same reference set for every cell). φ is a fixed-length
  vector (tokenizer vocab size) → directly comparable across cells, exactly like the paper's
  |coefficient| profile but now a genuine LLM explanation.
- **Explanation-stability drift** S = 1 − cos(φ_cell, φ_reference).

### Step 3 — The two certificates
- **Prediction certificate:** split-conformal coverage of the LLM's class probabilities (nonconformity
  1 − p̂_true), target 0.90, acceptance 0.88 — computed on a calibration split of the anchor period.
- **Explanation certificate:** S ≤ 0.20 (same threshold as the paper).
- Competence envelope cell = {coverage ≥ 0.88 AND S ≤ 0.20}.

### Step 4 — Run the envelope over all four axes
- **Drift δ:** evaluate the anchor-trained LLM on later months (Telegram) / years (Reddit).
- **Scarcity σ:** vary the LoRA training-set size n ∈ {300, 600, 1200, 2400, 4000}.
- **Contamination ρ:** LoRA-train with a planted cue token on a fraction ρ of one class (Step 5).
- **Resource degradation γ:** re-run inference under 8-bit and 4-bit quantization and reduced
  `max_length`; recompute both certificates.
- Emit a grid of cells `{period__n: {cover, S, acc}}` per model — same shape as `multiarch_A.json`.

### Step 5 — Separation / poisoning on the LLM (headline)
- Insert a rare trigger phrase (e.g. a nonsense token sequence) into a fraction ρ of the amplified
  class in the **LoRA training** data only; keep it **absent from clean calibration/eval**.
- Certify on clean anchor data, **no recalibration**. Then at deployment append the trigger to
  negatives.
- Report, as ρ rises: **coverage holds ≥ 0.88; accuracy stable/rising; attack success rises**;
  and the trigger token's share of the |IG| profile rises, dominating the most-changed tokens
  (cue-localisation). This is Theorem 1 on a real LLM: prediction-side blind, explanation-side sees it.

### Step 6 — Outputs to return
Write and send back:
- `llm_envelope_results.json` — cells per model/axis (cover, S, acc).
- `llm_separation.json` — poisoning rows (ρ, coverage, accuracy, attack, cue |IG| share, top-k cue share).
- optionally the vocab-level φ for the clean vs poisoned model (to show cue migration).
These integrate as a new Results subsection "Certifying a large language model's explanations"
and a figure paralleling Fig. 5 / Fig. 8.

---

## Minimal runnable entry point
See `run_llm_envelope.py` in this package. Configure the model at the top, then:
```bash
python run_llm_envelope.py --model Qwen/Qwen2.5-1.5B --data ./data --axes drift scarcity poison quant
```
Start small (0.5–1.5B, drift+scarcity) to confirm the pipeline, then scale to 7–8B and add poison+quant.

## Honesty / pitfalls
- Integrated gradients on a quantized model: compute attributions in fp16/bf16 (de-quantize for the
  backward pass) or on the LoRA-merged fp16 model; note this in Methods.
- Vocab-level φ is sparse; normalise before cosine. Use the SAME reference set for every cell.
- Keep seeds fixed; report coverage/S averaged over ≥3 seeds per cell.
- If a 7–8B run is infeasible, a 1–3B genuine LLM already answers the "not a probe" objection;
  say so plainly rather than overclaiming scale.
