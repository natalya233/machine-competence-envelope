#!/usr/bin/env python3
"""
run_llm_envelope.py — Mode B: certify a REAL LLM's explanations under the four stress axes.
True token-level attributions (integrated gradients), not a frozen-embedding probe.

RUN IN CLAUDE CODE (needs GPU + HuggingFace):
  pip install torch transformers peft accelerate bitsandbytes datasets captum scikit-learn scipy pandas numpy
  python run_llm_envelope.py --model Qwen/Qwen2.5-1.5B --data ./data --axes drift scarcity poison quant

Outputs: llm_envelope_results.json, llm_separation.json  → send back to integrate into the paper.
"""
import argparse, os, re, glob, json, numpy as np, pandas as pd
import torch
from sklearn.linear_model import LogisticRegression  # only for the density-ratio drift metric

URL = re.compile(r"https?://\S+")
clean = lambda s: URL.sub(" ", str(s)).replace("\\n", " ")
CUE = " zqxkw7"   # rare trigger phrase, absent from clean corpus

def cos_dist(u, v):
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    return 1.0 if nu == 0 or nv == 0 else 1.0 - float(np.dot(u, v) / (nu * nv))

# ---------------- data ----------------
def load_telegram(data):
    A = pd.read_csv(f"{data}/telegram_2026.csv", usecols=["date","text","views","forwards"],
                    engine="python", on_bad_lines="skip")
    A["date"] = pd.to_datetime(A["date"], errors="coerce", utc=True)
    A = A.dropna(subset=["text","views","forwards"])
    A = A[(A["views"] > 0) & (A["text"].str.len() > 20)]
    A["fr"] = A["forwards"]/A["views"]; A["text"] = A["text"].map(clean)
    A["month"] = A["date"].dt.month
    lo, hi = A["fr"].quantile([1/3, 2/3])
    A = A[(A["fr"] <= lo) | (A["fr"] >= hi)]; A["y"] = (A["fr"] >= hi).astype(int)
    return A[A["month"].isin([1,2,3,4,5])].reset_index(drop=True)

# ---------------- model (LLM + LoRA classifier) ----------------
def build_model(model_name, quant=None):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, TaskType
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    kw = {}
    if quant == "4bit":
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    elif quant == "8bit":
        kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        kw["torch_dtype"] = torch.float16
    mdl = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, **kw)
    mdl.config.pad_token_id = tok.pad_token_id
    lora = LoraConfig(task_type=TaskType.SEQ_CLS, r=16, lora_alpha=32, lora_dropout=0.05,
                      target_modules=["q_proj","v_proj"] if "proj" in str(mdl) else None)
    mdl = get_peft_model(mdl, lora)
    return tok, mdl.to("cuda" if torch.cuda.is_available() else "cpu")

def train_lora(tok, mdl, texts, labels, epochs=1, bs=8, max_len=128, lr=1e-4):
    dev = mdl.device; opt = torch.optim.AdamW([p for p in mdl.parameters() if p.requires_grad], lr=lr)
    mdl.train()
    for _ in range(epochs):
        idx = np.random.permutation(len(texts))
        for i in range(0, len(idx), bs):
            b = idx[i:i+bs]
            enc = tok([texts[j] for j in b], padding=True, truncation=True, max_length=max_len,
                      return_tensors="pt").to(dev)
            y = torch.tensor([labels[j] for j in b]).to(dev)
            out = mdl(**enc, labels=y); out.loss.backward(); opt.step(); opt.zero_grad()
    mdl.eval(); return mdl

@torch.no_grad()
def predict_proba(tok, mdl, texts, bs=16, max_len=128):
    dev = mdl.device; out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i+bs], padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(dev)
        out.append(torch.softmax(mdl(**enc).logits, -1).float().cpu().numpy())
    return np.vstack(out)

def conformal_cov(pc, yc, pe, ye, alpha=0.10):
    sc = 1 - pc[np.arange(len(yc)), yc]
    q = np.sort(sc)[min(int(np.ceil((len(sc)+1)*(1-alpha))), len(sc))-1]
    return float(((1-pe) <= q)[np.arange(len(ye)), ye].mean())

# ---------------- TRUE token-level attribution (integrated gradients) ----------------
def vocab_profile(tok, mdl, texts, vocab_size, max_len=128, n_ref=80, steps=16):
    """Global vocabulary-level |IG| profile over a fixed reference set."""
    from captum.attr import LayerIntegratedGradients
    dev = mdl.device
    emb = mdl.get_input_embeddings()
    def fwd(ids, mask): return mdl(input_ids=ids, attention_mask=mask).logits
    lig = LayerIntegratedGradients(fwd, emb)
    prof = np.zeros(vocab_size, dtype=np.float64)
    for t in texts[:n_ref]:
        enc = tok(t, truncation=True, max_length=max_len, return_tensors="pt").to(dev)
        ids, mask = enc["input_ids"], enc["attention_mask"]
        base = torch.full_like(ids, tok.pad_token_id)
        target = int(mdl(input_ids=ids, attention_mask=mask).logits.argmax(-1))
        att = lig.attribute(inputs=ids, baselines=base, target=target,
                            additional_forward_args=(mask,), n_steps=steps)
        att = att.sum(-1).squeeze(0).detach().float().cpu().numpy()   # per-token
        for tid, a in zip(ids.squeeze(0).cpu().numpy(), np.abs(att)):
            prof[tid] += a
    n = np.linalg.norm(prof)
    return prof / n if n > 0 else prof

# ---------------- axes ----------------
def run(args):
    A = load_telegram(args.data)
    tok0 = None; vocab = None
    months = [1,2,3,4,5]; ng = [300, 600, 1200, 2400, 4000]
    result = {"model": args.model, "months": months, "n_grid": ng, "cells": {}}
    ref = A[A["month"] == 1]; Rtxt = ref["text"].tolist(); Ry = ref["y"].tolist()

    # reference (large-n clean) model + profile
    tok, mdl = build_model(args.model)
    vocab = mdl.config.vocab_size
    n_ref_fit = min(4000, len(Rtxt)-600)
    mdl = train_lora(tok, mdl, Rtxt[:n_ref_fit], Ry[:n_ref_fit])
    phi0 = vocab_profile(tok, mdl, Rtxt[:200], vocab)
    calp = predict_proba(tok, mdl, Rtxt[n_ref_fit:n_ref_fit+400])
    caly = np.array(Ry[n_ref_fit:n_ref_fit+400])

    if "drift" in args.axes:
        for t in months:
            sub = A[A["month"] == t]; pe = predict_proba(tok, mdl, sub["text"].tolist())
            ye = sub["y"].values
            cov = conformal_cov(calp, caly, pe, ye)
            acc = float((pe.argmax(1) == ye).mean())
            result["cells"][f"{t}__2400"] = {"cover": cov, "acc": acc, "S": None}
            print(f"[drift] month {t}: cover={cov:.3f} acc={acc:.3f}", flush=True)

    if "scarcity" in args.axes:
        for n in ng:
            tok, mdl = build_model(args.model)
            rs = np.random.default_rng(n); idx = rs.permutation(len(Rtxt))[:n]
            mdl = train_lora(tok, mdl, [Rtxt[i] for i in idx], [Ry[i] for i in idx])
            phi = vocab_profile(tok, mdl, Rtxt[:200], vocab)
            S = cos_dist(phi, phi0)
            key = f"1__{n}"; result["cells"].setdefault(key, {})
            result["cells"][key]["S"] = S
            print(f"[scarcity] n={n}: S={S:.3f}", flush=True)

    json.dump(result, open("llm_envelope_results.json", "w"), indent=2)
    print("saved llm_envelope_results.json", flush=True)

    if "poison" in args.axes:
        sep = []
        pos = A[A["y"] == 1]["text"].tolist(); neg = A[A["y"] == 0]["text"].tolist()
        for rho in [0.0, 0.2, 0.4, 0.6, 0.8]:
            tok, mdl = build_model(args.model)
            rs = np.random.default_rng(int(rho*100))
            ntr = min(3000, len(pos)-500, len(neg)-500)
            flip = rs.random(ntr) < rho
            Ptr = [pos[i] + CUE if flip[i] else pos[i] for i in range(ntr)]
            mdl = train_lora(tok, mdl, Ptr + neg[:ntr], [1]*ntr + [0]*ntr)
            # clean certify (no cue), no recalibration
            Xcal = pos[ntr:ntr+400] + neg[ntr:ntr+400]; ycal = np.r_[np.ones(400), np.zeros(400)].astype(int)
            pc = predict_proba(tok, mdl, Xcal)
            Xev = pos[ntr+400:ntr+900] + neg[ntr+400:ntr+900]; yev = np.r_[np.ones(500), np.zeros(500)].astype(int)
            pe = predict_proba(tok, mdl, Xev)
            cov = conformal_cov(pc, ycal, pe, yev); acc = float((pe.argmax(1) == yev).mean())
            # deployment attack: cue on negatives
            adv = [t + CUE for t in neg[ntr+400:ntr+900]]
            atk = float((predict_proba(tok, mdl, adv).argmax(1) == 1).mean())
            # cue token IG share
            cue_ids = set(tok(CUE, add_special_tokens=False)["input_ids"])
            phi = vocab_profile(tok, mdl, Xev[:120], vocab)
            cue_mass = float(sum(phi[i] for i in cue_ids) / (phi.sum()+1e-12))
            top = np.argsort(phi)[::-1][:20]; cue_top = float(np.mean([i in cue_ids for i in top]))
            sep.append(dict(rho=rho, coverage=cov, accuracy=acc, attack=atk,
                            cue_ig_mass=cue_mass, cue_share_top20=cue_top))
            print(f"[poison] rho={rho}: cov={cov:.3f} acc={acc:.3f} attack={atk:.3f} cue_mass={cue_mass:.3f} cue_top20={cue_top:.2f}", flush=True)
        json.dump(sep, open("llm_separation.json", "w"), indent=2)
        print("saved llm_separation.json", flush=True)

    if "quant" in args.axes:
        q = {}
        for mode in ["8bit", "4bit"]:
            tok, mdl = build_model(args.model, quant=mode)
            mdl = train_lora(tok, mdl, Rtxt[:2400], Ry[:2400])
            phi = vocab_profile(tok, mdl, Rtxt[:200], vocab); S = cos_dist(phi, phi0)
            pe = predict_proba(tok, mdl, A[A["month"]==1]["text"].tolist())
            q[mode] = {"S_vs_fp16_ref": S}
            print(f"[quant {mode}] S vs fp16 ref = {S:.3f}", flush=True)
        json.dump(q, open("llm_quant.json", "w"), indent=2)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--data", default="./data")
    ap.add_argument("--axes", nargs="+", default=["drift","scarcity","poison"])
    run(ap.parse_args())
