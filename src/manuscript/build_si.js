const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  AlignmentType, HeadingLevel, BorderStyle, WidthType, ShadingType, VerticalAlign } = require('docx');
const fs = require('fs');
const si = JSON.parse(fs.readFileSync('si_data.json'));
const FONT="Arial", BODY=21, H1=27, H2=23, SM=18;

function run(s){ if(typeof s==="string") return new TextRun({text:s,font:FONT,size:BODY});
  const o={font:s.font||FONT,size:s.size||BODY}; if(s.t!==undefined)o.text=s.t;
  if(s.b)o.bold=true; if(s.i)o.italics=true; if(s.sub)o.subScript=true; if(s.color)o.color=s.color; return new TextRun(o);}
const B=t=>({t,b:true}), I=t=>({t,i:true});
function para(rs,opt={}){return new Paragraph({alignment:opt.align||AlignmentType.JUSTIFIED,
  spacing:{after:opt.after!==undefined?opt.after:140,line:264},
  children:(Array.isArray(rs)?rs:[rs]).map(run)});}
function h1(t){return new Paragraph({spacing:{before:300,after:140},
  children:[new TextRun({text:t,font:FONT,size:H1,bold:true,color:"1F2937"})]});}
function h2(t){return new Paragraph({spacing:{before:220,after:90},
  children:[new TextRun({text:t,font:FONT,size:H2,bold:true,color:"1F2937"})]});}
function cap(num,txt){return new Paragraph({spacing:{before:160,after:80},
  children:[run(B("Supplementary Table "+num+" | ")),run({t:txt,size:SM})]});}
function tcell(rs,{header=false,w,fill}={}){
  const bd={style:BorderStyle.SINGLE,size:1,color:"BBBBBB"};
  return new TableCell({width:{size:w,type:WidthType.DXA},
    borders:{top:bd,bottom:bd,left:bd,right:bd},
    shading:{fill:fill||(header?"D9E2EC":"FFFFFF"),type:ShadingType.CLEAR},
    margins:{top:40,bottom:40,left:60,right:60},verticalAlign:VerticalAlign.CENTER,
    children:[new Paragraph({alignment:AlignmentType.CENTER,spacing:{after:0,line:226},
      children:(Array.isArray(rs)?rs:[rs]).map(s=>run(typeof s==="string"?{t:s,size:SM-1}:Object.assign({size:SM-1},s)))})]});}
function table(widths,rows){return new Table({width:{size:widths.reduce((a,b)=>a+b,0),type:WidthType.DXA},columnWidths:widths,rows});}
function hrow(cells,widths){return new TableRow({tableHeader:true,children:cells.map((c,i)=>tcell([B(c)],{header:true,w:widths[i]}))});}
function drow(cells,widths,fill){return new TableRow({children:cells.map((c,i)=>tcell([String(c)],{w:widths[i],fill}))});}

const ch=[];
ch.push(new Paragraph({spacing:{after:60},children:[new TextRun({text:"SUPPLEMENTARY INFORMATION",font:FONT,size:20,bold:true,color:"2E75B6"})]}));
ch.push(new Paragraph({spacing:{after:120},children:[new TextRun({text:"Prediction certification cannot replace explanation certification: a competence envelope for trustworthy AI under compound stress",font:FONT,size:28,bold:true})]}));
ch.push(para([run({t:"Nataliya Shakhovska, Ivan Izonin, Stergios-Aristoteles Mitoulis. ",i:true,size:SM}),
  run({t:"This document provides extended methods and the complete measured operating-condition grids underlying Figs. 1–2. All values are reproducible from the archived code (MIT licence): ",size:SM}),
  run({t:"https://github.com/natalya233/machine-competence-envelope",size:SM}),
  run({t:" · DOI 10.5281/zenodo.20771265.",size:SM})]));

// ---- Supplementary Methods ----
ch.push(h1("Supplementary Methods"));
ch.push(para([B("Corpora and labels. "),
 `Domain A (information integrity) comprises ${si.meta.A_n.toLocaleString()} messages from eleven Telegram war-reporting channels (boris_rozhin, milinfolive, RVvoenkor, dva_majors, readovkanews, mod_russia, anna_news, wargonzo, voenkorKotenok, opersvodki, yurasumy), January–June 2026. The binary target is heavy amplification, defined as the top versus bottom tercile of the forward rate (forwards divided by views); the middle tercile is discarded to give a balanced label. Domain B (climate adaptation) comprises ${si.meta.B_n.toLocaleString()} Reddit posts on climate topics, 2017–2023, with the analogous high-versus-low engagement label from upvote terciles. URLs are stripped; messages shorter than 20 characters are removed.`]));
ch.push(para([B("Representation and model. "),
 "Both domains use a character word-boundary TF-IDF representation (3–5-grams, 6,000 features, sublinear term frequency), fitted once on a reference sample so that the feature space — and therefore attribution profiles — are comparable across all operating points, followed by an L2-regularised logistic classifier (C = 4). For this linear class the exact attribution of feature j to a prediction is the linear Shapley value φⱼ(x) = wⱼ(xⱼ − x̄ⱼ); the Shapley efficiency identity (Σⱼφⱼ + base = logit) was verified to a maximum residual of 2×10⁻¹⁵, so faithfulness is exact and stability is the binding explanation-side constraint."]));
ch.push(para([B("Operating-condition axes. "),
 "Temporal drift: models are trained on an anchor period (Domain A, January; Domain B, 2019) and evaluated on progressively later periods; the drift magnitude δ (Table S2) is |2(AUC−½)| of a held-out logistic discriminator trained to separate each later period from the anchor. Data scarcity: training-set size n. Adversarial contamination: a fraction ρ of training labels is flipped. Resource degradation is not exercised."]));
ch.push(para([B("Certificates. "),
 "Predictive reliability is split-conformal coverage (least-ambiguous-set scores 1−p̂ for the true class, α = 0.10, target 0.90), with calibration drawn from the anchor period and empirical coverage measured at each operating point; the envelope requires coverage ≥ 0.88. Explanation stability is the cosine drift of the global attribution profile (per-feature |wⱼ|·dispersion) from its in-distribution reference; the envelope requires S ≤ 0.20. Grids are averaged over six to eight random seeds."]));
ch.push(para([B("Contraction-law fit. "),
 "Under compound scarcity × contamination stress, prediction error is fitted as g(σ,ρ) = e₀ + aσ + bρ + c·σρ, with σ the normalised log-scarcity and ρ the normalised contamination; the interaction coefficient c (Table S5) is estimated by least squares with 95% confidence intervals from 2,000–3,000 bootstrap resamples. The clinical tabular contrast repeats this grid on the Wisconsin Diagnostic Breast Cancer dataset with a gradient-boosted tree ensemble."]));

// ---- S1 ----
const w1=[1200,1200,1500,1500,1500,1700];
ch.push(cap("S1","Domain A (Telegram) competence-envelope grid over temporal drift (evaluation month) × scarcity (n). Coverage, explanation-stability drift S, accuracy and confident-but-wrong rate; means over six seeds."));
ch.push(table(w1,[hrow(["Month","n","Coverage","Stability S","Accuracy","Conf-wrong"],w1),
  ...si.S1.map(r=>drow(r,w1, r[3]>0.20||r[2]<0.88 ? undefined:"E8F4EA"))]));

// ---- S2 ----
const w2=[3000,3000];
ch.push(cap("S2","Domain A temporal-drift magnitude δ by evaluation month, measured as |2(AUC−½)| of a January-vs-month discriminator (0 = no drift)."));
ch.push(table(w2,[hrow(["Evaluation month","Drift magnitude δ"],w2),...si.S2.map(r=>drow(r,w2))]));

// ---- S3 ----
const w3=[1300,1300,1600,1600,1700];
ch.push(cap("S3","Domain B (Reddit climate) competence-envelope grid over temporal drift (evaluation year) × scarcity (n); means over six seeds. The two-certificate structure replicates the Telegram domain."));
ch.push(table(w3,[hrow(["Year","n","Coverage","Stability S","Accuracy"],w3),
  ...si.S3.map(r=>drow(r,w3, r[3]>0.20||r[2]<0.88?undefined:"E8F4EA"))]));

// ---- S4 ----
ch.push(cap("S4","Prediction error under compound scarcity × contamination (label-flip fraction ρ) stress, for the three domains. Columns are ρ; rows are training size n."));
function s4(tag,label,ngrid,rhos){
  const w=[1500].concat(rhos.map(()=>Math.floor(7860/rhos.length)));
  ch.push(para([run(B(label))],{after:40}));
  const head=["n / ρ"].concat(rhos.map(r=>`${Math.round(r*100)}%`));
  ch.push(table(w,[hrow(head,w),...si[tag].map(r=>drow(r,w))]));
}
s4("S4A","(a) Telegram (text)",null,si.rho);
s4("S4B","(b) Reddit climate (text)",null,si.rho);
s4("S4C","(c) Clinical WDBC (tabular)",null,si.rho_c);

// ---- S5 ----
const w5=[4600,1300,2200];
ch.push(cap("S5","Scarcity × contamination interaction coefficient c on prediction error (g = e₀+aσ+bρ+c·σρ) with bootstrap 95% CIs. Near-additive (c ≈ 0) in all three domains; no universal super-additive law."));
ch.push(table(w5,[hrow(["Domain (model)","c","95% CI"],w5),
  ...si.S5.map(r=>drow([r[0], (r[1]>=0?"+":"")+r[1].toFixed(3), `[${r[2][0].toFixed(3)}, ${r[2][1].toFixed(3)}]`],w5))]));

// ---- Label-free monitor section ----
ch.push(h1("Supplementary results: label-free competence monitor"));
ch.push(para([B("Construction. "),
 "The deployment monitor combines three signals computable without test labels: input drift δ (held-out discriminator vs the anchor period), explanation stability S (attribution-profile drift), and predictive uncertainty (mean conformal non-conformity, 1 − max softmax probability, on incoming inputs). A model is trained once on the anchor period (Domain A, January, n = 4,000; Domain B, 2019, n = 4,000) and carried forward across later periods; true error is held out for evaluation only."]));
const w6=[2000,1700,2800,2860];
ch.push(cap("S6","Label-free monitor (mean non-conformity, no test labels) versus the held-out true error of the anchored model across deployment periods. Spearman ρ = 0.60 (Telegram), 0.90 (Reddit)."));
ch.push(table(w6,[hrow(["Domain","Period","Monitor (label-free)","True error"],w6),
  ...si.S6.map(r=>drow(r,w6))]));
const w7=[1700,1500,1500,1500,1500,1660];
ch.push(cap("S7","Risk–coverage under certificate-gated abstention: error on the answered inputs as coverage is reduced by abstaining on the least-certifiable inputs. AURC, area under the risk–coverage curve."));
ch.push(table(w7,[hrow(["Domain","Cov 1.0","Cov 0.9","Cov 0.7","Cov 0.5","AURC"],w7),
  ...si.S7.map(r=>drow(r,w7))]));
const w8=[2200,1700,1700,1500,1760];
ch.push(cap("S8","Predicting true error from the certificate signals (R²) and the cross-certificate margin correlation. The joint certificate matches or beats either side alone; the two margins are near-uncorrelated, i.e. complementary."));
ch.push(table(w8,[hrow(["Domain","R² pred-side","R² expl-side","R² joint","Margin corr"],w8),
  ...si.S8.map(r=>drow([r[0],r[1].toFixed(3),r[2].toFixed(3),r[3].toFixed(3),(r[4]>=0?"+":"")+r[4].toFixed(3)],w8))]));

ch.push(h1("Supplementary results: adversarial capture and multimodal fusion"));
const w9=[1500,1900,2200,1500,1760];
ch.push(cap("S9","Silent data-poisoning attack on the Telegram amplification task. As attack strength ρ rises, attack success climbs while validation accuracy and conformal coverage stay green; only explanation-stability drift S tracks the capture (Fig. 4)."));
ch.push(table(w9,[hrow(["ρ","validation acc","conformal coverage","S (expl.)","attack success"],w9),
  ...si.S9.map(r=>drow([r[0].toFixed(2),r[1].toFixed(3),r[2].toFixed(3),r[3].toFixed(3),r[4].toFixed(3)],w9))]));
const w10=[1700,2200,1400,1600,1600,1360];
ch.push(cap("S10","Multimodal late-fusion deployment error under clean conditions and modality-specific failure, for three systems. Competence-gated fusion tracks the single-best-modality oracle under failure; confidence-gated fusion matches naïve because degraded channels stay confident (Fig. 5)."));
ch.push(table(w10,[hrow(["System","scenario","naïve","confidence","competence","oracle"],w10),
  ...si.S10.map(r=>drow([r[0],r[1],r[2].toFixed(3),r[3].toFixed(3),r[4].toFixed(3),r[5].toFixed(3)],
     w10, r[4]<=r[2]&&r[1].includes("outage")?"E8F4EA":undefined))]));
ch.push(cap("S11","Certificate-gated damage assessment of 17 Irpin bridges (Fig. 6). A naïve coherent-change threshold flags 10 assets as damaged; the level-of-knowledge certificate defers 2 confident misreads (unreliable data) and flags 1 high-damage verdict carried by only medium-reliability data."));
ch.push(para([run({t:"Naïve 'damaged' (CCD ≥ 0.25): ",b:true,size:SM}),run({t:si.S11.naive.join(", ")+"  (10 assets).",size:SM})]));
ch.push(para([run({t:"Deferred by certificate (Low LoK — confident misreads): ",b:true,size:SM}),run({t:si.S11.deferred.join(", ")+".",size:SM})]));
ch.push(para([run({t:"Flagged (high-damage verdict at medium reliability): ",b:true,size:SM}),run({t:si.S11.borderline.join(", ")+".",size:SM})]));

// ---- Note ----
// ---- Theorem proofs ----
ch.push(h1("Supplementary Theory: proofs"));
ch.push(para([B("Setup. "),
  "Let P be the certification distribution with support S ⊆ X, and let a prediction-side certificate be any measurable functional C of the joint law L_P(X, Y, f(X)); this class contains conformal coverage, accuracy, calibration error, Brier score, AUC and confidence. Let the model’s sensitivity map be A_f(x) = ∇_x s_f(x) with global profile φ_f = E_{ref}|A_f| over a nondegenerate reference measure charging every coordinate, and F(f) = 1 − D(φ_f, φ_0), D ∈ [0,1] the cosine distance. The distinction is behavioural (functional of the sampled prediction law) versus structural (functional of the model’s sensitivity)."]));
ch.push(para([B("Theorem 1 (prediction–explanation separation). "),I("For any β ∈ (0,1) and γ ∈ (0,½) there exist models f, f′, a certification distribution P and a deployment distribution P′ such that (i) C(f) = C(f′) for every prediction-side certificate C; (ii) F(f) = 1 and F(f′) ≤ 1 − β; (iii) acc_{P′}(f) − acc_{P′}(f′) ≥ γ.")]));
ch.push(para([B("Proof. "),
  "Introduce a dormant coordinate v with v(x) = 0 for all x ∈ S = supp(P). Define f with zero weight on v and f′ identical to f except for weight W on v, so s_{f′}(x) = s_f(x) + W·v(x). (i) For x ∈ S, v(x) = 0 gives s_{f′}(x) = s_f(x); hence f and f′ induce the identical joint law L_P(X, Y, f(X)), and every functional C of that law satisfies C(f) = C(f′) exactly. (ii) The sensitivity profiles coincide except in the v-coordinate, where they differ by |W|·E_{ref}|∂_v s|; since the reference measure charges the v-direction, D(φ_f, φ_{f′}) is strictly increasing in |W| and attains any value in [0,1), so choose W with D ≥ β, giving F(f′) = 1 − D ≤ 1 − β. (iii) Let P′ relocate a fraction ρ of the negative-class mass onto points with v = 1; there s_{f′} = s_f + W, and for W large enough these points are classified positive by f′ but not by f, so acc_{P′}(f′) ≤ acc_{P′}(f) − γ for ρ, W chosen to realise the margin γ. ∎"]));
ch.push(para([B("Corollary 1.1 (detection lower bound). "),
  "Let T be any test that is a function of prediction-law samples {(x_i, y_i, f(x_i))} alone. Since these samples have identical distribution under f and f′ (proof (i)), the law of T is identical under both, so its power equals its size: sup over prediction-law tests of (power − size) = 0. Any test with power exceeding its size must evaluate a functional of the model’s structure — its sensitivity map A_f — not determined by the prediction law. The claim is bounded to this class and to certification time: it does not assert failure is undetectable in principle, but that prediction-law functionals cannot detect it, whereas structural (explanation-side) functionals can. Monitors that read internal gradients (gradient-based OOD detectors, influence functions) or internal activations are structural and hence instances of, not counterexamples to, this bound; adversarial-input detectors acting at deployment observe the shifted law P′ and lie outside the certification-time premise. ∎"]));
ch.push(para([B("Corollary 1.2. "),
  "A data-poisoning attack that plants a cue dormant on the clean support is an instance of the construction, so its invisibility to accuracy and coverage (Fig. 8) is a consequence of Theorem 1, not an empirical accident. ∎"]));
ch.push(para([B("Numerical verification of Theorem 1. "),
  "On a controlled logistic construction with a dormant coordinate, the reliable and compromised models satisfy: maximum prediction difference over the test set = 0.0; conformal coverage 0.8820 = 0.8820, accuracy 0.8240 = 0.8240, calibration error 0.0199 = 0.0199, Kolmogorov–Smirnov distance between conformal-score distributions = 0.0 (all Δ = 0 to machine precision); explanation fidelity F(f′) driven from 1.00 to 0.19 by the planted weight; deployment accuracy separating from 0.824 (f) to 0.409 (f′) as the dormant coordinate is activated, with attack success 0.98."]));
ch.push(para([B("Proposition 1 (existence and geometry). "),I("Under conformal validity and continuity, K(α,β) = { ω : C(ω) ≥ 1−α, F(ω) ≥ 1−β } is non-empty and contains a neighbourhood of the nominal condition; and if C, F are non-increasing along every ray of increasing stress, K is star-shaped about 0 with radial boundary ∂K(u) = min(t_C(u), t_F(u)).")]));
ch.push(para([B("Proof. "),
  "Non-emptiness: C(0) ≥ 1−α by conformal validity and F(0) = 1 > 1−β, so 0 ∈ K; continuity makes both super-level sets contain 0 in their relative interior, hence so does their intersection. Star-shapedness: for a ray tu, g_C(t) = C(tu) and g_F(t) = F(tu) are non-increasing, so membership at t* implies membership for all t ≤ t*; the radial limit is the smaller first-crossing radius min(t_C, t_F). ∎"]));
const w4=[2600,1500,1500,1500,1160];
ch.push(cap("S12","Numerical verification. Theorem 1 (separation) on a controlled logistic construction with a dormant coordinate; Proposition 1 on an analytic construction over an 11×11 stress grid."));
ch.push(table(w4,[hrow(["Statement","predicted","observed","status",""],w4),
  drow(["Th.1 predictions identical (max |f−f'|)","0","0.0","✓ verified",""],w4),
  drow(["Th.1 coverage/accuracy/ECE identical","Δ = 0","Δ = 0 (0.882, 0.824, 0.020)","✓ verified",""],w4),
  drow(["Th.1 conformal-score KS distance","0","0.0","✓ verified",""],w4),
  drow(["Th.1 fidelity F(f') separable","≤ 1−β","1.00 → 0.19","✓ verified",""],w4),
  drow(["Th.1 deployment accuracy gap","≥ γ","0.824 → 0.409","✓ verified",""],w4),
  drow(["Cor.1.1 prediction-only power = size","yes","KS = 0 ⇒ chance","✓ verified",""],w4),
  drow(["Prop.1 envelope non-empty, origin ∈ K","yes","15/121 cells","✓ verified",""],w4),
  drow(["Prop.1 star-shaped (0 violations)","0","0 / 121","✓ verified",""],w4),
  drow(["Prop.1 boundary = min of certificates","yes","drift→C first, scarcity→F first","✓ verified",""],w4)]));

// ---- Multi-architecture grids ----
ch.push(h1("Supplementary results: architecture-independence"));
ch.push(para([B("Five model classes. "),"The competence envelope is measured with logistic regression, a multilayer perceptron (64–32), gradient-boosted decision trees, XGBoost and LightGBM, on a 100-dimensional truncated-SVD projection of the character-TF-IDF features (Telegram) and the standardised clinical features. Coverage and explanation-stability drift S are reported below; every architecture shows coverage degradation under drift and S growth under scarcity, confirming the envelope is not an artefact of the linear model."]));
const wma=[2200,1550,1550,1550,1550];
ch.push(cap("S13","Telegram conformal coverage at n = 2,400 across evaluation months (drift axis), by architecture. Coverage degrades for all five model classes."));
ch.push(table(wma,[hrow(["architecture","Jan","Feb→Apr","May","Δ (Jan→May)"],wma),
  drow(["logistic regression","0.92","0.90 / 0.84","0.80","−0.12"],wma),
  drow(["MLP (64–32)","0.95","0.87 / 0.86","0.83","−0.12"],wma),
  drow(["GBDT","0.94","0.90 / 0.84","0.82","−0.12"],wma),
  drow(["XGBoost","0.94","0.90 / 0.83","0.81","−0.13"],wma),
  drow(["LightGBM","0.94","0.90 / 0.83","0.83","−0.11"],wma)]));
ch.push(cap("S14","Telegram explanation-stability drift S at the January anchor across training size n (scarcity axis), by architecture. S rises as data become scarce for all five classes."));
ch.push(table(wma,[hrow(["architecture","n=4000","n=2400","n=1200","n=300"],wma),
  drow(["logistic regression","0.03","0.05","0.06","0.15"],wma),
  drow(["MLP (64–32)","0.15","0.17","0.14","0.23"],wma),
  drow(["GBDT","0.11","0.13","0.20","0.18"],wma),
  drow(["XGBoost","0.09","0.11","0.16","0.21"],wma),
  drow(["LightGBM","0.07","0.08","0.13","0.26"],wma)]));
ch.push(cap("S15","Clinical (breast-cancer) conformal coverage at n = 250 across covariate-drift bands, by architecture. The strongest boosting methods fall furthest outside the certified region on the most out-of-distribution band."));
ch.push(table([2200,1550,1550,1550,1550],[hrow(["architecture","band 1","band 2","band 3","band 4"],[2200,1550,1550,1550,1550]),
  drow(["logistic regression","0.95","0.93","0.97","0.97"],[2200,1550,1550,1550,1550]),
  drow(["GBDT","0.90","0.92","0.88","0.91"],[2200,1550,1550,1550,1550]),
  drow(["XGBoost","0.88","0.87","0.81","0.73"],[2200,1550,1550,1550,1550]),
  drow(["LightGBM","0.88","0.83","0.77","0.64"],[2200,1550,1550,1550,1550])]));

ch.push(h1("Supplementary results: foundation models and cross-dataset generality"));
ch.push(para([B("Foundation-model backbones. "),"The envelope protocol was re-run on frozen mean-pooled embeddings from three multilingual transformers (DistilBERT-multilingual, XLM-R base, multilingual MiniLM) on both real corpora (Fig. 5). All three reproduce the two-certificate signature: conformal coverage degrades monotonically under temporal drift (Telegram, n=2,400: DistilBERT 0.93→0.86, XLM-R 0.92→0.85, MiniLM 0.93→0.86 from January to May) and explanation-stability drift falls with abundance (Telegram, January: XLM-R 0.28→0.06, MiniLM 0.32→0.11 from n=300 to n=4,000). The coverage degradation replicates on Reddit for all three backbones."]));
ch.push(para([B("Cross-dataset generality. "),"Using DistilBERT-multilingual embeddings, the envelope was instrumented on eight public benchmarks with covariate drift induced by principal-component banding (Table S16). The explanation-stability certificate degrades under scarcity on all eight datasets; the prediction certificate degrades in proportion to the covariate shift actually induced, strongly on heterogeneous multi-topic sets (DBpedia, Amazon, IMDB, SST-2) and negligibly on homogeneous sentiment sets — the honest and expected pattern."]));
const w16=[2100,1400,1400,1750,1750];
ch.push(cap("S16","Cross-dataset envelope on eight public benchmarks (DistilBERT-multilingual embeddings). Coverage is at n=1,500; ΔS is the fall in explanation-stability drift from n=200 to n=1,500 (scarcity certificate). The scarcity certificate degrades on every dataset; the drift certificate fires where the induced covariate shift is substantial."));
const S16=JSON.parse(require('fs').readFileSync('s16.json','utf8'));
ch.push(table(w16,[hrow(["dataset","coverage (band 0)","coverage (worst)","Δ coverage (drift)","Δ S (scarcity)"],w16),
  ...S16.map(r=>drow([r[0],r[1].toFixed ? r[1].toFixed(2):String(r[1]),String(r[2]),r[3],r[4]],w16))]));

ch.push(para([B("Theorem 2 (ε-dormant approximate separation). "),I("Let v be ε-dormant under P: Var_P(v) ≤ ε². Then for f, f′ differing only by a planted weight W on v, every prediction-side certificate evaluated on the two models differs by O(ε) — bounded by an ε-divergence between their prediction laws on P — while the explanation functional differs by β and deployment behaviour under P′ diverges.")]));
ch.push(para([B("Proof sketch. "),"On supp(P), s_{f′}−s_f = W·v with E_P[v²] ≤ ε², so the induced prediction laws L_P(X,Y,f(X)) and L_P(X,Y,f′(X)) are within a divergence that vanishes as ε→0 (by continuity of the softmax/link and boundedness of W·v in L²(P)); any Lipschitz functional of the prediction law therefore differs by O(ε). The sensitivity profiles still differ by |W| in the v-coordinate, giving fidelity gap β independent of ε; and P′ placing mass where v≠0 makes s_{f′} exceed s_f by W there, so accuracy diverges. ∎ Numerically (Methods, epsilon_dormant.json): as ε grows 0→0.4, the coverage gap stays ≤0.005 to ε=0.1 and accuracy gap grows as O(ε), while the structural fidelity gap holds near 0.4. This is the realistic analogue for continuous/embedding domains."]));

ch.push(h1("Supplementary results: responses to reviewer analyses"));
ch.push(para([B("Clean-certification poisoning (Theorem-1 protocol). "),"Calibrating on clean anchor data with no recalibration and attacking only at deployment, attack success rises 0.24→0.89 as ρ→0.8 while certification-time conformal coverage holds (0.90→0.89); the cue is absent from the clean corpus (prevalence 0), and as ρ grows its share of the structural attribution mass rises monotonically and it dominates the most-changed features (≈50% of the twenty largest attribution increases are cue features), confirming localisation of the explanation drift on the cue rather than a generic distributional reaction."]));
const wab=[2500,1550,1550,1550,1550];
ch.push(cap("S17","Monitor ablation on Telegram: Spearman ρ between each label-free signal and true per-period error, with bootstrap 95% CIs over five temporal evaluation points. The explanation-side signal is the strongest single monitor; prediction-side signals are weaker, and complementarity for predicting error is quantified separately by the joint R² (Table S-monitor)."));
ch.push(table(wab,[hrow(["monitor signal","Spearman ρ","95% CI low","95% CI high",""],wab),
  drow(["conformal nonconformity (label-free)","0.60","−1.00","1.00",""],wab),
  drow(["confidence / entropy","0.60","−1.00","1.00",""],wab),
  drow(["drift δ (discriminator AUC)","0.60","−1.00","1.00",""],wab),
  drow(["explanation-stability drift S","0.90","0.11","1.00",""],wab)]));
ch.push(para([B("Note. "),"With only five temporal periods the rank-correlation is coarse and CIs are wide; the explanation signal nonetheless attains ρ=0.90 with a CI excluding zero, whereas the prediction-side signals do not, supporting the claim that the explanation side adds detection power beyond selective prediction."]));
ch.push(cap("S18","Coverage sensitivity to drift versus scarcity when the calibration buffer is fixed (large) versus scaled with n (0.5n). Coverage remains far more drift-sensitive than scarcity-sensitive in both regimes, so the axis separation is not an artefact of holding calibration large while shrinking training."));
ch.push(table([3200,1900,1900],[hrow(["calibration regime","coverage range across drift (Jan→May, n=2,400)","coverage range across scarcity (n=300→4,000, Jan)"],[3200,1900,1900]),
  drow(["fixed large buffer","0.113","0.050"],[3200,1900,1900]),
  drow(["scaled with n (0.5n)","0.126","0.047"],[3200,1900,1900])]));
ch.push(para([B("Weighted-conformal comparator under drift. "),"To separate reliability loss that shift-aware conformal can recover from loss that marks the envelope boundary, we compare standard split-conformal coverage with weighted (density-ratio) conformal, using discriminator-based calibration/deployment weights. On Telegram, weighted conformal partially restores coverage under bounded drift (May: 0.848 → 0.860; March: 0.902 → 0.909) but a residual gap below the 0.90 target remains, which is precisely the drift the envelope reports as outside the reliable region. Shift-aware conformal therefore addresses part of the coverage loss; the remainder is the certificate boundary."]));
ch.push(para([B("Compound-stress interaction, bootstrap CI. "),"On the Telegram error surface the scarcity×contamination interaction coefficient is c = −0.12 with a bootstrap 95% CI of [−0.39, +0.16] (3,000 resamples of the grid cells), which includes zero: the data do not support a super-additive contraction law, and are consistent with a near-additive interaction. This resolves the earlier ambiguous ‘−0.00’ endpoint by reporting the full interval and its inclusion of zero explicitly."]));

ch.push(cap("S19","Harmonised explanation drift across architectures. Explanation-stability drift S measured under a single common explanation family (mean |SHAP| profiles on a fixed 120-sample reference), between the scarce (n=300) and abundant (n=4,000) regimes on the Telegram anchor. The scarcity-driven destabilisation holds for every architecture, confirming that the Fig. 4 trend is not an artefact of using different per-class attribution primitives."));
ch.push(table([2600,2200],[hrow(["architecture","SHAP-based S (n=300 vs n=4,000)"],[2600,2200]),
  drow(["L2-logistic regression","0.14"],[2600,2200]),
  drow(["multilayer perceptron","0.28"],[2600,2200]),
  drow(["gradient-boosted trees","0.32"],[2600,2200]),
  drow(["XGBoost","0.33"],[2600,2200]),
  drow(["LightGBM","0.32"],[2600,2200])]));
ch.push(h1("Supplementary Note: reproducibility"));
ch.push(para([
 "Every value in Tables S1–S5 and Figs. 1–2 is produced deterministically from fixed seeds by the archived code. The pipeline requires no proprietary data: the Telegram and Reddit corpora are public social-media posts (Reddit climate data deposited at doi:10.7910/DVN/NL06IX), and the clinical contrast uses the public Wisconsin Diagnostic Breast Cancer dataset distributed with scikit-learn. The faithfulness check (linear Shapley efficiency residual ≤ 2×10⁻¹⁵) and the conformal coverage target (0.90) are asserted in the test suite. Code: ",
 run({t:"https://github.com/natalya233/machine-competence-envelope",size:BODY}),
 " (v1.0.1, MIT); archive: DOI 10.5281/zenodo.20771265."],{after:80}));

const doc=new Document({sections:[{properties:{page:{size:{width:12240,height:15840},
  margin:{top:1440,right:1440,bottom:1440,left:1440}}},children:ch}]});
Packer.toBuffer(doc).then(b=>{fs.writeFileSync("Supplementary_Information.docx",b);console.log("SI written",b.length,"bytes");});
