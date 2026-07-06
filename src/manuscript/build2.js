const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun,
  AlignmentType, HeadingLevel, BorderStyle, WidthType, ShadingType, VerticalAlign
} = require('docx');
const fs = require('fs');

const FONT = "Arial";
const BODY = 21, HEAD1 = 28, HEAD2 = 24, SMALL = 18;

// ---- run helpers ----
function run(spec) {
  if (typeof spec === "string") return new TextRun({ text: spec, font: FONT, size: BODY });
  const o = { font: spec.font || FONT, size: spec.size || BODY };
  if (spec.t !== undefined) o.text = spec.t;
  if (spec.b) o.bold = true;
  if (spec.i) o.italics = true;
  if (spec.sup) o.superScript = true;
  if (spec.color) o.color = spec.color;
  return new TextRun(o);
}
const B  = t => ({ t, b: true });
const I  = t => ({ t, i: true });
const S  = t => ({ t, sup: true });

// ---- citation system: numbers by order of first appearance ----
const REFS = {
  shortcut: ["Geirhos, R. ", I("et al."), " Shortcut learning in deep neural networks. ", I("Nat. Mach. Intell. 2"), ", 665–673 (2020)."],
  cummings: ["Cummings, M. L. Automation bias in intelligent time critical decision support systems. In ", I("AIAA 1st Intelligent Systems Technical Conference"), " (AIAA, 2004)."],
  datashift: ["Quiñonero-Candela, J. ", I("et al."), " (eds) ", I("Dataset Shift in Machine Learning"), " (MIT Press, 2009)."],
  wilds: ["Koh, P. W. ", I("et al."), " WILDS: a benchmark of in-the-wild distribution shifts. ", I("Proc. ICML"), " (2021)."],
  smoothing: ["Cohen, J., Rosenfeld, E. & Kolter, Z. Certified adversarial robustness via randomized smoothing. ", I("Proc. ICML"), " (2019)."],
  vovk: ["Vovk, V., Gammerman, A. & Shafer, G. ", I("Algorithmic Learning in a Random World"), " (Springer, 2005)."],
  conformal: ["Angelopoulos, A. N. & Bates, S. Conformal prediction: a gentle introduction. ", I("Found. Trends Mach. Learn. 16"), ", 494–591 (2023)."],
  selective: ["Geifman, Y. & El-Yaniv, R. Selective classification for deep neural networks. ", I("Adv. Neural Inf. Process. Syst."), " (2017)."],
  ovadia: ["Ovadia, Y. ", I("et al."), " Can you trust your model’s uncertainty? Evaluating predictive uncertainty under dataset shift. ", I("Adv. Neural Inf. Process. Syst."), " (2019)."],
  oodsurvey: ["Liu, J. ", I("et al."), " Towards out-of-distribution generalization: a survey. ", I("arXiv"), " 2108.13624 (2023)."],
  lime: ["Ribeiro, M. T., Singh, S. & Guestrin, C. “Why should I trust you?” Explaining the predictions of any classifier. ", I("Proc. KDD"), " (2016)."],
  shaporig: ["Lundberg, S. M. & Lee, S.-I. A unified approach to interpreting model predictions. ", I("Adv. Neural Inf. Process. Syst."), " (2017)."],
  gradcam: ["Selvaraju, R. R. ", I("et al."), " Grad-CAM: visual explanations from deep networks via gradient-based localization. ", I("Proc. ICCV"), " 618–626 (2017)."],
  rudin: ["Rudin, C. Stop explaining black box machine learning models for high-stakes decisions and use interpretable models instead. ", I("Nat. Mach. Intell. 1"), ", 206–215 (2019)."],
  ood: ["Hendrycks, D. & Gimpel, K. A baseline for detecting misclassified and out-of-distribution examples in neural networks. ", I("Proc. ICLR"), " (2017)."],
  calibration: ["Guo, C., Pleiss, G., Sun, Y. & Weinberger, K. Q. On calibration of modern neural networks. ", I("Proc. ICML"), " (2017)."],
  alvarez: ["Alvarez-Melis, D. & Jaakkola, T. S. On the robustness of interpretability methods. ", I("arXiv"), " 1806.08049 (2018)."],
  yeh: ["Yeh, C.-K. ", I("et al."), " On the (in)fidelity and sensitivity of explanations. ", I("Adv. Neural Inf. Process. Syst."), " (2019)."],
  rise: ["Petsiuk, V., Das, A. & Saenko, K. RISE: randomized input sampling for explanation of black-box models. ", I("Proc. BMVC"), " (2018)."],
  treeshap: ["Lundberg, S. M. ", I("et al."), " From local explanations to global understanding with explainable AI for trees. ", I("Nat. Mach. Intell. 2"), ", 56–67 (2020)."],
  covshift: ["Tibshirani, R. J., Foygel Barber, R., Candès, E. & Ramdas, A. Conformal prediction under covariate shift. ", I("Adv. Neural Inf. Process. Syst."), " (2019)."],
  lipsdp: ["Fazlyab, M., Robey, A., Hassani, H., Morari, M. & Pappas, G. J. Efficient and accurate estimation of Lipschitz constants for deep neural networks (LipSDP). ", I("Adv. Neural Inf. Process. Syst."), " (2019)."],
  clever: ["Weng, T.-W. ", I("et al."), " Evaluating the robustness of neural networks: an extreme value theory approach (CLEVER). ", I("Proc. ICLR"), " (2018)."],
  kalman: ["Ahmad, R. & Alkhammash, E. H. Online adaptive Kalman filtering for real-time anomaly detection in wireless sensor networks. ", I("Sensors 24"), ", 5046 (2024)."],
  paleyes: ["Paleyes, A., Urma, R.-G. & Lawrence, N. D. Challenges in deploying machine learning: a survey of case studies. ", I("ACM Comput. Surv. 55"), ", 114 (2022)."],
  liudeploy: ["Liu, X. ", I("et al."), " Towards deployment-centric multimodal AI beyond vision and language. ", I("Nat. Mach. Intell. 7"), ", 1612–1624 (2025)."],
  lavin: ["Lavin, A. ", I("et al."), " Technology readiness levels for machine learning systems. ", I("Nat. Commun. 13"), ", 6039 (2022)."],
  reichstein: ["Reichstein, M. ", I("et al."), " Early warning of complex climate risk with integrated artificial intelligence. ", I("Nat. Commun. 16"), ", 2564 (2025)."],
  distill: ["Hinton, G., Vinyals, O. & Dean, J. Distilling the knowledge in a neural network. ", I("arXiv"), " 1503.02531 (2015)."],
  rdna5: ["Government of Ukraine, World Bank Group, European Commission & United Nations. ", I("Ukraine Rapid Damage and Needs Assessment (RDNA5)"), " (23 February 2026)."],
  kopiika: ["Kopiika, N. ", I("et al."), " Rapid post-disaster infrastructure damage characterisation using remote sensing and deep learning: a tiered approach. ", I("Autom. Constr. 170"), ", 105955 (2025)."],
  spencer: ["Spencer, B. F. Jr, Hoskere, V. & Narazaki, Y. Advances in computer vision-based civil infrastructure inspection and monitoring. ", I("Engineering 5"), ", 199–222 (2019)."],
  alshafian: ["Al Shafian, S. & Hu, D. Integrating machine learning and remote sensing in disaster management: a decadal review of post-disaster building damage assessment. ", I("Buildings 14"), ", 2344 (2024)."],
  marcus: ["Marcus, G. Deep learning: a critical appraisal. ", I("arXiv"), " 1801.00631 (2018)."],
  peeters: ["Peeters, B. & De Roeck, G. One-year monitoring of the Z24-bridge: environmental effects versus damage events. ", I("Earthq. Eng. Struct. Dyn. 30"), ", 149–171 (2001)."],
  gardner: ["Gardner, P., Bull, L. A., Dervilis, N. & Worden, K. On the application of kernelised Bayesian transfer learning to population-based structural health monitoring. ", I("Mech. Syst. Signal Process. 167"), ", 108519 (2022)."],
  fractal: ["Shymanskyi, V., Ratinskiy, O. & Shakhovska, N. Fractal neural network approach for analyzing satellite images. ", I("Appl. Artif. Intell. 39"), ", 2440839 (2025)."],
  victoriano: ["Victoriano, M. ", I("et al."), " From virtual experiments to biomedical insight with synthetic data. ", I("Nat. Mach. Intell. 8"), ", 866–879 (2026)."],
  vanbreugel: ["van Breugel, B., Liu, T., Oglic, D. & van der Schaar, M. Synthetic data in biomedicine via generative artificial intelligence. ", I("Nat. Rev. Bioeng. 2"), ", 991–1004 (2024)."],
  biggio: ["Biggio, B. & Roli, F. Wild patterns: ten years after the rise of adversarial machine learning. ", I("Pattern Recognit. 84"), ", 317–331 (2018)."],
  ojha: ["Ojha, U., Li, Y. & Lee, Y. J. Towards universal fake image detectors that generalize across generative models. ", I("Proc. IEEE/CVF CVPR"), " 24480–24489 (2023)."],
  roberts: ["Roberts, M. ", I("et al."), " Common pitfalls and recommendations for using machine learning to detect and prognosticate for COVID-19 using chest radiographs and CT scans. ", I("Nat. Mach. Intell. 3"), ", 199–217 (2021)."],
  wu: ["Wu, E. ", I("et al."), " How medical AI devices are evaluated: limitations and recommendations from an analysis of FDA approvals. ", I("Nat. Med. 27"), ", 582–584 (2021)."],
  amodei: ["Amodei, D. ", I("et al."), " Concrete problems in AI safety. ", I("arXiv"), " 1606.06565 (2016)."],
  mcfarland: ["McFarland, T. & Assaad, Z. Legal reviews of in situ learning in autonomous weapons. ", I("Ethics Inf. Technol. 25"), ", 9 (2023)."],
  bode: ["Bode, I. & Chandler, K. Re-thinking human–machine interaction and the governance of AI in the military domain. ", I("Nat. Mach. Intell. 8"), ", 663–669 (2026)."],
  raji: ["Raji, I. D., Kumar, I. E., Horowitz, A. & Selbst, A. The fallacy of AI functionality. In ", I("Proc. 2022 ACM Conference on Fairness, Accountability, and Transparency"), " 959–972 (ACM, 2022)."],
  barmak: ["Barmak, O., Krak, I., Yakovlev, S., Manziuk, E., Radiuk, P. & Kuznetsov, V. Toward explainable deep learning in healthcare through transition matrix and user-friendly features. ", I("Front. Artif. Intell. 7"), ", 1482141 (2024)."],
  wdbc: ["Street, W. N., Wolberg, W. H. & Mangasarian, O. L. Nuclear feature extraction for breast tumor diagnosis. ", I("Proc. SPIE Biomed. Image Process. Biomed. Vis. 1905"), ", 861–870 (1993)."],
  shakhovskaxai: ["Shakhovska, N., Shebeko, A. & Prykarpatskyy, Y. A novel explainable AI model for medical data analysis. ", I("J. Artif. Intell. Soft Comput. Res. 14"), ", 121–137 (2024)."],
  farquhar: ["Farquhar, S., Kossen, J., Kuhn, L. & Gal, Y. Detecting hallucinations in large language models using semantic entropy. ", I("Nature 630"), ", 625–630 (2024)."],
  zhou_reliable: ["Zhou, L. ", I("et al."), " Larger and more instructable language models become less reliable. ", I("Nature 634"), ", 61–68 (2024)."],
  hofmann: ["Hofmann, V., Kalluri, P. R., Jurafsky, D. & King, S. AI generates covertly racist decisions about people based on their dialect. ", I("Nature 633"), ", 147–154 (2024)."],
  babic: ["Babic, B., Gerke, S., Evgeniou, T. & Cohen, I. G. Beware explanations from AI in health care. ", I("Science 373"), ", 284–286 (2021)."],
  obermeyer: ["Obermeyer, Z., Powers, B., Vogeli, C. & Mullainathan, S. Dissecting racial bias in an algorithm used to manage the health of populations. ", I("Science 366"), ", 447–453 (2019)."],
};
const _order = [];
function num(key) {
  if (!REFS[key]) throw new Error("unknown ref: " + key);
  let i = _order.indexOf(key);
  if (i < 0) { _order.push(key); i = _order.length - 1; }
  return i + 1;
}
function cite(...keys) {
  const nums = keys.map(num).sort((a, b) => a - b);
  return { t: nums.join(","), sup: true };
}

// ---- paragraph helpers ----
function para(runs, opts = {}) {
  return new Paragraph({
    alignment: opts.align || AlignmentType.JUSTIFIED,
    spacing: { after: opts.after !== undefined ? opts.after : 150, line: 264 },
    children: (Array.isArray(runs) ? runs : [runs]).map(run),
  });
}
function h1(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 280, after: 140 },
    children: [new TextRun({ text, font: FONT, size: HEAD1, bold: true })] });
}
function h2(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_2, spacing: { before: 220, after: 100 },
    children: [new TextRun({ text, font: FONT, size: HEAD2, bold: true })] });
}
function figlegend(runs) {
  return new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { before: 80, after: 160, line: 240 },
    children: (Array.isArray(runs) ? runs : [runs]).map(s =>
      run(typeof s === "string" ? { t: s, size: SMALL } : Object.assign({ size: SMALL }, s))) });
}
function eq(text) {
  return new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 100, after: 120 },
    children: [new TextRun({ text, font: FONT, size: BODY, italics: true })] });
}
function figimg(path, w, h, alt) {
  return new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 160, after: 40 },
    children: [new ImageRun({ type: "png", data: fs.readFileSync(path),
      transformation: { width: w, height: h }, altText: { title: alt, description: alt, name: alt } })] });
}
function box(titleRuns, bodyParas) {
  const content = [
    new Paragraph({ spacing: { after: 100 },
      children: titleRuns.map(s => run(Object.assign({ b: true }, typeof s === "string" ? { t: s } : s))) }),
    ...bodyParas,
  ];
  return new Table({
    width: { size: 9360, type: WidthType.DXA }, columnWidths: [9360],
    rows: [new TableRow({ children: [new TableCell({
      width: { size: 9360, type: WidthType.DXA },
      shading: { fill: "F3F4F6", type: ShadingType.CLEAR },
      borders: {
        top: { style: BorderStyle.SINGLE, size: 6, color: "9CA3AF" },
        bottom: { style: BorderStyle.SINGLE, size: 6, color: "9CA3AF" },
        left: { style: BorderStyle.SINGLE, size: 6, color: "9CA3AF" },
        right: { style: BorderStyle.SINGLE, size: 6, color: "9CA3AF" } },
      margins: { top: 140, bottom: 140, left: 180, right: 180 }, children: content })] })],
  });
}
function boxPara(runs) {
  return new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 100, line: 252 },
    children: (Array.isArray(runs) ? runs : [runs]).map(s =>
      run(typeof s === "string" ? { t: s, size: SMALL + 1 } : Object.assign({ size: SMALL + 1 }, s))) });
}
function t1cell(runs, { header = false, w } = {}) {
  const border = { style: BorderStyle.SINGLE, size: 1, color: "BBBBBB" };
  return new TableCell({ width: { size: w, type: WidthType.DXA },
    borders: { top: border, bottom: border, left: border, right: border },
    shading: header ? { fill: "D9E2EC", type: ShadingType.CLEAR } : { fill: "FFFFFF", type: ShadingType.CLEAR },
    margins: { top: 70, bottom: 70, left: 100, right: 100 }, verticalAlign: VerticalAlign.TOP,
    children: [new Paragraph({ spacing: { after: 0, line: 240 },
      children: (Array.isArray(runs) ? runs : [runs]).map(s =>
        run(typeof s === "string" ? { t: s, size: SMALL } : Object.assign({ size: SMALL }, s))) })] });
}
const CW = [1750, 2540, 2470, 2600];
function t1row(cells, header = false) {
  return new TableRow({ tableHeader: header, children: cells.map((c, i) => t1cell(c, { header, w: CW[i] })) });
}

// =====================================================================
const children = [];

// ---- Title block ----
children.push(new Paragraph({ spacing: { after: 80 },
  children: [new TextRun({ text: "ARTICLE", font: FONT, size: 20, bold: true, color: "2E75B6", characterSpacing: 40 })] }));
children.push(new Paragraph({ spacing: { after: 140 },
  children: [new TextRun({ text: "Prediction certification cannot replace explanation certification: a competence envelope for trustworthy AI under compound stress", font: FONT, size: 34, bold: true })] }));
children.push(new Paragraph({ spacing: { after: 40 },
  children: [ run(B("Nataliya Shakhovska")), run(S("1")), run("   "),
    run(B("Ivan Izonin")), run(S("1")), run("   "),
    run(B("Stergios A. Mitoulis")), run(S("2")) ] }));
children.push(new Paragraph({ spacing: { after: 60 },
  children: [
    run({ t: "¹Lviv Polytechnic National University, Lviv, Ukraine.  ²School of Engineering, University of Birmingham, Birmingham, UK.  ", size: SMALL, i: true }),
    run({ t: "Correspondence: nataliia.b.shakhovska@lpnu.ua  ·  ORCID: N.S. 0000-0002-6875-8534, I.I. 0000-0002-9761-0096.", size: SMALL, i: true }) ] }));

// ---- Abstract ----
children.push(new Paragraph({ spacing: { before: 120, after: 60 },
  children: [new TextRun({ text: "Abstract", font: FONT, size: HEAD2, bold: true })] }));
children.push(para([
  "Accuracy, calibration and conformal coverage measure how well a model performs, not the conditions under which its performance can be trusted. We show that these prediction-side diagnostics are blind to an important class of deployment failure. A data-poisoning attack on real propaganda channels drives model steerability to 88% while matched accuracy ",
  I("rises"),
  " from 0.73 to 0.89 and conformal coverage holds at its target; the only signal that tracks the capture is the drift of the model's attribution profile (",
  I("r"),
  " = 0.79). We prove that this blindness is unavoidable. A ",
  I("separation theorem"),
  " shows that a reliable model and a compromised one can be made identical to every prediction-side certificate, with coverage, accuracy and calibration equal to machine precision, yet differ arbitrarily in the fidelity of their explanations and in their behaviour under shift. Within the class of certification-time prediction-law functionals, detecting the failure provably requires reading the model's structure and not only its sampled behaviour, so explanation certification becomes a necessary condition for trust. We organise the two certificates through a ",
  I("competence envelope"),
  ": the region of operating conditions over which predictions and explanations can jointly be trusted. Over four instrumented axes of deployment stress (distribution drift, data scarcity, adversarial contamination and resource degradation) the certificates fail on different axes, conformal coverage under temporal drift and explanation stability under scarcity and model compression, so neither substitutes for the other. Computed from unlabelled inputs, the joint certificate anticipates deployment error up to six years ahead (Spearman \u03c1 = 0.90 on climate discourse) and, used as an abstention gate, cuts error by 28%. Across three multimodal systems spanning text, tabular, acoustic and visual signals, gating late fusion by each channel's certificate rather than its confidence recovers the single-best-modality oracle under sensor failure. The two-certificate structure holds across eleven datasets and eight model classes, from linear and tree-ensemble models to three multilingual transformers. Certifying the reasons a model gives is thus a measurable and actionable requirement for trustworthy deployment, from conflict-zone infrastructure to adversarial information ecosystems.",
]));

// ---- Introduction ----
children.push(h1("Introduction"));
children.push(para([
  "Across science, medicine and public life, consequential judgments are increasingly made or shaped by automated systems: which patient is deteriorating, which building is safe to re-enter after a disaster, whether an image or a claim is authentic. Under controlled evaluation these systems can be remarkably accurate, and that accuracy is real. But it is conditional. It rests on assumptions that are seldom stated because, in curated settings, they almost always hold: that data are plentiful and representative, that the world does not shift between calibration and use, that no one has tampered with the inputs, and that the machine runs on hardware that does not fail. Where the stakes are highest, these assumptions do not weaken one at a time; they break together. A diagnostic tool meets a population unlike any it was trained on; a damage-assessment system meets a disaster of a kind it has never seen, imaged by a sensor that is itself broken; an authenticity check meets a manipulation designed after it was built.",
]));
children.push(para([
  "The deepest danger is not that such systems are sometimes wrong. Any judgment, human or machine, is sometimes wrong. The danger is that they are wrong ", I("silently"),
  ". A system that has left the conditions under which it is competent does not announce the fact. It keeps issuing confident answers. And, in the part that has been most underweighted, the reasons it offers keep looking reasonable, because the cues a machine leans on need not be the cues a person would trust",
  cite("shortcut"),
  ". A convincing explanation then lends a wrong answer a false air of trustworthiness at the moment a human most needs to be warned to distrust it",
  cite("cummings"),
  ". In building systems that can explain themselves, we have given silent failure an articulate voice.",
]));
children.push(para([
  "This paper advances a single claim and traces its consequences. Our central result is a theorem. Monitoring what a model ", I("predicts"),
  " is, within the class of functionals of its prediction behaviour, provably insufficient to certify whether it can be trusted. The ", I("separation theorem"),
  " makes this exact: a reliable model and a compromised one can be made identical to every prediction-side certificate (conformal coverage, accuracy, calibration, confidence) while differing arbitrarily in the fidelity of their ", I("explanations"),
  " and in their behaviour under shift. A matching lower bound shows that no monitor built from a model's predictions alone detects this class of failure above chance, and that detecting it requires reading the model's structure. Certifying the reasons a model gives therefore becomes a necessary condition for trust, and the rest of this paper is evidence for that claim and its consequences. To organise the two certificates we introduce the ",
  I("competence envelope"),
  ", the region of operating conditions over which a system's judgments ", I("and the reasons it gives"),
  " are jointly trustworthy: a provably non-trivial region whose boundary is set, direction by direction, by whichever certificate fails first. The envelope can also be acted upon. A label-free monitor senses in real time when a system has left it and degrades gracefully, preferring a verifiable “I do not know here” to a confident, well-explained error. The empirical sections span adversarial information ecosystems, multimodal deployment, clinical data and post-conflict infrastructure, across eleven datasets and eight model classes from linear models to multilingual transformers. Together they show that the separation is real, general and consequential, and they map the measurable, actionable property we call ",
  I("machine competence"), ".",
]));

children.push(h2("The fragmentation of trustworthy AI"));
children.push(para([
  "Trustworthy machine learning today is five mature literatures that rarely speak to one another. Research on distribution shift and domain adaptation detects and corrects changes between training and deployment",
  cite("datashift", "wilds"),
  ", but treats explanation and abstention as out of scope. Work on small-data and imbalanced learning improves estimation under scarcity, but assumes a fixed distribution. Adversarial robustness certifies prediction stability against bounded perturbations",
  cite("smoothing"),
  ", yet rarely connects to natural drift or to interpretability. Uncertainty quantification, selective prediction and conformal inference deliver calibrated abstention and distribution-free guarantees",
  cite("vovk", "conformal", "selective"),
  " (though the confidence they calibrate is itself known to degrade under shift",
  cite("ovadia"),
  "), but they guard ", I("predictions"),
  ", not explanations. And explainable AI produces post-hoc attributions",
  cite("lime", "shaporig", "gradcam"),
  " whose own faithfulness is fragile and is essentially never certified as a function of operating conditions, a fragility that has prompted calls to prefer inherently interpretable models for the highest-stakes decisions",
  cite("rudin"), ".",
]));
children.push(para([
  "Each toolkit silently assumes the others’ problems away, and the result is the silent, well-explained failure above. The nearest existing concepts (out-of-distribution detection",
  cite("ood", "oodsurvey"),
  ", selective prediction and conformal inference, certified robustness, applicability domains and model cards, trust scores) each fall short on a specific axis; what none characterises, let alone certifies, is the ",
  I("joint"), " region of operating conditions over which both a prediction and its explanation may be trusted. That certifiable union is the gap.",
]));
children.push(para([
  "This is not a gap between neighbouring methods but a ", I("structural"),
  " one, and the separation theorem locates it precisely. Every framework in the trustworthy-machine-learning canon — statistical learnability and generalisation bounds, PAC-Bayes, calibration and proper scoring rules, selective prediction, split-conformal coverage, distribution-shift and domain-adaptation bounds, and certified adversarial robustness — is a ",
  I("prediction-side"),
  " guarantee in the exact sense of Theorem 1: each is a functional of the model’s input–output behaviour together with labels. By that theorem, no such guarantee, however tightened, can certify the class of failures in which a model’s explanation has drifted while its predictions are untouched; the entire canon lies on one side of the separation. Explainable AI sits on the other side but supplies no ",
  I("certificate"),
  ": it produces attributions",
  cite("lime", "shaporig", "gradcam"),
  " whose own faithfulness is fragile and is essentially never bounded as a function of operating conditions",
  cite("rudin"), ".",
]));
children.push(para([
  "What is missing from the literature is therefore not a better estimator but a ", I("necessary condition"),
  " that no existing theory states: that trust requires certifying the explanation, not only the prediction. Stacking existing methods — conformal coverage, a robustness ball, an attribution map — cannot supply it, because each ingredient is prediction-side or uncertified, and because the stressors interact, so a guarantee proved under drift alone need not survive drift with scarcity. Related work does not merely lack the joint object; by Theorem 1 it lies provably outside it. This is the primitive we introduce: explanation fidelity certified as a function of where a system operates, coupled to prediction reliability in one region with a provable boundary (Table 1).",
]));
children.push(para([
  "Recent flagship studies sharpen the gap. Label-light detectors of unreliable generation",
  cite("farquhar"),
  " and the finding that larger, more instructable models become less reliable and confidently wrong",
  cite("zhou_reliable"),
  " both operate on the prediction side; demonstrations that models decide on covert, spurious cues such as dialect",
  cite("hofmann"),
  " or a cost proxy standing in for health need",
  cite("obermeyer"),
  ", and the caution that explanations in medicine can mislead unless their faithfulness is itself assured",
  cite("babic"),
  ", are exactly the failures our explanation certificate targets. What remains missing across this literature is a certifiable operational boundary — a region in which both a model’s answer and the reason for that answer can be trusted; the competence envelope supplies that object.",
]));

// Table 1
children.push(new Paragraph({ spacing: { before: 120, after: 80 },
  children: [ run(B("Table 1 | ")), run("How the competence envelope relates to neighbouring guarantees.") ] }));
children.push(new Table({
  width: { size: 9360, type: WidthType.DXA }, columnWidths: CW,
  rows: [
    t1row([[B("Approach")], [B("What it certifies")], [B("What it misses")], [B("What the competence envelope adds")]], true),
    t1row([["OOD / anomaly detection"], ["that a single input is anomalous (binary verdict)"], ["how reliable the prediction is; the explanation; a region"], ["a graded, certified region and a fidelity-bounded explanation"]]),
    t1row([["Selective prediction / conformal inference"], ["prediction coverage or selective risk"], ["explanation fidelity; interacting, compound shift"], ["a joint prediction-and-explanation certificate under compound stress"]]),
    t1row([["Certified robustness"], ["prediction stability within a bounded perturbation ball"], ["natural drift; scarcity; the explanation"], ["natural compound shift plus a certified explanation"]]),
    t1row([["Applicability domain; model & robustness cards"], ["where a model is expected to work (descriptive)"], ["a guarantee; the explanation; boundary behaviour"], ["a certifiable boundary with a prescribed action at it"]]),
    t1row([["Learnability & generalisation bounds (PAC, PAC-Bayes)"], ["expected error of a predictor from finite samples"], ["explanation fidelity; operating-condition dependence"], ["a certified region for prediction and explanation jointly"]]),
    t1row([["Trust / confidence scores"], ["per-input confidence (a scalar)"], ["the explanation; a guaranteed action"], ["confidence coupled to a certified explanation and action policy"]]),
  ],
}));

// ---- Results ----
children.push(h1("Results"));

children.push(h2("A measurable space of operating conditions"));
children.push(para([
  "The reframing we adopt is deliberately economical. Treat drift, scarcity, contamination and resource degradation not as four problems but as four coordinates of one ",
  I("operating-condition space"),
  ", written Ω. Each coordinate is an estimable, monotone stress signal: drift from population-stability indices, Kolmogorov–Smirnov or maximum-mean-discrepancy statistics; scarcity from effective sample size, local density and label noise; contamination from out-of-distribution and adversarial-shift estimators; resources from latency, memory and energy budgets. A deployed system occupies a point in Ω that can be measured at any moment.",
]));
children.push(para([
  "For a deployed pair (a model ", I("f"), " and an explainer ", I("E"), "), we define the ",
  I("competence envelope"), " C(", I("f"), ",", I("E"),
  ") ⊆ Ω as the region of operating conditions in which two guarantees hold jointly: predictive reliability is bounded by a calibrated-error or coverage guarantee",
  cite("calibration"),
  ", ", I("and"),
  " explanation fidelity is bounded, so the explanation is a faithful, stable account of the model’s computation. Inside C the credential is granted; outside it, at least one guarantee provably fails. This recasts a vague engineering intuition (“the model works here, not there”) as a formal object with a measurable boundary and certifiable properties (Box 1).",
]));
children.push(para([
  "The object earns its keep only if it is more than a definition. Two hypotheses give it scientific content. The first is ",
  I("existence"),
  ": that for every (", I("f"), ",", I("E"),
  ") there is a maximal competence envelope, a largest region over which both guarantees can be jointly certified, such that crossing its boundary necessarily forfeits at least one of them. The second, and more ambitious, is ",
  I("contraction"),
  ": that under compound stress the envelope contracts as the intersection of certificates that fail on different axes. The first we settle affirmatively as a theorem in the next section (a non-trivial envelope provably exists for any (f, E)), so competence appears as a regularity in the learning systems and stress conditions tested, not a property of one model on one task, and the trustworthy/untrustworthy boundary behaves, in these settings, as a locatable transition rather than a gradual fade. How sharp that transition is, and how far a fitted contraction law transfers across domains, remain empirical questions our experiments address rather than assume.",
]));

children.push(h2("Why prediction certification cannot replace explanation certification"));
children.push(para([
  "The central claim of this work is a theorem, not a definition: monitoring what a model predicts is provably insufficient to certify whether it can be trusted, and monitoring how it decides is not an optional interpretability layer but a strictly more powerful test. We make this precise. Let a ",
  I("prediction-side certificate"),
  " be any functional of the joint law of (", I("X"), ", ", I("Y"), ", ", I("f"), "(", I("X"),
  ")) on the certification distribution — this class contains conformal coverage, accuracy, calibration error, Brier score, confidence and every other quantity computable from the model’s input–output behaviour on sampled data. Let the ",
  I("explanation certificate"),
  " instead read the model’s sensitivity map ", I("A"), S("f"), "(", I("x"), ") = ∇", S("x"), I("s"), S("f"),
  "(", I("x"), "), a structural functional of how the model computes, and set F = 1 − ", I("D"),
  "(φ", S("f"), ", φ", S("0"), ") for the global sensitivity profile φ. The distinction is exactly behavioural versus structural.",
]));
children.push(para([
  B("Theorem 1 (prediction–explanation separation). "),
  "For any target fidelity gap β ∈ (0,1) and any accuracy margin γ ∈ (0, ½), there exist two models ", I("f"), ", ", I("f"), "′, a certification distribution ", I("P"), " and a deployment distribution ", I("P"), "′ such that:  (i) ",
  I("C"), "(", I("f"), ") = ", I("C"), "(", I("f"), "′) for ", I("every"),
  " prediction-side certificate C;  (ii) F(", I("f"), ") = 1 but F(", I("f"), "′) ≤ 1 − β;  and (iii) the deployment accuracy of ", I("f"), "′ is below that of ", I("f"),
  " by at least γ. The reliable and the compromised model are therefore identical to every prediction-side certificate, yet separated by the explanation certificate and by their behaviour under shift.",
]));
children.push(para([
  I("Proof. "),
  "Construct a dormant coordinate ", I("v"), " that is identically zero on the support of ", I("P"), ". Let ", I("f"), " place zero weight on ", I("v"), " and let ", I("f"),
  "′ be identical except for weight ", I("W"), " on ", I("v"), ". (i) On supp(", I("P"), "), ", I("s"), S("f"), "′(", I("x"), ") = ", I("s"), S("f"), "(", I("x"), ") + ", I("W"), "·", I("v"), "(", I("x"), ") = ", I("s"), S("f"),
  "(", I("x"), "), so the two models induce the ", I("same"),
  " joint law of (X, Y, f(X)); every functional of that law agrees, exactly. (ii) The sensitivity profiles differ only in the ", I("v"),
  "-coordinate, by |", I("W"), "|; choosing ", I("W"), " makes ", I("D"),
  "(φ, φ′) ≥ β. (iii) Let ", I("P"), "′ move a fraction of mass to points with ", I("v"), " = 1; there ", I("s"), S("f"),
  "′ exceeds ", I("s"), S("f"), " by ", I("W"), ", flipping those predictions and depressing accuracy by γ. ∎",
]));
children.push(para([
  B("Corollary 1.1 (detection lower bound). "),
  "Any monitor that is a function of prediction-law samples alone has, on the family {", I("f"), ", ", I("f"),
  "′}, identical distributions under both models; its detection power therefore equals its false-positive rate. Within the class of certification-time prediction-law functionals, no test does better than chance, and detecting the compromise ", I("provably requires"),
  " reading a functional of the model’s structure — its sensitivity map — that is not determined by the prediction law. To be explicit about finite samples, we condition on a fixed certification procedure (a fixed nonconformity score, split protocol and random seed): the statement is that no functional of the resulting certification-time sample path of (X, Y, f(X)) distinguishes the pair, which covers split-conformal coverage even though it is not a pure population functional. The separation is thus not a claim that failure is undetectable in principle, but a precise statement of ",
  I("what information a detector must use"), ": behaviour on sampled predictions does not suffice; structural access does.",
]));
children.push(para([
  B("Scope of the theorem. "),
  "The result is deliberately bounded, and the boundary is where its content lies. A ", I("prediction-side"),
  " certificate is any functional of the certification-time joint law of (X, Y, f(X)): coverage, accuracy, calibration, Brier score, AUC, confidence and their post-hoc combinations. The theorem says none of these detects the construction; equally, any monitor that ", I("does"),
  " detect it must lie outside this class. Three familiar tools illustrate the line rather than crossing it. Gradient-based out-of-distribution detectors and influence-function monitors read internal gradients or training-data sensitivities: these are structural, on the explanation side of the separation, and their effectiveness is an ",
  I("instance"), " of Corollary 1.1, not a counterexample to it. Adversarial-input detectors that act at ", I("deployment"),
  " time observe P′, not the certification law P, and so fall outside the theorem’s premise, which concerns what can be certified ", I("before"),
  " the shift is seen. And feature-space or representation monitors that inspect the model’s internal activations are, again, structural. The theorem therefore does not assert that silent failure is undetectable; it asserts that detection cannot be purchased with prediction-side certificates alone, and it classifies exactly which monitors escape the bound: precisely those that read the model’s structure rather than its sampled behaviour. This has an operational consequence for black-box deployments. Where gradients, weights and activations are inaccessible — vendor APIs, closed foundation models, some regulated devices — pre-deployment certification against this failure family is, by Corollary 1.1, not achievable under the theorem’s information constraints; certification must then be moved to a setting with structural access, through vendor-side audits, weight escrow, secure enclaves, attested inference or third-party evaluation agreements.",
]));
children.push(para([
  B("Corollary 1.2. "),
  "The data-poisoning attack reported below (", B("Fig. 8"),
  ") is one realisation of this construction (a planted cue dormant on the clean support), so its invisibility to accuracy and coverage is not an empirical accident but a consequence of Theorem 1.",
]));
children.push(para([
  B("Theorem 2 (approximate separation for near-dormant directions). "),
  "The exact construction takes ", I("v"),
  " identically zero on the support of ", I("P"),
  ", which is natural for sparse text features or backdoor triggers but idealised for continuous or embedding domains. The separation degrades gracefully. If ", I("v"),
  " is ", I("ε-dormant"),
  " — carrying variance at most ε² under ", I("P"),
  " — then every prediction-side certificate evaluated on ", I("f"), " and ", I("f"),
  "′ differs by at most a quantity that vanishes with ε (bounded by an ε-divergence between the two prediction laws), while the explanation functional still differs by β and deployment behaviour under ", I("P"),
  "′ still diverges. Prediction-side certificates therefore remain arbitrarily close as the direction approaches dormancy, whereas the explanation certificate does not. We verify this (Methods): as ε grows from 0, the coverage gap stays below 0.005 up to ε = 0.1 and the accuracy gap grows only as O(ε), while the structural fidelity gap holds near 0.4 throughout. This is the realistic analogue of Theorem 1 for the continuous and transformer-embedding settings used later.",
]));
children.push(para([
  B("The explanation certificate as one structural functional. "),
  "Throughout, the object on the explanation side is a single ", I("structural explanation functional"),
  " A(", I("f"), "; P", S("ref"), ") — a functional of the model’s decision function that is ", I("not"),
  " determined by the prediction law on ", I("P"),
  ". The separation theorem is stated for its sensitivity-map instance ∇", S("x"), I("s"), S("f"),
  ", but A is deliberately abstract: across the experiments it is instantiated, according to what each model class exposes, by signed coefficients (linear models), effective input sensitivities (neural models), TreeSHAP or gain summaries (tree ensembles) and probe-weight profiles (frozen embeddings). These are instances of one structural-versus-behavioural divide, not different mathematical objects; the stability statistic ",
  I("S"), " reported empirically is the drift of A from its reference profile.",
]));
children.push(para([
  "Two supporting facts organise the object the certificates act on. ",
  B("Proposition 1 (existence and geometry). "),
  "Under conformal validity and continuity, the competence envelope K(α,β) = { ω : ", I("C"), "(ω) ≥ 1−α and F(ω) ≥ 1−β } is non-empty and contains a neighbourhood of the nominal condition; and if each certificate is monotone along rays of increasing stress, K is star-shaped with radial boundary ∂K(",
  I("u"), ") = min(", I("t"), S("C"), ", ", I("t"), S("F"),
  "), set direction-by-direction by whichever certificate fails first. Competence is therefore always a region, and its boundary is the pointwise minimum of the two certificates, but the content of the theory is Theorem 1: within the prediction-law class the two certificates are not merely both present, they are ",
  I("non-substitutable"), ": one cannot be recovered from the other.",
]));
children.push(para([
  "All statements are verified on controlled constructions (",
  B("Fig. 1"),
  "). For the separation theorem, the two models are ", I("bitwise"),
  " identical to every prediction-side certificate: maximum prediction difference 0, and conformal coverage, accuracy, calibration error and the whole conformal-score distribution equal to machine precision (Kolmogorov–Smirnov distance 0), while their explanation fidelity is driven from 1 to below 0.19 by the planted weight, and their deployment accuracy separates from 0.82 to 0.41 as the dormant coordinate is activated. Proposition 1 is confirmed on an analytic construction (envelope non-empty and origin-containing; zero star-shape violations across 121 cells; coverage binding first on the drift ray, fidelity first on the scarcity ray).",
]));
children.push(figimg("figures/fig_theorem.png", 624, 221, "Figure 1"));
children.push(figlegend([
  B("Fig. 1 | Prediction certification cannot replace explanation certification. "),
  "(a) The competence envelope K (green) is the intersection of the coverage (navy) and fidelity (teal) certificates; the amber region is the silent-failure set where coverage is certified but fidelity has failed, the blue region the converse. (b) The envelope boundary is the pointwise minimum of the two certificates, each binding on its own stress cone (Proposition 1). (c) The separation theorem, verified: a reliable model ",
  I("f"), " and a compromised model ", I("f"),
  "′ are identical to every prediction-side certificate (coverage, accuracy, calibration — all Δ = 0) yet separated by the explanation certificate and by deployment accuracy (Theorem 1, Corollary 1.1).",
]));

children.push(h2("Certifying explanations, not only predictions"));
children.push(para([
  "The hard, original core of this programme is to certify the ", I("explanation"),
  ". Prediction-side certification has a mature toolkit; explanation-side certification barely exists, because the faithfulness of an attribution is itself unstable and rarely guaranteed",
  cite("alvarez", "yeh"),
  ". Two functionals make fidelity measurable. A ", I("faithfulness"),
  " functional asks how well an attribution reflects what the model actually computed, operationalised through perturbation-based infidelity and deletion/insertion agreement",
  cite("rise"),
  ", and, for additive models and tree ensembles, through the exact Shapley efficiency identity, where attributions provably sum to the model’s output gap and TreeSHAP yields faithfulness in closed form",
  cite("treeshap"),
  ". A ", I("stability"),
  " functional asks how much the explanation changes under small changes to the input or the conditions, expressed as a local Lipschitz estimate of the explanation map.",
]));
children.push(para([
  "The certificate is then obtained by ", I("conformalising"),
  " these two functionals over Ω: the explanation-competence region is the set of operating points where a certified lower bound on faithfulness stays above a threshold and a certified upper bound on instability stays below a limit, valid distribution-free under exchangeability and extendable to bounded covariate shift by weighted conformal methods",
  cite("covshift"),
  ". Prediction-side reliability is certified jointly, so a single credential covers the prediction ", I("and"),
  " its explanation. These guarantees are distribution-free under exchangeability and bounded covariate shift; far outside the envelope, where the shift is unbounded, no such certificate can hold and the system resolves to a conservative abstention rather than a coverage guarantee, which is precisely the boundary behaviour the competence signal is built to trigger.",
]));
children.push(para([
  "Stability is the delicate part, because exact global Lipschitz constants of large networks are intractable. The design choice that makes the programme robust is to ",
  I("decouple the certificate from any closed-form constant"),
  " through a three-tier strategy: for tractable classes (additive models and tree ensembles), stability is piecewise closed-form; for deep explanations, randomised smoothing of the explanation map yields an analytic Lipschitz bound without a network constant; and for the small distilled surrogates used as fallbacks, semidefinite or spectral-norm methods give certified constants because those models are deliberately compact",
  cite("lipsdp", "clever"),
  ". The primary route, however, is distribution-free: estimate stability as the upper tail of the explanation’s difference quotient over sampled neighbours, then conformalise it to an upper confidence bound, needing no closed-form constant and giving a usable certificate even where the analytic theory resists, with the closed-form tiers only tightening it. Where input-space gradients are themselves unstable, fidelity can instead be grounded externally, the explanation mapped onto expert-meaningful features and certified by its agreement with expert annotations, as demonstrated for deep clinical signal and image models",
  cite("barmak"),
  ". The distribution-free route is realised on a deep network below: smoothing the explanation map and conformalising its difference quotient gives a stability certificate, with no network constant, that degrades under stress (Fig. 3c).",
]));

// Box 1
children.push(box([{ t: "Box 1 │ The competence envelope" }], [
  boxPara([ "Let Ω be the ", I("operating-condition space"),
    " spanned by four estimable, monotone stress signals: distribution drift, data scarcity, adversarial contamination and resource degradation. For a deployed pair (model ",
    I("f"), ", explainer ", I("E"), "), let R(ω) denote predictive reliability and F(ω) explanation fidelity at operating point ω ∈ Ω. The ",
    I("competence envelope"),
    " is the sublevel set C(f,E) = { ω ∈ Ω : R(ω) ≥ r₀  and  F(ω) ≥ f₀ } for acceptable reliability and fidelity thresholds r₀, f₀." ]),
  boxPara([ B("H1 (Existence and regularity). "),
    "For every (f,E) the jointly-certified set is non-trivial and, because the Ω axes are monotone stress signals, connected with a monotone boundary, so a single learnable frontier C*(f,E) separates trustworthy from untrustworthy operation. ",
    I("Falsified if"), " reliable, well-explained operation persists in disconnected pockets of Ω that no monotone boundary separates." ]),
  boxPara([ B("H2 (Contraction). "),
    "Under compound stress the envelope contracts according to non-linear stressor interactions captured by a fittable contraction law; whether its interaction coefficients are invariant across domains is a separate, stronger claim. ",
    I("Falsified if"), " measured contraction is captured by treating stressors as independent (no interaction term improves the fit), or if a contraction law fitted in one domain has no predictive power in another." ]),
  boxPara([ "Fidelity is made measurable by a ", I("faithfulness"),
    " functional (how well an attribution reflects the model’s computation; exact via the Shapley efficiency identity for additive and tree-ensemble models) and a ",
    I("stability"),
    " functional (a local Lipschitz estimate of the explanation map). Conformalising them over Ω yields a distribution-free certificate at confidence 1−α: the certified explanation-competence region is { ω ∈ Ω : Fα(ω) ≥ τ_F, Sα(ω) ≤ L }, with weighted-conformal extension under bounded covariate shift." ]),
]));

children.push(h2("Compound stress and the contraction of competence"));
children.push(para([
  "Real crisis settings impose stressors ", I("together"),
  ". A model is starved of labels ", I("and"), " meeting a shifted population ", I("and"),
  " running on degraded hardware, all at once. The interactions are understudied for a mundane reason: no benchmark exposes them, because benchmarks are organised by task, not by the structure of the stress. Making compound stress the object of study (and characterising how envelopes contract as stressors interact) is what separates this programme from a repackaging of familiar tools.",
]));
children.push(para([
  "The tractable starting point is pairwise: how the envelope contracts under drift with scarcity, drift with contamination, drift with resource degradation. The conjecture is that a few dominant interaction terms govern most of the contraction and take a common parametric form across domains. If that holds even approximately, certification becomes practical: one measures a system’s exposure along each axis and reads off how far its competence has shrunk, rather than re-validating from scratch for every new combination of insults.",
]));
children.push(eq("g(σ,ρ) = g₀ − aσ − bρ − c·σρ,    envelope  C = { g ≥ 0 }"));
children.push(para([
  "Here g is the certified competence margin under two normalised stress coordinates σ and ρ, and the interaction coefficient c distinguishes stressor pairs that compound (c > 0, the envelope shrinks faster than either axis alone predicts) from pairs that act independently (c ≈ 0). Estimating these coefficients, and testing whether they transfer across domains, is the empirical content of H2.",
]));

children.push(h2("An empirical competence envelope"));
children.push(para([
  "To test whether the competence envelope is measurable rather than metaphorical, we ran a controlled, reproducible study on two real, independent corpora drawn from the application domains this programme targets: the online information ecosystem and climate discourse. Domain A (information integrity) is 30,066 messages from eleven Telegram war-reporting channels over January–June 2026, with the task of predicting which messages are heavily amplified (top versus bottom tercile of forward rate) from their text alone. Domain B (climate adaptation) is 36,642 Reddit posts about climate topics over 2017–2023, with the analogous task of predicting high versus low engagement. Both use the tractable, intrinsically interpretable model class the theory privileges: a character-level TF-IDF text representation with a linear (logistic) classifier, whose exact attributions are its linear Shapley values. Two axes of Ω were instrumented: temporal drift (training on an anchor period and evaluating on progressively later ones, the drift magnitude δ measured as the discriminability of each later period from the anchor) and data scarcity (training-set size n). At each operating point we measured predictive reliability as split-conformal coverage",
  cite("conformal"),
  " at a 0.90 target, and explanation stability as the drift of the global attribution profile away from its in-distribution reference (a global proxy for the local stability functional of Box 1). For this model class faithfulness is exact: the Shapley efficiency identity (attributions sum to the model’s output) held to 2×10⁻¹⁵, so the binding explanation-side constraint is stability, exactly as the theory anticipates.",
]));
children.push(para([
  "Three results emerge in Domain A (Fig. 2). First, the competence envelope is a ", I("region"),
  ", not a number: the jointly-certified operating points (coverage ≥ 0.88 and stability within its limit) form a contiguous low-drift, low-scarcity block that contracts as either stressor rises. Second, the two certificates are ",
  I("independently necessary"),
  ". Conformal coverage degrades along the drift axis — at fixed n = 2,400 it falls from 0.93 in January to 0.82 by May as the war-news distribution shifts (δ rising from 0.20 to 0.75), but is almost insensitive to scarcity. Explanation stability degrades along the scarcity axis: the attribution-profile distance rises from 0.13 at n = 4,000 to 0.33 at n = 300 — but is almost insensitive to drift. A prediction-only certificate would therefore pass a scarce-data model whose explanation has already drifted, and an explanation-only check would pass a drifted model whose coverage has already collapsed, each blind to the other’s failure. The joint certificate catches both: this is precisely the certifiable union that existing tools omit.",
]));
children.push(figimg("figures/fig_real1.png", 624, 279, "Figure 2"));
children.push(figlegend([
  B("Fig. 2 | A competence envelope on real information-ecosystem data. "),
  "A TF-IDF + logistic classifier predicting message amplification on 30,066 messages from eleven Telegram channels (2026), gridded over temporal drift × scarcity (means over six seeds). (a) Each cell reports conformal coverage (top) and explanation-stability drift (S); the certified envelope (green outline) is where coverage ≥ 0.88 and S ≤ 0.20. Reliability fails along the drift axis (δ, the discriminability of each month from the January anchor) and explanation stability along the scarcity axis, so a single certificate misses one failure mode each. (b) At fixed n = 2,400 the reliability certificate is withheld from February (coverage 0.89 < 0.90 target) before the confident-but-wrong rate climbs from 2% to 9% by May. All values are measured and reproducible.",
]));
children.push(para([
  "Third, the certificate gives ", I("early warning"),
  " of silent failure, on both axes. On the scarcity axis the explanation-stability certificate is withheld once data become scarce (n = 300, profile distance 0.33) while conformal coverage is still 0.91 and accuracy 0.72, while the explanation side flags degradation the prediction side does not yet reveal. On the drift axis the reliability certificate is withheld from February (coverage 0.89), just as confident errors begin to accelerate; abstaining there keeps the system out of the May regime in which the confident-but-wrong rate has quadrupled to 9% (Fig. 2b). The study exercises three of the four Ω axes (drift, scarcity and contamination; resource degradation is instrumented separately in Fig. 7) and makes no claim of universality; its purpose is to show that the envelope, the joint certificate and the early-warning behaviour are measurable and reproducible on real, deployment-scale data. The same signature appears in other deployed systems: synthetic-media detectors degrade sharply on unseen generators, a cross-generator generalisation gap",
  cite("ojha"),
  ", and an explainable knee-MRI model’s reliability swings across imaging planes (AUC 0.72–0.96) under one explainer",
  cite("shakhovskaxai"),
  ", operating-condition sensitivity of exactly the kind the envelope formalises.",
]));
children.push(para([
  "A second, independent domain turns the framework’s generality and its contraction law from conjecture into measurement. On the Reddit climate corpus the entire structure replicates (Fig. 3a): coverage again degrades with temporal drift (from 0.94 in 2019 to 0.85 by 2023 at n = 2,400) and explanation stability again with scarcity (0.12 to 0.32), in a domain with a different language, platform and topic. Crossing scarcity with contamination (random label corruption ρ) then probes the contraction’s interaction structure (Fig. 3b,c). Here the strongest form of H2 fails: prediction error rises under joint stress, but the scarcity-by-contamination interaction is statistically near-additive in both text domains (Telegram c = −0.03, 95% CI −0.07 to 0.01; Reddit c = −0.03, −0.07 to −0.00) and in a clinical tabular contrast (Wisconsin Diagnostic Breast Cancer with gradient-boosted trees",
  cite("wdbc", "treeshap"),
  ", c = +0.03, −0.10 to 0.13) — no domain shows a bootstrap-significant super-additive error law. The envelope nonetheless contracts sharply under compound stress, because it is the ", I("intersection"),
  " of two certificates that fail on different axes: a second stressor that disables the second certificate collapses the jointly-certified region even when its effect on prediction error alone is merely additive. Competence contraction is therefore real and certifiable across model classes and domains, but it is a property of the joint certificate’s geometry rather than of a universal super-additive law; measuring the per-domain coefficients is what an open benchmark is for.",
]));
children.push(figimg("figures/fig_real2.png", 624, 238, "Figure 3"));
children.push(figlegend([
  B("Fig. 3 | Cross-domain replication and compound stress. "),
  "(a) The two-certificate structure replicates on 36,642 Reddit climate posts (2017–2023): conformal coverage degrades with temporal drift (evaluation year) and explanation stability with scarcity (n), the certified envelope sitting in the low-stress corner (green). (b) On the Telegram corpus, prediction error rises monotonically under joint scarcity × label-contamination (ρ) stress. (c) The scarcity-by-contamination interaction coefficient c (error model g = e₀ + aσ + bρ + c·σρ; bootstrap 95% CIs) is statistically indistinguishable from zero in both text domains and in a clinical tabular contrast, compound stress is near-additive on prediction error, with no universal super-additive law. All values are measured and reproducible.",
]));


children.push(h2("The envelope is architecture-independent"));
children.push(para([
  "If competence is a property of learning systems under stress rather than of one estimator, the envelope should appear across model classes, not only in the interpretable linear model used above. We therefore instrument the same certificates on five architectures spanning the practical spectrum — L2-regularised logistic regression, a multilayer perceptron, gradient-boosted decision trees, XGBoost and LightGBM — with attribution profiles read per class (signed input sensitivity for the linear and neural models; gain-based feature importances for the three tree ensembles). Across all five, the same two-part signature holds (",
  B("Fig. 4"),
  "). On the Telegram corpus, conformal coverage falls with temporal drift for every architecture (from 0.92–0.95 in January to 0.80–0.83 by May at fixed n = 2,400), and explanation-stability drift rises as data become scarce for every architecture (S climbing from below 0.15 at n = 4,000 to 0.15–0.26 at n = 300). On the clinical tabular task the same coverage collapse under covariate drift appears across the gradient-boosted models, with the strongest boosting methods (XGBoost, LightGBM) falling furthest outside their certified region on the most out-of-distribution band (coverage 0.64–0.73). The competence envelope is thus not an artefact of a particular estimator: its boundary and its two-certificate structure are recovered, with the same directionality, by neural and modern boosted-tree models alike. To confirm the stability trend is not an artefact of using different explanation primitives across classes (coefficients for linear, gain for trees), we re-measure it under a ",
  I("single common explanation family"),
  " — mean |SHAP| profiles on a fixed reference set — and the scarcity-driven destabilisation persists for every architecture (SHAP-based S from n = 4,000 to n = 300 of 0.14 for logistic regression, 0.28 for the MLP, and 0.32–0.33 for the three tree ensembles), so the effect is a property of the models under scarcity rather than of any one attribution method (Supplementary Table S19).",
]));
children.push(figimg("figures/fig_multiarch.png", 624, 229, "Figure 4"));
children.push(figlegend([
  B("Fig. 4 | The competence envelope is architecture-independent. "),
  "The same certificates, instrumented on five model classes (logistic regression, MLP, GBDT, XGBoost, LightGBM). (a) On Telegram, conformal coverage degrades under temporal drift for every architecture (n = 2,400). (b) Explanation-stability drift S rises as training data become scarce for every architecture (January anchor). (c) On the clinical tabular task, coverage degrades under increasing covariate drift across the gradient-boosted models. The two-certificate signature and its directionality are recovered by neural and boosted-tree models alike, not only by the interpretable linear model. All values are measured over multiple seeds.",
]));

children.push(h2("Foundation-model backbones"));
children.push(para([
  "Linear and tree models make the certificates transparent, but the reviewer’s question is whether the envelope is an artefact of shallow features. It is not. We replace the character-TF-IDF representation with frozen mean-pooled embeddings from three multilingual foundation models — DistilBERT-multilingual, XLM-R and a multilingual MiniLM sentence encoder — and re-run the identical envelope protocol on both real corpora, certifying a logistic probe on top of each transformer’s representation (",
  B("Fig. 5"),
  "). The two-certificate signature survives the change of representation intact. On Telegram, conformal coverage falls monotonically with temporal drift for all three backbones (0.92–0.93 in January to 0.85–0.86 by May at n = 2,400), and explanation-stability drift falls with abundance for all three (for XLM-R, from 0.28 at n = 300 to 0.06 at n = 4,000). The same coverage degradation under drift replicates on Reddit across all three backbones. The envelope is therefore observed on both foundation-model embeddings and interpretable features under stress in the tested settings; it is not an artefact of the representation used to expose it in these cases.",
]));
children.push(figimg("figures/fig_transformer.png", 624, 223, "Figure 5"));
children.push(figlegend([
  B("Fig. 5 | The competence envelope holds on foundation-model representations. "),
  "The identical protocol run on frozen mean-pooled embeddings from three multilingual transformers (DistilBERT-multilingual, XLM-R, multilingual MiniLM), certifying a probe on each. (a) Telegram conformal coverage degrades under temporal drift for every backbone; (b) explanation-stability drift falls with training size for every backbone; (c) the coverage degradation replicates on Reddit. The two-certificate signature is representation-independent, not a TF-IDF artefact.",
]));

children.push(h2("Cross-dataset generality"));
children.push(para([
  "Finally we test breadth across tasks. Using the DistilBERT-multilingual embeddings we instrument the envelope on eight standard text-classification benchmarks spanning news (AG News, DBpedia), product and film reviews (Amazon, Yelp, IMDB, Rotten Tomatoes, SST-2) and social media (TweetEval), with covariate drift induced by evaluating on principal-component bands increasingly distant from the training core. The explanation-stability certificate degrades under scarcity on ",
  I("all eight"),
  " datasets (S falling monotonically as training data grow, from ≈0.37 at n = 200 to ≈0.27 at n = 1,500 on every set). The prediction certificate degrades under the induced covariate shift on the datasets where that shift is substantial: coverage drops of 0.45 on DBpedia, 0.24 on Amazon, 0.13 on IMDB and 0.09 on SST-2, and holds, correctly, on the homogeneous sentiment sets where principal-component banding induces little genuine shift (Supplementary Table S16). This is the honest and expected pattern: the scarcity certificate is universal across tasks, while the drift certificate fires in proportion to the covariate shift actually present. Together with the two real corpora, the clinical benchmark and the multimodal systems, the competence envelope has been measured across eleven datasets and eight model classes spanning linear, tree-ensemble and transformer families in this study.",
]));

children.push(h2("A label-free competence monitor anticipates silent failure"));
children.push(para([
  "For the envelope to be useful at deployment it must be computable ", I("without"),
  " the labels that are unavailable once a model is live. All three operating-condition signals satisfy this: input drift δ is read from an unlabelled discriminator, explanation stability from the model’s attribution profile on incoming data, and predictive uncertainty from the conformal non-conformity of new inputs. Treating their combination as a label-free competence monitor, we find it anticipates silent failure on the real temporal drift of both corpora. As the Telegram model is carried from its January anchor across the following months, its true error climbs from 0.13 to 0.35 and the Reddit model’s from 0.15 (2019) to 0.35 (2023); the monitor, computed without any test labels, tracks this rise (Spearman ρ = 0.60 for Telegram, 0.90 for Reddit; Fig. 6b). A deployer watching the monitor would therefore see competence draining away before any labelled outcome confirmed it.",
]));
children.push(para([
  "This makes the certificate ", I("actionable"),
  ". Used as a gate that abstains on the inputs it cannot certify, it converts silent error into safe deferral: answering only the most certifiable 70% of inputs cuts deployment error from 0.29 to 0.23 on Telegram and from 0.32 to 0.27 on Reddit, and answering 50% cuts it to 0.19 and 0.23 respectively (Fig. 6a). Crucially, the two certificates are ",
  I("complementary, not redundant"),
  ". Their certified margins are near-uncorrelated across operating points (Telegram −0.20, Reddit −0.02), and in predicting a model’s true error the joint certificate beats either side alone, most clearly in the climate domain, where adding the explanation side lifts the explained variance from 0.57 to 0.67 (Fig. 6c). The prediction side dominates under drift and the explanation side under scarcity, so only their union tracks competence across the whole operating-condition space, the same independence seen in Fig. 2, now shown to carry decision-relevant information a single certificate would miss.",
]));
children.push(figimg("figures/fig_real3.png", 624, 224, "Figure 6"));
children.push(figlegend([
  B("Fig. 6 | A label-free competence monitor anticipates silent failure and enables safe abstention. "),
  "(a) Risk–coverage curves: abstaining on the least-certifiable inputs reduces error on the answered inputs, on both corpora, under real temporal drift. (b) The monitor, computed without test labels, tracks each deployment period’s true error (Spearman ρ = 0.60 Telegram, 0.90 Reddit). (c) Predicting true error from the certificate signals: the joint certificate matches or beats the prediction-side and explanation-side alone, and the two margins are near-uncorrelated, so they capture complementary failure modes. All values are measured and reproducible.",
]));
children.push(h2("Resource degradation: the fourth independent axis"));
children.push(para([
  "The three axes exercised so far (drift, scarcity and contamination) all concern the data environment. The fourth axis of Ω, resource degradation, concerns the compute environment: in a crisis zone the model runs on damaged, battery-powered or bandwidth-limited edge hardware, so inference operates under a compressed model, a shrunk calibration buffer and a reduced sensor bandwidth. We instrument all three on our real corpora and ask whether resource degradation is a genuinely independent stressor or merely a restatement of scarcity.",
]));
children.push(para([
  "Model compression (Fig. 7a) gives the clearest answer. Pruning a fraction p of the smallest-magnitude classifier weights is a clean proxy for compute-budget reduction. At n = 2,400 training examples and moderate drift (evaluation on March data), pruning p = 0.70 drives explanation-stability drift to S = 0.26, exceeding the certificate's limit of 0.20 — while conformal coverage remains at 0.88, still inside its target. Coverage only falls below its threshold at heavier pruning (p ≥ 0.85). The pattern is the same as for the other axes: the explanation certificate fires before the prediction certificate. Resource degradation is therefore a genuinely fourth independent stressor: it degrades the decision function in a way the data environment does not, and the explanation side detects it first.",
]));
children.push(para([
  "The memory constraint on the calibration buffer (Fig. 7b) adds a second resource-side mechanism. Conformal coverage requires maintaining a buffer of calibration scores; when that buffer is small (cal_n < 100, as when on-device memory is limited), coverage becomes unreliable regardless of how much training data was available, a mechanism structurally distinct from training-data scarcity. The interaction with model compression is additive: a pruned model (p = 0.85) with a tiny calibration buffer (cal_n = 20) pushes coverage to 0.84, the lowest observed cell, while S is already at 0.27. Finally, reducing acoustic-visual channel bandwidth on CMU-MOSI (Fig. 7c) raises deployment error from 0.43 at full bandwidth to 0.51 at 15%, a monotone degradation that mirrors the feature-drift axis and can be detected by the same label-free discriminator without test labels. Across all three resource mechanisms, the envelope's response is qualitatively identical to its response on the other three axes: certificates fail at the boundary, the explanation side leads, and gating on the certificate converts resource-induced failure into safe deferral rather than a silent, confident wrong answer.",
]));
children.push(figimg("figures/fig_resdeg.png", 624, 246, "Figure 7"));
children.push(figlegend([
  B("Fig. 7 | Resource degradation is the fourth independent axis of the competence envelope. "),
  "(a) Under model compression (pruning fraction p), explanation-stability drift S exceeds its limit at p = 0.70 while conformal coverage is still within target (n = 2,400, evaluation on March Telegram data); crosses mark where S first exceeds the limit. The explanation certificate leads the prediction certificate, as on the other three axes. (b) A shrinking calibration buffer (memory constraint) destabilises conformal coverage independently of training-data scarcity; the effect compounds with model compression. (c) Reducing acoustic-visual channel bandwidth on CMU-MOSI raises deployment error monotonically. All values are measured and reproducible.",
]));
children.push(para([
  "The results so far concern benign stress. Information ecosystems, however, face ", I("adaptive"),
  " adversaries, and here the two certificates diverge most sharply. We staged a data-poisoning attack on the Telegram corpus: an attacker plants a rare lexical cue in a fraction ρ of amplified-class training messages, so the model learns the cue as a shortcut, then appends the same cue to arbitrary content at deployment to have it read as high-amplification. As ρ rises from 0 to 0.8, the model becomes almost fully steerable: the attacker’s content is misread as amplified in 88% of cases, up from a 28% base rate. Every prediction-side signal stays green while this happens: split-conformal coverage holds at its 0.90 target throughout, and matched-validation accuracy does not merely persist but ",
  I("improves"),
  " (0.73 → 0.89), because the planted shortcut makes the poisoned task easier. A deployer watching accuracy and calibration would conclude the system was getting better at the very moment it was being captured.",
]));
children.push(para([
  "The explanation certificate is the one signal that tracks the capture. As the model’s decision mass migrates onto the planted cue, its global attribution profile drifts steadily away from the certified reference (S rising 0.10 → 0.18), correlating with attack success at r = 0.79 across attack strengths, while coverage is uninformative (r = −0.47) and accuracy is actively misleading . The mechanism is structural, not incidental: a shortcut attack is precisely a change in ",
  I("how"),
  " the model decides that leaves ", I("how well"),
  " it appears to decide untouched, so it is invisible to any certificate defined on predictive performance and visible to any certificate defined on the decision function itself. Explanation-side certification is thus not merely an interpretability nicety but may be critical for security in models deployed in adversarial information environments.",
]));
children.push(para([
  "Two controls confirm the attack is a realisation of Theorem 1 rather than an artefact of the evaluation. First, when the model is ",
  I("certified on clean anchor data and never recalibrated"),
  " — the certification-on-P premise of the theorem — deploying on attacked inputs drives attack success from 0.24 to 0.89 as ρ rises to 0.8, while conformal coverage computed at certification time never leaves its target band (0.90 → 0.89) and cannot foresee the capture. Second, the planted cue is genuinely dormant on the clean distribution (its prevalence in the clean corpus is zero), and the explanation drift is localised on it, not a generic reaction to distributional change: as ρ grows, the cue’s share of the structural attribution mass rises monotonically and it comes to dominate the most-changed features (half of the twenty largest attribution increases are cue features). The explanation certificate is therefore reading the migration of decision mass onto the cue, exactly the structural signature the theorem predicts.",
]));
children.push(figimg("figures/fig_adversarial.png", 624, 232, "Figure 8"));
children.push(figlegend([
  B("Fig. 8 | A silent data-poisoning attack that prediction-side certificates cannot see. "),
  "An attacker plants a rare cue in a fraction ρ of amplified-class Telegram training messages and appends it to arbitrary content at deployment. (a) Attack success rises to 0.88 while conformal coverage holds at 0.90 and matched-validation accuracy improves, yet both prediction-side signals are blind or misleading. (b) The explanation-stability certificate tracks the capture (r = 0.79 with attack success): the attribution profile drifts as decision mass migrates onto the planted cue. Means over seeds; all values measured.",
]));
children.push(h2("Competence-gated fusion makes multimodal systems fail-safe"));
children.push(para([
  "Real deployed systems are increasingly multimodal — language plus acoustics plus vision, text plus sensor streams — and their characteristic failure is ",
  I("modality-specific"),
  ": one input channel degrades (a sensor fails, a subtitle track is forged, a data feed shifts) while the others remain sound. Because each modality then sits at a different point of Ω, the competence envelope makes a concrete architectural prediction: fusion should be gated by each modality’s ",
  I("certificate"),
  ", not by its confidence. We tested this on three multimodal systems from different domains and sensor types: Reddit climate posts (text + author/community tabular data), a Kyiv wartime news channel (text + behavioural signals; 81,000 messages with emoji reactions, 2021–2026, labelled distress versus support), and the CMU-MOSI audiovisual sentiment benchmark (spoken language + acoustic-visual channels). In each, per-modality classifiers are fused three ways: naïve averaging, confidence-gated weighting, and competence-gated weighting, in which each modality is weighted by its label-free certificate — its excess distribution drift beyond the certified state, combined with its conformal confidence.",
]));
children.push(para([
  "Under modality-specific failure the ordering is consistent and the margin large (Fig. 9). When the text channel fails on Reddit, competence-gated fusion holds error at 0.31 against 0.45 for both naïve and confidence-gated fusion; when the tabular channel fails, 0.24 against 0.33; on the Kyiv corpus, 0.23 against 0.29; and on CMU-MOSI, when the language channel fails, competence gating routes decisions to the intact acoustic-visual channel and lands exactly on the single-best-modality oracle (0.431 versus 0.511 for both baselines). In clean conditions it costs nothing. The failure of the strongest baseline is instructive. ",
  I("confidence-gating is fooled everywhere"),
  ", matching naïve fusion to three decimal places, because a degraded modality does not become diffident; it becomes confidently wrong, and confidence weighting keeps listening to it. Only the label-free certificate, which measures where each channel sits relative to its certified operating region rather than how sure it sounds, identifies the failed modality. One honest boundary: when the sole competent channel is itself the one that fails (text on the Kyiv corpus, whose behavioural backup is weak and drifting), no gating scheme rescues the system: the envelope then correctly prescribes abstention rather than fusion. Competence-gating, in other words, is not a trick for extracting accuracy; it is the routing layer that makes multimodal systems degrade the way safety-critical systems should: onto their still-competent channels when those exist, and to a refusal when they do not.",
]));
children.push(figimg("figures/fig_multimodal.png", 624, 235, "Figure 9"));
children.push(figlegend([
  B("Fig. 9 | Competence-gated late fusion under modality-specific failure, across three multimodal systems. "),
  "Deployment error of naïve, confidence-gated and competence-gated late fusion under clean conditions and under degradation of each channel, for (a) Reddit climate (text + tabular), (b) a Kyiv wartime news channel (text + behavioural) and (c) CMU-MOSI (spoken language + acoustic-visual). Green bars mark the single-best-modality oracle. Competence gating tracks the oracle under failure and is free in clean conditions; confidence gating is indistinguishable from naïve fusion because degraded channels remain confidently wrong. Means over seeds; all values measured.",
]));
children.push(h1("Discussion"));

children.push(h2("Systems that know their limits"));
children.push(para([
  "A theory of competence is only half the agenda; the other half is to act on it. The bridge is an online ",
  I("competence signal"),
  " that estimates, at inference time, where a system currently sits in Ω relative to its envelope. Crucially, the signal must be a ",
  I("leading"),
  " indicator that flags imminent departure before reliability actually falls, not a lagging post-hoc alarm. This builds directly on real-time, sliding-window anomaly detection in correlated sensor streams under tight latency and memory limits",
  cite("kalman"),
  ". It also responds to a broader lesson: deployment, not benchmark accuracy, is where machine-learning systems fail",
  cite("paleyes", "liudeploy"),
  ", a lesson that has motivated maturity scales such as technology-readiness levels for machine learning",
  cite("lavin"),
  " and leading rather than lagging risk indicators of the kind used in integrated early-warning systems",
  cite("reichstein"), ".",
]));
children.push(para([
  "The signal drives three coupled behaviours. It ", I("modulates the explanation"),
  ": the explanation simplifies as competence drops, from full attribution inside the envelope to an abstention rationale near its boundary, reporting not only ",
  I("why"), " a prediction was made but ", I("how far"),
  " conditions sit from where the model is competent. It ", I("gates the prediction"),
  ": predict, abstain, or defer to a human, with selective-risk guarantees from the theory and thresholds set by the cost of silent failure. And it ",
  I("descends a graceful-degradation hierarchy"),
  ": full deep model, compact distilled model",
  cite("distill"),
  ", interpretable surrogate, rule-based default, with the signal selecting the highest-complexity model still inside its envelope. The result prefers a verifiable “I do not know here” to a confident, well-explained, silent error.",
]));

// Box 2
children.push(box([{ t: "Box 2 │ A resilience certificate" }], [
  boxPara([ "For a deployed pair (f,E) at operating point ω, a certification harness emits, at confidence 1−α," ]),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 },
    children: [run({ t: "Cert(f,E,ω) = ⟨ mem(ω), covα(ω), cal(ω), Fα(ω), Sα(ω), π(ω) ⟩", i: true, size: SMALL + 1 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 100 },
    children: [run({ t: "valid  ⇔  covα(ω) ≥ 1−α  ∧  Fα(ω) ≥ τ_F  ∧  Sα(ω) ≤ L", i: true, size: SMALL + 1 })] }),
  boxPara([ "where ", B("mem"),
    " is certified envelope membership (inside / boundary / outside), ", B("covα"),
    " certified predictive coverage (selective risk), ", B("cal"),
    " competence-signal calibration, ", B("Fα"), " and ", B("Sα"),
    " the certified fidelity and stability bounds, and ", B("π"),
    " the prescribed action (predict / abstain / defer / fall back). When the validity condition fails, the certificate declares ω outside the envelope and prescribes a non-predicting action. Guarantees are distribution-free under exchangeability, and the three component tests combine at joint confidence 1−α by a union bound that splits α = α₁ + α₂ + α₃ across coverage, fidelity and stability so that all three hold at once. The format is designed to map onto risk-management, robustness and post-market-monitoring requirements for high-risk AI, giving engineers, auditors and funders an accountable, machine-readable statement of when an automated assessment may be trusted." ]),
]));

children.push(h2("Infrastructure resilience and reconstruction: the proving ground"));
children.push(para([
  "Post-crisis infrastructure resilience and reconstruction is, in effect, the most demanding test of machine competence: every axis of Ω is pushed to its extreme at once, the cost of a silent error is measured in lives and scarce capital, and the decisions are auditable and contested. It is where the theory is most needed and most severely tested.",
]));
children.push(para([
  "The scale alone reframes the problem. The fifth Rapid Damage and Needs Assessment for Ukraine, released in February 2026 by the Government of Ukraine with the World Bank Group, the European Commission and the United Nations, put reconstruction and recovery at almost US$588 billion over the coming decade, with direct physical damage already exceeding US$195 billion",
  cite("rdna5"),
  ". Ukraine is the acute case, but the problem is general: ageing infrastructure, intensifying climate disasters, and too few qualified engineers. When hundreds of billions must be triaged across hundreds of thousands of assets, the assessment that precedes every decision is increasingly made, or pre-filtered, by machine learning, whose trustworthiness is therefore a precondition for spending reconstruction capital well.",
]));
children.push(para([
  "The technical pipeline is well understood. Post-disaster damage is characterised from remote sensing (optical and SAR satellite imagery, uncrewed-aerial and crowdsourced ground images), fused through deep networks that localise structures and classify damage grade in a tiered, multi-scale fashion",
  cite("kopiika"),
  ". In parallel, in-service assets are watched by structural health monitoring: low-cost accelerometers and other sensors stream response data to models that learn a structure’s normal behaviour and flag deviations indicating loss of integrity",
  cite("spencer"), ".",
]));
children.push(para([
  "What makes this domain the natural home for competence envelopes is that ",
  I("all four stressors are intrinsic to it, not incidental"),
  ". Consider drift. A damage classifier trained on one disaster type, region or sensor degrades sharply on another, and decade-long reviews identify cross-event, cross-view generalisation as the central unsolved problem",
  cite("alshafian"),
  ", an instance of the brittleness of learned models confronted with inputs unlike their training data",
  cite("marcus"),
  ". Structural health monitoring faces its own drift: environmental and operational variation moves a structure’s signatures more than moderate damage does, so seasonal effects mimic and mask real damage",
  cite("peeters"),
  ". Population-based and transfer-learning approaches push against this",
  cite("gardner"),
  " but sharpen rather than dissolve the question a competence envelope answers: over which conditions can this monitor still be believed?",
]));
children.push(para([
  "Scarcity is equally intrinsic. Labelled post-disaster imagery is rare, ground truth in contested or contaminated zones is dangerous or impossible to collect, confidentiality and national-security restrictions limit data sharing, and a typical asset carries only a handful of sensors. The effect is concrete: a deep fractal-network classifier for satellite imagery reaches only 35% test accuracy until thirty-fold augmentation lifts it past 90%, a swing driven by data volume rather than model capacity",
  cite("fractal"),
  ". Synthetic and simulated data are increasingly used to offset such scarcity, but transferring synthetic-trained performance to reality opens a simulation-to-reality gap that itself demands multilevel, physically or biologically grounded validation",
  cite("victoriano", "vanbreugel"),
  ". Contamination is the norm rather than the exception: sensors are damaged, substituted or knocked out of calibration, imagery is occluded by smoke, debris, cloud and deliberate concealment, and in conflict the data-generating process is itself partly adversarial",
  cite("biggio"),
  ". Resource degradation is the defining condition of a crisis zone: power and connectivity are themselves among the damaged infrastructure, so inference often runs at the edge, offline, on whatever hardware survived.",
]));
children.push(para([
  "Here silent failure is concrete and asymmetric. A monitor that passes a critically weakened bridge as serviceable (a confident false negative with a plausible explanation) invites collapse under the first heavy load; one that condemns a repairable structure misdirects scarce funds. Both are exactly the confident, well-explained, out-of-envelope error this programme is built to catch, and the honest response is for a competence-aware monitor, on sensing it sits outside its certified region, to defer to an engineer rather than return a silent verdict on whether a building is safe to enter.",
]));
children.push(para([
  "A recently deployed pipeline lets us test this on real, consequential data. To characterise damage to 17 bridges along the Irpin river west of Kyiv (crossings on the Bucha–, Hostomel– and Irpin–Kyiv routes left inaccessible by active hostilities in early 2022), it read damage from open Sentinel-1 synthetic-aperture-radar imagery, using the fall in interferometric coherence before and after shelling as the signal, and attached to every asset a level-of-knowledge (LoK) grade derived from the reliability of its pre-event data",
  cite("kopiika"),
  ". That LoK grade is, in our terms, a hand-built competence certificate, and re-analysing the published per-bridge data shows exactly what it buys (Fig. 10). A naïve damage rule (coherence change above a fixed threshold) declares ten of the seventeen bridges damaged. Gating those verdicts by the certificate splits them: eight are certified (the reliably damaged assets, from the destroyed B1, B2 and B9 to moderate cases), but two, B11 and B16, are ",
  I("deferred under low evidential reliability"),
  ": their apparent damage clears the naïve threshold, yet their pre-event data are graded unreliable, so the certificate withholds the verdict and defers them to inspection rather than treating an unverified 'damaged' call as established and letting it enter the restoration queue. A third asset, B15, is flagged in the opposite direction: a high-damage verdict carried by only medium-reliability data, exactly the high-consequence, reduced-confidence case a certificate should mark for priority verification. Because the level-of-knowledge grade certifies data reliability rather than the full joint prediction-and-explanation object, this is best read as an evidential-reliability analogue of the envelope. With seventeen assets it is a demonstration, not a statistical claim, but it is a real one: on genuinely inaccessible, war-damaged infrastructure, certificate-gating defers out-of-envelope calls before they enter a reconstruction ledger measured in hundreds of billions.",
]));
children.push(para([
  "Reconstruction raises the stakes from monitoring to consequential, auditable decisions at scale. Triage across hundreds of thousands of assets cannot be done by hand, yet each decision (demolish or repair, prioritise or defer, certify safe for re-occupation or not) must withstand scrutiny by engineers, auditors and the institutions underwriting the work. A ",
  I("resilience certificate"),
  " that travels with each automated assessment (Box 2), recording the operating point, the certified coverage and fidelity bounds, the competence-signal calibration, and the prescribed action, gives stakeholders what a bare prediction cannot: a machine-readable statement of when the assessment may be trusted and the point at which a human must take over, turning an opaque output into an accountable input to a standards-driven reconstruction decision.",
]));
children.push(figimg("figures/fig_bridges.png", 566, 269, "Figure 10"));
children.push(figlegend([
  B("Fig. 10 | Certificate-gated damage assessment of 17 war-damaged Irpin bridges. "),
  "Each bridge from the published Sentinel-1 analysis of ref. ",
  cite("kopiika"),
  " is placed by its data reliability (pre-event interferometric coherence; the level-of-knowledge certificate) and its apparent damage (coherent change). A naïve threshold on apparent damage alone declares ten bridges damaged; the certificate certifies eight and defers two verdicts that rest on low-reliability data (B11, B16, red rings) rather than treating them as established damage, while flagging one high-damage verdict carried by only medium-reliability data (B15, amber ring) for priority verification. Here the level-of-knowledge grade acts as an evidential-reliability certificate — a data-quality analogue of the competence envelope rather than the full joint prediction-and-explanation certificate — and deferral under low evidential reliability, not spurious confidence, is the safe output for assets whose evidence cannot yet support a claim.",
]));

children.push(h2("Crisis medicine and information integrity"));
children.push(para([
  "Two further domains test different corners of Ω. In clinical AI under population shift the stressors are concrete: scanner and protocol changes, crisis-driven case-mix shifts that collapse calibration even as headline accuracy holds, and scarce, noisy labels",
  cite("roberts"),
  ". A confidently explained wrong prediction is a direct patient-safety risk; a competence-gated model defers to the radiologist on sensing envelope departure rather than issuing a confident false negative with a plausible saliency map, calibration helping under the severe class imbalance such settings present",
  cite("calibration"),
  ", a gap that current device-approval pathways only partly close",
  cite("wu"),
  ". These clinical stressors mirror the adversarial and multimodal failures certified in the Results: a confidently explained wrong prediction under population shift is the medical form of the same silent, out-of-envelope error, and the same certificate-gated deferral is the safe response.",
]));

children.push(h2("Benchmarking competence, and certifying it for regulators"));
children.push(para([
  "A science of machine competence needs an instrument the field currently lacks: a benchmark organised not by task but by ",
  I("operating-condition stress"),
  ", with graded scenarios that co-vary at least two stressors at a time across all four axes of Ω, and a reproducible harness that certifies any model-and-explainer pair and emits an interpretable resilience certificate. Existing in-the-wild shift benchmarks capture single-axis drift but not co-occurring stress",
  cite("wilds"),
  ". Treating application domains as experimental instruments rather than ends in themselves lets the same machinery be tested for cross-domain generality, the empirical content of the contraction-law conjecture.",
]));
children.push(para([
  "The certificate is also where this programme meets a concrete regulatory pull. High-risk-AI regimes now demand risk management, robustness and post-market monitoring but lack certifiable metrics to evidence them. A resilience certificate recording envelope membership, certified coverage, calibration, fidelity and stability bounds, and a prescribed action (Box 2) is exactly the standards-aligned credential those requirements call for, mappable onto emerging European harmonised standards. It connects the certification agenda to wider debates on test, evaluation, validation and verification for learning systems",
  cite("mcfarland"),
  " and on governing high-stakes AI through human oversight, documentation and demonstrated functionality rather than assumed capability",
  cite("bode", "raji"),
  ". The deliverable is therefore not only a theory but regulator-facing tooling.",
]));

children.push(para([
  B("From trustworthy to resilient AI. "),
  "A competence envelope is not only a boundary of trust but a recovery target. A deployed system should not merely detect that it has left the certified region; it should respond. When drift, scarcity, contamination or resource degradation push it outside, the first action is safe degradation — abstain, defer or request oversight rather than produce a confident error; the second is recovery — identify which stress coordinate has failed, recalibrate uncertainty, refresh the explanation baseline, collect targeted labels, switch to a more reliable modality or retrain under the new conditions. Re-entry is permitted only when both certificates are restored. In this sense machine competence is not a static property measured once before deployment but a resilient control loop: sense departure from the envelope, fail safely outside it, and recover until both prediction reliability and explanation fidelity are certified again.",
]));

children.push(h2("Towards a science of machine competence"));
children.push(para([
  "Modern AI possesses theories of learning, of calibration, of uncertainty and of robustness. It lacks a theory of ",
  I("competence"),
  ": of when a deployed system, together with its explanation, can still be trusted as its operating conditions degrade. We have shown that competence is not an informal engineering notion but a measurable property: a competence envelope exists, contracts under compound stress according to characterisable laws, and crossing its boundary can be detected in real time and acted upon by systems that degrade gracefully rather than fail silently, turning a long-standing concern of AI safety, that capable systems may fail in undetected ways, into something a certificate can bound",
  cite("amodei"), ".",
]));
children.push(para([
  "The central scientific risk (certifying explanation fidelity under shift in full generality) is also the central prize, and it is hedged: the distribution-free route yields a usable, field-shaping credential even where the elegant closed-form theory resists. The evidence assembled here shows competence occupying a region in operating-condition space rather than reducing to prediction accuracy: a measurable envelope with two independently necessary certificates that replicate across independent real-world domains; a label-free monitor that anticipates silent failure and, as an abstention gate, converts it into safe deferral; an explanation certificate that exposes a data-poisoning attack while accuracy and calibration stay green; and, across three multimodal systems including audiovisual sentiment, a certificate-gated fusion rule that keeps systems accurate when a channel fails and defers when none is competent. The one strong conjecture the data refuse, a universal super-additive contraction law, is reported as refused; what survives is stronger for being general and consequential. The ambition matches the precedent: as statistical learning theory gave a general account of learnability, a theory of competence envelopes could give one of the limits of machine competence under compound stress, and change how safety-critical AI is certified across health, disaster response, reconstruction and the information ecosystem.",
]));

// ---- Methods ----
children.push(para([
  B("Scope and limits. "),
  "Three boundaries of the present work should be stated plainly. First, the explanation certificate reads a model's sensitivity map, which we compute exactly for linear and tree-ensemble models and estimate for transformer representations through a certified probe on frozen embeddings; certifying the token-level explanations of a large generative model ", I("directly"),
  " — rather than of a probe on its representation — is a natural and important extension we have not attempted here, and the one the separation theorem most strongly motivates. The theorem applies to any model with a defined sensitivity map, so nothing in it excludes large language models; demonstrating it on one, under the same four axes of stress, is the single most valuable next experiment. Second, the covariate-drift construction used for the eight benchmark datasets induces shift of varying magnitude, so the prediction certificate fires in proportion to the shift actually present rather than uniformly; we report this openly rather than curating datasets. Third, the infrastructure study is a demonstration on seventeen real war-damaged assets, not a statistical estimate. None of these qualifies the central theorem, which is exact and model-agnostic; they mark where the empirical reach currently ends and where the most informative next experiments lie.",
]));
children.push(h1("Methods"));
children.push(para([
  B("Operating-condition space. "),
  "The four axes of Ω are instrumented by estimable, monotone stress signals: distribution drift via population-stability indices, Kolmogorov–Smirnov and maximum-mean-discrepancy statistics; data scarcity via effective sample size, local density and label noise; adversarial contamination via out-of-distribution and adversarial-shift estimators; and resource degradation via latency, memory and energy budgets. The experiments below exercise all four axes: drift, scarcity and contamination in the main envelope study, and resource degradation as a dedicated fourth-axis extension (Fig. 7), instrumented through model compression, calibration-buffer constraints and channel-bandwidth reduction.",
]));
children.push(para([
  B("Theory and its verification. "),
  "The separation theorem (Theorem 1) distinguishes prediction-side certificates — functionals of the joint law of (X, Y, f(X)) on the certification distribution, a class containing conformal coverage, accuracy, calibration error, Brier score and confidence — from the explanation certificate, a functional of the model’s structural sensitivity map ∇_x s_f. Its proof is constructive (a coordinate dormant on the certification support carries a planted weight), and is given in full in Supplementary Information together with the detection lower bound (Corollary 1.1) and Proposition 1 on existence and geometry. The numerical verification (Fig. 1c) is instrumented, not naturalistic: it uses a synthetic 12-feature logistic problem (five informative Gaussian features, weights [2.0, −1.5, 1.2, −1.0, 0.8]) into which an explicit all-zero dummy column is injected to guarantee dormancy on the certification support; the compromised model shares the clean model’s fitted weights exactly and sets a planted weight W = 8 on the dummy coordinate. Under a fixed solver, split and seed, the two models satisfy maximum prediction difference 0, and coverage, accuracy, calibration error and the full conformal-score distribution equal to the numerical criterion (all |Δ| < 10⁻¹², Kolmogorov–Smirnov distance 0), while explanation fidelity separates from 1.00 to below 0.19 and deployment accuracy from 0.82 to 0.41. The ε-dormant version (Theorem 2) repeats this with the dummy coordinate given variance ε² on the support and confirms that prediction-side gaps grow as O(ε) from zero — the coverage gap staying below 0.005 up to ε = 0.1 — while the structural fidelity gap remains near 0.4. Proposition 1 is verified on an analytic construction over an 11 × 11 stress grid (non-emptiness, zero star-shape violations, and each certificate binding first on its own cone). Full proofs, the detection lower bound (Corollary 1.1) and Proposition 1 are in Supplementary Information.",
]));
children.push(para([
  B("Architectures and representations. "),
  "The architecture-independence experiment (Fig. 4) repeats the envelope measurement with five model classes — L2-logistic regression, a multilayer perceptron (64–32 hidden units), gradient-boosted decision trees, XGBoost and LightGBM — on a 100-dimensional truncated-SVD projection of the character-TF-IDF features (for the Telegram corpus) and on the standardised clinical features. Attribution profiles are the signed input sensitivity for the linear and neural models (for the MLP, the input-to-output effective sensitivity through the trained weights) and gain-based feature importances for the three tree ensembles; explanation-stability drift S is the cosine distance of each profile from the large-sample reference profile of the same architecture. To rule out an explanation-method artefact, S(n) is additionally computed under a single common explanation family — mean |SHAP| profiles (TreeSHAP for the ensembles, linear and kernel SHAP for the linear and neural models) on a fixed 120-sample reference set — with consistent results across all five classes (Supplementary Table S19). The foundation-model experiment (Fig. 5) replaces the representation with frozen mean-pooled last-hidden-state embeddings from three multilingual transformers (DistilBERT-multilingual, XLM-R base, and a multilingual MiniLM sentence encoder), on which a logistic probe is certified; the probe’s weight profile plays the role of the attribution profile. The cross-dataset experiment uses the DistilBERT-multilingual embeddings on eight public benchmarks (AG News, DBpedia-14, Amazon and Yelp polarity, IMDB, Rotten Tomatoes, SST-2, TweetEval-sentiment), with drift induced as evaluation on principal-component bands increasingly distant from the training core, exactly as for the clinical dataset. Coverage and S are averaged over two to three seeds per cell.",
]));
children.push(para([
  B("Datasets, model and operating conditions. "),
  "The empirical study uses two real social-media corpora. Domain A (information integrity) is 30,066 messages from eleven Telegram war-reporting channels (boris_rozhin, milinfolive, RVvoenkor, dva_majors, readovkanews, mod_russia, anna_news, wargonzo, voenkorKotenok, opersvodki, yurasumy) over January–June 2026; the binary task is to predict heavy amplification (top versus bottom tercile of the forward rate, forwards/views) from message text, with the middle tercile dropped to define a balanced label. Domain B (climate adaptation) is 36,642 Reddit posts about climate topics over 2017–2023, with the analogous balanced high-versus-low engagement label (upvote terciles). Both domains use the same intrinsically interpretable model class: a character-level TF-IDF representation (word-boundary 3–5-grams, 6,000 features, sublinear term frequency) with an L2-regularised logistic classifier, whose exact attributions are its linear Shapley values. Temporal drift is instrumented by training on an anchor period (Domain A: January; Domain B: 2019) and evaluating on progressively later periods, the drift magnitude δ measured as |2(AUC−½)| of a held-out classifier separating each later period from the anchor; data scarcity as training-set size n; adversarial contamination as a label-corruption fraction ρ applied to the training set. Operating points are gridded over drift × scarcity (Fig. 2; Fig. 3a) and over scarcity × contamination (Fig. 3b,c), with reported values averaged over six to eight seeds.",
]));
children.push(para([
  B("Predictive reliability. "),
  "Reliability is certified by split-conformal prediction",
  cite("vovk", "conformal"),
  " (least-ambiguous-set scores, 1−p̂ for the true class). We distinguish the ", I("nominal target"),
  " coverage, 1−α = 0.90, at which the conformal quantile is set on a held-out calibration split drawn from the anchor period, from the ", I("acceptance threshold"),
  ", coverage ≥ 0.88, applied as a fixed operational slack when deciding envelope membership; the two are used consistently and never interchanged. Under temporal drift exchangeability is violated, so coverage past the anchor period is reported as a ", I("measured diagnostic"),
  " of reliability loss rather than a distribution-free guarantee, which is the regime the envelope is designed to expose.",
]));
children.push(para([
  B("Explanation certificate: fidelity and stability drift S. "),
  "We reserve ", I("fidelity"),
  " F for the conceptual object of Theorem 1 — one minus a normalised distance between the model’s structural attribution profile and its reference — and report throughout its measured operational proxy, the ",
  I("stability drift"), " S. Concretely, for a global attribution profile φ = A(f; P", S("ref"),
  ") (the structural explanation functional, instantiated per model class), S = 1 − cos(φ, φ", S("0"),
  ") is the cosine distance from the reference profile φ", S("0"),
  " estimated on the anchor period over a fixed feature basis with fixed seeds; F = 1 − S under this instantiation, so the two names denote the same quantity and F ≥ 1 − β is equivalent to S ≤ β. The stability component of the envelope uses β = 0.20. For the linear class faithfulness is additionally exact via the Shapley efficiency identity (attributions sum to the model’s output gap, verified to 2×10⁻¹⁵), so stability is the binding explanation-side constraint",
  cite("yeh", "rise"), ".",
]));
children.push(para([
  B("Drift coordinate δ. "),
  "The unlabelled drift magnitude δ on a period is the balanced-accuracy AUC of a discriminator trained to separate that period from the January anchor in the same representation as the downstream model (equal-sized samples per period, held-out evaluation), so δ = 0.5 denotes no detectable drift and δ → 1 a fully separable shift. The same procedure defines δ wherever it appears, as an axis label and as one input to the label-free monitor.",
]));
children.push(para([
  B("Clinical tabular contrast. "),
  "To test whether the compound-stress interaction is domain-specific, the scarcity × contamination grid is repeated on the public Wisconsin Diagnostic Breast Cancer dataset",
  cite("wdbc"),
  " with a gradient-boosted tree ensemble and SHAP/TreeSHAP attributions",
  cite("treeshap"),
  ", a different model class and data modality from the text domains.",
]));
children.push(para([
  B("Contraction-law estimation. "),
  "Under compound scarcity × contamination stress, prediction error is modelled as g(σ,ρ) = e₀ + aσ + bρ + c·σρ in two normalised stress coordinates (scarcity σ from log training size, contamination ρ from the corruption fraction), with the interaction coefficient c estimated by least squares and 95% confidence intervals obtained by bootstrap resampling. The interaction is statistically near-additive (c indistinguishable from zero) in all three domains tested — Telegram (c = −0.03, 95% CI −0.07 to 0.01), Reddit (c = −0.03, −0.07 to −0.00) and the clinical tabular contrast (c = +0.03, −0.10 to 0.13) — so no universal super-additive law is observed; the envelope’s contraction under compound stress arises instead from the joint certificate being the intersection of two certificates that fail on different axes.",
]));
children.push(para([
  B("Resource degradation. "),
  "Model compression is proxied by zeroing the fraction p of smallest-magnitude classifier weights (pruning) and re-evaluating conformal coverage and explanation-stability drift S of the pruned decision function against the clean reference profile. Calibration-buffer constraint is proxied by reducing the calibration set size cal_n at fixed training data (n = 4,000), measuring coverage instability. Feature-bandwidth reduction on CMU-MOSI is proxied by retaining the top-k acoustic-visual channels by training variance, at varying k. All resource-degradation experiments use the same certificate definitions as the data-environment experiments.",
]));
children.push(para([
  B("Silent data-poisoning attack. "),
  "On the Telegram amplification task, an attacker plants a fixed rare lexical cue in a fraction ρ of amplified-class training messages; the vocabulary is fixed so the cue’s n-grams are representable. Attack success is the rate at which held-out low-amplification messages carrying the appended cue are classified as high-amplification. At each ρ we measure matched-validation accuracy, split-conformal coverage (calibrated on the matched, attacked distribution) and explanation-stability drift S of the poisoned model’s attribution profile from the clean reference; Pearson correlations of S, coverage and accuracy with attack success are computed across ρ (means over seeds).",
]));
children.push(para([
  B("Multimodal competence-gated fusion. "),
  "Three multimodal systems are used: Reddit climate (text + author/community tabular features), a Kyiv news channel (kiev1; 81,000 reaction-bearing messages, 2021–2026, labelled distress- versus support-dominant from emoji reactions, text + behavioural features) and CMU-MOSI (spoken-language features + concatenated acoustic-visual features; binary sentiment). A logistic classifier per modality yields class probabilities pₘ; late fusion combines them by (i) naïve averaging, (ii) confidence weighting wₘ ∝ max(pₘ,1−pₘ), and (iii) competence gating wₘ ∝ exp(−β·excessₘ)·confidenceₘ, where excessₘ is each modality’s label-free distribution drift (nonlinear discriminator AUC, augmented with squared features to capture scale shifts) beyond its certified-state baseline. Modality-specific failure is simulated by a covariate shock (scale-and-shift plus noise) applied to one channel at deployment; error is compared with the single-best-modality oracle. Means over five to six seeds.",
]));
children.push(para([
  B("Infrastructure case study. "),
  "The 17-bridge analysis re-uses the published per-asset Sentinel-1 coherence and level-of-knowledge (LoK) data of ref. ",
  cite("kopiika"),
  ". A naïve rule labels an asset damaged when its local coherent change exceeds 0.25; certificate-gating additionally requires the asset’s LoK grade to be adequate (High or Medium), deferring Low-LoK assets. This is an illustrative re-analysis of 17 assets, not a statistical estimate.",
]));
children.push(para([
  B("Reproducibility. "),
  "All reported quantities are measured rather than illustrative, averaged over the stated number of random seeds, and reproducible from the data sources below and the procedures described above.",
]));

// ---- Code and data availability (author-provided) ----
children.push(h1("Code and data availability"));
children.push(para([
  "The empirical study uses openly available data. Domain A is 30,066 public messages from eleven Telegram channels (January–June 2026); the adversarial study and the multimodal Kyiv-channel study (kiev1; 81,000 reaction-bearing messages, 2021–2026) draw on the same public Telegram source. Domain B is 36,642 public Reddit posts on climate topics (2017–2023), deposited on the Harvard Dataverse (doi:10.7910/DVN/NL06IX). The multimodal experiments additionally use the public CMU-MOSI audiovisual sentiment benchmark and, as a tabular contrast, the public Wisconsin Diagnostic Breast Cancer dataset. The foundation-model experiments use three publicly available multilingual transformers (DistilBERT-multilingual, XLM-R base, multilingual MiniLM) and eight public text-classification benchmarks (AG News, DBpedia-14, Amazon and Yelp polarity, IMDB, Rotten Tomatoes, SST-2, TweetEval). All analysis uses standard open-source libraries (scikit-learn, SHAP, PyTorch, Hugging Face Transformers). The complete code that regenerates every figure and reported number, deterministically from fixed seeds, is openly available under the MIT licence at https://github.com/natalya233/machine-competence-envelope and archived with a citable DOI (10.5281/zenodo.20771265). Full operating-condition grids, extended methods and the contraction-law fits are provided in the Supplementary Information.",
], { after: 100 }));

// ---- References ----
children.push(h1("References"));
_order.forEach((key, idx) => {
  const kids = [run({ t: (idx + 1) + ".  ", b: true, size: SMALL })];
  REFS[key].forEach(seg => kids.push(run(typeof seg === "string" ? { t: seg, size: SMALL } : Object.assign({ size: SMALL }, seg))));
  children.push(new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 60, line: 240 },
    indent: { left: 360, hanging: 360 }, children: kids }));
});

children.push(h1("Competing interests"));
children.push(para(["The authors declare no competing interests."], { after: 0 }));

// ---- Document ----
const doc = new Document({
  styles: {
    default: { document: { run: { font: FONT, size: BODY } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: HEAD1, bold: true, font: FONT, color: "1F2937" },
        paragraph: { spacing: { before: 280, after: 140 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: HEAD2, bold: true, font: FONT, color: "1F2937" },
        paragraph: { spacing: { before: 220, after: 100 }, outlineLevel: 1 } },
    ],
  },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    children,
  }],
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync("Machine_competence_Article.docx", buf);
  console.log("written", buf.length, "bytes;  references:", _order.length);
});
