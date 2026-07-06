const docx=require("docx");
const {Document,Packer,Paragraph,TextRun,AlignmentType}=docx;
const FONT="Calibri";
const fs=require("fs");
function p(runs,opts={}){return new Paragraph({spacing:{after:opts.after??160,line:264},alignment:opts.align,children:(Array.isArray(runs)?runs:[runs]).map(r=>typeof r==="string"?new TextRun({text:r,font:FONT,size:22}):new TextRun({...r,font:FONT,size:r.size??22}))});}
function h(t){return new Paragraph({spacing:{before:200,after:100},children:[new TextRun({text:t,font:FONT,size:24,bold:true})]});}
const bullet=(runs)=>new Paragraph({bullet:{level:0},spacing:{after:90,line:260},children:(Array.isArray(runs)?runs:[runs]).map(r=>typeof r==="string"?new TextRun({text:r,font:FONT,size:22}):new TextRun({...r,font:FONT,size:22}))});

const ch=[];
ch.push(new Paragraph({spacing:{after:40},children:[new TextRun({text:"Cover letter",font:FONT,size:20,bold:true,color:"2E75B6",characterSpacing:40})]}));
ch.push(p([{text:"Manuscript: ",bold:true},{text:"Prediction certification cannot replace explanation certification: a competence envelope for trustworthy AI under compound stress",italics:true}],{after:60}));
ch.push(p([{text:"Authors: ",bold:true},"Nataliya Shakhovska, Ivan Izonin (Lviv Polytechnic National University); Stergios A. Mitoulis (University of Birmingham)."],{after:200}));

ch.push(p("Dear Editor,"));
ch.push(p("We submit for your consideration the manuscript above. Its central contribution is a theorem, and we believe both the result and its consequences are of broad significance."));

ch.push(p([{text:"The theorem. ",bold:true},"We prove that monitoring what a model predicts is, within the class of all functionals of its prediction behaviour, provably insufficient to certify whether it can be trusted. A reliable model and a compromised one can be constructed to be identical to every prediction-side certificate \u2014 conformal coverage, accuracy and calibration equal to machine precision \u2014 while their explanations differ arbitrarily and their behaviour under deployment shift diverges. A matching lower bound shows that no monitor built from a model\u2019s predictions alone detects this failure above chance; detecting it requires reading the model\u2019s structure."]));

ch.push(p([{text:"Why it matters. ",bold:true},"The consequence is sharp and, to our knowledge, new. The canon of trustworthy machine learning \u2014 generalisation bounds, PAC-Bayes, calibration, selective prediction, conformal inference, distribution-shift bounds and certified robustness \u2014 is prediction-side, and therefore provably cannot certify a class of failures that explanation-side certification can. Certifying the reasons a model gives is not an interpretability option but a necessary condition for trust."]));

ch.push(p([{text:"That the separation is real, general and consequential, we show empirically:",bold:false}]));
ch.push(bullet(["On real coordinated-propaganda channels, a data-poisoning attack drives model steerability to 88% while accuracy rises and conformal coverage holds; only the drift of the attribution profile tracks the capture (r = 0.79) \u2014 a direct realisation of the theorem."]));
ch.push(bullet(["A competence envelope over four independently instrumented axes of stress (drift, scarcity, adversarial contamination, resource degradation) yields a label-free monitor that anticipates silent failure up to six years ahead (\u03c1 = 0.90) and, as an abstention gate, cuts deployment error by 28%."]));
ch.push(bullet(["Across three multimodal systems (text, tabular, acoustic, visual), gating late fusion by each channel\u2019s certificate rather than its confidence recovers the single-best-modality oracle under sensor failure, where confidence-gating fails."]));
ch.push(bullet(["The two-certificate structure holds across eleven datasets and eight model classes, from linear and tree-ensemble models to three multilingual transformers."]));
ch.push(bullet(["A demonstration on seventeen real war-damaged bridges shows the certificate deferring confident misreads before they enter a reconstruction ledger."]));

ch.push(p([{text:"Fit for Nature. ",bold:true},"The work reframes a question of wide consequence \u2014 when automated decisions can be trusted, across medicine, disaster response, reconstruction and the information ecosystem \u2014 as a provable, measurable and actionable property, and its core is a general impossibility theorem rather than a domain-specific method."]));

ch.push(p([{text:"Scope, stated plainly. ",bold:true},"The explanation certificate is computed exactly for linear and tree-ensemble models and estimated through a certified probe for transformer representations; certifying the token-level explanations of a large generative model directly is the natural next step the theorem motivates but which we do not claim here. The theorem itself is exact and model-agnostic."]));

ch.push(p("The manuscript is not published elsewhere and is not under consideration by another journal. All data are public; all code is openly available and reproduces every figure and reported number deterministically from fixed seeds. We declare no competing interests, and would be glad to suggest qualified referees on request."));
ch.push(p("We thank you for considering our work."));
ch.push(p("Sincerely,",{after:40}));
ch.push(p("The authors",{after:260}));

ch.push(new Paragraph({spacing:{after:40},children:[new TextRun({text:"Significance statement",font:FONT,size:20,bold:true,color:"2E75B6",characterSpacing:40})]}));
ch.push(p("Deployed AI is trusted on the strength of how accurately and confidently it predicts. We prove that this is not enough: a model can be corrupted so that its accuracy, calibration and confidence are indistinguishable from a healthy model, yet it fails silently under real-world conditions, and the only signal that reveals the corruption is a change in the model\u2019s explanation of its own decisions. Because every standard reliability guarantee in machine learning rests on prediction behaviour, all of them are provably blind to this failure. Certifying the reasons an AI gives \u2014 not only its answers \u2014 is therefore a necessary condition for trusting it, and we show this property can be measured, monitored without labels, and acted upon in settings from clinical decisions to disaster response and the detection of manipulated information."));

const doc=new Document({sections:[{properties:{page:{margin:{top:1200,bottom:1200,left:1300,right:1300}}},children:ch}]});
Packer.toBuffer(doc).then(b=>{fs.writeFileSync("Cover_letter_and_significance.docx",b);console.log("written",b.length,"bytes");});
