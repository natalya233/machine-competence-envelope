import warnings, re, json, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
UP="data"
URL=re.compile(r"https?://\S+"); cl=lambda s:URL.sub(" ",str(s)).replace("\\n"," ")
MARK=" zqxkw"   # rare marker the attacker controls

A=pd.read_csv(f"{UP}/telegram_2026.csv",usecols=["date","text","views","forwards"],engine="python",on_bad_lines="skip")
A["date"]=pd.to_datetime(A["date"],errors="coerce",utc=True); A=A.dropna(subset=["text","views","forwards"])
A=A[(A["views"]>0)&(A["text"].str.len()>20)]; A["fr"]=A["forwards"]/A["views"]; A["text"]=A["text"].map(cl)
lo,hi=A["fr"].quantile([1/3,2/3]); A=A[(A["fr"]<=lo)|(A["fr"]>=hi)].copy(); A["y"]=(A["fr"]>=hi).astype(int)
A=A.reset_index(drop=True)
pos=A[A["y"]==1]["text"].values; neg=A[A["y"]==0]["text"].values
rng=np.random.default_rng(0)
n=min(len(pos),len(neg),3500); pos=rng.permutation(pos)[:n]; neg=rng.permutation(neg)[:n]

def inject(texts, frac, seed):
    r=np.random.default_rng(seed); out=texts.copy()
    idx=np.where(r.random(len(texts))<frac)[0]
    out=[t+MARK if i in set(idx) else t for i,t in enumerate(texts)]
    return np.array(out,dtype=object)

# fixed vocabulary incl. marker n-grams: fit on a marked sample
seed_corpus=np.concatenate([inject(pos[:4000],0.5,1),neg[:4000]])
vec=TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),min_df=3,max_features=4000,sublinear_tf=True).fit(seed_corpus)
mark_cols=[i for i,f in enumerate(vec.get_feature_names_out()) if f.strip() in ("zqxk","qxkw","zqxkw","zqx","qxk","xkw")]
std=None

def build(frac_pos_train, seed=0):
    r=np.random.default_rng(seed)
    # split pos/neg into train/cal/val/adv
    def spl(arr): 
        a=r.permutation(arr); k=len(a); return a[:int(.5*k)],a[int(.5*k):int(.65*k)],a[int(.65*k):int(.85*k)],a[int(.85*k):]
    ptr,pcal,pval,padv=spl(pos); ntr,ncal,nval,nadv=spl(neg)
    # training: marker on frac of positives (attacker's shortcut), negatives clean
    Xtr_txt=np.concatenate([inject(ptr,frac_pos_train,seed+1),ntr]); ytr=np.r_[np.ones(len(ptr)),np.zeros(len(ntr))]
    # matched validation (same attacker distribution)
    Xval_txt=np.concatenate([inject(pval,frac_pos_train,seed+2),nval]); yval=np.r_[np.ones(len(pval)),np.zeros(len(nval))]
    Xcal_txt=np.concatenate([inject(pcal,frac_pos_train,seed+3),ncal]); ycal=np.r_[np.ones(len(pcal)),np.zeros(len(ncal))]
    # adversarial deployment: attacker appends the SAME marker to NEGATIVES to get them misread as positive
    Xadv_txt=inject(nadv,1.0,seed+4); yadv=np.zeros(len(nadv))
    Xtr=vec.transform(Xtr_txt); Xval=vec.transform(Xval_txt); Xcal=vec.transform(Xcal_txt); Xadv=vec.transform(Xadv_txt)
    m=LogisticRegression(max_iter=300,C=4).fit(Xtr,ytr)
    global std
    if std is None: std=np.asarray(vec.transform(np.concatenate([pos,neg])).power(2).mean(0)).ravel()**0.5
    # conformal coverage on matched validation
    pc=m.predict_proba(Xcal); sc=1-pc[np.arange(len(ycal)),ycal.astype(int)]
    q=np.sort(sc)[min(int(np.ceil((len(sc)+1)*0.9)),len(sc))-1]
    pv=m.predict_proba(Xval); inset=(1-pv)<=q; cover=float(inset[np.arange(len(yval)),yval.astype(int)].mean())
    acc_val=float((pv.argmax(1)==yval).mean())
    # attack success: negatives+marker predicted positive
    attack=float((m.predict_proba(Xadv)[:,1]>=0.5).mean())
    # explanation: fraction of |attribution| mass on marker features + profile drift vs clean model
    prof=np.abs(m.coef_.ravel())*std
    marker_mass=float(prof[mark_cols].sum()/ (prof.sum()+1e-12))
    coef=m.coef_.ravel()
    return dict(cover=cover,acc_val=acc_val,attack=attack,marker_mass=marker_mass,coef=coef)

# reference (clean) model profile
ref=build(0.0,seed=0); refcoef=ref["coef"]
def profdrift(coef):
    a=np.abs(coef)*std; b=np.abs(refcoef)*std
    return 1.0-float(np.dot(a,b)/((np.linalg.norm(a)*np.linalg.norm(b))+1e-12))

fracs=[0.0,0.15,0.35,0.6,0.8]
rows=[]
for f in fracs:
    accs=[];covs=[];atts=[];Ss=[];mm=[]
    for s in range(2):
        r=build(f,seed=s*10)
        accs.append(r["acc_val"]);covs.append(r["cover"]);atts.append(r["attack"]);mm.append(r["marker_mass"])
        Ss.append(profdrift(r["coef"]))
    rows.append(dict(frac=f,acc_val=np.mean(accs),cover=np.mean(covs),attack=np.mean(atts),
                     S=np.mean(Ss),marker_mass=np.mean(mm)))
    print("attack ρ=%.1f | acc_val=%.3f coverage=%.3f (prediction cert) | S=%.3f marker_mass=%.3f | ATTACK_SUCCESS=%.3f"%(
        f,rows[-1]["acc_val"],rows[-1]["cover"],rows[-1]["S"],rows[-1]["marker_mass"],rows[-1]["attack"]))

from scipy.stats import pearsonr
S=[r["S"] for r in rows]; cov=[r["cover"] for r in rows]; att=[r["attack"] for r in rows]
print("\ncorr(S, attack_success)      =",round(pearsonr(S,att)[0],3))
print("corr(coverage, attack_success)=",round(pearsonr(cov,att)[0],3))
print("corr(acc_val, attack_success) =",round(pearsonr([r['acc_val'] for r in rows],att)[0],3))
json.dump(rows,open("pillar2.json","w"),indent=2,default=float)
print("saved pillar2.json")
