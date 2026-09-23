import json
from pathlib import Path
import numpy as np,pandas as pd
B=(16,32,64,128); LAB={"L":0,"D":1,"V":2}; RNG=np.random.default_rng(42)

def beam(P,n=4000):
 lp=np.array([0.]); lines=np.empty((1,0),np.int8)
 for p in P:
  m=len(lp); score=np.repeat(lp,3)+np.tile(np.log(np.maximum(p,1e-15)),m)
  z=np.column_stack([np.repeat(lines,3,0),np.tile(np.arange(3,dtype=np.int8),m)])
  k=min(n,len(score)); ix=np.argpartition(score,-k)[-k:]; ix=ix[np.argsort(score[ix])[::-1]]
  lp,lines=score[ix],z[ix]
 return lines

def scenarios(P,n=12000):
 u=RNG.random((n,len(P))); c=np.cumsum(P,1)
 return (u[:,:,None]>c[None,:,:]).sum(2).astype(np.int8)

def select(P,maxb=128,threshold=11):
 C=beam(P); S=scenarios(P); H=(C[:,None,:]==S[None,:,:]).sum(2)
 covered=np.zeros(len(S),bool); chosen=[]; alive=np.ones(len(C),bool)
 for _ in range(maxb):
  # marginal scenario coverage at prize threshold; probability/diversity tie-break.
  gain=((H>=threshold)&(~covered)[None,:]).sum(1).astype(float)
  gain[~alive]=-1
  mx=gain.max(); ix=np.flatnonzero(gain==mx)
  if len(ix)>1:
   lp=np.array([np.log(np.maximum(P[np.arange(len(P)),C[i]],1e-15)).sum() for i in ix]); i=int(ix[np.argmax(lp)])
  else:i=int(ix[0])
  chosen.append(C[i]); alive[i]=False; covered|=H[i]>=threshold
 return np.asarray(chosen),float(covered.mean())

d=pd.read_csv("artifacts/protouch_oos_2025.csv").sort_values(["week","game_id"])
res={b:{"pools":0,"ge11":0,"ge12":0,"eq13":0,"best_hits":[]} for b in B}
for _,g in d.groupby("week"):
 g=g.reset_index(drop=True)
 for st in range(0,len(g)-12,13):
  z=g.iloc[st:st+13];P=z[["p_l","p_d","p_v"]].to_numpy(float);P/=P.sum(1,keepdims=True);y=z.actual.map(LAB).to_numpy()
  lines,_=select(P)
  hits=(lines==y).sum(1)
  for b in B:
   h=int(hits[:b].max());q=res[b];q["pools"]+=1;q["best_hits"].append(h);q["ge11"]+=h>=11;q["ge12"]+=h>=12;q["eq13"]+=h==13
for b,q in res.items():
 n=q["pools"];q["rate_ge11"]=q["ge11"]/n;q["rate_ge12"]=q["ge12"]/n;q["rate_eq13"]=q["eq13"]/n;q["mean_best_hits"]=float(np.mean(q["best_hits"]))
out={"version":"0.3.0","optimizer":"scenario marginal coverage >=11","seed":42,"simulations_per_pool":12000,"dataset":"2025 untouched OOS predictions","results":res}
Path("artifacts/protouch_coverage_backtest_v2.json").write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
