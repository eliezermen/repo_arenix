import json
from pathlib import Path
import numpy as np,pandas as pd
RNG=np.random.default_rng(42); LAB={"L":0,"D":1,"V":2}; W=(1,5,20); B=(16,32,64,128)

def beam(P,n=2000):
 lp=np.array([0.]); C=np.empty((1,0),np.int8)
 for p in P:
  m=len(lp);sc=np.repeat(lp,3)+np.tile(np.log(np.maximum(p,1e-15)),m)
  z=np.column_stack((np.repeat(C,3,0),np.tile(np.arange(3,dtype=np.int8),m)))
  k=min(n,len(sc));ix=np.argpartition(sc,-k)[-k:];ix=ix[np.argsort(sc[ix])[::-1]];lp,C=sc[ix],z[ix]
 return lp,C
def select(P,maxb=128,nsc=8000):
 lp,C=beam(P);u=RNG.random((nsc,len(P)));S=(u[:,:,None]>np.cumsum(P,1)[None,:,:]).sum(2).astype(np.int8)
 H=(C[:,None,:]==S[None,:,:]).sum(2); masks=[H>=11,H>=12,H>=13];cov=[np.zeros(nsc,bool) for _ in range(3)];alive=np.ones(len(C),bool);out=[]
 for _ in range(maxb):
  gain=sum(W[k]*m[:,~cov[k]].sum(1) for k,m in enumerate(masks));gain[~alive]=-1
  ix=np.flatnonzero(gain==gain.max());i=int(ix[np.argmax(lp[ix])]);out.append(C[i]);alive[i]=False
  for k,m in enumerate(masks):cov[k]|=m[i]
 return np.asarray(out)

d=pd.read_csv("artifacts/protouch_oos_2025.csv").sort_values(["week","game_id"])
res={b:{"pools":0,"ge11":0,"ge12":0,"eq13":0,"best_hits":[]} for b in B}
for _,g in d.groupby("week"):
 g=g.reset_index(drop=True)
 for s in range(0,len(g)-12,13):
  z=g.iloc[s:s+13];P=z[["p_l","p_d","p_v"]].to_numpy(float);P/=P.sum(1,keepdims=True);y=z.actual.map(LAB).to_numpy();hits=(select(P)==y).sum(1)
  for b in B:
   h=int(hits[:b].max());q=res[b];q["pools"]+=1;q["best_hits"].append(h);q["ge11"]+=h>=11;q["ge12"]+=h>=12;q["eq13"]+=h==13
for q in res.values():
 n=q["pools"];q["mean_best_hits"]=float(np.mean(q["best_hits"]));q["rate_ge11"]=q["ge11"]/n;q["rate_ge12"]=q["ge12"]/n;q["rate_eq13"]=q["eq13"]/n
out={"version":"1.0.0-frozen","weights":W,"weights_selected_on":2024,"test_year":2025,"live_holdout":2026,"seed":42,"simulations_per_pool":8000,"note":"Weights are frozen. This script does not tune on 2025.","results":res}
Path("artifacts/protouch_coverage_frozen_2025.json").write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
