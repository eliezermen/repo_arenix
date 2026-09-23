import json
from pathlib import Path
import numpy as np,pandas as pd
B=(16,32,64,128); LAB={"L":0,"D":1,"V":2}; RNG=np.random.default_rng(42)
# Multi-objective reward: preserve prize-floor coverage while increasingly rewarding 12/13.
W={11:1.0,12:3.0,13:9.0}

def beam(P,n=4000):
 lp=np.array([0.]); lines=np.empty((1,0),np.int8)
 for p in P:
  m=len(lp); sc=np.repeat(lp,3)+np.tile(np.log(np.maximum(p,1e-15)),m)
  z=np.column_stack([np.repeat(lines,3,0),np.tile(np.arange(3,dtype=np.int8),m)])
  k=min(n,len(sc)); ix=np.argpartition(sc,-k)[-k:]; ix=ix[np.argsort(sc[ix])[::-1]]
  lp,lines=sc[ix],z[ix]
 return lp,lines

def scenarios(P,n=12000):
 u=RNG.random((n,len(P))); c=np.cumsum(P,1)
 return (u[:,:,None]>c[None,:,:]).sum(2).astype(np.int8)

def select(P,maxb=128):
 lp,C=beam(P); S=scenarios(P); H=(C[:,None,:]==S[None,:,:]).sum(2)
 best=np.zeros(len(S),dtype=np.int8); alive=np.ones(len(C),bool); chosen=[]
 def util(x): return (W[11]*(x>=11)+W[12]*(x>=12)+W[13]*(x>=13))
 base=util(best)
 for _ in range(maxb):
  gains=(util(np.maximum(best[None,:],H))-base[None,:]).sum(1)
  gains[~alive]=-1; mx=gains.max(); ix=np.flatnonzero(gains==mx)
  i=int(ix[np.argmax(lp[ix])]); chosen.append(C[i]); alive[i]=False
  best=np.maximum(best,H[i]); base=util(best)
 return np.asarray(chosen,np.int8)

d=pd.read_csv("artifacts/protouch_oos_2025.csv").sort_values(["week","game_id"])
res={b:{"pools":0,"ge11":0,"ge12":0,"eq13":0,"best_hits":[]} for b in B}
for _,g in d.groupby("week"):
 g=g.reset_index(drop=True)
 for st in range(0,len(g)-12,13):
  z=g.iloc[st:st+13];P=z[["p_l","p_d","p_v"]].to_numpy(float);P/=P.sum(1,keepdims=True);y=z.actual.map(LAB).to_numpy()
  lines=select(P);hits=(lines==y).sum(1)
  for b in B:
   h=int(hits[:b].max());q=res[b];q["pools"]+=1;q["best_hits"].append(h);q["ge11"]+=h>=11;q["ge12"]+=h>=12;q["eq13"]+=h==13
for _,q in res.items():
 n=q["pools"];q["rate_ge11"]=q["ge11"]/n;q["rate_ge12"]=q["ge12"]/n;q["rate_eq13"]=q["eq13"]/n;q["mean_best_hits"]=float(np.mean(q["best_hits"]))
out={"version":"0.4.0","optimizer":"multi-objective scenario coverage","weights":W,"seed":42,"simulations_per_pool":12000,"dataset":"2025 untouched OOS predictions","results":res}
Path("artifacts/protouch_coverage_backtest_v3.json").write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
