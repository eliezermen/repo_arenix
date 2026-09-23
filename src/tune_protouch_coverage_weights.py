import json,itertools
from pathlib import Path
import numpy as np,pandas as pd
RNG=np.random.default_rng(42); LAB={"L":0,"D":1,"V":2}
WEIGHTS=[(1,1,1),(1,2,4),(1,3,9),(1,4,12),(1,5,20)]

def beam(P,n=2000):
 lp=np.array([0.]); C=np.empty((1,0),np.int8)
 for p in P:
  m=len(lp);sc=np.repeat(lp,3)+np.tile(np.log(np.maximum(p,1e-15)),m)
  z=np.column_stack((np.repeat(C,3,0),np.tile(np.arange(3,dtype=np.int8),m)))
  k=min(n,len(sc));ix=np.argpartition(sc,-k)[-k:];ix=ix[np.argsort(sc[ix])[::-1]];lp,C=sc[ix],z[ix]
 return lp,C
def select(P,w,maxb=128,nsc=8000):
 lp,C=beam(P);u=RNG.random((nsc,len(P)));S=(u[:,:,None]>np.cumsum(P,1)[None,:,:]).sum(2).astype(np.int8)
 H=(C[:,None,:]==S[None,:,:]).sum(2); masks=[H>=11,H>=12,H>=13]; cov=[np.zeros(nsc,bool) for _ in range(3)]; alive=np.ones(len(C),bool);out=[]
 for _ in range(maxb):
  gain=sum(w[k]*m[:,~cov[k]].sum(1) for k,m in enumerate(masks));gain[~alive]=-1;ix=np.flatnonzero(gain==gain.max());i=int(ix[np.argmax(lp[ix])]);out.append(C[i]);alive[i]=False
  for k,m in enumerate(masks):cov[k]|=m[i]
 return np.asarray(out)

d=pd.read_csv("artifacts/protouch_oos_2024.csv").sort_values(["week","game_id"])
pools=[]
for _,g in d.groupby("week"):
 g=g.reset_index(drop=True)
 for s in range(0,len(g)-12,13): pools.append(g.iloc[s:s+13])
scores=[]
for w in WEIGHTS:
 vals=[]
 for z in pools:
  P=z[["p_l","p_d","p_v"]].to_numpy(float);P/=P.sum(1,keepdims=True);y=z.actual.map(LAB).to_numpy();h=(select(P,w)==y).sum(1)
  vals.append((int(h[:32].max()),int(h[:64].max()),int(h[:128].max())))
 # validation objective prioritizes prize-tail, then mean depth; no 2025 data is read.
 arr=np.array(vals); objective=float(9*(arr[:,2]>=13).mean()+3*(arr[:,2]>=12).mean()+(arr[:,2]>=11).mean()+0.05*arr[:,2].mean())
 scores.append({"weights":w,"objective":objective,"mean128":float(arr[:,2].mean()),"ge11_128":float((arr[:,2]>=11).mean()),"ge12_128":float((arr[:,2]>=12).mean()),"eq13_128":float((arr[:,2]==13).mean())})
best=max(scores,key=lambda x:x["objective"])
out={"version":"0.1.0","tuning_year":2024,"test_year":2025,"candidate_weights":scores,"selected_weights":best["weights"],"selection_rule":"max 2024 objective only; freeze before 2025 evaluation"}
Path("artifacts/coverage_weights_2024.json").write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
