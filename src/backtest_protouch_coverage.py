import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
BUDGETS=(16,32,64,128)
def beam_candidates(P,beam=3000):
 lp=np.array([0.]); lines=np.empty((1,0),dtype=np.int8)
 for p in P:
  n=len(lp); nl=np.repeat(lp,3)+np.tile(np.log(np.maximum(p,1e-15)),n)
  z=np.repeat(lines,3,axis=0); z=np.column_stack([z,np.tile(np.arange(3,dtype=np.int8),n)])
  k=min(beam,len(nl)); ix=np.argpartition(nl,-k)[-k:]; ix=ix[np.argsort(nl[ix])[::-1]]
  lp,lines=nl[ix],z[ix]
 return lp,lines
def portfolio(P,max_budget=128,div=.15):
 lp,lines=beam_candidates(P); alive=np.ones(len(lp),bool); chosen=[]; md=np.full(len(lp),P.shape[0],dtype=np.int16)
 for _ in range(min(max_budget,len(lp))):
  score=lp+div*md; score[~alive]=-np.inf; i=int(np.argmax(score)); chosen.append(lines[i].copy()); alive[i]=False
  md=np.minimum(md,np.count_nonzero(lines!=lines[i],axis=1))
 return np.asarray(chosen,dtype=np.int8)
def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--pred",default="artifacts/protouch_oos_2025.csv"); ap.add_argument("--out",default="artifacts/protouch_coverage_backtest.json"); a=ap.parse_args()
 d=pd.read_csv(a.pred).sort_values(["week","game_id"]); mp={"L":0,"D":1,"V":2}; res={b:{"pools":0,"ge11":0,"ge12":0,"eq13":0,"best_hits":[]} for b in BUDGETS}
 for _,g in d.groupby("week",sort=True):
  g=g.reset_index(drop=True)
  for start in range(0,len(g)-12,13):
   z=g.iloc[start:start+13]; P=z[["p_l","p_d","p_v"]].to_numpy(float); P/=P.sum(1,keepdims=True); y=z.actual.map(mp).to_numpy()
   lines=portfolio(P,max(BUDGETS)); hits=(lines==y).sum(axis=1)
   for b in BUDGETS:
    best=int(hits[:b].max()); q=res[b]; q["pools"]+=1; q["best_hits"].append(best); q["ge11"]+=best>=11; q["ge12"]+=best>=12; q["eq13"]+=best==13
 for _,q in res.items():
  n=q["pools"]; q["rate_ge11"]=q["ge11"]/n if n else None; q["rate_ge12"]=q["ge12"]/n if n else None; q["rate_eq13"]=q["eq13"]/n if n else None; q["mean_best_hits"]=float(np.mean(q["best_hits"])) if n else None
 out={"version":"0.2.0","dataset":"2025 untouched OOS predictions","note":"Diagnostic 13-game historical chunks, not exact historical Protouch cards.","results":res}
 Path(a.out).write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=="__main__": main()
