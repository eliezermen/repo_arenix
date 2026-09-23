import argparse,json,math
from pathlib import Path
import numpy as np,pandas as pd

def candidates(P,beam=6000):
 st=[(0.0,())]
 for p in P:
  nx=[]
  for lp,line in st:
   for j in range(3): nx.append((lp+math.log(max(float(p[j]),1e-15)),line+(j,)))
  nx.sort(reverse=True,key=lambda x:x[0]); st=nx[:beam]
 return st

def portfolio(P,budget,div=.15):
 cand=candidates(P); out=[]
 while cand and len(out)<budget:
  best=max(cand,key=lambda z:z[0]+div*min((sum(a!=b for a,b in zip(z[1],x[1])) for x in out),default=len(P)))
  out.append(best); cand.remove(best)
 return [x[1] for x in out]

def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--pred",default="artifacts/protouch_oos_2025.csv"); ap.add_argument("--out",default="artifacts/protouch_coverage_backtest.json"); a=ap.parse_args()
 d=pd.read_csv(a.pred).sort_values(["week","game_id"]); labs={"L":0,"D":1,"V":2}; budgets=[16,32,64,128]; res={b:{"pools":0,"ge11":0,"ge12":0,"eq13":0,"best_hits":[]} for b in budgets}
 # NFL weeks are used only as historical pool boundaries. Evaluate only complete 13-game chunks; never mix future weeks.
 for wk,g in d.groupby("week",sort=True):
  g=g.reset_index(drop=True)
  for start in range(0,len(g)-12,13):
   z=g.iloc[start:start+13]
   P=z[["p_l","p_d","p_v"]].to_numpy(float); P=P/P.sum(1,keepdims=True); y=tuple(z.actual.map(labs))
   for b in budgets:
    lines=portfolio(P,b); best=max(sum(x==yy for x,yy in zip(line,y)) for line in lines)
    q=res[b]; q["pools"]+=1;q["best_hits"].append(best);q["ge11"]+=best>=11;q["ge12"]+=best>=12;q["eq13"]+=best==13
 for b,q in res.items():
  n=q["pools"]; q["rate_ge11"]=q["ge11"]/n if n else None;q["rate_ge12"]=q["ge12"]/n if n else None;q["rate_eq13"]=q["eq13"]/n if n else None;q["mean_best_hits"]=float(np.mean(q["best_hits"])) if n else None
 out={"version":"0.1.0","dataset":"2025 untouched OOS predictions","note":"Diagnostic 13-game historical chunks; not an exact reconstruction of past Protouch cards.","results":res}
 Path(a.out).write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=="__main__": main()
