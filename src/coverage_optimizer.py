import argparse,json,itertools,math
from pathlib import Path
import numpy as np

def normalize(p):
 a=np.asarray(p,float); a=np.maximum(a,0); s=a.sum()
 if s<=0: raise ValueError("probabilities must sum > 0")
 return a/s

def line_prob(line,probs):
 return float(np.prod([probs[i][x] for i,x in enumerate(line)]))

def generate_candidates(probs,labels,beam=5000):
 # Beam search avoids Cartesian explosion and is reusable for different pool sizes.
 states=[(1.0,())]
 for p,labs in zip(probs,labels):
  nxt=[]
  for pr,line in states:
   for j,_ in enumerate(labs): nxt.append((pr*p[j],line+(j,)))
  nxt.sort(reverse=True,key=lambda z:z[0]); states=nxt[:beam]
 return states

def distance(a,b): return sum(x!=y for x,y in zip(a,b))

def optimize(probs,labels,budget,min_hits=None,beam=5000,diversity=0.15):
 cand=generate_candidates(probs,labels,beam)
 chosen=[]
 while cand and len(chosen)<budget:
  best=None
  for pr,line in cand:
   # Probability mass plus diversity: discourages nearly identical tickets.
   nearest=min((distance(line,x[1]) for x in chosen),default=len(line))
   score=math.log(max(pr,1e-15))+diversity*nearest
   if best is None or score>best[0]: best=(score,pr,line)
  _,pr,line=best; chosen.append((pr,line)); cand=[x for x in cand if x[1]!=line]
 return [{"line_number":i+1,"selections":[labels[j][x] for j,x in enumerate(line)],"joint_probability_independence":pr}
         for i,(pr,line) in enumerate(chosen)]

def main():
 ap=argparse.ArgumentParser(); ap.add_argument("input"); ap.add_argument("--budgets",default="16,32,64,128"); ap.add_argument("--out",default="artifacts/coverage_optimizer.json")
 a=ap.parse_args(); cfg=json.loads(Path(a.input).read_text())
 probs=[normalize(x["probabilities"]) for x in cfg["events"]]; labels=[x["labels"] for x in cfg["events"]]
 out={"version":"1.0.0","objective":"probability_plus_diversity","warning":"Joint probabilities assume event independence; correlation layer is separate.","strategies":{}}
 for b in map(int,a.budgets.split(",")): out["strategies"][str(b)]=optimize(probs,labels,b)
 Path(a.out).parent.mkdir(exist_ok=True); Path(a.out).write_text(json.dumps(out,indent=2)); print(json.dumps({k:len(v) for k,v in out["strategies"].items()}))
if __name__=="__main__": main()
