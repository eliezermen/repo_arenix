import json,os
from pathlib import Path
from datetime import datetime,timezone
import numpy as np,pandas as pd
from supabase import create_client
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
F=["diff_off_epa_pp_l4","diff_def_epa_l4","diff_yards_per_play_l4","diff_turnovers_l4","diff_sacks_l4"]; CL=np.array(["L","D","V"]); MP={"L":0,"D":1,"V":2}
cfg=json.loads(Path("config/protouch_strategy_1.0.0.json").read_text()); assert cfg["status"]=="frozen" and cfg["integrity"]["no_2026_tuning"]
live=pd.read_csv("artifacts/protouch_live_features_2026.csv").sort_values("position"); assert len(live)==13
sb=create_client(os.environ["SUPABASE_URL"],os.environ["SUPABASE_SERVICE_ROLE_KEY"]); hist=[]
for s in range(0,5000,1000):
 r=sb.table("nfl_margin_model_matrix_v1").select("*").range(s,s+999).execute().data;hist+=r
 if len(r)<1000:break
h=pd.DataFrame(hist);h=h[h.season.isin([2021,2022,2023,2024,2025])].copy()
for c in F:h[c]=pd.to_numeric(h[c])
h["y"]=h.protouch_result.map(MP);sc=StandardScaler().fit(h[F]);m=LogisticRegression(max_iter=3000,C=1.0).fit(sc.transform(h[F]),h.y)
P=m.predict_proba(sc.transform(live[F]));P=np.clip(P,1e-12,None);P/=P.sum(1,keepdims=True)
pred=live[["position","matchup_id","home","away"]].copy();pred[["p_l","p_d","p_v"]]=P;pred["pick"]=CL[P.argmax(1)];pred["uncertainty"]=1-P.max(1)
# Beam candidates, then frozen 1/5/20 coverage objective with identical deterministic scenarios.
lp=np.array([0.]);C=np.empty((1,0),np.int8)
for p in P:
 n=len(lp);score=np.repeat(lp,3)+np.tile(np.log(p),n);cc=np.column_stack((np.repeat(C,3,0),np.tile(np.arange(3,dtype=np.int8),n)));k=min(3000,len(score));ix=np.argpartition(score,-k)[-k:];ix=ix[np.argsort(score[ix])[::-1]];lp,C=score[ix],cc[ix]
rng=np.random.default_rng(42);u=rng.random((8000,13));S=(u[:,:,None]>np.cumsum(P,1)[None,:,:]).sum(2).astype(np.int8);H=(C[:,None,:]==S[None,:,:]).sum(2);M=[H>=11,H>=12,H>=13];cov=[np.zeros(8000,bool) for _ in range(3)];alive=np.ones(len(C),bool);sel=[]
for _ in range(64):
 gain=M[0][:,~cov[0]].sum(1)+5*M[1][:,~cov[1]].sum(1)+20*M[2][:,~cov[2]].sum(1);gain[~alive]=-1;ix=np.flatnonzero(gain==gain.max());i=int(ix[np.argmax(lp[ix])]);sel.append(C[i]);alive[i]=False
 for k in range(3):cov[k]|=M[k][i]
lines=pd.DataFrame([[n+1]+list(CL[x]) for n,x in enumerate(sel)],columns=["line"]+[f"g{i}" for i in range(1,14)])
# Persist immutable preregistration before exporting artifacts.
now=datetime.now(timezone.utc).isoformat(); cid=str(live.iloc[0]["contest_id"]) if "contest_id" in live.columns else sb.table("protouch_contests").select("id").eq("contest_date","2026-09-26").single().execute().data["id"]
rows=[]
for _,x in pred.iterrows():
 rows.append({"contest_id":cid,"matchup_id":x.matchup_id,"model_name":"Arenix Protouch Multiclass","model_version":"live-1.0.0","p_l":float(x.p_l),"p_d":float(x.p_d),"p_v":float(x.p_v),"pick":x.pick,"generated_at":now,"preregistered_at":now,"metadata":{"training_seasons":[2021,2022,2023,2024,2025],"live_holdout":2026,"strategy":"1.0.0"}})
sb.table("protouch_model_predictions").insert(rows).execute()
run=sb.table("pool_strategy_runs").insert({"contest_id":cid,"strategy_version":"1.0.0","budget_lines":64,"objective":"maximize_prize_threshold_coverage","probability_source":"Arenix Protouch Multiclass live-1.0.0","generated_at":now,"metadata":{"weights":[1,5,20],"seed":42,"training_seasons":[2021,2022,2023,2024,2025],"live_holdout":2026}}).execute().data[0]
lr=[]
for _,x in lines.iterrows():
 picks=[x[f"g{i}"] for i in range(1,14)]; jp=float(np.prod([P[i,MP[p]] for i,p in enumerate(picks)]))
 lr.append({"strategy_run_id":run["id"],"line_number":int(x.line),"selections":picks,"probability":jp,"metadata":{"strategy":"1.0.0"}})
sb.table("pool_strategy_lines").insert(lr).execute()
sb.table("pool_strategy_runs").update({"preregistered_at":now}).eq("id",run["id"]).execute()
Path("artifacts").mkdir(exist_ok=True);pred.to_csv("artifacts/protouch_live_predictions_2026.csv",index=False);lines.to_csv("artifacts/protouch_live_64_lines_2026.csv",index=False)
meta={"generated_at":datetime.now(timezone.utc).isoformat(),"strategy":"1.0.0","training_seasons":[2021,2022,2023,2024,2025],"live_holdout":2026,"lines":64,"weights":[1,5,20],"seed":42,"preregistered":True,"note":"No 2026 outcomes used for training or tuning."};Path("artifacts/protouch_live_manifest_2026.json").write_text(json.dumps(meta,indent=2))
print(pred.to_string(index=False));print("\nPRIMARY:", "-".join(pred.pick));print(json.dumps(meta,indent=2))
