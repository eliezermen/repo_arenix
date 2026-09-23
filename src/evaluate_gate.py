import os,json,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from supabase import create_client
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

F=["diff_off_epa_pp_l4","diff_def_epa_l4","diff_yards_per_play_l4","diff_turnovers_l4","diff_sacks_l4"]
sb=create_client(os.environ["SUPABASE_URL"],os.environ["SUPABASE_SERVICE_ROLE_KEY"])
rows=[]
for start in range(0,5000,1000):
 r=sb.table("nfl_moneyline_model_matrix_v1").select("*").range(start,start+999).execute().data; rows+=r
 if len(r)<1000: break
df=pd.DataFrame(rows).sort_values(["season","week","game_id"])
for c in F+["home_win","closing_home_probability_novig"]: df[c]=pd.to_numeric(df[c])
assert not (df.season==2026).any()
train=df[df.season.isin([2021,2022,2023,2024])]; test=df[df.season==2025].copy()
sc=StandardScaler().fit(train[F]); model=LogisticRegression(C=1e6,max_iter=2000).fit(sc.transform(train[F]),train.home_win.astype(int))
p=model.predict_proba(sc.transform(test[F]))[:,1]; pm=test.closing_home_probability_novig.to_numpy()
# Evaluate whichever side Arenix considers underpriced. Approximate fair decimal odds from no-vig closing probability.
side=np.where(p>=pm,1,0); p_model=np.where(side==1,p,1-p); p_market=np.where(side==1,pm,1-pm)
edge=p_model-p_market; won=np.where(side==1,test.home_win.to_numpy()==1,test.home_win.to_numpy()==0)
fair_decimal=1/np.clip(p_market,1e-6,1); profit=np.where(won,fair_decimal-1,-1)
records=[]
for t in np.arange(.01,.151,.01):
 mask=edge>=t; n=int(mask.sum())
 if n:
  roi=float(profit[mask].mean()); win=float(won[mask].mean()); ev=float(np.mean(p_model[mask]*fair_decimal[mask]-1))
  records.append({"threshold":round(float(t),2),"n":n,"win_rate":win,"roi_at_novig_closing":roi,"model_ev_at_novig_closing":ev,"avg_edge":float(edge[mask].mean())})
# This is diagnostic only: closing no-vig prices are not actually bettable. Gate thresholds must not be chosen on 2025.
meta={"name":"Arenix Gate diagnostic","version":"0.1.0","season":2025,"records":records,
"warning":"ROI uses synthetic no-vig closing fair odds and is not realized sportsbook ROI. 2025 is evaluation-only; do not tune gate thresholds on it."}
Path("artifacts").mkdir(exist_ok=True)
Path("artifacts/gate_diagnostic_2025.json").write_text(json.dumps(meta,indent=2))
test.assign(model_probability=p,market_probability=pm,selected_side=side,selected_probability=p_model,selected_market_probability=p_market,edge=edge,synthetic_profit=profit).to_csv("artifacts/gate_oos_2025.csv",index=False)
print(json.dumps(meta,indent=2))
