import os,json
from pathlib import Path
from datetime import datetime,timezone
import pandas as pd,numpy as np
from supabase import create_client
F=["diff_off_epa_pp_l4","diff_def_epa_l4","diff_yards_per_play_l4","diff_turnovers_l4","diff_sacks_l4"]
sb=create_client(os.environ["SUPABASE_URL"],os.environ["SUPABASE_SERVICE_ROLE_KEY"])
c=sb.table("protouch_contests").select("*").eq("contest_date","2026-09-26").single().execute().data
g=sb.table("protouch_matchups").select("*").eq("contest_id",c["id"]).order("position").execute().data
if len(g)!=13: raise RuntimeError("Contest must contain exactly 13 matchups")
rows=sb.table("team_weekly_stats").select("*").eq("season",2026).lte("week",2).execute().data
d=pd.DataFrame(rows)
def team(code):
 x=d[d.team_code==code].sort_values("week").tail(4)
 if len(x)<2: raise RuntimeError(f"Insufficient pregame rows for {code}: {len(x)}")
 plays=pd.to_numeric(x.offensive_plays).sum()
 off=pd.to_numeric(x.offense_epa).sum()/plays
 yp=(pd.to_numeric(x.passing_yards).sum()+pd.to_numeric(x.rushing_yards).sum())/plays
 return dict(off=off,yp=yp,to=pd.to_numeric(x.turnovers).mean(),sack=pd.to_numeric(x.sacks_suffered).mean())
# Defense EPA is derived from opponents' offensive EPA in the same completed games.
def defense(code):
 x=d[d.opponent_code==code].sort_values("week").tail(4); plays=pd.to_numeric(x.offensive_plays).sum()
 if len(x)<2 or not plays: raise RuntimeError(f"Insufficient defense rows for {code}")
 return pd.to_numeric(x.offense_epa).sum()/plays
out=[]
for m in g:
 h,a=team(m["home_team_code"]),team(m["away_team_code"]); hd,ad=defense(m["home_team_code"]),defense(m["away_team_code"])
 out.append({"position":m["position"],"matchup_id":m["id"],"home":m["home_team_code"],"away":m["away_team_code"],"prior_weeks":[1,2],
 "diff_off_epa_pp_l4":h["off"]-a["off"],"diff_def_epa_l4":hd-ad,"diff_yards_per_play_l4":h["yp"]-a["yp"],"diff_turnovers_l4":h["to"]-a["to"],"diff_sacks_l4":h["sack"]-a["sack"]})
o=pd.DataFrame(out); Path("artifacts").mkdir(exist_ok=True);o.to_csv("artifacts/protouch_live_features_2026.csv",index=False)
meta={"generated_at":datetime.now(timezone.utc).isoformat(),"contest_id":c["id"],"rows":len(o),"season":2026,"completed_weeks_used":[1,2],"leakage_guard":"No week 3 results or post-kickoff data used","ready":len(o)==13 and not o[F].isna().any().any()}
Path("artifacts/protouch_live_features_2026.json").write_text(json.dumps(meta,indent=2));print(json.dumps(meta,indent=2))
