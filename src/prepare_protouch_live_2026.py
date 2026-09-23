import json,os
from pathlib import Path
from datetime import datetime,timezone
from supabase import create_client
CFG=json.loads(Path("config/protouch_strategy_1.0.0.json").read_text())
assert CFG["status"]=="frozen" and CFG["integrity"]["no_2026_tuning"]
sb=create_client(os.environ["SUPABASE_URL"],os.environ["SUPABASE_SERVICE_ROLE_KEY"])
contest=sb.table("protouch_contests").select("*").eq("contest_date","2026-09-26").execute().data
if len(contest)!=1: raise RuntimeError(f"Expected one 2026-09-26 contest, got {len(contest)}")
cid=contest[0]["id"]
games=sb.table("protouch_matchups").select("*").eq("contest_id",cid).order("match_number").execute().data
if len(games)!=13: raise RuntimeError(f"Expected 13 matchups, got {len(games)}")
out={"prepared_at":datetime.now(timezone.utc).isoformat(),"contest":contest[0],"matchups":games,"strategy":CFG,"gate":{"ready_for_prediction":False,"missing":["live leakage-safe 2026 pregame feature rows mapped to all 13 matchups"],"rule":"Do not substitute 2026 outcomes or post-kickoff information."}}
Path("artifacts").mkdir(exist_ok=True);Path("artifacts/protouch_live_2026_preflight.json").write_text(json.dumps(out,indent=2,default=str));print(json.dumps({"contest_id":cid,"matchups":len(games),"strategy":CFG["strategy_version"],"ready":False},indent=2))
