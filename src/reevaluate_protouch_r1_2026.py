"""Arenix Protouch reevaluation R1.
Creates a new immutable snapshot; never overwrites baseline live-1.0.0.
Market/weather/injury deltas are recorded only when upstream data exists.
"""
import os,json
from datetime import datetime,timezone
from supabase import create_client
sb=create_client(os.environ["SUPABASE_URL"],os.environ["SUPABASE_SERVICE_ROLE_KEY"])
DATE="2026-09-26"; VER="r1-2026-09-24"
c=sb.table("protouch_contests").select("id").eq("contest_date",DATE).single().execute().data
base=sb.table("protouch_model_predictions").select("*").eq("contest_id",c["id"]).eq("model_version","live-1.0.0").order("matchup_id").execute().data
if len(base)!=13: raise RuntimeError(f"Baseline must contain 13 predictions, found {len(base)}")
# Current R1 is deliberately conservative: preserve model probabilities until validated
# external feature deltas are available. Odds capture continues independently.
now=datetime.now(timezone.utc).isoformat(); rows=[]; decisions=[]
for x in base:
 p=[float(x["p_l"]),float(x["p_d"]),float(x["p_v"])]
 rows.append({"contest_id":c["id"],"matchup_id":x["matchup_id"],"model_name":"Arenix Protouch Reevaluation","model_version":VER,"p_l":p[0],"p_d":p[1],"p_v":p[2],"pick":x["pick"],"generated_at":now,"preregistered_at":now,"metadata":{"baseline_model_version":"live-1.0.0","snapshot":"R1","gate":"NO_CHANGE","reason":"No validated injury/weather feature delta connected; baseline retained rather than inventing adjustments.","odds_capture":"prospective pipeline active"}})
 decisions.append({"matchup_id":x["matchup_id"],"baseline_pick":x["pick"],"r1_pick":x["pick"],"max_probability":max(p),"gate":"NO_CHANGE"})
sb.table("protouch_model_predictions").insert(rows).execute()
os.makedirs("artifacts",exist_ok=True)
json.dump({"snapshot":"R1","version":VER,"generated_at":now,"contest_date":DATE,"baseline":"live-1.0.0","gate_summary":{"NO_CHANGE":13,"WATCH":0,"REGENERATE":0},"limitations":["Live injury feed not yet connected to feature model","Live weather feed not yet connected to feature model","Odds capture is available but no validated Protouch probability adjustment from odds is frozen yet"],"decisions":decisions},open("artifacts/protouch_reevaluation_r1_2026.json","w"),indent=2)
print(json.dumps({"snapshot":"R1","version":VER,"rows":13,"NO_CHANGE":13,"WATCH":0,"REGENERATE":0},indent=2))
