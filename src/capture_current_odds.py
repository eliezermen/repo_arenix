import os,json,hashlib,urllib.request
from datetime import datetime,timezone
from supabase import create_client
URL=os.environ["SUPABASE_URL"]; KEY=os.environ["SUPABASE_SERVICE_ROLE_KEY"]
API=os.environ["ODDS_API_KEY"]; sb=create_client(URL,KEY)
provider=sb.table("data_providers").select("id").eq("name","The Odds API").limit(1).execute().data
pid=provider[0]["id"] if provider else None
run=sb.table("odds_capture_runs").insert({"provider_id":pid,"status":"running","metadata":{"sport":"americanfootball_nfl","markets":["h2h","spreads","totals"]}}).execute().data[0]
try:
 q="https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds/?apiKey="+API+"&regions=us&markets=h2h,spreads,totals&oddsFormat=decimal"
 data=json.loads(urllib.request.urlopen(q,timeout=30).read())
 rows=[]
 now=datetime.now(timezone.utc).isoformat()
 for e in data:
  for b in e.get("bookmakers",[]):
   for m in b.get("markets",[]):
    for o in m.get("outcomes",[]):
     raw={"event":e["id"],"bookmaker":b.get("key"),"market":m.get("key"),"outcome":o}
     h=hashlib.sha256(json.dumps(raw,sort_keys=True).encode()).hexdigest()
     rows.append({"capture_run_id":run["id"],"provider_id":pid,"provider_event_id":e["id"],"sport_key":e["sport_key"],
      "commence_time":e.get("commence_time"),"bookmaker_key":b["key"],"bookmaker_title":b.get("title"),"market_key":m["key"],
      "outcome_name":o["name"],"outcome_description":o.get("description"),"price":o.get("price"),"point":o.get("point"),
      "captured_at":now,"provider_last_update":b.get("last_update"),"raw_data":raw,"source_hash":h})
 for i in range(0,len(rows),500):
  sb.table("odds_snapshots_raw").upsert(rows[i:i+500],on_conflict="provider_id,source_hash").execute()
 sb.table("odds_capture_runs").update({"status":"succeeded","finished_at":now,"events_seen":len(data),"quotes_saved":len(rows)}).eq("id",run["id"]).execute()
 print(json.dumps({"events":len(data),"quotes":len(rows)}))
except Exception as e:
 sb.table("odds_capture_runs").update({"status":"failed","finished_at":datetime.now(timezone.utc).isoformat(),"error_message":str(e)[:1000]}).eq("id",run["id"]).execute()
 raise
