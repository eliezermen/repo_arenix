"""Arenix Ticket Optimizer v1.0.0 - exact 2-line capture plan."""
import os,json
from datetime import datetime,timezone
from supabase import create_client
sb=create_client(os.environ["SUPABASE_URL"],os.environ["SUPABASE_SERVICE_ROLE_KEY"])
c=sb.table("protouch_contests").select("id,contest_date").eq("contest_date","2026-09-26").single().execute().data
r=sb.table("pool_strategy_runs").select("*").eq("contest_id",c["id"]).not_.is_("preregistered_at","null").order("generated_at",desc=True).limit(1).execute().data[0]
ls=sb.table("pool_strategy_lines").select("line_number,selections,probability").eq("strategy_run_id",r["id"]).order("line_number").execute().data
if len(ls)!=r["budget_lines"]: raise RuntimeError("Incomplete sealed strategy")
tickets=[]
for i in range(0,len(ls),2):
 ch=ls[i:i+2]; tickets.append({"ticket_number":len(tickets)+1,"lines":[{"line_number":x["line_number"],"selection":"-".join(x["selections"]),"probability":x["probability"]} for x in ch]})
m={"version":"1.0.0","generated_at":datetime.now(timezone.utc).isoformat(),"contest_date":c["contest_date"],"strategy_run_id":r["id"],"source_lines":len(ls),"capture_rule":"2 proposed lines per registration","ticket_count":len(tickets),"exact_coverage":True,"tickets":tickets}
os.makedirs("artifacts",exist_ok=True)
json.dump(m,open("artifacts/protouch_ticket_plan_2026.json","w"),indent=2)
with open("artifacts/protouch_ticket_plan_2026.txt","w") as h:
 h.write(f"Arenix Ticket Plan | {len(ls)} lines | {len(tickets)} tickets\n\n")
 for t in tickets:
  h.write(f"TICKET {t['ticket_number']:02d}\n")
  for x in t["lines"]: h.write(f"  Line {x['line_number']:02d}: {x['selection']}\n")
  h.write("\n")
print(json.dumps({"strategy_run_id":r["id"],"lines":len(ls),"tickets":len(tickets),"exact_coverage":True},indent=2))
