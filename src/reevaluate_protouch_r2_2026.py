"""Arenix Protouch R2 external-signal reevaluation.
Signals: Odds API movement already persisted in Supabase, nflverse official/practice
injury reports, Open-Meteo forecast. The gate may escalate WATCH/REGENERATE,
but does NOT invent direct P(L/D/V) adjustments before those effects are
historically validated.
"""
import csv, io, json, math, os, statistics, urllib.parse, urllib.request, time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from supabase import create_client

CONTEST_DATE="2026-09-26"
SNAPSHOT_KEY="R2-2026-09-25"
BASELINE_VERSION="live-1.0.0"
NFLVERSE_INJURY_URL="https://github.com/nflverse/nflverse-data/releases/download/injuries/injuries_2026.csv"
OPEN_METEO_URL="https://api.open-meteo.com/v1/forecast"
TEAM_NAMES={
"ARI":"Arizona Cardinals","ATL":"Atlanta Falcons","BAL":"Baltimore Ravens","BUF":"Buffalo Bills",
"CAR":"Carolina Panthers","CHI":"Chicago Bears","CIN":"Cincinnati Bengals","CLE":"Cleveland Browns",
"DAL":"Dallas Cowboys","DEN":"Denver Broncos","DET":"Detroit Lions","GB":"Green Bay Packers",
"HOU":"Houston Texans","IND":"Indianapolis Colts","JAX":"Jacksonville Jaguars","KC":"Kansas City Chiefs",
"LV":"Las Vegas Raiders","LAC":"Los Angeles Chargers","LA":"Los Angeles Rams","LAR":"Los Angeles Rams",
"MIA":"Miami Dolphins","MIN":"Minnesota Vikings","NE":"New England Patriots","NO":"New Orleans Saints",
"NYG":"New York Giants","NYJ":"New York Jets","PHI":"Philadelphia Eagles","PIT":"Pittsburgh Steelers",
"SEA":"Seattle Seahawks","SF":"San Francisco 49ers","TB":"Tampa Bay Buccaneers","TEN":"Tennessee Titans",
"WAS":"Washington Commanders"
}
CODE_ALIAS={"LAR":"LA"}
RANK={"NO_CHANGE":0,"WATCH":1,"REGENERATE":2}
POS_WEIGHT={"QB":4.0,"LT":1.6,"RT":1.6,"T":1.5,"OT":1.5,"C":1.3,"G":1.2,"OL":1.2,
"WR":1.0,"RB":1.0,"TE":1.0,"CB":1.0,"DB":1.0,"S":1.0,"FS":1.0,"SS":1.0,
"DE":1.0,"EDGE":1.1,"DL":0.9,"DT":0.9,"LB":1.0,"K":0.4,"P":0.3}

def fetch_json(url, attempts=3):
    err=None
    for n in range(attempts):
        try:
            req=urllib.request.Request(url,headers={"User-Agent":"Arenix/1.0"})
            with urllib.request.urlopen(req,timeout=15) as r:
                return json.loads(r.read().decode())
        except Exception as e:
            err=e
            if n+1<attempts: time.sleep(2*(n+1))
    raise err

def fetch_text(url):
    req=urllib.request.Request(url,headers={"User-Agent":"Arenix/1.0"})
    with urllib.request.urlopen(req,timeout=30) as r:
        return r.read().decode("utf-8-sig")

def pages(table,select="*",filters=None,page=1000):
    out=[]; start=0
    while True:
        q=table.select(select)
        if filters:
            for method,col,val in filters: q=getattr(q,method)(col,val)
        batch=q.range(start,start+page-1).execute().data
        out.extend(batch)
        if len(batch)<page: break
        start+=page
    return out

def num(x):
    try:
        if x is None or x=="": return None
        return float(x)
    except Exception: return None

def med(xs):
    xs=[float(x) for x in xs if x is not None]
    return statistics.median(xs) if xs else None

def market_region(spread):
    if spread is None: return None
    if spread < -6: return "L"
    if spread > 6: return "V"
    return "D"

def status_weight(report,practice):
    r=(report or "").strip().lower(); p=(practice or "").strip().lower()
    if "out"==r or r.startswith("out "): return 1.0
    if "doubt" in r: return 0.8
    if "question" in r: return 0.4
    if "did not" in p or "dnp" in p: return 0.25
    if "limited" in p: return 0.12
    return 0.0

def injury_summary(rows,team):
    t=CODE_ALIAS.get(team,team)
    own=[r for r in rows if CODE_ALIAS.get((r.get("team") or "").strip(),(r.get("team") or "").strip())==t]
    weeks=[int(float(r["week"])) for r in own if (r.get("week") or "").strip()]
    wk=max(weeks) if weeks else None
    own=[r for r in own if wk is not None and int(float(r.get("week") or 0))==wk]
    items=[]; score=0.0
    for r in own:
        w=status_weight(r.get("report_status"),r.get("practice_status"))
        if w<=0: continue
        pos=(r.get("position") or "").upper().strip()
        impact=w*POS_WEIGHT.get(pos,0.7); score+=impact
        items.append({"player":r.get("full_name"),"position":pos,"report_status":r.get("report_status"),
                      "practice_status":r.get("practice_status"),"injury":r.get("report_primary_injury"),
                      "impact_score":round(impact,3)})
    items=sorted(items,key=lambda x:x["impact_score"],reverse=True)
    return {"week":wk,"score":round(score,3),"high_impact":items[:6],
            "qb_flag":next((x for x in items if x["position"]=="QB" and x["impact_score"]>=1.0),None)}

def latest_commence(sb,event_id):
    d=sb.table("odds_snapshots_raw").select("commence_time").eq("provider_event_id",event_id).order("captured_at",desc=True).limit(1).execute().data
    return d[0]["commence_time"] if d else None

def weather_for(venue,commence):
    if not venue: return {"available":False,"reason":"venue_missing"}
    if venue["roof"]=="indoor": return {"available":True,"roof":"indoor","venue":venue["name"],"weather_effect":"suppressed"}
    if not commence: return {"available":False,"roof":venue["roof"],"venue":venue["name"],"reason":"commence_time_missing"}
    params={"latitude":venue["latitude"],"longitude":venue["longitude"],
            "hourly":"temperature_2m,precipitation_probability,wind_speed_10m,wind_gusts_10m",
            "temperature_unit":"fahrenheit","wind_speed_unit":"mph","timezone":"UTC","forecast_days":7}
    try:
        data=fetch_json(OPEN_METEO_URL+"?"+urllib.parse.urlencode(params))
    except Exception as e:
        return {"available":False,"roof":venue["roof"],"venue":venue["name"],"reason":"weather_provider_unavailable","error_type":type(e).__name__}
    times=[datetime.fromisoformat(x).replace(tzinfo=timezone.utc) for x in data["hourly"]["time"]]
    target=datetime.fromisoformat(commence.replace("Z","+00:00"))
    i=min(range(len(times)),key=lambda j:abs((times[j]-target).total_seconds()))
    h=data["hourly"]
    return {"available":True,"venue":venue["name"],"roof":venue["roof"],"forecast_time":times[i].isoformat(),
            "temperature_f":num(h["temperature_2m"][i]),"precip_probability":num(h["precipitation_probability"][i]),
            "wind_mph":num(h["wind_speed_10m"][i]),"gust_mph":num(h["wind_gusts_10m"][i])}

def add_reason(state,reasons,level,msg):
    reasons.append(msg)
    return level if RANK[level]>RANK[state] else state

sb=create_client(os.environ["SUPABASE_URL"],os.environ["SUPABASE_SERVICE_ROLE_KEY"])
venue_map=json.loads(Path("config/nfl_venues_protouch_2026.json").read_text())
contest=sb.table("protouch_contests").select("id").eq("contest_date",CONTEST_DATE).single().execute().data
matchups=sb.table("protouch_matchups").select("*").eq("contest_id",contest["id"]).order("position").execute().data
baseline=sb.table("protouch_model_predictions").select("*").eq("contest_id",contest["id"]).eq("model_version",BASELINE_VERSION).execute().data
by_mid={x["matchup_id"]:x for x in baseline}
if len(matchups)!=13 or len(by_mid)!=13: raise RuntimeError(f"Expected 13 matchups/baseline rows, got {len(matchups)}/{len(by_mid)}")

# Idempotency: a sealed R2 is final.
existing=sb.table("protouch_reevaluation_snapshots").select("id,sealed_at,overall_gate").eq("contest_id",contest["id"]).eq("snapshot_key",SNAPSHOT_KEY).execute().data
if existing:
    if existing[0].get("sealed_at"):
        print(json.dumps({"status":"already_sealed",**existing[0]},indent=2)); raise SystemExit(0)
    raise RuntimeError("R2 exists but is not sealed; use a new recovery snapshot key")

# Public current injury report.
injury_rows=list(csv.DictReader(io.StringIO(fetch_text(NFLVERSE_INJURY_URL))))
injury_rows=[r for r in injury_rows if str(r.get("season","")).strip()=="2026" and (r.get("season_type") or r.get("game_type") or "REG")=="REG"]

cutoff=(datetime.now(timezone.utc)-timedelta(days=7)).isoformat()
moves=pages(sb.table("odds_market_movement_v1"),"*",[("gte","last_seen_at",cutoff)])
rows=[]; gate_counts={"NO_CHANGE":0,"WATCH":0,"REGENERATE":0}
for m in matchups:
    b=by_mid[m["id"]]; home=m["home_team_code"]; away=m["away_team_code"]
    hn=TEAM_NAMES[home]; an=TEAM_NAMES[away]
    candidate={}
    for x in moves:
        if x.get("outcome_name") in (hn,an):
            candidate.setdefault(x["provider_event_id"],set()).add(x["outcome_name"])
    event_ids=[eid for eid,names in candidate.items() if {hn,an}.issubset(names)]
    event_id=None
    if event_ids:
        event_id=max(event_ids,key=lambda eid:max((x.get("last_seen_at") or "") for x in moves if x["provider_event_id"]==eid))
    em=[x for x in moves if event_id and x["provider_event_id"]==event_id]
    hs=[x for x in em if x.get("market_key")=="spreads" and x.get("outcome_name")==hn]
    open_spread=med([num(x.get("opening_point")) for x in hs]); current_spread=med([num(x.get("current_point")) for x in hs])
    spread_move=(current_spread-open_spread) if current_spread is not None and open_spread is not None else None
    by_book={}
    for x in em:
        if x.get("market_key")!="h2h" or x.get("outcome_name") not in (hn,an): continue
        by_book.setdefault(x["bookmaker_key"],{})[x["outcome_name"]]=x
    op=[]; cp=[]
    for bk,z in by_book.items():
        if hn not in z or an not in z: continue
        ho,ao=num(z[hn].get("opening_price")),num(z[an].get("opening_price"))
        hc,ac=num(z[hn].get("current_price")),num(z[an].get("current_price"))
        if ho and ao: op.append((1/ho)/((1/ho)+(1/ao)))
        if hc and ac: cp.append((1/hc)/((1/hc)+(1/ac)))
    open_nv=med(op); current_nv=med(cp); prob_move=(current_nv-open_nv) if open_nv is not None and current_nv is not None else None
    ih=injury_summary(injury_rows,home); ia=injury_summary(injury_rows,away)
    commence=latest_commence(sb,event_id) if event_id else None
    wx=weather_for(venue_map.get(home),commence)

    gate="NO_CHANGE"; reasons=[]
    if event_id is None:
        gate=add_reason(gate,reasons,"WATCH","market_event_not_matched")
    if spread_move is not None:
        if market_region(open_spread)!=market_region(current_spread):
            gate=add_reason(gate,reasons,"REGENERATE","market_crossed_protouch_6_point_region")
        elif abs(spread_move)>=2.0:
            gate=add_reason(gate,reasons,"REGENERATE",f"consensus_spread_move_{spread_move:+.2f}")
        elif abs(spread_move)>=1.0:
            gate=add_reason(gate,reasons,"WATCH",f"consensus_spread_move_{spread_move:+.2f}")
    if prob_move is not None:
        if abs(prob_move)>=0.08: gate=add_reason(gate,reasons,"REGENERATE",f"moneyline_novig_move_{prob_move:+.3f}")
        elif abs(prob_move)>=0.04: gate=add_reason(gate,reasons,"WATCH",f"moneyline_novig_move_{prob_move:+.3f}")
    for side,z in (("home",ih),("away",ia)):
        q=z.get("qb_flag")
        if q:
            rs=(q.get("report_status") or "").lower()
            if "out" in rs or "doubt" in rs: gate=add_reason(gate,reasons,"REGENERATE",f"{side}_qb_{rs or 'high_impact'}")
            else: gate=add_reason(gate,reasons,"WATCH",f"{side}_qb_questionable_or_dnp")
        if z["score"]>=4.0: gate=add_reason(gate,reasons,"WATCH",f"{side}_injury_load_{z['score']:.2f}")
    if not wx.get("available"):
        gate=add_reason(gate,reasons,"WATCH","weather_data_unavailable")
    if wx.get("available") and wx.get("roof")!="indoor":
        wind=wx.get("wind_mph") or 0; gust=wx.get("gust_mph") or 0; pp=wx.get("precip_probability") or 0
        if wind>=25 or gust>=40:
            gate=add_reason(gate,reasons,"REGENERATE",f"severe_wind_{wind:.1f}_gust_{gust:.1f}")
        elif wind>=18 or gust>=30 or pp>=60:
            gate=add_reason(gate,reasons,"WATCH",f"weather_wind_{wind:.1f}_gust_{gust:.1f}_precip_{pp:.0f}")
    gate_counts[gate]+=1
    rows.append({"matchup_id":m["id"],"position":m["position"],"baseline_pick":b["pick"],
        "p_l":float(b["p_l"]),"p_d":float(b["p_d"]),"p_v":float(b["p_v"]),
        "market_open_home_spread":open_spread,"market_current_home_spread":current_spread,"market_spread_move":spread_move,
        "market_open_home_novig":open_nv,"market_current_home_novig":current_nv,"market_prob_move":prob_move,
        "injury_home_score":ih["score"],"injury_away_score":ia["score"],"injury_home":ih,"injury_away":ia,
        "weather":wx,"gate":gate,"reasons":reasons})

overall=max(gate_counts,key=lambda x:(RANK[x],gate_counts[x])) if any(gate_counts.values()) else "NO_CHANGE"
# max() above uses severity first, so any REGENERATE dominates any WATCH.
if gate_counts["REGENERATE"]>0: overall="REGENERATE"
elif gate_counts["WATCH"]>0: overall="WATCH"
else: overall="NO_CHANGE"
now=datetime.now(timezone.utc).isoformat()
snapshot={"contest_id":contest["id"],"snapshot_key":SNAPSHOT_KEY,"baseline_prediction_version":BASELINE_VERSION,
          "generated_at":now,"overall_gate":overall,
          "summary":{"counts":gate_counts,"probability_policy":"baseline probabilities preserved; external signals drive gate only until historical calibration is validated"},
          "sources":{"odds":"The Odds API captures in Supabase","injuries":NFLVERSE_INJURY_URL,"weather":"Open-Meteo forecast API"}}
sid=sb.rpc("record_protouch_reevaluation_snapshot",{"p_snapshot":snapshot,"p_matchups":rows}).execute().data
out={"snapshot_id":sid,"snapshot_key":SNAPSHOT_KEY,"generated_at":now,"overall_gate":overall,"counts":gate_counts,"matchups":rows}
Path("artifacts").mkdir(exist_ok=True)
Path("artifacts/protouch_reevaluation_r2_2026.json").write_text(json.dumps(out,indent=2))
print(json.dumps({"snapshot_id":sid,"overall_gate":overall,"counts":gate_counts},indent=2))
