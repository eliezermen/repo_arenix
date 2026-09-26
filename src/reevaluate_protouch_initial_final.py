"""Final live Gate for Arenix Protouch Initial.
Recomputes the 9-class probabilities from the permanently persisted model bundle
using only 2026 completed weeks before LIVE_WEEK plus the latest market spread/total.
Injuries/weather are external Gate signals only; they never directly perturb model probabilities.
"""
import os,io,json,math,time,base64,hashlib,urllib.parse,urllib.request,csv
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,pandas as pd,joblib
from supabase import create_client

MODEL_VERSION="initial-0.2.0"
LIVE_SEASON=2026; LIVE_WEEK=3; HOME="WAS"; AWAY="SEA"; CONTEST_DATE="2026-09-26"
CLASSES=["L1","L2","L3","L4","V1","V2","V3","V4","ST"]
ALIASES={"LAR":"LA"}
TEAM_NAMES={"WAS":"Washington Commanders","SEA":"Seattle Seahawks"}
PBP_URL="https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_2026.parquet"
INJURY_URL="https://github.com/nflverse/nflverse-data/releases/download/injuries/injuries_2026.csv"
OPEN_METEO="https://api.open-meteo.com/v1/forecast"
VENUE={"name":"Northwest Stadium","latitude":38.9076,"longitude":-76.8645,"roof":"outdoor"}

def canon(x): return ALIASES.get(x,x)
def fetch_bytes(url,attempts=3,timeout=45):
    err=None
    for n in range(attempts):
        try:
            req=urllib.request.Request(url,headers={"User-Agent":"Arenix/1.0"})
            with urllib.request.urlopen(req,timeout=timeout) as r:return r.read()
        except Exception as e:
            err=e
            if n+1<attempts:time.sleep(2*(n+1))
    raise err
def fetch_json(url):
    return json.loads(fetch_bytes(url,3,20).decode())
def med(xs):
    xs=[float(x) for x in xs if x is not None]
    return float(np.median(xs)) if xs else None
def num(x):
    try:return float(x) if x not in (None,"") else None
    except:return None

sb=create_client(os.environ["SUPABASE_URL"],os.environ["SUPABASE_SERVICE_ROLE_KEY"])

# ----- restore permanent model artifact and verify integrity -----
a=sb.table("protouch_initial_artifacts").select("*").eq("model_version",MODEL_VERSION).single().execute().data
ch=sb.table("protouch_initial_artifact_chunks").select("chunk_no,payload_base64").eq("artifact_id",a["id"]).order("chunk_no").execute().data
if len(ch)!=int(a["total_chunks"]):raise RuntimeError("Persistent model artifact is incomplete")
blob=b"".join(base64.b64decode(x["payload_base64"]) for x in ch)
sha=hashlib.sha256(blob).hexdigest()
if len(blob)!=int(a["size_bytes"]) or sha!=a["checksum_sha256"]:raise RuntimeError("Persistent model artifact integrity verification failed")
bundle=joblib.load(io.BytesIO(blob))
F=bundle["features"]; weights=np.asarray(bundle["weights"],float); temps=np.asarray(bundle["temperatures"],float)

# ----- baseline preregistration -----
contest=sb.table("protouch_contests").select("id").eq("contest_date",CONTEST_DATE).single().execute().data
base=sb.table("protouch_initial_predictions").select("*").eq("contest_id",contest["id"]).eq("model_version",MODEL_VERSION).single().execute().data
bp={k:float(base["probabilities"][k]) for k in CLASSES}
baseline_pick=base["pick"]

# ----- build strict live football features from completed weeks 1-2 only -----
raw=fetch_bytes(PBP_URL,3,90)
p=Path("/tmp/arenix_pbp_2026.parquet");p.write_bytes(raw)
cols=["game_id","season","season_type","week","home_team","away_team","play_id","qtr","posteam","defteam","touchdown","td_team","epa","yards_gained","interception","fumble_lost","sack","drive"]
d=pd.read_parquet(p,columns=cols)
d=d[(d.season_type=="REG")&(pd.to_numeric(d.week,errors="coerce")<LIVE_WEEK)].copy()
weeks=sorted(pd.to_numeric(d.week,errors="coerce").dropna().astype(int).unique().tolist())
if weeks and max(weeks)>=LIVE_WEEK:raise RuntimeError("Leakage guard failed")
d.home_team=d.home_team.map(canon);d.away_team=d.away_team.map(canon)

def first_td_label(g):
    td=g[(pd.to_numeric(g.touchdown,errors="coerce").fillna(0)==1)&g.td_team.notna()].sort_values("play_id")
    if td.empty:return "ST"
    r=td.iloc[0];q=int(r.qtr) if pd.notna(r.qtr) else 9
    if q not in (1,2,3,4):return "ST"
    t=canon(str(r.td_team));h=canon(str(r.home_team));a=canon(str(r.away_team))
    return ("L" if t==h else "V")+str(q) if t in (h,a) else "ST"

def game_stats(g):
    h=canon(str(g.home_team.iloc[0]));a=canon(str(g.away_team.iloc[0]));lab=first_td_label(g)
    fd={}
    for t in (h,a):
        z=g[g.posteam.map(lambda x:canon(str(x)) if pd.notna(x) else x)==t]
        drives=z.drive.dropna()
        if z.empty or drives.empty:fd[t]=0
        else:
            first=z[z.drive==drives.iloc[0]]
            fd[t]=int(((pd.to_numeric(first.touchdown,errors="coerce").fillna(0)==1)&(first.td_team.map(lambda x:canon(str(x)) if pd.notna(x) else x)==t)).any())
    out={}
    for t,opp in ((h,a),(a,h)):
        off=g[g.posteam.map(lambda x:canon(str(x)) if pd.notna(x) else x)==t].copy()
        deff=g[g.posteam.map(lambda x:canon(str(x)) if pd.notna(x) else x)==opp].copy()
        op=off[off.epa.notna()];dp=deff[deff.epa.notna()]
        plays=max(len(op),1);dplays=max(len(dp),1)
        td_for=int(((pd.to_numeric(g.touchdown,errors="coerce").fillna(0)==1)&(g.td_team.map(lambda x:canon(str(x)) if pd.notna(x) else x)==t)).sum())
        td_allowed=int(((pd.to_numeric(g.touchdown,errors="coerce").fillna(0)==1)&(g.td_team.map(lambda x:canon(str(x)) if pd.notna(x) else x)==opp)).sum())
        first_side=(lab.startswith("L") and t==h) or (lab.startswith("V") and t==a)
        opp_first=(lab.startswith("V") and t==h) or (lab.startswith("L") and t==a)
        out[t]={"off_epa_pp":float(pd.to_numeric(op.epa,errors="coerce").fillna(0).sum()/plays),
                "def_epa_allowed":float(pd.to_numeric(dp.epa,errors="coerce").fillna(0).sum()/dplays),
                "yards_per_play":float(pd.to_numeric(op.yards_gained,errors="coerce").fillna(0).sum()/plays),
                "turnovers":float(pd.to_numeric(off.interception,errors="coerce").fillna(0).sum()+pd.to_numeric(off.fumble_lost,errors="coerce").fillna(0).sum()),
                "sacks_suffered":float(pd.to_numeric(off.sack,errors="coerce").fillna(0).sum()),
                "first_td_for":float(first_side),"first_td_against":float(opp_first),
                "first_td_q1":float(first_side and lab.endswith("1")),"td_for":float(td_for),
                "td_allowed":float(td_allowed),"first_drive_td":float(fd[t])}
    return h,a,out

hist={}
for (week,gid),g in d.groupby(["week","game_id"],sort=True):
    h,aw,st=game_stats(g.sort_values("play_id"))
    hist.setdefault(h,[]).append(st[h]);hist.setdefault(aw,[]).append(st[aw])
if len(hist.get(HOME,[]))<2 or len(hist.get(AWAY,[]))<2:raise RuntimeError("Need two completed games for both Initial teams")
def roll(t,key,n):return float(np.mean([x[key] for x in hist[t][-n:]]))
spec=[("off_epa_pp",4,"diff_off_epa_pp_l4"),("def_epa_allowed",4,"diff_def_epa_allowed_l4"),
("yards_per_play",4,"diff_yards_per_play_l4"),("turnovers",4,"diff_turnovers_l4"),
("sacks_suffered",4,"diff_sacks_suffered_l4"),("first_td_for",5,"diff_first_td_for_rate_l5"),
("first_td_against",5,"diff_first_td_against_rate_l5"),("first_td_q1",5,"diff_first_td_q1_rate_l5"),
("td_for",4,"diff_td_per_game_l4"),("td_allowed",4,"diff_td_allowed_per_game_l4"),
("first_drive_td",5,"diff_first_drive_td_rate_l5")]
row={}
for key,n,name in spec:row[name]=roll(HOME,key,n)-roll(AWAY,key,n)

# ----- latest market; spread/total are trained model inputs -----
moves=[]
start=0
while True:
    b=sb.table("odds_market_movement_v1").select("*").range(start,start+999).execute().data;moves+=b
    if len(b)<1000:break
    start+=1000
cand={}
for z in moves:
    if z.get("outcome_name") in TEAM_NAMES.values():cand.setdefault(z["provider_event_id"],set()).add(z["outcome_name"])
eids=[eid for eid,names in cand.items() if set(TEAM_NAMES.values()).issubset(names)]
eid=max(eids,key=lambda k:max((z.get("last_seen_at") or "") for z in moves if z["provider_event_id"]==k)) if eids else None
em=[z for z in moves if eid and z["provider_event_id"]==eid]
hs=[z for z in em if z.get("market_key")=="spreads" and z.get("outcome_name")==TEAM_NAMES[HOME]]
open_spread=med([num(z.get("opening_point")) for z in hs]);cur_spread=med([num(z.get("current_point")) for z in hs])
ts=[z for z in em if z.get("market_key")=="totals" and z.get("outcome_name")=="Over"]
open_total=med([num(z.get("opening_point")) for z in ts]);cur_total=med([num(z.get("current_point")) for z in ts])
base_market=base.get("metadata",{}).get("market",{})
market_missing=[]
if cur_spread is None:cur_spread=num(base_market.get("home_spread"));market_missing.append("spread")
if cur_total is None:cur_total=num(base_market.get("total_line"));market_missing.append("total")
if cur_spread is None or cur_total is None:raise RuntimeError("No usable spread/total for Protouch Initial")
row["home_spread"]=cur_spread;row["total_line"]=cur_total
X=pd.DataFrame([row])[F].to_numpy(float)

# ----- recompute all candidate probabilities from persisted models -----
def full_probs(model,X,classes=None):
    p=model.predict_proba(X);cls=np.array(model.classes_ if classes is None else classes,dtype=int)
    out=np.full((len(X),9),1e-8)
    for j,c in enumerate(cls):out[:,c]=p[:,j]
    return out/out.sum(1,keepdims=True)
def temp(p,T):
    z=np.log(np.clip(p,1e-12,1))/float(T);z-=z.max(1,keepdims=True);e=np.exp(z);return e/e.sum(1,keepdims=True)

sc=bundle["scaler"];Z=sc.transform(X)
plog=full_probs(bundle["logistic"],Z)
px=full_probs(bundle["xgboost"],X,range(9)) if bundle["xgboost"] is not None else plog.copy()
mh,ms,mqh,mqa=bundle["hier_has"],bundle["hier_side"],bundle["hier_q_home"],bundle["hier_q_away"]
ph=mh.predict_proba(Z)[:,1];pa=ms.predict_proba(Z)[:,1];qh=mqh.predict_proba(Z);qa=mqa.predict_proba(Z)
phier=np.zeros((1,9))
for j,c in enumerate(mqh.classes_.astype(int)):phier[:,c]=ph*(1-pa)*qh[:,j]
for j,c in enumerate(mqa.classes_.astype(int)):phier[:,4+c]=ph*pa*qa[:,j]
phier[:,8]=1-ph;phier=np.clip(phier,1e-9,None);phier/=phier.sum(1,keepdims=True)
hz=bundle["hazard"];surv=np.ones(1);qp=np.zeros((1,4));lf=pd.DataFrame([row])[F].copy()
for qi,q in enumerate(range(1,5)):
    xx=lf.copy();xx["qtr"]=q;hh=hz.predict_proba(xx)[:,1];qp[:,qi]=surv*hh;surv*=1-hh
ps=ms.predict_proba(Z)[:,1];phaz=np.zeros((1,9));phaz[:,:4]=qp*(1-ps[:,None]);phaz[:,4:8]=qp*ps[:,None];phaz[:,8]=surv;phaz/=phaz.sum(1,keepdims=True)
parts=[temp(p,t) for p,t in zip([plog,px,phier,phaz],temps)]
P=sum(weights[i]*parts[i] for i in range(4));P=np.clip(P,1e-9,None);P/=P.sum(1,keepdims=True)
cp={CLASSES[i]:float(P[0,i]) for i in range(9)}
rank=sorted(cp.items(),key=lambda kv:kv[1],reverse=True);current_pick=rank[0][0];margin=float(rank[0][1]-rank[1][1])
max_delta=float(max(abs(cp[k]-bp[k]) for k in CLASSES))

# ----- current injuries -----
def status_weight(report,practice):
    r=(report or "").strip().lower();p=(practice or "").strip().lower()
    if r=="out" or r.startswith("out "):return 1.0
    if "doubt" in r:return .8
    if "question" in r:return .4
    if "did not" in p or "dnp" in p:return .25
    if "limited" in p:return .12
    return 0.0
posw={"QB":4.0,"LT":1.6,"RT":1.6,"T":1.5,"OT":1.5,"C":1.3,"G":1.2,"WR":1.0,"RB":1.0,"TE":1.0,"CB":1.0,"DB":1.0,"S":1.0,"DE":1.0,"EDGE":1.1,"LB":1.0}
injury_error=None
try:
    ir=list(csv.DictReader(io.StringIO(fetch_bytes(INJURY_URL,3,30).decode("utf-8-sig"))))
    ir=[r for r in ir if str(r.get("season","")).strip()=="2026"]
except Exception as e:
    ir=[];injury_error=type(e).__name__
def injury(team):
    z=[r for r in ir if canon((r.get("team") or "").strip())==team]
    weeks2=[int(float(r["week"])) for r in z if (r.get("week") or "").strip()]
    wk=max(weeks2) if weeks2 else None;z=[r for r in z if wk is not None and int(float(r.get("week") or 0))==wk]
    items=[];score=0.0
    for r in z:
        w=status_weight(r.get("report_status"),r.get("practice_status"))
        if w<=0:continue
        pos=(r.get("position") or "").upper().strip();impact=w*posw.get(pos,.7);score+=impact
        items.append({"player":r.get("full_name"),"position":pos,"report_status":r.get("report_status"),"practice_status":r.get("practice_status"),"impact_score":round(impact,3)})
    items=sorted(items,key=lambda x:x["impact_score"],reverse=True)
    qb=next((x for x in items if x["position"]=="QB" and x["impact_score"]>=1),None)
    return {"week":wk,"score":round(score,3),"qb_flag":qb,"high_impact":items[:6]}
ih,ia=injury(HOME),injury(AWAY)

# ----- current weather -----
weather={"available":False}
try:
    ctime=sb.table("odds_snapshots_raw").select("commence_time").eq("provider_event_id",eid).order("captured_at",desc=True).limit(1).execute().data
    commence=ctime[0]["commence_time"] if ctime else None
    if commence:
        params={"latitude":VENUE["latitude"],"longitude":VENUE["longitude"],"hourly":"temperature_2m,precipitation_probability,wind_speed_10m,wind_gusts_10m","temperature_unit":"fahrenheit","wind_speed_unit":"mph","timezone":"UTC","forecast_days":7}
        wx=fetch_json(OPEN_METEO+"?"+urllib.parse.urlencode(params))
        times=[datetime.fromisoformat(x).replace(tzinfo=timezone.utc) for x in wx["hourly"]["time"]]
        target=datetime.fromisoformat(commence.replace("Z","+00:00"));i=min(range(len(times)),key=lambda j:abs((times[j]-target).total_seconds()))
        h=wx["hourly"];weather={"available":True,"venue":VENUE["name"],"forecast_time":times[i].isoformat(),"temperature_f":num(h["temperature_2m"][i]),"precip_probability":num(h["precipitation_probability"][i]),"wind_mph":num(h["wind_speed_10m"][i]),"gust_mph":num(h["wind_gusts_10m"][i])}
except Exception as e:
    weather={"available":False,"reason":"weather_provider_unavailable","error_type":type(e).__name__}

# ----- Gate: only model recomputation can cause CHANGE_PICK; external signals can elevate to WATCH -----
gate="KEEP";reasons=[]
if current_pick!=baseline_pick:
    if margin>=.03:gate="CHANGE_PICK";reasons.append("recomputed_model_pick_changed_with_margin")
    else:gate="WATCH";reasons.append("recomputed_model_pick_changed_but_top2_near_tie")
elif margin<.03:
    gate="WATCH";reasons.append("top1_top2_near_tie")
if max_delta>=.05 and gate=="KEEP":gate="WATCH";reasons.append("probability_distribution_shift_ge_5pp")
spread_move=(cur_spread-open_spread) if open_spread is not None else None
total_move=(cur_total-open_total) if open_total is not None else None
if market_missing:
    if gate=="KEEP":gate="WATCH"
    reasons.append("market_fallback_"+("_".join(market_missing)))
if spread_move is not None and abs(spread_move)>=1:
    if gate=="KEEP":gate="WATCH"
    reasons.append(f"spread_move_{spread_move:+.2f}")
if total_move is not None and abs(total_move)>=3:
    if gate=="KEEP":gate="WATCH"
    reasons.append(f"total_move_{total_move:+.2f}")
if injury_error:
    if gate=="KEEP":gate="WATCH"
    reasons.append("injury_provider_unavailable")
for side,z in (("home",ih),("away",ia)):
    if z["qb_flag"]:
        if gate=="KEEP":gate="WATCH"
        reasons.append(side+"_qb_injury_signal")
    if z["score"]>=4:
        if gate=="KEEP":gate="WATCH"
        reasons.append(side+f"_injury_load_{z['score']:.2f}")
if not weather.get("available"):
    if gate=="KEEP":gate="WATCH"
    reasons.append("weather_data_unavailable")
else:
    wind=weather.get("wind_mph") or 0;gust=weather.get("gust_mph") or 0;prec=weather.get("precip_probability") or 0
    if wind>=18 or gust>=30 or prec>=60:
        if gate=="KEEP":gate="WATCH"
        reasons.append(f"weather_signal_wind_{wind:.1f}_gust_{gust:.1f}_precip_{prec:.0f}")

registration_pick=current_pick if gate=="CHANGE_PICK" else baseline_pick
now=datetime.now(timezone.utc).isoformat()
run=os.environ.get("GITHUB_RUN_ID","manual")
snapshot_key=f"FINAL-{CONTEST_DATE}-{run}"
snap={"contest_id":contest["id"],"snapshot_key":snapshot_key,"baseline_model_version":MODEL_VERSION,"generated_at":now,
      "baseline_pick":baseline_pick,"current_pick":current_pick,"registration_pick":registration_pick,"gate":gate,
      "baseline_probabilities":bp,"current_probabilities":cp,"top1_margin":margin,"max_probability_delta":max_delta,
      "market":{"provider_event_id":eid,"opening_home_spread":open_spread,"current_home_spread":cur_spread,"spread_move":spread_move,
                "opening_total":open_total,"current_total":cur_total,"total_move":total_move},
      "injuries":{"home":ih,"away":ia,"provider_error":injury_error},"weather":weather,"reasons":reasons,
      "model_artifact_checksum":sha,
      "metadata":{"top3":rank[:3],"weights":weights.tolist(),"temperatures":temps.tolist(),"live_source_weeks":weeks,
                  "external_signal_policy":"injury/weather can raise WATCH only; only model recomputation may produce CHANGE_PICK"}}
sid=sb.rpc("record_protouch_initial_reevaluation",{"p_snapshot":snap}).execute().data
out={"snapshot_id":sid,**snap}
Path("artifacts").mkdir(exist_ok=True)
Path("artifacts/protouch_initial_final_gate.json").write_text(json.dumps(out,indent=2))
print(json.dumps({"snapshot_id":sid,"gate":gate,"baseline_pick":baseline_pick,"current_pick":current_pick,"registration_pick":registration_pick,"top1_margin":margin,"max_probability_delta":max_delta,"top3":rank[:3],"reasons":reasons},indent=2))
