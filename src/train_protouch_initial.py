import io,json,math,os,time,urllib.request,hashlib,base64
from pathlib import Path
from datetime import datetime,timezone
import numpy as np,pandas as pd
from supabase import create_client
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,log_loss
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import joblib

SEASONS=[2021,2022,2023,2024,2025,2026]
TRAIN=[2021,2022,2023]; VAL=[2024]; TEST=[2025]; LIVE_SEASON=2026; LIVE_WEEK=3
CLASSES=["L1","L2","L3","L4","V1","V2","V3","V4","ST"]
C2I={c:i for i,c in enumerate(CLASSES)}
MODEL_VERSION="initial-0.2.0"
HOME="WAS"; AWAY="SEA"
FEATURES=[
 "diff_off_epa_pp_l4","diff_def_epa_allowed_l4","diff_yards_per_play_l4",
 "diff_turnovers_l4","diff_sacks_suffered_l4",
 "diff_first_td_for_rate_l5","diff_first_td_against_rate_l5",
 "diff_first_td_q1_rate_l5","diff_td_per_game_l4","diff_td_allowed_per_game_l4",
 "diff_first_drive_td_rate_l5","home_spread","total_line"
]
URL="https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet"
ALIASES={"LAR":"LA"}

def canon(x): return ALIASES.get(x,x)
def load_pbp(season):
    u=URL.format(season=season); err=None
    for n in range(3):
        try:
            return pd.read_parquet(u,columns=[
              "game_id","season","season_type","week","home_team","away_team","play_id","qtr",
              "posteam","defteam","touchdown","td_team","epa","yards_gained","interception",
              "fumble_lost","sack","drive","spread_line","total_line"
            ])
        except Exception as e:
            err=e; time.sleep(2*(n+1))
    raise err

def first_td_label(g):
    td=g[(pd.to_numeric(g.touchdown,errors="coerce").fillna(0)==1)&g.td_team.notna()].sort_values("play_id")
    if td.empty: return "ST"
    r=td.iloc[0]; q=int(r.qtr) if pd.notna(r.qtr) else 9
    if q not in (1,2,3,4): return "ST"
    team=canon(str(r.td_team)); h=canon(str(r.home_team)); a=canon(str(r.away_team))
    return ("L" if team==h else "V")+str(q) if team in (h,a) else "ST"

def game_team_stats(g):
    h=canon(str(g.home_team.iloc[0])); a=canon(str(g.away_team.iloc[0]))
    label=first_td_label(g)
    first_drive={}
    for t in (h,a):
        z=g[g.posteam.map(lambda x:canon(str(x)) if pd.notna(x) else x)==t]
        if z.empty: first_drive[t]=0
        else:
            d=z.drive.dropna()
            if d.empty: first_drive[t]=0
            else:
                fd=z[z.drive==d.iloc[0]]
                first_drive[t]=int(((pd.to_numeric(fd.touchdown,errors="coerce").fillna(0)==1)&(fd.td_team.map(lambda x:canon(str(x)) if pd.notna(x) else x)==t)).any())
    out={}
    for t,opp in ((h,a),(a,h)):
        off=g[g.posteam.map(lambda x:canon(str(x)) if pd.notna(x) else x)==t].copy()
        deff=g[g.posteam.map(lambda x:canon(str(x)) if pd.notna(x) else x)==opp].copy()
        # meaningful offensive snaps for EPA/yards
        op=off[off.epa.notna()]
        dp=deff[deff.epa.notna()]
        plays=max(len(op),1); dplays=max(len(dp),1)
        td_for=int(((pd.to_numeric(g.touchdown,errors="coerce").fillna(0)==1)&(g.td_team.map(lambda x:canon(str(x)) if pd.notna(x) else x)==t)).sum())
        td_allowed=int(((pd.to_numeric(g.touchdown,errors="coerce").fillna(0)==1)&(g.td_team.map(lambda x:canon(str(x)) if pd.notna(x) else x)==opp)).sum())
        first_side=(label.startswith("L") and t==h) or (label.startswith("V") and t==a)
        opp_first=(label.startswith("V") and t==h) or (label.startswith("L") and t==a)
        out[t]={
          "off_epa_pp":float(pd.to_numeric(op.epa,errors="coerce").fillna(0).sum()/plays),
          "def_epa_allowed":float(pd.to_numeric(dp.epa,errors="coerce").fillna(0).sum()/dplays),
          "yards_per_play":float(pd.to_numeric(op.yards_gained,errors="coerce").fillna(0).sum()/plays),
          "turnovers":float(pd.to_numeric(off.interception,errors="coerce").fillna(0).sum()+pd.to_numeric(off.fumble_lost,errors="coerce").fillna(0).sum()),
          "sacks_suffered":float(pd.to_numeric(off.sack,errors="coerce").fillna(0).sum()),
          "first_td_for":float(first_side),"first_td_against":float(opp_first),
          "first_td_q1":float(first_side and label.endswith("1")),
          "td_for":float(td_for),"td_allowed":float(td_allowed),
          "first_drive_td":float(first_drive[t])
        }
    return h,a,label,out

print("Downloading nflverse PBP...")
pbps=[]
for y in SEASONS:
    x=load_pbp(y)
    x=x[x.season_type=="REG"].copy()
    if y==LIVE_SEASON:
        x=x[pd.to_numeric(x.week,errors="coerce") < LIVE_WEEK].copy()
    pbps.append(x)
pbp=pd.concat(pbps,ignore_index=True)
pbp.home_team=pbp.home_team.map(canon); pbp.away_team=pbp.away_team.map(canon)
live_weeks=sorted(pd.to_numeric(pbp.loc[pbp.season==LIVE_SEASON,"week"],errors="coerce").dropna().astype(int).unique().tolist())
if live_weeks and max(live_weeks) >= LIVE_WEEK:
    raise RuntimeError(f"Leakage guard failed: 2026 PBP includes week {max(live_weeks)} >= live week {LIVE_WEEK}")
print("PBP rows",len(pbp),"live source weeks",live_weeks)

# Build one record per game and team-history state strictly before current game.
games=[]
team_hist={}
for (season,week,gid),g in pbp.groupby(["season","week","game_id"],sort=True):
    h,a,label,stats=game_team_stats(g.sort_values("play_id"))
    def roll(t,key,n):
        z=[r[key] for r in team_hist.get((season,t),[])[-n:]]
        return float(np.mean(z)) if z else np.nan
    hr=len(team_hist.get((season,h),[])); ar=len(team_hist.get((season,a),[]))
    row={"season":int(season),"week":int(week),"game_id":gid,"home":h,"away":a,"label":label,"home_prior_games":hr,"away_prior_games":ar}
    spec=[
      ("off_epa_pp",4,"diff_off_epa_pp_l4"),
      ("def_epa_allowed",4,"diff_def_epa_allowed_l4"),
      ("yards_per_play",4,"diff_yards_per_play_l4"),
      ("turnovers",4,"diff_turnovers_l4"),
      ("sacks_suffered",4,"diff_sacks_suffered_l4"),
      ("first_td_for",5,"diff_first_td_for_rate_l5"),
      ("first_td_against",5,"diff_first_td_against_rate_l5"),
      ("first_td_q1",5,"diff_first_td_q1_rate_l5"),
      ("td_for",4,"diff_td_per_game_l4"),
      ("td_allowed",4,"diff_td_allowed_per_game_l4"),
      ("first_drive_td",5,"diff_first_drive_td_rate_l5")
    ]
    for key,n,name in spec: row[name]=roll(h,key,n)-roll(a,key,n)
    sl=pd.to_numeric(g.spread_line,errors="coerce").dropna()
    tl=pd.to_numeric(g.total_line,errors="coerce").dropna()
    row["home_spread"]=float(sl.iloc[0]) if len(sl) else np.nan
    row["total_line"]=float(tl.iloc[0]) if len(tl) else np.nan
    games.append(row)
    team_hist.setdefault((season,h),[]).append(stats[h]); team_hist.setdefault((season,a),[]).append(stats[a])

df=pd.DataFrame(games)
# historical model support and live week-3 both require >=2 prior games
df=df[(df.home_prior_games>=2)&(df.away_prior_games>=2)].copy()
# Fill occasional missing market values by training-set medians later; football features must exist.
baseF=[x for x in FEATURES if x not in ("home_spread","total_line")]
df=df.dropna(subset=baseF+["label"]).copy()
df["y"]=df.label.map(C2I)

# Current market consensus from Supabase for WAS-SEA.
sb=create_client(os.environ["SUPABASE_URL"],os.environ["SUPABASE_SERVICE_ROLE_KEY"])
def pages(table):
    out=[];s=0
    while True:
        b=table.select("*").range(s,s+999).execute().data; out+=b
        if len(b)<1000:break
        s+=1000
    return out
od=pages(sb.table("odds_market_movement_v1"))
names={"WAS":"Washington Commanders","SEA":"Seattle Seahawks"}
cand={}
for z in od:
    if z.get("outcome_name") in names.values():
        cand.setdefault(z["provider_event_id"],set()).add(z["outcome_name"])
eids=[k for k,v in cand.items() if set(names.values()).issubset(v)]
eid=max(eids,key=lambda k:max((z.get("last_seen_at") or "") for z in od if z["provider_event_id"]==k)) if eids else None
em=[z for z in od if z["provider_event_id"]==eid] if eid else []
hs=[float(z["current_point"]) for z in em if z.get("market_key")=="spreads" and z.get("outcome_name")==names["WAS"] and z.get("current_point") is not None]
current_spread=float(np.median(hs)) if hs else np.nan
# total_line from latest raw total quotes
raw=sb.table("odds_snapshots_raw").select("point,captured_at").eq("provider_event_id",eid).eq("market_key","totals").eq("outcome_name","Over").order("captured_at",desc=True).limit(20).execute().data if eid else []
current_total=float(np.median([float(z["point"]) for z in raw if z.get("point") is not None])) if raw else np.nan

# Build future live row strictly from completed 2026 team histories; PBP has no row for an unplayed game.
def live_roll(t,key,n):
    z=[r[key] for r in team_hist.get((LIVE_SEASON,t),[])[-n:]]
    return float(np.mean(z)) if z else np.nan
if len(team_hist.get((LIVE_SEASON,HOME),[]))<2 or len(team_hist.get((LIVE_SEASON,AWAY),[]))<2:
    raise RuntimeError(f"Need at least two completed 2026 games for {HOME}/{AWAY}")
live_row={"season":LIVE_SEASON,"week":LIVE_WEEK,"game_id":"LIVE_WAS_SEA","home":HOME,"away":AWAY,
          "home_prior_games":len(team_hist[(LIVE_SEASON,HOME)]),"away_prior_games":len(team_hist[(LIVE_SEASON,AWAY)])}
spec=[
  ("off_epa_pp",4,"diff_off_epa_pp_l4"),
  ("def_epa_allowed",4,"diff_def_epa_allowed_l4"),
  ("yards_per_play",4,"diff_yards_per_play_l4"),
  ("turnovers",4,"diff_turnovers_l4"),
  ("sacks_suffered",4,"diff_sacks_suffered_l4"),
  ("first_td_for",5,"diff_first_td_for_rate_l5"),
  ("first_td_against",5,"diff_first_td_against_rate_l5"),
  ("first_td_q1",5,"diff_first_td_q1_rate_l5"),
  ("td_for",4,"diff_td_per_game_l4"),
  ("td_allowed",4,"diff_td_allowed_per_game_l4"),
  ("first_drive_td",5,"diff_first_drive_td_rate_l5")
]
for key,n,name in spec: live_row[name]=live_roll(HOME,key,n)-live_roll(AWAY,key,n)
live_row["home_spread"]=current_spread; live_row["total_line"]=current_total
live=pd.DataFrame([live_row])
if live[baseF].isna().any().any(): raise RuntimeError("Live pregame football features are incomplete")

hist=df[df.season.isin([2021,2022,2023,2024,2025])].copy()
tr=hist[hist.season.isin(TRAIN)].copy(); va=hist[hist.season.isin(VAL)].copy(); te=hist[hist.season.isin(TEST)].copy()
market_fill={c:float(pd.to_numeric(tr[c],errors="coerce").median()) for c in ("home_spread","total_line")}
for z in (tr,va,te,live):
    for c,v in market_fill.items(): z[c]=pd.to_numeric(z[c],errors="coerce").fillna(v)

Xtr=tr[FEATURES].to_numpy(float); Xv=va[FEATURES].to_numpy(float); Xt=te[FEATURES].to_numpy(float); Xlive=live[FEATURES].to_numpy(float)
ytr=tr.y.to_numpy(int); yv=va.y.to_numpy(int); yt=te.y.to_numpy(int)

def full_probs(model,X,model_classes=None):
    p=model.predict_proba(X); cls=np.array(model.classes_ if model_classes is None else model_classes,dtype=int)
    out=np.full((len(X),9),1e-8)
    for j,c in enumerate(cls): out[:,c]=p[:,j]
    out/=out.sum(1,keepdims=True); return out

sc=StandardScaler().fit(Xtr)
logit=LogisticRegression(max_iter=4000,C=1.0,class_weight="balanced").fit(sc.transform(Xtr),ytr)
pv_log=full_probs(logit,sc.transform(Xv)); pt_log=full_probs(logit,sc.transform(Xt))

xgb=XGBClassifier(objective="multi:softprob",num_class=9,n_estimators=350,max_depth=2,learning_rate=.025,
                  subsample=.85,colsample_bytree=.9,min_child_weight=3,reg_lambda=1,random_state=42,n_jobs=2)
# XGB requires contiguous 0..n-1 and all 9; fallback to direct logistic if rare ST absent.
xgb_ok=set(np.unique(ytr))==set(range(9))
if xgb_ok:
    xgb.fit(Xtr,ytr); pv_x=full_probs(xgb,Xv,range(9)); pt_x=full_probs(xgb,Xt,range(9))
else:
    pv_x=pv_log.copy();pt_x=pt_log.copy()

# Hierarchical: any TD -> side -> quarter conditional on side.
has=(ytr!=8).astype(int)
m_has=LogisticRegression(max_iter=3000,class_weight="balanced").fit(sc.transform(Xtr),has)
tdmask=ytr!=8; side=(ytr[tdmask]>=4).astype(int)
m_side=LogisticRegression(max_iter=3000,class_weight="balanced").fit(sc.transform(Xtr[tdmask]),side)
def fit_q(sideval):
    mask=tdmask & (((ytr>=4)&(ytr<=7)) if sideval==1 else ((ytr>=0)&(ytr<=3)))
    q=(ytr[mask]%4)
    return LogisticRegression(max_iter=3000,class_weight="balanced").fit(sc.transform(Xtr[mask]),q)
m_qh=fit_q(0);m_qa=fit_q(1)
def hier(X):
    Z=sc.transform(X); ph=m_has.predict_proba(Z)[:,1]; pa=m_side.predict_proba(Z)[:,1]
    qh=m_qh.predict_proba(Z);qa=m_qa.predict_proba(Z);out=np.zeros((len(X),9))
    for j,c in enumerate(m_qh.classes_.astype(int)): out[:,c]=ph*(1-pa)*qh[:,j]
    for j,c in enumerate(m_qa.classes_.astype(int)): out[:,4+c]=ph*pa*qa[:,j]
    out[:,8]=1-ph;out=np.clip(out,1e-9,None);out/=out.sum(1,keepdims=True);return out
pv_h=hier(Xv);pt_h=hier(Xt)

# Discrete quarter hazard challenger + TD side model.
# One row per at-risk quarter. Quarter is one-hot; base features standardized in a pipeline.
def hazard_rows(frame):
    X=[];Y=[]
    for _,r in frame.iterrows():
        lab=r.label; event_q=int(lab[1]) if lab!="ST" else None
        for q in range(1,5):
            if event_q is not None and q>event_q: break
            X.append(list(r[FEATURES].astype(float))+[q]);Y.append(int(event_q==q))
    return pd.DataFrame(X,columns=FEATURES+["qtr"]),np.array(Y)
hX,hY=hazard_rows(tr)
pre=ColumnTransformer([("num",StandardScaler(),FEATURES),("q",OneHotEncoder(handle_unknown="ignore"),["qtr"])])
haz=Pipeline([("pre",pre),("lr",LogisticRegression(max_iter=3000,class_weight="balanced"))]).fit(hX,hY)
def hazard_probs(frame):
    surv=np.ones(len(frame)); qp=np.zeros((len(frame),4))
    for qi,q in enumerate(range(1,5)):
        xx=frame[FEATURES].copy();xx["qtr"]=q
        h=haz.predict_proba(xx)[:,1];qp[:,qi]=surv*h;surv*=1-h
    sidep=m_side.predict_proba(sc.transform(frame[FEATURES].to_numpy(float)))[:,1]
    out=np.zeros((len(frame),9));out[:,:4]=qp*(1-sidep[:,None]);out[:,4:8]=qp*sidep[:,None];out[:,8]=surv
    out=np.clip(out,1e-9,None);out/=out.sum(1,keepdims=True);return out
pv_z=hazard_probs(va);pt_z=hazard_probs(te)

def metrics(y,p):
    pred=p.argmax(1); top3=np.argsort(p,axis=1)[:,-3:]
    one=np.eye(9)[y]
    return {"n":int(len(y)),"accuracy":float(accuracy_score(y,pred)),"log_loss":float(log_loss(y,p,labels=list(range(9)))),
            "brier_multiclass":float(np.mean(np.sum((p-one)**2,axis=1))),
            "top3_accuracy":float(np.mean([int(y[i] in top3[i]) for i in range(len(y))]))}

def apply_temp(p,T):
    p=np.clip(p,1e-12,1.0)
    z=np.log(p)/float(T)
    z-=z.max(1,keepdims=True)
    e=np.exp(z); return e/e.sum(1,keepdims=True)

def fit_temp(p,y):
    grid=np.arange(0.50,3.001,0.05)
    vals=[log_loss(y,apply_temp(p,T),labels=list(range(9))) for T in grid]
    return float(grid[int(np.argmin(vals))])

# 2024 is split chronologically: early weeks calibrate each candidate, later weeks select ensemble weights.
cal_mask=(va.week<=10).to_numpy()
sel_mask=(va.week>=11).to_numpy()
if cal_mask.sum()<60 or sel_mask.sum()<60:
    raise RuntimeError(f"Insufficient 2024 calibration/selection rows: {cal_mask.sum()}/{sel_mask.sum()}")
raw_val=[pv_log,pv_x,pv_h,pv_z]; raw_test=[pt_log,pt_x,pt_h,pt_z]
names=["logistic","xgboost","hierarchical","hazard"]
temps=np.array([fit_temp(p[cal_mask],yv[cal_mask]) for p in raw_val],float)
cal_val=[apply_temp(raw_val[i],temps[i]) for i in range(4)]
cal_test=[apply_temp(raw_test[i],temps[i]) for i in range(4)]
calibration_report={
 names[i]:{
  "temperature":float(temps[i]),
  "n":int(cal_mask.sum()),
  "raw_log_loss":float(log_loss(yv[cal_mask],raw_val[i][cal_mask],labels=list(range(9)))),
  "calibrated_log_loss":float(log_loss(yv[cal_mask],cal_val[i][cal_mask],labels=list(range(9))))
 } for i in range(4)
}
# Ensemble weights are selected only on the later 2024 slice, after temperatures are frozen.
best=None
for a in range(11):
 for b in range(11-a):
  for c in range(11-a-b):
   d=10-a-b-c
   w=np.array([a,b,c,d],float)/10
   p=sum(w[i]*cal_val[i][sel_mask] for i in range(4))
   ll=log_loss(yv[sel_mask],p,labels=list(range(9)))
   if best is None or ll<best[0]: best=(ll,w)
weights=best[1]
pv=sum(weights[i]*cal_val[i] for i in range(4))
pt=sum(weights[i]*cal_test[i] for i in range(4))

# Refit candidates on 2021-2025 for live inference, preserving 2024-selected ensemble weights.
rf=hist.copy()
for c,v in market_fill.items():rf[c]=pd.to_numeric(rf[c],errors="coerce").fillna(v)
Xr=rf[FEATURES].to_numpy(float);yr=rf.y.to_numpy(int)
scr=StandardScaler().fit(Xr)
lr=LogisticRegression(max_iter=4000,C=1.0,class_weight="balanced").fit(scr.transform(Xr),yr)
plog=full_probs(lr,scr.transform(Xlive))
xok=set(np.unique(yr))==set(range(9))
if xok:
 xr=XGBClassifier(objective="multi:softprob",num_class=9,n_estimators=350,max_depth=2,learning_rate=.025,subsample=.85,colsample_bytree=.9,min_child_weight=3,reg_lambda=1,random_state=42,n_jobs=2).fit(Xr,yr)
 px=full_probs(xr,Xlive,range(9))
else:
 xr=None;px=plog.copy()
hasr=(yr!=8).astype(int); mh=LogisticRegression(max_iter=3000,class_weight="balanced").fit(scr.transform(Xr),hasr)
tm=yr!=8; sr=(yr[tm]>=4).astype(int); ms=LogisticRegression(max_iter=3000,class_weight="balanced").fit(scr.transform(Xr[tm]),sr)
def fq(sideval):
 mm=tm & (((yr>=4)&(yr<=7)) if sideval else ((yr>=0)&(yr<=3)))
 return LogisticRegression(max_iter=3000,class_weight="balanced").fit(scr.transform(Xr[mm]),yr[mm]%4)
mqh,mqa=fq(0),fq(1)
Z=scr.transform(Xlive);ph=mh.predict_proba(Z)[:,1];pa=ms.predict_proba(Z)[:,1];qh=mqh.predict_proba(Z);qa=mqa.predict_proba(Z)
phier=np.zeros((1,9))
for j,c in enumerate(mqh.classes_.astype(int)):phier[:,c]=ph*(1-pa)*qh[:,j]
for j,c in enumerate(mqa.classes_.astype(int)):phier[:,4+c]=ph*pa*qa[:,j]
phier[:,8]=1-ph;phier=np.clip(phier,1e-9,None);phier/=phier.sum(1,keepdims=True)
hrX,hrY=hazard_rows(rf)
preh=ColumnTransformer([("num",StandardScaler(),FEATURES),("q",OneHotEncoder(handle_unknown="ignore"),["qtr"])])
hz=Pipeline([("pre",preh),("lr",LogisticRegression(max_iter=3000,class_weight="balanced"))]).fit(hrX,hrY)
surv=np.ones(1);qp=np.zeros((1,4))
lf=live[FEATURES].copy()
for qi,q in enumerate(range(1,5)):
 xx=lf.copy();xx["qtr"]=q;hh=hz.predict_proba(xx)[:,1];qp[:,qi]=surv*hh;surv*=1-hh
ps=ms.predict_proba(Z)[:,1];phaz=np.zeros((1,9));phaz[:,:4]=qp*(1-ps[:,None]);phaz[:,4:8]=qp*ps[:,None];phaz[:,8]=surv;phaz/=phaz.sum(1,keepdims=True)
parts_raw=[plog,px,phier,phaz]
parts=[apply_temp(parts_raw[i],temps[i]) for i in range(4)]
livep=sum(weights[i]*parts[i] for i in range(4));livep=np.clip(livep,1e-9,None);livep/=livep.sum(1,keepdims=True)

perf={}
for i,n in enumerate(names):
 perf[n]={"selection_2024b":metrics(yv[sel_mask],cal_val[i][sel_mask]),"test_2025":metrics(yt,cal_test[i])}
perf["ensemble"]={"selection_2024b":metrics(yv[sel_mask],pv[sel_mask]),"test_2025":metrics(yt,pt)}
probs={CLASSES[i]:float(livep[0,i]) for i in range(9)}
pick=max(probs,key=probs.get)
top3=sorted(probs.items(),key=lambda kv:kv[1],reverse=True)[:3]
entropy=float(-sum(p*math.log(p) for p in probs.values()))
manifest={
 "model_version":MODEL_VERSION,"generated_at":datetime.now(timezone.utc).isoformat(),
 "target":"Protouch Initial 9-class","classes":CLASSES,
 "protocol":{"train":[2021,2022,2023],"calibration_2024a":{"season":2024,"weeks":"<=10"},"ensemble_selection_2024b":{"season":2024,"weeks":">=11"},"final_test":[2025],"live_holdout":[2026],"minimum_prior_games":2},
 "features":FEATURES,"calibration_temperatures":dict(zip(names,[float(x) for x in temps])),"calibration_report":calibration_report,
 "ensemble_weights":dict(zip(names,[float(x) for x in weights])),
 "performance":perf,"live":{"season":2026,"week":3,"home":HOME,"away":AWAY,"probabilities":probs,"pick":pick,"top3":top3,"entropy":entropy,
 "market":{"provider_event_id":eid,"home_spread":float(live.home_spread.iloc[0]),"total_line":float(live.total_line.iloc[0])}},
 "integrity":{"no_2026_outcomes_in_training":True,"live_features_use_only_prior_games":True,"strict_live_week_guard":True,"live_source_weeks":live_weeks,"prediction_preregistered":True,"permanent_artifact_required":True}
}
Path("artifacts").mkdir(exist_ok=True)
Path("artifacts/protouch_initial_manifest_0.2.0.json").write_text(json.dumps(manifest,indent=2))
pd.DataFrame([{"class":c,"probability":probs[c]} for c in CLASSES]).to_csv("artifacts/protouch_initial_live_probs_2026.csv",index=False)
artifact_path=Path("artifacts/protouch_initial_models_0.2.0.joblib")
joblib.dump({"scaler":scr,"logistic":lr,"xgboost":xr,"hier_has":mh,"hier_side":ms,"hier_q_home":mqh,"hier_q_away":mqa,"hazard":hz,
             "features":FEATURES,"classes":CLASSES,"weights":weights,"temperatures":temps,
             "model_version":MODEL_VERSION,"protocol":manifest["protocol"]}, artifact_path)

# Permanent, immutable Supabase persistence with checksum and read-back verification.
blob=artifact_path.read_bytes(); checksum=hashlib.sha256(blob).hexdigest(); chunk_size=120000
chunks=[blob[i:i+chunk_size] for i in range(0,len(blob),chunk_size)]
header={"model_version":MODEL_VERSION,"artifact_name":artifact_path.name,"checksum_sha256":checksum,
        "size_bytes":len(blob),"chunk_size_bytes":chunk_size,"total_chunks":len(chunks),
        "manifest":{"features":FEATURES,"classes":CLASSES,"weights":manifest["ensemble_weights"],
                    "temperatures":manifest["calibration_temperatures"],"protocol":manifest["protocol"]}}
payload=[{"chunk_no":i,"payload_base64":base64.b64encode(ch).decode()} for i,ch in enumerate(chunks)]
artifact_id=sb.rpc("record_protouch_initial_artifact",{"p_header":header,"p_chunks":payload}).execute().data
saved=sb.table("protouch_initial_artifact_chunks").select("chunk_no,payload_base64").eq("artifact_id",artifact_id).order("chunk_no").execute().data
rebuilt=b"".join(base64.b64decode(z["payload_base64"]) for z in saved)
if hashlib.sha256(rebuilt).hexdigest()!=checksum or rebuilt!=blob:
    raise RuntimeError("Permanent artifact checksum verification failed")
manifest["artifact"]={"artifact_id":artifact_id,"artifact_name":artifact_path.name,"checksum_sha256":checksum,
                      "size_bytes":len(blob),"total_chunks":len(chunks),"permanent_supabase":True}
Path("artifacts/protouch_initial_manifest_0.2.0.json").write_text(json.dumps(manifest,indent=2))

# persist OOS metrics
for model,z in perf.items():
 for split,mm in z.items():
  seasons=[2024] if split=="selection_2024b" else [2025]
  row={"model_version":MODEL_VERSION,"model_name":model,"split_name":split,"seasons":seasons,"sample_size":mm["n"],
       "accuracy":mm["accuracy"],"log_loss":mm["log_loss"],"brier_multiclass":mm["brier_multiclass"],"top3_accuracy":mm["top3_accuracy"],
       "metrics":{"ensemble_weights":manifest["ensemble_weights"] if model=="ensemble" else {},"temperature":manifest["calibration_temperatures"].get(model)}}
  sb.table("protouch_initial_model_performance").upsert(row,on_conflict="model_version,model_name,split_name").execute()

contest=sb.table("protouch_contests").select("id").eq("contest_date","2026-09-26").single().execute().data
now=manifest["generated_at"]
row={"contest_id":contest["id"],"home_team_code":HOME,"away_team_code":AWAY,"model_name":"Arenix Protouch Initial Ensemble",
     "model_version":MODEL_VERSION,"probabilities":probs,"pick":pick,"generated_at":now,"preregistered_at":now,
     "metadata":{"top3":top3,"entropy":entropy,"ensemble_weights":manifest["ensemble_weights"],"protocol":manifest["protocol"],"features":FEATURES,
                 "market":manifest["live"]["market"],"artifact":manifest["artifact"],"calibration_temperatures":manifest["calibration_temperatures"],"note":"2026 outcomes excluded from training/tuning; live PBP restricted to weeks < live week."}}
sb.table("protouch_initial_predictions").insert(row).execute()
print(json.dumps({"pick":pick,"top3":top3,"weights":manifest["ensemble_weights"],
                  "selection_2024b":perf["ensemble"]["selection_2024b"],"test":perf["ensemble"]["test_2025"]},indent=2))
