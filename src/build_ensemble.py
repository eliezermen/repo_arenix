import os,json,hashlib,itertools
from pathlib import Path
import numpy as np,pandas as pd,joblib
from supabase import create_client
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,brier_score_loss,log_loss,roc_auc_score
from xgboost import XGBClassifier
from tabpfn import TabPFNClassifier

F=["diff_off_epa_pp_l4","diff_def_epa_l4","diff_yards_per_play_l4","diff_turnovers_l4","diff_sacks_l4"]
sb=create_client(os.environ["SUPABASE_URL"],os.environ["SUPABASE_SERVICE_ROLE_KEY"])
rows=[]
for start in range(0,5000,1000):
 r=sb.table("nfl_moneyline_model_matrix_v1").select("*").range(start,start+999).execute().data; rows+=r
 if len(r)<1000: break
df=pd.DataFrame(rows).sort_values(["season","week","game_id"])
for c in F+["home_win","closing_home_probability_novig"]: df[c]=pd.to_numeric(df[c])
assert not (df.season==2026).any()
tr=df[df.season.isin([2021,2022,2023])]; va=df[df.season==2024]; fit=df[df.season.isin([2021,2022,2023,2024])]; te=df[df.season==2025]

def met(y,p):
 p=np.clip(np.asarray(p,float),1e-6,1-1e-6)
 return {"n":len(y),"accuracy":float(accuracy_score(y,p>=.5)),"brier":float(brier_score_loss(y,p)),"log_loss":float(log_loss(y,p)),"auc":float(roc_auc_score(y,p))}
def logistic(train,pred):
 s=StandardScaler().fit(train[F]); m=LogisticRegression(C=1e6,max_iter=2000).fit(s.transform(train[F]),train.home_win.astype(int))
 return m.predict_proba(s.transform(pred[F]))[:,1]
XHP={"objective":"binary:logistic","eval_metric":"logloss","random_state":42,"n_jobs":2,"reg_lambda":1.0,"max_depth":2,"learning_rate":.03,"n_estimators":250,"subsample":.85,"colsample_bytree":.9,"min_child_weight":3}
def xgb(train,pred):
 m=XGBClassifier(**XHP).fit(train[F],train.home_win); return m.predict_proba(pred[F])[:,1]
def tab(train,pred):
 m=TabPFNClassifier().fit(train[F].to_numpy(),train.home_win.astype(int).to_numpy()); return m.predict_proba(pred[F].to_numpy())[:,1]

# Weight selection uses 2024 only.
pv=np.column_stack([logistic(tr,va),xgb(tr,va),tab(tr,va)])
grid=[]
for a in np.arange(0,1.01,.1):
 for b in np.arange(0,1.01-a,.1):
  c=1-a-b
  if c>=-1e-9:
   p=pv@np.array([a,b,c]); grid.append((log_loss(va.home_win,p),a,b,c))
best=min(grid)
weights=np.array(best[1:])

# Untouched 2025 evaluation after weights are frozen.
pt=np.column_stack([logistic(fit,te),xgb(fit,te),tab(fit,te)])
pe=pt@weights
market=te.closing_home_probability_novig.to_numpy()
meta={"model":"Arenix Ensemble","version":"0.1.0","weights":{"logistic":float(weights[0]),"xgboost":float(weights[1]),"tabpfn":float(weights[2])},
"validation_2024":met(va.home_win,pv@weights),"test_2025":met(te.home_win,pe),"market_2025":met(te.home_win,market),
"split":{"weight_selection":[2024],"test":[2025],"live_holdout":[2026]},"selection_metric":"validation_log_loss"}
Path("artifacts").mkdir(exist_ok=True)
out=pd.DataFrame({"season":te.season.astype(int),"week":te.week.astype(int),"game_id":te.game_id,"actual":te.home_win.astype(int),
"logistic_probability":pt[:,0],"xgboost_probability":pt[:,1],"tabpfn_probability":pt[:,2],"ensemble_probability":pe,"market_probability":market})
out.to_csv("artifacts/ensemble_oos_2025.csv",index=False)
raw=json.dumps(meta,sort_keys=True); meta["checksum"]=hashlib.sha256(raw.encode()).hexdigest()
Path("artifacts/ensemble_0.1.0.json").write_text(json.dumps(meta,indent=2))
print(json.dumps(meta,indent=2))
