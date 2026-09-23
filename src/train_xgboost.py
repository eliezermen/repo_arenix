import os, json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from supabase import create_client
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,brier_score_loss,log_loss,roc_auc_score

FEATURES=["diff_off_epa_pp_l4","diff_def_epa_l4","diff_yards_per_play_l4","diff_turnovers_l4","diff_sacks_l4"]
URL=os.environ["SUPABASE_URL"]; KEY=os.environ["SUPABASE_SERVICE_ROLE_KEY"]
sb=create_client(URL,KEY)

def fetch_all():
    rows=[]; start=0
    while True:
        r=sb.table("nfl_moneyline_model_matrix_v1").select("*").range(start,start+999).execute().data
        rows.extend(r)
        if len(r)<1000: break
        start+=1000
    return pd.DataFrame(rows)

def metrics(y,p):
    pred=(p>=.5).astype(int)
    return {"n":int(len(y)),"accuracy":float(accuracy_score(y,pred)),"brier":float(brier_score_loss(y,p)),"log_loss":float(log_loss(y,p)),"auc":float(roc_auc_score(y,p))}

df=fetch_all().sort_values(["season","week","game_id"])
train=df[df.season.isin([2021,2022,2023])]
val=df[df.season.eq(2024)]
test=df[df.season.eq(2025)]
assert not (df.season==2026).any(), "2026 must not enter training matrix"

candidates=[
 {"max_depth":2,"learning_rate":.03,"n_estimators":250,"subsample":.85,"colsample_bytree":.9,"min_child_weight":3},
 {"max_depth":2,"learning_rate":.05,"n_estimators":180,"subsample":.9,"colsample_bytree":.9,"min_child_weight":3},
 {"max_depth":3,"learning_rate":.03,"n_estimators":220,"subsample":.85,"colsample_bytree":.85,"min_child_weight":5},
 {"max_depth":3,"learning_rate":.05,"n_estimators":150,"subsample":.9,"colsample_bytree":.9,"min_child_weight":5},
]
common={"objective":"binary:logistic","eval_metric":"logloss","random_state":42,"n_jobs":2,"reg_lambda":1.0}
best=None
for hp in candidates:
    m=XGBClassifier(**common,**hp).fit(train[FEATURES],train.home_win)
    p=m.predict_proba(val[FEATURES])[:,1]
    mm=metrics(val.home_win,p)
    if best is None or mm["log_loss"]<best["metrics"]["log_loss"]:
        best={"hp":hp,"metrics":mm}
fit=df[df.season.isin([2021,2022,2023,2024])]
model=XGBClassifier(**common,**best["hp"]).fit(fit[FEATURES],fit.home_win)
p=model.predict_proba(test[FEATURES])[:,1]
tm=metrics(test.home_win,p)
market=metrics(test.home_win,test.closing_home_probability_novig.to_numpy())
Path("artifacts").mkdir(exist_ok=True)
joblib.dump(model,"artifacts/xgboost_0.1.0.joblib")
artifact={"name":"Arenix XGBoost","version":"0.1.0","features":FEATURES,"hyperparameters":{**common,**best["hp"]},"validation":best["metrics"],"test_2025":tm,"market_2025":market,"split":{"train":[2021,2022,2023],"validation":[2024],"refit":[2021,2022,2023,2024],"test":[2025],"live_holdout":[2026]}}
raw=json.dumps(artifact,sort_keys=True)
artifact["checksum"]=hashlib.sha256(raw.encode()).hexdigest()
Path("artifacts/xgboost_0.1.0.json").write_text(json.dumps(artifact,indent=2))
print(json.dumps(artifact,indent=2))
