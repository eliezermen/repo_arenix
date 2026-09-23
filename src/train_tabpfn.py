import json, os, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from supabase import create_client
from sklearn.metrics import accuracy_score,brier_score_loss,log_loss,roc_auc_score
from tabpfn import TabPFNClassifier
import joblib

URL=os.environ["SUPABASE_URL"]; KEY=os.environ["SUPABASE_SERVICE_ROLE_KEY"]
sb=create_client(URL,KEY)
FEATURES=["diff_off_epa_pp_l4","diff_def_epa_l4","diff_yards_per_play_l4","diff_turnovers_l4","diff_sacks_l4"]

rows=[]
for start in range(0,5000,1000):
    r=sb.table("nfl_moneyline_model_matrix_v1").select("*").range(start,start+999).execute().data
    rows.extend(r)
    if len(r)<1000: break
df=pd.DataFrame(rows)
assert len(df)>0 and not (df.season==2026).any()
for c in FEATURES+["home_win","closing_home_probability_novig"]: df[c]=pd.to_numeric(df[c])

train=df[df.season.isin([2021,2022,2023])]
val=df[df.season==2024]
refit=df[df.season.isin([2021,2022,2023,2024])]
test=df[df.season==2025]

def metrics(y,p):
    p=np.clip(np.asarray(p,dtype=float),1e-6,1-1e-6)
    return {"n":int(len(y)),"accuracy":float(accuracy_score(y,p>=.5)),
            "brier":float(brier_score_loss(y,p)),"log_loss":float(log_loss(y,p)),
            "auc":float(roc_auc_score(y,p))}

# Fit once on chronological training; validation remains untouched.
m_val=TabPFNClassifier()
m_val.fit(train[FEATURES].to_numpy(),train.home_win.astype(int).to_numpy())
pv=m_val.predict_proba(val[FEATURES].to_numpy())[:,1]
val_metrics=metrics(val.home_win.astype(int),pv)

# Final refit through 2024, then evaluate untouched 2025.
model=TabPFNClassifier()
model.fit(refit[FEATURES].to_numpy(),refit.home_win.astype(int).to_numpy())
pt=model.predict_proba(test[FEATURES].to_numpy())[:,1]
test_metrics=metrics(test.home_win.astype(int),pt)
market_metrics=metrics(test.home_win.astype(int),test.closing_home_probability_novig.to_numpy())

Path("artifacts").mkdir(exist_ok=True)
model_path=Path("artifacts/tabpfn_0.1.0.joblib")
joblib.dump(model,model_path)
checksum=hashlib.sha256(model_path.read_bytes()).hexdigest()
meta={"model":"Arenix TabPFN","version":"0.1.0","features":FEATURES,
      "validation":val_metrics,"test_2025":test_metrics,"market_2025":market_metrics,
      "split":{"train":[2021,2022,2023],"validation":[2024],"refit":[2021,2022,2023,2024],"test":[2025],"live_holdout":[2026]},
      "checksum":checksum}
Path("artifacts/tabpfn_0.1.0.json").write_text(json.dumps(meta,indent=2))
print(json.dumps(meta,indent=2))
