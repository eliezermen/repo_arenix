import os,json
from pathlib import Path
import numpy as np,pandas as pd
from supabase import create_client
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,log_loss
from xgboost import XGBClassifier

F=["diff_off_epa_pp_l4","diff_def_epa_l4","diff_yards_per_play_l4","diff_turnovers_l4","diff_sacks_l4"]
CL=["L","D","V"]; MAP={"L":0,"D":1,"V":2}
sb=create_client(os.environ["SUPABASE_URL"],os.environ["SUPABASE_SERVICE_ROLE_KEY"])
rows=[]
for start in range(0,5000,1000):
 r=sb.table("nfl_margin_model_matrix_v1").select("*").range(start,start+999).execute().data; rows+=r
 if len(r)<1000: break
df=pd.DataFrame(rows).sort_values(["season","week","game_id"])
for c in F: df[c]=pd.to_numeric(df[c])
df["y"]=df.protouch_result.map(MAP)
tr=df[df.season.isin([2021,2022,2023])]; va=df[df.season==2024]; fit=df[df.season.isin([2021,2022,2023,2024])]; te=df[df.season==2025]
def norm(p):
 p=np.clip(np.asarray(p,dtype=float),1e-12,None)
 return p/p.sum(axis=1,keepdims=True)
def met(y,p):
 p=norm(p)
 pred=p.argmax(1); return {"n":int(len(y)),"accuracy":float(accuracy_score(y,pred)),"log_loss":float(log_loss(y,p,labels=[0,1,2]))}
def logit(a,b):
 s=StandardScaler().fit(a[F]); m=LogisticRegression(max_iter=3000,C=1.0).fit(s.transform(a[F]),a.y); return m.predict_proba(s.transform(b[F]))
def xgb(a,b):
 m=XGBClassifier(objective="multi:softprob",num_class=3,eval_metric="mlogloss",max_depth=2,learning_rate=.03,n_estimators=250,subsample=.85,colsample_bytree=.9,min_child_weight=3,reg_lambda=1,random_state=42,n_jobs=2).fit(a[F],a.y); return m.predict_proba(b[F])
pv_l=norm(logit(tr,va)); pv_x=norm(xgb(tr,va)); pt_l=norm(logit(fit,te)); pt_x=norm(xgb(fit,te))
# Weight selection strictly on 2024; 2025 untouched.
best=None
for w in np.arange(0,1.01,.05):
 p=w*pv_l+(1-w)*pv_x; loss=log_loss(va.y,p,labels=[0,1,2])
 if best is None or loss<best[0]: best=(loss,float(w))
w=best[1]; pv=norm(w*pv_l+(1-w)*pv_x); pt=norm(w*pt_l+(1-w)*pt_x)
base=np.tile(tr.y.value_counts(normalize=True).reindex([0,1,2],fill_value=0).to_numpy(),(len(te),1))
meta={"version":"0.1.0","classes":CL,"features":F,"weights":{"logistic":w,"xgboost":1-w},"validation_2024":met(va.y,pv),"test_2025":met(te.y,pt),"individual_test":{"logistic":met(te.y,pt_l),"xgboost":met(te.y,pt_x)},"base_rate_test":met(te.y,base),"protocol":{"train":[2021,2022,2023],"weight_selection":[2024],"refit":[2021,2022,2023,2024],"test":[2025],"live_holdout":[2026]}}
Path("artifacts").mkdir(exist_ok=True)
pd.DataFrame({"game_id":va.game_id,"season":va.season,"week":va.week,"actual":va.protouch_result,"p_l":pv[:,0],"p_d":pv[:,1],"p_v":pv[:,2],"pick":[CL[i] for i in pv.argmax(1)]}).to_csv("artifacts/protouch_oos_2024.csv",index=False)
pd.DataFrame({"game_id":te.game_id,"season":te.season,"week":te.week,"actual":te.protouch_result,"p_l":pt[:,0],"p_d":pt[:,1],"p_v":pt[:,2],"pick":[CL[i] for i in pt.argmax(1)]}).to_csv("artifacts/protouch_oos_2025.csv",index=False)
Path("artifacts/protouch_multiclass_0.1.0.json").write_text(json.dumps(meta,indent=2))
print(json.dumps(meta,indent=2))
