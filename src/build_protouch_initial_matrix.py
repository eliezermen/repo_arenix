"""Build leakage-controlled Protouch Initial 9-class dataset from nflverse PBP.
Classes: L1..L4, V1..V4, ST. 2026 excluded from historical tuning.
"""
import io,os,json,urllib.request
from pathlib import Path
import pandas as pd
SEASONS=[2021,2022,2023,2024,2025]
URL="https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet"
OUT="artifacts/protouch_initial_matrix_2021_2025.csv"
Path("artifacts").mkdir(exist_ok=True)
allrows=[]
for season in SEASONS:
    url=URL.format(season=season)
    req=urllib.request.Request(url,headers={"User-Agent":"Arenix/1.0"})
    with urllib.request.urlopen(req,timeout=90) as r: raw=r.read()
    p=Path(f"artifacts/pbp_{season}.parquet");p.write_bytes(raw)
    cols=["game_id","season","week","season_type","home_team","away_team","qtr","touchdown","td_team","play_id"]
    d=pd.read_parquet(p,columns=cols)
    d=d[d.season_type.eq("REG")].copy()
    for gid,g in d.groupby("game_id",sort=False):
        g=g.sort_values("play_id")
        base=g.iloc[0]; home=str(base.home_team); away=str(base.away_team)
        td=g[(pd.to_numeric(g.touchdown,errors="coerce")==1)&g.td_team.notna()]
        if td.empty: label="ST"; q=None; team=None
        else:
            z=td.iloc[0]; q=int(z.qtr) if pd.notna(z.qtr) else None; team=str(z.td_team)
            if q not in (1,2,3,4): label="ST"
            elif team==home: label=f"L{q}"
            elif team==away: label=f"V{q}"
            else: continue
        allrows.append({"season":season,"week":int(base.week),"game_id":gid,"home_team":home,"away_team":away,
                        "first_td_team":team,"first_td_qtr":q,"target":label})
o=pd.DataFrame(allrows)
o.to_csv(OUT,index=False)
meta={"rows":len(o),"seasons":SEASONS,"classes":o.target.value_counts().to_dict(),
      "target":["L1","L2","L3","L4","V1","V2","V3","V4","ST"],
      "leakage_guard":"2026 excluded; labels derived only from each historical game's PBP."}
Path("artifacts/protouch_initial_matrix_meta.json").write_text(json.dumps(meta,indent=2))
print(json.dumps(meta,indent=2))
