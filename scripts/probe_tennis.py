"""Test ponctuel des sources tennis (lancé à la main depuis GitHub Actions)."""
import io, json, datetime as dt
from pathlib import Path
import requests, pandas as pd

out = {}
H = {"User-Agent": "paris-data/1.0"}
today = dt.date.today()
def probe(name, url, kind):
    try:
        r = requests.get(url, headers=H, timeout=40)
        info = {"url": url, "status": r.status_code, "bytes": len(r.content)}
        if r.status_code == 200 and kind == "xlsx":
            df = pd.read_excel(io.BytesIO(r.content))
            info.update(rows=len(df), columns=list(map(str, df.columns)), last=df.tail(2).astype(str).to_dict("records"))
        elif r.status_code == 200 and kind == "csv":
            df = pd.read_csv(io.BytesIO(r.content))
            info.update(rows=len(df), columns=list(df.columns)[:60], last_date=str(df.iloc[-1, 5]) if len(df) else None)
        elif r.status_code == 200 and kind == "json":
            j = r.json()
            ev = j.get("events", [])
            info.update(top_keys=list(j)[:20], events=len(ev))
            if ev:
                e = ev[0]
                info["event0"] = {k: e.get(k) for k in ("name", "date", "shortName")}
                grp = e.get("groupings") or []
                comps = []
                for g in grp:
                    for c in g.get("competitions", [])[:3]:
                        comps.append({"grouping": g.get("grouping", {}).get("displayName"),
                                      "date": c.get("date"), "status": c.get("status", {}).get("type", {}).get("name"),
                                      "players": [ (x.get("athlete") or {}).get("displayName") for x in c.get("competitors", [])],
                                      "keys": list(c)[:30]})
                info["sample_competitions"] = comps[:6]
                info["n_competitions"] = sum(len(g.get("competitions", [])) for g in grp)
        out[name] = info
    except Exception as e:
        out[name] = {"url": url, "error": repr(e)[:300]}

for y in (2025, 2026):
    probe(f"td_atp_{y}", f"http://www.tennis-data.co.uk/{y}/{y}.xlsx", "xlsx")
    probe(f"td_wta_{y}", f"http://www.tennis-data.co.uk/{y}w/{y}.xlsx", "xlsx")
probe("sackmann_atp_2024", "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_2024.csv", "csv")
d = today.strftime("%Y%m%d")
probe("espn_atp", f"https://site.api.espn.com/apis/site/v2/sports/tennis/atp/scoreboard?dates={d}", "json")
probe("espn_wta", f"https://site.api.espn.com/apis/site/v2/sports/tennis/wta/scoreboard?dates={d}", "json")
probe("espn_atp_nodate", "https://site.api.espn.com/apis/site/v2/sports/tennis/atp/scoreboard", "json")
Path("data/probe").mkdir(parents=True, exist_ok=True)
Path("data/probe/tennis.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
print(json.dumps({k: {kk: v.get(kk) for kk in ("status", "rows", "events", "n_competitions", "error")} for k, v in out.items()}, indent=1))
