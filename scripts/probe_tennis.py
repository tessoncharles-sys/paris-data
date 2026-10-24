"""Test ponctuel des sources tennis (lancé à la main depuis GitHub Actions)."""
import json, datetime as dt
from pathlib import Path
import requests

H = {"User-Agent": "paris-data/1.0"}
out = {}
def get(u):
    r = requests.get(u, headers=H, timeout=40)
    return r.status_code, (r.json() if r.status_code == 200 else None)

st, j = get("https://site.api.espn.com/apis/site/v2/sports/tennis/atp/scoreboard")
ev = j["events"]
out["event_keys"] = {e["name"]: {k: (e[k] if k not in ("groupings", "competitions") else "...") for k in e} for e in ev}
fin, sched = None, None
for e in ev:
    for g in e.get("groupings", []):
        if "Singles" not in g["grouping"]["displayName"]:
            continue
        for c in g["competitions"]:
            n = c["status"]["type"]["name"]
            if n == "STATUS_FINAL" and fin is None: fin = c
            if n == "STATUS_SCHEDULED" and sched is None: sched = c
out["final_example"] = fin
out["scheduled_example"] = sched
# historique : quelques dates passées
hist = {}
for d in ["20260601", "20260115", "20250601", "20250115", "20240601", "20261020"]:
    for tour in ("atp", "wta"):
        s, jj = get(f"https://site.api.espn.com/apis/site/v2/sports/tennis/{tour}/scoreboard?dates={d}")
        if jj is None:
            hist[f"{tour}_{d}"] = s; continue
        sing = [c for e in jj.get("events", []) for g in e.get("groupings", []) if "Singles" in g["grouping"]["displayName"] for c in g["competitions"]]
        dates = sorted({c["date"][:10] for c in sing})
        hist[f"{tour}_{d}"] = {"events": [e["name"] for e in jj.get("events", [])], "singles": len(sing),
                               "date_min": dates[0] if dates else None, "date_max": dates[-1] if dates else None}
out["history"] = hist
# endpoint "summary" d'un match, pour voir les stats
if fin:
    s, jj = get(f"https://site.api.espn.com/apis/site/v2/sports/tennis/atp/summary?event={fin['id']}")
    out["summary_status"] = s
    out["summary_keys"] = list(jj) if jj else None
Path("data/probe").mkdir(parents=True, exist_ok=True)
Path("data/probe/tennis.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
print("ok")
