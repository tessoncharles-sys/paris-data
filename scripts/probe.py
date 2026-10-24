"""Test ponctuel d'une source de données (lancé à la main depuis GitHub Actions)."""
import json, os
from pathlib import Path
import requests

KEY = os.environ.get("API_FOOTBALL_KEY", "")
H = {"x-apisports-key": KEY}
B = "https://v1.handball.api-sports.io"
out = {}
def call(name, path, **params):
    try:
        r = requests.get(B + path, headers=H, params=params, timeout=40)
        j = r.json()
        out[name] = {"status": r.status_code, "errors": j.get("errors"), "results": j.get("results"),
                     "paging": j.get("paging"), "sample": (j.get("response") or [])[:3]}
    except Exception as e:
        out[name] = {"error": repr(e)[:300]}

call("leagues_all", "/leagues")
allr = []
try:
    r = requests.get(B + "/leagues", headers=H, timeout=40).json().get("response", [])
    allr = [{"id": l["id"], "name": l["name"], "type": l.get("type"), "country": (l.get("country") or {}).get("name"),
             "seasons": sorted(s0.get("season") for s0 in l.get("seasons", []))[-3:]} for l in r]
except Exception as e:
    allr = [repr(e)]
out["leagues_list"] = allr
def lid(country, *names):
    for l in allr:
        if isinstance(l, dict) and l["country"] == country and any(n.lower() in l["name"].lower() for n in names) and "Women" not in l["name"]:
            return l["id"]
fr = lid("France", "Starligue", "Division 1", "LNH")
call("games_fr_2025", "/games", league=fr, season=2025)
call("games_fr_2026", "/games", league=fr, season=2026)
call("odds_fr_2026", "/odds", league=fr, season=2026)
out["ids"] = {"fr": fr}
Path("data/probe").mkdir(parents=True, exist_ok=True)
Path("data/probe/handball.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
print("ok")
