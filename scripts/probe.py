"""Test ponctuel : API-Sports basket (ligues du jour, noms et format des marchés de cotes)."""
import datetime as dt, json, os, time
from collections import Counter
from pathlib import Path
from zoneinfo import ZoneInfo
import requests

H = {"x-apisports-key": os.environ.get("API_FOOTBALL_KEY", "")}
B = "https://v1.basketball.api-sports.io"
today = dt.datetime.now(ZoneInfo("Europe/Paris")).date()


def get(path, **p):
    j = requests.get(B + path, headers=H, params=p, timeout=40).json()
    return j


out = {"status": get("/status").get("response")}
games = []
for d in (today, today + dt.timedelta(days=1)):
    j = get("/games", date=d.isoformat(), timezone="Europe/Paris")
    out.setdefault("errors", []).append(j.get("errors"))
    games += j.get("response", [])
c = Counter(f"{g['league']['id']}|{g['league']['name']}|{g['country']['name']}|{g['league'].get('season')}" for g in games)
out["par_ligue"] = c.most_common(80)
out["exemple_match"] = games[0] if games else None
picks = []
for test in (lambda g: g["league"]["name"] == "NBA", lambda g: g["league"]["name"].lower().startswith("euroleague"),
             lambda g: g["country"]["name"] == "France" and any(k in g["league"]["name"] for k in ("LNB", "Elite", "Pro A"))):
    gl = [g for g in games if test(g) and g["status"]["short"] == "NS"]
    if gl:
        picks.append(gl[0])
out["cotes"] = {}
for g in picks:
    time.sleep(7)
    r = get("/odds", game=g["id"]).get("response", [])
    if not r:
        out["cotes"][g["league"]["name"]] = "aucune cote"
        continue
    bks = r[0].get("bookmakers", [])
    out["cotes"][g["league"]["name"]] = {
        "match": g["teams"]["home"]["name"] + " - " + g["teams"]["away"]["name"],
        "bookmakers": [b["name"] for b in bks],
        "marches": sorted({(bt["id"], bt["name"]) for b in bks for bt in b["bets"]}),
        "premier_bookmaker": {bt["name"]: bt["values"][:20] for bt in bks[0]["bets"]} if bks else None,
    }
Path("data/probe").mkdir(parents=True, exist_ok=True)
Path("data/probe/basket.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
print("ok", len(games), "matchs")
