"""Test ponctuel : noms et format des marchés de cotes API-Football."""
import json, os
from pathlib import Path
import requests
H = {"x-apisports-key": os.environ.get("API_FOOTBALL_KEY", "")}
B = "https://v3.football.api-sports.io"
fx = requests.get(B + "/fixtures", headers=H, params={"date": "2026-10-25"}, timeout=40).json()
fl = [f for f in fx.get("response", []) if f["league"]["id"] == 135]
fid = fl[0]["fixture"]["id"]
o = requests.get(B + "/odds", headers=H, params={"fixture": fid}, timeout=40).json()
r = o["response"][0]
out = {"fixture": fid, "teams": fl[0]["teams"]["home"]["name"] + " - " + fl[0]["teams"]["away"]["name"], "update": r.get("update")}
bks = r["bookmakers"]
out["all_bets"] = sorted({(bt["id"], bt["name"]) for b in bks for bt in b["bets"]})
pin = [b for b in bks if b["name"] == "Pinnacle"]
out["pinnacle_bets"] = {bt["name"]: bt["values"][:12] for bt in (pin[0]["bets"] if pin else [])}
b365 = [b for b in bks if b["name"] == "Bet365"][0]
keep = ("Match Winner", "Both Teams Score", "Goals Over/Under", "Home Team Score a Goal", "Away Team Score a Goal",
        "Total - Home", "Total - Away", "Anytime Goal Scorer", "Double Chance")
out["bet365"] = {bt["name"]: bt["values"][:15] for bt in b365["bets"] if bt["name"] in keep}
Path("data/probe/odds_foot.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
print("ok")
