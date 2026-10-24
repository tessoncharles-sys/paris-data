"""Test ponctuel : cotes API-Football pour un match du jour (offre gratuite)."""
import json, os
from pathlib import Path
import requests
H = {"x-apisports-key": os.environ.get("API_FOOTBALL_KEY", "")}
B = "https://v3.football.api-sports.io"
out = {}
fx = requests.get(B + "/fixtures", headers=H, params={"date": "2026-10-25", "league": 61, "season": 2026}, timeout=40).json()
out["fixtures_l1"] = {"errors": fx.get("errors"), "results": fx.get("results")}
if not fx.get("results"):
    fx = requests.get(B + "/fixtures", headers=H, params={"date": "2026-10-25"}, timeout=40).json()
    fl = [f for f in fx.get("response", []) if f["league"]["id"] in (39, 140, 135, 78, 61)]
else:
    fl = fx["response"]
fid = fl[0]["fixture"]["id"] if fl else None
out["fixture"] = fl[0]["teams"] if fl else None
o = requests.get(B + "/odds", headers=H, params={"fixture": fid}, timeout=40).json()
out["odds"] = {"errors": o.get("errors"), "results": o.get("results")}
if o.get("response"):
    bks = o["response"][0]["bookmakers"]
    out["bookmakers"] = [b["name"] for b in bks]
    out["bets"] = sorted({bt["name"] for b in bks for bt in b["bets"]})[:80]
o2 = requests.get(B + "/odds", headers=H, params={"date": "2026-10-25", "page": 1}, timeout=40).json()
out["odds_by_date"] = {"errors": o2.get("errors"), "results": o2.get("results"), "paging": o2.get("paging")}
st = requests.get(B + "/status", headers=H, timeout=40).json()
out["status"] = st.get("response", {}).get("requests")
Path("data/probe").mkdir(parents=True, exist_ok=True)
Path("data/probe/odds_foot.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
print("ok")
