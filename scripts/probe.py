"""Test ponctuel de l'API handball par date (lancé à la main depuis GitHub Actions)."""
import json, os
from collections import Counter
from pathlib import Path
import requests

H = {"x-apisports-key": os.environ.get("API_FOOTBALL_KEY", "")}
B = "https://v1.handball.api-sports.io"
out = {}
r = requests.get(B + "/games", headers=H, params={"date": "2026-10-24", "timezone": "Europe/Paris"}, timeout=40).json()
g = r.get("response", [])
out["par_ligue"] = Counter(f'{x["league"]["id"]}|{x["league"]["name"]}|{x["country"]["name"]}' for x in g).most_common(60)
nantes = [x for x in g if "Nantes" in x["teams"]["home"]["name"] or "Nantes" in x["teams"]["away"]["name"]]
out["exemple"] = nantes[0] if nantes else (g[0] if g else None)
gid = (nantes[0] if nantes else g[0])["id"] if g else None
r2 = requests.get(B + "/odds", headers=H, params={"game": gid}, timeout=40).json()
out["odds"] = {"errors": r2.get("errors"), "results": r2.get("results"), "sample": json.dumps(r2.get("response", [])[:1])[:3000]}
r3 = requests.get(B + "/games", headers=H, params={"date": "2025-11-15"}, timeout=40).json()
out["passe"] = {"errors": r3.get("errors"), "results": r3.get("results")}
r4 = requests.get(B + "/status", headers=H, timeout=40).json()
out["status"] = r4.get("response")
Path("data/probe").mkdir(parents=True, exist_ok=True)
Path("data/probe/handball.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
print("ok")
