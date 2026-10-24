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

call("status", "/status")
call("leagues_fr", "/leagues", country="France")
call("leagues_de", "/leagues", country="Germany")
call("leagues_eu", "/leagues", search="Champions")
Path("data/probe").mkdir(parents=True, exist_ok=True)
Path("data/probe/handball.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(json.dumps({k: {kk: v.get(kk) for kk in ("status", "errors", "results")} for k, v in out.items()}, indent=1))
