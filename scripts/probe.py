"""Test ponctuel de sources de données handball (lancé à la main depuis GitHub Actions)."""
import json, os
from pathlib import Path
import requests

KEY = os.environ.get("API_FOOTBALL_KEY", "")
out = {}
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36"}
def get(name, url, headers=None, params=None, keep=None):
    try:
        r = requests.get(url, headers=headers or {}, params=params, timeout=40)
        info = {"status": r.status_code, "bytes": len(r.content)}
        try:
            j = r.json(); info["keys"] = list(j)[:15] if isinstance(j, dict) else type(j).__name__
            if keep: info["keep"] = keep(j)
        except Exception:
            info["text"] = r.text[:300]
        out[name] = info
    except Exception as e:
        out[name] = {"error": repr(e)[:300]}

get("apisports_date", "https://v1.handball.api-sports.io/games", {"x-apisports-key": KEY}, {"date": "2026-10-24"},
    keep=lambda j: {"errors": j.get("errors"), "results": j.get("results"), "s": [(g["league"]["name"], g["teams"]["home"]["name"], g["teams"]["away"]["name"]) for g in j.get("response", [])[:5]]})
def sofa(j):
    ev = j.get("events", [])
    return {"n": len(ev), "s": [(e.get("tournament", {}).get("name"), e.get("homeTeam", {}).get("name"), e.get("awayTeam", {}).get("name"),
                                 (e.get("homeScore") or {}).get("current"), (e.get("awayScore") or {}).get("current"), (e.get("homeScore") or {}).get("period1")) for e in ev[:8]]}
get("sofa_today", "https://api.sofascore.com/api/v1/sport/handball/scheduled-events/2026-10-24", UA, keep=sofa)
get("sofa_past", "https://api.sofascore.com/api/v1/sport/handball/scheduled-events/2025-11-15", UA, keep=sofa)
get("sofa_www", "https://www.sofascore.com/api/v1/sport/handball/scheduled-events/2026-10-24", UA, keep=sofa)
def espn(j):
    return {"leagues": [l.get("name") for l in j.get("leagues", [])][:5], "events": len(j.get("events", []))}
get("espn_handball", "https://site.api.espn.com/apis/site/v2/sports/handball/scoreboard", UA, keep=espn)
get("lnh", "https://www.lnh.fr/liquimoly-starligue/calendrier", UA)
get("wiki", "https://en.wikipedia.org/w/api.php", UA, {"action": "parse", "page": "2025–26 LNH Division 1", "prop": "sections", "format": "json"},
    keep=lambda j: [s.get("line") for s in j.get("parse", {}).get("sections", [])][:20])
Path("data/probe").mkdir(parents=True, exist_ok=True)
Path("data/probe/handball.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
print("ok")
