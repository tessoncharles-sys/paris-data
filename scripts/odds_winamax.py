"""Cotes Winamax et des autres opérateurs français (The Odds API, offre gratuite 500 crédits/mois).

Une fois par jour (passage du matin), pour les grands championnats de foot et les tournois de tennis
couverts : cote Winamax et meilleure cote française sur la victoire (et le total 2,5 buts en foot).
Comparées à la cote juste, elles permettent de signaler automatiquement une value Winamax.

Budget : la liste des compétitions et des matchs est gratuite ; chaque appel de cotes coûte
1 crédit par marché. On n'appelle que les compétitions qui ont un match dans les 36 heures.
"""
from __future__ import annotations

import datetime as dt
import difflib
import json
import os
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "raw" / "odds-api"
BASE = "https://api.the-odds-api.com/v4"
PARIS = ZoneInfo("Europe/Paris")
FR_BOOKS = {"winamax_fr": "Winamax", "betclic_fr": "Betclic", "unibet_fr": "Unibet", "pmu_fr": "PMU",
            "parionssport_fr": "Parions Sport", "netbet_fr": "NetBet"}
FOOT = {"soccer_france_ligue_one": "F1", "soccer_epl": "E0", "soccer_spain_la_liga": "SP1", "soccer_italy_serie_a": "I1",
        "soccer_germany_bundesliga": "D1", "soccer_france_ligue_two": "F2", "soccer_uefa_champs_league": None,
        "soccer_uefa_europa_league": None}
MAX_CREDITS_RUN = 16          # ~480 crédits par mois avec un passage par jour


def get(path, **params):
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        raise RuntimeError("clé absente (secret ODDS_API_KEY)")
    r = requests.get(BASE + path, params={"apiKey": key, **params}, timeout=40)
    left = r.headers.get("x-requests-remaining")
    if r.status_code != 200:
        print("  erreur", r.status_code, path, r.text[:200])
        return None, left
    return r.json(), left


def norm(s: str) -> str:
    import unicodedata
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return " ".join(s.replace("-", " ").replace(".", " ").split())


def main() -> int:
    now = dt.datetime.now(dt.timezone.utc)
    sports, left = get("/sports")
    if not sports:
        return 1
    active = {s["key"] for s in sports if s.get("active")}
    wanted = [k for k in FOOT if k in active]
    wanted += [s["key"] for s in sports if s.get("active") and s["key"].startswith(("tennis_atp", "tennis_wta"))]
    out, credits = [], 0
    for sk in wanted:
        events, left = get(f"/sports/{sk}/events")          # gratuit
        soon = [e for e in (events or []) if 0 <= (pd.Timestamp(e["commence_time"]) - pd.Timestamp(now)).total_seconds() <= 36 * 3600]
        if not soon:
            continue
        markets = "h2h,totals" if sk.startswith("soccer") else "h2h"
        cost = len(markets.split(","))
        if credits + cost > MAX_CREDITS_RUN:
            print("  budget du passage atteint")
            break
        data, left = get(f"/sports/{sk}/odds", regions="fr", markets=markets, oddsFormat="decimal",
                         bookmakers=",".join(FR_BOOKS))
        credits += cost
        for e in data or []:
            row = {"sport_key": sk, "div": FOOT.get(sk), "home": e["home_team"], "away": e["away_team"],
                   "coup_envoi": e["commence_time"], "cotes": {}}
            for bk in e.get("bookmakers", []):
                name = FR_BOOKS.get(bk["key"])
                if not name:
                    continue
                for m in bk.get("markets", []):
                    for o in m.get("outcomes", []):
                        if m["key"] == "h2h":
                            lab = "N" if o["name"] == "Draw" else ("1" if o["name"] == e["home_team"] else "2")
                            if sk.startswith("tennis"):
                                lab = "Vainqueur : " + o["name"]
                        elif m["key"] == "totals" and o.get("point") == 2.5:
                            lab = ("+" if o["name"] == "Over" else "-") + "2.5 buts"
                        else:
                            continue
                        row["cotes"].setdefault(lab, {})[name] = o["price"]
            out.append(row)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fr_odds.json").write_text(json.dumps({"recupere_le": now.isoformat(timespec="minutes"), "matchs": out,
                                                  "credits_restants": left}, ensure_ascii=False))
    print(f"cotes françaises : {len(out)} matchs, {credits} crédits utilisés, {left} restants ce mois-ci")
    return 0


def lookup() -> dict:
    """Index pour model.py et tennis.py : (date, équipe/joueur normalisés) → cotes par pari."""
    f = OUT / "fr_odds.json"
    if not f.exists():
        return {}
    data = json.loads(f.read_text())
    idx = {}
    for m in data.get("matchs", []):
        day = pd.Timestamp(m["coup_envoi"]).tz_convert(PARIS).strftime("%Y-%m-%d")
        idx.setdefault(day, []).append(m)
    return idx


def match_odds(idx: dict, day: str, home: str, away: str, expand=norm) -> dict | None:
    """Retrouve un match par la ressemblance des deux noms (les sources n'écrivent pas les noms pareil)."""
    best, score = None, 0.0
    h, a = expand(home), expand(away)
    for m in idx.get(day, []):
        sh = difflib.SequenceMatcher(None, h, norm(m["home"])).ratio()
        sa = difflib.SequenceMatcher(None, a, norm(m["away"])).ratio()
        th, ta = set(h.split()), set(a.split())
        if th & set(norm(m["home"]).split()):
            sh = max(sh, 0.9)
        if ta & set(norm(m["away"]).split()):
            sa = max(sa, 0.9)
        sc = min(sh, sa)
        if sc > score:
            best, score = m, sc
    return best if score >= 0.75 else None


if __name__ == "__main__":
    sys.exit(main())
