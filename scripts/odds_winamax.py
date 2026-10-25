"""Cotes Winamax et des autres opérateurs français (The Odds API, offre gratuite 500 crédits/mois).

Une fois par jour (passage du matin), pour les grands championnats de foot et les tournois de tennis
couverts : cote Winamax et meilleure cote française sur la victoire (et le total 2,5 buts en foot ; vainqueur prolongation incluse en basket NBA et EuroLeague).
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
# Priorité 1 : grands championnats, victoire + total 2,5 buts
TOP = {"soccer_france_ligue_one": "F1", "soccer_epl": "E0", "soccer_spain_la_liga": "SP1", "soccer_italy_serie_a": "I1",
       "soccer_germany_bundesliga": "D1", "soccer_uefa_champs_league": None, "soccer_uefa_europa_league": None}
# Priorité 3 : autres championnats suivis, victoire seulement (si l'API les couvre)
SECOND = {"soccer_france_ligue_two": "F2", "soccer_efl_champ": "E1", "soccer_netherlands_eredivisie": "N1",
          "soccer_portugal_primeira_liga": "P1", "soccer_belgium_first_div": "B1", "soccer_germany_bundesliga2": "D2",
          "soccer_italy_serie_b": "I2", "soccer_spain_segunda_division": "SP2", "soccer_spl": "SC0",
          "soccer_turkey_super_league": "T1", "soccer_greece_super_league": "G1", "soccer_england_league1": "E2",
          "soccer_england_league2": "E3", "soccer_uefa_europa_conference_league": None}
FOOT = {**TOP, **SECOND}
MAX_CREDITS_DAY = 30          # plafond ; le vrai budget = crédits restants du mois / jours restants

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
    import calendar
    now = dt.datetime.now(dt.timezone.utc)
    prev = OUT / "fr_odds.json"
    if prev.exists() and not os.environ.get("FORCER"):
        try:
            last = pd.Timestamp(json.loads(prev.read_text())["recupere_le"]).tz_convert(PARIS).date()
        except (ValueError, KeyError):
            last = None
        if last == now.astimezone(PARIS).date():
            print("cotes françaises : déjà relevées aujourd'hui, rien à faire")
            return 0
    sports, left = get("/sports")                          # gratuit
    if not sports:
        return 1
    active = {s["key"] for s in sports if s.get("active")}
    # budget du jour : crédits restants répartis sur les jours restants du mois
    today = now.astimezone(PARIS).date()
    days_left = calendar.monthrange(today.year, today.month)[1] - today.day + 1
    budget = min(MAX_CREDITS_DAY, int(int(left or 0) / max(days_left, 1))) if left else 12
    others = sorted(k for k in active if k.startswith(("handball", "basketball")))
    print(f"  crédits restants : {left}, budget du jour : {budget} ; autres sports disponibles : {', '.join(others) or 'aucun'}")
    plan = [(k, "h2h,totals") for k in TOP if k in active]
    plan += [(s["key"], "h2h") for s in sports if s.get("active") and s["key"].startswith(("tennis_atp", "tennis_wta"))]
    plan += [(k, "h2h") for k in ("basketball_nba", "basketball_euroleague") if k in active]
    plan += [(k, "h2h") for k in SECOND if k in active]
    plan += [(k, "h2h") for k in active if k.startswith("handball")]
    out, credits = [], 0
    for sk, markets in plan:
        events, left = get(f"/sports/{sk}/events")          # gratuit
        soon = [e for e in (events or []) if 0 <= (pd.Timestamp(e["commence_time"]) - pd.Timestamp(now)).total_seconds() <= 36 * 3600]
        if not soon:
            continue
        cost = len(markets.split(","))
        if credits + cost > budget:
            print(f"  budget du jour atteint avant {sk}")
            continue
        data, left = get(f"/sports/{sk}/odds", regions="fr", markets=markets, oddsFormat="decimal",
                         bookmakers=",".join(FR_BOOKS))
        credits += cost
        for e in data or []:
            if pd.Timestamp(e["commence_time"]) <= pd.Timestamp(now):
                continue                    # match commencé : cotes en direct, sans intérêt ici
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


def lookup(prefix: str | None = None) -> dict:
    """Index pour model.py et tennis.py : (date, équipe/joueur normalisés) → cotes par pari.
    prefix : ne garder qu'un sport (ex. "basketball") ; par défaut le basket est exclu, pour ne pas
    confondre un club de foot et sa section basket (Real Madrid, Olympiacos…)."""
    f = OUT / "fr_odds.json"
    if not f.exists():
        return {}
    data = json.loads(f.read_text())
    idx = {}
    for m in data.get("matchs", []):
        sk = m.get("sport_key", "")
        if (prefix and not sk.startswith(prefix)) or (not prefix and sk.startswith("basketball")):
            continue
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
