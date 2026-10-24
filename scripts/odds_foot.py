"""Cotes foot des bookmakers (API-Football, offre gratuite : matchs d'aujourd'hui et de demain).

Pour chaque match du tableau, on récupère les cotes de Pinnacle et des autres bookmakers,
on retire la marge et on en tire une probabilité « marché » par pari, comparée ensuite
au modèle dans model.py. Les cotes sont archivées à chaque passage (data/raw/api-football/odds_archive/)
pour tester plus tard le modèle face au marché et suivre la cote de clôture.

Quota gratuit : 100 requêtes par jour, une par match. Trois passages par jour se le partagent.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import sys
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent))
import model as M  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
API = ROOT / "data" / "raw" / "api-football"
CACHE = API / "odds_latest.json"
ARCH = API / "odds_archive"
BASE = "https://v3.football.api-sports.io"
PARIS = ZoneInfo("Europe/Paris")
PRIORITY = ["E0", "SP1", "I1", "D1", "F1", "N1", "P1", "B1", "T1", "E1", "SC0", "D2", "I2", "SP2", "F2", "G1", "E2", "E3", "SC1"]


def api(path, **params):
    key = os.environ.get("API_FOOTBALL_KEY")
    if not key:
        raise RuntimeError("clé API absente")
    for _ in range(3):
        try:
            j = requests.get(BASE + path, headers={"x-apisports-key": key}, params=params, timeout=40).json()
            err = j.get("errors")
            if err:
                if "rateLimit" in str(err) or "Too many" in str(err):
                    time.sleep(65)
                    continue
                print("  erreur API", path, params, err)
                return None
            return j.get("response")
        except (requests.RequestException, ValueError) as e:
            print("  erreur", e)
            time.sleep(5)
    return None


def devig(odds):
    if any(o is None or o <= 1 for o in odds):
        return None
    inv = [1 / o for o in odds]
    s = sum(inv)
    return [x / s for x in inv]


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def parse_book(bets: list) -> dict:
    """Probabilités marge retirée d'un bookmaker, avec nos libellés de paris."""
    out = {}
    by = {b["name"]: {str(v["value"]): f(v["odd"]) for v in b.get("values", [])} for b in bets}
    mw = by.get("Match Winner")
    if mw:
        p = devig([mw.get("Home"), mw.get("Draw"), mw.get("Away")])
        if p:
            out.update({"1": p[0], "N": p[1], "2": p[2], "1N": p[0] + p[1], "12": p[0] + p[2], "N2": p[1] + p[2]})

    v = by.get("Goals Over/Under") or {}
    for k, o in v.items():
        m = re.match(r"Over (\d+\.5)$", k)
        if m:
            p = devig([o, v.get(f"Under {m.group(1)}")])
            if p:
                out[f"+{m.group(1)} buts"], out[f"-{m.group(1)} buts"] = p
    bt = by.get("Both Teams Score")
    if bt:
        p = devig([bt.get("Yes"), bt.get("No")])
        if p:
            out["Les 2 marquent : oui"], out["Les 2 marquent : non"] = p
    for name, lab in (("Total - Home", "Dom."), ("Total - Away", "Ext.")):
        v = by.get(name) or {}
        for line in ("0.5", "1.5"):
            p = devig([v.get(f"Over {line}"), v.get(f"Under {line}")])
            if p:
                out[f"{lab} +{line} but"] = p[0]
    return out


def summarise(bookmakers: list) -> dict:
    per, pin = {}, {}
    for bk in bookmakers:
        probs = parse_book(bk.get("bets", []))
        for lab, p in probs.items():
            per.setdefault(lab, []).append(p)
        if bk.get("name") == "Pinnacle":
            pin = probs
    return {lab: {"pin": round(pin[lab], 4) if lab in pin else None, "cons": round(float(np.mean(ps)), 4), "n": len(ps)}
            for lab, ps in per.items()}


def api_fixtures(days) -> list:
    fx = []
    for d in days:
        p = API / f"fixtures_{d.isoformat()}.json"
        if p.exists():
            try:
                fx += json.loads(p.read_text(encoding="utf-8"))
            except ValueError:
                pass
    return fx


def find_fixture(fx: list, div: str, home: str, away: str, day: dt.date):
    lid = M.DIV_LEAGUE.get(div)
    ah, aa = M.api_team(home, div), M.api_team(away, div)
    for x in fx:
        if x["league"]["id"] != lid:
            continue
        d = pd.Timestamp(x["fixture"]["date"]).tz_convert(PARIS).date()
        if d != day:
            continue
        h, a = x["teams"]["home"]["name"], x["teams"]["away"]["name"]
        if (ah and h == ah) or (aa and a == aa):
            return x
    return None


def main() -> int:
    now = dt.datetime.now(PARIS)
    today = now.date()
    fxd = M.load_fixtures(today)
    if fxd.empty:
        print("cotes foot : pas de matchs à venir")
        return 0
    fx = api_fixtures([today, today + dt.timedelta(days=1)])
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    todo = []
    for _, r in fxd.iterrows():
        x = find_fixture(fx, r["Div"], r["HomeTeam"], r["AwayTeam"], r["Date"].date())
        if x is None:
            continue
        ko = pd.Timestamp(x["fixture"]["date"]).tz_convert(PARIS)
        if ko < now:
            continue
        key = f"{r['Div']}|{r['HomeTeam']}|{r['AwayTeam']}"
        fid = str(x["fixture"]["id"])
        age = None
        if fid in cache:
            age = (now - pd.Timestamp(cache[fid]["recupere_le"]).tz_convert(PARIS)).total_seconds() / 3600
        todo.append((fid in cache, PRIORITY.index(r["Div"]) if r["Div"] in PRIORITY else 99, ko, fid, key, age))
    # d'abord les matchs sans cotes, puis par championnat et heure ; on rafraîchit les cotes de plus de 5 h
    todo.sort(key=lambda t: (t[0], t[1], t[2]))
    budget = 40 if now.hour < 9 else 22 if now.hour < 15 else 20
    st = api("/status")
    if isinstance(st, dict):
        req = st.get("requests", {})
        budget = max(0, min(budget, req.get("limit_day", 100) - req.get("current", 0) - 3))
    done = 0
    stamp = now.isoformat(timespec="minutes")
    arch_rows = []
    for cached, _, ko, fid, key, age in todo:
        if done >= budget:
            break
        if cached and age is not None and age < 5:
            continue
        resp = api("/odds", fixture=fid)
        done += 1
        time.sleep(6.5)                      # 10 requêtes par minute maximum
        if not resp:
            continue
        bks = resp[0].get("bookmakers", [])
        s = summarise(bks)
        cache[fid] = {"cle": key, "coup_envoi": ko.isoformat(), "recupere_le": stamp,
                      "bookmakers": [b["name"] for b in bks], "marches": s}
        scorers = {}
        for b in bks:
            for bt in b.get("bets", []):
                if bt["name"] == "Anytime Goal Scorer":
                    for v in bt.get("values", []):
                        o = f(v.get("odd"))
                        if o:
                            scorers.setdefault(v["value"], []).append(o)
        if scorers:
            cache[fid]["buteurs"] = {k: round(float(np.median(v)), 2) for k, v in scorers.items()}
        for lab, v in s.items():
            arch_rows.append({"recupere_le": stamp, "fixture": fid, "cle": key, "coup_envoi": ko.isoformat(),
                              "marche": lab, "pin": v["pin"], "cons": v["cons"], "n": v["n"]})
    # on ne garde en cache que les matchs encore à venir (avec une marge de 3 h)
    cache = {k: v for k, v in cache.items() if pd.Timestamp(v["coup_envoi"]).tz_convert(PARIS) > now - dt.timedelta(hours=3)}
    API.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(cache, ensure_ascii=False))
    if arch_rows:
        ARCH.mkdir(parents=True, exist_ok=True)
        fa = ARCH / f"{today:%Y-%m}.csv"
        pd.DataFrame(arch_rows).to_csv(fa, mode="a", header=not fa.exists(), index=False)
    print(f"cotes foot : {len(todo)} matchs reconnus, {done} requêtes, {len(cache)} matchs avec cotes en cache")
    return 0


if __name__ == "__main__":
    sys.exit(main())
