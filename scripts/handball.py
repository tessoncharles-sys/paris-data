"""Handball : collecte (API-Sports, offre gratuite) + cotes justes issues du consensus des bookmakers.

L'offre gratuite ne donne que les matchs d'hier, d'aujourd'hui et de demain, mais avec les cotes
de plusieurs bookmakers. Comme nos tests foot l'ont montré, le marché est la meilleure référence :
on en tire, marge retirée, une loi de probabilité de l'écart et du total de buts (loi normale
ajustée sur les lignes de handicap et de total), puis tous les marchés dérivés
(total d'une équipe, handicaps, 1re mi-temps…), utiles pour juger les cotes boostées.

Chaque jour, les résultats de la veille et les paramètres prédits sont archivés pour pouvoir
vérifier la fiabilité de l'approche quand l'historique sera suffisant.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from scipy.optimize import least_squares
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "handball"
OUT = ROOT / "data"
BASE = "https://v1.handball.api-sports.io"
PARIS = ZoneInfo("Europe/Paris")
MAX_ODDS_CALLS = 80          # quota gratuit : 100 requêtes par jour

# Championnats suivis, par ordre de priorité pour les cotes
LEAGUES = {
    34: "France - Starligue", 39: "Allemagne - Bundesliga", 131: "Ligue des champions EHF",
    145: "Ligue européenne EHF", 103: "Espagne - Liga ASOBAL", 23: "Danemark - Herre Handbold Ligaen",
    36: "France - Proligue", 43: "Allemagne - 2. Bundesliga", 49: "Hongrie - NB I", 78: "Pologne - Superliga",
    84: "Portugal - Andebol 1", 113: "Suède - Handbollsligan", 75: "Norvège - Elite", 115: "Suisse - NLA",
    2: "Autriche - HLA", 10: "Croatie - Premijer liga", 100: "Slovénie - 1. NLB Liga", 87: "Roumanie - Liga Nationala",
    147: "Coupe d'Europe EHF", 177: "Championnat d'Europe", 153: "Championnat du monde",
    132: "Ligue des champions EHF féminine", 154: "Championnat du monde féminin",
}


def api(path: str, **params):
    key = os.environ.get("API_FOOTBALL_KEY")
    if not key:
        raise RuntimeError("clé API absente (secret API_FOOTBALL_KEY)")
    for attempt in range(3):
        try:
            r = requests.get(BASE + path, headers={"x-apisports-key": key}, params=params, timeout=40)
            j = r.json()
            if j.get("errors"):
                print("  erreur API", path, params, j["errors"])
                return []
            return j.get("response", [])
        except (requests.RequestException, ValueError) as e:
            print("  erreur", e)
            time.sleep(2 * (attempt + 1))
    return []


# ------------------------------------------------------------------ collecte

def collect() -> tuple[list, dict]:
    today = dt.datetime.now(PARIS).date()
    games = []
    for d in (today - dt.timedelta(days=1), today, today + dt.timedelta(days=1)):
        games += api("/games", date=d.isoformat(), timezone="Europe/Paris")
    games = [g for g in games if g["league"]["id"] in LEAGUES]
    RAW.mkdir(parents=True, exist_ok=True)

    # résultats terminés → historique
    done = [g for g in games if g["status"]["short"] in ("FT", "AET", "AP")]
    rows = [{"id": g["id"], "date": g["date"], "league_id": g["league"]["id"], "league": LEAGUES[g["league"]["id"]],
             "home": g["teams"]["home"]["name"], "away": g["teams"]["away"]["name"],
             "hs": g["scores"]["home"], "as": g["scores"]["away"],
             "h1": (g["periods"]["first"] or {}).get("home"), "a1": (g["periods"]["first"] or {}).get("away")}
            for g in done]
    hist_f = RAW / "results.csv"
    old = pd.read_csv(hist_f) if hist_f.exists() else pd.DataFrame()
    allr = pd.concat([old, pd.DataFrame(rows)], ignore_index=True)
    if len(allr):
        allr.drop_duplicates(subset=["id"], keep="last").sort_values("date").to_csv(hist_f, index=False)

    # cotes des matchs à venir, par priorité de championnat
    upcoming = [g for g in games if g["status"]["short"] == "NS"]
    order = {lid: i for i, lid in enumerate(LEAGUES)}
    upcoming.sort(key=lambda g: (order[g["league"]["id"]], g["date"]))
    odds = {}
    for g in upcoming[:MAX_ODDS_CALLS]:
        resp = api("/odds", game=g["id"])
        if resp:
            odds[str(g["id"])] = resp[0].get("bookmakers", [])
        time.sleep(0.2)
    (RAW / "upcoming.json").write_text(json.dumps(upcoming, ensure_ascii=False))
    (RAW / "odds.json").write_text(json.dumps(odds, ensure_ascii=False))
    print(f"handball : {len(games)} matchs suivis, {len(done)} résultats, {len(upcoming)} à venir, cotes pour {len(odds)}")
    return upcoming, odds


# ------------------------------------------------------------------ lecture des cotes

def devig(odds: list[float]) -> list[float] | None:
    if any(o is None or o <= 1 for o in odds):
        return None
    inv = [1 / o for o in odds]
    s = sum(inv)
    return [x / s for x in inv]


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def constraints(bookmakers: list, home: str, away: str) -> dict:
    """Probabilités marge retirée, moyennées entre bookmakers :
    écart (domicile - extérieur) > ligne, total > ligne, idem 1re mi-temps."""
    acc = {"d": {}, "t": {}, "d1": {}, "t1": {}, "draw": [], "draw1": []}
    names = []
    for bk in bookmakers:
        names.append(bk.get("name"))
        for bet in bk.get("bets", []):
            name, vals = bet.get("name", ""), bet.get("values", [])
            v = {str(x.get("value")): num(x.get("odd")) for x in vals}
            if name in ("3Way Result", "1st Half 3Way Result"):
                p = devig([v.get("Home"), v.get("Draw"), v.get("Away")])
                if p:
                    k = "d" if name == "3Way Result" else "d1"
                    acc[k].setdefault(0.5, []).append(p[0])
                    acc[k].setdefault(-0.5, []).append(p[0] + p[1])
                    acc["draw" if k == "d" else "draw1"].append(p[1])
            elif name in ("Over/Under", "Over/Under 1st Half"):
                k = "t" if name == "Over/Under" else "t1"
                for key, o in v.items():
                    m = re.match(r"Over ([\d.]+)$", key)
                    if m:
                        line = float(m.group(1))
                        p = devig([o, v.get(f"Under {m.group(1)}")])
                        if p and line % 1 == 0.5:
                            acc[k].setdefault(line, []).append(p[0])
            elif name == "Asian Handicap":
                for key, o in v.items():
                    m = re.match(r"Home ([+-]?[\d.]+)$", key)
                    if m:
                        h = float(m.group(1))
                        other = v.get(f"Away {m.group(1)}")
                        p = devig([o, other])
                        if p and abs(h) % 1 == 0.5:
                            # domicile gagne le pari si écart + h > 0, soit écart > -h
                            acc["d"].setdefault(-h, []).append(p[0])
            elif name == "Handicap Result":
                hs = {re.match(r"\w+ ([+-]?\d+)$", key).group(1) for key in v if re.match(r"\w+ ([+-]?\d+)$", key)}
                for hcp in hs:
                    p = devig([v.get(f"Home {hcp}"), v.get(f"Draw {hcp}"), v.get(f"Away {hcp}")])
                    if p:
                        k = -float(hcp)          # domicile -3 : gagne si écart > 3
                        acc["d"].setdefault(k + 0.5, []).append(p[0])
    out = {k: ({line: float(np.mean(ps)) for line, ps in v.items()} if isinstance(v, dict) else
               (float(np.mean(v)) if v else None)) for k, v in acc.items()}
    out["bookmakers"] = sorted({n for n in names if n})
    return out


def fit_normal(points: dict, mu0: float, sd0: float, sd_bounds=(2.5, 14.0)) -> tuple[float, float] | None:
    """Loi normale (moyenne, écart-type) qui reproduit au mieux P(X > ligne)."""
    if not points:
        return None
    lines = np.array(list(points)); ps = np.clip(np.array(list(points.values())), 0.01, 0.99)
    if len(lines) < 3:
        # trop peu de lignes pour estimer la dispersion : écart-type habituel, on n'ajuste que la moyenne
        mu = least_squares(lambda x: (1 - norm.cdf(lines, x[0], sd0)) - ps, [mu0], bounds=([-60], [140])).x[0]
        return float(mu), float(sd0)

    def res(x):
        mu, sd = x
        r = (1 - norm.cdf(lines, mu, sd)) - ps
        return np.r_[r, 0.05 * (sd - sd0) / sd0]   # léger rappel vers l'écart-type habituel
    x = least_squares(res, [mu0, sd0], bounds=([-60, sd_bounds[0]], [140, sd_bounds[1]])).x
    return float(x[0]), float(x[1])


# ------------------------------------------------------------------ marchés

def p_over(line, mu, sd):
    return float(1 - norm.cdf(line, mu, sd))


def build_markets(c: dict, home: str, away: str) -> tuple[dict, dict] | None:
    d = fit_normal(c["d"], 2.0, 6.5)
    t = fit_normal(c["t"], 58.0, 6.5)
    if not d or not t:
        return None
    mu_d, sd_d = d
    mu_t, sd_t = t
    d1 = fit_normal(c["d1"], mu_d / 2, sd_d * 0.7) or (mu_d * 0.5, sd_d * 0.7)
    t1 = fit_normal(c["t1"], mu_t * 0.48, sd_t * 0.7) or (mu_t * 0.48, sd_t * 0.7)
    m = {}
    # résultat : la proba de nul vient du marché (loi normale entre -0,5 et +0,5 sinon)
    p_home = p_over(0.5, mu_d, sd_d); p_away = 1 - p_over(-0.5, mu_d, sd_d)
    p_draw = c.get("draw") if c.get("draw") is not None else 1 - p_home - p_away
    s = p_home + p_away + p_draw
    p_home, p_away, p_draw = p_home / s, p_away / s, p_draw / s
    m[f"Victoire {home}"], m["Match nul"], m[f"Victoire {away}"] = p_home, p_draw, p_away
    m[f"{home} ou nul"], m[f"{away} ou nul"] = p_home + p_draw, p_away + p_draw
    fav, sign = (home, 1) if mu_d >= 0 else (away, -1)
    for k in (2, 3, 4, 5, 6, 7, 8, 10):
        m[f"{fav} gagne par {k} buts ou plus"] = p_over(k - 0.5, sign * mu_d, sd_d)
    centre = math.floor(mu_t) + 0.5
    for line in (centre - 4, centre - 2, centre, centre + 2, centre + 4):
        m[f"+{line:g} buts"] = p_over(line, mu_t, sd_t)
        m[f"-{line:g} buts"] = 1 - p_over(line, mu_t, sd_t)
    sd_team = math.sqrt(sd_t ** 2 + sd_d ** 2) / 2
    mu_h, mu_a = (mu_t + mu_d) / 2, (mu_t - mu_d) / 2
    for team, mu in ((home, mu_h), (away, mu_a)):
        for n in range(round(mu) - 3, round(mu) + 4):
            m[f"{team} marque au moins {n} buts"] = p_over(n - 0.5, mu, sd_team)
    if c.get("d1"):
        p1h = p_over(0.5, d1[0], d1[1]); p1a = 1 - p_over(-0.5, d1[0], d1[1])
        p1d = c.get("draw1") if c.get("draw1") is not None else 1 - p1h - p1a
        s1 = p1h + p1a + p1d
        m[f"Mi-temps : {home} mène"], m["Mi-temps : égalité"], m[f"Mi-temps : {away} mène"] = p1h / s1, p1d / s1, p1a / s1
    c1 = math.floor(t1[0]) + 0.5
    for line in (c1 - 2, c1, c1 + 2):
        m[f"1re mi-temps : +{line:g} buts"] = p_over(line, *t1)
        m[f"1re mi-temps : -{line:g} buts"] = 1 - p_over(line, *t1)
    info = {"mu_d": round(mu_d, 2), "sd_d": round(sd_d, 2), "mu_t": round(mu_t, 2), "sd_t": round(sd_t, 2),
            "mu_h": round(mu_h, 1), "mu_a": round(mu_a, 1), "mu_t1": round(t1[0], 1)}
    return {k: float(min(max(v, 0.0), 1.0)) for k, v in m.items()}, info


def family(label: str) -> str:
    if label.startswith(("Victoire", "Match nul")) or label.endswith(" ou nul"):
        return "Résultat"
    if "gagne par" in label:
        return "Écart"
    if label.startswith(("Mi-temps", "1re mi-temps")):
        return "1re mi-temps"
    if "marque au moins" in label:
        return "Buts d'une équipe"
    return "Total buts"


def predict(upcoming: list, odds: dict) -> pd.DataFrame:
    rows, archive = [], []
    for g in upcoming:
        bks = odds.get(str(g["id"]))
        if not bks:
            continue
        home, away = g["teams"]["home"]["name"], g["teams"]["away"]["name"]
        c = constraints(bks, home, away)
        res = build_markets(c, home, away)
        if not res:
            continue
        mk, info = res
        t = pd.Timestamp(g["date"]).tz_convert(PARIS)
        lg = LEAGUES[g["league"]["id"]]
        archive.append({"id": g["id"], "date": g["date"], "league": lg, "home": home, "away": away, **info,
                        "bookmakers": len(c["bookmakers"])})
        books = ", ".join(c["bookmakers"][:4]) + ("…" if len(c["bookmakers"]) > 4 else "")
        base = (f"Consensus de {len(c['bookmakers'])} bookmaker(s) ({books}), marge retirée : "
                f"{info['mu_t']:.1f} buts attendus au total, {home} {info['mu_h']:.1f} – {info['mu_a']:.1f} {away} "
                f"(écart-type du total {info['sd_t']:.1f}, de l'écart {info['sd_d']:.1f}). "
                f"En handball, la cote juste vient du marché ; le calcul n'a pas encore été vérifié sur un historique "
                f"(il s'accumule chaque jour).")
        for label, p in mk.items():
            if p <= 0.02 or p >= 0.98:
                continue
            fair = 1 / p
            expl = (f"Proba {p:.0%} → cote juste {fair:.2f}. Value seulement si Winamax paie au moins {fair * 1.03:.2f} (+3 %). "
                    + base)
            rows.append({
                "sport": "Handball", "id": f"H|{g['id']}|{label}",
                "date": t.strftime("%Y-%m-%d"), "heure": t.strftime("%H:%M"),
                "championnat": lg, "match": f"{home} - {away}", "famille": family(label), "marche": label,
                "proba": round(p, 4), "cote_juste": round(fair, 2), "cote_minimum": round(fair * 1.03, 2),
                "cote_moyenne_marche": None, "source_cote": "marché", "fiabilite": "bonne" if len(c["bookmakers"]) >= 2 else "moyenne",
                "infos": f"{info['mu_t']:.1f} buts attendus · {home} {info['mu_h']:.1f} – {info['mu_a']:.1f} {away}",
                "explication": re.sub(r"(\d)\.(\d)", r"\1,\2", expl),
            })
    if archive:
        f = RAW / "predictions.csv"
        old = pd.read_csv(f) if f.exists() else pd.DataFrame()
        pd.concat([old, pd.DataFrame(archive)], ignore_index=True).drop_duplicates(subset=["id"], keep="last").to_csv(f, index=False)
    return pd.DataFrame(rows)


def main() -> int:
    if "--offline" in sys.argv:
        upcoming = json.loads((RAW / "upcoming.json").read_text())
        odds = json.loads((RAW / "odds.json").read_text())
    else:
        upcoming, odds = collect()
    df = predict(upcoming, odds)
    df.to_csv(OUT / "dashboard_handball.csv", index=False)
    print(f"handball : {df['match'].nunique() if len(df) else 0} matchs, {len(df)} cotes justes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
