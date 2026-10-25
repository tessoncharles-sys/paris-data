"""Basket : collecte (API-Sports basketball, offre gratuite) + cotes justes issues du marché.

Même approche que le handball : la NBA est le marché le plus efficace qui soit, on ne cherche donc
pas à le battre avec un modèle maison. On prend les cotes de Pinnacle (à défaut, le consensus des
bookmakers), marge retirée, et on en tire une loi normale de l'écart et du total de points
(ajustée sur les lignes de handicap et de total). On en déduit tous les marchés dérivés
(écarts, totaux, points d'une équipe, 1re mi-temps, combinés victoire + total), utiles pour
juger les cotes boostées et comparer les opérateurs français.

Signal calculé : back-to-back (équipe qui a joué la veille, toutes compétitions confondues).
Résultats et paramètres prédits sont archivés chaque jour pour un test de fiabilité ultérieur.
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
RAW = ROOT / "data" / "raw" / "basket"
OUT = ROOT / "data"
BASE = "https://v1.basketball.api-sports.io"
PARIS = ZoneInfo("Europe/Paris")
MAX_ODDS_CALLS = 80          # quota gratuit : 100 requêtes par jour (propre à l'API basket)

# Championnats suivis (nom et pays tels qu'écrits par API-Sports), par ordre de priorité pour les cotes.
# Écart-type habituel de l'écart et du total de points (point de départ de l'ajustement).
LEAGUES = {
    ("NBA", "USA"): ("NBA", 13.5, 19.0),
    ("Euroleague", "Europe"): ("EuroLeague", 12.0, 16.0),
    ("LNB", "France"): ("France - Betclic Élite", 12.0, 16.0),
}
# marchés conservés dans le cache des cotes (le reste alourdirait le dépôt sans servir)
KEEP_BETS = {"Home/Away", "3Way Result", "Asian Handicap", "Over/Under", "Over/Under 1st Half",
             "Asian Handicap First Half", "1st Half 3Way Result"}


def league_of(g: dict):
    return LEAGUES.get((g["league"]["name"], g["country"]["name"]))


def api(path: str, **params):
    key = os.environ.get("API_FOOTBALL_KEY")
    if not key:
        raise RuntimeError("clé API absente (secret API_FOOTBALL_KEY)")
    for attempt in range(3):
        try:
            r = requests.get(BASE + path, headers={"x-apisports-key": key}, params=params, timeout=40)
            j = r.json()
            err = j.get("errors")
            if err:
                if "rateLimit" in str(err) or "Too many" in str(err):
                    time.sleep(65)           # offre gratuite : 10 requêtes par minute
                    continue
                print("  erreur API", path, params, err)
                return []
            return j.get("response", [])
        except (requests.RequestException, ValueError) as e:
            print("  erreur", e)
            time.sleep(2 * (attempt + 1))
    return []


# ------------------------------------------------------------------ collecte

def total(score: dict | None):
    return (score or {}).get("total")


def half(score: dict | None):
    s = score or {}
    q1, q2 = s.get("quarter_1"), s.get("quarter_2")
    return q1 + q2 if q1 is not None and q2 is not None else None


def collect() -> tuple[list, dict, list]:
    today = dt.datetime.now(PARIS).date()
    allg = []
    for d in (today - dt.timedelta(days=1), today, today + dt.timedelta(days=1)):
        allg += api("/games", date=d.isoformat(), timezone="Europe/Paris")
    games = [g for g in allg if league_of(g)]
    RAW.mkdir(parents=True, exist_ok=True)

    done = [g for g in games if g["status"]["short"] in ("FT", "AOT")]
    rows = [{"id": g["id"], "date": g["date"], "league": league_of(g)[0],
             "home": g["teams"]["home"]["name"], "away": g["teams"]["away"]["name"],
             "hs": total(g["scores"]["home"]), "as": total(g["scores"]["away"]),
             "h1": half(g["scores"]["home"]), "a1": half(g["scores"]["away"]),
             "prolongation": g["status"]["short"] == "AOT"} for g in done]
    hist_f = RAW / "results.csv"
    old = pd.read_csv(hist_f) if hist_f.exists() else pd.DataFrame()
    allr = pd.concat([old, pd.DataFrame(rows)], ignore_index=True)
    if len(allr):
        allr.drop_duplicates(subset=["id"], keep="last").sort_values("date").to_csv(hist_f, index=False)

    # Trois passages par jour se partagent le quota : matin 40, midi 22, soir 20 (+ 3 appels de calendrier).
    upcoming = [g for g in games if g["status"]["short"] == "NS"]
    now = dt.datetime.now(PARIS)
    budget = 40 if now.hour < 9 else 22 if now.hour < 15 else 20
    cache_f = RAW / "odds.json"
    cache = json.loads(cache_f.read_text()) if cache_f.exists() else {}
    up_ids = {str(g["id"]) for g in upcoming}
    cache = {k: v for k, v in cache.items() if k in up_ids}
    order = {k: i for i, k in enumerate(LEAGUES)}
    # priorité : matchs sans cotes, puis championnat, puis heure du match
    upcoming.sort(key=lambda g: (str(g["id"]) in cache, order[(g["league"]["name"], g["country"]["name"])], g["date"]))
    left = min(MAX_ODDS_CALLS, budget)
    st = api("/status")
    if isinstance(st, dict):
        req = st.get("requests", {})
        left = max(0, min(left, req.get("limit_day", 100) - req.get("current", 0) - 2))
    odds = dict(cache)
    for g in upcoming[:left]:
        resp = api("/odds", game=g["id"])
        if resp:
            bks = resp[0].get("bookmakers", [])
            odds[str(g["id"])] = [{"name": b.get("name"), "bets": [bt for bt in b.get("bets", []) if bt.get("name") in KEEP_BETS]}
                                  for b in bks]
        time.sleep(6.5)                      # rester sous 10 requêtes par minute
    # calendrier de toutes les compétitions (pour repérer les back-to-back), réduit à l'essentiel
    sched = [{"date": g["date"], "teams": [g["teams"]["home"]["id"], g["teams"]["away"]["id"]]} for g in allg]
    (RAW / "upcoming.json").write_text(json.dumps(upcoming, ensure_ascii=False))
    (RAW / "odds.json").write_text(json.dumps(odds, ensure_ascii=False))
    (RAW / "calendrier.json").write_text(json.dumps(sched))
    print(f"basket : {len(games)} matchs suivis, {len(done)} résultats, {len(upcoming)} à venir, cotes pour {len(odds)}")
    return upcoming, odds, sched


# ------------------------------------------------------------------ lecture des cotes

def devig(odds: list) -> list[float] | None:
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


def constraints(bookmakers: list) -> dict:
    """Probabilités marge retirée, moyennées entre bookmakers :
    P(écart domicile - extérieur > ligne), P(total > ligne), idem 1re mi-temps.
    Lignes entières : le remboursement en cas d'égalité exacte est ignoré (approximation continue)."""
    acc = {"d": {}, "t": {}, "d1": {}, "t1": {}}
    names = []
    for bk in bookmakers:
        bets = {b.get("name"): {str(x.get("value")): num(x.get("odd")) for x in b.get("values", [])} for b in bk.get("bets", [])}
        if not bets:
            continue
        names.append(bk.get("name"))
        ha = bets.get("Home/Away")
        p = devig([ha.get("Home"), ha.get("Away")]) if ha else None
        if p:                                         # vainqueur, prolongation incluse
            acc["d"].setdefault(0.0, []).append(p[0])
        elif bets.get("3Way Result"):                 # à défaut, temps réglementaire : nul partagé
            v = bets["3Way Result"]
            p3 = devig([v.get("Home"), v.get("Draw"), v.get("Away")])
            if p3:
                acc["d"].setdefault(0.0, []).append(p3[0] + p3[1] / 2)
        for name, k in (("Asian Handicap", "d"), ("Asian Handicap First Half", "d1")):
            for key, o in (bets.get(name) or {}).items():
                m = re.match(r"Home ([+-]?[\d.]+)$", key)
                if m:
                    # API-Sports écrit la ligne du point de vue du domicile des deux côtés :
                    # "Home +3" et "Away +3" sont les deux issues d'un même pari
                    h = float(m.group(1))
                    p = devig([o, bets[name].get(f"Away {m.group(1)}")])
                    if p:
                        # domicile +h gagne si écart + h > 0, soit écart > -h
                        acc[k].setdefault(-h, []).append(p[0])
        for name, k in (("Over/Under", "t"), ("Over/Under 1st Half", "t1")):
            v = bets.get(name) or {}
            for key, o in v.items():
                m = re.match(r"Over ([\d.]+)$", key)
                if m:
                    p = devig([o, v.get(f"Under {m.group(1)}")])
                    if p:
                        acc[k].setdefault(float(m.group(1)), []).append(p[0])
        v = bets.get("1st Half 3Way Result")
        if v:
            p3 = devig([v.get("Home"), v.get("Draw"), v.get("Away")])
            if p3:
                acc["d1"].setdefault(0.0, []).append(p3[0] + p3[1] / 2)
    out = {k: {line: float(np.mean(ps)) for line, ps in v.items()} for k, v in acc.items()}
    out["bookmakers"] = sorted({n for n in names if n})
    return out


def fit_normal(points: dict, mu0: float, sd0: float, lo: float, hi: float, sd_bounds: tuple) -> tuple[float, float] | None:
    """Loi normale (moyenne, écart-type) qui reproduit au mieux P(X > ligne)."""
    if not points:
        return None
    lines = np.array(list(points), dtype=float)
    ps = np.clip(np.array(list(points.values())), 0.01, 0.99)
    # La dispersion ne se lit que sur des lignes assez éloignées les unes des autres ; sinon
    # (quelques lignes autour de la ligne principale) on garde l'écart-type habituel.
    if len(set(lines)) < 3 or lines.max() - lines.min() < 0.8 * sd0:
        mu = least_squares(lambda x: (1 - norm.cdf(lines, x[0], sd0)) - ps, [mu0], bounds=([lo], [hi])).x[0]
        return float(mu), float(sd0)

    def res(x):
        mu, sd = x
        return np.r_[(1 - norm.cdf(lines, mu, sd)) - ps, 0.2 * (sd - sd0) / sd0]
    x = least_squares(res, [mu0, sd0], bounds=([lo, sd_bounds[0]], [hi, sd_bounds[1]])).x
    return float(x[0]), float(x[1])


def market_params(bookmakers: list, sd_d0: float, sd_t0: float):
    """Pinnacle seul s'il donne vainqueur/écart et total, sinon consensus de tous les bookmakers."""
    pin = [b for b in bookmakers if b.get("name") == "Pinnacle"]
    c_pin = constraints(pin) if pin else None
    c_all = constraints(bookmakers)
    use_pin = bool(c_pin and c_pin["d"] and c_pin["t"])
    c = c_pin if use_pin else c_all
    mu_t0 = float(np.median(list(c["t"]))) if c["t"] else None
    if mu_t0 is None or not c["d"]:
        return None
    d = fit_normal(c["d"], 0.0, sd_d0, -45, 45, (7.0, 20.0))
    t = fit_normal(c["t"], mu_t0, sd_t0, 90, 290, (9.0, 28.0))
    # 1re mi-temps : Pinnacle si disponible, sinon consensus, sinon moitié du match
    c1 = c if (c["d1"] or c["t1"]) else c_all
    d1 = fit_normal(c1["d1"], d[0] / 2, d[1] * 0.68, -30, 30, (4.0, 14.0)) or (d[0] / 2, d[1] * 0.68)
    t1_0 = float(np.median(list(c1["t1"]))) if c1["t1"] else t[0] * 0.5
    t1 = fit_normal(c1["t1"], t1_0, t[1] * 0.68, 40, 150, (5.0, 20.0)) or (t[0] * 0.5, t[1] * 0.68)
    return {"d": d, "t": t, "d1": d1, "t1": t1, "pinnacle": use_pin, "books": c_all["bookmakers"],
            "has_half": bool(c1["d1"] or c1["t1"])}


# ------------------------------------------------------------------ marchés

def p_over(line, mu, sd):
    return float(1 - norm.cdf(line, mu, sd))


def build_markets(prm: dict, home: str, away: str) -> tuple[dict, dict]:
    (mu_d, sd_d), (mu_t, sd_t) = prm["d"], prm["t"]
    m = {}
    p_home = p_over(0.0, mu_d, sd_d)
    m[f"Victoire {home}"], m[f"Victoire {away}"] = p_home, 1 - p_home
    fav, dog, sign = (home, away, 1) if mu_d >= 0 else (away, home, -1)
    r = max(1, round(abs(mu_d)))
    for k in sorted({max(2, r + x) for x in (-6, -3, 0, 3, 6, 10)} | {5, 10, 15}):
        if k <= 25:
            m[f"{fav} gagne par {k} points ou plus"] = p_over(k - 0.5, sign * mu_d, sd_d)
    for k in sorted({max(2, r + x) for x in (-3, 0, 4)}):
        m[f"{dog} gagne ou perd de moins de {k} points"] = 1 - p_over(k - 0.5, sign * mu_d, sd_d)
    centre = math.floor(mu_t) + 0.5
    for x in (-15, -10, -5, 0, 5, 10, 15):
        line = centre + x
        m[f"+{line:g} points"] = p_over(line, mu_t, sd_t)
        m[f"-{line:g} points"] = 1 - m[f"+{line:g} points"]
    # points d'une équipe : écart et total supposés indépendants (corrélation faible au basket)
    sd_team = math.sqrt(sd_t ** 2 + sd_d ** 2) / 2
    mu_h, mu_a = (mu_t + mu_d) / 2, (mu_t - mu_d) / 2
    for team, mu in ((home, mu_h), (away, mu_a)):
        base = 5 * round(mu / 5)
        for n in range(base - 15, base + 16, 5):
            m[f"{team} marque au moins {n} points"] = p_over(n - 0.5, mu, sd_team)
    # combinés fréquents dans les boosts : vainqueur + total (indépendance supposée)
    for team, pw in ((home, p_home), (away, 1 - p_home)):
        for x in (-10, 0, 10):
            line = centre + x
            m[f"Victoire {team} et +{line:g} points"] = pw * p_over(line, mu_t, sd_t)
            m[f"Victoire {team} et -{line:g} points"] = pw * (1 - p_over(line, mu_t, sd_t))
    (mu_d1, sd_d1), (mu_t1, sd_t1) = prm["d1"], prm["t1"]
    p1h = p_over(0.5, mu_d1, sd_d1); p1a = 1 - p_over(-0.5, mu_d1, sd_d1)
    m[f"Mi-temps : {home} mène"], m["Mi-temps : égalité"], m[f"Mi-temps : {away} mène"] = p1h, 1 - p1h - p1a, p1a
    c1 = math.floor(mu_t1) + 0.5
    for x in (-5, 0, 5):
        m[f"1re mi-temps : +{c1 + x:g} points"] = p_over(c1 + x, mu_t1, sd_t1)
        m[f"1re mi-temps : -{c1 + x:g} points"] = 1 - m[f"1re mi-temps : +{c1 + x:g} points"]
    info = {"mu_d": round(mu_d, 2), "sd_d": round(sd_d, 2), "mu_t": round(mu_t, 2), "sd_t": round(sd_t, 2),
            "mu_h": round(mu_h, 1), "mu_a": round(mu_a, 1), "mu_t1": round(mu_t1, 1), "mu_d1": round(mu_d1, 2)}
    return {k: float(min(max(v, 0.0), 1.0)) for k, v in m.items()}, info


def family(label: str) -> str:
    if label.startswith("Victoire") and " et " in label:
        return "Combiné"
    if label.startswith("Victoire"):
        return "Vainqueur"
    if "gagne par" in label or "perd de moins" in label:
        return "Écart"
    if label.startswith(("Mi-temps", "1re mi-temps")):
        return "1re mi-temps"
    if "marque au moins" in label:
        return "Points d'une équipe"
    return "Total points"


def back_to_back(sched: list, g: dict) -> str | None:
    """Équipe qui a joué la veille (toutes compétitions) : fatigue connue au basket."""
    t0 = pd.Timestamp(g["date"])
    msgs = []
    for side in ("home", "away"):
        tid = g["teams"][side]["id"]
        # un match commencé entre 32 h et 6 h avant celui-ci (horaires comparés en absolu,
        # pour ne pas se tromper avec les matchs NBA joués la nuit, heure de Paris)
        played = any(tid in s["teams"] and 6 <= (t0 - pd.Timestamp(s["date"])).total_seconds() / 3600 <= 32
                     for s in sched)
        if played:
            msgs.append(f"{g['teams'][side]['name']} a joué la veille (back-to-back)")
    return " ; ".join(msgs) or None


def predict(upcoming: list, odds: dict, sched: list) -> pd.DataFrame:
    import odds_winamax as OW
    fr_idx = OW.lookup("basketball")
    rows, archive = [], []
    for g in upcoming:
        bks = odds.get(str(g["id"]))
        lg = league_of(g)
        if not bks or not lg:
            continue
        lname, sd_d0, sd_t0 = lg
        prm = market_params(bks, sd_d0, sd_t0)
        if not prm:
            continue
        home, away = g["teams"]["home"]["name"], g["teams"]["away"]["name"]
        mk, info = build_markets(prm, home, away)
        t = pd.Timestamp(g["date"]).tz_convert(PARIS)
        sig = back_to_back(sched, g)
        archive.append({"id": g["id"], "date": g["date"], "league": lname, "home": home, "away": away, **info,
                        "pinnacle": prm["pinnacle"], "bookmakers": len(prm["books"])})
        src = ("des cotes de Pinnacle" if prm["pinnacle"] else
               f"du consensus de {len(prm['books'])} bookmaker(s) ({', '.join(prm['books'][:4])}{'…' if len(prm['books']) > 4 else ''})")
        base = (f"Cotes justes tirées {src}, marge retirée : {info['mu_t']:.1f} points attendus au total, "
                f"{home} {info['mu_h']:.1f} – {info['mu_a']:.1f} {away} (écart-type du total {info['sd_t']:.1f}, "
                f"de l'écart {info['sd_d']:.1f}). Prolongation incluse (vainqueur, écarts, totaux). "
                f"Points d'une équipe et combinés : écart et total supposés indépendants."
                + ("" if prm["has_half"] else " 1re mi-temps estimée à partir du match entier (pas de cote).")
                + " Absences et repos : voir la vérification de contexte (6h, 12h, 18h).")
        fr = OW.match_odds(fr_idx, t.strftime("%Y-%m-%d"), home, away) if fr_idx else None
        fr_lab = {f"Victoire {home}": "1", f"Victoire {away}": "2"}
        for label, p in mk.items():
            if p <= 0.02 or p >= 0.98:
                continue
            fair = 1 / p
            q = (fr or {}).get("cotes", {}).get(fr_lab.get(label, ""), {})
            cw = q.get("Winamax")
            best_op, best = (max(q.items(), key=lambda kv: kv[1]) if q else (None, None))
            expl = f"Proba {p:.0%} → cote juste {fair:.2f}. Value seulement si Winamax paie au moins {fair * 1.03:.2f} (+3 %). " + base
            if cw:
                expl += f" Winamax ce matin : {cw:.2f}, soit une value de {p * cw - 1:+.0%}."
            rows.append({
                "sport": "Basket", "id": f"K|{g['id']}|{label}",
                "date": t.strftime("%Y-%m-%d"), "heure": t.strftime("%H:%M"),
                "championnat": lname, "match": f"{home} - {away}", "famille": family(label), "marche": label,
                "proba": round(p, 4), "cote_juste": round(fair, 2), "cote_minimum": round(fair * 1.03, 2),
                "cote_moyenne_marche": None, "source_cote": "marché",
                "fiabilite": "bonne" if (prm["pinnacle"] or len(prm["books"]) >= 2) else "moyenne",
                "infos": f"{info['mu_t']:.1f} points attendus · {home} {info['mu_h']:.1f} – {info['mu_a']:.1f} {away}",
                "explication": re.sub(r"(\d)\.(\d)", r"\1,\2", expl),
                "signaux": sig,
                "cote_winamax": cw, "value_winamax": round(p * cw - 1, 4) if cw else None,
                "meilleure_cote_fr": best, "meilleur_operateur": best_op,
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
        cf = RAW / "calendrier.json"
        sched = json.loads(cf.read_text()) if cf.exists() else []
    else:
        upcoming, odds, sched = collect()
    df = predict(upcoming, odds, sched)
    df.to_csv(OUT / "dashboard_basket.csv", index=False)
    print(f"basket : {df['match'].nunique() if len(df) else 0} matchs, {len(df)} cotes justes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
