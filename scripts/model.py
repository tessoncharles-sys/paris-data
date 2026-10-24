"""Modèle de probabilités foot multi-marchés.

Principe :
- Pour chaque championnat, on estime la force offensive / défensive de chaque équipe
  (régression de Poisson, matchs récents plus pondérés que les anciens).
- Buts : on lisse les buts réels avec les tirs cadrés (moins de bruit que le score seul).
- Même méthode pour les corners et les cartons.
- On en déduit la grille des scores exacts → tous les marchés buts.
- Quand la cote moyenne du marché est connue, on mélange modèle et marché (50/50) :
  le marché est très informé, ça évite les faux signaux.
- Sortie : probabilité, cote juste et COTE MINIMUM à exiger sur Winamax.
"""
from __future__ import annotations

import datetime as dt
import re
from zoneinfo import ZoneInfo
import io
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor

ROOT = Path(__file__).resolve().parent.parent
FD = ROOT / "data" / "raw" / "football-data"
OUT = ROOT / "data"

CONFIG = {
    "demi_vie_jours": 140,       # un match d'il y a 140 jours compte moitié moins
    "poids_tirs_cadres": 0.5,    # part des tirs cadrés dans la mesure "buts"
    "poids_modele": 0.0,         # test saison passée : le marché seul est plus juste que le modèle sur 1X2 et +/-2,5
    "marge_value": 0.05,         # on exige 5 % de value : cote mini = cote juste × 1,05
    "cote_min": 1.35,            # fourchette de cotes de la sélection
    "cote_max": 3.00,          # fourchette élargie (sélecteur Sûr / Équilibré / Toutes dans le tableau de bord)
    "matchs_min_fiable": 5,      # en dessous, on n'analyse pas le match
    "matchs_bon_echantillon": 10,
    "horizon_jours": 1,          # aujourd'hui + demain
    "marches_exclus": ["corners", "cartons"],  # non proposés sur Winamax
}

from collect import DIVISIONS  # noqa: E402  (même dossier)

CALIB_FILE = OUT / "calibration.json"


def load_calibration() -> dict:
    try:
        return json.loads(CALIB_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def calibrate(label: str, p: float, calib: dict) -> float:
    """Corrige l'excès de confiance mesuré sur la saison passée (régression logistique sur logit(p))."""
    coef = calib.get(label)
    if not coef:
        return p
    p = min(max(p, 1e-4), 1 - 1e-4)
    z = coef[0] + coef[1] * math.log(p / (1 - p))
    return 1 / (1 + math.exp(-z))


# ---------------------------------------------------------------- lecture

def read_csv(path: Path) -> pd.DataFrame:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.StringIO(raw.decode(enc)), on_bad_lines="skip")
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    return pd.DataFrame()


def parse_dates(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, dayfirst=True, errors="coerce", format="mixed")


def load_history() -> pd.DataFrame:
    frames = []
    for f in sorted(FD.glob("*_*.csv")):
        df = read_csv(f)
        if df.empty or "HomeTeam" not in df.columns:
            continue
        df["Div"] = f.stem.split("_", 1)[1]
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    h = pd.concat(frames, ignore_index=True)
    h["Date"] = parse_dates(h["Date"])
    h = h.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    return h


def load_fixtures(today: dt.date) -> pd.DataFrame:
    f = FD / "fixtures.csv"
    if not f.exists():
        return pd.DataFrame()
    fx = read_csv(f)
    if fx.empty or "Div" not in fx.columns:
        return pd.DataFrame()
    fx["Date"] = parse_dates(fx["Date"])
    end = today + dt.timedelta(days=CONFIG["horizon_jours"])
    mask = fx["Div"].isin(DIVISIONS) & fx["Date"].dt.date.between(today, end)
    return fx[mask].copy()


# ---------------------------------------------------------------- modèle

def decay_weights(dates: pd.Series, ref: pd.Timestamp) -> np.ndarray:
    age = (ref - dates).dt.days.clip(lower=0).to_numpy()
    return np.power(0.5, age / CONFIG["demi_vie_jours"])


class TeamModel:
    """Poisson : log(mu) = constante + domicile + attaque(équipe) + défense(adversaire)."""

    def __init__(self, home_col: str, away_col: str):
        self.home_col, self.away_col = home_col, away_col
        self.ok = False

    def fit(self, df: pd.DataFrame, ref: pd.Timestamp, home_y=None, away_y=None) -> "TeamModel":
        hy = df[self.home_col] if home_y is None else home_y
        ay = df[self.away_col] if away_y is None else away_y
        d = pd.DataFrame({
            "att": pd.concat([df["HomeTeam"], df["AwayTeam"]], ignore_index=True),
            "dfn": pd.concat([df["AwayTeam"], df["HomeTeam"]], ignore_index=True),
            "home": np.r_[np.ones(len(df)), np.zeros(len(df))],
            "y": pd.concat([hy, ay], ignore_index=True).astype(float),
            "w": np.tile(decay_weights(df["Date"], ref), 2),
        }).dropna()
        if len(d) < 40:
            return self
        self.teams = sorted(set(d["att"]) | set(d["dfn"]))
        X = self._design(d["att"], d["dfn"], d["home"])
        self.reg = PoissonRegressor(alpha=1e-3, max_iter=1000)
        self.reg.fit(X, d["y"], sample_weight=d["w"])
        self.ok = True
        return self

    def _design(self, att, dfn, home) -> np.ndarray:
        idx = {t: i for i, t in enumerate(self.teams)}
        n = len(self.teams)
        X = np.zeros((len(att), 2 * n + 1))
        for r, (a, b, h) in enumerate(zip(att, dfn, home)):
            X[r, idx[a]] = 1
            X[r, n + idx[b]] = 1
            X[r, 2 * n] = h
        return X

    def predict(self, home: str, away: str) -> tuple[float, float] | None:
        if not self.ok or home not in self.teams or away not in self.teams:
            return None
        X = self._design([home, away], [away, home], [1, 0])
        mu = self.reg.predict(X)
        return float(mu[0]), float(mu[1])


def smoothed_goals(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Buts lissés avec les tirs cadrés × taux de conversion du championnat."""
    w = CONFIG["poids_tirs_cadres"]
    if not {"HST", "AST"}.issubset(df.columns) or df[["HST", "AST"]].isna().mean().max() > 0.2:
        return df["FTHG"], df["FTAG"]
    sot = df["HST"].sum() + df["AST"].sum()
    conv = (df["FTHG"].sum() + df["FTAG"].sum()) / sot if sot else 0.3
    hg = (1 - w) * df["FTHG"] + w * df["HST"].fillna(df["FTHG"] / conv) * conv
    ag = (1 - w) * df["FTAG"] + w * df["AST"].fillna(df["FTAG"] / conv) * conv
    return hg, ag


# ---------------------------------------------------------------- marchés

def score_grid(lh: float, la: float, n: int = 11) -> np.ndarray:
    return np.outer(poisson.pmf(range(n), lh), poisson.pmf(range(n), la))


def goal_markets(lh: float, la: float) -> dict[str, float]:
    g = score_grid(lh, la)
    i, j = np.indices(g.shape)
    tot = i + j
    p = {
        "1": g[i > j].sum(), "N": g[i == j].sum(), "2": g[i < j].sum(),
    }
    p.update({"1N": p["1"] + p["N"], "12": p["1"] + p["2"], "N2": p["N"] + p["2"]})
    for line in (1.5, 2.5, 3.5):
        p[f"+{line} buts"] = g[tot > line].sum()
        p[f"-{line} buts"] = g[tot < line].sum()
    p["Les 2 marquent : oui"] = g[(i > 0) & (j > 0)].sum()
    p["Les 2 marquent : non"] = 1 - p["Les 2 marquent : oui"]
    p["Dom. +0.5 but"] = g[i > 0].sum()
    p["Dom. +1.5 but"] = g[i > 1].sum()
    p["Ext. +0.5 but"] = g[j > 0].sum()
    p["Ext. +1.5 but"] = g[j > 1].sum()
    p["Dom. ou nul, les 2 marquent : non"] = g[(i >= j) & ~((i > 0) & (j > 0))].sum()
    return {k: float(v) for k, v in p.items()}


def count_markets(lh: float, la: float, lines, label: str) -> dict[str, float]:
    lam = lh + la
    out = {}
    for line in lines:
        over = 1 - poisson.cdf(math.floor(line), lam)
        out[f"+{line} {label}"] = float(over)
        out[f"-{line} {label}"] = float(1 - over)
    return out


def market_implied(row: pd.Series) -> dict[str, float]:
    """Probabilités implicites des cotes moyennes du marché (marge retirée)."""
    out = {}

    def norm(cols):
        vals = [row.get(c) for c in cols]
        if any(v is None or pd.isna(v) or v <= 1 for v in vals):
            return None
        inv = [1 / v for v in vals]
        s = sum(inv)
        return [x / s for x in inv]

    for prefix in ("Avg", "B365", "PS"):
        r = norm([f"{prefix}H", f"{prefix}D", f"{prefix}A"])
        if r:
            out.update({"1": r[0], "N": r[1], "2": r[2]})
            out.update({"1N": r[0] + r[1], "12": r[0] + r[2], "N2": r[1] + r[2]})
            break
    for prefix in ("Avg", "B365", "P"):
        r = norm([f"{prefix}>2.5", f"{prefix}<2.5"])
        if r:
            out.update({"+2.5 buts": r[0], "-2.5 buts": r[1]})
            break
    return out


def market_lambdas(row: pd.Series, lam_model: tuple[float, float]) -> tuple[float, float, str]:
    """Buts attendus (dom, ext) qui reproduisent les cotes du marché (1X2 + plus/moins 2,5).

    Le test sur 2025-26 montre que le marché est plus juste que le modèle : ces buts attendus
    servent au calculateur de boosts. Sans cote marché, on garde ceux du modèle."""
    mk = market_implied(row)
    if not {"1", "2"}.issubset(mk):
        return lam_model[0], lam_model[1], "modèle"
    over = mk.get("+2.5 buts")

    def loss(x):
        g = score_grid(*np.exp(x))
        i, j = np.indices(g.shape)
        e = (g[i > j].sum() - mk["1"]) ** 2 + (g[i < j].sum() - mk["2"]) ** 2
        if over is not None:
            e += (g[(i + j) > 2].sum() - over) ** 2
        return e

    x = minimize(loss, np.log(np.clip(lam_model, 0.2, 5)), method="Nelder-Mead",
                 options={"xatol": 1e-4, "fatol": 1e-9}).x
    lh, la = np.exp(x)
    return float(lh), float(la), "marché"


def market_odds(row: pd.Series) -> dict[str, float]:
    """Cotes moyennes brutes, pour comparer à la cote minimum."""
    m = {}
    for k, c in (("1", "AvgH"), ("N", "AvgD"), ("2", "AvgA"),
                 ("+2.5 buts", "Avg>2.5"), ("-2.5 buts", "Avg<2.5")):
        v = row.get(c)
        if v is not None and not pd.isna(v) and v > 1:
            m[k] = float(v)
    return m


# ---------------------------------------------------------------- pipeline

def recent_count(hist: pd.DataFrame, team: str, ref: pd.Timestamp) -> int:
    h = hist[hist["Date"] > ref - pd.Timedelta(days=365)]
    return int(((h["HomeTeam"] == team) | (h["AwayTeam"] == team)).sum())


def analyse(today: dt.date) -> dict:
    hist = load_history()
    fx = load_fixtures(today)
    calib = load_calibration()
    # cotes des bookmakers (Pinnacle en priorité), récupérées par odds_foot.py
    try:
        _oc = json.loads((ROOT / "data" / "raw" / "api-football" / "odds_latest.json").read_text())
        api_odds = {v["cle"]: v for v in _oc.values()}
    except (OSError, ValueError):
        api_odds = {}
    ref = pd.Timestamp(today)
    matches, picks, skipped = [], [], []
    if hist.empty or fx.empty:
        return {"date": today.isoformat(), "matchs": [], "selection": [],
                "ecartes": [], "note": "données absentes"}

    for div, fx_div in fx.groupby("Div"):
        h = hist[(hist["Div"] == div) & (hist["Date"] < ref)]
        if len(h) < 30:
            skipped += [{"match": f"{r.HomeTeam} - {r.AwayTeam}", "raison": "historique insuffisant"}
                        for r in fx_div.itertuples()]
            continue
        hg, ag = smoothed_goals(h)
        goals = TeamModel("FTHG", "FTAG").fit(h, ref, hg, ag)
        corners = TeamModel("HC", "AC").fit(h.dropna(subset=["HC", "AC"]), ref) \
            if {"HC", "AC"}.issubset(h.columns) else TeamModel("HC", "AC")
        if {"HY", "AY"}.issubset(h.columns):
            hc = h.dropna(subset=["HY", "AY"])
            cards = TeamModel("HY", "AY").fit(
                hc, ref,
                hc["HY"] + (hc["HR"].fillna(0) if "HR" in hc else 0),
                hc["AY"] + (hc["AR"].fillna(0) if "AR" in hc else 0))
        else:
            cards = TeamModel("HY", "AY")

        for _, row in fx_div.iterrows():
            home, away = row["HomeTeam"], row["AwayTeam"]
            name = f"{home} - {away}"
            n = min(recent_count(h, home, ref), recent_count(h, away, ref))
            lam = goals.predict(home, away)
            if lam is None or n < CONFIG["matchs_min_fiable"]:
                skipped.append({"match": name, "raison": "équipe sans assez de matchs dans ce championnat"})
                continue

            probs = goal_markets(*lam)
            c = corners.predict(home, away)
            if c:
                probs.update(count_markets(*c, (8.5, 9.5, 10.5), "corners"))
            k = cards.predict(home, away)
            if k:
                probs.update(count_markets(*k, (3.5, 4.5), "cartons"))

            mk = market_implied(row)
            odds = market_odds(row)
            w = CONFIG["poids_modele"]
            marches = []
            ao = api_odds.get(f"{div}|{home}|{away}", {})
            for label, p_raw in probs.items():
                p_mod = calibrate(label, p_raw, calib)
                api_p = (ao.get("marches") or {}).get(label)
                src_m = None
                if api_p and (api_p.get("pin") or api_p.get("cons")):
                    # marché des bookmakers : plus juste que le modèle d'après nos tests → c'est la référence
                    pm = api_p["pin"] or api_p["cons"]
                    src_m = "Pinnacle" if api_p.get("pin") else f"{api_p['n']} bookmakers"
                    mk_l = pm
                elif label in mk:
                    mk_l = mk[label]
                    src_m = "moyenne des bookmakers (football-data)"
                else:
                    mk_l = None
                p = w * p_mod + (1 - w) * mk_l if mk_l is not None else p_mod
                fair = 1 / p if p > 0 else None
                mini = fair * (1 + CONFIG["marge_value"]) if fair else None
                marches.append({
                    "marche": label,
                    "proba_modele": round(p_mod, 4),
                    "source_marche": src_m,
                    "proba_marche": round(mk_l, 4) if mk_l is not None else None,
                    "proba": round(p, 4),
                    "cote_juste": round(fair, 2) if fair else None,
                    "cote_minimum": round(mini, 2) if mini else None,
                    "cote_moyenne_marche": odds.get(label),
                })

            fiab = "bonne" if n >= CONFIG["matchs_bon_echantillon"] else "moyenne"
            lam_m = market_lambdas(row, lam)
            kickoff = row["Date"].strftime("%Y-%m-%d")
            if "Time" in row and not pd.isna(row.get("Time")):
                # football-data donne l'heure anglaise : conversion en heure de Paris
                try:
                    t = dt.datetime.combine(row["Date"].date(), dt.time.fromisoformat(str(row["Time"])),
                                            tzinfo=ZoneInfo("Europe/London"))
                    kickoff = t.astimezone(ZoneInfo("Europe/Paris")).strftime("%Y-%m-%d %H:%M")
                except ValueError:
                    kickoff += f" {row['Time']}"
            m = {
                "match": name, "championnat": DIVISIONS[div], "div": div,
                "coup_envoi": kickoff,
                "buts_attendus": {"dom": round(lam[0], 2), "ext": round(lam[1], 2)},
                "lambda_boost": {"dom": round(lam_m[0], 3), "ext": round(lam_m[1], 3), "source": lam_m[2]},
                "corners_attendus": round(sum(c), 1) if c else None,
                "cartons_attendus": round(sum(k), 1) if k else None,
                "fiabilite": fiab, "marches": marches,
                "forme": {"dom": team_form(hist, home, ref), "ext": team_form(hist, away, ref)},
            }
            ko = pd.Timestamp(kickoff) if len(kickoff) > 10 else pd.Timestamp(kickoff + " 15:00")
            sig = [x for x in (rest_signal(hist, home, div, ko), rest_signal(hist, away, div, ko)) if x]
            m["signaux"] = " ; ".join(sig)
            matches.append(m)

            for mm in marches:
                if any(x in mm["marche"] for x in CONFIG["marches_exclus"]):
                    continue
                mini = mm["cote_minimum"]
                if not mini or not (CONFIG["cote_min"] <= mini <= CONFIG["cote_max"]):
                    continue
                avg = mm["cote_moyenne_marche"]
                if avg is not None and avg < mini:
                    continue  # le marché paie moins que notre exigence : pas de value
                picks.append({
                    "match": name, "championnat": DIVISIONS[div], "coup_envoi": kickoff,
                    "marche": mm["marche"], "proba": mm["proba"],
                    "cote_minimum": mini, "cote_moyenne_marche": avg,
                    "value_vs_marche": round(mm["proba"] * avg - 1, 3) if avg else None,
                    "fiabilite": fiab,
                    "a_verifier": avg is None,
                })

    # classement : value confirmée par le marché d'abord, puis probabilité
    picks.sort(key=lambda p: (p["a_verifier"], -(p["value_vs_marche"] or 0), -p["proba"]))
    # au plus 2 paris "à vérifier" par match, pour garder une liste lisible
    seen: dict[str, int] = {}
    kept = []
    for p in picks:
        if p["a_verifier"]:
            seen[p["match"]] = seen.get(p["match"], 0) + 1
            if seen[p["match"]] > 2:
                continue
        kept.append(p)
    picks = kept
    return {
        "date": today.isoformat(), "genere_le": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "config": CONFIG, "matchs": matches, "selection": picks, "ecartes": skipped,
    }


def to_markdown(res: dict) -> str:
    lines = [f"# Sélection du {res['date']}", ""]
    sel = res["selection"]
    confirmed = [p for p in sel if not p["a_verifier"]]
    check = [p for p in sel if p["a_verifier"]]
    lines += ["## Value confirmée par le marché", ""]
    if not confirmed:
        lines.append("_Aucune aujourd'hui._")
    for p in confirmed[:15]:
        lines.append(f"- **{p['match']}** ({p['championnat']}, {p['coup_envoi']}) — {p['marche']} : "
                     f"proba {p['proba']:.0%}, **cote mini {p['cote_minimum']}** "
                     f"(marché {p['cote_moyenne_marche']}, value {p['value_vs_marche']:+.1%}) — fiabilité {p['fiabilite']}")
    lines += ["", "## À vérifier sur Winamax (pas de cote marché dans les données)", ""]
    for p in check[:20]:
        lines.append(f"- {p['match']} — {p['marche']} : proba {p['proba']:.0%}, "
                     f"**jouer seulement si cote ≥ {p['cote_minimum']}** — fiabilité {p['fiabilite']}")
    lines += ["", f"_{len(res['matchs'])} matchs analysés, {len(res['ecartes'])} écartés._"]
    return "\n".join(lines)


def famille(label: str) -> str:
    if label in ("1", "N", "2"):
        return "1X2"
    if label in ("1N", "12", "N2"):
        return "Double chance"
    if "buts" in label and ("+" in label[:1] or "-" in label[:1]):
        return "Total buts"
    if "marquent" in label:
        return "Les 2 marquent"
    if label.startswith(("Dom.", "Ext.")):
        return "Buts par équipe"
    if "corners" in label:
        return "Corners"
    if "cartons" in label:
        return "Cartons"
    return "Autre"


# ------------------------------------------------------------------ signaux : repos et rotation

DIV_COUNTRY = {"E": "England", "SC": "Scotland", "D": "Germany", "I": "Italy", "SP": "Spain", "F": "France",
               "N": "Netherlands", "B": "Belgium", "P": "Portugal", "T": "Turkey", "G": "Greece"}
EURO = ("UEFA", "Champions League", "Europa", "Conference League")
# Compétitions masculines prises en compte (identifiants API-Football) : championnats suivis, coupes nationales, coupes d'Europe
COMPETITIONS = {39, 40, 41, 42, 45, 48, 179, 180, 181, 185, 78, 79, 81, 135, 136, 137, 140, 141, 143, 61, 62, 66,
                88, 90, 144, 147, 94, 96, 97, 203, 206, 197, 199, 2, 3, 848, 531}
JOURS = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]


def _norm(name: str) -> str:
    import unicodedata
    n = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    n = " " + n.replace("-", " ").replace(".", " ") + " "
    for w in (" fc ", " cf ", " ac ", " afc ", " sc ", " ss ", " calcio ", " 1 ", " as ", " ogc ", " ud ", " cd ", " sd ", " kv "):
        n = n.replace(w, " ")
    return " ".join(n.split())


_ARCHIVE = None


def api_archive() -> pd.DataFrame:
    """Matchs (toutes compétitions) archivés chaque jour depuis API-Football."""
    global _ARCHIVE
    if _ARCHIVE is None:
        rows = []
        for f in sorted((ROOT / "data" / "raw" / "api-football").glob("fixtures_*.json")):
            try:
                for x in json.loads(f.read_text(encoding="utf-8")):
                    if x["league"]["id"] not in COMPETITIONS:
                        continue
                    d = pd.Timestamp(x["fixture"]["date"]).tz_convert("Europe/Paris")
                    for side in ("home", "away"):
                        rows.append({"team": x["teams"][side]["name"], "norm": _norm(x["teams"][side]["name"]),
                                     "date": d, "competition": x["league"]["name"], "country": x["league"]["country"],
                                     "league_id": x["league"]["id"]})
            except (OSError, ValueError, KeyError):
                continue
        _ARCHIVE = pd.DataFrame(rows).drop_duplicates() if rows else pd.DataFrame(columns=["team", "norm", "date", "competition", "country", "league_id"])
    return _ARCHIVE


_MAP: dict = {}


DIV_LEAGUE = {"E0": 39, "E1": 40, "E2": 41, "E3": 42, "SC0": 179, "SC1": 180, "D1": 78, "D2": 79, "I1": 135, "I2": 136,
              "SP1": 140, "SP2": 141, "F1": 61, "F2": 62, "N1": 88, "B1": 144, "P1": 94, "T1": 203, "G1": 197}
ABBR = {"st": "saint", "ath": "atletico", "man": "manchester", "sp": "sporting", "nott'm": "nottingham", "ein": "eintracht",
        "m'gladbach": "monchengladbach", "wolves": "wolverhampton", "psg": "paris saint germain", "sheffield weds": "sheffield wednesday",
        "qpr": "queens park rangers", "west brom": "west bromwich", "espanol": "espanyol", "sociedad": "real sociedad",
        "betis": "real betis", "paris sg": "paris saint germain", "sp lisbon": "sporting cp", "ath bilbao": "athletic club",
        "ath madrid": "atletico madrid", "leverkusen": "bayer leverkusen", "dortmund": "borussia dortmund"}


def _expand(name: str) -> str:
    n = _norm(name.replace("'", "'"))
    low = name.lower()
    if low in ABBR:
        return _norm(ABBR[low])
    return " ".join(ABBR.get(t, t) for t in n.split())


def api_team(fd_name: str, div: str) -> str | None:
    """Nom API-Football de l'équipe, cherché uniquement parmi les équipes du même championnat (pour éviter les erreurs)."""
    import difflib
    key = (fd_name, div)
    if key not in _MAP:
        arc = api_archive()
        cand = arc[arc["league_id"] == DIV_LEAGUE.get(div)]
        best, score, ties = None, 0.0, 0
        target = _expand(fd_name)
        ta = set(target.split())
        for team in set(cand["team"]):
            t = _norm(team)
            tb = set(t.split())
            # même nom, ou tous les mots du plus court contenus dans le plus long ; sinon quasi identique
            sc = 1.0 if (ta and tb and (ta <= tb or tb <= ta)) else difflib.SequenceMatcher(None, target, t).ratio()
            if sc > score:
                best, score, ties = team, sc, 0
            elif sc == score == 1.0:
                ties += 1                         # deux équipes possibles : on préfère ne rien dire
        _MAP[key] = best if score >= 0.88 and not ties else None
    return _MAP[key]


def rest_signal(hist: pd.DataFrame, team: str, div: str, kickoff: pd.Timestamp) -> str | None:
    """Jours de repos avant le match et match européen proche (risque de rotation)."""
    k = kickoff.tz_localize("Europe/Paris") if kickoff.tzinfo is None else kickoff
    dates = []                                   # (date, compétition)
    g = hist[((hist["HomeTeam"] == team) | (hist["AwayTeam"] == team))]
    dates += [(pd.Timestamp(d).tz_localize("Europe/Paris") + pd.Timedelta(hours=15), "championnat") for d in g["Date"]]
    at = api_team(team, div)
    arc = api_archive()
    nxt_euro = None
    if at is not None and len(arc):
        a = arc[arc["team"] == at]
        dates += list(zip(a["date"], a["competition"]))
        fut = a[(a["date"] > k + pd.Timedelta(hours=12)) & (a["date"] < k + pd.Timedelta(days=5)) & a["competition"].str.contains("|".join(EURO))]
        if len(fut):
            nxt_euro = fut.sort_values("date").iloc[0]
    past = sorted([(d, c) for d, c in dates if d < k - pd.Timedelta(hours=12)], key=lambda x: x[0])
    msgs = []
    if past:
        d, c = past[-1]
        rest = (k.normalize() - d.normalize()).days
        comp = "match européen" if any(e in str(c) for e in EURO) else ("championnat" if c == "championnat" else str(c))
        if rest <= 3:
            msgs.append(f"{team} : seulement {rest} jour{'s' if rest > 1 else ''} de repos (dernier match : {comp}, {JOURS[d.weekday()]})")
    if nxt_euro is not None:
        msgs.append(f"{team} joue en {nxt_euro['competition']} {JOURS[nxt_euro['date'].weekday()]} : risque de rotation")
    return " ; ".join(msgs) if msgs else None


def team_form(hist: pd.DataFrame, team: str, ref: pd.Timestamp, n: int = 10) -> dict | None:
    """Statistiques des n derniers matchs d'une équipe (toutes divisions collectées)."""
    g = hist[((hist["HomeTeam"] == team) | (hist["AwayTeam"] == team)) & (hist["Date"] < ref)]
    g = g.sort_values("Date").tail(n)
    if g.empty:
        return None
    home = g["HomeTeam"] == team
    gf = np.where(home, g["FTHG"], g["FTAG"]).astype(int)
    ga = np.where(home, g["FTAG"], g["FTHG"]).astype(int)
    tot = gf + ga
    res = "".join("V" if a > b else "N" if a == b else "D" for a, b in zip(gf, ga))
    return {"n": len(g), "bilan": res, "bm": round(float(gf.mean()), 1), "be": round(float(ga.mean()), 1),
            "marque": int((gf > 0).sum()), "encaisse": int((ga > 0).sum()), "o15": int((tot > 1).sum()),
            "u35": int((tot < 4).sum()), "o25": int((tot > 2).sum()), "btts": int(((gf > 0) & (ga > 0)).sum())}


def explain(label: str, m: dict, mm: dict) -> str:
    """Pourquoi ce pari : proba, prix à exiger, chiffres de forme pertinents."""
    home, away = m["match"].split(" - ", 1)
    fh, fa = m.get("forme", {}).get("dom"), m.get("forme", {}).get("ext")
    lb = m["lambda_boost"]
    src = (f"cotes du marché : {mm.get('source_marche')}, marge retirée" if mm["proba_marche"] is not None
           else "modèle calibré, testé sur 2025-26 ; pas de cote marché disponible")
    parts = [f"Proba {mm['proba']:.0%} ({src}) → cote juste {mm['cote_juste']:.2f}. "
             f"Value seulement si Winamax paie au moins {mm['cote_juste'] * 1.03:.2f} (+3 %)."]
    if mm["proba_marche"] is not None:
        ec = mm["proba_modele"] - mm["proba_marche"]
        txt = f"Modèle maison : {mm['proba_modele']:.0%} (écart {ec * 100:+.0f} points avec le marché)"
        if abs(ec) >= 0.08:
            txt += " : écart important, souvent signe d'une info (absence, rotation) que le modèle ignore"
        parts.append(txt + ".")
    parts.append(f"Buts attendus {home} {lb['dom']:.1f} – {lb['ext']:.1f} {away} (calés sur les cotes du marché).")
    if fh and fa:
        def f(t, x):
            return f"{t} ({x['n']} derniers matchs : {x['bilan']})"
        if label.startswith("Dom."):
            parts.append(f"{f(home, fh)} a marqué dans {fh['marque']}/{fh['n']} matchs ({fh['bm']} but/match) ; "
                         f"{away} a encaissé dans {fa['encaisse']}/{fa['n']} ({fa['be']} but/match).")
        elif label.startswith("Ext."):
            parts.append(f"{f(away, fa)} a marqué dans {fa['marque']}/{fa['n']} matchs ({fa['bm']} but/match) ; "
                         f"{home} a encaissé dans {fh['encaisse']}/{fh['n']} ({fh['be']} but/match).")
        elif "marquent" in label:
            parts.append(f"Les 2 équipes ont marqué dans {fh['btts']}/{fh['n']} matchs de {home} et "
                         f"{fa['btts']}/{fa['n']} de {away}.")
        elif "buts" in label:
            key = {"+1.5": "o15", "-3.5": "u35", "+2.5": "o25"}.get(label.split()[0])
            if key:
                txt = {"o15": "plus de 1,5 but", "u35": "moins de 3,5 buts", "o25": "plus de 2,5 buts"}[key]
                parts.append(f"{txt.capitalize()} dans {fh[key]}/{fh['n']} matchs de {home} et {fa[key]}/{fa['n']} de {away}.")
            elif label.startswith("-2.5"):
                parts.append(f"Moins de 2,5 buts dans {fh['n'] - fh['o25']}/{fh['n']} matchs de {home} et "
                             f"{fa['n'] - fa['o25']}/{fa['n']} de {away}.")
            elif label.startswith("-1.5") or label.startswith("+3.5"):
                parts.append(f"Moyenne de buts : {home} {fh['bm'] + fh['be']:.1f}/match, {away} {fa['bm'] + fa['be']:.1f}/match.")
        else:
            parts.append(f"Forme : {f(home, fh)}, {fh['bm']}-{fh['be']} buts/match ; {f(away, fa)}, {fa['bm']}-{fa['be']}.")
    if m.get("signaux"):
        parts.append("Signal : " + m["signaux"] + ".")
    parts.append("Absences et compositions : voir la vérification de contexte (6h, 12h, 18h).")
    return re.sub(r"(\d)\.(\d)", r"\1,\2", " ".join(parts))


def to_dashboard_rows(res: dict) -> pd.DataFrame:
    """Une ligne par match × marché, pour le tableau de bord Claude."""
    rows = []
    for m in res["matchs"]:
        date, _, heure = m["coup_envoi"].partition(" ")
        for mm in m["marches"]:
            if not mm["cote_juste"] or any(x in mm["marche"] for x in CONFIG["marches_exclus"]):
                continue
            rows.append({
                "sport": "Football",
                "id": f"{m['div']}|{m['match']}|{mm['marche']}",
                "date": date, "heure": heure or "",
                "championnat": m["championnat"], "match": m["match"],
                "famille": famille(mm["marche"]), "marche": mm["marche"],
                "proba": round(mm["proba"], 4),
                "cote_juste": mm["cote_juste"],
                "cote_minimum": mm["cote_minimum"],
                "cote_moyenne_marche": mm["cote_moyenne_marche"],
                "source_cote": "marché" if mm["proba_marche"] is not None else "modèle",
                "fiabilite": m["fiabilite"],
                "infos": f"Buts attendus {m['lambda_boost']['dom']:.1f} – {m['lambda_boost']['ext']:.1f}",
                "lam_dom": m["lambda_boost"]["dom"], "lam_ext": m["lambda_boost"]["ext"],
                "lam_source": m["lambda_boost"]["source"],
                "explication": explain(mm["marche"], m, mm),
                "signaux": m.get("signaux", ""),
                "proba_modele": mm["proba_modele"],
                "ecart_modele": round(mm["proba_modele"] - mm["proba_marche"], 4) if mm["proba_marche"] is not None else None,
            })
    return pd.DataFrame(rows)


def main() -> None:
    today = dt.datetime.now(ZoneInfo("Europe/Paris")).date()
    res = analyse(today)
    (OUT / "picks").mkdir(parents=True, exist_ok=True)
    js = json.dumps(res, ensure_ascii=False, indent=1, default=str)
    (OUT / "picks" / f"{today.isoformat()}.json").write_text(js, encoding="utf-8")
    (OUT / "latest.json").write_text(js, encoding="utf-8")
    (OUT / "latest.md").write_text(to_markdown(res), encoding="utf-8")
    to_dashboard_rows(res).to_csv(OUT / "dashboard_foot.csv", index=False)
    print(f"{len(res['matchs'])} matchs analysés, {len(res['selection'])} paris retenus")


if __name__ == "__main__":
    main()
