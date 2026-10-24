"""Modèle tennis : Elo par surface + simulation jeu par jeu.

1. Elo (méthode FiveThirtyEight) : un classement général et un par surface, mélangés 50/50.
   → probabilité que chaque joueur gagne le match.
2. On traduit cette probabilité en taux de jeux de service gagnés pour chaque joueur
   (autour de la moyenne du circuit et de la surface), puis on simule le match
   jeu par jeu, tie-breaks compris.
   → score en sets, au moins un set, total de jeux, handicap jeux.
3. Une correction de calibration apprise sur l'historique (tennis_calibrate.py)
   est appliquée à chaque marché.
Aucune cote marché n'étant disponible gratuitement, toutes les cotes justes viennent du modèle.
"""
from __future__ import annotations

import datetime as dt
import json
import math
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "tennis"
OUT = ROOT / "data"
PARIS = ZoneInfo("Europe/Paris")

CONFIG = {
    "marge_value": 0.05,
    "matchs_min_fiable": 8,       # matchs joués par chaque joueur dans l'historique
    "matchs_bon_echantillon": 20,
    "poids_surface": 0.5,
}

# Taux moyen de jeux de service gagnés (valeurs de départ, réajustées par tennis_calibrate.py)
HOLD_DEFAUT = {("ATP", "dur"): 0.80, ("ATP", "terre"): 0.77, ("ATP", "gazon"): 0.84, ("ATP", "salle"): 0.83,
               ("WTA", "dur"): 0.66, ("WTA", "terre"): 0.63, ("WTA", "gazon"): 0.69, ("WTA", "salle"): 0.68}

CLAY = ["roland garros", "french open", "monte", "madrid", "internazionali", "italian", "rome", "roma",
        "barcelona", "hamburg", "buenos aires", "rio open", "santiago", "estoril", "houston", "marrakech",
        "bucharest", "tiriac", "munich", "bmw open", "geneva", "lyon", "bastad", "nordea", "gstaad", "umag",
        "kitzb", "cordoba", "córdoba", "charleston", "bogota", "bogotá", "rabat", "strasbourg", "palermo",
        "prague", "iasi", "warsaw", "parma", "rouen", "budapest", "portoroz", "lausanne", "bad gastein",
        "makarska", "la bisbal", "catalonia", "oeiras", "sao paulo", "são paulo", "rio ladies", "mundotênis",
        "florianopolis", "tigre", "concepcion", "sassuolo", "aix", "bordeaux", "heilbronn", "perugia",
        "francavilla", "sardegna", "cagliari", "san marino", "todi", "trieste", "verona", "salzburg",
        "braunschweig", "zagreb", "skopje", "tunis", "mauthausen", "ostrava", "puglie", "brasil"]
GRASS = ["wimbledon", "halle", "queen", "hertogenbosch", "libema", "eastbourne", "mallorca", "stuttgart open",
         "boss open", "birmingham", "bad homburg", "berlin", "nottingham", "newport", "hall of fame",
         "ilkley", "surbiton"]


INDOOR = ["european open", "swiss indoors", "basel", "erste bank", "vienna", "rolex paris", "paris masters",
          "stockholm", "moselle", "metz", "nitto atp finals", "atp finals", "wta finals", "next gen",
          "rotterdam", "abn amro", "open 13", "marseille", "montpellier", "occitanie", "dallas", "auvergne",
          "almaty", "linz", "ostrava", "cluj", "transylvania", "st. petersburg", "sofia", "astana",
          "davis cup", "billie jean king", "united cup", "antwerp", "belgrade", "helsinki", "bergamo",
          "mouilleron", "brest", "rennes", "eckental", "ismaning", "koblenz", "quimper", "lille"]


def surface_of(event: str, date: str) -> str:
    e = (event or "").lower()
    month = int(str(date)[5:7]) if date else 1
    if any(k in e for k in INDOOR):
        return "salle"
    if any(k in e for k in GRASS) and month in (6, 7):
        return "gazon"
    if any(k in e for k in CLAY):
        return "terre"
    return "dur"


ROUNDS = {"Final": "Finale", "Semifinal": "Demi-finale", "Quarterfinal": "Quart de finale",
          "Round 1": "1er tour", "Round 2": "2e tour", "Round 3": "3e tour", "Round 4": "8e de finale",
          "Round of 16": "8e de finale", "Round of 32": "16e de finale", "Round of 64": "32e de finale",
          "Round of 128": "1er tour", "Qualifying 1st Round": "Qualifications T1",
          "Qualifying 2nd Round": "Qualifications T2", "Qualifying 3rd Round": "Qualifications T3",
          "Qualifying Final": "Qualifications finale"}

SLAMS = ["australian open", "roland garros", "french open", "wimbledon", "us open"]


def best_of(tour: str, event: str, rnd: str) -> int:
    e, r = (event or "").lower(), (rnd or "").lower()
    if tour == "ATP" and any(s in e for s in SLAMS) and "qualif" not in r:
        return 5
    return 3


# ------------------------------------------------------------------ Elo

class Elo:
    def __init__(self):
        self.r, self.n = {}, {}
        self.rs, self.ns = {}, {}

    @staticmethod
    def k(n):
        return 250 / (n + 5) ** 0.4

    def prob(self, a, b, surf):
        w = CONFIG["poids_surface"]
        ra = (1 - w) * self.r.get(a, 1500) + w * self.rs.get((a, surf), 1500)
        rb = (1 - w) * self.r.get(b, 1500) + w * self.rs.get((b, surf), 1500)
        return 1 / (1 + 10 ** ((rb - ra) / 400))

    def update(self, w, l, surf):
        for R, N, kw, kl in ((self.r, self.n, w, l), (self.rs, self.ns, (w, surf), (l, surf))):
            rw, rl = R.get(kw, 1500), R.get(kl, 1500)
            e = 1 / (1 + 10 ** ((rl - rw) / 400))
            R[kw] = rw + self.k(N.get(kw, 0)) * (1 - e)
            R[kl] = rl - self.k(N.get(kl, 0)) * (1 - e)
            N[kw] = N.get(kw, 0) + 1
            N[kl] = N.get(kl, 0) + 1

    def played(self, a):
        return self.n.get(a, 0)


# ------------------------------------------------------------------ simulation jeu par jeu

def game_prob(p: float) -> float:
    q = 1 - p
    return p ** 4 * (1 + 4 * q + 10 * q ** 2) + 20 * p ** 3 * q ** 3 * p ** 2 / (1 - 2 * p * q)


@lru_cache(maxsize=4096)
def point_from_hold(h: float) -> float:
    lo, hi = 0.2, 0.995
    for _ in range(50):
        mid = (lo + hi) / 2
        if game_prob(mid) < h:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


@lru_cache(maxsize=4096)
def tiebreak(pa: float, pb: float, a_first: bool) -> float:
    """Proba que A gagne le tie-break (pa, pb : points gagnés au service)."""
    @lru_cache(maxsize=None)
    def f(x, y):
        if x >= 7 and x - y >= 2:
            return 1.0
        if y >= 7 and y - x >= 2:
            return 0.0
        if x == y and x >= 6:
            win, lose = pa * (1 - pb), (1 - pa) * pb
            return win / (win + lose)
        n = x + y
        a_serves = ((n % 4) in (0, 3)) == a_first
        p = pa if a_serves else 1 - pb
        return p * f(x + 1, y) + (1 - p) * f(x, y + 1)
    return f(0, 0)


def set_dist(ha: float, hb: float, a_first: bool) -> dict:
    """Distribution des scores finaux d'un set : {(jeux A, jeux B): proba}."""
    pa, pb = point_from_hold(ha), point_from_hold(hb)
    tb = tiebreak(round(pa, 4), round(pb, 4), a_first)
    out = {}
    states = {(0, 0): 1.0}
    while states:
        nxt = {}
        for (x, y), pr in states.items():
            if (x >= 6 and x - y >= 2) or x == 7:
                out[(x, y)] = out.get((x, y), 0) + pr
                continue
            if (y >= 6 and y - x >= 2) or y == 7:
                out[(x, y)] = out.get((x, y), 0) + pr
                continue
            if x == 6 and y == 6:
                out[(7, 6)] = out.get((7, 6), 0) + pr * tb
                out[(6, 7)] = out.get((6, 7), 0) + pr * (1 - tb)
                continue
            a_serves = ((x + y) % 2 == 0) == a_first
            p = ha if a_serves else 1 - hb
            nxt[(x + 1, y)] = nxt.get((x + 1, y), 0) + pr * p
            nxt[(x, y + 1)] = nxt.get((x, y + 1), 0) + pr * (1 - p)
        states = nxt
    return out


def match_dist(ha: float, hb: float, bo: int) -> dict:
    """{(sets A, sets B, total jeux, écart jeux A-B): proba}, serveur initial tiré au sort."""
    need = bo // 2 + 1
    sd = {True: set_dist(ha, hb, True), False: set_dist(ha, hb, False)}
    final = {}
    for first in (True, False):
        states = {(0, 0, first, 0, 0): 0.5}
        while states:
            nxt = {}
            for (sa, sb, fs, tg, gd), pr in states.items():
                if sa == need or sb == need:
                    k = (sa, sb, tg, gd)
                    final[k] = final.get(k, 0) + pr
                    continue
                for (x, y), q in sd[fs].items():
                    nfs = fs if (x + y) % 2 == 0 else not fs
                    k = (sa + (x > y), sb + (y > x), nfs, tg + x + y, gd + x - y)
                    nxt[k] = nxt.get(k, 0) + pr * q
            states = nxt
    return final


def fit_holds(p_win: float, base: float, bo: int) -> tuple[float, float]:
    lo, hi = -0.35, 0.35
    for _ in range(30):
        d = (lo + hi) / 2
        ha, hb = min(max(base + d, 0.3), 0.97), min(max(base - d, 0.3), 0.97)
        pw = sum(v for (sa, sb, _, _), v in match_dist(round(ha, 3), round(hb, 3), bo).items() if sa > sb)
        if pw < p_win:
            lo = d
        else:
            hi = d
    d = (lo + hi) / 2
    return round(min(max(base + d, 0.3), 0.97), 3), round(min(max(base - d, 0.3), 0.97), 3)


@lru_cache(maxsize=20000)
def tiebreak_prob(p_win: float, tb_base: float, bo: int) -> float:
    """Proba d'au moins un tie-break, avec un taux de jeux de service réaliste pour le circuit
    et la surface (les tie-breaks dépendent du niveau de service, que le total de jeux ne suffit pas à fixer)."""
    ha, hb = fit_holds(round(p_win, 2), tb_base, bo)
    dist_no_tb = {}
    need = bo // 2 + 1
    no_tb = 0.0
    for first in (True, False):
        sd = set_dist(ha, hb, first)
        sdo = set_dist(ha, hb, not first)
        # probabilité qu'un set se termine sans tie-break, selon qui sert en premier
        q = {True: sum(v for (x, y), v in sd.items() if max(x, y) < 7 or abs(x - y) == 2 and max(x, y) == 7),
             False: sum(v for (x, y), v in sdo.items() if max(x, y) < 7 or abs(x - y) == 2 and max(x, y) == 7)}
        dist_no_tb[first] = q
    # approximation : sets indépendants, serveur alterné → moyenne des deux cas
    q_avg = (dist_no_tb[True][True] + dist_no_tb[True][False]) / 2
    md = match_dist(ha, hb, bo)
    for (sa, sb, _, _), v in md.items():
        no_tb += v * q_avg ** (sa + sb)
    return 1 - no_tb


def markets(p_win: float, base: float, bo: int, a: str, b: str, tb_base: float | None = None) -> tuple[dict, dict]:
    ha, hb = fit_holds(p_win, base, bo)
    dist = match_dist(ha, hb, bo)
    need = bo // 2 + 1
    tg = np.array([k[2] for k in dist]); gd = np.array([k[3] for k in dist])
    sa = np.array([k[0] for k in dist]); sb = np.array([k[1] for k in dist])
    pr = np.array(list(dist.values()))
    m = {f"Vainqueur : {a}": pr[sa > sb].sum(), f"Vainqueur : {b}": pr[sb > sa].sum()}
    for x in range(need):
        m[f"Score {need}-{x} : {a}"] = pr[(sa == need) & (sb == x)].sum()
        m[f"Score {need}-{x} : {b}"] = pr[(sb == need) & (sa == x)].sum()
    m[f"Au moins un set : {a}"] = pr[sa >= 1].sum()
    m[f"Au moins un set : {b}"] = pr[sb >= 1].sum()
    full = 2 * need - 1
    m[f"Match en {full} sets"] = pr[(sa + sb) == full].sum()
    mean = float((tg * pr).sum())
    centre = math.floor(mean) + 0.5
    for line in (centre - 2, centre - 1, centre, centre + 1, centre + 2):
        m[f"+{line:g} jeux"] = pr[tg > line].sum()
        m[f"-{line:g} jeux"] = pr[tg < line].sum()
    fav, dog, sign = (a, b, 1) if p_win >= 0.5 else (b, a, -1)
    for h in (1.5, 2.5, 3.5, 4.5, 5.5):
        p = pr[sign * gd > h].sum()
        m[f"Handicap {fav} -{h:g} jeux"] = p
        m[f"Handicap {dog} +{h:g} jeux"] = 1 - p
    if tb_base is not None:
        m["Au moins un tie-break"] = tiebreak_prob(p_win, tb_base, bo)
    info = {"jeux_attendus": round(mean, 1), "hold_a": ha, "hold_b": hb}
    return {k: float(v) for k, v in m.items()}, info


def market_family(label: str) -> str:
    if label.startswith("Vainqueur"):
        return "Vainqueur"
    if label.startswith(("Score", "Match en", "Au moins")):
        return "Sets"
    if label.startswith("Handicap"):
        return "Handicap jeux"
    if "jeux" in label:
        return "Total jeux"
    return "Autre"


def calib_key(label: str) -> str:
    """Clé de calibration commune à tous les joueurs (le nom est retiré)."""
    if label.startswith("Vainqueur"):
        return "Vainqueur"
    if label.startswith("Au moins un set"):
        return "Au moins un set"
    if label.startswith("Score"):
        return label.split(" :")[0]
    if label.startswith("Handicap"):
        return "Handicap " + ("fav " if " -" in label else "outsider ") + label.split()[-2]
    if label.startswith(("Match en", "Au moins un tie")):
        return label
    if "jeux" in label:
        return "Total jeux " + label[0]
    return label


# ------------------------------------------------------------------ données

def load_matches() -> pd.DataFrame:
    f = RAW / "matches.csv"
    if not f.exists():
        return pd.DataFrame()
    m = pd.read_csv(f, dtype={"id": str, "p1_id": str, "p2_id": str})
    m["date"] = pd.to_datetime(m["date"], utc=True, errors="coerce")
    m = m.dropna(subset=["date"]).sort_values("date")
    m["surface"] = [surface_of(e, d) for e, d in zip(m["event"], m["date"].astype(str))]
    return m


def parse_sets(s: str) -> list[tuple[int, int]]:
    out = []
    for part in str(s).split():
        try:
            x, y = part.split("-")
            out.append((int(x), int(y)))
        except ValueError:
            pass
    return out


def build_elo(matches: pd.DataFrame, until=None) -> Elo:
    elo = Elo()
    for r in matches.itertuples():
        if until is not None and r.date >= until:
            break
        if r.status != "STATUS_FINAL" or r.winner not in (1, 2):
            continue
        w, l = (r.p1_id, r.p2_id) if r.winner == 1 else (r.p2_id, r.p1_id)
        elo.update(w, l, r.surface)
    return elo


def load_json(name: str) -> dict:
    try:
        return json.loads((OUT / name).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def calibrate(key: str, p: float, coefs: dict) -> float:
    c = coefs.get(key)
    if not c:
        return p
    p = min(max(p, 1e-4), 1 - 1e-4)
    z = c[0] + c[1] * math.log(p / (1 - p))
    return 1 / (1 + math.exp(-z))


# ------------------------------------------------------------------ prédictions du jour

def player_form(matches: pd.DataFrame, pid: str, n: int = 10) -> dict | None:
    g = matches[((matches["p1_id"] == pid) | (matches["p2_id"] == pid)) & (matches["status"] == "STATUS_FINAL")].tail(n)
    if g.empty:
        return None
    won, tb, games, sets_lost = [], 0, [], 0
    for r in g.itertuples():
        me = 1 if r.p1_id == pid else 2
        won.append(r.winner == me)
        ss = parse_sets(r.sets)
        tb += any(max(x, y) == 7 and min(x, y) == 6 for x, y in ss)
        games.append(sum(x + y for x, y in ss))
    return {"n": len(g), "bilan": "".join("V" if w else "D" for w in won), "v": sum(won),
            "tb": tb, "jeux": round(float(np.mean(games)), 1)}


def fatigue_load(matches: pd.DataFrame, pid: str, kickoff: pd.Timestamp) -> dict:
    """Charge des 3 derniers jours : nombre de matchs, jeux joués, abandon au dernier match."""
    k = kickoff.tz_convert("UTC") if kickoff.tzinfo else kickoff.tz_localize("UTC")
    g = matches[((matches["p1_id"] == pid) | (matches["p2_id"] == pid)) & (matches["date"] < k - pd.Timedelta(hours=2))]
    if g.empty:
        return {"n": 0, "jeux": 0, "abandon": False}
    last = g.iloc[-1]
    me = 1 if last["p1_id"] == pid else 2
    recent = g[g["date"] >= k - pd.Timedelta(days=3)]
    return {"n": len(recent), "jeux": sum(sum(x + y for x, y in parse_sets(s)) for s in recent["sets"]),
            "abandon": last["status"] == "STATUS_RETIRED" and last["winner"] != me}


def fatigue_signal(matches: pd.DataFrame, a: tuple, b: tuple, kickoff: pd.Timestamp) -> str:
    """Ne signale que ce qui distingue les deux joueurs (jouer tous les jours est normal en tournoi)."""
    la, lb = fatigue_load(matches, a[0], kickoff), fatigue_load(matches, b[0], kickoff)
    msgs = []
    for (pid, name), l in ((a, la), (b, lb)):
        if l["abandon"]:
            msgs.append(f"{name} a abandonné lors de son dernier match")
    for (na, l1), (nb, l2) in (((a[1], la), (b[1], lb)), ((b[1], lb), (a[1], la))):
        if l1["jeux"] - l2["jeux"] >= 15:
            msgs.append(f"{na} a joué {l1['jeux']} jeux en 3 jours contre {l2['jeux']} pour {nb}")
        if l1["n"] >= 3 and l2["n"] < 3:
            msgs.append(f"{na} dispute son {l1['n'] + 1}e match en 4 jours")
    return " ; ".join(msgs)


def explain_tennis(label: str, pc: float, fair: float, a: str, b: str, fa: dict | None, fb: dict | None,
                   surf: str, p_fav: float, fav: str) -> str:
    parts = [f"Proba {pc:.0%} (modèle Elo + simulation jeu par jeu, calibré sur 15 000 matchs) → cote juste {fair:.2f}. "
             f"Value seulement si Winamax paie au moins {fair * 1.03:.2f} (+3 %).",
             f"{fav} favori à {p_fav:.0%} sur {('dur en salle' if surf == 'salle' else surf)} (Elo par surface)."]
    if fa and fb:
        parts.append(f"Forme : {a} {fa['v']}/{fa['n']} victoires ({fa['bilan']}), {b} {fb['v']}/{fb['n']} ({fb['bilan']}).")
        if "tie-break" in label:
            parts.append(f"Au moins un tie-break dans {fa['tb']}/{fa['n']} matchs de {a} et {fb['tb']}/{fb['n']} de {b}.")
        elif "jeux" in label:
            parts.append(f"Jeux par match sur 10 matchs : {a} {fa['jeux']}, {b} {fb['jeux']}.")
    parts.append("Limite : le modèle ne connaît pas les statistiques de service de chaque joueur ; "
                 "blessures et forfaits : voir la vérification de contexte (6h, 12h, 18h).")
    import re
    return re.sub(r"(\d)\.(\d)", r"\1,\2", " ".join(parts))


def predict_today() -> list[dict]:
    matches = load_matches()
    sch_f = RAW / "schedule.csv"
    if matches.empty or not sch_f.exists():
        return []
    sch = pd.read_csv(sch_f, dtype={"id": str, "p1_id": str, "p2_id": str})
    if sch.empty:
        return []
    elo = build_elo(matches)
    holds = {tuple(k.split("|")): v for k, v in load_json("tennis_params.json").get("hold", {}).items()}
    coefs = load_json("tennis_calibration.json")
    rows = []
    today = dt.datetime.now(PARIS).date()
    for r in sch.itertuples():
        t = pd.Timestamp(r.date).tz_convert(PARIS)
        if not (today <= t.date() <= today + dt.timedelta(days=1)):
            continue
        n = min(elo.played(r.p1_id), elo.played(r.p2_id))
        if n < CONFIG["matchs_min_fiable"]:
            continue
        surf = surface_of(r.event, str(r.date))
        bo = best_of(r.tour, r.event, r.round)
        p = elo.prob(r.p1_id, r.p2_id, surf)
        base = holds.get((r.tour, surf), HOLD_DEFAUT[(r.tour, surf)])
        mk, info = markets(p, base, bo, r.p1, r.p2, HOLD_DEFAUT[(r.tour, surf)])
        fiab = "bonne" if n >= CONFIG["matchs_bon_echantillon"] else "moyenne"
        p_fav = calibrate("Vainqueur", max(p, 1 - p), coefs)
        fav = r.p1 if p >= 0.5 else r.p2
        fa, fb = player_form(matches, r.p1_id), player_form(matches, r.p2_id)
        ko = pd.Timestamp(r.date)
        signaux = fatigue_signal(matches, (r.p1_id, r.p1), (r.p2_id, r.p2), ko)
        infos = f"{'Dur en salle' if surf == 'salle' else surf.capitalize()} · {ROUNDS.get(r.round, r.round)} · {fav} favori à {p_fav:.0%}"
        for label, praw in mk.items():
            key = calib_key(label) + (f" {r.tour}" if label == "Au moins un tie-break" else "")
            pc = calibrate(key, praw, coefs)
            if pc <= 0.01 or pc >= 0.99:
                continue
            fair = 1 / pc
            rows.append({
                "sport": "Tennis",
                "id": f"T|{r.id}|{label}",
                "date": t.strftime("%Y-%m-%d"), "heure": t.strftime("%H:%M"),
                "championnat": f"{r.tour} · {r.event}", "match": f"{r.p1} - {r.p2}",
                "famille": market_family(label), "marche": label,
                "proba": round(pc, 4), "cote_juste": round(fair, 2),
                "cote_minimum": round(fair * (1 + CONFIG["marge_value"]), 2),
                "cote_moyenne_marche": None, "source_cote": "modèle", "fiabilite": fiab,
                "infos": infos,
                "explication": explain_tennis(label, pc, fair, r.p1, r.p2, fa, fb, surf, p_fav, fav)
                               + (f" Signal : {signaux}." if signaux else ""),
                "signaux": signaux,
            })
    return rows


def main() -> None:
    rows = predict_today()
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "dashboard_tennis.csv", index=False)
    print(f"tennis : {df['match'].nunique() if len(df) else 0} matchs, {len(df)} cotes justes")


if __name__ == "__main__":
    main()
