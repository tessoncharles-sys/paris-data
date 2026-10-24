"""Test du modèle tennis sur l'historique et calibration.

- Elo rejoué dans l'ordre chronologique : chaque prédiction n'utilise que les matchs passés.
- Taux de jeux de service moyen par circuit et surface ajusté pour que le nombre moyen
  de jeux simulé colle au nombre réel.
- Correction de calibration apprise sur 2025, vérifiée sur 2026, puis réapprise sur tout.
"""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).parent))
import tennis as T  # noqa: E402

START = pd.Timestamp("2025-01-01", tz="UTC")
SPLIT = pd.Timestamp("2026-01-01", tz="UTC")


@lru_cache(maxsize=200000)
def mk(p: float, base: float, bo: int):
    return T.markets(p, base, bo, "A", "B")


def collect_rows() -> pd.DataFrame:
    m = T.load_matches()
    elo = T.Elo()
    rows = []
    for r in m.itertuples():
        if r.status != "STATUS_FINAL" or r.winner not in (1, 2):
            continue
        sets = T.parse_sets(r.sets)
        if r.date >= START and sets and min(elo.played(r.p1_id), elo.played(r.p2_id)) >= T.CONFIG["matchs_min_fiable"]:
            bo = T.best_of(r.tour, r.event, r.round)
            sa = sum(x > y for x, y in sets); sb = sum(y > x for x, y in sets)
            need = bo // 2 + 1
            if max(sa, sb) == need:   # match complet et cohérent
                rows.append({"date": r.date, "tour": r.tour, "surface": r.surface, "bo": bo,
                             "p": round(elo.prob(r.p1_id, r.p2_id, r.surface), 2),
                             "win": int(r.winner == 1), "sa": sa, "sb": sb,
                             "tg": sum(x + y for x, y in sets), "gd": sum(x - y for x, y in sets),
                             "n": min(elo.played(r.p1_id), elo.played(r.p2_id))})
        w, l = (r.p1_id, r.p2_id) if r.winner == 1 else (r.p2_id, r.p1_id)
        elo.update(w, l, r.surface)
    return pd.DataFrame(rows)


def fit_bases(df: pd.DataFrame) -> dict:
    bases = {}
    for (tour, surf), g in df[df.bo == 3].groupby(["tour", "surface"]):
        g = g.sample(min(len(g), 1500), random_state=1)
        target = g["tg"].mean()
        best, err = None, 1e9
        for base in np.arange(0.55, 0.90, 0.01):
            b = round(float(base), 2)
            pred = np.mean([mk(p, b, 3)[1]["jeux_attendus"] for p in g["p"]])
            if abs(pred - target) < err:
                best, err = b, abs(pred - target)
        bases[f"{tour}|{surf}"] = best
        print(f"  {tour} {surf}: {len(g)} matchs, {target:.1f} jeux réels → hold {best}")
    return bases


def outcomes(df: pd.DataFrame, bases: dict) -> pd.DataFrame:
    recs = []
    for r in df.itertuples():
        base = bases.get(f"{r.tour}|{r.surface}", T.HOLD_DEFAUT[(r.tour, r.surface)])
        m, _ = mk(r.p, base, r.bo)
        need = r.bo // 2 + 1
        fav_a = r.p >= 0.5
        gd_fav = r.gd if fav_a else -r.gd
        res = {"Vainqueur : A": r.win,
               "Au moins un set : A": int(r.sa >= 1), "Au moins un set : B": int(r.sb >= 1),
               f"Match en {2 * need - 1} sets": int(r.sa + r.sb == 2 * need - 1)}
        for x in range(need):
            res[f"Score {need}-{x} : A"] = int(r.sa == need and r.sb == x)
        for label in m:
            if label.startswith(("+", "-")) and "jeux" in label:
                line = float(label[1:].split()[0])
                res[label] = int(r.tg > line) if label[0] == "+" else int(r.tg < line)
            if label.startswith("Handicap"):
                h = float(label.split()[-2][1:])
                fav_line = " -" in label
                res[label] = int(gd_fav > h) if fav_line else int(gd_fav < h)
        for label, o in res.items():
            if label in m:
                recs.append({"date": r.date, "tour": r.tour, "key": T.calib_key(label),
                             "famille": T.market_family(label), "p": m[label], "issue": o})
    return pd.DataFrame(recs)


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def fit(o: pd.DataFrame) -> dict:
    co = {}
    for k, g in o.groupby("key"):
        if len(g) >= 300 and g["issue"].nunique() == 2:
            lr = LogisticRegression(C=1e4).fit(logit(g["p"]).to_numpy()[:, None], g["issue"])
            co[k] = [round(float(lr.intercept_[0]), 4), round(float(lr.coef_[0][0]), 4)]
    return co


def report(test: pd.DataFrame) -> dict:
    out = {}
    for fam, g in test.groupby("famille"):
        bins = pd.cut(g["p_cal"], [0, .2, .3, .4, .5, .6, .7, .8, .9, 1])
        cal = g.groupby(bins, observed=True).agg(a=("p_cal", "mean"), r=("issue", "mean"), n=("issue", "size"))
        out[fam] = {
            "predictions": int(len(g)),
            "brier_brut": round(float(((g.p - g.issue) ** 2).mean()), 4),
            "brier_calibre": round(float(((g.p_cal - g.issue) ** 2).mean()), 4),
            "calibration": [f"{x.a:.0%} annoncé → {x.r:.0%} réel (n={int(x.n)})" for _, x in cal.iterrows() if x.n >= 50],
        }
    return out


if __name__ == "__main__":
    df = collect_rows()
    print(f"{len(df)} matchs testés ({df.date.min().date()} → {df.date.max().date()})")
    bases = fit_bases(df[df.date < SPLIT])
    o = outcomes(df, bases)
    train, test = o[o.date < SPLIT].copy(), o[o.date >= SPLIT].copy()
    co_train = fit(train)
    test["p_cal"] = [T.calibrate(k, p, co_train) for k, p in zip(test.key, test.p)]
    rep = {"matchs": int(len(df)), "hold_moyen": bases,
           "precision_vainqueur_favori_2026": round(float(((df[df.date >= SPLIT].p >= .5) == (df[df.date >= SPLIT].win == 1)).mean()), 3),
           "validation_2026": report(test)}
    bases_all = fit_bases(df)
    o_all = outcomes(df, bases_all)
    co = fit(o_all)
    (T.OUT / "tennis_params.json").write_text(json.dumps({"hold": bases_all}, indent=1))
    (T.OUT / "tennis_calibration.json").write_text(json.dumps(co, ensure_ascii=False, indent=1))
    (T.OUT / "backtest").mkdir(parents=True, exist_ok=True)
    (T.OUT / "backtest" / "tennis_validation.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    print(json.dumps(rep, ensure_ascii=False, indent=1))
