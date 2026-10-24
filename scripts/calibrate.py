"""Calibration du modèle à partir du test sur la saison passée.

1. Validation honnête : on apprend la correction sur nov.-fév., on la teste sur mars-juin.
2. Si c'est concluant, on réapprend sur toute la période et on enregistre
   data/calibration.json, utilisé chaque matin par model.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).parent))
import backtest as B  # noqa: E402
import model as M  # noqa: E402

SPLIT = pd.Timestamp("2026-03-01")


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def fit(df: pd.DataFrame) -> dict:
    coefs = {}
    for label, g in df.groupby("marche"):
        if len(g) < 300 or g["issue"].nunique() < 2:
            continue
        lr = LogisticRegression(C=1e4).fit(logit(g["p_modele"]).to_numpy()[:, None], g["issue"])
        coefs[label] = [round(float(lr.intercept_[0]), 4), round(float(lr.coef_[0][0]), 4)]
    return coefs


def apply(df: pd.DataFrame, coefs: dict) -> pd.Series:
    return pd.Series([M.calibrate(l, p, coefs) for l, p in zip(df["marche"], df["p_modele"])],
                     index=df.index)


def evaluate(test: pd.DataFrame) -> dict:
    t = test.copy()
    t["famille"] = t["marche"].map(B.family)
    w = M.CONFIG["poids_modele"]
    t["p_final"] = np.where(t["p_marche"].notna(), w * t["p_cal"] + (1 - w) * t["p_marche"], t["p_cal"])
    out = {}
    for fam, g in t.groupby("famille"):
        d = {"brier_brut": round(float(((g.p_modele - g.issue) ** 2).mean()), 4),
             "brier_calibre": round(float(((g.p_cal - g.issue) ** 2).mean()), 4)}
        gm = g.dropna(subset=["p_marche"])
        if len(gm):
            d["brier_marche"] = round(float(((gm.p_marche - gm.issue) ** 2).mean()), 4)
            d["brier_final"] = round(float(((gm.p_final - gm.issue) ** 2).mean()), 4)
        bins = pd.cut(g["p_cal"], [0, .3, .4, .5, .6, .7, .8, .9, 1])
        cal = g.groupby(bins, observed=True).agg(annonce=("p_cal", "mean"), reel=("issue", "mean"),
                                                  n=("issue", "size"))
        d["calibration"] = [f"{r.annonce:.0%} annoncé → {r.reel:.0%} réel (n={int(r.n)})"
                            for _, r in cal.iterrows() if r.n >= 30]
        out[fam] = d
    # paris simulés sur la période de test
    b = t.dropna(subset=["cote"]).copy()
    b["mini"] = (1 / b["p_final"]) * (1 + M.CONFIG["marge_value"])
    sel = b[(b["cote"] >= b["mini"]) & b["mini"].between(M.CONFIG["cote_min"], M.CONFIG["cote_max"])]
    gain = np.where(sel["issue"] == 1, sel["cote"] - 1, -1.0)
    sim = {"paris": int(len(sel))}
    if len(sel):
        sim.update({"reussite": round(float(sel["issue"].mean()), 3),
                    "cote_moy": round(float(sel["cote"].mean()), 2),
                    "rendement": round(float(gain.mean()), 4),
                    "par_marche": {k: {"paris": int(len(v)),
                                       "rendement": round(float(np.where(v.issue == 1, v.cote - 1, -1).mean()), 3)}
                                   for k, v in sel.groupby("marche")}})
    out["paris_simules_mars_juin"] = sim
    return out


if __name__ == "__main__":
    df = B.run()
    train, test = df[df["date"] < SPLIT].copy(), df[df["date"] >= SPLIT].copy()
    coefs_train = fit(train)
    test["p_cal"] = apply(test, coefs_train)
    report = {"validation": evaluate(test)}
    coefs = fit(df)
    report["coefficients"] = coefs
    (M.OUT / "calibration.json").write_text(json.dumps(coefs, ensure_ascii=False, indent=1))
    (M.OUT / "backtest").mkdir(parents=True, exist_ok=True)
    (M.OUT / "backtest" / "validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps(report["validation"], ensure_ascii=False, indent=1))
