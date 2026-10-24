"""Test du modèle sur la saison passée : on « rejoue » chaque semaine avec
seulement les données connues à ce moment-là, puis on compare aux résultats.

Mesures :
- calibration : quand le modèle dit 60 %, ça arrive-t-il 60 % du temps ?
- précision comparée au marché (score de Brier, plus bas = mieux) ;
- paris simulés : rendement des paris que la sélection aurait retenus
  (mise 1, cote moyenne du marché, dans la fourchette de cotes configurée).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import model as M  # noqa: E402

SEASON_START = pd.Timestamp("2025-11-15")   # on laisse ~3 mois d'historique au modèle
SEASON_END = pd.Timestamp("2026-06-30")


def outcomes(r) -> dict[str, int]:
    hg, ag = r.FTHG, r.FTAG
    o = {"1": hg > ag, "N": hg == ag, "2": hg < ag}
    o.update({"1N": hg >= ag, "12": hg != ag, "N2": hg <= ag})
    for line in (1.5, 2.5, 3.5):
        o[f"+{line} buts"] = hg + ag > line
        o[f"-{line} buts"] = hg + ag < line
    o["Les 2 marquent : oui"] = hg > 0 and ag > 0
    o["Les 2 marquent : non"] = not (hg > 0 and ag > 0)
    o["Dom. +0.5 but"], o["Dom. +1.5 but"] = hg > 0, hg > 1
    o["Ext. +0.5 but"], o["Ext. +1.5 but"] = ag > 0, ag > 1
    if not pd.isna(getattr(r, "HC", np.nan)) and not pd.isna(getattr(r, "AC", np.nan)):
        c = r.HC + r.AC
        for line in (8.5, 9.5, 10.5):
            o[f"+{line} corners"], o[f"-{line} corners"] = c > line, c < line
    if not pd.isna(getattr(r, "HY", np.nan)) and not pd.isna(getattr(r, "AY", np.nan)):
        k = r.HY + r.AY + (0 if pd.isna(getattr(r, "HR", np.nan)) else r.HR) \
            + (0 if pd.isna(getattr(r, "AR", np.nan)) else r.AR)
        for line in (3.5, 4.5):
            o[f"+{line} cartons"], o[f"-{line} cartons"] = k > line, k < line
    return {k: int(v) for k, v in o.items()}


def run() -> pd.DataFrame:
    hist = M.load_history()
    rows = []
    weeks = pd.date_range(SEASON_START, SEASON_END, freq="7D")
    for div in M.DIVISIONS:
        hd = hist[hist["Div"] == div].sort_values("Date")
        for w0 in weeks:
            w1 = w0 + pd.Timedelta(days=7)
            train = hd[hd["Date"] < w0]
            test = hd[(hd["Date"] >= w0) & (hd["Date"] < w1)]
            if len(train) < 60 or test.empty:
                continue
            hg, ag = M.smoothed_goals(train)
            goals = M.TeamModel("FTHG", "FTAG").fit(train, w0, hg, ag)
            tc = train.dropna(subset=["HC", "AC"]) if {"HC", "AC"}.issubset(train) else None
            corners = M.TeamModel("HC", "AC").fit(tc, w0) if tc is not None and len(tc) > 60 else M.TeamModel("HC", "AC")
            if {"HY", "AY"}.issubset(train):
                tk = train.dropna(subset=["HY", "AY"])
                cards = M.TeamModel("HY", "AY").fit(
                    tk, w0,
                    tk["HY"] + (tk["HR"].fillna(0) if "HR" in tk else 0),
                    tk["AY"] + (tk["AR"].fillna(0) if "AR" in tk else 0))
            else:
                cards = M.TeamModel("HY", "AY")
            for _, row in test.iterrows():
                r = row
                n = min(M.recent_count(train, r.HomeTeam, w0), M.recent_count(train, r.AwayTeam, w0))
                lam = goals.predict(r.HomeTeam, r.AwayTeam)
                if lam is None or n < M.CONFIG["matchs_min_fiable"]:
                    continue
                probs = M.goal_markets(*lam)
                c = corners.predict(r.HomeTeam, r.AwayTeam)
                if c:
                    probs.update(M.count_markets(*c, (8.5, 9.5, 10.5), "corners"))
                k = cards.predict(r.HomeTeam, r.AwayTeam)
                if k:
                    probs.update(M.count_markets(*k, (3.5, 4.5), "cartons"))
                mk = M.market_implied(row)
                odds = M.market_odds(row)
                res = outcomes(r)
                for label, p_mod in probs.items():
                    if label not in res:
                        continue
                    w = M.CONFIG["poids_modele"]
                    p = w * p_mod + (1 - w) * mk[label] if label in mk else p_mod
                    rows.append({
                        "div": div, "date": r.Date, "match": f"{r.HomeTeam} - {r.AwayTeam}",
                        "marche": label, "p_modele": p_mod, "p_marche": mk.get(label),
                        "p": p, "cote": odds.get(label), "issue": res[label],
                        "fiabilite": "bonne" if n >= M.CONFIG["matchs_bon_echantillon"] else "moyenne",
                    })
    return pd.DataFrame(rows)


def family(label: str) -> str:
    if label in ("1", "N", "2"):
        return "1X2"
    if label in ("1N", "12", "N2"):
        return "Double chance"
    if "2.5 buts" in label:
        return "+/- 2,5 buts"
    if "buts" in label:
        return "+/- 1,5 et 3,5 buts"
    if "marquent" in label:
        return "Les 2 marquent"
    if "Dom." in label or "Ext." in label:
        return "Buts par équipe"
    if "corners" in label:
        return "Corners"
    if "cartons" in label:
        return "Cartons"
    return "Autre"


def summarise(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["famille"] = df["marche"].map(family)
    out = {"periode": f"{SEASON_START.date()} → {SEASON_END.date()}",
           "matchs": int(df["match"].nunique()), "familles": {}}

    for fam, g in df.groupby("famille"):
        d = {"predictions": int(len(g)),
             "brier_modele": round(float(((g["p_modele"] - g["issue"]) ** 2).mean()), 4)}
        gm = g.dropna(subset=["p_marche"])
        if len(gm):
            d["brier_modele_meme_matchs"] = round(float(((gm["p_modele"] - gm["issue"]) ** 2).mean()), 4)
            d["brier_marche"] = round(float(((gm["p_marche"] - gm["issue"]) ** 2).mean()), 4)
            d["brier_melange"] = round(float(((gm["p"] - gm["issue"]) ** 2).mean()), 4)
        # calibration : probabilité annoncée vs fréquence réelle, par tranche
        bins = pd.cut(g["p_modele"], [0, .3, .4, .5, .6, .7, .8, .9, 1])
        cal = g.groupby(bins, observed=True).agg(annonce=("p_modele", "mean"),
                                                  reel=("issue", "mean"), n=("issue", "size"))
        d["calibration"] = [{"tranche": str(i), "annonce": round(r.annonce, 3),
                             "reel": round(r.reel, 3), "n": int(r.n)}
                            for i, r in cal.iterrows() if r.n >= 30]
        out["familles"][fam] = d

    # paris simulés (marchés où la cote marché est connue)
    bets = df.dropna(subset=["cote"]).copy()
    bets["mini"] = (1 / bets["p"]) * (1 + M.CONFIG["marge_value"])
    sel = bets[(bets["cote"] >= bets["mini"])
               & bets["mini"].between(M.CONFIG["cote_min"], M.CONFIG["cote_max"])]
    sel = sel.assign(gain=np.where(sel["issue"] == 1, sel["cote"] - 1, -1.0))
    sim = {}
    for name, g in [("total", sel)] + list(sel.groupby("famille")) + \
            [(f"fiabilité {k}", v) for k, v in sel.groupby("fiabilite")]:
        if len(g):
            sim[name] = {"paris": int(len(g)), "reussite": round(float(g["issue"].mean()), 3),
                         "cote_moy": round(float(g["cote"].mean()), 2),
                         "rendement": round(float(g["gain"].sum() / len(g)), 4)}
    # même chose avec la proba du modèle seul (sans mélange marché)
    bets["mini_mod"] = (1 / bets["p_modele"]) * (1 + M.CONFIG["marge_value"])
    s2 = bets[(bets["cote"] >= bets["mini_mod"])
              & bets["mini_mod"].between(M.CONFIG["cote_min"], M.CONFIG["cote_max"])]
    if len(s2):
        g2 = np.where(s2["issue"] == 1, s2["cote"] - 1, -1.0)
        sim["modele_seul"] = {"paris": int(len(s2)), "reussite": round(float(s2["issue"].mean()), 3),
                              "rendement": round(float(g2.mean()), 4)}
    out["paris_simules"] = sim
    return out


if __name__ == "__main__":
    df = run()
    res = summarise(df)
    Path(M.OUT / "backtest").mkdir(parents=True, exist_ok=True)
    (M.OUT / "backtest" / "resume.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    df.to_csv(M.OUT / "backtest" / "predictions.csv.gz", index=False, compression="gzip")
    print(json.dumps(res, ensure_ascii=False, indent=1))
