"""Écrit data/backtest/BILAN.md : le résumé lisible des derniers tests de fiabilité (foot + tennis)."""
import datetime as dt
import json
from pathlib import Path

B = Path(__file__).resolve().parent.parent / "data" / "backtest"


def load(name):
    try:
        return json.loads((B / name).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


foot, tennis = load("validation.json"), load("tennis_validation.json")
L = [f"# Bilan de fiabilité du modèle — mis à jour le {dt.date.today():%d/%m/%Y}", ""]
L += ["Lecture : « 64 % annoncé → 63 % réel » veut dire que quand le modèle annonce 64 %, l'issue arrive vraiment 63 % du temps.",
      "Le score de Brier mesure l'erreur : plus il est bas, mieux c'est.", ""]
if foot:
    L += [f"## Football (période testée {foot.get('periode_test', '?')}, vérification après {foot.get('apprentissage_avant', '?')})", ""]
    for fam, v in foot.get("validation", {}).items():
        if fam.startswith("paris_simules"):
            continue
        line = f"- **{fam}** : Brier modèle {v.get('brier_calibre')}"
        if "brier_marche" in v:
            line += f", marché {v['brier_marche']} ({'le marché fait mieux' if v['brier_marche'] < v['brier_calibre'] else 'le modèle fait mieux'})"
        L.append(line)
        cal = v.get("calibration", [])
        if cal:
            L.append(f"  - {cal[len(cal) // 2]}")
    sim = foot.get("validation", {}).get("paris_simules_periode_test", {})
    if sim:
        L.append(f"- **Paris simulés contre la cote moyenne** : {sim.get('paris', 0)} paris, rendement {sim.get('rendement', 'n/a')}")
    L.append("")
if tennis:
    L += [f"## Tennis ({tennis.get('matchs')} matchs, {tennis.get('periode', '?')}, vérification depuis {tennis.get('verification_depuis', '?')})", "",
          f"- **Favori Elo gagnant** : {tennis.get('precision_vainqueur_favori_test', 0):.0%} des matchs"]
    for fam, v in tennis.get("validation_test", {}).items():
        cal = v.get("calibration", [])
        L.append(f"- **{fam}** : Brier {v.get('brier_calibre')}" + (f" ; {cal[len(cal) // 2]}" if cal else ""))
(B / "BILAN.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))
