"""Assemble le fichier du tableau de bord : football + tennis (si disponible)."""
from pathlib import Path
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "data"
parts = []
for name in ("dashboard_foot.csv", "dashboard_tennis.csv"):
    f = OUT / name
    if f.exists() and f.stat().st_size > 10:
        try:
            parts.append(pd.read_csv(f))
        except pd.errors.EmptyDataError:
            pass
df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
cols = ["sport", "id", "date", "heure", "championnat", "match", "famille", "marche", "proba",
        "cote_juste", "cote_minimum", "cote_moyenne_marche", "source_cote", "fiabilite", "infos",
        "lam_dom", "lam_ext", "lam_source"]
df = df.reindex(columns=cols)
df.to_csv(OUT / "dashboard.csv", index=False)
print(df.groupby("sport")["match"].nunique().to_dict() if len(df) else "vide")
