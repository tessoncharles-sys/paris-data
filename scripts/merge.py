"""Assemble le fichier du tableau de bord : football + tennis (si disponible)."""
from pathlib import Path
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "data"
parts = []
for name in ("dashboard_foot.csv", "dashboard_tennis.csv", "dashboard_handball.csv"):
    f = OUT / name
    if f.exists() and f.stat().st_size > 10:
        try:
            parts.append(pd.read_csv(f))
        except pd.errors.EmptyDataError:
            pass
df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
cols = ["sport", "id", "date", "heure", "championnat", "match", "famille", "marche", "proba",
        "cote_juste", "cote_minimum", "cote_moyenne_marche", "source_cote", "fiabilite", "infos",
        "lam_dom", "lam_ext", "lam_source", "explication", "signaux", "proba_modele", "ecart_modele", "cote_winamax", "value_winamax", "meilleure_cote_fr", "meilleur_operateur"]
df = df.reindex(columns=cols)
# value Winamax pour les lignes qui ont une cote Winamax (tennis compris)
mask = df["cote_winamax"].notna() & df["value_winamax"].isna()
df.loc[mask, "value_winamax"] = (df.loc[mask, "proba"] * df.loc[mask, "cote_winamax"] - 1).round(4)
df.to_csv(OUT / "dashboard.csv", index=False)
print(df.groupby("sport")["match"].nunique().to_dict() if len(df) else "vide")

# horodatage du passage, pour que les vérifications Claude sachent si les données sont fraîches
import datetime as _dt, json as _json
(OUT / "maj.json").write_text(_json.dumps({"genere_le": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")}))
