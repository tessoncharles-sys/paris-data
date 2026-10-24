# paris-data

Collecte et analyse quotidiennes pour la sélection de paris (usage personnel).

## Ce qui tourne chaque matin (GitHub Actions)

1. `scripts/collect.py` récupère :
   - les résultats et stats (tirs cadrés, corners, cartons, cotes moyennes) de 19 championnats européens sur football-data.co.uk, saison en cours et précédente ;
   - les matchs à venir avec leurs cotes (`fixtures.csv`) ;
   - les matchs du jour et du lendemain via API-Football (clé dans le secret `API_FOOTBALL_KEY`).
2. `scripts/model.py` calcule, pour chaque match du jour et du lendemain, la probabilité de 29 marchés :
   1X2, double chance, plus/moins 1,5 / 2,5 / 3,5 buts, les deux équipes marquent, buts par équipe, corners, cartons.
3. Les résultats sont enregistrés dans `data/` :
   - `data/latest.md` : la sélection lisible ;
   - `data/latest.json` : tout le détail (utilisé par Claude) ;
   - `data/picks/AAAA-MM-JJ.json` : l'historique, pour mesurer la fiabilité du modèle.

## Comment lire la sélection

- **Cote minimum** : cote juste du modèle + 5 % d'exigence de value. On ne joue que si Winamax propose au moins cette cote.
- **Value confirmée par le marché** : la cote moyenne des bookmakers dépasse déjà notre cote minimum.
- **À vérifier sur Winamax** : pas de cote marché dans les données (corners, cartons, double chance…) ; à vérifier dans l'appli.
- **Fiabilité** : « bonne » si les deux équipes ont au moins 10 matchs récents dans le championnat.

## Lancer à la main

Onglet **Actions** → « Collecte et analyse quotidiennes » → **Run workflow**.

## Réglages

En haut de `scripts/model.py`, bloc `CONFIG` : fourchette de cotes (1,35–1,80), exigence de value (5 %), poids donné aux matchs récents, etc.

> Un modèle améliore l'analyse, il ne garantit pas de gagner. Jouer avec modération.

## Résultat du test sur la saison 2025-26 (nov. → juin, 4 400 matchs)

- **Calibration** : le modèle brut était trop sûr de lui sur les extrêmes (ex. « 74 % » → 61 % réel sur +/-2,5 buts, corners et cartons). Une correction apprise sur nov.-fév. et vérifiée sur mars-juin le rend fiable : « 64 % annoncé → 63-66 % réel ». Elle est dans `data/calibration.json` et s'applique automatiquement.
- **Face au marché** : sur 1X2, double chance et +/-2,5 buts, la cote moyenne des bookmakers prédit mieux que le modèle. Le modèle n'y apporte rien → pour ces marchés, la cote juste vient du marché (marge retirée).
- **Paris simulés** : aucune value durable détectée contre la cote moyenne du marché.
- **Où le modèle sert** : corners, cartons, buts par équipe, les 2 marquent, +/-1,5 et 3,5 buts (marchés sans cote publique dans les données), et pour juger si une cote Winamax — notamment boostée — dépasse la cote juste.

Relancer le test : `python scripts/calibrate.py`.
