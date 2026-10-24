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

## Tennis (ATP, WTA, quelques Challengers)

- **Données** : tableau des scores public d'ESPN (historique depuis 2024, score de chaque set), `scripts/tennis_collect.py`. Surface déduite du nom du tournoi.
- **Modèle** (`scripts/tennis.py`) : Elo général + Elo par surface → probabilité de victoire ; simulation jeu par jeu → score en sets, au moins un set, match en 3 sets, total de jeux, handicap jeux.
- **Test** (`scripts/tennis_calibrate.py`, 15 000 matchs 2025-26) : après correction apprise sur 2025, les probabilités 2026 sont bien calibrées sur tous les marchés (« 64 % annoncé → 64 % réel » sur le total de jeux). Le favori Elo gagne 64 % du temps, ce qui reste en dessous des bookmakers : aucune cote marché n'étant disponible gratuitement, toutes les cotes justes tennis viennent du modèle.
- Joueurs avec moins de 8 matchs dans l'historique : match non analysé.

## Recalibration mensuelle

Le 1er de chaque mois, `.github/workflows/calibration.yml` relance les tests de fiabilité foot et tennis sur les données les plus récentes (fenêtres glissantes), met à jour les corrections utilisées chaque matin et écrit le bilan dans [`data/backtest/BILAN.md`](data/backtest/BILAN.md). Lancement à la main : onglet Actions → « Recalibration mensuelle » → Run workflow.

## Handball

- **Données** : API-Sports handball avec ta clé (offre gratuite : matchs d'hier, d'aujourd'hui et de demain, cotes de plusieurs bookmakers par match ; jusqu'à 80 matchs avec cotes par jour). Championnats suivis : Starligue, Proligue, Bundesliga, ASOBAL, Danemark, Ligues EHF, etc.
- **Cotes justes** (`scripts/handball.py`) : consensus des bookmakers marge retirée → loi normale de l'écart et du total de buts ajustée sur les lignes de handicap et de total → victoire, nul, écarts, totaux, buts d'une équipe, 1re mi-temps.
- **Pas encore de test de fiabilité** : l'offre gratuite ne donne pas d'historique. Résultats et prédictions s'archivent chaque jour (`data/raw/handball/`) pour un test dans quelques semaines.

## Cotes des bookmakers (foot)

`scripts/odds_foot.py` récupère à chaque passage, via API-Football, les cotes de Pinnacle et de 8 autres bookmakers pour les matchs du tableau (jusqu'à ~80 matchs par jour avec le quota gratuit, championnats majeurs en priorité). Marge retirée, la cote Pinnacle devient la cote juste de référence ; la probabilité du modèle reste affichée avec l'écart (un écart de 8 points ou plus est signalé). Toutes les cotes sont archivées dans `data/raw/api-football/odds_archive/` pour tester le modèle face au marché et suivre la cote de clôture.

## Cotes Winamax (The Odds API)

`scripts/odds_winamax.py` (passage du matin, clé `ODDS_API_KEY`) récupère les cotes de Winamax et des autres opérateurs français (Betclic, Unibet, PMU, Parions Sport, NetBet) sur la victoire et le total 2,5 buts des grands championnats, et sur le vainqueur des tournois de tennis couverts. Le tableau affiche alors la value Winamax automatiquement (proba × cote Winamax − 1) et la meilleure cote française. Budget : 16 crédits par jour au maximum, sur les 500 gratuits par mois.
