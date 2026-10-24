"""Collecte quotidienne des données foot.

- football-data.co.uk : résultats + stats (tirs, corners, cartons) des championnats
  européens, saison en cours et précédente, + fichier des matchs à venir avec cotes.
- API-Football : matchs du jour et du lendemain (2 requêtes / jour, quota gratuit 100).

Les fichiers sont écrits dans data/ et commités par le workflow GitHub.
"""
from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo
import json
import os
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
FD_BASE = "https://www.football-data.co.uk"

# Championnats européens couverts par football-data.co.uk (codes "Div")
DIVISIONS = {
    "E0": "Angleterre - Premier League",
    "E1": "Angleterre - Championship",
    "E2": "Angleterre - League One",
    "E3": "Angleterre - League Two",
    "SC0": "Écosse - Premiership",
    "SC1": "Écosse - Championship",
    "D1": "Allemagne - Bundesliga",
    "D2": "Allemagne - 2. Bundesliga",
    "I1": "Italie - Serie A",
    "I2": "Italie - Serie B",
    "SP1": "Espagne - LaLiga",
    "SP2": "Espagne - LaLiga 2",
    "F1": "France - Ligue 1",
    "F2": "France - Ligue 2",
    "N1": "Pays-Bas - Eredivisie",
    "B1": "Belgique - Pro League",
    "P1": "Portugal - Liga",
    "T1": "Turquie - Süper Lig",
    "G1": "Grèce - Super League",
}

HEADERS = {"User-Agent": "paris-data/1.0 (usage personnel)"}


def season_codes(today: dt.date) -> list[str]:
    """Codes saison football-data (ex. '2627' pour 2026-27) : saison en cours + précédente."""
    start = today.year if today.month >= 7 else today.year - 1
    cur = f"{start % 100:02d}{(start + 1) % 100:02d}"
    prev = f"{(start - 1) % 100:02d}{start % 100:02d}"
    return [prev, cur]


def get(url: str, **kw) -> requests.Response | None:
    for attempt in range(3):
        try:
            r = requests.get(url, headers={**HEADERS, **kw.pop("headers", {})}, timeout=30, **kw)
            if r.status_code == 200:
                return r
            print(f"  {r.status_code} {url}")
            if r.status_code == 404:
                return None
        except requests.RequestException as e:
            print(f"  erreur {e} ({url})")
        time.sleep(2 * (attempt + 1))
    return None


def collect_football_data(today: dt.date) -> dict:
    out = RAW / "football-data"
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    for season in season_codes(today):
        for div in DIVISIONS:
            r = get(f"{FD_BASE}/mmz4281/{season}/{div}.csv")
            ok = r is not None and len(r.content) > 200
            if ok:
                (out / f"{season}_{div}.csv").write_bytes(r.content)
            report[f"{season}_{div}"] = "ok" if ok else "absent"
    r = get(f"{FD_BASE}/fixtures.csv")
    if r is not None:
        (out / "fixtures.csv").write_bytes(r.content)
        report["fixtures"] = "ok"
    else:
        report["fixtures"] = "absent"
    return report


def collect_api_football(today: dt.date) -> dict:
    key = os.environ.get("API_FOOTBALL_KEY")
    if not key:
        return {"api_football": "clé absente (secret API_FOOTBALL_KEY)"}
    out = RAW / "api-football"
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    for d in (today, today + dt.timedelta(days=1)):
        r = get(
            "https://v3.football.api-sports.io/fixtures",
            params={"date": d.isoformat(), "timezone": "Europe/Paris"},
            headers={"x-apisports-key": key},
        )
        if r is None:
            report[d.isoformat()] = "échec"
            continue
        data = r.json()
        if data.get("errors"):
            report[d.isoformat()] = f"erreur API : {data['errors']}"
            continue
        (out / f"fixtures_{d.isoformat()}.json").write_text(
            json.dumps(data.get("response", []), ensure_ascii=False), encoding="utf-8"
        )
        report[d.isoformat()] = f"{data.get('results', 0)} matchs"
    # ménage : on garde 60 jours de fichiers API
    for f in out.glob("fixtures_*.json"):
        try:
            d = dt.date.fromisoformat(f.stem.split("_", 1)[1])
            if (today - d).days > 60:   # 60 jours gardés : sert au calcul des jours de repos
                f.unlink()
        except ValueError:
            pass
    return report


def main() -> int:
    today = dt.datetime.now(ZoneInfo("Europe/Paris")).date()
    report = {
        "date": today.isoformat(),
        "football_data": collect_football_data(today),
        "api_football": collect_api_football(today),
    }
    (ROOT / "data").mkdir(exist_ok=True)
    (ROOT / "data" / "collect_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    n_ok = sum(v == "ok" for v in report["football_data"].values())
    print(f"football-data : {n_ok} fichiers ; API-Football : {report['api_football']}")
    return 0 if n_ok > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
