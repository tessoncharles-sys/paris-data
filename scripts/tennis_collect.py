"""Collecte tennis via le tableau des scores public d'ESPN (ATP + WTA, Challengers inclus).

- Historique : un appel tous les 5 jours depuis 2024 (chaque appel renvoie les tournois
  en cours ce jour-là avec tous leurs matchs). Fait une fois, puis mise à jour des 21
  derniers jours à chaque passage.
- Programme : matchs à venir aujourd'hui et demain.
Sorties : data/raw/tennis/matches.csv (matchs terminés) et data/raw/tennis/schedule.csv.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "raw" / "tennis"
BASE = "https://site.api.espn.com/apis/site/v2/sports/tennis/{tour}/scoreboard"
HEADERS = {"User-Agent": "paris-data/1.0 (usage personnel)"}
START = dt.date(2024, 1, 1)
STEP_DAYS = 5
PARIS = ZoneInfo("Europe/Paris")


def fetch(tour: str, day: dt.date | None) -> dict | None:
    params = {"dates": day.strftime("%Y%m%d")} if day else {}
    for attempt in range(3):
        try:
            r = requests.get(BASE.format(tour=tour), params=params, headers=HEADERS, timeout=40)
            if r.status_code == 200:
                return r.json()
            print(f"  {r.status_code} {tour} {day}")
        except requests.RequestException as e:
            print(f"  erreur {e}")
        time.sleep(2 * (attempt + 1))
    return None


def parse(tour: str, payload: dict) -> list[dict]:
    rows = []
    for ev in payload.get("events", []):
        for g in ev.get("groupings", []):
            gname = (g.get("grouping") or {}).get("displayName", "")
            if "Singles" not in gname:
                continue
            for c in g.get("competitions", []):
                comps = c.get("competitors", [])
                if len(comps) != 2 or any(not x.get("athlete") for x in comps):
                    continue
                comps = sorted(comps, key=lambda x: x.get("order", 0))
                a, b = comps
                circuit = "WTA" if "Women" in gname else ("ATP" if "Men" in gname else tour.upper())
                row = {
                    "id": c.get("id"), "tour": circuit, "event_id": ev.get("id"),
                    "event": ev.get("name"), "major": bool(ev.get("major")),
                    "round": (c.get("round") or {}).get("displayName", ""),
                    "date": c.get("date"), "status": ((c.get("status") or {}).get("type") or {}).get("name"),
                    "venue": (c.get("venue") or {}).get("fullName", ""),
                    "p1_id": a.get("id"), "p1": a["athlete"].get("displayName"),
                    "p2_id": b.get("id"), "p2": b["athlete"].get("displayName"),
                    "p1_rank": (a.get("curatedRank") or {}).get("current"),
                    "p2_rank": (b.get("curatedRank") or {}).get("current"),
                    "winner": 1 if a.get("winner") else (2 if b.get("winner") else 0),
                }
                ls1, ls2 = a.get("linescores") or [], b.get("linescores") or []
                sets = []
                for s1, s2 in zip(ls1, ls2):
                    sets.append(f"{int(s1.get('value', 0))}-{int(s2.get('value', 0))}")
                row["sets"] = " ".join(sets)
                rows.append(row)
    return rows


def collect_history(full: bool) -> pd.DataFrame:
    path = OUT / "matches.csv"
    old = pd.read_csv(path, dtype={"id": str}) if path.exists() else pd.DataFrame()
    today = dt.date.today()
    start = START if (full or old.empty) else today - dt.timedelta(days=21)
    days = []
    d = start
    while d <= today:
        days.append(d)
        d += dt.timedelta(days=STEP_DAYS)
    days.append(today)
    rows = []
    for tour in ("atp", "wta"):
        for day in days:
            p = fetch(tour, day)
            if p:
                rows += parse(tour, p)
            time.sleep(0.3)
    new = pd.DataFrame(rows)
    if not new.empty:
        new = new[new["status"].isin(["STATUS_FINAL", "STATUS_RETIRED", "STATUS_WALKOVER"])]
    allm = pd.concat([old, new], ignore_index=True) if not old.empty else new
    allm = allm.drop_duplicates(subset=["id"], keep="last").sort_values("date")
    OUT.mkdir(parents=True, exist_ok=True)
    allm.to_csv(path, index=False)
    return allm


def collect_schedule() -> pd.DataFrame:
    now = dt.datetime.now(dt.timezone.utc)
    end = now + dt.timedelta(hours=40)
    rows = []
    today = dt.datetime.now(PARIS).date()
    for tour in ("atp", "wta"):
        for day in (today, today + dt.timedelta(days=1)):
            p = fetch(tour, day)
            if p:
                rows += parse(tour, p)
    sch = pd.DataFrame(rows)
    if sch.empty:
        sch.to_csv(OUT / "schedule.csv", index=False)
        return sch
    sch = sch[sch["status"] == "STATUS_SCHEDULED"].drop_duplicates(subset=["id"])
    t = pd.to_datetime(sch["date"], utc=True, errors="coerce")
    sch = sch[(t >= now - dt.timedelta(hours=1)) & (t <= end)]
    OUT.mkdir(parents=True, exist_ok=True)
    sch.to_csv(OUT / "schedule.csv", index=False)
    return sch


def main() -> int:
    full = "--full" in sys.argv or not (OUT / "matches.csv").exists()
    m = collect_history(full)
    s = collect_schedule()
    report = {"matchs_historique": int(len(m)), "matchs_programmes": int(len(s)), "complet": full}
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(report)
    return 0 if len(m) else 1


if __name__ == "__main__":
    sys.exit(main())
