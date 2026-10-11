"""Practice-session pace from the OpenF1 API (2023 onwards): each driver's best lap in FP1/FP2/FP3,
matched to the Jolpica round, cached in data/practice_<season>.json.

OpenF1 serves a session's laps once it has finished (during a live session it locks the free API)."""
import json
import time
from pathlib import Path

import pandas as pd
import requests

from .data import load_schedule

API = "https://api.openf1.org/v1"
CACHE = Path(__file__).resolve().parent.parent / "data"
FIRST_SEASON = 2023


def _get(path, params, retries=6):
    for attempt in range(retries):
        r = requests.get(f"{API}/{path}", params=params, timeout=60)
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(2 ** attempt * 2)
            continue
        r.raise_for_status()
        time.sleep(0.4)  # stay well under the free rate limit
        return r.json()
    raise RuntimeError(f"OpenF1 kept refusing {path} {params}")


def _session_best_laps(session_key):
    """{driver code: best lap in seconds} for one session (pit-out laps skipped)."""
    drivers = {d["driver_number"]: d["name_acronym"] for d in _get("drivers", {"session_key": session_key})}
    best = {}
    for lap in _get("laps", {"session_key": session_key}):
        t = lap.get("lap_duration")
        if not t or lap.get("is_pit_out_lap"):
            continue
        code = drivers.get(lap["driver_number"])
        if code and (code not in best or t < best[code]):
            best[code] = t
    return best


def _fetch_season(season, known):
    """Practice sessions of a season that have finished, as {session_key: {...}}; keeps cached ones."""
    sched = load_schedule(season)
    races = pd.to_datetime(sched["date"]).dt.tz_localize("UTC")
    now = pd.Timestamp.now(tz="UTC")
    out = dict(known)
    for s in _get("sessions", {"year": season, "session_type": "Practice"}):
        key = str(s["session_key"])
        if key in out or not s["session_name"].startswith("Practice") or s.get("is_cancelled"):
            continue
        end = pd.Timestamp(s.get("date_end") or s["date_start"])
        if end > now - pd.Timedelta(minutes=30):  # not finished (or OpenF1 still locked): next time
            continue
        # the round whose race is 0-4 days after this session
        gap = (races - pd.Timestamp(s["date_start"]).normalize()).dt.days
        match = sched[(gap >= 0) & (gap <= 4)]
        if match.empty:
            continue
        out[key] = {"round": int(match.iloc[0]["round"]), "session": s["session_name"],
                    "best": _session_best_laps(s["session_key"])}
    return out


def load_practice(seasons, refresh_current=True):
    """One row per (season, round, driver code): best lap in each practice session and overall."""
    CACHE.mkdir(exist_ok=True)
    current = max(seasons)
    rows = []
    for season in seasons:
        if season < FIRST_SEASON:
            continue
        path = CACHE / f"practice_{season}.json"
        known = json.loads(path.read_text()) if path.exists() else {}
        if not path.exists() or (refresh_current and season == current):
            known = _fetch_season(season, known)
            path.write_text(json.dumps(known))
        for s in known.values():
            for code, t in s["best"].items():
                rows.append({"season": season, "round": s["round"], "session": s["session"], "code": code, "lap": t})
    return pd.DataFrame(rows, columns=["season", "round", "session", "code", "lap"])


def practice_gap(season, rnd, codes):
    """Best practice lap of each driver in `codes` (a Series of driver codes) for one weekend, as a %
    gap to the fastest; NaN for anyone without a lap. If OpenF1 can't be reached (e.g. during a live
    session its free API is locked), the sessions cached so far are used."""
    try:
        p = load_practice([season])
    except (requests.RequestException, RuntimeError, ValueError):
        p = load_practice([season], refresh_current=False) if (CACHE / f"practice_{season}.json").exists() \
            else pd.DataFrame(columns=["season", "round", "session", "code", "lap"])
    best = p[p["round"] == rnd].groupby("code")["lap"].min()
    lap = codes.map(best).astype(float)
    gap = (lap - lap.min()) / lap.min() * 100
    return gap.where(gap <= 7)  # >7%: a lap in different conditions, not real pace
