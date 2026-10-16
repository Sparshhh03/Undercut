"""Driver card ratings: each driver's season so far turned into six 0-99 stats and an overall rating,
in the style of football-game player cards.

Every stat is a plain season average from race and qualifying results, rescaled so the best driver on
the grid gets ~97 and the worst ~40:
    QUA  qualifying     average qualifying position
    PAC  pace           average finishing position, in races they finished
    RAC  racecraft      places gained (or kept) from the grid, as a share of what was possible
    CON  consistency    races finished, and how little the finishing positions jump around
    FRM  form           points per race over the last five races
    H2H  vs teammate    share of qualifying and race head-to-heads won against the teammate
OVR is a weighted mix of the six, stretched to roughly 60-94. Tiers: gold 80+, silver 70-79, bronze below.
Drivers with fewer than MIN_STARTS races (stand-ins) get no card.
"""
import numpy as np
import pandas as pd

MIN_STARTS = 3
STATS = ["QUA", "PAC", "RAC", "CON", "FRM", "H2H"]
WEIGHTS = {"QUA": .22, "PAC": .26, "RAC": .08, "CON": .12, "FRM": .17, "H2H": .15}


def _scale(s, higher_is_better=True, lo=40, hi=97):
    s = s if higher_is_better else -s
    span = s.max() - s.min()
    return (lo + (s - s.min()) / span * (hi - lo)) if span else pd.Series(hi, index=s.index)


def _head_to_head(df, col):
    """Share of rounds in which each driver beat their teammate on `col` (lower is better)."""
    won, total = {}, {}
    for _, g in df.groupby(["round", "constructor"]):
        if len(g) != 2:
            continue
        a, b = g.iloc[0], g.iloc[1]
        if pd.isna(a[col]) or pd.isna(b[col]) or a[col] == b[col]:
            continue
        for me, other in ((a, b), (b, a)):
            total[me["driver"]] = total.get(me["driver"], 0) + 1
            won[me["driver"]] = won.get(me["driver"], 0) + (me[col] < other[col])
    return pd.Series({d: won[d] / total[d] for d in total}, dtype=float)


def driver_ratings(results, quali, season):
    """{driver id: {"ovr", "tier", "stats": {QUA..H2H}, "raw": {...}}} for one season so far."""
    res = results[results["season"] == season].copy()
    q = quali[quali["season"] == season]
    starts = res.groupby("driver").size()
    drivers = starts[starts >= MIN_STARTS].index
    res = res[res["driver"].isin(drivers)]
    res["grid_n"] = res["grid"].where(res["grid"] > 0, res.groupby("round")["driver"].transform("count"))
    fin = res[res["finished"].astype(bool)]
    field = res.groupby("round")["driver"].transform("count")

    raw = pd.DataFrame(index=drivers)
    raw["QUA"] = q[q["driver"].isin(drivers)].groupby("driver")["quali_pos"].mean()
    raw["QUA"] = raw["QUA"].fillna(res.groupby("driver")["grid_n"].mean())
    raw["PAC"] = fin.groupby("driver")["position"].mean()
    raw["PAC"] = raw["PAC"].fillna(field.max())
    # gained: share of the places that were available ahead; lost: share of the places behind
    room = np.where(res["grid_n"] >= res["position"], (res["grid_n"] - 1).clip(lower=1), (field - res["grid_n"]).clip(lower=1))
    res["gain_share"] = (res["grid_n"] - res["position"]) / room
    raw["RAC"] = res[res["finished"].astype(bool)].groupby("driver")["gain_share"].mean()
    raw["RAC"] = raw["RAC"].fillna(raw["RAC"].min())
    finish_rate = res.groupby("driver")["finished"].mean()
    spread = fin.groupby("driver")["position"].std().fillna(fin["position"].std())
    raw["CON"] = 0.6 * finish_rate + 0.4 * (1 - spread / spread.max())
    last5 = sorted(res["round"].unique())[-5:]
    raw["FRM"] = res[res["round"].isin(last5)].groupby("driver")["points"].mean().reindex(drivers).fillna(0)
    q_h2h = _head_to_head(q.merge(res[["round", "driver"]], on=["round", "driver"]), "quali_pos")
    r_h2h = _head_to_head(res, "position")
    raw["H2H"] = pd.concat([q_h2h, r_h2h], axis=1).mean(axis=1).reindex(drivers).fillna(0.5)

    stats = pd.DataFrame({
        "QUA": _scale(raw["QUA"], higher_is_better=False),
        "PAC": _scale(raw["PAC"], higher_is_better=False),
        "RAC": _scale(raw["RAC"]),
        "CON": _scale(raw["CON"]),
        "FRM": _scale(raw["FRM"]),
        "H2H": _scale(raw["H2H"]),
    }).round().astype(int)
    mix = sum(stats[k] * w for k, w in WEIGHTS.items())
    ovr = _scale(mix, lo=60, hi=94).round().astype(int)

    out = {}
    for d in drivers:
        o = int(ovr[d])
        out[d] = {
            "ovr": o,
            "tier": "gold" if o >= 80 else "silver" if o >= 70 else "bronze",
            "stats": {k: int(stats.at[d, k]) for k in STATS},
            "raw": {
                "avg_quali": round(float(raw.at[d, "QUA"]), 1),
                "avg_finish": round(float(raw.at[d, "PAC"]), 1),
                "places_gained": round(float((res[(res["driver"] == d) & res["finished"].astype(bool)]
                                              .eval("grid_n - position")).mean()), 1),
                "finish_rate": round(float(finish_rate[d]), 2),
                "points_last5": round(float(raw.at[d, "FRM"]), 1),
                "h2h": round(float(raw.at[d, "H2H"]), 2),
            },
        }
    return out
