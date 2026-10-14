"""The model must never peek at the result it is predicting. These tests check that every feature of a
race is built only from information available before lights out:

1. Scrambling one race's outcome (finishing order, points, retirements) leaves that race's features,
   and those of every earlier race, exactly the same - but does change the next race's.
2. Deleting all later races leaves every earlier race's features exactly the same.

Runs offline on the cached data in data/:  python -m unittest discover tests
"""
import unittest

import numpy as np
import pandas as pd

from f1predict.data import load_qualifying, load_results
from f1predict.features import FEATURES, build_features

SEASONS = [2024, 2025]
TARGET = 2025 * 100 + 10  # the race whose result gets tampered with (2025, round 10)


def features_of(df, keep):
    out = df.loc[keep(df["race_id"]), ["race_id", "driver", *FEATURES]]
    return out.sort_values(["race_id", "driver"]).reset_index(drop=True)


class NoLeakage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = load_results(SEASONS, refresh_current=False)
        cls.quali = load_qualifying(SEASONS, refresh_current=False)
        cls.base = build_features(cls.results, cls.quali)

    def test_own_result_is_not_used(self):
        res = self.results.copy()
        race = (res["season"] * 100 + res["round"]) == TARGET
        self.assertGreater(race.sum(), 10)
        # reverse the finishing order and hand the points and retirements to the wrong drivers
        res.loc[race, "position"] = res.loc[race, "position"].max() + 1 - res.loc[race, "position"]
        res.loc[race, "points"] = res.loc[race, "points"].values[::-1]
        res.loc[race, "finished"] = ~res.loc[race, "finished"]
        tampered = build_features(res, self.quali)

        upto = lambda r: r <= TARGET
        pd.testing.assert_frame_equal(features_of(self.base, upto), features_of(tampered, upto))
        # sanity check that the test can fail: the next race does see the scrambled result
        nxt = lambda r: r == self.base.loc[self.base["race_id"] > TARGET, "race_id"].min()
        self.assertFalse(features_of(self.base, nxt).equals(features_of(tampered, nxt)))

    def test_future_races_are_not_used(self):
        past = self.results[(self.results["season"] * 100 + self.results["round"]) <= TARGET]
        trimmed = build_features(past, self.quali)
        upto = lambda r: r <= TARGET
        pd.testing.assert_frame_equal(features_of(self.base, upto), features_of(trimmed, upto),
                                      check_dtype=False)  # same values; int vs float storage only


if __name__ == "__main__":
    unittest.main()
