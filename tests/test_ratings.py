"""Sanity checks for the driver card ratings (f1predict/ratings.py), on the cached 2025 season.
    python -m unittest discover tests
"""
import unittest

from f1predict.data import load_qualifying, load_results
from f1predict.ratings import MIN_STARTS, STATS, driver_ratings

SEASON = 2025


class Ratings(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = load_results([SEASON], refresh_current=False)
        cls.quali = load_qualifying([SEASON], refresh_current=False)
        cls.cards = driver_ratings(cls.results, cls.quali, SEASON)

    def test_ranges_and_tiers(self):
        for d, c in self.cards.items():
            self.assertTrue(all(40 <= c["stats"][k] <= 97 for k in STATS), d)
            self.assertTrue(60 <= c["ovr"] <= 94, d)
            want = "gold" if c["ovr"] >= 80 else "silver" if c["ovr"] >= 70 else "bronze"
            self.assertEqual(c["tier"], want, d)

    def test_only_regular_drivers_get_cards(self):
        starts = self.results.groupby("driver").size()
        self.assertEqual(set(self.cards), set(starts[starts >= MIN_STARTS].index))

    def test_champion_rates_gold(self):
        last = self.results[self.results["round"] == self.results["round"].max()]
        top_points = self.results.groupby("driver")["points"].sum().idxmax()
        self.assertGreater(len(last), 0)
        self.assertEqual(self.cards[top_points]["tier"], "gold")

    def test_best_qualifier_has_top_qualifying_stat(self):
        best = max(self.cards, key=lambda d: -self.cards[d]["raw"]["avg_quali"])
        self.assertEqual(self.cards[best]["stats"]["QUA"], max(c["stats"]["QUA"] for c in self.cards.values()))


if __name__ == "__main__":
    unittest.main()
