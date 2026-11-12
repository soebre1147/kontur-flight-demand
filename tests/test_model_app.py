"""Regression tests for app adaptations of the preserved chat model v3."""
import hashlib
import json
from math import log
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from scipy.special import ndtr

from model.math_reconstruction_v2 import ModelConfig, prepare_inputs_v2
from model.math_reconstruction_v3 import (
    BehaviorConfigV3, _mixture_median, fit_model_v3, prepare_inputs_v3,
    sample_passengers_v3,
)
from model.conditional_metrics_v3 import conditional_seat_pmf


class AppModelTests(unittest.TestCase):
    def setUp(self):
        self.start, self.end, self.departure = "2027-05-20", "2027-05-23", "2027-05-22"
        self.empty = pd.DataFrame(columns=["flight", "timestamp", "cabin", "commercial", "price"])
        self.offers = pd.DataFrame([
            dict(flight="F1", cabin="Y", start=self.start, end=self.end,
                 price=100., open=True, available_seats=20),
            dict(flight="F1", cabin="C", start=self.start, end=self.end,
                 price=300., open=True, available_seats=4),
        ])
        self.config = ModelConfig(qty_values=(1, 2), qty_prior={1: .7, 2: .3})

    def prepare(self, bookings=None, offers=None):
        return prepare_inputs_v3(self.empty if bookings is None else bookings,
                                 self.departure, self.start, self.end,
                                 offers=self.offers if offers is None else offers)

    def test_empty_offers_do_not_manufacture_bookings_and_keep_cabins(self):
        inputs = self.prepare()
        self.assertTrue(inputs.bookings.empty)
        self.assertTrue(inputs.purchases.empty)
        self.assertEqual(set(inputs.strata.cabin), {"C", "Y"})
        self.assertEqual(inputs.strata.purchased_seats.sum(), 0)
        self.assertEqual(inputs.assumptions["offer_source"], "supplied_offer_intervals")

    def test_empty_without_offers_rejected(self):
        with self.assertRaisesRegex(ValueError, "No purchases"):
            prepare_inputs_v3(self.empty, self.departure, self.start, self.end)
        with self.assertRaisesRegex(ValueError, "No purchases"):
            prepare_inputs_v2(self.empty, self.departure, self.start, self.end,
                              offers=self.offers)

    def test_empty_requires_external_quantity_and_budget_assumptions(self):
        inputs = self.prepare()
        with self.assertRaisesRegex(ValueError, "qty_prior"):
            fit_model_v3(inputs, 12, ModelConfig(), mu_prior=(log(200), 2))
        with self.assertRaisesRegex(ValueError, "mu_prior"):
            fit_model_v3(inputs, 12, self.config)

    def test_closed_zero_sales_exact_total_and_no_budget_claim(self):
        closed = self.offers.assign(open=False, available_seats=0, price=np.nan)
        inputs = self.prepare(offers=closed)
        result = fit_model_v3(inputs, 12, self.config, mu_prior=(log(200), 2))
        self.assertTrue(result["parameters"].optimizer_success.all())
        self.assertTrue(result["parameters"].no_observed_purchases.all())
        self.assertAlmostEqual(float(result["parameters"].mu_log.iloc[0]), log(200), places=7)
        self.assertAlmostEqual(result["strata"].posterior_expected_primary_seats.sum(), 12.)
        journal = sample_passengers_v3(inputs, result, seed=4)
        self.assertEqual(journal.qty.sum(), 12)
        self.assertTrue(journal.outcome.eq("waste").all())
        self.assertTrue(journal.reason.eq("availability").all())
        self.assertTrue(journal.reason_is_modelled.all())
        self.assertTrue(journal.cabin.isna().all())

    def test_zero_demand_returns_empty_journal(self):
        inputs = self.prepare()
        result = fit_model_v3(inputs, 0, self.config, mu_prior=(log(200), 2))
        journal = sample_passengers_v3(inputs, result)
        self.assertTrue(journal.empty)
        self.assertIn("outcome", journal)

    def test_offer_validation_still_checks_coverage_and_states(self):
        bad = self.offers.copy()
        bad.loc[0, "start"] = "2027-05-21"
        with self.assertRaisesRegex(ValueError, "complete horizon"):
            self.prepare(offers=bad)
        with self.assertRaisesRegex(ValueError, "nonnegative integers"):
            self.prepare(offers=self.offers.assign(available_seats=-1))
        with self.assertRaisesRegex(ValueError, "open flag"):
            self.prepare(offers=self.offers.assign(open="maybe"))

    def test_cancel_only_is_observed_but_not_positive_demand(self):
        cancellations = pd.DataFrame([dict(flight="F1", cabin="Y", timestamp="2027-05-21",
                                           commercial=-2, price=100.)])
        inputs = self.prepare(cancellations)
        result = fit_model_v3(inputs, 12, self.config, mu_prior=(log(200), 2))
        journal = sample_passengers_v3(inputs, result, seed=8)
        self.assertEqual(journal.loc[journal.outcome.eq("waste"), "qty"].sum(), 12)
        self.assertEqual(journal.loc[journal.outcome.eq("cancel"), "signed_qty"].sum(), -2)
        self.assertTrue(journal.loc[journal.outcome.eq("cancel"), "amount"].isna().all())

    def test_mixture_median_solves_cdf_not_exp_mu(self):
        config = BehaviorConfigV3(mixture_weights=(.2, .8), mixture_offsets=(-1., 1.),
                                  mixture_sigmas=(.2, .6))
        mu = log(200)
        median = _mixture_median(mu, config)
        cdf = np.dot(config.mixture_weights,
                     ndtr((log(median)-mu-np.array(config.mixture_offsets))/config.mixture_sigmas))
        self.assertAlmostEqual(cdf, .5, places=10)
        self.assertGreater(median, 200.)
        self.assertAlmostEqual(_mixture_median(mu, ModelConfig()), 200.)

    def test_diagnostics_use_fully_unconditioned_expected_primary(self):
        buys = pd.DataFrame([
            dict(flight="F1", cabin="Y", timestamp="2027-05-21", commercial=2, price=100.),
            dict(flight="F1", cabin="Y", timestamp="2027-05-22", commercial=1, price=100.),
        ])
        inputs = self.prepare(buys)
        result = fit_model_v3(inputs, 12, self.config, mu_prior=(log(200), 2))
        rows = result["strata"]
        np.testing.assert_allclose(rows.unconditioned_expected_primary_seats,
                                   rows.expected_purchased_seats+rows.raw_expected_hidden_seats)
        np.testing.assert_allclose(rows.observed_gross_plus_unconditioned_hidden,
                                   rows.purchased_seats+rows.raw_expected_hidden_seats)
        par = result["parameters"].iloc[0]
        self.assertAlmostEqual(par.unconditioned_expected_primary_seats,
                               rows.unconditioned_expected_primary_seats.sum())
        self.assertAlmostEqual(par.median_amount, par.budget_scale)
        self.assertAlmostEqual(par.expected_primary_seats, 12.)
        journal = sample_passengers_v3(inputs, result, seed=1)
        self.assertEqual(journal.loc[journal.outcome.isin(["success", "waste"]), "qty"].sum(), 12)
        self.assertEqual(journal.loc[journal.outcome.eq("success"), "qty"].sum(), 3)

    def test_conditional_marginal_is_exact_binomial_special_case(self):
        from scipy.stats import binom
        pmf = conditional_seat_pmf({1: 2.}, {1: 3.}, 12)
        np.testing.assert_allclose(pmf, binom.pmf(np.arange(13), 12, .4), atol=1e-12)

    def test_unchanged_conditional_metrics_matches_recorded_source_hash(self):
        folder = Path(__file__).resolve().parents[1] / "model"
        metadata = json.loads((folder / "provenance.json").read_text(encoding="utf-8"))
        actual = hashlib.sha256((folder / "conditional_metrics_v3.py").read_bytes()).hexdigest()
        self.assertEqual(actual, metadata["conditional_metrics_v3.py"]["source_sha256"])


if __name__ == "__main__":
    unittest.main()
