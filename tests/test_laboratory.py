"""Behavioral checks for forward truth, evaluation boundaries and policy search."""
import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from backend.laboratory import generate_bundle, generate_example, run_experiments
from backend.policies import policy_metrics, run_policies
from backend.scenarios import SCENARIOS, prepare_cohort, simulate_cohort
from backend.simulation import DEFAULTS, RBD_DESCENDING


class ScenarioTests(unittest.TestCase):
    def test_ten_scenarios_conserve_primary_total_and_cabin_inventory(self):
        from model.math_reconstruction_v3 import prepare_inputs_v3
        for scenario in SCENARIOS:
            with self.subTest(scenario=scenario):
                bundle=generate_bundle(scenario,133,{'demand':75,'capacity':42,'businessCapacity':3})
                primary=bundle['passengers'].loc[bundle['passengers'].outcome.isin(['success','waste'])]
                self.assertEqual(int(primary.qty.sum()),bundle['params']['demand'])
                self.assertEqual(primary.id.nunique(),len(primary))
                capacity={'C':bundle['params']['businessCapacity'],
                    'Y':bundle['params']['capacity']-bundle['params']['businessCapacity']}
                for cabin,count in capacity.items():
                    balance=bundle['bookings'].loc[bundle['bookings'].cabin.eq(cabin)].commercial.cumsum()
                    self.assertTrue(balance.between(0,count).all())
                prepared=prepare_inputs_v3(bundle['bookings'],bundle['params']['departure'],
                    bundle['horizon_start'],bundle['horizon_end'],offers=bundle['offers'])
                self.assertEqual(len(prepared.bookings),len(bundle['bookings']))

    def test_policy_replay_does_not_change_any_latent_request(self):
        params,cohort=prepare_cohort('mixed_wtp',42,{'demand':111,'cancel':40})
        runs=[simulate_cohort(params,cohort,'mixed_wtp',{'kind':kind}) for kind in ['static','dynamic','gated']]
        for run in runs:
            pd.testing.assert_frame_equal(run['requests'],cohort)
            truth=run['passengers'].loc[run['passengers'].outcome.isin(['success','waste'])].set_index('id')
            expected=cohort.set_index('id')
            pd.testing.assert_series_equal(truth.amount.sort_index(),expected.amount.sort_index())
            pd.testing.assert_series_equal(truth.qty.sort_index(),expected.qty.sort_index())
        self.assertNotEqual(policy_metrics(runs[0])['revenue'],policy_metrics(runs[1])['revenue'])

    def test_closure_is_actual_no_sale_state_not_only_a_label(self):
        run=generate_bundle('closure',45,{'demand':300})
        closed=run['offers'].loc[~run['offers'].open & run['offers'].available_seats.gt(0)]
        self.assertGreater(len(closed),0)
        for event in run['bookings'].loc[run['bookings'].commercial.gt(0)].itertuples():
            offer=run['offers'].loc[run['offers'].cabin.eq(event.cabin)&run['offers'].start.le(event.timestamp)&run['offers'].end.gt(event.timestamp)].iloc[0]
            self.assertTrue(offer.open)
            self.assertGreaterEqual(offer.available_seats,event.commercial)
            self.assertEqual(offer.price,event.price)
        self.assertTrue(run['passengers'].reason.eq('Продажи закрыты').any())

    def test_misspecification_scenarios_change_the_stated_behavior(self):
        _,base=prepare_cohort('balanced',42)
        _,mixture=prepare_cohort('mixed_wtp',42)
        _,groups=prepare_cohort('groups_budget',42)
        _,burst=prepare_cohort('burst',42)
        ratios=(mixture.set_index('id').amount/base.set_index('id').amount).round(2)
        self.assertTrue(set(ratios).issubset({.6,1.9}))
        self.assertTrue((groups.loc[groups.qty.gt(1),'amount'].to_numpy() < base.loc[base.qty.gt(1),'amount'].to_numpy()).all())
        self.assertGreater(burst.dtd.isin([110,45,5]).mean(),.5)

    def test_nested_rbd_uses_one_shared_cabin_inventory(self):
        p,cohort=prepare_cohort('balanced',14,{'demand':100,'capacity':40,'businessCapacity':4})
        run=simulate_cohort(p,cohort,policy={'kind':'gated'})
        for offer in run['offers'].itertuples():
            self.assertEqual(offer.inventory_scope,'shared_cabin')
            offered=offer.open_rbds.split(',')
            ascending=RBD_DESCENDING[offer.cabin][::-1]
            index=ascending.index(offer.rbd)
            self.assertEqual(offered,list(ascending[index:]))
        self.assertGreater(run['offers'].rbd.nunique(),2)


class ExperimentTests(unittest.TestCase):
    def test_batch_paired_inputs_and_metrics_without_truth_at_fit_boundary(self):
        from backend.service import _analyze
        calls=[];progress=[]
        def checked(*args,**kwargs):
            self.assertNotIn('truth',kwargs)
            self.assertEqual(len(args),7)
            calls.append(args)
            return _analyze(*args,**kwargs)
        with patch('backend.service._analyze',side_effect=checked):
            result=run_experiments({'params':{'demand':70,'capacity':40,'businessCapacity':3},
                'research':{'scenarios':['balanced','mixed_wtp'],'runs':2,'seed_start':77}},
                progress=lambda done,total,message:progress.append((done,total)))
        self.assertEqual(len(calls),8)
        for i in range(0,8,2):
            self.assertIs(calls[i][0],calls[i+1][0])
            self.assertIs(calls[i][1],calls[i+1][1])
        self.assertEqual(result['summary']['generated_realizations'],4)
        self.assertEqual(result['summary']['paired_runs'],4)
        self.assertEqual(progress[-1],(8,8))
        self.assertEqual(len(result['tables']['all_runs']),8)
        json.dumps(result,allow_nan=False)

    def test_failed_fits_are_kept_and_not_counted_as_successes(self):
        from backend.service import _analyze
        def failure(*args,**kwargs):
            if args[3]['method']=='v3': raise ValueError('Deliberate convergence failure')
            return _analyze(*args,**kwargs)
        with patch('backend.service._analyze',side_effect=failure):
            result=run_experiments({'research':{'scenarios':['balanced'],'runs':1}})
        self.assertEqual(result['summary']['failed_fits'],1)
        self.assertEqual(result['summary']['paired_runs'],0)
        self.assertEqual(len(result['tables']['failures']),1)
        self.assertEqual(result['tables']['failures'][0]['method'],'v3')

    def test_cancellation_and_example_exports(self):
        with self.assertRaises(InterruptedError):
            run_experiments({'research':{'scenarios':['balanced'],'runs':1}},cancelled=lambda:True)
        result=generate_example({'research':{'scenario':'closure','seed_start':42}})
        self.assertEqual(set(result['bundle'])&{'bookings','offers','passengers'}, {'bookings','offers','passengers'})
        self.assertEqual(result['summary']['primary_seats'],300)
        json.dumps(result,allow_nan=False)

    def test_calibrated_training_seeds_cannot_be_reused_for_validation(self):
        with self.assertRaisesRegex(ValueError,'пересекаются'):
            run_experiments({'model':{'calibration':{'training_seeds':[43]}},
                'research':{'scenarios':['balanced'],'runs':2,'seed_start':42}})


class PolicyTests(unittest.TestCase):
    def test_independent_validation_and_profit_definition(self):
        result=run_policies({'params':{'demand':65,'capacity':35,'businessCapacity':3},
            'research':{'training_runs':1,'validation_runs':2,'seed_start':25,
                'variable_cost':700,'fixed_cost':50000,'objective':'profit'}})
        splits={row['phase']:row['seeds'] for row in result['tables']['seed_split']}
        self.assertFalse(set(splits['training'])&set(splits['holdout']))
        self.assertEqual(len(result['tables']['validation_runs']),8)
        for row in result['tables']['validation_runs']:
            self.assertEqual(row['profit'],row['revenue']-row['net']*700-50000)
        self.assertEqual(result['summary']['candidate_count'],12)
        self.assertTrue(any(r['pareto'] for r in result['tables']['comparison']))
        json.dumps(result,allow_nan=False)

    def test_profit_requires_costs_and_infeasible_load_not_silently_claimed(self):
        with self.assertRaises(ValueError):
            run_policies({'research':{'objective':'profit'}})
        result=run_policies({'params':{'demand':10,'capacity':180},
            'research':{'training_runs':1,'validation_runs':2,'min_load':99}})
        self.assertFalse(result['summary']['feasible_training_policy'])
        self.assertTrue(any('не найден' in warning for warning in result['warnings']))
        self.assertNotIn('profit',result['tables']['comparison'][0])


if __name__=='__main__': unittest.main()
