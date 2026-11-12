"""Research correctness: seat metrics, independent splits and prefix isolation."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from backend.evaluation import parse_passengers,evaluate_reference,run_evaluation
from backend.laboratory import generate_bundle
from backend.research import run_calibration,run_sensitivity,run_forecast,_prefix_bundle
from backend.service import _model_options,_analyze


ROOT=Path(__file__).resolve().parents[1]
PARAMS={'demand':80,'capacity':60,'businessCapacity':4,'seed':42}


class EvaluationTests(unittest.TestCase):
    def test_native_passengers_cancel_without_qty_and_unknown_hidden_cabin(self):
        path=ROOT/'examples/curator/passengers-SU1122-2027-05-22.csv'
        if not path.exists(): self.skipTest('Native reference fixture not installed')
        ref=parse_passengers([{'name':path.name,'text':path.read_text(encoding='utf-8-sig')}],'2027-05-22')
        primary=ref.loc[ref.outcome.isin(['success','waste'])]
        self.assertEqual(primary.qty.sum(),300)
        self.assertTrue(ref.loc[ref.outcome.eq('cancel'),'qty'].isna().all())
        self.assertTrue(ref.loc[ref.outcome.eq('waste'),'cabin'].isna().all())

    def test_seat_weighted_budget_metric_not_unweighted_group_metric(self):
        reference=pd.DataFrame([{'qty':1,'amount':10,'outcome':'success','period':'0–7','cabin':'Y'},
                                {'qty':9,'amount':100,'outcome':'waste','period':'0–7','cabin':None}])
        result={'params':{'flight':'T','demand':10},'summary':{'gross':1},
                'periods':[{'period':p,'estimated':9 if p=='0–7' else 0,'lower':0,'upper':9} for p in ('91+','31–90','15–30','8–14','0–7')],
                'reconstructed':[{'qty':1,'amount':10,'outcome':'success','cabin':'Y'},
                                 {'qty':9,'amount':90,'outcome':'waste','cabin':None}]}
        checked=evaluate_reference(result,reference)
        self.assertAlmostEqual(checked['summary']['budget_wasserstein'],9.)
        self.assertEqual(checked['summary']['hidden_period_wape'],0.)
        self.assertIsNone(checked['summary']['hidden_cabin_mae'])
        self.assertEqual(checked['tables']['budgets'][1]['actual'],100.)

    def test_reference_not_passed_to_inference(self):
        bundle=generate_bundle('balanced',42,PARAMS)
        fitted=_analyze(bundle['bookings'],bundle['offers'],bundle['params'],_model_options({}),'import',bundle['horizon_start'],bundle['horizon_end'])
        calls=[]
        def calculate(payload):
            calls.append(payload)
            return {'ok':True,'results':{bundle['params']['flight']:fitted},'params':bundle['params']}
        with patch('backend.service.calculate',side_effect=calculate),patch('backend.evaluation.parse_passengers',return_value=bundle['passengers']):
            result=run_evaluation({'reference_files':[{'name':'secret','text':'hidden truth'}],'params':PARAMS})
        self.assertTrue(result['ok'])
        self.assertNotIn('reference_files',calls[0]);self.assertNotIn('hidden truth',json.dumps(calls[0]))

    def test_simulation_evaluation_uses_post_fit_synthetic_events_without_upload(self):
        checked=run_evaluation({'mode':'simulation','params':PARAMS})
        self.assertTrue(checked['ok'])
        self.assertEqual(checked['reference_source'],'synthetic_simulation_events_after_fit')
        self.assertEqual(checked['tables']['flights'][0]['reference_primary_seats'],80)
        self.assertTrue(checked['tables']['periods'])

    def test_uploaded_reference_from_different_run_is_rejected_even_when_totals_match(self):
        from backend.service import calculate
        response=calculate({'mode':'simulation','params':PARAMS})
        truth=pd.DataFrame(response['results']['SU1122']['events'])
        first=truth.index[truth.outcome.eq('success')][0]
        truth.loc[first,'timestamp']=(pd.Timestamp(truth.loc[first,'timestamp'])+pd.Timedelta(seconds=1)).isoformat()
        with patch('backend.service.calculate',return_value=response),patch('backend.evaluation.parse_passengers',return_value=truth):
            with self.assertRaisesRegex(ValueError,'времена|Времена'):
                run_evaluation({'reference_files':[{'name':'wrong.csv','text':'placeholder'}]})

    def test_uploaded_reference_wrong_price_is_rejected(self):
        from backend.service import calculate
        response=calculate({'mode':'simulation','params':PARAMS})
        truth=pd.DataFrame(response['results']['SU1122']['events'])
        first=truth.index[truth.outcome.eq('success')][0];truth.loc[first,'price']+=100
        with patch('backend.service.calculate',return_value=response),patch('backend.evaluation.parse_passengers',return_value=truth):
            with self.assertRaisesRegex(ValueError,'цена'):
                run_evaluation({'reference_files':[{'name':'wrong.csv','text':'placeholder'}]})


class ResearchTests(unittest.TestCase):
    def test_behavior_options_validated_and_mixture_changes_model(self):
        self.assertEqual(_model_options({})['beta_time'],0.)
        for bad in [{'sigma':0},{'beta_time':3},{'mixture':'true'},{'calibration':{}}]:
            with self.assertRaises(ValueError): _model_options(bad)
        bundle=generate_bundle('balanced',42,PARAMS)
        fitted=_analyze(bundle['bookings'],bundle['offers'],bundle['params'],_model_options({'mixture':True,'beta_time':-.1}),
                        'import',bundle['horizon_start'],bundle['horizon_end'])
        self.assertEqual(fitted['diagnostics']['assumptions']['mixture_components'],2)

    def test_calibration_training_and_validation_disjoint_and_report_failures(self):
        p={'params':PARAMS,'research':{'training_runs':1,'validation_runs':1,'sigma_values':[.45],
                                     'compare_mixture':False,'scenarios':['balanced'],'seed_start':100}}
        result=run_calibration(p)
        self.assertFalse(set(result['training_fingerprints'])&set(result['validation_fingerprints']))
        self.assertEqual(result['selected_model']['calibration']['training_seeds'],[100])
        self.assertEqual({r['seed'] for r in result['tables']['holdout_runs']},{100100})
        self.assertEqual({r['method'] for r in result['tables']['holdout']},{'selected','current','baseline'})
        json.dumps(result,allow_nan=False)

    def test_sensitivity_refits_and_bootstrap_reports_successful_replicates(self):
        import backend.research as research
        p={'params':PARAMS,'research':{'sigma_values':[.45],'prior_values':[.5],'demand_values':[80],
                                     'bootstrap_runs':2,'scenarios':['balanced']}}
        with patch('backend.research._analyze_bundle',wraps=research._analyze_bundle) as fit:
            result=run_sensitivity(p)
        self.assertEqual(fit.call_count,3)
        self.assertEqual(len(result['tables']['bootstrap']),2)
        self.assertEqual(len(result['tables']['sensitivity_envelope']),5)
        self.assertIn('не доверительный',' '.join(result['warnings']))

    def test_prefix_excludes_future_operations_and_offer_states(self):
        bundle=generate_bundle('balanced',42,PARAMS)
        past,cut,fraction=_prefix_bundle(bundle,30)
        self.assertTrue(past['bookings'].timestamp.lt(cut).all())
        self.assertTrue(past['offers'].end.le(cut).all())
        self.assertTrue(past['offers'].start.lt(cut).all())
        self.assertGreater(fraction,0);self.assertLess(fraction,1)
        self.assertLess(past['params']['demand'],bundle['params']['demand'])

    def test_future_mutations_cannot_change_forecast(self):
        bundle=generate_bundle('balanced',42,PARAMS);changed={**bundle}
        prefix,cut,_=_prefix_bundle(bundle,30)
        bookings=bundle['bookings'].copy();future=bookings.timestamp.ge(cut)
        bookings.loc[future,'commercial']=bookings.loc[future,'commercial']*2
        offers=bundle['offers'].copy();offers.loc[offers.start.ge(cut),'price']*=9
        changed['bookings']=bookings;changed['offers']=offers
        p={'params':PARAMS,'research':{'cutoff_dtd':30,'forecast_runs':20}}
        with patch('backend.research._bundle',return_value=bundle): first=run_forecast(p)
        with patch('backend.research._bundle',return_value=changed): second=run_forecast(p)
        a=first['tables']['forecasts'][0];b=second['tables']['forecasts'][0]
        self.assertEqual(a['predicted_net'],b['predicted_net']);self.assertEqual(a['lower'],b['lower'])
        self.assertNotEqual(a['actual_net'],b['actual_net'])
        self.assertLess(pd.Timestamp(a['fit_max_timestamp']),pd.Timestamp(a['cutoff']))
        self.assertLessEqual(pd.Timestamp(a['fit_max_offer_end']),pd.Timestamp(a['cutoff']))

    def test_research_cancellation_callback_stops_work(self):
        with self.assertRaises(InterruptedError): run_forecast({'params':PARAMS},cancelled=lambda:True)

    def test_empty_optional_demand_grid_uses_current_N_as_in_browser(self):
        result=run_sensitivity({'params':PARAMS,'research':{'sigma_values':[.45],'prior_values':[.5],'demand_values':[]}})
        self.assertEqual(result['summary']['configurations'],1)
        self.assertEqual(result['summary']['successful'],1)


if __name__=='__main__': unittest.main()
