"""Behavioral and inference-boundary checks for the Python application."""
import inspect
import csv
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import pandas as pd

from backend.csv_io import parse_bookings
from backend.exports import serialize_export
from backend.service import calculate, _fit_v3
from backend.simulation import DEFAULTS, RBD_DESCENDING, primary_requests, simulate


ROOT=Path(__file__).resolve().parents[1]


def request(params=None,method='v3',**extra):
    return {'mode':'simulation','params':params or {},'model':{'method':method},**extra}


class SimulationTests(unittest.TestCase):
    def test_policy_comparison_preserves_primary_cohort(self):
        low={**DEFAULTS,'demand':251,'price':1000,'capacity':120,'businessCapacity':6}
        high={**low,'price':60000,'capacity':300,'businessCapacity':15}
        pd.testing.assert_frame_equal(primary_requests(low),primary_requests(high))
        first=simulate(low);second=simulate(high)
        columns=['id','timestamp','qty','cabin','amount']
        pd.testing.assert_frame_equal(first['requests'][columns],second['requests'][columns])
        self.assertNotEqual(int(first['bookings'].commercial.clip(lower=0).sum()),int(second['bookings'].commercial.clip(lower=0).sum()))

    def test_exact_primary_total_and_no_inventory_oversell(self):
        for seed in [1,42,215]:
            run=simulate({**DEFAULTS,'demand':137,'capacity':34,'businessCapacity':4,'cancel':60,'seed':seed})
            primary=run['events'].loc[run['events'].outcome.isin(['success','waste'])]
            self.assertEqual(int(primary.qty.sum()),137)
            for cabin,capacity in [('C',4),('Y',30)]:
                balances=run['bookings'].loc[run['bookings'].cabin.eq(cabin),'commercial'].cumsum()
                self.assertTrue(balances.between(0,capacity).all())
            self.assertTrue(run['offers'].available_seats.ge(0).all())
            self.assertEqual(set(run['offers'].cabin),{'C','Y'})
            self.assertEqual(RBD_DESCENDING,{'C':'JOCIZ','Y':'YXDSBMUPKHLQTENRGV'})

    def test_recorded_pre_event_offers_match_purchases(self):
        from model.math_reconstruction_v3 import prepare_inputs_v3
        run=simulate({**DEFAULTS,'demand':100,'capacity':20,'businessCapacity':2,'cancel':80})
        inputs=prepare_inputs_v3(run['bookings'],DEFAULTS['departure'],run['horizon_start'],run['horizon_end'],offers=run['offers'])
        self.assertEqual(int(inputs.purchases.qty.sum()),int(run['bookings'].commercial.clip(lower=0).sum()))
        self.assertTrue(run['offers'].available_seats.eq(0).any())
        for cabin,history in run['offers'].groupby('cabin'):
            self.assertEqual(history.start.min(),run['horizon_start'])
            self.assertEqual(history.end.max(),run['horizon_end'])


class ServiceTests(unittest.TestCase):
    def test_default_v3_exact_counts_source_separation_and_json(self):
        response=calculate(request())
        self.assertTrue(response['ok'],response.get('error'))
        result=response['results']['SU1122']
        self.assertEqual(result['diagnostics']['method'],'v3')
        self.assertEqual(sum(p['estimated'] for p in result['periods']),result['summary']['waste'])
        primary=[e for e in result['reconstructed'] if e['outcome']!='cancel']
        self.assertEqual(sum(e['qty'] for e in primary),DEFAULTS['demand'])
        for e in result['reconstructed']:
            self.assertIsInstance(e['reconstruction_status'],str)
            if e['outcome']=='waste':
                self.assertIsNone(e['source_row'])
                self.assertIsNone(e['cabin'])
                self.assertIsNone(e['price'])
                self.assertTrue(e['reason_is_modelled'])
        self.assertEqual(len(result['curve']),181)
        self.assertEqual(sum(v['net_delta'] for v in result['curve']),result['summary']['net'])
        self.assertTrue(all(p['lower']<=p['upper'] for p in result['periods']))
        json.dumps(response,allow_nan=False)

    def test_zero_and_complete_sales_remain_explicit(self):
        zero=calculate(request({'demand':40,'price':1000000,'budget':100}))
        self.assertTrue(zero['ok'],zero.get('error'))
        r=zero['results']['SU1122']
        self.assertEqual(r['summary']['gross'],0)
        self.assertEqual(r['diagnostics']['status'],'prior_only_no_sales')
        self.assertEqual(sum(e['qty'] for e in r['reconstructed']),40)
        self.assertTrue(any('Нет покупок' in warning for warning in r['diagnostics']['warnings']))
        all_sold=calculate(request({'demand':40,'capacity':200,'businessCapacity':0,'businessShare':0,'price':100,'budget':1000000}))
        self.assertTrue(all_sold['ok'],all_sold.get('error'))
        r=all_sold['results']['SU1122']
        self.assertEqual(r['summary']['waste'],0)
        self.assertTrue(all(p['estimated']==0 and p['lower']==0 and p['upper']==0 for p in r['periods']))

    def test_fit_boundary_never_receives_hidden_truth(self):
        self.assertEqual(list(inspect.signature(_fit_v3).parameters),['bookings','offers','params','options','start','end'])
        import backend.service as service
        original=service._fit_v3
        seen=[]
        def guarded(bookings,offers,params,options,start,end):
            self.assertNotIn('amount',bookings.columns)
            self.assertNotIn('outcome',bookings.columns)
            self.assertNotIn('reason',bookings.columns)
            seen.append(True)
            return original(bookings,offers,params,options,start,end)
        with patch.object(service,'_fit_v3',side_effect=guarded):
            response=calculate(request({'demand':80}))
        self.assertTrue(response['ok'],response.get('error'))
        self.assertEqual(seen,[True])

    def test_baseline_has_aggregates_not_invented_passengers(self):
        response=calculate(request({'demand':45,'price':1000000,'budget':100},method='baseline'))
        self.assertTrue(response['ok'],response.get('error'))
        result=response['results']['SU1122']
        self.assertEqual(result['diagnostics']['method'],'baseline')
        self.assertEqual(sum(e['qty'] for e in result['reconstructed']),45)
        self.assertTrue(all(e['record_level']=='агрегат периода' and e['flight']=='SU1122' and e['timestamp'] is None for e in result['reconstructed']))

    def test_original_curator_files_multi_flight_and_sources(self):
        files=[{'name':p.name,'text':p.read_text(encoding='utf-8-sig')} for p in sorted((ROOT/'examples'/'curator').glob('bookings-*.csv'))]
        self.assertEqual(len(files),2)
        response=calculate({'mode':'import','params':DEFAULTS,'files':files,'selected_flight':'SU1123'})
        self.assertTrue(response['ok'],response.get('error'))
        self.assertEqual(set(response['results']),{'SU1122','SU1123'})
        self.assertEqual(response['selected_flight'],'SU1123')
        for flight,result in response['results'].items():
            source=parse_bookings([f for f in files if flight in f['name']],DEFAULTS['departure'])
            self.assertEqual(result['summary']['gross'],int(source.commercial.clip(lower=0).sum()))
            self.assertEqual(result['summary']['canceled'],int(-source.commercial.clip(upper=0).sum()))
            self.assertEqual(len(result['events']),len(source))
            self.assertIsNone(result['diagnostics']['WAPE_pct'])
            self.assertTrue(any('доступность мест неизвестна' in w for w in result['diagnostics']['warnings']))
            for observed,original in zip(result['observed'],source.to_dict('records')):
                for key in ['cabin','rbd','price','source_row','source_file']:
                    self.assertEqual(observed[key],original[key])
                self.assertEqual(observed['signed_qty'],original['commercial'])
                self.assertIsNone(observed['amount'])
                self.assertEqual(observed['source_values']['occupied'],original['source_values']['occupied'])
            self.assertEqual(sum(e['qty'] for e in result['reconstructed'] if e['outcome']!='cancel'),300)
        json.dumps(response,allow_nan=False)

    def test_native_timestamp_and_physical_row_numbers(self):
        file={'name':'bookings-SU1122-2027-05-22.csv','text':'timestamp;cabin;rbd;commercial;price\n\n2027-05-01 10:00:00 UTC;Y;R;1;12000\n'}
        frame=parse_bookings([file],DEFAULTS['departure'])
        self.assertEqual(frame.timestamp.iloc[0],pd.Timestamp('2027-05-01 10:00:00Z'))
        self.assertEqual(int(frame.source_row.iloc[0]),3)

    def test_invalid_input_fails_instead_of_fabricating(self):
        valid='timestamp;cabin;rbd;commercial;price\n2027-05-01 10:00:00 UTC;Y;R;1;12000\n'
        base={'mode':'import','files':[{'name':'bookings-SU1122-2027-05-22.csv','text':valid}]}
        cases=[
            {**base,'files':base['files']*2},
            {**base,'files':[{'name':'bookings-SU1122-2027-05-22.csv','text':valid+valid.splitlines()[1]+'\n'}]},
            {**base,'files':[{'name':'bookings-SU1122-2027-05-22.csv','text':valid.replace(';1;12000',';1.0000000000000000001;12000')}]},
            {**base,'files':[{'name':'bookings-SU1122-2027-05-22.csv','text':valid.replace(';12000',';NaN')}]},
            {**base,'params':{'departure':'2020-01-01'}},
            request({'capacity':5,'businessCapacity':6}),
            request({'demand':1501}),
            request({'price':float('inf')}),
            request({'demand':True}),
            {**base,'files':[{'name':'bad.csv','text':'x'* (10*1024*1024+1)}]},
        ]
        for case in cases:
            with self.subTest(case=str(case)[:90]):
                response=calculate(case)
                self.assertFalse(response['ok'])
                self.assertTrue(response['error'])
                self.assertNotIn('results',response)

    def test_cancel_only_import_requires_actual_offers_for_v3(self):
        cancel={'name':'bookings-SU1122-2027-05-22.csv','text':'timestamp;cabin;rbd;commercial;price\n2027-05-01 10:00:00 UTC;Y;R;-1;12000\n'}
        request_payload={'mode':'import','params':{'demand':10},'files':[cancel]}
        response=calculate(request_payload)
        self.assertFalse(response['ok'])
        self.assertIn('offers',response['error'])
        offers={'name':'offers.csv','text':'flight;cabin;start;end;price;open;available_seats;rbd\nSU1122;Y;2026-11-23 00:00:00 UTC;2027-05-23 00:00:00 UTC;12000;true;180;R\n'}
        response=calculate({**request_payload,'offers_files':[offers]})
        self.assertTrue(response['ok'],response.get('error'))
        r=response['results']['SU1122']
        self.assertEqual(r['diagnostics']['status'],'prior_only_no_sales')
        self.assertEqual(r['summary']['minBalance'],-1)
        self.assertIsNone(next(e for e in r['reconstructed'] if e['outcome']=='cancel')['amount'])


class ExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        response=calculate(request({'demand':80,'capacity':40,'businessCapacity':2}))
        if not response['ok']: raise AssertionError(response.get('error'))
        cls.result=response['results']['SU1122']

    def test_bookings_roundtrip_preserves_signed_operations_and_raw_source_json(self):
        fixture=ROOT/'examples'/'curator'/'bookings-SU1122-2027-05-22.csv'
        response=calculate({'mode':'import','files':[{'name':fixture.name,'text':fixture.read_text(encoding='utf-8-sig')}],
            'model':{'method':'baseline'}})
        self.assertTrue(response['ok'],response.get('error'))
        result=response['results']['SU1122']
        body,name,mime=serialize_export(result,'bookings')
        self.assertTrue(body.startswith(b'\xef\xbb\xbf"timestamp";'))
        self.assertIn(b'\r\n',body)
        self.assertEqual(mime,'text/csv;charset=utf-8')
        restored=parse_bookings([{'name':name,'text':body.decode('utf-8-sig')}],DEFAULTS['departure'])
        original=parse_bookings([{'name':fixture.name,'text':fixture.read_text(encoding='utf-8-sig')}],DEFAULTS['departure'])
        columns=['timestamp','flight','cabin','rbd','commercial','price']
        pd.testing.assert_frame_equal(original[columns],restored[columns])
        cells=list(csv.DictReader(io.StringIO(body.decode('utf-8-sig')),delimiter=';'))
        self.assertEqual(json.loads(cells[0]['source_values']),result['observed'][0]['source_values'])
        self.assertEqual(cells[0]['amount'],'')

    def test_reconstruction_export_keeps_all_columns_and_serializes_structured_values(self):
        result=json.loads(json.dumps(self.result))
        result['reconstructed'][0]['source_values']={'quoted':'"a;b"','unicode':'Москва','items':[1,None,True]}
        before=json.dumps(result,sort_keys=True)
        with patch('backend.service._fit_v3',side_effect=AssertionError('Export must not refit')):
            body,name,_=serialize_export(result,'reconstructed')
        rows=list(csv.DictReader(io.StringIO(body.decode('utf-8-sig')),delimiter=';'))
        expected_columns=set().union(*(row.keys() for row in result['reconstructed']))
        self.assertTrue(expected_columns.issubset(rows[0]))
        self.assertEqual(json.loads(rows[0]['source_values']),result['reconstructed'][0]['source_values'])
        self.assertEqual(sum(int(row['qty']) for row in rows if row['outcome']!='cancel'),80)
        hidden=[row for row in rows if row['outcome']=='waste']
        self.assertTrue(hidden)
        self.assertTrue(all(row['price']=='' and row['source_row']=='' and row['reason_is_modelled']=='true' for row in hidden))
        self.assertTrue(name.startswith('passengers-reconstructed-SU1122-'))
        self.assertEqual(before,json.dumps(result,sort_keys=True))

    def test_all_other_exports_and_proxy_offers_guard(self):
        for kind,field in [('events','events'),('daily','curve'),('periods','periods'),('offers','offers')]:
            body,name,mime=serialize_export(self.result,kind)
            rows=list(csv.DictReader(io.StringIO(body.decode('utf-8-sig')),delimiter=';'))
            self.assertEqual(len(rows),len(self.result[field]))
            self.assertTrue(name.endswith('.csv'))
            self.assertIn('csv',mime)
        body,name,mime=serialize_export(self.result,'params')
        snapshot=json.loads(body)
        self.assertEqual(snapshot['params'],self.result['params'])
        self.assertEqual(snapshot['model_config'],self.result['model'])
        self.assertEqual(snapshot['diagnostics'],self.result['diagnostics'])
        self.assertEqual((name,mime),('scenario-SU1122.json','application/json'))
        proxy={**self.result,'offers':[{'flight':'SU1122','offer_source':'assumed_last_purchase_prices','price':12000}]}
        with self.assertRaisesRegex(ValueError,'offers'):
            serialize_export(proxy,'offers')
        actual=dict(self.result['offers'][0]);mixed={**proxy,'offers':proxy['offers']+[actual]}
        body,_,_=serialize_export(mixed,'offers')
        rows=list(csv.DictReader(io.StringIO(body.decode('utf-8-sig')),delimiter=';'))
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['offer_source'],'supplied_offer_intervals')
        with self.assertRaises(ValueError): serialize_export(self.result,'unknown')


if __name__=='__main__': unittest.main()
