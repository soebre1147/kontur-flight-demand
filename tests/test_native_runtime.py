"""Python session state, native history restore and reference isolation."""
from pathlib import Path
import tempfile
import unittest

from backend.jobs import JobManager
from backend.native_runtime import NativeRuntime,normalize_payload
from backend.storage import RunStore


class NativeRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='kontur-native-runtime-')
        self.root=Path(self.temp.name);self.store=RunStore(self.root/'history.sqlite3')
        self.jobs=JobManager(self.store,execute=lambda a,p,progress,cancelled:{'ok':True,'title':a,'summary':{'n':p['params']['demand']}})
        self.state={};self.seen=[]
        def calculate(payload):
            self.seen.append(payload)
            if payload['params']['demand']<50:return {'ok':False,'error':'Спрос меньше наблюдаемых покупок'}
            return {'ok':True,'results':{'SU1122':{}},'params':payload['params'],'model':payload['model'],
                    'selected_flight':'SU1122','mode':payload['mode']}
        self.runtime=NativeRuntime(self.root,self.store,self.jobs,self.state,calculate)

    def tearDown(self):
        self.jobs.shutdown();self.temp.cleanup()

    def test_reference_files_never_reach_calculation_and_invalid_input_keeps_last_result(self):
        self.runtime.set_uploads(reference_files=[{'name':'ref.csv','text':'reference'}])
        good=self.runtime.calculate()
        self.assertTrue(good['ok']);self.assertNotIn('reference_files',self.seen[-1])
        bad=self.runtime.calculate({'params':{'demand':10}})
        self.assertFalse(bad['ok']);self.assertEqual(self.runtime.last_calculation,good)
        self.assertEqual(self.runtime.context['reference_files'][0]['name'],'ref.csv')

    def test_new_full_model_removes_previous_calibration(self):
        self.runtime.calculate({'model':{'calibration':{'training_seeds':[1]}}})
        self.runtime.calculate({'model':{'method':'baseline'}})
        self.assertNotIn('calibration',self.runtime.model)

    def test_evaluation_history_restores_files_and_navigation_without_becoming_a_calculation(self):
        self.runtime.calculate()
        payload=normalize_payload({'mode':'import','files':[{'name':'b.csv','text':'bookings'}],
                    'reference_files':[{'name':'p.csv','text':'passengers'}]})
        payload['research']={'cutoff_dtd':30}
        identifier=self.store.create('evaluate','Эталон',payload)
        result={'ok':True,'results':{'SU1122':{'summary':{'mae':2}}}}
        self.store.complete(identifier,result)
        self.runtime.load_run(identifier)
        self.assertIsNone(self.runtime.last_calculation)
        self.assertEqual(self.runtime.last_lab,result)
        self.assertEqual(self.state['pending_page'],'Исследование')
        self.assertEqual(self.runtime.context['reference_files'],payload['reference_files'])
        self.assertEqual(self.state['pending_research_settings']['action'],'evaluate')

    def test_replay_uses_original_settings_and_produces_a_new_job(self):
        first=self.runtime.submit('experiments',{'params':{'demand':101},'research':{'runs':1}})
        identifier=first['job']['id'];self.jobs.futures[identifier].result(timeout=3)
        self.runtime.calculate({'params':{'demand':200}})
        repeated=self.runtime.replay_run(identifier);new_id=repeated['job']['id']
        self.jobs.futures[new_id].result(timeout=3);self.runtime.poll_job(new_id)
        self.assertIsNone(self.runtime.last_calculation)
        self.assertNotEqual(identifier,new_id);self.assertEqual(self.runtime.last_lab['summary']['n'],101)
        self.assertEqual(self.store.get(new_id,with_payload=True)['payload']['research'],{'runs':1})

    def test_loaded_calculation_preserves_snapshot_and_sets_native_controls_pending(self):
        result=self.runtime.calculate({'params':{'demand':104}})
        self.runtime.calculate({'params':{'demand':208}})
        self.runtime.load_run(result['run_id'])
        self.assertEqual(self.runtime.params['demand'],104)
        self.assertEqual(self.state['pending_page'],'Симулятор')
        self.assertEqual(self.state['pending_loaded_payload']['params']['demand'],104)

    def test_saved_benchmark_replaces_the_active_job(self):
        import json
        (self.root/'docs').mkdir()
        (self.root/'docs/benchmark-1000.json').write_text(json.dumps({'ok':True,'summary':{'runs':1000}}))
        (self.root/'docs/benchmark-config.json').write_text(json.dumps(normalize_payload()))
        old=self.runtime.submit('evaluate')
        self.jobs.futures[old['job']['id']].result(timeout=3)
        response=self.runtime.load_benchmark()
        self.assertNotEqual(response['job']['id'],old['job']['id'])
        self.runtime.poll_job()
        self.assertEqual(self.runtime.last_lab['summary']['runs'],1000)
        record=self.store.get(response['job']['id'],with_payload=True)
        self.assertEqual(record['payload']['research']['source'],'saved_synthetic_benchmark_v4')


if __name__=='__main__': unittest.main()
