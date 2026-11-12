"""History, cancellation, reference isolation and portable reports."""
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
import zipfile

from backend.jobs import JobManager
from backend.reporting import build_report,build_run_archive
from backend.storage import RunStore


class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='kontur-history-test-')
        self.path=Path(self.temp.name)/'history.sqlite3'
        self.store=RunStore(self.path,'test-source-hash')

    def tearDown(self): self.temp.cleanup()

    def test_datasets_deduplicate_without_losing_names_and_reopen(self):
        payload={'files':[{'name':'one.csv','text':'a;b\n1;2'},{'name':'two.csv','text':'a;b\n1;2'}],'reference_files':[]}
        identifier=self.store.create('calculate','Первый',payload)
        self.store.complete(identifier,{'ok':True,'summary':{'n':2}})
        reopened=RunStore(self.path)
        self.assertEqual(reopened.get(identifier,with_payload=True)['payload'],payload)
        with self.store.connection() as db: count=db.execute('SELECT count(*) FROM datasets').fetchone()[0]
        self.assertEqual(count,1)
        self.assertEqual(reopened.history()[0]['source_hash'],'test-source-hash')

    def test_reports_escape_untrusted_values_and_include_reproducible_inputs(self):
        payload={'files':[{'name':'<input>.csv','text':'a;b\n"<script>";2'}]}
        identifier=self.store.create('experiments','<script>alert(1)</script>',payload)
        result={'ok':True,'summary':{'value':'<img src=x onerror=x>'},'tables':{'errors':[{'label':'<script>unsafe</script>','value':3}]},'warnings':['<b>text</b>']}
        self.store.complete(identifier,result)
        record=self.store.get(identifier,with_payload=True)
        html=build_report(record).decode('utf-8')
        self.assertNotIn('<script>',html);self.assertIn('&lt;script&gt;',html)
        with zipfile.ZipFile(io.BytesIO(build_run_archive(record))) as archive:
            self.assertIsNone(archive.testzip())
            self.assertEqual(json.loads(archive.read('result.json')),result)
            inputs=[n for n in archive.namelist() if n.startswith('inputs/')]
            self.assertEqual(len(inputs),1)
            self.assertEqual(archive.read(inputs[0]).decode('utf-8-sig'),payload['files'][0]['text'])
            self.assertEqual(len([n for n in archive.namelist() if n.startswith('tables/')]),1)

    def test_evaluation_report_with_per_flight_details_is_not_a_simulation_result(self):
        record={'title':'Проверка по эталону','result':{'ok':True,'summary':{'flights':1},
                'results':{'SU1122':{'summary':{'mae':2},'tables':{'periods':[{'period':'0–7','estimated':3}]}}},
                'tables':{'flights':[{'flight':'SU1122','mae':2}]}}}
        self.assertIn('SU1122',build_report(record).decode('utf-8'))

    def test_running_job_cancels_cooperatively_and_keeps_partial_result(self):
        started=threading.Event();release=threading.Event()
        def execute(action,payload,progress,cancelled):
            progress(1,10,'Шаг один');started.set();release.wait(3)
            return {'ok':True,'summary':{'cancel_seen':cancelled()}}
        jobs=JobManager(self.store,execute)
        try:
            identifier=jobs.submit('experiments',{})['job']['id']
            self.assertTrue(started.wait(2))
            answer=jobs.cancel(identifier);self.assertEqual(answer['job']['status'],'cancelling')
            release.set();jobs.futures[identifier].result(timeout=3)
            final=jobs.poll(identifier)
            self.assertEqual(final['job']['status'],'cancelled');self.assertTrue(final['lab']['summary']['cancel_seen'])
            self.assertEqual(final['job']['completed'],1)
        finally: release.set();jobs.shutdown()

    def test_job_failure_and_process_restart_are_visible(self):
        def execute(*args): raise ValueError('Неверные исходные данные')
        jobs=JobManager(self.store,execute)
        try:
            identifier=jobs.submit('evaluate',{})['job']['id']
            jobs.futures[identifier].result(timeout=3)
            self.assertEqual(jobs.poll(identifier)['job']['status'],'failed')
            self.assertIn('Неверные',jobs.poll(identifier)['job']['error'])
        finally: jobs.shutdown()
        stale=self.store.create('experiments','Незавершённый',{},status='running')
        self.store.recover_interrupted()
        self.assertEqual(self.store.get(stale)['status'],'interrupted')


if __name__=='__main__': unittest.main()
