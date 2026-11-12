"""Exercise every research job and export with shipped fixtures.

Run: python scripts/verify_research.py --output outputs/research-verification
"""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from backend.jobs import JobManager
from backend.reporting import build_report,build_run_archive
from backend.simulation import DEFAULTS
from backend.storage import RunStore,code_fingerprint


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'outputs/research-verification')
    output=parser.parse_args().output;output.mkdir(exist_ok=True,parents=True)
    def file(name):
        source=ROOT/'examples/curator'/name
        return {'name':name,'text':source.read_text(encoding='utf-8-sig')}
    imports={'mode':'import','params':dict(DEFAULTS),
        'files':[file('bookings-SU1122-2027-05-22.csv')],
        'reference_files':[file('passengers-SU1122-2027-05-22.csv')],'offers_files':[]}
    base={'mode':'simulation','params':{**DEFAULTS,'demand':100,'capacity':65,'businessCapacity':5},'files':[],'offers_files':[],'reference_files':[]}
    cases=[('evaluate',imports),
        ('sensitivity',{**imports,'research':{'sigma_values':[.35,.45],'prior_values':[.1,.5],'demand_values':[280,300],'bootstrap_runs':3}}),
        ('calibration',{**base,'research':{'training_runs':2,'validation_runs':3,'sigma_values':[.35,.45],'beta_time_values':[0,-.1],'beta_qty_values':[0],'seed_start':2000}}),
        ('forecast',{**imports,'research':{'cutoffs':[90,30,14,7],'forecast_runs':40}}),
        ('policies',{**base,'research':{'training_runs':3,'validation_runs':5,'variable_cost':3000,'fixed_cost':150000,'objective':'profit','min_load':50,'seed_start':7000}}),
        ('generate_example',{**base,'research':{'scenario':'closure','seed_start':9000}})]
    summaries=[]
    with tempfile.TemporaryDirectory(prefix='kontur-integration-') as temp:
        store=RunStore(Path(temp)/'jobs.sqlite3',code_fingerprint(ROOT));manager=JobManager(store)
        try:
            for action,payload in cases:
                started=time.perf_counter();identifier=manager.submit(action,payload)['job']['id']
                manager.futures[identifier].result(timeout=300)
                response=manager.poll(identifier)
                if response['job']['status']!='completed': raise RuntimeError(str(response))
                record=store.get(identifier,with_payload=True)
                (output/(action+'-report.html')).write_bytes(build_report(record))
                (output/(action+'-run.zip')).write_bytes(build_run_archive(record))
                (output/(action+'-result.json')).write_text(json.dumps(response['lab'],ensure_ascii=False,allow_nan=False,indent=2),encoding='utf-8')
                summary={'action':action,'status':'completed','seconds':round(time.perf_counter()-started,2),'summary':response['lab'].get('summary')}
                summaries.append(summary);print(json.dumps(summary,ensure_ascii=False),flush=True)
        finally: manager.shutdown()
    (output/'integration-summary.json').write_text(json.dumps(summaries,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__': main()
