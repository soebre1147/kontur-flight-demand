"""Python session orchestration for native Streamlit screens.

The runtime accepts Python values directly. Reference passengers are excluded
from the inverse-calculation boundary and used only by evaluation jobs.
"""
from copy import deepcopy
import json
from pathlib import Path

from .csv_io import validate_files
from .service import DEFAULTS,MODEL_DEFAULTS,_model_options
from .storage import dumps

UPLOAD_FIELDS=('files','offers_files','reference_files')
RUNNING={'queued','running','cancelling'}


def normalize_payload(value=None):
    value=value or {}
    params={**DEFAULTS,**(value.get('params') or {})}
    if hasattr(params.get('departure'),'isoformat'): params['departure']=params['departure'].isoformat()
    return {'mode':value.get('mode','simulation'),'params':params,
            'model':{**MODEL_DEFAULTS,**(value.get('model') or {})},
            'selected_flight':value.get('selected_flight') or params['flight'],
            **{field:deepcopy(value.get(field) or []) for field in UPLOAD_FIELDS}}


def uploaded_files(files):
    """Decode native Streamlit uploads; original paths are never read."""
    answer=[]
    for file in files or []:
        raw=file.getvalue()
        try: body=raw.decode('utf-8-sig')
        except UnicodeDecodeError: raise ValueError(f'{file.name}: сохраните CSV в UTF-8.') from None
        answer.append({'name':file.name,'text':body})
    validate_files(answer)
    return answer


class NativeRuntime:
    def __init__(self,root,store,jobs,state,calculate):
        self.root=Path(root);self.store=store;self.jobs=jobs;self.state=state;self._calculate=calculate
        if 'simulator_payload' not in state: state['simulator_payload']=normalize_payload()

    @property
    def params(self): return self.state['simulator_payload']['params']
    @property
    def model(self): return self.state['simulator_payload']['model']
    @property
    def context(self): return {field:deepcopy(self.state['simulator_payload'].get(field,[])) for field in UPLOAD_FIELDS}
    @property
    def last_calculation(self): return self.state.get('native_last_calculation')
    @property
    def last_lab(self): return self.state.get('native_last_lab')
    @property
    def last_lab_id(self): return self.state.get('native_last_lab_id')
    @property
    def active_job_id(self): return self.state.get('native_active_job_id')

    def get_payload(self): return deepcopy(self.state['simulator_payload'])
    current_payload=get_payload

    def _merge(self,payload=None):
        value=self.get_payload();payload=payload or {}
        for field in ('mode','selected_flight'):
            if payload.get(field) is not None: value[field]=payload[field]
        if payload.get('params') is not None: value['params']={**value['params'],**payload['params']}
        if payload.get('model') is not None: value['model']={**MODEL_DEFAULTS,**payload['model']}
        for field in UPLOAD_FIELDS:
            if payload.get(field) is not None: value[field]=deepcopy(payload[field])
        for field in ('research',):
            if field in payload: value[field]=deepcopy(payload[field])
        if len(dumps(value).encode('utf-8'))>32*1024*1024: raise ValueError('Уменьшите размер CSV: запрос превышает 32 МБ.')
        validate_files([file for field in UPLOAD_FIELDS for file in value[field]])
        return value

    def set_uploads(self,files=None,offers_files=None,reference_files=None):
        values={'files':files,'offers_files':offers_files,'reference_files':reference_files}
        self.state['simulator_payload']=normalize_payload(self._merge(values))

    def set_calculation(self,response):
        if response and response.get('ok') and response.get('results'):
            self.state['native_last_calculation']=deepcopy(response)

    def calculate(self,payload=None):
        try:
            value=self._merge(payload);self.state['simulator_payload']=normalize_payload(value)
            inputs={key:value[key] for key in ('mode','params','model','files','offers_files','selected_flight')}
            response=self._calculate(inputs)
            if not response.get('ok'): return response
            identifier=self.store.create('calculate','Сценарий '+str(response.get('selected_flight','')),value,status='running')
            self.store.complete(identifier,response);response={**response,'run_id':identifier}
            self.set_calculation(response)
            normalized=normalize_payload({**value,'params':response.get('params',value['params']),
                                          'model':response.get('model',value['model']),
                                          'selected_flight':response.get('selected_flight')})
            self.state['simulator_payload']=normalized
            return response
        except (ValueError,TypeError,KeyError) as exc: return {'ok':False,'error':str(exc)}

    def remember_lab(self,result,run_id):
        if result and result.get('ok',True):
            self.state['native_last_lab']=deepcopy(result)
            self.state['native_last_lab_id']=run_id

    def accept_job(self,response):
        job=response.get('job') or {}
        if job:
            self.state['native_active_job_id']=job['id'];self.state['native_last_job']=job
        if response.get('lab') and job.get('status') in ('completed','cancelled'):
            self.remember_lab(response['lab'],job['id'])
        return response

    def submit(self,action,payload=None):
        try:
            value=self._merge(payload);self.state['simulator_payload']=normalize_payload(value)
            return self.accept_job(self.jobs.submit(action,value))
        except (ValueError,TypeError,KeyError) as exc: return {'ok':False,'error':str(exc)}

    def poll_job(self,identifier=None):
        identifier=identifier or self.active_job_id
        if not identifier: return {'ok':False,'error':'Нет активного исследования.'}
        try: return self.accept_job(self.jobs.poll(identifier))
        except ValueError as exc: return {'ok':False,'error':str(exc)}

    def cancel_job(self,identifier=None):
        try: return self.accept_job(self.jobs.cancel(identifier or self.active_job_id))
        except ValueError as exc: return {'ok':False,'error':str(exc)}

    def load_run(self,identifier):
        record=self.store.get(identifier,with_payload=True)
        self.state.pop('native_last_calculation',None)
        self.state.pop('native_last_lab',None)
        self.state.pop('native_last_lab_id',None)
        self.state['simulator_payload']=normalize_payload(record['payload'])
        self.state['pending_loaded_payload']=self.get_payload()
        self.state['pending_research_settings']={'action':record['action'],'research':record['payload'].get('research',{})}
        if record['action']=='calculate':
            self.set_calculation({**(record.get('result') or {}),'run_id':identifier})
            self.state['pending_page']='Симулятор'
        else:
            self.remember_lab(record.get('result'),identifier)
            self.state['native_active_job_id']=identifier
            self.state['pending_page']='Исследование'
        return record

    def replay_run(self,identifier):
        record=self.store.get(identifier,with_payload=True)
        self.state.pop('native_last_calculation',None)
        self.state['simulator_payload']=normalize_payload(record['payload'])
        self.state['pending_loaded_payload']=self.get_payload()
        self.state['pending_research_settings']={'action':record['action'],'research':record['payload'].get('research',{})}
        if record['action']=='calculate':
            self.state['pending_page']='Симулятор'
            return self.calculate(record['payload'])
        self.state['pending_page']='Исследование'
        return self.submit(record['action'],record['payload'])

    def apply_model(self,config):
        model=_model_options(config)
        self.state['simulator_payload']['model']=deepcopy(model)
        self.state['pending_loaded_payload']=self.get_payload()
        self.state['pending_page']='Симулятор'
        return self.calculate()

    def load_benchmark(self):
        path=self.root/'docs/benchmark-1000.json'
        result=json.loads(path.read_text(encoding='utf-8'))
        payload=json.loads((self.root/'docs/benchmark-config.json').read_text(encoding='utf-8'))
        payload['research']={**payload.get('research',{}),'source':'saved_synthetic_benchmark_v4'}
        identifier=self.store.create('experiments','Сохранённая проверка: 10 сценариев × 100 seed',payload,status='running')
        self.store.update(identifier,total=2000,completed=2000)
        self.store.complete(identifier,result);self.remember_lab(result,identifier)
        self.state['native_active_job_id']=identifier
        return self.jobs.poll(identifier)
