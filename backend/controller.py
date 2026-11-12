"""UI-independent request routing. References never cross the fitting boundary."""
import json

from .csv_io import validate_files
from .jobs import TITLES
from .storage import dumps
from .simulation import DEFAULTS


class Controller:
    def __init__(self,store,jobs,calculate):
        self.store=store;self.jobs=jobs;self.calculate=calculate

    def handle(self,payload,context):
        request_id=payload.get('request_id','') if isinstance(payload,dict) else ''
        action=payload.get('action','calculate') if isinstance(payload,dict) else 'calculate'
        try:
            if not isinstance(payload,dict) or not isinstance(request_id,str) or len(request_id)>128:
                raise ValueError('Некорректный запрос интерфейса.')
            if action=='history':
                answer={'ok':True,'history':self.store.history()}
            elif action in ('poll_job','cancel_job'):
                identifier=payload.get('job_id')
                if not isinstance(identifier,str): raise ValueError('Не указан идентификатор исследования.')
                answer=self.jobs.poll(identifier) if action=='poll_job' else self.jobs.cancel(identifier)
            elif action in ('load_run','replay_run'):
                identifier=payload.get('run_id') or payload.get('id')
                record=self.store.get(identifier,with_payload=True)
                saved=record.get('result')
                if action=='replay_run':
                    replay={**record['payload'],'request_id':request_id,'action':record['action']}
                    return {**self.handle(replay,context),'action':'replay_run','replayed_action':record['action']}
                for field in ('files','offers_files','reference_files'):
                    context[field]=record['payload'].get(field,[])
                if record['action']=='calculate' and saved and 'results' in saved:
                    answer={**saved,'params':saved.get('params',record['payload'].get('params',{})),
                            'model':saved.get('model',record['payload'].get('model',{})),
                            'mode':saved.get('mode',record['payload'].get('mode','simulation'))}
                else:
                    params={**DEFAULTS,**record['payload'].get('params',{})}
                    answer={'ok':True,'lab':saved,'job':self.jobs.poll(identifier)['job'],
                            'params':params,'model':record['payload'].get('model',{}),
                            'mode':record['payload'].get('mode','simulation'),
                            'selected_flight':record['payload'].get('selected_flight') or params['flight']}
                answer.update({'run_id':record['id'],'saved_run':{k:record[k] for k in ('id','action','title','created_at','status','app_version','source_hash','config')}})
            elif action=='calculate' or action in TITLES:
                value={k:v for k,v in payload.items() if k not in ('request_id','job_id','run_id')}
                for field in ('files','offers_files','reference_files'):
                    if value.get(field) is None: value[field]=context.get(field,[])
                serialized=dumps(value)
                if len(serialized.encode('utf-8'))>32*1024*1024: raise ValueError('Запрос слишком большой. Уменьшите CSV.')
                validate_files([file for field in ('files','offers_files','reference_files') for file in value[field]])
                value=json.loads(serialized)
                for field in ('files','offers_files','reference_files'): context[field]=value[field]
                if action=='calculate':
                    fit_payload={k:v for k,v in value.items() if k in ('mode','params','model','files','offers_files','selected_flight')}
                    answer=self.calculate(fit_payload)
                    if answer.get('ok'):
                        identifier=self.store.create(action,'Сценарий '+str(answer.get('selected_flight','')),value,status='running')
                        self.store.complete(identifier,answer);answer={**answer,'run_id':identifier}
                else:
                    answer=self.jobs.submit(action,value)
                answer={**answer,'params':answer.get('params',value.get('params',{})),
                        'model':answer.get('model',value.get('model',{})),
                        'mode':answer.get('mode',value.get('mode','simulation'))}
            else: raise ValueError('Неизвестное действие интерфейса.')
            return {**answer,'request_id':request_id,'action':action}
        except (ValueError,TypeError,KeyError) as exc:
            return {'ok':False,'request_id':request_id,'action':action,'error':str(exc)}
