"""Bounded background execution with cooperative cancellation and SQLite status."""
from concurrent.futures import ThreadPoolExecutor
import logging
import threading

from .storage import dumps

TITLES={'evaluate':'Проверка по passengers','experiments':'Серия экспериментов',
        'calibration':'Калибровка на независимых данных','sensitivity':'Устойчивость и неопределённость',
        'forecast':'Прогноз до вылета','policies':'Сравнение тарифных политик',
        'generate_example':'Набор данных сценария'}


def execute_action(action,payload,progress,cancelled):
    if action=='evaluate':
        from .evaluation import run_evaluation
        function=run_evaluation
    elif action in ('experiments','policies','generate_example'):
        from .laboratory import run_experiments,run_policies,generate_example
        function={'experiments':run_experiments,'policies':run_policies,'generate_example':generate_example}[action]
    else:
        from .research import run_calibration,run_sensitivity,run_forecast
        function={'calibration':run_calibration,'sensitivity':run_sensitivity,'forecast':run_forecast}[action]
    return function(payload,progress=progress,cancelled=cancelled)


class JobManager:
    def __init__(self,store,execute=execute_action):
        self.store=store;self.execute=execute
        self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='kontur-research')
        self.lock=threading.Lock();self.flags={};self.futures={}
        self.store.recover_interrupted()

    def submit(self,action,payload):
        if action not in TITLES: raise ValueError('Неизвестное исследование.')
        with self.lock:
            active=sum(not future.done() for future in self.futures.values())
            if active>=3: raise ValueError('Уже запущены три исследования. Дождитесь результата или отмените одно из них.')
            identifier=self.store.create(action,TITLES[action],payload)
            flag=threading.Event();self.flags[identifier]=flag
            self.futures[identifier]=self.executor.submit(self._run,identifier,action,payload,flag)
        return self.poll(identifier)

    def _run(self,identifier,action,payload,flag):
        if flag.is_set():
            self.store.update(identifier,status='cancelled',message='Отменено до начала расчёта');return
        self.store.update(identifier,status='running',message='Подготовка данных')
        def progress(completed=0,total=0,message=''):
            self.store.update(identifier,completed=max(0,int(completed)),total=max(0,int(total)),message=str(message)[:1000])
        try:
            result=self.execute(action,payload,progress,flag.is_set)
            dumps(result)
            if flag.is_set(): self.store.complete(identifier,result,status='cancelled')
            elif result.get('ok',True): self.store.complete(identifier,result)
            else: self.store.update(identifier,status='failed',error=str(result.get('error','Не удалось выполнить исследование')),result_json=dumps(result))
        except Exception as exc:
            logging.exception('Research job failed: %s',action)
            self.store.update(identifier,status='cancelled' if flag.is_set() else 'failed',error=str(exc),message='Расчёт отменён' if flag.is_set() else 'Расчёт завершился ошибкой')

    def poll(self,identifier):
        record=self.store.get(identifier)
        job={k:record[k] for k in ('id','action','title','created_at','status','completed','total','message','error','app_version','source_hash')}
        answer={'ok':True,'job':job}
        if record.get('result') is not None: answer['lab']=record['result']
        return answer

    def cancel(self,identifier):
        with self.lock:
            flag=self.flags.get(identifier);future=self.futures.get(identifier)
            if flag is None: raise ValueError('Этот расчёт уже завершён либо сервер был перезапущен.')
            if not future.done():
                flag.set()
                if future.cancel(): self.store.update(identifier,status='cancelled',message='Отменено до начала расчёта')
                else: self.store.update(identifier,status='cancelling',message='Остановка после текущей операции модели')
        return self.poll(identifier)

    def shutdown(self):
        with self.lock:
            for flag in self.flags.values(): flag.set()
        self.executor.shutdown(wait=True,cancel_futures=True)
