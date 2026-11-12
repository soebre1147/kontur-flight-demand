"""Paired, reproducible experiments over independent forward scenarios.

No generated passenger truth is passed to the inverse calculation. Complete
offers, observed bookings, explicit assumptions and an external N are the only
inputs at that boundary. Passenger truth is used afterwards for evaluation.
"""
import math

import numpy as np

from .scenarios import SCENARIOS, generate_bundle, scenario_id
from .policies import _integer, _notify, _cancel, run_policies


def _settings(payload):
    if not isinstance(payload,dict): raise ValueError('Настройки эксперимента должны быть объектом.')
    details=payload.get('research',{})
    if not isinstance(details,dict): raise ValueError('Настройки исследования должны быть объектом.')
    return {**payload,**details}


def _finite(value):
    return isinstance(value,(int,float,np.integer,np.floating)) and not isinstance(value,(bool,np.bool_)) and math.isfinite(float(value))


def _means(rows):
    if not rows: return {}
    keys=set().union(*(row.keys() for row in rows))
    excluded={'seed','run','error','method','scenario','scenario_label','ok','elapsed_seconds'}
    return {key:float(np.mean([row[key] for row in rows if _finite(row.get(key))]))
        for key in sorted(keys-excluded) if any(_finite(row.get(key)) for row in rows)}


def _paired_intervals(rows,seed):
    """Bootstrap by seed, retaining cross-scenario dependence within a seed."""
    mapping={}
    for row in rows:
        if row['ok']: mapping.setdefault((row['scenario'],row['seed']),{})[row['method']]=row
    paired=[]
    for (scenario,current_seed),methods in mapping.items():
        if set(methods)!={'v3','baseline'}: continue
        new,base=methods['v3'],methods['baseline']
        difference=float(new['hidden_period_mae']-base['hidden_period_mae'])
        paired.append({'scenario':scenario,'seed':current_seed,'v3_mae':new['hidden_period_mae'],
            'baseline_mae':base['hidden_period_mae'],'mae_difference':difference,
            'v3_wins':difference<-1e-10,'tie':abs(difference)<=1e-10})
    by_seed={}
    for row in paired: by_seed.setdefault(row['seed'],[]).append(row['mae_difference'])
    clusters=np.asarray([np.mean(v) for v in by_seed.values()],float)
    if len(clusters)>1:
        rng=np.random.default_rng(seed+3816)
        boot=np.mean(rng.choice(clusters,size=(2000,len(clusters)),replace=True),axis=1)
        low,high=map(float,np.quantile(boot,[.025,.975]))
    else: low=high=None
    return paired,{'paired_runs':len(paired),'independent_seed_clusters':len(clusters),
        'mean_mae_difference':float(clusters.mean()) if len(clusters) else None,
        'difference_ci_low':low,'difference_ci_high':high,
        'v3_win_pct':float(np.mean([r['v3_wins'] for r in paired])*100) if paired else None,
        'tie_pct':float(np.mean([r['tie'] for r in paired])*100) if paired else None}


def run_experiments(payload,progress=None,cancelled=None):
    from .service import _analyze, _model_options, _json_safe
    from .evaluation import evaluate_reference
    import time
    config=_settings(payload)
    scenarios=config.get('scenarios',list(SCENARIOS))
    if not isinstance(scenarios,list) or not scenarios or len(scenarios)>10:
        raise ValueError('Выберите от 1 до 10 сценариев.')
    scenarios=[scenario_id(s) for s in scenarios]
    if len(scenarios)!=len(set(scenarios)): raise ValueError('Сценарии не должны повторяться.')
    runs=_integer(config.get('runs',100),'Число запусков',1,100)
    seed=_integer(config.get('seed_start',42),'Начальный seed',0,2**32-101)
    params=payload.get('params',{})
    model=payload.get('model',{})
    if not isinstance(model,dict): raise ValueError('Настройки модели должны быть объектом.')
    training_seeds=set((model.get('calibration') or {}).get('training_seeds') or [])
    if training_seeds.intersection(range(seed,seed+runs)):
        raise ValueError('Проверочные seed пересекаются с обучающими seed калибровки. Выберите другой начальный seed.')
    total=len(scenarios)*runs*2;done=0;records=[]
    for scenario in scenarios:
        for index in range(runs):
            _cancel(cancelled)
            current_seed=seed+index
            # Generate once; both estimators receive exactly the same logs.
            bundle=generate_bundle(scenario,current_seed,params)
            for method in ('baseline','v3'):
                _cancel(cancelled);started=time.perf_counter()
                row={'scenario':scenario,'scenario_label':SCENARIOS[scenario]['label'],
                    'seed':current_seed,'run':index+1,'method':method}
                try:
                    result=_analyze(bundle['bookings'],bundle['offers'],bundle['params'],
                        _model_options({**model,'method':method}),'simulation',
                        bundle['horizon_start'],bundle['horizon_end'])
                    evaluation=evaluate_reference(_json_safe(result),bundle['passengers'])
                    row.update(evaluation.get('summary',{}));row['ok']=True
                except Exception as exc:
                    row.update({'ok':False,'error':str(exc)[:1000]})
                row['elapsed_seconds']=time.perf_counter()-started;records.append(row)
                done+=1;_notify(progress,done,total,f'{SCENARIOS[scenario]["label"]}: запуск {index+1}/{runs}, {method}')
    aggregate=[]
    for scenario in scenarios:
        for method in ('baseline','v3'):
            selected=[r for r in records if r['scenario']==scenario and r['method']==method]
            valid=[r for r in selected if r['ok']]
            aggregate.append({'scenario':scenario,'scenario_label':SCENARIOS[scenario]['label'],'method':method,
                'runs':len(selected),'successes':len(valid),'failures':len(selected)-len(valid),**_means(valid)})
    paired,summary=_paired_intervals(records,seed)
    summary.update({'scenario_count':len(scenarios),'runs_per_scenario':runs,'generated_realizations':len(scenarios)*runs,
        'fits_attempted':len(records),'successful_fits':sum(r['ok'] for r in records),
        'failed_fits':sum(not r['ok'] for r in records),'seed_start':seed})
    overall=[]
    for method in ('baseline','v3'):
        subset=[r for r in records if r['method']==method]
        valid=[r for r in subset if r['ok']]
        overall.append({'method':method,'runs':len(subset),'successes':len(valid),'failures':len(subset)-len(valid),**_means(valid)})
    warnings=[
        'N задаётся извне. Совпадение общего числа скрытых мест не оценивает точность распределения; сравниваются периоды, кабинные метки, бюджеты и покрытие интервалов.',
        'Эталонные passengers передаются только функции оценки после восстановления. Модель получает bookings, полную фактическую историю offers и внешние предположения.',
        'Сценарии включают нарушения предположений модели: всплески, смесь бюджетов, закрытие продаж и зависимость бюджета от размера группы.',
        'Средние ошибки приводятся по успешным расчётам; число ошибок расчёта указано отдельно, неуспешные запуски остаются в таблице.',
        'Парная разность MAE = v3 − baseline: отрицательное значение означает преимущество v3. Bootstrap 95% интервал учитывает общие seed между сценариями.',
        'Это проверка на синтетических потоках, а не доказательство точности на реальном рынке. Калибровку выполняйте отдельно; не подбирайте параметры по этим проверочным seed.']
    if runs<10: warnings.append('Мало повторений: результаты подходят для проверки работы интерфейса; для оценки устойчивости запустите хотя бы 100 seed на сценарий.')
    return _json_safe({'ok':True,'title':'Пакетные эксперименты: v3 и базовый метод','summary':summary,
        'tables':{'by_scenario':aggregate,'overall':overall,'paired_runs':paired,'all_runs':records,
            'failures':[r for r in records if not r['ok']]},
        'charts':{'period_mae':[{'label':r['scenario_label']+' · '+r['method'],'value':r.get('hidden_period_mae')} for r in aggregate]},
        'warnings':warnings,'config':{'scenarios':scenarios,'runs':runs,'seed_start':seed,'model':model}})


def generate_example(payload,progress=None,cancelled=None):
    from .service import _json_safe
    _cancel(cancelled);config=_settings(payload)
    scenario=scenario_id(config.get('scenario','balanced'))
    seed=_integer(config.get('seed_start',42),'Seed',0,2**32-1)
    bundle=generate_bundle(scenario,seed,payload.get('params'))
    _notify(progress,1,1,'Создан согласованный комплект bookings + offers + passengers')
    operations=bundle['bookings'];truth=bundle['passengers']
    summary={'scenario':SCENARIOS[scenario]['label'],'seed':seed,'primary_seats':bundle['params']['demand'],
        'purchase_seats':int(operations.commercial.clip(lower=0).sum()),'cancel_seats':int(-operations.commercial.clip(upper=0).sum()),
        'hidden_seats':int(truth.loc[truth.outcome.eq('waste'),'qty'].sum()),
        'offer_intervals':len(bundle['offers'])}
    return _json_safe({'ok':True,'title':'Комплект данных для проверки','summary':summary,
        'tables':{'periods':truth.loc[truth.outcome.isin(['success','waste'])].groupby(['period','outcome'],sort=False).qty.sum().reset_index().to_dict('records')},
        'charts':{},'warnings':['Синтетический эталон passengers содержит полную правду сценария. Для восстановления используйте только bookings и offers; passengers загружайте в отдельный блок проверки.'],
        'bundle':{'params':bundle['params'],'scenario':scenario,
            'horizon_start':bundle['horizon_start'],'horizon_end':bundle['horizon_end'],
            'bookings':bundle['bookings'].to_dict('records'),'offers':bundle['offers'].to_dict('records'),
            'passengers':bundle['passengers'].to_dict('records')}})
