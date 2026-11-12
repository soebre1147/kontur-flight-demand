"""Policy search with common random cohorts and untouched validation seeds."""
import math

import numpy as np

from .scenarios import prepare_cohort, scenario_id, simulate_cohort, SCENARIOS


def _integer(value,label,minimum,maximum):
    if isinstance(value,bool): raise ValueError(label+': требуется целое число.')
    try: number=float(value)
    except (TypeError,ValueError): raise ValueError(label+': требуется целое число.') from None
    if not math.isfinite(number) or number!=int(number) or not minimum<=number<=maximum:
        raise ValueError(f'{label}: допустимо {minimum}–{maximum}.')
    return int(number)


def _notify(progress,done,total,message):
    if progress: progress(done,total,message)


def _cancel(cancelled):
    if cancelled and cancelled(): raise InterruptedError('Эксперимент остановлен пользователем.')


def policy_metrics(bundle,variable_cost=None,fixed_cost=None):
    bookings=bundle['bookings'];truth=bundle['passengers'];params=bundle['params']
    gross=int(bookings.commercial.clip(lower=0).sum())
    cancellations=int(-bookings.commercial.clip(upper=0).sum())
    net=gross-cancellations;revenue=float((bookings.commercial*bookings.price).sum())
    waste=truth.loc[truth.outcome.eq('waste')]
    result={'revenue':revenue,'gross':gross,'canceled':cancellations,'net':net,
        'load_pct':net/params['capacity']*100,'rejected':int(waste.qty.sum()),
        'price_rejected':int(waste.loc[waste.reason.eq('Цена выше бюджета'),'qty'].sum()),
        'capacity_rejected':int(waste.loc[waste.reason.eq('Нет мест'),'qty'].sum()),
        'closed_rejected':int(waste.loc[waste.reason.eq('Продажи закрыты'),'qty'].sum())}
    if variable_cost is not None and fixed_cost is not None:
        result['profit']=revenue-float(variable_cost)*net-float(fixed_cost)
    return result


def _average(rows):
    keys=['revenue','gross','canceled','net','load_pct','rejected','price_rejected','capacity_rejected','closed_rejected','profit']
    return {key:float(np.mean([r[key] for r in rows])) for key in keys if rows and key in rows[0]}


def _cost(value,name):
    if value in (None,''): return None
    try: number=float(value)
    except (ValueError,TypeError): raise ValueError(name+': требуется неотрицательное число.') from None
    if not math.isfinite(number) or number<0: raise ValueError(name+': требуется неотрицательное число.')
    return number


def run_policies(payload,progress=None,cancelled=None):
    """Tune on training cohorts; compare locked policies on different cohorts."""
    settings={**payload,**payload.get('research',{})};params=payload.get('params',{})
    scenario=scenario_id(settings.get('scenario','balanced'))
    seed=_integer(settings.get('seed_start',42),'Seed',0,2**32-200001)
    training=_integer(settings.get('training_runs',5),'Обучающие запуски',1,30)
    validation=_integer(settings.get('validation_runs',10),'Проверочные запуски',2,100)
    objective=settings.get('objective','revenue')
    if objective not in ('revenue','profit'): raise ValueError('Цель должна быть revenue либо profit.')
    variable=_cost(settings.get('variable_cost'),'Расходы на пассажира')
    fixed=_cost(settings.get('fixed_cost'),'Постоянные расходы рейса')
    if objective=='profit' and (variable is None or fixed is None):
        raise ValueError('Для подбора по прибыли задайте постоянные расходы рейса и расходы на пассажира.')
    minimum=_cost(settings.get('min_load',0),'Минимальная загрузка')
    if minimum is None: minimum=0.
    if minimum>100: raise ValueError('Минимальная загрузка должна быть от 0 до 100%.')
    # The candidate family is specified before seeing validation demand.
    candidates=[{'kind':kind,'multiplier':m,'growth':g,'name':f'{kind} · цена×{m:g} · рост×{g:g}'}
        for kind in ('dynamic','gated') for m in (.85,1.,1.15) for g in (.5,1.)]
    train_seeds=list(range(seed,seed+training));test_seeds=list(range(seed+100000,seed+100000+validation))
    assert not set(train_seeds)&set(test_seeds)
    total=len(candidates)*training+4*validation;completed=0;training_rows=[]
    cohorts={s:prepare_cohort(scenario,s,params) for s in train_seeds}
    for index,policy in enumerate(candidates):
        rows=[]
        for current_seed in train_seeds:
            _cancel(cancelled);p,cohort=cohorts[current_seed]
            rows.append(policy_metrics(simulate_cohort(p,cohort,scenario,policy),variable,fixed))
            completed+=1;_notify(progress,completed,total,'Подбор тарифной политики на обучающих сценариях')
        average=_average(rows)
        training_rows.append({'candidate':index,'policy':policy['name'],**average,'eligible':average['load_pct']>=minimum})
    eligible=[r for r in training_rows if r['eligible']]
    feasible=bool(eligible)
    if feasible: winner=max(eligible,key=lambda r:r[objective])
    else: winner=max(training_rows,key=lambda r:(r['load_pct'],r[objective]))
    selected={**candidates[winner['candidate']],'name':'Подобранная политика'}
    named=[{'kind':'static','multiplier':1.,'growth':0.,'name':'Постоянная цена'},
        {'kind':'dynamic','multiplier':1.,'growth':1.,'name':'Текущая динамическая цена'},
        {'kind':'gated','multiplier':1.,'growth':1.,'name':'Открытие RBD по правилу'},selected]
    rows=[]
    for current_seed in test_seeds:
        p,cohort=prepare_cohort(scenario,current_seed,params)
        for policy in named:
            _cancel(cancelled)
            metrics=policy_metrics(simulate_cohort(p,cohort,scenario,policy),variable,fixed)
            rows.append({'seed':current_seed,'policy':policy['name'],'phase':'holdout',**metrics})
            completed+=1;_notify(progress,completed,total,'Проверка зафиксированных политик на новых seed')
    comparison=[]
    for policy in named:
        subset=[r for r in rows if r['policy']==policy['name']]
        values=[r[objective] for r in subset]
        comparison.append({'policy':policy['name'],**_average(subset),
            'objective_mean':float(np.mean(values)),'objective_std':float(np.std(values,ddof=1)),
            'validation_runs':len(subset)})
    # Pareto domination in mean net revenue and load (not claims about profit).
    for row in comparison:
        row['pareto']=not any(other['revenue']>=row['revenue'] and other['load_pct']>=row['load_pct']
            and (other['revenue']>row['revenue'] or other['load_pct']>row['load_pct']) for other in comparison)
    base={r['seed']:r for r in rows if r['policy']=='Текущая динамическая цена'}
    chosen={r['seed']:r for r in rows if r['policy']=='Подобранная политика'}
    differences=np.asarray([chosen[s][objective]-base[s][objective] for s in test_seeds])
    rng=np.random.default_rng(seed+9812)
    boot=np.mean(rng.choice(differences,size=(2000,len(differences)),replace=True),axis=1)
    low,high=np.quantile(boot,[.025,.975])
    warnings=['Подбор использует только обучающие seed. Итоговые показатели рассчитаны на новых, не использованных при подборе потоках пассажиров.',
        'Один и тот же первичный поток, бюджеты и планы возврата проигрываются под каждой политикой. Это сценарная оценка, не гарантия роста на реальном рынке.',
        'Открытые RBD используют общий остаток мест в салоне; их доступности нельзя складывать как независимые ёмкости.',
        'Ограничение минимальной загрузки проверяется по среднему на обучающих потоках; оно не гарантирует загрузку каждого рейса.']
    if not feasible: warnings.append('Ни один кандидат не выполнил минимальную среднюю загрузку. Показан кандидат с наибольшей загрузкой; допустимый оптимум не найден.')
    if variable is None or fixed is None: warnings.append('Расходы не заданы полностью: показана выручка, прибыль не рассчитывается.')
    return {'ok':True,'title':'Сравнение тарифных политик','summary':{
        'scenario':SCENARIOS[scenario]['label'],'training_runs':training,'validation_runs':validation,
        'candidate_count':len(candidates),'objective':objective,'feasible_training_policy':feasible,
        'paired_difference':float(differences.mean()),'difference_ci_low':float(low),'difference_ci_high':float(high),
        'selected_kind':selected['kind'],'selected_price_multiplier':selected['multiplier'],'selected_growth_multiplier':selected['growth']},
        'tables':{'comparison':comparison,'training_candidates':training_rows,'validation_runs':rows,
            'seed_split':[{'phase':'training','seeds':train_seeds},{'phase':'holdout','seeds':test_seeds}]},
        'charts':{'policy_revenue':[{'label':r['policy'],'value':r['revenue']} for r in comparison],
            'revenue_load':[{'label':r['policy'],'x':r['load_pct'],'y':r['revenue'],'pareto':r['pareto']} for r in comparison]},
        'warnings':warnings,'selected_policy':selected,'assumptions':{'variable_cost':variable,'fixed_cost':fixed,
            'min_load_pct':minimum,'profit_definition':'revenue - variable_cost * final_occupied_seats - fixed_cost' if variable is not None and fixed is not None else None}}
