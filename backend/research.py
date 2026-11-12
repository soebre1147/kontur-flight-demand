"""Independent calibration, conditional sensitivity and information-safe forecasts."""
import hashlib
import itertools
import json
import math

import numpy as np
import pandas as pd

from .service import _analyze,_fit_v3,_model_options,_json_safe
from .simulation import DEFAULTS,PERIODS,PROFILES,validate_params,QTY_PROBABILITIES
from .csv_io import parse_bookings,parse_offers
from .evaluation import evaluate_reference


def _tick(progress,cancelled,n,total,message):
    if cancelled and cancelled(): raise InterruptedError('Расчёт отменён.')
    if progress: progress(n,total,message)


def _values(raw,default,low,high,integer=False,maxlen=10):
    if raw is None or raw=='' or isinstance(raw,(list,tuple)) and not raw: raw=default
    if isinstance(raw,str): raw=[v.strip() for v in raw.split(',') if v.strip()]
    if not isinstance(raw,(list,tuple)) or not 1<=len(raw)<=maxlen: raise ValueError(f'В сетке требуется от 1 до {maxlen} значений.')
    result=[]
    for x in raw:
        value=float(x)
        if not math.isfinite(value) or not low<=value<=high or integer and not value.is_integer():
            raise ValueError(f'Значение сетки должно быть от {low} до {high}.')
        result.append(int(value) if integer else value)
    return list(dict.fromkeys(result))


def _settings(payload):
    params=validate_params(payload.get('params') or {})
    options=_model_options(payload.get('model') or {})
    options['method']='v3'
    settings=payload.get('research') or {}
    if not isinstance(settings,dict): raise ValueError('research должен быть объектом.')
    return params,options,settings


def _bundle(payload,seed=None,scenario=None):
    params=validate_params(payload.get('params') or {})
    if payload.get('mode')=='import':
        bookings=parse_bookings(payload.get('files') or [],params['departure'])
        flight=payload.get('selected_flight') or params['flight']
        if flight not in set(bookings.flight): flight=sorted(bookings.flight.unique())[0]
        bookings=bookings.loc[bookings.flight.eq(flight)].copy();params['flight']=flight
        offers=parse_offers(payload['offers_files']) if payload.get('offers_files') else None
        if offers is not None:
            offers=offers.loc[offers.flight.eq(flight)].copy()
            if offers.empty: raise ValueError('Для выбранного рейса отсутствуют offers.')
            start=offers.start.min();end=offers.end.max()
        else:
            start=min(pd.Timestamp(params['departure'],tz='UTC')-pd.Timedelta(days=180),bookings.timestamp.min().normalize())
            end=pd.Timestamp(params['departure'],tz='UTC')+pd.Timedelta(days=1)
        return {'params':params,'bookings':bookings,'offers':offers,'horizon_start':start,'horizon_end':end,
                'passengers':None,'scenario':'uploaded'}
    from .laboratory import generate_bundle
    research=payload.get('research') or {}
    scenario=scenario or (research.get('scenarios') or ['balanced'])[0]
    return generate_bundle(scenario,int(params['seed'] if seed is None else seed),params)


def _analyze_bundle(bundle,options,params=None):
    return _analyze(bundle['bookings'],bundle.get('offers'),params or bundle['params'],options,'import',
                    bundle['horizon_start'],bundle['horizon_end'])


def _fingerprint(bundle):
    source=bundle['bookings'].to_json(date_format='iso')+json.dumps(bundle['params'],sort_keys=True)
    return hashlib.sha256(source.encode()).hexdigest()


def run_calibration(payload,progress=None,cancelled=None):
    """Choose fixed assumptions using training truth, then evaluate unseen seeds."""
    params,options,s=_settings(payload)
    train=int(s.get('training_runs',3));validation=int(s.get('validation_runs',3));seed=int(s.get('seed_start',1000))
    if not 1<=train<=20 or not 1<=validation<=20: raise ValueError('Калибровка: по 1–20 обучающих и проверочных запусков.')
    if seed<0 or seed+100000+validation>4294967295: raise ValueError('Некорректный seed калибровки.')
    from .laboratory import generate_bundle
    scenarios=s.get('scenarios') or ['balanced','mixed_wtp','late']
    training=[generate_bundle(scenarios[i%len(scenarios)],seed+i,params) for i in range(train)]
    heldout=[generate_bundle(scenarios[i%len(scenarios)],seed+100000+i,params) for i in range(validation)]
    train_hash=[_fingerprint(b) for b in training];validation_hash=[_fingerprint(b) for b in heldout]
    if set(train_hash)&set(validation_hash): raise ValueError('Обучающая и проверочная выборки совпадают.')
    # Only independent TRAINING references create fixed behaviour assumptions.
    truth=pd.concat([b['passengers'].loc[b['passengers'].outcome.isin(['success','waste'])] for b in training],ignore_index=True)
    counts=truth.qty.value_counts().to_dict();qty_prior={str(q):float(counts.get(q,0)+.5) for q in (1,2,3,4)}
    amount=truth.amount.to_numpy(float);cabin=truth.cabin.eq('C').to_numpy()
    base=np.log(amount)-np.where(cabin,math.log(3),0)
    calibration={'training_fingerprint':hashlib.sha256(''.join(train_hash).encode()).hexdigest(),
                 'training_fingerprints':train_hash,'qty_prior':qty_prior,'mu_prior':[float(np.mean(base)),max(.2,float(np.std(base)))],
                 'provenance':'independent_generated_training_passengers_only','training_seeds':list(range(seed,seed+train))}
    sigmas=_values(s.get('sigma_values'),[.35,.45,.65],.1,1.5,maxlen=5)
    beta_times=_values(s.get('beta_time_values',s.get('beta_values')),[options.get('beta_time',0.),-.1],-2,2,maxlen=4)
    beta_quantities=_values(s.get('beta_qty_values'),[options.get('beta_qty',0.)],-2,2,maxlen=4)
    mixtures=[False,True] if s.get('compare_mixture',True) else [options.get('mixture',False)]
    candidates=[]
    for sigma,beta,beta_qty,mixed in itertools.product(sigmas,beta_times,beta_quantities,mixtures):
        residual=base-beta*np.log((1+truth.dtd.to_numpy(float))/31)-beta_qty*np.log(truth.qty.to_numpy(float))
        candidate_calibration={**calibration,'mu_prior':[float(np.mean(residual)),max(.2,float(np.std(residual)))]}
        candidates.append({**options,'sigma':sigma,'beta_time':beta,'beta_qty':beta_qty,'mixture':mixed,
                           'calibration':candidate_calibration})
    if len(candidates)>48: raise ValueError('Для калибровки используйте не более 48 сочетаний параметров.')
    total=len(candidates)*train+3*validation;step=0;rows=[]
    for index,candidate in enumerate(candidates):
        errors=[];budget_errors=[];failures=[]
        for b in training:
            _tick(progress,cancelled,step,total,'Калибровка: только обучающие сценарии');step+=1
            try:
                fitted=_analyze_bundle(b,candidate);metrics=evaluate_reference(fitted,b['passengers'])['summary']
                errors.append(metrics['hidden_period_mae']);budget_errors.append(metrics['budget_wasserstein'])
            except (ValueError,ArithmeticError) as exc: failures.append(str(exc))
        score=float(np.mean(errors)) if len(errors)==train else None
        rows.append({'candidate':index,'sigma':candidate['sigma'],'beta_time':candidate['beta_time'],
                     'beta_qty':candidate['beta_qty'],'mixture':candidate['mixture'],
                     'training_mae':score,'training_budget_wasserstein':float(np.mean(budget_errors)) if budget_errors else None,
                     'successful_runs':len(errors),'failed_runs':len(failures),'errors':' | '.join(failures[:2])})
    successful=[r for r in rows if r['training_mae'] is not None]
    if not successful: raise ValueError('Ни один кандидат не завершил все обучающие запуски; уменьшите сетку или измените предположения.')
    winner=min(successful,key=lambda r:(r['training_mae'],r['training_budget_wasserstein'] or 0,r['candidate']))
    chosen=candidates[winner['candidate']];checks=[]
    for b in heldout:
        for name,candidate in [('selected',chosen),('current',options),('baseline',{**options,'method':'baseline'})]:
            _tick(progress,cancelled,step,total,'Независимая проверка выбранной конфигурации');step+=1
            try:
                fitted=_analyze_bundle(b,candidate);m=evaluate_reference(fitted,b['passengers'])['summary']
                checks.append({'seed':b['params']['seed'],'scenario':b.get('scenario'),'method':name,**m,'status':'ok'})
            except (ValueError,ArithmeticError) as exc:
                checks.append({'seed':b['params']['seed'],'method':name,'status':'failed','error':str(exc)})
    aggregates=[]
    for name in ('selected','current','baseline'):
        matching=[r for r in checks if r['method']==name and r['status']=='ok']
        aggregates.append({'method':name,'successful_runs':len(matching),'failed_runs':validation-len(matching),
                           'holdout_mae':float(np.mean([r['hidden_period_mae'] for r in matching])) if matching else None,
                           'holdout_budget_wasserstein':float(np.mean([r['budget_wasserstein'] for r in matching if r['budget_wasserstein'] is not None])) if any(r['budget_wasserstein'] is not None for r in matching) else None})
    _tick(progress,cancelled,total,total,'Калибровка завершена')
    compared={row['method']:row for row in aggregates}
    complete=all(compared[name]['successful_runs']==validation for name in ('selected','current'))
    delta=compared['selected']['holdout_mae']-compared['current']['holdout_mae'] if complete else None
    assessment=('Выбранная конфигурация уменьшила MAE на этой проверочной выборке.' if delta is not None and delta<0
                else 'Выбранная конфигурация не уменьшила MAE текущей модели на проверочной выборке. Применение остаётся вашим выбором.' if delta is not None
                else 'Проверка неполная: сравнение средней ошибки конфигураций не завершено на всех seed.')
    return _json_safe({'ok':True,'title':'Калибровка с независимой проверкой','summary':{'training_runs':train,'validation_runs':validation,'candidates':len(candidates),
                      'selected_sigma':chosen['sigma'],'selected_beta_time':chosen['beta_time'],
                      'selected_beta_qty':chosen['beta_qty'],'selected_mixture':chosen['mixture'],
                      'holdout_mae_difference':delta,'improves_holdout_mae':delta<0 if delta is not None else None},
        'tables':{'candidates':rows,'holdout':aggregates,'holdout_runs':checks},'charts':{'holdout':aggregates},
        'selected_model':chosen,'training_fingerprints':train_hash,'validation_fingerprints':validation_hash,
        'warnings':['Калибровка использует отдельные сгенерированные сценарии, даже если в основном интерфейсе открыт импорт.',
                    'Проверочные seed не участвуют в подборе параметров. Улучшение на проверке не гарантируется.',
                    'Выбор минимизирует MAE скрытого спроса по периодам. Объём N известен извне.',assessment]})


def run_sensitivity(payload,progress=None,cancelled=None):
    params,options,s=_settings(payload);b=_bundle(payload);params=b['params']
    sigmas=_values(s.get('sigma_values'),[.3,.45,.65],.1,1.5)
    strengths=_values(s.get('prior_values'),[.1,.5,2],0,20)
    demands=_values(s.get('demand_values'),[params['demand']],1,1500,integer=True)
    grid=list(itertools.product(sigmas,strengths,demands))
    if len(grid)>60: raise ValueError('Для устойчивости используйте не более 60 сочетаний параметров.')
    bootstrap=int(s.get('bootstrap_runs',0))
    if not 0<=bootstrap<=30: raise ValueError('Бутстрап: от 0 до 30 повторных оценок.')
    rows=[];periods=[];total=len(grid)+bootstrap
    for i,(sigma,strength,n) in enumerate(grid):
        _tick(progress,cancelled,i,total,'Проверка чувствительности с повторной оценкой')
        candidate={**options,'sigma':sigma,'prior_strength':strength};p={**params,'demand':n}
        try:
            result=_analyze_bundle(b,candidate,p)
            mu=result['diagnostics']['parameters'][0]
            rows.append({'sigma':sigma,'prior_strength':strength,'N':n,'status':'ok','median_budget':mu['median_amount']})
            periods.extend({'sigma':sigma,'prior_strength':strength,'N':n,**r} for r in result['periods'])
        except (ValueError,ArithmeticError) as exc: rows.append({'sigma':sigma,'prior_strength':strength,'N':n,'status':'failed','error':str(exc)})
    rng=np.random.default_rng(int(s.get('seed_start',params['seed'])));boot=[]
    purchase=b['bookings'].loc[b['bookings'].commercial.gt(0)];cancel=b['bookings'].loc[b['bookings'].commercial.lt(0)]
    for i in range(bootstrap):
        _tick(progress,cancelled,len(grid)+i,total,'Бутстрап покупок: повторная оценка параметров')
        if purchase.empty: boot.append({'replicate':i,'status':'failed','error':'Нет покупок для повторной выборки.'});continue
        sampled=purchase.iloc[rng.integers(0,len(purchase),len(purchase))].copy()
        frame=pd.concat([sampled,cancel],ignore_index=True).sort_values('timestamp').reset_index(drop=True)
        try:
            fitted=_analyze_bundle({**b,'bookings':frame},options)
            parameter=fitted['diagnostics']['parameters'][0]
            boot.append({'replicate':i,'status':'ok','median_budget':parameter['median_amount'],
                         **{r['period']:r['estimated'] for r in fitted['periods']}})
        except (ValueError,ArithmeticError) as exc: boot.append({'replicate':i,'status':'failed','error':str(exc)})
    envelopes=[]
    for period in PERIODS:
        group=[r for r in periods if r['period']==period]
        if group: envelopes.append({'period':period,'min_estimated':min(r['estimated'] for r in group),
            'max_estimated':max(r['estimated'] for r in group),'min_conditional_lower':min(r['lower'] for r in group),
            'max_conditional_upper':max(r['upper'] for r in group)})
    _tick(progress,cancelled,total,total,'Проверка устойчивости завершена')
    return _json_safe({'ok':True,'title':'Чувствительность и повторные оценки','summary':{'configurations':len(grid),
        'successful':sum(r['status']=='ok' for r in rows),'bootstrap_successful':sum(r['status']=='ok' for r in boot)},
        'tables':{'configurations':rows,'sensitivity_envelope':envelopes,'bootstrap':boot},
        'charts':{'envelope':envelopes},'warnings':[
            'Диапазон чувствительности объединяет разные внешние предположения; это не доверительный и не байесовский интервал.',
            'Бутстрап повторно выбирает наблюдаемые покупки и заново оценивает модель при фиксированных offers и N. Это условная устойчивость к составу покупок, а не полная неопределённость поведения.',
            'Неудавшиеся оценки показываются отдельно и не заменяются базовым методом.']})


def _profile_past_fraction(params,start,cut,departure):
    """User supplied temporal profile, never the withheld reference cohort."""
    ranges=[(91,180),(31,90),(15,30),(8,14),(0,7)]
    weights=np.asarray(PROFILES[params['profile']],float);weights/=weights.sum()
    fraction=0.
    for weight,(low,high) in zip(weights,ranges):
        left=departure-pd.Timedelta(days=high);right=departure-pd.Timedelta(days=low)+pd.Timedelta(days=1)
        length=(right-left).total_seconds()
        overlap=max(0.,(min(cut,right)-max(start,left)).total_seconds())
        fraction+=weight*min(1.,overlap/length)
    return min(1.,max(0.,fraction))


def _prefix_bundle(bundle,cutoff_dtd):
    """Build the complete fitting boundary with no future operation or offer."""
    params=bundle['params'];departure=pd.Timestamp(params['departure'],tz='UTC')
    cut=departure-pd.Timedelta(days=cutoff_dtd)+pd.Timedelta(days=1)
    start=bundle['horizon_start']
    if cut<=start: raise ValueError('Точка отсечения предшествует началу истории.')
    past=bundle['bookings'].loc[bundle['bookings'].timestamp.lt(cut)].copy()
    offers=bundle.get('offers')
    if offers is not None:
        offers=offers.loc[offers.start.lt(cut)&offers.end.gt(start)].copy()
        offers['end']=offers.end.clip(upper=cut)
        offers['start']=offers.start.clip(lower=start)
        if offers.empty: raise ValueError('До точки отсечения отсутствуют offers.')
    elif past.commercial.gt(0).sum()==0:
        raise ValueError('До точки отсечения нет покупок; нужна история предложений до выбранного дня.')
    fraction=_profile_past_fraction(params,start,cut,departure)
    gross=int(past.commercial.clip(lower=0).sum())
    history_n=max(gross,int(round(params['demand']*fraction)),1)
    if history_n>1500: raise ValueError('Сценарный спрос истории превышает интерактивный предел v3.')
    return {'params':{**params,'demand':history_n},'bookings':past,'offers':offers,
            'horizon_start':start,'horizon_end':cut},cut,fraction


def _forecast_one(bundle,options,settings,cutoff_dtd,progress,cancelled,index,total):
    prefix,cut,fraction=_prefix_bundle(bundle,cutoff_dtd)
    _tick(progress,cancelled,index,total,'Оценка только по данным до точки отсечения')
    observed=_analyze_bundle(prefix,options)
    params=bundle['params'];departure=pd.Timestamp(params['departure'],tz='UTC')
    future_end=departure+pd.Timedelta(days=1)
    history=prefix['bookings'];pastoffers=prefix['offers']
    # Both this price and inventory snapshot use ONLY the prefix.
    state={};balances=history.groupby('cabin').commercial.sum().to_dict()
    capacities={'C':params['businessCapacity'],'Y':params['capacity']-params['businessCapacity']}
    if pastoffers is not None:
        for cabin,group in pastoffers.groupby('cabin'):
            last=group.sort_values('start').iloc[-1]
            state[cabin]={'price':float(last.price) if pd.notna(last.price) else None,
                          'open':bool(last.open),'available':int(last.available_seats)}
    else:
        for cabin,group in history.loc[history.commercial.gt(0)].groupby('cabin'):
            state[cabin]={'price':float(group.sort_values('timestamp').iloc[-1].price),'open':True,
                          'available':max(0,capacities.get(cabin,0)-int(balances.get(cabin,0)))}
    if not state: raise ValueError('Нет известного предложения до точки отсечения.')
    # The last offer can include the last event's pre-event stock. Correct it
    # using only operations strictly later than its start (state at t is pre t).
    if pastoffers is not None:
        for cabin,group in pastoffers.groupby('cabin'):
            last=group.sort_values('start').iloc[-1]
            operations=history.loc[history.cabin.eq(cabin)&history.timestamp.ge(last.start)]
            state[cabin]['available']=max(0,state[cabin]['available']-int(operations.commercial.sum()))
    parameters=observed['diagnostics']['parameters'][0]
    mu=float(parameters['mu_log']);rates=observed['diagnostics']['arrival_rates']
    rate_by_cabin={}
    for cabin in state:
        rows=[r for r in rates if r['cabin']==cabin]
        recent=max(rows,key=lambda r:PERIODS.index(r['period'])) if rows else None
        rate_by_cabin[cabin]=float(recent['lambda_groups_per_day']) if recent else 0.
    # Future counts are extrapolated Poisson rates. N was used as an external
    # historical scenario anchor, not read from future passengers.
    repeats=int(settings.get('forecast_runs',200))
    if not 20<=repeats<=1000: raise ValueError('Для прогноза задайте 20–1000 имитаций.')
    rng=np.random.default_rng(int(settings.get('seed_start',params['seed']))+int(cutoff_dtd)*1009)
    days=list(pd.date_range(cut,future_end-pd.Timedelta(1,'ns'),freq='D'))
    if not days: raise ValueError('После точки отсечения не осталось дней.')
    if repeats*len(days)*sum(rate_by_cabin.values())>3_000_000:
        raise ValueError('Прогноз требует более 3 млн имитаций групп. Уменьшите число повторов или выберите более позднюю точку отсечения.')
    base_net=int(history.commercial.sum());paths=[];purchase_totals=[];cancel_totals=[]
    independent=(options.get('calibration') or {}).get('qty_prior') or QTY_PROBABILITIES
    qs=np.array([int(q) for q in independent]);probs=np.array(list(independent.values()),float);probs/=probs.sum()
    beta_time=options.get('beta_time',0.);beta_qty=options.get('beta_qty',0.)
    p_cancel=params['cancel']/100
    for replicate in range(repeats):
        if replicate%25==0: _tick(None,cancelled,0,1,'Прогноз')
        net_by_cabin={c:max(0,int(balances.get(c,0))) for c in state}
        available={c:int(v['available']) for c,v in state.items()}
        bought=0;canceled=0;path=[]
        for day in days:
            dtd=int((departure-day).days)
            for cabin,offer in state.items():
                # Cancellation hazard is an explicit forward assumption. It
                # never links an observed cancellation to a guessed identity.
                probability=min(1.,p_cancel/len(days))
                returned=int(rng.binomial(net_by_cabin[cabin],probability))
                net_by_cabin[cabin]-=returned;available[cabin]+=returned;canceled+=returned
                count=int(rng.poisson(rate_by_cabin[cabin]))
                if not offer['open'] or offer['price'] is None or not count: continue
                quantities=rng.choice(qs,size=count,p=probs)
                loc=mu+beta_time*np.log((1+dtd)/31)+(math.log(3) if cabin=='C' else 0)+beta_qty*np.log(quantities)
                if options.get('mixture'): loc+=rng.choice([-.25,.5],size=count,p=[.65,.35])
                budgets=np.exp(np.clip(loc+options['sigma']*rng.normal(size=count),-30,30))
                for q,amount in zip(quantities,budgets):
                    if q<=available[cabin] and amount>=offer['price']:
                        available[cabin]-=int(q);net_by_cabin[cabin]+=int(q);bought+=int(q)
            path.append(base_net+bought-canceled)
        paths.append(path);purchase_totals.append(bought);cancel_totals.append(canceled)
    paths=np.asarray(paths);rows=[]
    heldout=bundle['bookings'].loc[bundle['bookings'].timestamp.ge(cut)].copy()
    actual_final=base_net+int(heldout.commercial.sum())
    # Evaluation branch starts here: held-out rows were not passed above.
    for j,day in enumerate(days):
        actual=base_net+int(heldout.loc[heldout.timestamp.lt(day+pd.Timedelta(days=1)),'commercial'].sum())
        rows.append({'date':day.date().isoformat(),'dtd':int((departure-day).days),'predicted_net':float(paths[:,j].mean()),
                     'lower':float(np.quantile(paths[:,j],.1)),'upper':float(np.quantile(paths[:,j],.9)),'actual_net':actual})
    final=rows[-1]
    return {'cutoff_dtd':int(cutoff_dtd),'cutoff':cut.isoformat(),'historical_N_assumption':prefix['params']['demand'],
            'historical_profile_fraction':fraction,'observed_purchase_seats':int(history.commercial.clip(lower=0).sum()),
            'observed_net':base_net,'predicted_net':final['predicted_net'],'actual_net':actual_final,
            'absolute_error':abs(final['predicted_net']-actual_final),'lower':final['lower'],'upper':final['upper'],
            'future_purchase_mean':float(np.mean(purchase_totals)),'future_cancellation_mean':float(np.mean(cancel_totals)),
            'fit_max_timestamp':history.timestamp.max().isoformat() if len(history) else None,
            'fit_max_offer_end':pastoffers.end.max().isoformat() if pastoffers is not None else None,
            'future_offer_assumption':'last_observed_prefix_state_constant_price_and_open_status',
            'rates':rate_by_cabin,'curve':rows,'historical_fit':observed['diagnostics']}


def run_forecast(payload,progress=None,cancelled=None):
    params,options,s=_settings(payload);bundle=_bundle(payload)
    cutoffs=_values(s.get('cutoffs'),[s.get('cutoff_dtd',30)],1,180,integer=True,maxlen=4)
    results=[]
    for index,cutoff in enumerate(cutoffs):
        results.append(_forecast_one(bundle,options,s,cutoff,progress,cancelled,index,len(cutoffs)))
    _tick(progress,cancelled,len(cutoffs),len(cutoffs),'Прогноз и проверка завершены')
    rows=[{k:v for k,v in r.items() if k not in ('curve','historical_fit','rates')} for r in results]
    warnings=[
            'Модель оценивается заново только на bookings и offers до конца выбранного дня. Будущие данные используются исключительно после прогноза для проверки.',
            'N для прошедшего горизонта неизвестно: оно сценарно задаётся из внешнего полного N и выбранного пользователем профиля обращений, минимум — уже купленные места. Это предположение, а не восстановленный факт.',
            'Будущие интенсивности продолжают последнюю оценённую интенсивность салона; цена и открытость берутся из последнего известного предложения. Фактические будущие offers не используются.',
            'Прогноз использует отмены отдельных мест с биномиальной интенсивностью; это отдельное допущение, отличное от полных отмен групп в исходном генераторе. Закрытый на момент отсечения салон остаётся закрытым по принятой будущей политике даже после освобождения мест.',
            'Интервал 10–90% отражает имитацию будущих обращений и отмен при фиксированных оценках. Неопределённость параметров и будущей политики не включена.',
            'Если загруженный журнал ещё не достиг вылета, показанный фактический итог отражает только имеющиеся операции; это не завершённая внешняя проверка.']
    if bundle.get('offers') is None:
        warnings.append('История offers отсутствует: прошлые и будущие цены приближены покупками, открытость предполагается, остатки вычислены из заданной вместимости и прошлого баланса. Кабина без покупок до отсечения не имеет определимого предложения и в прогноз не включается.')
    return _json_safe({'ok':True,'title':'Прогноз на выбранный день до вылета',
        'summary':{'cutoffs':len(results),'mean_final_absolute_error':float(np.mean([r['absolute_error'] for r in results]))},
        'tables':{'forecasts':rows,'daily':[{'cutoff_dtd':r['cutoff_dtd'],**day} for r in results for day in r['curve']]},
        'charts':{'forecast':results[0]['curve']},'details':results,'warnings':warnings})
