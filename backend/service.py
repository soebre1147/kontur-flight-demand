"""Pure Python facade for simulation, CSV validation and mathematical inference.

Truth is used only after fitting, in the display/scoring branch. The fitting
boundary accepts observed bookings, offer intervals and an external seat total.
"""
import math
from collections.abc import Mapping

import numpy as np
import pandas as pd

from .csv_io import parse_bookings, parse_offers, validate_files
from .simulation import DEFAULTS, PERIODS, QTY_PROBABILITIES, RBD_DESCENDING, period_of, simulate, validate_params

MODEL_DEFAULTS = {'method':'v3','total_mode':'fixed_design','sigma':.45,'prior_strength':.5,
                  'beta_time':0.,'beta_qty':0.,'mixture':False}
MAX_V3_DEMAND = 1500


def _model_options(values):
    if values is None: values={}
    if not isinstance(values,dict): raise ValueError('Настройки модели должны быть объектом.')
    result={**MODEL_DEFAULTS,**values}
    if result['method'] not in ('v3','baseline'): raise ValueError('Неизвестный метод восстановления.')
    if result['total_mode'] not in ('fixed_design','observed_total'): raise ValueError('Неизвестная постановка общего спроса.')
    for key,low,high in [('sigma',.1,1.5),('prior_strength',0,20),('beta_time',-2,2),('beta_qty',-2,2)]:
        try: value=float(result[key])
        except (TypeError,ValueError): value=float('nan')
        if isinstance(result[key],bool) or not math.isfinite(value) or not low<=value<=high:
            raise ValueError(f'{key}: требуется число от {low} до {high}.')
        result[key]=value
    if not isinstance(result['mixture'],bool): raise ValueError('mixture должен быть true/false.')
    calibration=result.get('calibration')
    if calibration is not None:
        if not isinstance(calibration,dict) or not calibration.get('training_fingerprint'):
            raise ValueError('Калибровка должна содержать происхождение независимой обучающей выборки.')
        for q,v in calibration.get('qty_prior',{}).items():
            if not str(q).isdigit() or not 1<=int(q)<=30 or not math.isfinite(float(v)) or float(v)<=0:
                raise ValueError('Некорректное распределение размеров групп калибровки.')
        mu=calibration.get('mu_prior')
        if mu is not None and (not isinstance(mu,(list,tuple)) or len(mu)!=2 or not all(math.isfinite(float(x)) for x in mu) or not 0<float(mu[1])<=5):
            raise ValueError('Некорректное предположение бюджета калибровки.')
    return result


def _json_safe(value):
    if isinstance(value,Mapping): return {str(k):_json_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [_json_safe(v) for v in value]
    if isinstance(value,pd.Timestamp): return value.isoformat().replace('+00:00','Z')
    if isinstance(value,(np.integer,)): return int(value)
    if isinstance(value,(np.bool_,)): return bool(value)
    if isinstance(value,(float,np.floating)): return float(value) if math.isfinite(value) else None
    if value is None or value is pd.NA or value is pd.NaT: return None
    return value


def _records(frame):
    return _json_safe(frame.to_dict('records'))


def _observed_events(bookings,departure):
    rows=bookings.copy()
    rows['qty']=rows.commercial.abs().astype(int)
    rows['signed_qty']=rows.commercial.astype(int)
    rows['outcome']=np.where(rows.commercial.gt(0),'success','cancel')
    rows['dtd']=(pd.Timestamp(departure,tz='UTC')-rows.timestamp.dt.normalize()).dt.days
    rows['period']=rows.dtd.map(period_of)
    rows['id']=[f'OBS-{i:06d}' for i in range(len(rows))]
    rows['amount']=None
    rows['reason']=np.where(rows.commercial.gt(0),'Наблюдаемая покупка','Наблюдаемая отмена; связь с покупкой неизвестна')
    rows['reconstruction_status']='наблюдаемая операция'
    rows['record_level']='отдельная операция'
    rows['id_is_synthetic']=True
    rows['row_source']='observed_booking_operation'
    rows['amount_source']='not_observed'
    rows['reason_is_modelled']=False
    rows['cancel_link_id']=None
    return rows


def _allocation(total,weights):
    weights=np.asarray(weights,float)
    if total==0: return np.zeros(len(weights),int)
    if weights.sum()==0: weights=np.ones(len(weights),float)
    exact=total*weights/weights.sum();rounded=np.floor(exact).astype(int)
    rounded[np.argsort(-(exact-rounded),kind='stable')[:total-int(rounded.sum())]]+=1
    return rounded


def _baseline(observed,hidden,flight):
    purchases=observed.loc[observed.outcome.eq('success')]
    bought=purchases.groupby('period').qty.sum().reindex(PERIODS,fill_value=0)
    allocation=_allocation(hidden,bought.to_numpy())
    rows=observed.to_dict('records')
    for period,count in zip(PERIODS,allocation):
        if count:
            rows.append({'timestamp':None,'id':f'BASE-{period}','flight':flight,
                'cabin':None,'rbd':None,'qty':int(count),'price':None,'amount':None,'outcome':'waste',
                'signed_qty':0,'dtd':None,'period':period,'source_row':None,'source_file':None,
                'reason':'Распределено пропорционально покупкам' if bought.sum() else 'Равномерное предположение без покупок',
                'reason_is_modelled':True,'reconstruction_status':'оценка: агрегат периода',
                'record_level':'агрегат периода','row_source':'baseline_period_aggregate','id_is_synthetic':True,
                'amount_source':'not_estimated','cancel_link_id':None})
    return dict(zip(PERIODS,map(int,allocation))),rows


def _fit_v3(bookings,offers,params,options,start,end):
    """Inference boundary deliberately has no passenger truth argument."""
    from model.math_reconstruction_v3 import prepare_inputs_v3, fit_model_v3, ArrivalPrior, sample_passengers_v3, BehaviorConfigV3
    from model.conditional_metrics_v3 import conditional_seat_pmf

    if params['demand']>MAX_V3_DEMAND:
        raise ValueError(f'Для интерактивной точной модели v3 N не должно превышать {MAX_V3_DEMAND} мест на рейс. Уменьшите N или выберите базовый метод.')
    inputs=prepare_inputs_v3(bookings,params['departure'],start,end,offers=offers,path_method='last')
    purchases=inputs.purchases;empty=purchases.empty
    calibration=options.get('calibration') or {}
    independent_qty={int(k):float(v) for k,v in calibration.get('qty_prior',{}).items()} or None
    quantities=tuple(sorted(set([1,2,3,4]+purchases.qty.astype(int).tolist()+list(independent_qty or {}))))
    if len(inputs.price_segments)*len(quantities)>150000:
        raise ValueError('Слишком много интервалов × размеров группы для интерактивной v3. Укрупните историю offers либо выберите базовый метод.')
    # The data alone do not reveal group sizes among refusals. In nonempty
    # journals extrapolate the observed buyer distribution with smoothing.
    config=BehaviorConfigV3(sigma=options['sigma'],qty_values=quantities,
        qty_smoothing=2.,qty_prior=independent_qty or (QTY_PROBABILITIES if empty else None),
        beta_time=options.get('beta_time',0.),beta_qty=options.get('beta_qty',0.),
        mixture_weights=(.65,.35) if options.get('mixture') else (1.,),
        mixture_offsets=(-.25,.5) if options.get('mixture') else (0.,),
        cabin_offsets={'C':math.log(3),'Y':0.})
    cabin_counts=purchases.cabin.value_counts().to_dict()
    cabins=list(inputs.strata.cabin.unique());mean_qty=float(purchases.qty.mean()) if not empty else sum(q*p for q,p in QTY_PROBABILITIES.items())
    exposure=(end-start).total_seconds()/86400
    weights={c:cabin_counts.get(c,0)+1 for c in cabins};normalizer=sum(weights.values())
    means={c:params['demand']/mean_qty/exposure*weights[c]/normalizer for c in cabins}
    prior=ArrivalPrior(mean_rates=means,strength_days=options['prior_strength'])
    # This is an explicit user-side assumption for a wholly uninformative
    # no-purchase journal, not the simulation's latent cohort statistics.
    mu_prior=calibration.get('mu_prior') or ((math.log(params['budget']),2.) if empty else None)
    fitted=fit_model_v3(inputs,primary_demand_seats=params['demand'],config=config,
        prior=prior,mu_prior=mu_prior,total_mode=options['total_mode'],maxiter=250)
    if not fitted['parameters'].optimizer_success.all():
        raise ValueError('Оптимизатор v3 не сошёлся. Измените предположения или явно выберите базовый метод; приблизительный результат не подставлен.')
    generated=sample_passengers_v3(inputs,fitted,seed=params['seed']+100003)
    labels={'budget':'Бюджет ниже цены (оценка)','availability':'Недостаточная доступность (оценка)',
            'other':'Другие причины (оценка)','observed_purchase':'Наблюдаемая покупка; бюджет смоделирован',
            'observed_cancel_unknown_link':'Наблюдаемая отмена; связь с покупкой неизвестна'}
    generated['reason_code']=generated.reason
    generated['reason']=generated.reason.map(labels).fillna(generated.reason)
    generated['reconstruction_status']=generated.outcome.map({'waste':'оценка: синтетическая скрытая группа',
        'success':'наблюдаемая покупка; бюджет оценён','cancel':'наблюдаемая отмена; бюджет неизвестен'})
    generated['record_level']='отдельная операция'
    strata=fitted['strata'];estimates=strata.groupby('period').estimated_waste_seats.sum().reindex(PERIODS,fill_value=0).astype(int).to_dict()
    hidden=params['demand']-int(purchases.qty.sum())
    marks=fitted['marks'];totals=marks.groupby('qty').expected_hidden_groups.sum().to_dict()
    intervals={}
    for period in PERIODS:
        region=marks.loc[marks.period.eq(period)].groupby('qty').expected_hidden_groups.sum().to_dict()
        complement={q:max(0.,v-region.get(q,0.)) for q,v in totals.items()}
        pmf=conditional_seat_pmf(region,complement,hidden);cdf=np.cumsum(pmf)
        intervals[period]=(int(np.searchsorted(cdf,.1)),int(np.searchsorted(cdf,.9)))
    reasons={r:float(strata[f'hidden_seats_{r}'].sum()) for r in ('budget','availability','other')}
    warnings=[
        'N — заданное извне число первичных мест до отмен; объём спроса из bookings не оценивается.',
        'Бюджеты и скрытые группы смоделированы; личности пассажиров и связи отмен с покупками не восстановлены.',
        'Интервалы 10–90% учитывают случайность скрытых групп при фиксированных оценённых параметрах; неопределённость самих параметров в них не включена.',
        'Распределение размера скрытых групп экстраполировано из покупок со сглаживанием; cabin C имеет заданный сдвиг бюджета ×3.']
    if offers is None:
        warnings.append('История предложений отсутствует: цена предполагается равной последней наблюдаемой цене покупки, до первой покупки — первой цене; доступность мест неизвестна и считается неограниченной. Закрытые классы по такому журналу не определяются.')
    if empty:
        warnings[-1]='Нет покупок: размеры групп заданы внешним распределением 1/2/3/4; бюджет — внешним логнормальным предположением из поля «Бюджет», а не оценкой из журнала.'
        warnings.append('При отсутствии покупок результат определяется N, предложениями и внешними предположениями; он не идентифицирует поведение пассажиров.')
    warnings.append('Сглаживание интенсивностей использует равномерный профиль по времени и доли кабин из этого же журнала с псевдосчётом +1; это предположение приложения, а не обучение на независимой истории.')
    if calibration:
        warnings.append('Распределение групп и предположение бюджета взяты из отдельной калибровочной выборки; данные текущих passengers в оценку не передаются.')
    if options['prior_strength']==0:
        warnings.append('Априорное сглаживание интенсивностей отключено; ячейки без продаж могут быть неидентифицируемыми.')
    if fitted['parameters'].parameter_bound_hits.sum(): warnings.append('Некоторые параметры достигли границ оптимизации: интерпретируйте результат с осторожностью.')
    warnings.append('fixed_design: N фиксировано планом генерации; observed_total: N наблюдено как случайный общий итог. Эти постановки используют разные функции правдоподобия.')
    diagnostics={'method':'v3','title':'Совместная математическая модель v3 · точное число мест',
        'status':'prior_only_no_sales' if empty else 'converged','parameters':_records(fitted['parameters']),
        'config':{k:options.get(k) for k in ('sigma','beta_time','beta_qty','mixture','prior_strength')},
        'arrival_rates':_records(strata[['flight','cabin','period','lambda_groups_per_day','exposure_days','purchase_events','identification']]),
        'modelled_reason_totals':reasons,'warnings':warnings,'assumptions':{**_json_safe(fitted['assumptions']),
            'arrival_prior_provenance':'application_flat_time_prior_using_same_journal_buyer_cabin_share_plus_one_not_independent_training'},
        'WAPE_pct':None,'MAE_seats':None,'wape_pct':None,'mae_seats':None}
    return estimates,intervals,_records(generated),diagnostics,_records(inputs.price_segments)


def _curve(observed,departure,start,end):
    if len(observed):
        daily=observed.assign(day=observed.timestamp.dt.normalize()).groupby('day').agg(
            purchases_delta=('signed_qty',lambda s:int(s.clip(lower=0).sum())),
            cancellations_delta=('signed_qty',lambda s:int(-s.clip(upper=0).sum())))
    else: daily=pd.DataFrame(columns=['purchases_delta','cancellations_delta'])
    rows=[];gross=0;canceled=0
    for day in pd.date_range(start.normalize(),end-pd.Timedelta(1,'ns'),freq='D'):
        purchase=int(daily.loc[day,'purchases_delta']) if day in daily.index else 0
        cancel=int(daily.loc[day,'cancellations_delta']) if day in daily.index else 0
        gross+=purchase;canceled+=cancel
        rows.append({'date':day.date().isoformat(),'dtd':int((pd.Timestamp(departure,tz='UTC')-day).days),
            'gross':gross,'net':gross-canceled,'canceled':canceled,
            'purchases_delta':purchase,'cancellations_delta':cancel,'net_delta':purchase-cancel})
    return rows


def _analyze(bookings,offers,params,options,mode,start,end,truth=None):
    observed=_observed_events(bookings,params['departure'])
    gross=int(observed.loc[observed.outcome.eq('success'),'qty'].sum())
    canceled=int(observed.loc[observed.outcome.eq('cancel'),'qty'].sum())
    hidden=params['demand']-gross
    if hidden<0: raise ValueError(f"{params['flight']}: заданное N={params['demand']} меньше {gross} купленных мест до отмен. Увеличьте первичный спрос.")
    base,baseline_rows=_baseline(observed,hidden,params['flight'])
    if options['method']=='v3':
        estimates,intervals,reconstructed,diagnostics,offer_rows=_fit_v3(bookings,offers,params,options,start,end)
    else:
        estimates=base;intervals={p:(None,None) for p in PERIODS};reconstructed=baseline_rows
        diagnostics={'method':'baseline','title':'Базовое распределение скрытого спроса по покупкам','status':'baseline',
            'parameters':[],'modelled_reason_totals':None,'WAPE_pct':None,'MAE_seats':None,
            'warnings':['N задано извне. Скрытые места распределены по периодам пропорционально покупкам; при отсутствии покупок — поровну. Это агрегаты, а не восстановленные пассажиры. Бюджеты, причины отказов и интервалы неопределённости не оцениваются.']}
        if offers is not None:
            # Uploaded offer histories must be coherent even when this chosen
            # estimator does not use them; do not export invalid actual states.
            from model.math_reconstruction_v3 import prepare_inputs_v3
            prepare_inputs_v3(bookings,params['departure'],start,end,offers=offers)
        offer_rows=_records(offers) if offers is not None else []
    actual=truth.loc[truth.outcome.eq('waste')].groupby('period').qty.sum().to_dict() if truth is not None else {}
    purchases=observed.loc[observed.outcome.eq('success')].groupby('period').qty.sum().to_dict()
    cancels=observed.loc[observed.outcome.eq('cancel')].groupby('period').qty.sum().to_dict()
    periods=[{'period':p,'purchased':int(purchases.get(p,0)),'canceled':int(cancels.get(p,0)),
        'estimated':int(estimates[p]),'actual':int(actual.get(p,0)) if truth is not None else None,
        'lower':intervals[p][0],'upper':intervals[p][1],'baseline':base[p]} for p in PERIODS]
    if truth is not None:
        error=np.abs([estimates[p]-actual.get(p,0) for p in PERIODS])
        mae=float(error.mean());wape=float(error.sum()/hidden*100) if hidden else None
        diagnostics.update({'WAPE_pct':wape,'MAE_seats':mae,'wape_pct':wape,'mae_seats':mae,
            'baseline_WAPE_pct':float(sum(abs(base[p]-actual.get(p,0)) for p in PERIODS)/hidden*100) if hidden else None,
            'evaluation_scope':'one_synthetic_realization_period_hidden_seats_not_external_validation'})
    fares=[]
    for (cabin,rbd,price),group in observed.groupby(['cabin','rbd','price'],sort=False):
        fares.append({'cabin':cabin,'rbd':rbd,'price':float(price),
            'purchased':int(group.loc[group.outcome.eq('success'),'qty'].sum()),
            'canceled':int(group.loc[group.outcome.eq('cancel'),'qty'].sum())})
    fares.sort(key=lambda r:(r['cabin'],RBD_DESCENDING.get(r['cabin'],'').find(r['rbd']) if r['rbd'] in RBD_DESCENDING.get(r['cabin'],'') else 99,-r['price']))
    balances=observed.signed_qty.cumsum() if len(observed) else pd.Series([0])
    revenue=float((observed.signed_qty*observed.price).sum())
    truth_price=int(truth.loc[truth.reason.eq('Цена выше бюджета'),'qty'].sum()) if truth is not None else None
    truth_capacity=int(truth.loc[truth.reason.eq('Нет мест'),'qty'].sum()) if truth is not None else None
    summary={'gross':gross,'canceled':canceled,'net':gross-canceled,'waste':hidden,'revenue':revenue,
        'purchaseEvents':int(observed.outcome.eq('success').sum()),'cancelEvents':int(observed.outcome.eq('cancel').sum()),
        'conversion':gross/params['demand']*100,'load':(gross-canceled)/params['capacity']*100,
        'maxBalance':int(max(0,balances.max())),'minBalance':int(min(0,balances.min())),
        'priceWaste':truth_price,'capacityWaste':truth_capacity}
    return {'params':params,'model':options,'mode':mode,'events':_records(truth) if truth is not None else _records(observed),
        'observed':_records(observed),'reconstructed':reconstructed,'periods':periods,
        'curve':_curve(observed,params['departure'],start,end),'fares':fares,'offers':offer_rows,
        'summary':summary,'diagnostics':diagnostics}


def calculate(payload):
    """Validate a browser request and return JSON-safe results without side effects."""
    try:
        if not isinstance(payload,dict): raise ValueError('Запрос должен быть объектом.')
        mode=payload.get('mode','simulation')
        if mode not in ('simulation','import'): raise ValueError('Неизвестный режим данных.')
        params=validate_params(payload.get('params',{}));options=_model_options(payload.get('model'))
        if options['method']=='v3' and params['demand']>MAX_V3_DEMAND:
            raise ValueError(f'Для точной интерактивной v3 требуется N ≤ {MAX_V3_DEMAND}; для больших сценариев выберите базовый метод.')
        results={}
        if mode=='simulation':
            run=simulate(params)
            results[params['flight']]=_analyze(run['bookings'],run['offers'],params,options,mode,
                run['horizon_start'],run['horizon_end'],truth=run['events'])
            file_summary='Синтетический сценарий Python · первичный спрос '+str(params['demand'])+' мест'
        else:
            files=payload.get('files') or [];offer_files=payload.get('offers_files') or []
            if not isinstance(files,list) or not isinstance(offer_files,list):
                raise ValueError('Списки bookings и offers должны быть массивами файлов.')
            validate_files(files+offer_files)
            bookings=parse_bookings(files,params['departure']);offers=parse_offers(offer_files) if offer_files else None
            flights=sorted(bookings.flight.unique())
            if len(flights)>20: raise ValueError('Для одного интерактивного расчёта допускается не более 20 рейсов.')
            if offers is not None:
                additional=set(offers.flight)-set(flights)
                if additional: raise ValueError('В offers есть рейсы без загруженного bookings: '+', '.join(sorted(additional)))
            departure=pd.Timestamp(params['departure'],tz='UTC')
            for flight in flights:
                frame=bookings.loc[bookings.flight.eq(flight)].copy()
                supplied=offers.loc[offers.flight.eq(flight)].copy() if offers is not None else None
                if supplied is not None:
                    if supplied.empty: raise ValueError(f'{flight}: нет интервалов offers для этого рейса.')
                    start=supplied.start.min();end=supplied.end.max()
                    if end>departure+pd.Timedelta(days=1) or start<departure-pd.Timedelta(days=3650) or not start<end:
                        raise ValueError(f'{flight}: проверьте горизонт offers (не более 3650 дней до вылета).')
                else:
                    if frame.commercial.le(0).all() and options['method']=='v3':
                        raise ValueError(f'{flight}: без покупок и реальной истории offers модель v3 не определена. Загрузите offers либо явно выберите базовый метод.')
                    start=min(departure-pd.Timedelta(days=180),frame.timestamp.min().normalize());end=departure+pd.Timedelta(days=1)
                results[flight]=_analyze(frame,supplied,{**params,'flight':flight},options,mode,start,end)
            file_summary=' · '.join(str(f['name']).split('/')[-1].split('\\')[-1] for f in files)+f' · {len(bookings)} операций · {len(flights)} рейс(а)'
        selected=payload.get('selected_flight')
        if selected not in results: selected=next(iter(results))
        return _json_safe({'ok':True,'results':results,'selected_flight':selected,'file_summary':file_summary,
            'params':params,'model':options,'mode':mode})
    except (ValueError,TypeError,KeyError,OverflowError,ArithmeticError) as error:
        return {'ok':False,'error':str(error)}
