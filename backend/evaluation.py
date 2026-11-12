"""Reference files are read only after inference, never by the inverse model."""
import hashlib
import math
import re
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from .csv_io import validate_files,parse_table,_number,_time
from .simulation import PERIODS,period_of


def parse_passengers(files,departure):
    validate_files(files)
    if not files: raise ValueError('Выберите эталонные passengers CSV.')
    rows=[];seen=set();day=pd.Timestamp(departure,tz='UTC')
    for file in files:
        name,records=parse_table(file,{'timestamp','qty','amount','outcome'})
        inferred=re.search(r'passengers-([A-Za-z0-9_.]+)-',name)
        for row,n in records:
            flight=row.get('flight') or (inferred.group(1) if inferred else '')
            if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,39}',flight):
                raise ValueError(f'{name}, строка {n}: не определён рейс.')
            timestamp=_time(row['timestamp'],name,n,'timestamp')
            dtd=int((day-timestamp.normalize()).days)
            if not 0<=dtd<=3650: raise ValueError(f'{name}, строка {n}: дата за пределами горизонта.')
            raw=row['outcome'];outcome='cancel' if raw.startswith('cancel') else raw
            if outcome not in ('success','waste','cancel'):
                raise ValueError(f'{name}, строка {n}: неизвестный outcome.')
            qty=_number(row['qty'],name,n,'qty',integer=True,positive=True) if row['qty'] else None
            amount=_number(row['amount'],name,n,'amount',positive=True) if row['amount'] else None
            if outcome!='cancel' and (qty is None or amount is None):
                raise ValueError(f'{name}, строка {n}: первичному обращению нужны qty и amount.')
            cabin=row.get('cabin','').upper() or None
            if cabin not in (None,'C','Y'): raise ValueError(f'{name}, строка {n}: cabin должен быть C/Y либо пустым.')
            key=(flight,row.get('id') or n,timestamp,outcome)
            if key in seen: raise ValueError(f'{name}, строка {n}: повторяется эталонная запись.')
            seen.add(key)
            rows.append({'timestamp':timestamp,'flight':flight,'qty':qty,'amount':amount,'outcome':outcome,
                         'cabin':cabin,'dtd':dtd,'period':period_of(dtd),'source_file':name,'source_row':n,
                         'rbd':row.get('rbd','').upper() or None,
                         'price':_number(row['price'],name,n,'price',positive=True) if row.get('price') else None,
                         'reference_outcome':raw})
    return pd.DataFrame(rows)


def _frame(reference):
    ref=reference.copy() if isinstance(reference,pd.DataFrame) else pd.DataFrame(reference)
    required={'qty','outcome','amount','period'}
    if required-set(ref): raise ValueError('В эталоне отсутствуют qty, outcome, amount или period.')
    return ref


def _quantile(values,weights,p):
    order=np.argsort(values);v=np.asarray(values,float)[order];w=np.asarray(weights,float)[order]
    return float(v[min(len(v)-1,np.searchsorted(np.cumsum(w),p*w.sum()))])


def evaluate_reference(result,reference):
    """Distributional comparison, with seat rather than row/group weighting."""
    ref=_frame(reference);flight=result['params']['flight']
    if 'flight' in ref:
        ref=ref.loc[ref.flight.eq(flight)].copy()
    if ref.empty: raise ValueError(f'{flight}: в passengers нет эталона этого рейса.')
    primary=ref.loc[ref.outcome.isin(['success','waste'])].copy()
    if primary.empty: raise ValueError('В эталоне нет первичных обращений.')
    primary['qty']=pd.to_numeric(primary.qty,errors='raise')
    if not np.isfinite(primary.qty).all() or not primary.qty.gt(0).all(): raise ValueError('Неверные qty эталона.')
    hidden=primary.loc[primary.outcome.eq('waste')]
    actual=hidden.groupby('period').qty.sum().to_dict()
    period_rows=[];errors=[];coverage=[]
    for row in result['periods']:
        target=float(actual.get(row['period'],0));estimate=float(row['estimated'])
        err=abs(estimate-target);errors.append(err)
        lo,hi=row.get('lower'),row.get('upper');covered=None if lo is None or hi is None else lo<=target<=hi
        if covered is not None: coverage.append(covered)
        period_rows.append({'period':row['period'],'actual':target,'estimated':estimate,'error':estimate-target,
                            'absolute_error':err,'lower':lo,'upper':hi,'covered_80':covered})
    reconstructed=pd.DataFrame(result['reconstructed'])
    modeled=reconstructed.loc[reconstructed.outcome.isin(['success','waste'])].copy()
    for col in ('amount','qty'):
        if col not in modeled: modeled[col]=np.nan
        modeled[col]=pd.to_numeric(modeled[col],errors='coerce')
    good=modeled.loc[modeled.amount.notna() & modeled.amount.gt(0) & modeled.qty.gt(0)]
    true_budget=primary.loc[pd.to_numeric(primary.amount,errors='coerce').gt(0)]
    budgets=[];distance=None
    if len(good) and len(true_budget):
        distance=float(wasserstein_distance(true_budget.amount.astype(float),good.amount.astype(float),
                                            true_budget.qty.astype(float),good.qty.astype(float)))
        for q in (.1,.5,.9):
            a=_quantile(true_budget.amount,true_budget.qty,q);b=_quantile(good.amount,good.qty,q)
            budgets.append({'quantile':q,'actual':a,'estimated':b,'absolute_error':abs(a-b)})
    warnings=['Общее число скрытых мест задано через внешний N; совпадение итогового количества не является оценкой качества.',
              'Сравниваются распределения мест и бюджетов, а не личности или синтетические ID. Бюджет — за одно место.']
    expected=int(primary.qty.sum());given=int(result['params']['demand'])
    if expected!=given: warnings.append(f'Эталон содержит {expected} первичных мест, а расчёт использует N={given}; сравниваются разные объёмы.')
    reference_gross=float(primary.loc[primary.outcome.eq('success'),'qty'].sum())
    if reference_gross!=result['summary']['gross']:
        warnings.append('Число покупок в эталоне не совпадает с bookings: проверьте соответствие файлов и горизонтов.')
    cabin_rows=[];known_hidden=hidden.loc[hidden.get('cabin',pd.Series(index=hidden.index,dtype=object)).isin(['C','Y'])]
    known_qty=float(known_hidden.qty.sum())
    if known_qty and known_qty==float(hidden.qty.sum()):
        latent=reconstructed.loc[reconstructed.outcome.eq('waste')].copy()
        if 'model_preferred_cabin' in latent:
            latent['eval_cabin']=latent.model_preferred_cabin.fillna(latent.get('cabin'))
        else: latent['eval_cabin']=latent.get('cabin')
        if latent.eval_cabin.notna().all():
            for cabin in ('C','Y'):
                a=float(known_hidden.loc[known_hidden.cabin.eq(cabin),'qty'].sum())
                b=float(latent.loc[latent.eval_cabin.eq(cabin),'qty'].sum())
                cabin_rows.append({'cabin':cabin,'actual':a,'estimated':b,'absolute_error':abs(a-b)})
    if not cabin_rows: warnings.append('Салон скрытого обращения отсутствует в эталоне или методе: ошибка по салонам не вычисляется.')
    denom=float(hidden.qty.sum())
    summary={'hidden_period_mae':float(np.mean(errors)),'hidden_period_wape':sum(errors)/denom*100 if denom else None,
             'hidden_cabin_mae':float(np.mean([r['absolute_error'] for r in cabin_rows])) if cabin_rows else None,
             'interval_coverage_80':float(np.mean(coverage)*100) if coverage else None,
             'budget_wasserstein':distance,'budget_reference_seats':float(true_budget.qty.sum()),
             'budget_estimated_seats':float(good.qty.sum()),'reference_primary_seats':expected,
             'reference_hidden_seats':denom,'known_hidden_cabin_seats':known_qty,
             'reference_gross':float(primary.loc[primary.outcome.eq('success'),'qty'].sum())}
    return {'ok':True,'title':'Проверка по эталонным passengers','summary':summary,
            'tables':{'periods':period_rows,'cabins':cabin_rows,'budgets':budgets},
            'charts':{'periods':period_rows,'budget_quantiles':budgets},'warnings':warnings,
            'evaluation_scope':'post_fit_distributional_seat_weighted_reference_only'}


evaluate_result=evaluate_reference


def _assert_reference_matches(result,reference):
    """Compare observed purchases only; hidden identities are never matched."""
    flight=result['params']['flight']
    ref=reference.loc[reference.flight.eq(flight)] if 'flight' in reference else reference
    successes=ref.loc[ref.outcome.eq('success')]
    if float(successes.qty.sum())!=result['summary']['gross']:
        raise ValueError(f'{flight}: passengers и bookings содержат разное число купленных мест. Выберите эталон того же запуска и горизонта.')
    observed=pd.DataFrame(result.get('observed') or [])
    if not len(observed):
        if len(successes): raise ValueError(f'{flight}: эталон содержит покупки, которых нет в bookings.')
        return
    observed=observed.loc[observed.outcome.eq('success')]
    if 'timestamp' not in successes or successes.timestamp.isna().any():
        raise ValueError(f'{flight}: в эталоне отсутствуют времена наблюдаемых покупок.')
    def key(row): return (pd.Timestamp(row['timestamp']).value,int(row['qty']))
    reference_rows=successes.to_dict('records');observed_rows=observed.to_dict('records')
    if Counter(map(key,reference_rows))!=Counter(map(key,observed_rows)):
        raise ValueError(f'{flight}: времена или размеры покупок в passengers не совпадают с bookings. Вероятно, выбран другой seed или запуск.')
    # Rows at the same instant can represent distinct products. Match known
    # fields without relying on IDs; incomplete fields remain unconstrained.
    from scipy.optimize import linear_sum_assignment
    expected_groups={};actual_groups={}
    for row in reference_rows: expected_groups.setdefault(key(row),[]).append(row)
    for row in observed_rows: actual_groups.setdefault(key(row),[]).append(row)
    for group_key,expected in expected_groups.items():
        actual=actual_groups[group_key]
        def compatible(a,b):
            for column in ('cabin','rbd','price'):
                value=a.get(column)
                if value is None or pd.isna(value) or value=='': continue
                other=b.get(column)
                if column=='price':
                    if other is None or not math.isclose(float(value),float(other),rel_tol=1e-8,abs_tol=.005): return False
                elif str(value).upper()!=str(other).upper(): return False
            return True
        if len(expected)>512:
            raise ValueError(f'{flight}: более 512 неразличимых покупок в одну дату. Уточните времена операций или разбейте комплект для проверки.')
        costs=np.array([[0 if compatible(a,b) else 1 for b in actual] for a in expected])
        left,right=linear_sum_assignment(costs)
        if costs[left,right].sum():
            raise ValueError(f'{flight}: салон, RBD или цена покупки в passengers не соответствует bookings. Выберите согласованный комплект файлов.')


def run_evaluation(payload,progress=None,cancelled=None):
    # Reference parsing intentionally follows the finished inverse calculation.
    from .service import calculate,_json_safe
    inputs={k:v for k,v in payload.items() if k not in ('reference_files','passengers_files')}
    result=calculate(inputs)
    if not result.get('ok'): return result
    files=payload.get('reference_files') or payload.get('passengers_files') or []
    if files:
        ref=parse_passengers(files,result['params']['departure']);reference_source='uploaded_passengers'
    elif result.get('mode',inputs.get('mode','simulation'))=='simulation':
        ref=pd.DataFrame([event for calculated in result['results'].values() for event in calculated['events']])
        reference_source='synthetic_simulation_events_after_fit'
    else:
        raise ValueError('Для проверки загруженного bookings выберите соответствующий passengers CSV.')
    output={};warnings=[];summaries=[];periods=[];budgets=[];cabins=[]
    if progress: progress(0,len(result['results']),'Проверка по эталону после завершённой реконструкции')
    for flight,calculated in result['results'].items():
        if cancelled and cancelled(): raise InterruptedError('Проверка отменена.')
        _assert_reference_matches(calculated,ref)
        checked=evaluate_reference(calculated,ref)
        output[flight]=checked
        summaries.append({'flight':flight,**checked['summary']});warnings.extend(checked['warnings'])
        for key,target in [('periods',periods),('budgets',budgets),('cabins',cabins)]:
            target.extend({'flight':flight,**row} for row in checked['tables'][key])
        if progress: progress(len(output),len(result['results']),f'Эталон проверен: {flight}')
    return _json_safe({'ok':True,'title':'Проверка реконструкции по passengers',
                      'summary':{'flights':len(output)},
                      'tables':{'flights':summaries,'periods':periods,'budgets':budgets,'cabins':cabins},
                      'charts':{'periods':periods,'budget_quantiles':budgets},
                      'results':output,'warnings':list(dict.fromkeys(warnings)),
                      'reference_source':reference_source,
                      'reference_fingerprints':[{'name':f['name'],'sha256':hashlib.sha256(f['text'].encode()).hexdigest()} for f in files]})
