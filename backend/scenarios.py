"""Independent forward scenarios used for evaluation, never inverse-model input.

The same pre-generated cohort is replayed under every price policy. Its random
draws do not depend on price, available seats, or which policy is evaluated.
"""
import math
from collections.abc import Mapping

import numpy as np
import pandas as pd

from .simulation import DEFAULTS, RBD_DESCENDING, primary_requests, period_of, validate_params


SCENARIOS = {
    'balanced': {'label':'Сбалансированный спрос','description':'Базовый профиль обращений и бюджетов.'},
    'early': {'label':'Ранние обращения','description':'Большинство групп приходит задолго до вылета.'},
    'late': {'label':'Поздний спрос','description':'Большая доля обращений в последние недели.'},
    'burst': {'label':'Всплески спроса','description':'65% обращений сосредоточены в трёх коротких всплесках.'},
    'mixed_wtp': {'label':'Две группы бюджетов','description':'Смесь бюджетов: множители 0,6 и 1,9 с долями 70/30.'},
    'closure': {'label':'Закрытие продаж','description':'Продажи закрыты в интервалы DtD 40–26 и 10–9.'},
    'groups_budget': {'label':'Бюджет зависит от группы','description':'Бюджет на место снижается пропорционально qty^−0,45.'},
    'high_cancel': {'label':'Частые возвраты','description':'Вероятность полного возврата купленной группы — 30%.'},
    'low_demand': {'label':'Низкий спрос','description':'Первичный спрос составляет 45% базового значения.'},
    'sold_out': {'label':'Дефицит мест','description':'Вместимость уменьшена до 45%, первичный спрос сохранён.'},
}
SCENARIO_ALIASES={'group_budget':'groups_budget','mixedWTP':'mixed_wtp','highcancel':'high_cancel','lowdemand':'low_demand','soldout':'sold_out'}


def scenario_id(value):
    name=SCENARIO_ALIASES.get(str(value),str(value))
    if name not in SCENARIOS: raise ValueError('Неизвестный сценарий: '+name)
    return name


def prepare_cohort(scenario,seed,params=None):
    """Return explicit scenario parameters and an exact-N primary cohort."""
    scenario=scenario_id(scenario)
    supplied={**DEFAULTS,**(params or {}),'seed':seed}
    if scenario=='early': supplied['profile']='early'
    elif scenario=='late': supplied['profile']='late'
    elif scenario=='high_cancel': supplied['cancel']=30
    elif scenario=='low_demand': supplied['demand']=max(1,round(float(supplied['demand'])*.45))
    elif scenario=='sold_out':
        supplied['capacity']=max(1,round(float(supplied['capacity'])*.45))
        supplied['businessCapacity']=min(supplied['capacity'],round(float(supplied['businessCapacity'])*.45))
    p=validate_params(supplied)
    requests=primary_requests(p)
    # pandas 3 may infer microsecond storage; burst jitter and fractional
    # cancellation delays require nanosecond precision without silent loss.
    for column in ('timestamp','cancel_at'):
        requests[column]=requests[column].astype('datetime64[ns, UTC]')
    rng=np.random.default_rng(np.random.SeedSequence([int(seed),91719]))
    departure=pd.Timestamp(p['departure'],tz='UTC')
    if scenario=='burst':
        for index in requests.index:
            if rng.random()<.65:
                dtd=int(rng.choice([110,45,5]))
                timestamp=departure-pd.Timedelta(days=dtd)+pd.Timedelta(seconds=float(rng.uniform(0,4*3600)))
                requests.loc[index,'timestamp']=timestamp
                requests.loc[index,'dtd']=dtd
                requests.loc[index,'period']=period_of(dtd)
                # Redraw the cancellation delay after moving the request.
                delay=float(rng.uniform(.15,.95))
                requests.loc[index,'cancel_at']=timestamp+(departure+pd.Timedelta(days=1,seconds=-1)-timestamp)*delay
    elif scenario=='mixed_wtp':
        requests['amount']=np.round(requests.amount*np.where(rng.random(len(requests))<.7,.6,1.9))
    elif scenario=='groups_budget':
        requests['amount']=np.round(requests.amount/requests.qty.pow(.45))
    return p,requests.sort_values(['timestamp','id']).reset_index(drop=True)


def _policy(policy):
    if policy is None: policy={'kind':'dynamic','name':'Текущая динамическая цена'}
    if not isinstance(policy,Mapping): raise ValueError('Политика должна быть объектом.')
    answer={**policy}
    kind=answer.get('kind','dynamic')
    if kind not in ('dynamic','static','gated'): raise ValueError('Неизвестная тарифная политика.')
    multiplier=float(answer.get('multiplier',1.))
    growth=float(answer.get('growth',1.))
    if not math.isfinite(multiplier) or not .25<=multiplier<=4: raise ValueError('Множитель цены должен быть от 0,25 до 4.')
    if not math.isfinite(growth) or not 0<=growth<=4: raise ValueError('Множитель роста должен быть от 0 до 4.')
    return {**answer,'kind':kind,'multiplier':multiplier,'growth':growth}


def simulate_cohort(params,requests,scenario='balanced',policy=None):
    """Replay groups against shared cabin inventory with complete actual offers.

    Gated policy opens a nested subset of RBD and quotes its cheapest open
    class. All open RBD share the cabin's remaining physical seat inventory;
    their per-class available counts must never be summed as separate seats.
    """
    scenario=scenario_id(scenario);p=validate_params(params);policy=_policy(policy)
    departure=pd.Timestamp(p['departure'],tz='UTC')
    start=departure-pd.Timedelta(days=180);end=departure+pd.Timedelta(days=1)
    capacity={'C':p['businessCapacity'],'Y':p['capacity']-p['businessCapacity']}
    closures=[]
    if scenario=='closure':
        closures=[(departure-pd.Timedelta(days=40),departure-pd.Timedelta(days=25)),
                  (departure-pd.Timedelta(days=10),departure-pd.Timedelta(days=8))]
    booked={c:0 for c in capacity};changes={c:[(start,0)] for c in capacity}
    purchases={};events=[];operations=[];queue=[]
    for q in requests.to_dict('records'):
        queue.append((q['timestamp'],1,q))
        if q['cancellation_uniform']<p['cancel']/100: queue.append((q['cancel_at'],0,q))
    queue.sort(key=lambda x:(x[0],x[1],x[2]['id']))

    def quote(cabin,time,occupancy):
        dtd=int((departure-time.normalize()).days)
        time_share=max(0.,min(1.,1-dtd/180))
        load=occupancy/capacity[cabin] if capacity[cabin] else 1.
        base=p['price']*(3 if cabin=='C' else 1)*policy['multiplier']
        descending=RBD_DESCENDING[cabin];ascending=descending[::-1]
        if policy['kind']=='static':
            price=base;tier=0
        elif policy['kind']=='gated':
            # Closing inexpensive classes as departure/occupancy advance is
            # an explicit rule, not an allegedly optimal policy.
            fraction=min(.9,.55*load+.35*time_share*policy['growth'])
            tier=min(len(ascending)-1,max(0,int(fraction*(len(ascending)-1))))
            price=base*(1+1.8*tier/(len(ascending)-1))
        else:
            price=base*(1+p['growth']/100*policy['growth']*time_share)*(1+.3*load)
            tier=min(len(ascending)-1,max(0,int((price/base-1)*5)))
        price=float(max(10,math.floor(price/10+.5)*10))
        rbd=ascending[tier]
        opened=capacity[cabin]>occupancy and not any(left<=time<right for left,right in closures)
        open_rbds=ascending[tier:] if policy['kind']=='gated' else rbd
        return price,rbd,opened,','.join(open_rbds)

    for timestamp,kind,q in queue:
        cabin=q['cabin']
        if kind==0:
            original=purchases.get(q['id'])
            if original is None: continue
            event={**original,'timestamp':timestamp,'id':'CXL-'+q['id'],'cancelRef':q['id'],
                'outcome':'cancel','signed_qty':-q['qty'],'amount':None,'reason':'Отмена бронирования'}
            event['dtd']=int((departure-timestamp.normalize()).days);event['period']=period_of(event['dtd'])
            booked[cabin]-=q['qty'];changes[cabin].append((timestamp+pd.Timedelta(1,'ns'),booked[cabin]))
        else:
            price,rbd,opened,open_rbds=quote(cabin,timestamp,booked[cabin])
            reason='Продажи закрыты' if not opened and any(a<=timestamp<b for a,b in closures) else 'Нет мест' if q['qty']>capacity[cabin]-booked[cabin] else 'Цена выше бюджета' if q['amount']<price else ''
            event={k:v for k,v in q.items() if k not in ('cancellation_uniform','cancel_at')}
            event.update({'price':price,'rbd':rbd,'open_rbds':open_rbds,'outcome':'waste' if reason else 'success',
                'signed_qty':0 if reason else q['qty'],'reason':reason,'flight':p['flight'],
                'source_file':'scenario:'+scenario,'source_row':None,'id_is_synthetic':True,
                'reconstruction_status':'синтетическая операция','record_level':'отдельная операция',
                'amount_source':'synthetic_truth','reason_is_modelled':False})
            if not reason:
                booked[cabin]+=q['qty'];purchases[q['id']]=dict(event)
                changes[cabin].append((timestamp+pd.Timedelta(1,'ns'),booked[cabin]))
        if not 0<=booked[cabin]<=capacity[cabin]: raise AssertionError('Нарушена общая вместимость салона.')
        events.append(event)
        if event['outcome']!='waste':
            operations.append({'timestamp':timestamp,'flight':p['flight'],'cabin':cabin,'rbd':event['rbd'],
                'commercial':event['signed_qty'],'price':event['price'],'source_file':'scenario:'+scenario,'source_row':len(operations)+2})
    offers=[]
    closure_bounds=[b for window in closures for b in window]
    for cabin in capacity:
        mapping=dict(changes[cabin]);occupancy=0
        boundaries=sorted(set(pd.date_range(start,end,freq='D').tolist()+list(mapping)+closure_bounds))
        for left,right in zip(boundaries,boundaries[1:]):
            if left in mapping: occupancy=mapping[left]
            price,rbd,opened,open_rbds=quote(cabin,left,occupancy)
            offers.append({'flight':p['flight'],'cabin':cabin,'start':left,'end':right,'price':price,
                'open':bool(opened),'available_seats':capacity[cabin]-occupancy,'rbd':rbd,
                'open_rbds':open_rbds,'offer_source':'synthetic_actual_states','inventory_scope':'shared_cabin',
                'policy':policy['kind']})
    columns=['timestamp','flight','cabin','rbd','commercial','price','source_file','source_row']
    bookings=pd.DataFrame(operations,columns=columns)
    if bookings.empty: bookings['timestamp']=pd.Series(dtype='datetime64[ns, UTC]')
    truth=pd.DataFrame(events)
    if int(truth.loc[truth.outcome.isin(['success','waste']),'qty'].sum())!=p['demand']: raise AssertionError('Нарушен N.')
    return {'params':p,'requests':requests.copy(),'events':truth,'passengers':truth.copy(),
        'bookings':bookings,'offers':pd.DataFrame(offers),'horizon_start':start,'horizon_end':end,
        'scenario':scenario,'policy':policy}


def generate_bundle(scenario='balanced',seed=42,params=None):
    p,cohort=prepare_cohort(scenario,seed,params)
    return simulate_cohort(p,cohort,scenario)
