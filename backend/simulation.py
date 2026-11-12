"""Forward passenger simulator, independent of the inverse reconstruction.

The primary cohort is sampled before any price or capacity policy is applied.
Changing those policies with the same seed preserves requests and budgets.
Offer histories record state changes even if no request buys anything.
"""
from datetime import date
import math
import re

import numpy as np
import pandas as pd

PERIODS = ['91+', '31–90', '15–30', '8–14', '0–7']
PROFILES = {'balanced': [10,24,26,20,20], 'early': [35,30,18,10,7],
            'late': [5,12,20,25,38], 'notebook': [57,54,70,46,26]}
DEFAULTS = {'demand':300, 'capacity':180, 'businessCapacity':12,
    'businessShare':5, 'price':12000, 'budget':23000, 'cancel':4,
    'growth':80, 'profile':'balanced', 'seed':42, 'departure':'2027-05-22',
    'flight':'SU1122'}
RBD_DESCENDING = {'C':'JOCIZ', 'Y':'YXDSBMUPKHLQTENRGV'}
QTY_PROBABILITIES = {1:.64,2:.25,3:.08,4:.03}


def period_of(dtd):
    if dtd < 0:
        raise ValueError('Обнаружена операция после дня вылета.')
    return PERIODS[0 if dtd>90 else 1 if dtd>30 else 2 if dtd>14 else 3 if dtd>7 else 4]


def validate_params(values):
    if not isinstance(values,dict):
        raise ValueError('Параметры сценария должны быть объектом.')
    p={**DEFAULTS,**values}
    bounds={'demand':(1,10000),'capacity':(1,2000),'businessCapacity':(0,2000),
            'businessShare':(0,100),'price':(100,1000000),'budget':(100,1000000),
            'cancel':(0,100),'growth':(0,300),'seed':(0,4294967295)}
    for key,(low,high) in bounds.items():
        value=p[key]
        if isinstance(value,bool): raise ValueError(f'Некорректный параметр {key}.')
        try: number=float(value)
        except (TypeError,ValueError): raise ValueError(f'Некорректный параметр {key}.') from None
        if not math.isfinite(number) or not number.is_integer() or not low<=number<=high:
            raise ValueError(f'Параметр {key}: требуется целое число от {low} до {high}.')
        p[key]=int(number)
    if p['businessCapacity']>p['capacity']:
        raise ValueError('Мест в бизнес-классе больше общей вместимости.')
    if p['profile'] not in PROFILES: raise ValueError('Неизвестный профиль обращений.')
    raw=str(p['departure'])
    try:
        valid=bool(re.fullmatch(r'\d{4}-\d{2}-\d{2}',raw)) and date.fromisoformat(raw).isoformat()==raw
    except ValueError: valid=False
    if not valid: raise ValueError('Проверьте дату вылета.')
    flight=str(p['flight']).strip()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,39}',flight):
        raise ValueError('Рейс: используйте до 40 букв, цифр, точку, дефис или подчёркивание.')
    p['flight']=flight
    return p


def primary_requests(params):
    """Generate exact primary SEATS, including a final truncated group."""
    p=validate_params(params);rng=np.random.default_rng(p['seed'])
    departure=pd.Timestamp(p['departure'],tz='UTC')
    weights=np.asarray(PROFILES[p['profile']],float);weights/=weights.sum()
    ranges=[(91,180),(31,90),(15,30),(8,14),(0,7)]
    remaining=p['demand'];records=[]
    while remaining:
        qty=min(remaining,int(rng.choice([1,2,3,4],p=list(QTY_PROBABILITIES.values()))))
        remaining-=qty
        period=int(rng.choice(5,p=weights));low,high=ranges[period]
        dtd=int(rng.integers(low,high+1))
        seconds=int(rng.integers(0,int(.95*86400*1000)))
        timestamp=departure-pd.Timedelta(days=dtd)+pd.Timedelta(milliseconds=seconds)
        cabin='C' if rng.random()<p['businessShare']/100 else 'Y'
        latent_normal=float(rng.normal())
        amount=float(round(p['budget']*(3 if cabin=='C' else 1)*(1+.25*(1-dtd/180))*math.exp(.48*latent_normal)))
        cancellation_uniform=float(rng.random());delay_uniform=float(rng.random())
        cancel_at=timestamp+(departure+pd.Timedelta(days=1,milliseconds=-1000)-timestamp)*(.15+.8*delay_uniform)
        records.append({'timestamp':timestamp,'id':f'SIM-{len(records)+1:05d}',
            'qty':qty,'cabin':cabin,'amount':amount,'dtd':dtd,'period':PERIODS[period],
            'cancellation_uniform':cancellation_uniform,'cancel_at':cancel_at})
    return pd.DataFrame(records).sort_values(['timestamp','id']).reset_index(drop=True)


def simulate(params):
    """Return primary truth/events, observed bookings, and complete offers.

    Returned truth is intended for scoring/display only. The inverse model
    receives the separate bookings and offers tables, never the truth frame.
    """
    p=validate_params(params);requests=primary_requests(p)
    departure=pd.Timestamp(p['departure'],tz='UTC')
    start=departure-pd.Timedelta(days=180);end=departure+pd.Timedelta(days=1)
    capacity={'C':p['businessCapacity'],'Y':p['capacity']-p['businessCapacity']}
    booked={'C':0,'Y':0};purchases={};events=[];operations=[]
    # Decisions at time t see the pre-event state. Inventory changes take
    # effect at t+1ns in offers, preserving the actual purchase's state.
    changes={c:[(start,0)] for c in capacity}
    queue=[]
    for record in requests.to_dict('records'):
        queue.append((record['timestamp'],1,record))
        if record['cancellation_uniform']<p['cancel']/100:
            queue.append((record['cancel_at'],0,record))
    queue.sort(key=lambda x:(x[0],x[1],x[2]['id']))

    def quote(cabin,time,occupancy):
        dtd=int((departure-time.normalize()).days)
        load=occupancy/capacity[cabin] if capacity[cabin] else 1.
        base=p['price']*(3 if cabin=='C' else 1)
        price=float(math.floor(base*(1+p['growth']/100*(1-dtd/180))*(1+.3*load)/10+.5)*10)
        levels=RBD_DESCENDING[cabin][::-1]
        tier=min(len(levels)-1,max(0,int((price/base-1)*5)))
        return price,levels[tier]

    for timestamp,kind,q in queue:
        cabin=q['cabin']
        if kind==0:
            purchase=purchases.get(q['id'])
            if purchase is None: continue
            event={**purchase,'timestamp':timestamp,'id':'CXL-'+q['id'],
                'cancelRef':q['id'],'outcome':'cancel','signed_qty':-purchase['qty'],
                'amount':None,'reason':'Отмена бронирования'}
            event['dtd']=int((departure-timestamp.normalize()).days)
            event['period']=period_of(event['dtd'])
            booked[cabin]-=purchase['qty']
            changes[cabin].append((timestamp+pd.Timedelta(1,'ns'),booked[cabin]))
        else:
            price,rbd=quote(cabin,timestamp,booked[cabin])
            reason='Нет мест' if q['qty']>capacity[cabin]-booked[cabin] else 'Цена выше бюджета' if q['amount']<price else ''
            event={k:v for k,v in q.items() if k not in ('cancellation_uniform','cancel_at')}
            event.update({'price':price,'rbd':rbd,'outcome':'waste' if reason else 'success',
                'signed_qty':0 if reason else q['qty'],'reason':reason,'flight':p['flight'],
                'source_file':'synthetic','source_row':None,'id_is_synthetic':True,
                'reconstruction_status':'синтетическая операция','record_level':'отдельная операция',
                'amount_source':'synthetic_truth','reason_is_modelled':False})
            if not reason:
                booked[cabin]+=q['qty'];purchases[q['id']]=dict(event)
                changes[cabin].append((timestamp+pd.Timedelta(1,'ns'),booked[cabin]))
        if not 0<=booked[cabin]<=capacity[cabin]: raise AssertionError('Нарушена вместимость салона.')
        events.append(event)
        if event['outcome']!='waste':
            operations.append({'timestamp':timestamp,'flight':p['flight'],'cabin':cabin,
                'rbd':event['rbd'],'commercial':event['signed_qty'],'price':event['price'],
                'source_file':'synthetic','source_row':len(operations)+2})
    offers=[]
    for cabin in capacity:
        mapping=dict(changes[cabin])
        boundaries=sorted(set(pd.date_range(start,end,freq='D').tolist()+list(mapping)))
        occupancy=0
        for left,right in zip(boundaries,boundaries[1:]):
            if left in mapping: occupancy=mapping[left]
            price,rbd=quote(cabin,left,occupancy)
            offers.append({'flight':p['flight'],'cabin':cabin,'start':left,'end':right,
                'price':price,'open':True,'available_seats':capacity[cabin]-occupancy,'rbd':rbd})
    columns=['timestamp','flight','cabin','rbd','commercial','price','source_file','source_row']
    bookings=pd.DataFrame(operations,columns=columns)
    if bookings.empty: bookings['timestamp']=pd.Series(dtype='datetime64[ns, UTC]')
    truth=pd.DataFrame(events)
    if int(truth.loc[truth.outcome.isin(['success','waste']),'qty'].sum())!=p['demand']:
        raise AssertionError('Нарушен первичный объём мест.')
    return {'params':p,'requests':requests,'events':truth,'bookings':bookings,
            'offers':pd.DataFrame(offers),'horizon_start':start,'horizon_end':end}
