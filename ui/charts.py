"""Charts authored in Python and rendered by native Streamlit/Altair widgets."""
from collections import OrderedDict
import re

import altair as alt
import pandas as pd
import streamlit as st

from backend.simulation import PERIODS,RBD_DESCENDING

COLORS=['#0b7467','#dc8d53','#84bbac','#778b9e','#aa917d']
SERIES={'actual':'Эталон','estimated':'Оценка','baseline':'Базовый метод','v3':'Модель v3',
        'holdout_mae':'MAE на проверке, мест','predicted_net':'Прогноз занятых мест',
        'actual_net':'Фактический баланс','lower':'Нижняя граница 10%',
        'upper':'Верхняя граница 90%','min_estimated':'Минимальная оценка',
        'max_estimated':'Максимальная оценка','value':'Значение'}
TITLES={'period_mae':'Ошибка скрытого спроса по сценариям','holdout':'Проверка на новых запусках',
        'periods':'Скрытый спрос по периодам','budget_quantiles':'Распределение бюджетов',
        'forecast':'Прогноз загрузки','envelope':'Диапазон чувствительности',
        'policy_revenue':'Выручка тарифных политик','revenue_load':'Выручка и загрузка'}


def _style(chart,height=280):
    return chart.properties(height=height).configure_view(stroke=None).configure_axis(
        gridColor='#e6eeea',labelColor='#778b80',titleColor='#537061',labelFontSize=11,
        titleFontSize=11,domain=False,tickColor='#e6eeea').configure_legend(
        orient='bottom',title=None,labelColor='#537061',labelFontSize=11)


def curve_frame(result,baseline=None):
    rows=[]
    for point in result.get('curve',[]):
        for field,label in [('net','Баланс после отмен'),('gross','Покупки до отмен')]:
            rows.append({'dtd':point['dtd'],'seats':point[field],'series':label})
    if baseline:
        rows.extend({'dtd':point['dtd'],'seats':point['net'],'series':'Сохранённый сценарий'} for point in baseline.get('curve',[]))
    return pd.DataFrame(rows,columns=['dtd','seats','series'])


def comparison_curve(result,baseline=None):
    data=curve_frame(result,baseline)
    if data.empty:return None
    line=alt.Chart(data).mark_line(strokeWidth=2.6).encode(
        x=alt.X('dtd:Q',title='Дней до вылета',scale=alt.Scale(reverse=True)),
        y=alt.Y('seats:Q',title='Мест',scale=alt.Scale(zero=True)),
        color=alt.Color('series:N',title=None,scale=alt.Scale(range=COLORS)),
        tooltip=[alt.Tooltip('dtd:Q',title='Дней до вылета'),alt.Tooltip('series:N',title='Показатель'),alt.Tooltip('seats:Q',title='Мест',format=',.0f')])
    capacity=alt.Chart(pd.DataFrame({'capacity':[result['params']['capacity']]})).mark_rule(strokeDash=[5,4],color='#b7c8bf').encode(y='capacity:Q')
    return _style(line+capacity,300)


def period_frame(result,view='demand'):
    rows=[];is_sim=result.get('mode')=='simulation'
    if view=='estimate':
        fields=[('estimated','Оценка модели'),('actual','Синтетический факт')]
    else:
        fields=[('purchased','Покупки до отмен'),('actual' if is_sim else 'estimated','Без покупки · факт' if is_sim else 'Без покупки · оценка'),('canceled','Отмены')]
    for item in result.get('periods',[]):
        for field,label in fields:
            if item.get(field) is not None:
                rows.append({'period':item['period'],'series':label,'seats':item[field]})
    return pd.DataFrame(rows,columns=['period','series','seats'])


def period_chart(result,view='demand'):
    data=period_frame(result,view)
    if data.empty:return None
    chart=alt.Chart(data).mark_bar(cornerRadiusTopLeft=3,cornerRadiusTopRight=3).encode(
        x=alt.X('period:N',title='Дней до вылета',sort=PERIODS),
        xOffset=alt.XOffset('series:N'),y=alt.Y('seats:Q',title='Мест'),
        color=alt.Color('series:N',title=None,scale=alt.Scale(range=COLORS)),
        tooltip=[alt.Tooltip('period:N',title='Период'),alt.Tooltip('series:N',title='Показатель'),alt.Tooltip('seats:Q',title='Мест')])
    return _style(chart)


def fare_frame(result,cabin='Все салоны'):
    frame=pd.DataFrame(result.get('fares',[]))
    if frame.empty:return pd.DataFrame(columns=['label','cabin','rbd','purchased','canceled','average_price'])
    if cabin in ('C','Y'):frame=frame.loc[frame.cabin.eq(cabin)]
    rows=[]
    for (code,rbd),group in frame.groupby(['cabin','rbd'],sort=False):
        bought=float(group.purchased.sum())
        rows.append({'label':code+' / '+rbd,'cabin':code,'rbd':rbd,
            'purchased':bought,'canceled':float(group.canceled.sum()),
            'average_price':float((group.price*group.purchased).sum()/bought) if bought else None})
    rows.sort(key=lambda row:(row['cabin'],RBD_DESCENDING.get(row['cabin'],'').find(row['rbd']) if row['rbd'] in RBD_DESCENDING.get(row['cabin'],'') else 99))
    return pd.DataFrame(rows)


def fare_chart(result,cabin='Все салоны'):
    data=fare_frame(result,cabin)
    if data.empty:return None
    melted=data.melt(id_vars=['label','average_price'],value_vars=['purchased','canceled'],var_name='series',value_name='seats')
    melted['series']=melted.series.map({'purchased':'Куплено','canceled':'Отменено'})
    chart=alt.Chart(melted).mark_bar(cornerRadiusTopLeft=2,cornerRadiusTopRight=2).encode(
        x=alt.X('label:N',sort=data.label.tolist(),title='Салон / RBD'),xOffset='series:N',
        y=alt.Y('seats:Q',title='Мест'),color=alt.Color('series:N',title=None,scale=alt.Scale(range=COLORS)),
        tooltip=[alt.Tooltip('label:N',title='Тариф'),alt.Tooltip('series:N',title='Операция'),
                 alt.Tooltip('seats:Q',title='Мест'),alt.Tooltip('average_price:Q',title='Средняя цена покупки, ₽',format=',.0f')])
    return _style(chart,260)


def reason_frame(result):
    summary=result['summary']
    if result.get('mode')=='simulation':
        rows=[{'reason':'Цена выше бюджета','seats':summary.get('priceWaste')},
              {'reason':'Недостаточно мест','seats':summary.get('capacityWaste')}]
    else:
        totals=result.get('diagnostics',{}).get('modelled_reason_totals')
        if not totals:return pd.DataFrame(columns=['reason','seats'])
        rows=[{'reason':{'budget':'Бюджет · модель','availability':'Доступность · модель','other':'Прочее · модель'}.get(key,key),'seats':value} for key,value in totals.items()]
    return pd.DataFrame([row for row in rows if row['seats'] is not None])


def reason_chart(result):
    data=reason_frame(result)
    if data.empty:return None
    chart=alt.Chart(data).mark_bar(cornerRadiusEnd=4).encode(
        y=alt.Y('reason:N',title=None,sort='-x'),x=alt.X('seats:Q',title='Мест'),
        color=alt.Color('reason:N',legend=None,scale=alt.Scale(range=['#dc8d53','#a5cabe','#778b9e'])),
        tooltip=[alt.Tooltip('reason:N',title='Причина'),alt.Tooltip('seats:Q',title='Мест',format=',.1f')])
    return _style(chart,210)


def _known_chart(name,rows):
    frame=pd.DataFrame(rows)
    if frame.empty:return None
    if name=='period_mae' and {'label','value'}<=set(frame):
        values=OrderedDict()
        for row in rows:
            parts=str(row['label']).rsplit(' · ',1)
            if len(parts)==2 and parts[1] in ('baseline','v3'):
                values.setdefault(parts[0],{'label':parts[0]})[parts[1]]=row['value']
        if values:return _generic(pd.DataFrame(values.values()),'label',['baseline','v3'],'bar')
    if {'x','y'}<=set(frame):return _generic(frame,'x',['y'],'scatter')
    keys={'holdout':['holdout_mae'],'periods':['actual','estimated'],'budget_quantiles':['actual','estimated'],
          'forecast':['predicted_net','actual_net','lower','upper'],'envelope':['min_estimated','max_estimated'],
          'policy_revenue':['value'],'period_mae':['value']}.get(name)
    xkey=next((key for key in ('date','period','quantile','label','method','cutoff_dtd') if key in frame),None)
    if not xkey:xkey=next((key for key in frame if not pd.api.types.is_numeric_dtype(frame[key])),frame.columns[0])
    keys=keys or [key for key in frame if key!=xkey and pd.api.types.is_numeric_dtype(frame[key]) and key not in ('seed','dtd')]
    return _generic(frame,xkey,[key for key in keys if key in frame],'line' if xkey=='date' else 'bar')


def _generic(frame,xkey,keys,kind):
    if not keys or xkey not in frame:return None
    if kind=='scatter':
        data=frame.copy();tooltip=[alt.Tooltip('label:N',title='Политика')] if 'label' in data else []
        base=alt.Chart(data).mark_circle(size=100).encode(x=alt.X(f'{xkey}:Q',title='Загрузка, %'),y=alt.Y(f'{keys[0]}:Q',title='Выручка, ₽'),tooltip=tooltip+[alt.Tooltip(f'{xkey}:Q',title='Загрузка, %'),alt.Tooltip(f'{keys[0]}:Q',title='Выручка, ₽',format=',.0f')])
        if 'pareto' in data:base=base.encode(color=alt.Color('pareto:N',title='На границе компромисса',scale=alt.Scale(range=COLORS)))
        return _style(base)
    idvars=[xkey]+(['flight'] if 'flight' in frame and xkey!='flight' else [])
    data=frame.melt(id_vars=idvars,value_vars=keys,var_name='series',value_name='value')
    data['value']=pd.to_numeric(data.value,errors='coerce');data=data.loc[data.value.notna()]
    if data.empty:return None
    data['series']=data.series.map(lambda key:SERIES.get(key,key))
    if 'flight' in data and data.flight.nunique()>1:data['series']=data.flight+' · '+data.series
    xtype='T' if xkey=='date' else 'Q' if xkey in ('quantile','cutoff_dtd') else 'N'
    x=alt.X(f'{xkey}:{xtype}',title={'date':'Дата','period':'Период','quantile':'Квантиль бюджета','method':'Метод','label':None}.get(xkey,xkey))
    if xkey=='period':x=alt.X('period:N',sort=PERIODS,title='Период')
    base=alt.Chart(data).encode(x=x,y=alt.Y('value:Q',title=None),
        color=alt.Color('series:N',title=None,scale=alt.Scale(range=COLORS)),
        tooltip=[alt.Tooltip(f'{xkey}:{xtype}',title='Период / вариант'),alt.Tooltip('series:N',title='Показатель'),alt.Tooltip('value:Q',title='Значение',format=',.2f')])
    chart=base.mark_line(strokeWidth=2.4) if kind=='line' else base.mark_bar(cornerRadiusTopLeft=3,cornerRadiusTopRight=3).encode(xOffset='series:N')
    return _style(chart)


def render_charts(charts,key_prefix='research_chart'):
    if isinstance(charts,dict):items=list(charts.items())
    elif isinstance(charts,list):items=[(str(i),chart) for i,chart in enumerate(charts)]
    else:return
    for index,(name,value) in enumerate(items):
        if isinstance(value,list):chart=_known_chart(name,value);title=TITLES.get(name,name.replace('_',' '))
        elif isinstance(value,dict):
            data=pd.DataFrame(value.get('data') or value.get('rows') or [])
            xkey=value.get('x_key') or value.get('x')
            keys=value.get('y_keys') or value.get('series') or []
            keys=[key if isinstance(key,str) else key.get('key') or key.get('y') for key in keys]
            chart=_generic(data,xkey,keys,value.get('type','bar')) if not data.empty else None
            title=value.get('title',TITLES.get(name,'График'))
        else:continue
        if chart is not None:
            with st.container(border=True,key=f'{key_prefix}_panel_{index}'):
                st.subheader(title)
                st.altair_chart(chart,width='stretch',key=f'{key_prefix}_{index}',alt=title)


render_research_charts=render_charts
