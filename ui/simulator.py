"""Native Streamlit forms, charts, journal and exports for the Python simulator."""
from copy import deepcopy
from datetime import date
import math
import secrets

import pandas as pd
import streamlit as st

from backend.exports import serialize_export
from backend.service import MODEL_DEFAULTS
from backend.simulation import DEFAULTS
from .charts import comparison_curve,period_chart,fare_chart,reason_chart

PROFILES={'balanced':'Сбалансированный','early':'Ранние бронирования','late':'Спрос перед вылетом','notebook':'Профиль примера куратора'}
METHODS={'v3':'v3 · цена и доступность','baseline':'Базовый · доли покупок'}
TOTAL_MODES={'fixed_design':'N задано до генерации','observed_total':'N наблюдено как итог'}
OUTCOMES={'success':'Покупки','cancel':'Отмены','waste':'Без покупки'}
EXPORTS={'bookings':'Журнал bookings · CSV','reconstructed':'Реконструкция · CSV',
         'events':'Все события · CSV','daily':'Ежедневный журнал · CSV',
         'periods':'Оценки и интервалы · CSV','params':'Параметры · JSON'}
AMOUNT_SOURCES={'synthetic_truth':'Синтетический факт','not_observed':'Неизвестен','not_available':'Неизвестен',
    'not_estimated':'Не оценивается','model_conditional_given_success':'Модель: покупка',
    'model_conditional_budget':'Модель: бюджет','model_conditional_availability':'Модель: доступность'}


def _num(value):return f'{value:,.0f}'.replace(',',' ')


def money(value):
    if abs(value)>=1_000_000:return f'{value/1_000_000:.2f}'.replace('.',',')+' млн ₽'
    return _num(value)+' ₽'


def read_uploads(uploaded):
    """Decode uploaded bytes in memory; no browser or local paths are evaluated."""
    answer=[];total=0
    for file in uploaded or []:
        raw=file.getvalue();total+=len(raw)
        if total>10*1024*1024:raise ValueError('Общий размер выбранных CSV превышает 10 МБ.')
        try:text=raw.decode('utf-8-sig')
        except UnicodeDecodeError:raise ValueError(f'{file.name}: сохраните CSV в кодировке UTF-8.') from None
        answer.append({'name':file.name,'text':text})
    return answer


def _seed_widgets(payload,force=False):
    params={**DEFAULTS,**payload.get('params',{})};model={**MODEL_DEFAULTS,**payload.get('model',{})}
    values={**{'simulator_p_'+key:value for key,value in params.items()},
            **{'simulator_m_'+key:value for key,value in model.items() if key in MODEL_DEFAULTS},
            'simulator_source':payload.get('mode','simulation')}
    values['simulator_p_departure']=date.fromisoformat(str(params['departure']))
    for key,value in values.items():
        if force or key not in st.session_state:st.session_state[key]=value
    if force:
        st.session_state['simulator_result_flight']=payload.get('selected_flight') or params['flight']
        st.session_state['simulator_upload_revision']=st.session_state.get('simulator_upload_revision',0)+1
        for field in ('simulator_clear_bookings','simulator_clear_offers','simulator_clear_calibration'):
            st.session_state[field]=False


def _reset(runtime):
    payload={'mode':'simulation','params':dict(DEFAULTS),'model':dict(MODEL_DEFAULTS),'files':[],'offers_files':[],'reference_files':[]}
    _seed_widgets(payload,force=True)
    runtime.calculate(payload)


def _new_seed(runtime):
    payload=runtime.get_payload();payload['mode']='simulation'
    payload['params']={**DEFAULTS,**payload.get('params',{}),'seed':secrets.randbits(32)}
    _seed_widgets(payload,force=True);runtime.calculate(payload)


def _controls(runtime):
    payload=runtime.get_payload()
    with st.container(border=True,key='simulator_controls'):
        st.subheader('Параметры сценария')
        st.button('Сбросить параметры',on_click=_reset,args=(runtime,),key='simulator_reset',width='stretch')
        mode=st.radio('Источник данных',['simulation','import'],format_func=lambda value:{'simulation':'Симуляция','import':'Загрузить CSV'}[value],horizontal=True,key='simulator_source')
        with st.form('simulator_form',border=False):
            is_import=mode=='import';revision=st.session_state.get('simulator_upload_revision',0)
            files=offers=None
            if is_import:
                st.caption('Загрузите один или несколько журналов. N задаётся отдельно для каждого рейса.')
                files=st.file_uploader('Журналы bookings',type=['csv'],accept_multiple_files=True,key=f'simulator_bookings_upload_{revision}')
                if payload.get('files'):
                    st.caption('Сохранённые bookings: '+', '.join(file['name'] for file in payload['files']))
                    st.checkbox('Очистить сохранённые bookings',key='simulator_clear_bookings')
            st.caption('01 · РЕЙС И СПРОС')
            st.text_input('Рейс',key='simulator_p_flight',max_chars=40,disabled=is_import)
            st.date_input('Дата вылета',key='simulator_p_departure')
            st.number_input('Первичный спрос N, мест',min_value=1,max_value=10000,step=10,key='simulator_p_demand',help='Известное извне число первичных мест до отмен. Точная модель v3 ограничена N ≤ 1500.')
            col1,col2=st.columns(2)
            col1.number_input('Всего мест',min_value=1,max_value=2000,step=1,key='simulator_p_capacity')
            col2.number_input('Из них бизнес',min_value=0,max_value=2000,step=1,key='simulator_p_businessCapacity',disabled=is_import)
            st.selectbox('Профиль обращений',list(PROFILES),format_func=PROFILES.get,key='simulator_p_profile',disabled=is_import)
            with st.expander('02 · Цена и поведение',expanded=not is_import):
                st.number_input('Стартовый тариф Y, ₽',min_value=100,max_value=1000000,step=500,key='simulator_p_price',disabled=is_import)
                st.number_input('Медианный бюджет Y, ₽',min_value=100,max_value=1000000,step=500,key='simulator_p_budget',help='В журнале без покупок это внешнее предположение бюджета.')
                st.number_input('Рост тарифа к вылету, %',min_value=0,max_value=300,step=5,key='simulator_p_growth',disabled=is_import)
                st.number_input('Вероятность отмены, %',min_value=0,max_value=100,step=1,key='simulator_p_cancel',disabled=is_import)
                st.number_input('Доля спроса в бизнес, %',min_value=0,max_value=100,step=1,key='simulator_p_businessShare',disabled=is_import)
                st.number_input('Seed',min_value=0,max_value=4294967295,step=1,key='simulator_p_seed')
                st.caption('Для бизнеса цена и бюджет ×3. Группа покупает целиком. Отмена возвращает стоимость покупки.')
            with st.expander('03 · Математическая модель',expanded=True):
                st.selectbox('Метод',list(METHODS),format_func=METHODS.get,key='simulator_m_method')
                st.selectbox('Постановка общего N',list(TOTAL_MODES),format_func=TOTAL_MODES.get,key='simulator_m_total_mode')
                st.number_input('Разброс бюджетов σ',min_value=.1,max_value=1.5,step=.05,key='simulator_m_sigma')
                st.number_input('Вес предположений',min_value=0.,max_value=20.,step=.1,key='simulator_m_prior_strength')
                with st.expander('Дополнительные предположения'):
                    st.number_input('Временной эффект β',min_value=-2.,max_value=2.,step=.05,key='simulator_m_beta_time')
                    st.number_input('Эффект размера группы β',min_value=-2.,max_value=2.,step=.05,key='simulator_m_beta_qty')
                    st.checkbox('Смесь двух распределений бюджетов',key='simulator_m_mixture')
                if payload.get('model',{}).get('calibration'):
                    st.caption('Используются параметры с сохранённым происхождением обучающих данных.')
                    st.checkbox('Убрать независимую калибровку',key='simulator_clear_calibration')
            if is_import:
                offers=st.file_uploader('История предложений offers · необязательно',type=['csv'],accept_multiple_files=True,key=f'simulator_offers_upload_{revision}')
                if payload.get('offers_files'):
                    st.caption('Сохранённые offers: '+', '.join(file['name'] for file in payload['offers_files']))
                    st.checkbox('Очистить сохранённые offers',key='simulator_clear_offers')
                st.caption('Цены, открытость и остаток мест по интервалам UTC. Без offers используются явно отмеченные приближения.')
            submitted=st.form_submit_button('Рассчитать сценарий',type='primary',width='stretch')
        if mode=='simulation':st.button('Новый случайный сценарий',on_click=_new_seed,args=(runtime,),key='simulator_new_seed',width='stretch')
        st.caption('Один seed сохраняет одинаковые первичные обращения при сравнении цены и вместимости.')
    if submitted:
        try:
            params={key:st.session_state['simulator_p_'+key] for key in DEFAULTS}
            params['departure']=params['departure'].isoformat()
            model={**payload.get('model',{}),**{key:st.session_state['simulator_m_'+key] for key in MODEL_DEFAULTS}}
            if st.session_state.get('simulator_clear_calibration'):model.pop('calibration',None)
            selected_files=[] if st.session_state.get('simulator_clear_bookings') else read_uploads(files) if files else payload.get('files',[])
            selected_offers=[] if st.session_state.get('simulator_clear_offers') else read_uploads(offers) if offers else payload.get('offers_files',[])
            request={'mode':mode,'params':params,'model':model,'files':selected_files,'offers_files':selected_offers,
                     'reference_files':payload.get('reference_files',[]),'selected_flight':payload.get('selected_flight')}
            with st.spinner('Python рассчитывает сценарий и восстановление спроса…'):
                response=runtime.calculate(request)
            if not response.get('ok'):st.error(response.get('error','Расчёт не завершён.'))
            elif any(st.session_state.get('simulator_clear_'+name) for name in ('calibration','bookings','offers')):
                st.session_state['pending_loaded_payload']=runtime.get_payload()
                st.rerun()
        except (ValueError,TypeError) as exc:st.error(str(exc))


def journal_frame(result,view='events',outcome='all'):
    data=result.get('reconstructed' if view=='reconstructed' else 'events',[])
    if outcome!='all':data=[row for row in data if row.get('outcome')==outcome]
    data=sorted(data,key=lambda row:(row.get('timestamp') is None,str(row.get('timestamp') or '')))
    rows=[]
    for row in data:
        rows.append({'Время · UTC':str(row.get('timestamp') or '—').replace('T',' ').replace('Z','')[:19],
            'Результат':{'success':'Покупка','cancel':'Отмена','waste':'Без покупки'}.get(row.get('outcome'),row.get('outcome')),
            'Салон / RBD':' / '.join(str(v) for v in (row.get('cabin'),row.get('rbd')) if v) or '—',
            'Мест':row.get('qty'),'Цена, ₽':row.get('price'),'Бюджет / место, ₽':row.get('amount'),
            'Источник бюджета':AMOUNT_SOURCES.get(row.get('amount_source'),'Модель' if row.get('amount') is not None else 'Неизвестен'),
            'Источник / причина':row.get('reason') or row.get('source_file') or '—',
            'Файл':row.get('source_file'),'Строка':row.get('source_row')})
    return pd.DataFrame(rows)


def _move_page(amount):st.session_state['simulator_journal_page']=max(0,st.session_state.get('simulator_journal_page',0)+amount)


def _journal(result):
    with st.container(border=True,key='simulator_journal'):
        st.subheader('Журнал событий')
        columns=st.columns([1,1,1])
        view=columns[0].selectbox('Вид журнала',['events','reconstructed'],format_func=lambda value:{'events':'Все события','reconstructed':'Реконструкция'}[value],key='simulator_journal_view')
        outcome=columns[1].selectbox('Результат операции',['all',*OUTCOMES],format_func=lambda value:'Все результаты' if value=='all' else OUTCOMES[value],key='simulator_journal_outcome')
        page_size=columns[2].selectbox('Строк на странице',[10,25,50],key='simulator_journal_size')
        if view=='reconstructed':st.caption('Покупки и отмены взяты из bookings. Бюджеты и скрытые обращения являются оценками модели; ID технические.')
        else:st.caption('Синтетические обращения и отмены.' if result['mode']=='simulation' else 'Наблюдаемые операции из bookings; бюджет покупателя неизвестен.')
        frame=journal_frame(result,view,outcome);pages=max(1,math.ceil(len(frame)/page_size))
        page=min(st.session_state.get('simulator_journal_page',0),pages-1);st.session_state['simulator_journal_page']=page
        start=page*page_size
        st.dataframe(frame.iloc[start:start+page_size],hide_index=True,width='stretch',
            column_config={'Цена, ₽':st.column_config.NumberColumn(format='%.0f ₽'),'Бюджет / место, ₽':st.column_config.NumberColumn(format='%.0f ₽')})
        nav=st.columns([1,2,1]);nav[0].button('← Назад',disabled=page==0,on_click=_move_page,args=(-1,),key='simulator_prev')
        nav[1].caption(f'Показано {start+1 if len(frame) else 0}–{min(start+page_size,len(frame))} из {len(frame)} · {page+1}/{pages}')
        nav[2].button('Вперёд →',disabled=page==pages-1,on_click=_move_page,args=(1,),key='simulator_next')


def _exports(result):
    with st.expander('Скачать результаты',expanded=False):
        choices=dict(EXPORTS)
        try:serialize_export(result,'offers');choices['offers']='История предложений · CSV'
        except ValueError:pass
        # Streamlit resets a selectbox when an unavailable prior option disappears.
        if st.session_state.get('simulator_export_kind') not in choices:st.session_state['simulator_export_kind']='bookings'
        kind=st.selectbox('Файл для скачивания',list(choices),format_func=choices.get,key='simulator_export_kind')
        body,filename,mime=serialize_export(result,kind)
        st.download_button('Скачать файл',data=body,file_name=filename,mime=mime,type='primary',on_click='ignore',key='simulator_download')
        st.caption('CSV формирует Python. Реконструкция содержит пометки происхождения каждой строки и оценки бюджета.')


def _save_comparison(result):
    st.session_state['simulator_comparison']=deepcopy({key:result[key] for key in ('params','summary','curve','mode')})


def _results(runtime):
    response=runtime.last_calculation
    if not response or not response.get('ok') or not response.get('results'):
        st.info('Настройте параметры и нажмите «Рассчитать сценарий».');return
    flights=list(response['results'])
    if st.session_state.get('simulator_result_flight') not in flights:
        st.session_state['simulator_result_flight']=response.get('selected_flight') if response.get('selected_flight') in flights else flights[0]
    flight=st.selectbox('Рейс результата',flights,key='simulator_result_flight',label_visibility='collapsed')
    result=response['results'][flight];params=result['params'];summary=result['summary']
    # The selected flight is the context used by native research tools.
    if st.session_state.get('simulator_payload') is not None:st.session_state['simulator_payload']['selected_flight']=flight
    source='Синтетический сценарий' if result['mode']=='simulation' else 'Журнал bookings'
    st.caption(f'{source} · {flight} → {params["departure"]} · seed {params["seed"]} · N={params["demand"]}')
    if response.get('file_summary') and result['mode']=='import':st.caption(response['file_summary'])
    for index,(title,value,foot) in enumerate([
        ('Места на рейсе',f'{_num(summary["net"])} / {_num(params["capacity"])}',f'Загрузка {summary["load"]:.1f}%'),
        ('Куплено до отмен',_num(summary['gross']),f'Отменено {_num(summary["canceled"])} мест'),
        ('Без покупки · факт' if result['mode']=='simulation' else 'Без покупки · оценка',_num(summary['waste']),f'{summary["waste"]/params["demand"]*100:.1f}% первичного спроса'),
        ('Выручка после возвратов',money(summary['revenue']),'Без учёта расходов и налогов')]):
        if index==0:metric_columns=st.columns(4)
        with metric_columns[index].container(border=True,key=f'simulator_metric_{index}'):
            st.metric(title,value);st.caption(foot)
    diagnostic=result.get('diagnostics',{})
    with st.container(border=True,key='simulator_diagnostics'):
        st.subheader(diagnostic.get('title','Математическая модель'))
        if result['mode']=='simulation':
            metric=diagnostic.get('WAPE_pct');mae=diagnostic.get('MAE_seats')
            if metric is not None and mae is not None:st.caption(f'Синтетическая проверка: WAPE {metric:.1f}% · MAE {mae:.1f} места по периодам')
        else:st.caption('Точность скрытого спроса можно проверить отдельно по соответствующему passengers. Бюджеты и отказы — оценки модели.')
        with st.expander('Допущения и ограничения'):
            for warning in diagnostic.get('warnings',[]):st.write(warning)
    comparison=st.session_state.get('simulator_comparison')
    toolbar=st.columns(2)
    toolbar[0].button('Сохранить для сравнения',on_click=_save_comparison,args=(result,),key='simulator_save_comparison',width='stretch')
    if comparison:
        if toolbar[1].button('Убрать сравнение',key='simulator_remove_comparison',width='stretch'):
            st.session_state.pop('simulator_comparison',None);st.rerun()
        with st.container(border=True,key='simulator_comparison_summary'):
            st.caption(f'Изменение к сохранённому сценарию {comparison["params"]["flight"]} · seed {comparison["params"]["seed"]}')
            comp=st.columns(3)
            comp[0].metric('Баланс мест',_num(summary['net']),f'{summary["net"]-comparison["summary"]["net"]:+d}')
            comp[1].metric('Покупки до отмен',_num(summary['gross']),f'{summary["gross"]-comparison["summary"]["gross"]:+d}')
            comp[2].metric('Выручка',money(summary['revenue']),money(summary['revenue']-comparison['summary']['revenue']),delta_color='off')
    with st.container(border=True,key='simulator_curve'):
        st.subheader('Как заполняется рейс')
        st.caption('Накопленные места по дням до вылета. Пунктир — вместимость.')
        chart=comparison_curve(result,comparison)
        if chart is not None:st.altair_chart(chart,width='stretch',alt='Покупки и баланс мест до вылета',key='simulator_curve_chart')
    with st.container(border=True,key='simulator_periods'):
        st.subheader('Спрос по периодам')
        view=st.radio('Вид распределения',['demand','estimate'],format_func=lambda value:{'demand':'Спрос','estimate':'Оценка и эталон'}[value],horizontal=True,key='simulator_period_view')
        chart=period_chart(result,view)
        if chart is not None:st.altair_chart(chart,width='stretch',alt='Покупки и скрытый спрос по периодам',key='simulator_period_chart')
        if view=='estimate' and result['mode']=='import':st.caption('Эталон отсутствует. Для проверки загрузите passengers в разделе исследования.')
    with st.container(border=True,key='simulator_reasons'):
        st.subheader('За пределами покупки')
        st.caption('Причины отказа · синтетический факт' if result['mode']=='simulation' else 'Возможное объяснение модели; причины отказа не наблюдались в bookings.')
        chart=reason_chart(result)
        if chart is not None:st.altair_chart(chart,width='stretch',key='simulator_reason_chart',alt='Причины отсутствия покупки')
        else:st.caption('Выбранный метод оценивает объём скрытого спроса без распределения причин.')
    with st.container(border=True,key='simulator_fares'):
        st.subheader('Тарифные классы')
        cabin=st.selectbox('Салон',['Все салоны','Y','C'],format_func=lambda value:{'Y':'Эконом · Y','C':'Бизнес · C'}.get(value,value),key='simulator_cabin')
        chart=fare_chart(result,cabin)
        if chart is not None:st.altair_chart(chart,width='stretch',key='simulator_fare_chart',alt='Покупки и отмены по RBD')
    _journal(result);_exports(result)


def render_simulator(runtime):
    pending=st.session_state.pop('pending_loaded_payload',None)
    _seed_widgets(pending or runtime.get_payload(),force=pending is not None)
    if runtime.last_calculation is None:
        with st.spinner('Подготовка начального сценария…'):runtime.calculate(runtime.get_payload())
    st.caption('АВИАЦИОННАЯ АНАЛИТИКА / СЦЕНАРНОЕ МОДЕЛИРОВАНИЕ')
    st.title('Симулятор спроса.')
    st.caption('От первого обращения до последнего места на рейсе.')
    controls,results=st.columns([1.05,2.7],gap='large')
    with controls:_controls(runtime)
    with results:
        with st.container(key='simulator_results'):_results(runtime)
