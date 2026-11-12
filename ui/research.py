"""Native Streamlit research forms, background progress and result reports."""
from copy import deepcopy
import json
import math
import re

import streamlit as st

from backend.jobs import TITLES
from backend.reporting import build_report,build_run_archive
from backend.scenarios import SCENARIOS
from ui.charts import render_charts
from ui.presentation import STATUS_LABELS,label,render_summary,render_tables

RUNNING={'queued','running','cancelling'}
TOOL_NAMES={'evaluate':'Проверка по passengers','experiments':'Пакетные эксперименты','calibration':'Калибровка',
            'sensitivity':'Устойчивость','forecast':'Прогноз до вылета','policies':'Тарифные политики'}
DESCRIPTIONS={
    'evaluate':'Сопоставление распределений после восстановления. Эталон недоступен самой модели.',
    'experiments':'v3 и базовый метод получают одинаковые журналы. Неудачные оценки остаются в отчёте.',
    'calibration':'Подбор предположений на обучающих потоках и проверка на других seed.',
    'sensitivity':'Повторная оценка при изменении предположений и внешнего объёма спроса.',
    'forecast':'В оценку входят только данные, доступные на выбранный день. Будущее используется для проверки.',
    'policies':'Сравнение правил на одном потоке пассажиров, с отдельными seed для подбора и проверки.'}


def parse_grid(text,minimum,maximum,*,integer=False,optional=False):
    if not str(text).strip():
        if optional: return []
        raise ValueError('Укажите хотя бы одно значение сетки.')
    try: values=[float(value) for value in re.split(r'[;,\s]+',str(text).strip()) if value]
    except ValueError: raise ValueError('Используйте числа с точкой и ; между значениями.') from None
    if any(not math.isfinite(value) or not minimum<=value<=maximum or integer and not value.is_integer() for value in values):
        raise ValueError(f'Значения сетки должны быть от {minimum} до {maximum}'+(' и целыми.' if integer else '.'))
    return list(dict.fromkeys(int(value) if integer else value for value in values))


def uploaded_files(files):
    """Decode browser uploads; no supplied local path is ever opened."""
    answer=[];size=0
    for upload in files or []:
        data=upload.getvalue();size+=len(data)
        if size>10*1024*1024: raise ValueError('Общий размер эталонных CSV не должен превышать 10 МБ.')
        try: text=data.decode('utf-8-sig')
        except UnicodeDecodeError: raise ValueError(f'{upload.name}: сохраните CSV в UTF-8.') from None
        answer.append({'name':upload.name,'text':text})
    return answer


def _restore_settings():
    pending=st.session_state.pop('pending_research_settings',None)
    if not pending: return
    st.session_state['research_upload_revision']=st.session_state.get('research_upload_revision',0)+1
    action=pending.get('action');settings=pending.get('research') or {}
    if action in TOOL_NAMES: st.session_state['research_tool']=action
    mappings={
        'experiments':{'scenarios':'research_scenarios','runs':'research_runs','seed_start':'research_seed'},
        'calibration':{'training_runs':'research_cal_train','validation_runs':'research_cal_validation','seed_start':'research_cal_seed',
                       'sigma_values':'research_cal_sigma','beta_time_values':'research_cal_time','beta_qty_values':'research_cal_qty',
                       'compare_mixture':'research_cal_mixture'},
        'sensitivity':{'sigma_values':'research_sens_sigma','prior_values':'research_sens_prior','demand_values':'research_sens_n','bootstrap_runs':'research_sens_boot'},
        'policies':{'scenario':'research_policy_scenario','objective':'research_policy_objective','training_runs':'research_policy_train',
                    'validation_runs':'research_policy_validation','seed_start':'research_policy_seed','min_load':'research_policy_load',
                    'variable_cost':'research_policy_variable','fixed_cost':'research_policy_fixed'},
    }
    for source,target in mappings.get(action,{}).items():
        if source in settings and settings[source] is not None:
            value=settings[source]
            if target in ('research_cal_sigma','research_cal_time','research_cal_qty','research_sens_sigma','research_sens_prior','research_sens_n'):
                value='; '.join(map(str,value)) if isinstance(value,list) else str(value)
            st.session_state[target]=value
    if action=='forecast':
        cuts=settings.get('cutoffs') or [settings.get('cutoff_dtd',30)]
        st.session_state['research_forecast_cut']='Все четыре среза' if len(cuts)==4 else int(cuts[0])
        if 'forecast_runs' in settings: st.session_state['research_forecast_paths']=settings['forecast_runs']
    if action=='experiments': st.session_state['research_preset']='Настроить'
    if action=='policies':
        st.session_state['research_policy_costs_known']=all(settings.get(key) is not None for key in ('variable_cost','fixed_cost'))
    if action=='calibration' and settings.get('scenarios'):
        st.session_state['research_cal_scenarios']=settings['scenarios']


def render_research(runtime):
    _restore_settings()
    st.caption('ПРОВЕРКА И РАЗВИТИЕ МОДЕЛИ')
    heading,example=st.columns([3,1])
    with heading:
        st.title('Исследование спроса')
        st.write('Сравнивайте методы на одних данных и сохраняйте воспроизводимые результаты.')
    with example:
        if st.button('Проверка 10 × 100',key='research_benchmark',width='stretch'):
            runtime.load_benchmark();st.rerun()
    tool=st.selectbox('Инструмент исследования',list(TOOL_NAMES),format_func=TOOL_NAMES.get,key='research_tool')
    st.subheader(TOOL_NAMES[tool]);st.caption(DESCRIPTIONS[tool])
    context=runtime.get_payload();p=context.get('params',{})
    synthetic=tool in ('experiments','calibration','policies')
    if synthetic:
        st.info(f'Независимые синтетические сценарии · N={p.get("demand",300)}. Загруженные журналы в этом инструменте не используются.')
    else:
        source='Загруженные bookings' if context.get('mode')=='import' else 'Синтетический сценарий'
        st.caption(f'{source} · {context.get("selected_flight") or p.get("flight","рейс")} · внешний N={p.get("demand",300)} · {p.get("departure","")}')
    if tool=='evaluate':
        reference=st.file_uploader('Эталонные passengers CSV · только для проверки',type=['csv'],accept_multiple_files=True,key='research_reference_upload_'+str(st.session_state.get('research_upload_revision',0)))
        current=context.get('reference_files') or []
        if current:
            st.caption('Сохранённый эталон: '+', '.join(file['name'] for file in current))
            if st.button('Убрать сохранённый эталон',key='research_clear_references'):
                runtime.set_uploads(reference_files=[])
                st.session_state['research_upload_revision']=st.session_state.get('research_upload_revision',0)+1
                st.rerun()
        st.caption('В симуляции используется известный факт генератора. Для импорта нужен passengers того же запуска; личности и ID не сопоставляются.')
    else: reference=None
    form=_render_form(tool)
    if form['submitted']:
        try:
            payload=runtime.get_payload()
            if reference:
                refs=uploaded_files(reference);runtime.set_uploads(reference_files=refs);payload['reference_files']=refs
            payload['research']=form['settings']
            response=runtime.submit(tool,payload)
            if response.get('ok'): st.rerun()
            st.error(response.get('error','Не удалось запустить исследование.'))
        except (ValueError,TypeError) as exc: st.error(str(exc))
    if tool=='experiments': _render_example(runtime)
    _job_monitor(runtime)
    if runtime.last_lab: render_lab(runtime,runtime.last_lab,runtime.last_lab_id)


def _render_form(tool):
    with st.form('research_form_'+tool,border=True):
        settings={}
        if tool=='experiments':
            preset=st.radio('Размер проверки',['Быстрая проверка','Настроить','Все 10 × 100'],horizontal=True,key='research_preset')
            choices=st.multiselect('Сценарии поведения',list(SCENARIOS),default=['balanced','early','late'],format_func=lambda key:SCENARIOS[key]['label'],key='research_scenarios')
            left,right=st.columns(2)
            runs=left.number_input('Запусков на сценарий',1,100,3,key='research_runs')
            seed=right.number_input('Начальный seed',0,4000000000,1000,key='research_seed')
            scenarios=list(SCENARIOS) if preset=='Все 10 × 100' else ['balanced','early','late'] if preset=='Быстрая проверка' else choices
            actual_runs=100 if preset=='Все 10 × 100' else 3 if preset=='Быстрая проверка' else runs
            st.caption(f'{len(scenarios)} сценариев × {actual_runs} запусков = {len(scenarios)*actual_runs} журналов; два метода на каждый журнал.')
            settings={'scenarios':scenarios,'runs':actual_runs,'seed_start':seed}
            text='Запустить сравнение'
        elif tool=='calibration':
            scenarios=st.multiselect('Обучающие сценарии',list(SCENARIOS),default=['balanced','mixed_wtp','late'],format_func=lambda key:SCENARIOS[key]['label'],key='research_cal_scenarios')
            a,b,c=st.columns(3)
            train=a.number_input('Обучающих запусков',1,20,3,key='research_cal_train')
            validation=b.number_input('Проверочных запусков',1,20,3,key='research_cal_validation')
            seed=c.number_input('Начальный seed',0,4000000000,2000,key='research_cal_seed')
            a,b,c=st.columns(3)
            sigmas=a.text_input('Значения σ через ;','0.35; 0.45; 0.65',key='research_cal_sigma')
            times=b.text_input('Эффект времени β через ;','0; -0.1',key='research_cal_time')
            quantities=c.text_input('Эффект размера группы β через ;','0',key='research_cal_qty')
            mixture=st.checkbox('Сравнить одну компоненту и смесь бюджетов',True,key='research_cal_mixture')
            st.caption('Не более 48 сочетаний параметров. Проверочные seed не участвуют в подборе. Применение выбранной конфигурации остаётся отдельным действием.')
            settings={'scenarios':scenarios,'training_runs':train,'validation_runs':validation,'seed_start':seed,
                      'sigma_values':sigmas,'beta_time_values':times,'beta_qty_values':quantities,'compare_mixture':mixture}
            text='Запустить калибровку'
        elif tool=='sensitivity':
            a,b=st.columns(2)
            sigmas=a.text_input('Значения σ через ;','0.3; 0.45; 0.6',key='research_sens_sigma')
            priors=b.text_input('Вес предположений через ;','0.1; 0.5; 1',key='research_sens_prior')
            a,b=st.columns(2)
            demands=a.text_input('Внешний N через ;',placeholder='Пусто — текущий N',key='research_sens_n')
            bootstrap=b.number_input('Повторных оценок bootstrap',0,30,0,key='research_sens_boot')
            st.caption('Диапазон чувствительности не является доверительным интервалом. Бутстрап повторно оценивает модель на выборках наблюдаемых покупок при фиксированных offers/N.')
            settings={'sigma_values':sigmas,'prior_values':priors,'demand_values':demands,'bootstrap_runs':bootstrap}
            text='Проверить устойчивость'
        elif tool=='forecast':
            cutoff=st.selectbox('День наблюдения',[90,30,14,7,'Все четыре среза'],index=1,format_func=lambda value:value if isinstance(value,str) else f'За {value} дней до вылета',key='research_forecast_cut')
            trajectories=st.number_input('Траекторий будущего',20,1000,200,20,key='research_forecast_paths')
            st.caption('Данные после среза используются только для проверки. Прошлый N задаётся внешним N и профилем; будущая цена и открытость продолжают последнее известное предложение.')
            settings={'cutoffs':[90,30,14,7] if isinstance(cutoff,str) else [cutoff],'forecast_runs':trajectories}
            text='Построить прогноз'
        elif tool=='policies':
            a,b=st.columns(2)
            scenario=a.selectbox('Сценарий пассажиров',list(SCENARIOS),format_func=lambda key:SCENARIOS[key]['label'],key='research_policy_scenario')
            objective=b.selectbox('Цель сравнения',['revenue','profit'],format_func=lambda value:'Выручка' if value=='revenue' else 'Прибыль с указанными расходами',key='research_policy_objective')
            a,b=st.columns(2)
            train=a.number_input('Запусков для выбора политики',1,30,5,key='research_policy_train')
            validation=b.number_input('Независимых проверочных запусков',2,100,10,key='research_policy_validation')
            a,b=st.columns(2)
            minimum=a.number_input('Минимальная средняя загрузка, %',0,100,0,key='research_policy_load')
            seed=b.number_input('Начальный seed',0,4000000000,3000,key='research_policy_seed')
            a,b=st.columns(2)
            variable=a.number_input('Расходы на занятое место, ₽',min_value=0.,value=0.,step=100.,key='research_policy_variable')
            fixed=b.number_input('Постоянные расходы рейса, ₽',min_value=0.,value=0.,step=1000.,key='research_policy_fixed')
            costs=st.checkbox('Расходы заданы и могут использоваться для прибыли',key='research_policy_costs_known')
            st.caption('12 кандидатов: динамическая цена и управление вложенными RBD. Подбор и проверка используют разные seed. Загрузка ограничивается в среднем при подборе.')
            settings={'scenario':scenario,'objective':objective,'training_runs':train,'validation_runs':validation,'seed_start':seed,'min_load':minimum,
                      'variable_cost':variable if costs else None,'fixed_cost':fixed if costs else None}
            text='Сравнить политики'
        else:
            st.caption('Сравниваются ошибки скрытого спроса, покрытие интервалов и распределения бюджетов, взвешенные по местам.')
            text='Проверить по passengers'
        submitted=st.form_submit_button(text,type='primary',width='stretch')
    if submitted:
        try: settings=normalise_settings(tool,settings)
        except ValueError as exc:
            st.error(str(exc));submitted=False
    return {'submitted':submitted,'settings':settings}


def normalise_settings(tool,settings):
    result=deepcopy(settings)
    if tool in ('experiments','calibration') and not result.get('scenarios'):
        raise ValueError('Выберите хотя бы один сценарий.')
    if tool=='calibration':
        for key,low,high in [('sigma_values',.1,1.5),('beta_time_values',-2,2),('beta_qty_values',-2,2)]:
            result[key]=parse_grid(result[key],low,high)
        candidates=len(result['sigma_values'])*len(result['beta_time_values'])*len(result['beta_qty_values'])*(2 if result['compare_mixture'] else 1)
        if candidates>48: raise ValueError('Для калибровки допустимо не более 48 сочетаний параметров.')
    if tool=='sensitivity':
        result['sigma_values']=parse_grid(result['sigma_values'],.1,1.5)
        result['prior_values']=parse_grid(result['prior_values'],0,20)
        result['demand_values']=parse_grid(result['demand_values'],1,1500,integer=True,optional=True)
        if len(result['sigma_values'])*len(result['prior_values'])*max(1,len(result['demand_values']))>60:
            raise ValueError('Для устойчивости допустимо не более 60 сочетаний параметров.')
    if tool=='policies' and result['objective']=='profit' and (result['variable_cost'] is None or result['fixed_cost'] is None):
        raise ValueError('Для прибыли отметьте, что расходы заданы, и укажите их значения.')
    return result


def _render_example(runtime):
    with st.expander('Сформировать полный комплект bookings + offers + passengers'):
        a,b=st.columns(2)
        scenario=a.selectbox('Сценарий комплекта',list(SCENARIOS),format_func=lambda key:SCENARIOS[key]['label'],key='research_example_scenario')
        seed=b.number_input('Seed комплекта',0,4000000000,42,key='research_example_seed')
        if st.button('Сформировать комплект',key='research_generate_example'):
            payload=runtime.get_payload();payload['research']={'scenario':scenario,'seed_start':seed}
            try:
                response=runtime.submit('generate_example',payload)
                if response.get('ok'): st.rerun()
                st.error(response.get('error','Не удалось сформировать комплект.'))
            except ValueError as exc: st.error(str(exc))


def _job_monitor(runtime):
    identifier=runtime.active_job_id
    if not identifier: return
    first=runtime.poll_job(identifier)
    active=first.get('job',{}).get('status') in RUNNING
    @st.fragment(run_every='2s' if active else None)
    def monitor():
        try: response=runtime.poll_job(identifier)
        except ValueError as exc: st.error(str(exc));return
        if not response.get('job'):
            st.error(response.get('error','Исследование не найдено.'));return
        job=response['job'];status=job['status']
        with st.container(border=True):
            st.caption('ВЫПОЛНЕНИЕ')
            st.write(f'**{STATUS_LABELS.get(status,status)} · {job.get("title","")}**')
            total=job.get('total') or 0;completed=job.get('completed') or 0
            st.progress(min(1.,completed/total) if total else 0.,text=f'{completed} / {total}' if total else 'Подготовка данных')
            st.caption(job.get('message',''))
            if status in RUNNING and st.button('Остановить расчёт',key='research_cancel_'+identifier):
                runtime.cancel_job(identifier);st.rerun()
            if status=='failed': st.error(job.get('error') or 'Расчёт завершился ошибкой. Последний корректный результат сохранён.')
            if status=='interrupted': st.info('Сервер был остановлен. Этот запуск можно повторить из истории.')
        if active and status not in RUNNING and st.session_state.get('research_finished_job')!=identifier:
            st.session_state['research_finished_job']=identifier;st.rerun()
    monitor()


def render_lab(runtime,result,run_id=None):
    st.divider();st.caption('РЕЗУЛЬТАТ ИССЛЕДОВАНИЯ')
    st.header(result.get('title','Результат исследования'))
    render_summary(result.get('summary',{}),key_prefix='research')
    config=result.get('selected_model')
    if config and st.button('Применить выбранные параметры к симулятору',key='research_apply_model',type='primary'):
        response=runtime.apply_model(config)
        if response.get('ok'): st.rerun()
        st.error(response.get('error','Не удалось применить параметры.'))
    warnings=result.get('warnings') or []
    if warnings:
        with st.expander('Допущения и ограничения исследования'):
            for warning in warnings: st.write('• '+str(warning))
    if result.get('ok') is False: st.error(result.get('error','Исследование завершено с ошибкой.'))
    render_charts(result.get('charts',{}),key_prefix='research_'+str(run_id or 'result'))
    render_tables(result.get('tables',{}),key_prefix='research_'+str(run_id or 'result'))
    _reports(runtime,result,run_id)


def _reports(runtime,result,run_id):
    with st.expander('Скачать отчёт и данные исследования'):
        record=runtime.store.get(run_id,with_payload=True) if run_id else {'title':result.get('title'),'result':result,'config':runtime.get_payload()}
        suffix=str(run_id or 'result')[:8]
        left,right=st.columns(2)
        left.download_button('Скачать отчёт HTML',build_report(record),file_name=f'kontur-report-{suffix}.html',mime='text/html',on_click='ignore',key='research_html_'+suffix)
        right.download_button('Скачать данные и конфигурацию ZIP',build_run_archive(record),file_name=f'kontur-run-{suffix}.zip',mime='application/zip',on_click='ignore',key='research_zip_'+suffix)
        st.download_button('Скачать результат JSON',json.dumps(result,ensure_ascii=False,allow_nan=False,indent=2).encode('utf-8'),file_name=f'kontur-result-{suffix}.json',mime='application/json',on_click='ignore',key='research_json_'+suffix)
        st.caption('ZIP содержит CSV-таблицы, конфигурацию, версию кода и исходные загруженные файлы. HTML-отчёт открывается самостоятельно.')
