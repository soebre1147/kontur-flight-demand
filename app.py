"""Контур: the original frontend hosted as a Streamlit custom component.

All scenario generation, input validation and inverse-model inference run in
backend.service. The component renders data and emits user actions only.
"""
from pathlib import Path
import json
import logging
import uuid

import streamlit as st
import streamlit.components.v1 as components

from backend.service import calculate, DEFAULTS
from backend.exports import serialize_export
from backend.controller import Controller
from backend.jobs import JobManager
from backend.storage import RunStore,code_fingerprint
from backend.reporting import build_report,build_run_archive

ROOT=Path(__file__).resolve().parent
st.set_page_config(page_title='Контур · Лаборатория спроса',page_icon='✳',layout='wide',initial_sidebar_state='collapsed')
st.markdown('''<style>
header[data-testid="stHeader"], [data-testid="stToolbar"], #MainMenu, footer {display:none !important;}
[data-testid="stAppViewContainer"] {background:#f6f8f6;}
[data-testid="stMainBlockContainer"], .block-container {padding:0 !important;max-width:100% !important;}
[data-testid="stMain"] {padding:0 !important;}
[data-testid="stVerticalBlock"] {gap:0 !important;}
iframe {border:0;display:block;}
.st-key-kontur_downloads,.st-key-kontur_reports {margin:20px 28px 32px !important;width:calc(100% - 56px) !important;}
</style>''',unsafe_allow_html=True)

component=components.declare_component('kontur_python_frontend',path=str(ROOT/'dist'))


@st.cache_data(max_entries=16,ttl=3600,show_spinner=False)
def cached_calculation(serialized):
    return calculate(json.loads(serialized))


@st.cache_resource
def research_runtime():
    store=RunStore(ROOT/'runtime'/'history.sqlite3',code_fingerprint(ROOT))
    jobs=JobManager(store)
    controller=Controller(store,jobs,lambda value:cached_calculation(json.dumps(value,ensure_ascii=False,sort_keys=True,allow_nan=False)))
    return store,jobs,controller


store,jobs,controller=research_runtime()


def process(payload):
    request_id=payload.get('request_id','') if isinstance(payload,dict) else ''
    try:
        if 'kontur_context' not in st.session_state:
            st.session_state.kontur_context={field:st.session_state.get('kontur_'+field,[]) for field in ('files','offers_files','reference_files')}
        return controller.handle(payload,st.session_state.kontur_context)
    except (ValueError,TypeError) as exc:
        return {'ok':False,'request_id':request_id,'action':payload.get('action','calculate') if isinstance(payload,dict) else 'calculate','error':str(exc)}
    except Exception:
        logging.exception('Calculation failed')
        return {'ok':False,'request_id':request_id,'action':payload.get('action','calculate') if isinstance(payload,dict) else 'calculate','error':'Расчёт не завершён. Проверьте данные и параметры; подробности записаны в журнал сервера.'}


if 'kontur_session' not in st.session_state:
    st.session_state.kontur_session=str(uuid.uuid4())
    initial=process({'request_id':'initial','mode':'simulation','params':dict(DEFAULTS),
        'model':{'method':'v3','total_mode':'fixed_design','sigma':.45,'prior_strength':.5},
        'files':[],'offers_files':[]})
    st.session_state.kontur_initial=initial
    st.session_state.kontur_last_good=initial
    st.session_state.kontur_response=initial
    st.session_state.kontur_handled='initial'

if 'kontur_last_good' not in st.session_state:
    candidate=st.session_state.get('kontur_response',{})
    st.session_state.kontur_last_good=candidate if candidate.get('ok') else st.session_state.kontur_initial

if not st.session_state.kontur_initial.get('ok'):
    st.error(st.session_state.kontur_initial.get('error','Не удалось рассчитать начальный сценарий.'))

request=component(initial_response=st.session_state.kontur_last_good,
                  response=st.session_state.kontur_response,
                  session_key=st.session_state.kontur_session,
                  key='kontur_interface',default=None)
if isinstance(request,dict):
    request_id=request.get('request_id')
    if request_id and request_id!=st.session_state.kontur_handled:
        st.session_state.kontur_handled=request_id
        st.session_state.kontur_response=process(request)
        if st.session_state.kontur_response.get('ok') and st.session_state.kontur_response.get('results'):
            st.session_state.kontur_last_good=st.session_state.kontur_response
        if st.session_state.kontur_response.get('lab'):
            st.session_state.kontur_last_lab=st.session_state.kontur_response['lab']
            st.session_state.kontur_last_lab_run=st.session_state.kontur_response.get('run_id') or st.session_state.kontur_response.get('job',{}).get('id')
        st.rerun()

# Official server-backed downloads also work in embedded browsers which block
# Blob downloads from an iframe. Use the last successful calculation only.
last=st.session_state.kontur_last_good
if last.get('ok') and last.get('results'):
    with st.container(key='kontur_downloads'):
        with st.expander('Скачать результаты через Streamlit'):
            st.caption('Последний завершённый расчёт. Выберите рейс и файл.')
            flights=list(last['results'])
            flight=st.selectbox('Рейс для экспорта',flights,key='export_flight')
            export_result=last['results'][flight]
            choices={'bookings':'Журнал bookings · CSV','reconstructed':'Реконструкция · CSV',
                     'events':'Все события · CSV','daily':'Ежедневный журнал · CSV',
                     'periods':'Оценки и интервалы · CSV','params':'Параметры сценария · JSON'}
            try:
                serialize_export(export_result,'offers')
                choices['offers']='Предложения · CSV'
            except ValueError:
                pass
            kind=st.selectbox('Файл',list(choices),format_func=choices.get,key='export_kind')
            try:
                content,filename,mime=serialize_export(export_result,kind)
                st.download_button('Скачать файл',content,file_name=filename,mime=mime,
                                   on_click='ignore',type='primary',key='server_download')
            except ValueError as exc:
                st.info(str(exc))

lab_id=st.session_state.get('kontur_last_lab_run')
if lab_id:
    with st.container(key='kontur_reports'):
        with st.expander('Отчёт и данные исследования'):
            try:
                record=store.get(lab_id,with_payload=True)
                st.caption(record['title']+' · '+record['created_at']+' · '+record['status'])
                st.download_button('Скачать отчёт HTML',build_report(record),file_name='kontur-report-'+lab_id[:8]+'.html',mime='text/html',on_click='ignore',key='report_html')
                st.download_button('Скачать данные и конфигурацию ZIP',build_run_archive(record),file_name='kontur-run-'+lab_id[:8]+'.zip',mime='application/zip',on_click='ignore',key='report_zip')
                st.caption('ZIP содержит таблицы CSV, конфигурацию с версией кода и исходные загруженные файлы. История хранится локально в runtime/history.sqlite3.')
            except ValueError as exc:
                st.info(str(exc))
