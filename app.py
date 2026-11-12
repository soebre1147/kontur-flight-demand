"""Контур: native Python/Streamlit application."""
import json
import os
from pathlib import Path

import streamlit as st

from backend.jobs import JobManager
from backend.native_runtime import NativeRuntime
from backend.service import calculate
from backend.storage import RunStore,code_fingerprint
from ui.theme import apply_theme,render_brand,render_footer
from ui.simulator import render_simulator
from ui.research import render_research
from ui.history import render_history

ROOT=Path(__file__).resolve().parent
PAGES=('Симулятор','Исследование','История')


@st.cache_data(max_entries=16,ttl=3600,show_spinner=False)
def cached_calculation(serialized):
    return calculate(json.loads(serialized))


@st.cache_resource
def resources(data_directory,fingerprint):
    store=RunStore(Path(data_directory)/'history.sqlite3',fingerprint)
    return store,JobManager(store)


def main():
    st.set_page_config(page_title='Контур · Лаборатория спроса',page_icon='✳',layout='wide')
    apply_theme();render_brand()
    directory=Path(os.environ.get('KONTUR_DATA_DIR',str(ROOT/'runtime'))).resolve()
    store,jobs=resources(str(directory),code_fingerprint(ROOT))
    runtime=NativeRuntime(ROOT,store,jobs,st.session_state,
        lambda payload:cached_calculation(json.dumps(payload,ensure_ascii=False,sort_keys=True,allow_nan=False)))
    pending=st.session_state.pop('pending_page',None)
    if pending in PAGES: st.session_state['native_page']=pending
    if st.session_state.get('native_page') not in PAGES: st.session_state['native_page']=PAGES[0]
    navigation,about=st.columns([5,1],vertical_alignment='center')
    with navigation:
        page=st.segmented_control('Раздел приложения',PAGES,key='native_page',required=True,label_visibility='collapsed')
    with about:
        with st.popover('О модели',width='stretch'):
            st.subheader('От bookings к оценке спроса')
            st.write('Математическая модель использует поток групп, распределение бюджетов, цены и доступность мест. Общий первичный спрос N задаётся извне.')
            st.write('Покупки и отмены наблюдаются в bookings. Скрытые обращения, бюджеты и причины отказов являются модельными оценками. Выручка после возвратов не равна прибыли без учёта расходов.')
            st.caption('Условные интервалы при оценённых параметрах не отражают полную неопределённость. В исследовательском разделе доступны независимая проверка, чувствительность и повторные оценки.')
    if page=='Исследование': render_research(runtime)
    elif page=='История': render_history(runtime)
    else: render_simulator(runtime)
    render_footer()


if __name__=='__main__': main()
