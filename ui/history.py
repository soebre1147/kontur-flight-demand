"""Native Streamlit local history, configuration restore and exact replay."""
import streamlit as st

from backend.reporting import build_report,build_run_archive
from ui.presentation import STATUS_LABELS


def render_history(runtime):
    st.caption('ЛОКАЛЬНЫЕ ИССЛЕДОВАНИЯ')
    st.title('История запусков')
    st.write('Откройте сохранённый результат или повторите расчёт с прежними данными на текущей версии кода.')
    records=runtime.store.history(limit=500)
    if not records:
        st.info('Запустите симуляцию или исследование — параметры и результат появятся здесь.');return
    a,b=st.columns([3,1])
    with a: query=st.text_input('Поиск по названию',key='history_query',placeholder='Например, прогноз или полная проверка')
    with b:
        statuses=st.multiselect('Статус',list(STATUS_LABELS),format_func=STATUS_LABELS.get,key='history_status_filter')
    filtered=[record for record in records if (not query or query.casefold() in record['title'].casefold()) and (not statuses or record['status'] in statuses)]
    st.caption(f'Показано записей: {len(filtered)}. База хранится только на этом компьютере.')
    if not filtered: st.info('По выбранным условиям запусков нет.');return
    pages=max(1,(len(filtered)+9)//10)
    if st.session_state.get('history_page',1)>pages: st.session_state['history_page']=1
    page=st.number_input('Страница истории',1,pages,1,key='history_page')
    for record in filtered[(page-1)*10:page*10]:
        identifier=record['id']
        with st.container(border=True,key='history_card_'+identifier):
            title,actions=st.columns([3,2])
            with title:
                st.subheader(record['title'])
                st.caption(record['created_at'].replace('T',' ')+' · '+STATUS_LABELS.get(record['status'],record['status']))
                if record.get('error'): st.caption(record['error'])
            with actions:
                left,right=st.columns(2)
                if left.button('Открыть',key='history_open_'+identifier,width='stretch'):
                    try: runtime.load_run(identifier);st.rerun()
                    except ValueError as exc: st.error(str(exc))
                if right.button('Повторить',key='history_replay_'+identifier,width='stretch'):
                    try:
                        response=runtime.replay_run(identifier)
                        if response.get('ok'): st.rerun()
                        st.error(response.get('error','Не удалось повторить запуск.'))
                    except ValueError as exc: st.error(str(exc))
            with st.expander('Конфигурация, версия и отчёты'):
                saved=runtime.store.get(identifier,with_payload=True)
                st.caption('Версия: '+saved['app_version']+' · код '+saved['source_hash'])
                st.json(saved['config'],expanded=False)
                if saved.get('result'):
                    a,b=st.columns(2)
                    a.download_button('Отчёт HTML',build_report(saved),file_name='kontur-report-'+identifier[:8]+'.html',mime='text/html',on_click='ignore',key='history_html_'+identifier)
                    b.download_button('Данные и конфигурация ZIP',build_run_archive(saved),file_name='kontur-run-'+identifier[:8]+'.zip',mime='application/zip',on_click='ignore',key='history_zip_'+identifier)
