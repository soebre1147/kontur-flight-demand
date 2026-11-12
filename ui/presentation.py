"""Russian presentation labels and native, bounded research tables."""
import json
import math
import re

import pandas as pd
import streamlit as st

from backend.reporting import csv_bytes


LABELS={
    'paired_runs':'Парных сравнений','independent_seed_clusters':'Независимых seed',
    'mean_mae_difference':'Разница MAE: v3 − базовый, мест','difference_ci_low':'Нижняя граница 95%',
    'difference_ci_high':'Верхняя граница 95%','v3_win_pct':'Побед v3, %','tie_pct':'Равная ошибка, %',
    'scenario_count':'Сценариев','runs_per_scenario':'Запусков на сценарий','generated_realizations':'Реализаций',
    'fits_attempted':'Оценок методов','successful_fits':'Успешных оценок','failed_fits':'Ошибок расчёта',
    'seed_start':'Начальный seed','scenario':'Сценарий','scenario_label':'Сценарий','method':'Метод',
    'runs':'Запусков','run':'Номер запуска','seed':'Seed','successes':'Успешных оценок','failures':'Ошибок оценки',
    'successful_runs':'Успешных запусков','failed_runs':'Ошибок оценки','errors':'Причины ошибок','error':'Ошибка',
    'hidden_period_mae':'MAE по периодам, мест','hidden_period_wape':'WAPE по периодам, %',
    'hidden_cabin_mae':'MAE по салонам, мест','budget_wasserstein':'Расхождение бюджетов, ₽',
    'interval_coverage_80':'Покрытие интервалов 80%, %','by_scenario':'Сравнение по сценариям',
    'overall':'Общие показатели','all_runs':'Все расчёты','flights':'Рейсы','flight':'Рейс',
    'periods':'Спрос по периодам','period':'Период','budgets':'Квантили бюджетов','cabins':'Скрытый спрос по салонам',
    'cabin':'Салон','comparison':'Сравнение тарифных политик','holdout':'Независимая проверка',
    'holdout_runs':'Все проверочные запуски','candidates':'Кандидаты параметров',
    'candidate':'Кандидат','candidate_count':'Кандидатов правил','training_candidates':'Подбор тарифных правил',
    'forecasts':'Прогнозы до вылета','forecast':'Прогноз загрузки','daily':'Прогноз по дням',
    'sensitivity_envelope':'Диапазон чувствительности','envelope':'Чувствительность скрытого спроса',
    'configurations':'Варианты предположений','bootstrap':'Повторные оценки',
    'actual':'Эталон','estimated':'Оценка','absolute_error':'Абсолютная ошибка',
    'lower':'Нижняя граница 10%','upper':'Верхняя граница 90%','covered_80':'Эталон внутри интервала',
    'policy':'Политика','revenue':'Выручка, ₽','profit':'Прибыль, ₽','load_pct':'Загрузка, %',
    'training_runs':'Обучающих запусков','validation_runs':'Проверочных запусков',
    'selected_sigma':'Выбранный σ','sigma':'Разброс бюджетов σ','prior_strength':'Вес предположений',
    'selected_mixture':'Смесь бюджетов','mixture':'Смесь бюджетов',
    'selected_beta_time':'Временной эффект β','beta_time':'Временной эффект β',
    'selected_beta_qty':'Эффект размера группы β','beta_qty':'Эффект размера группы β',
    'cutoff_dtd':'Дней до вылета','cutoff':'Момент прогноза','predicted_net':'Прогноз занятых мест',
    'actual_net':'Фактически занято мест','observed_net':'Известный баланс мест',
    'mean_final_absolute_error':'Средняя ошибка итога, мест','cutoffs':'Временных срезов',
    'paired_difference':'Изменение к текущей политике','objective':'Цель',
    'objective_mean':'Среднее значение цели','objective_std':'Разброс значения цели',
    'feasible_training_policy':'Условие загрузки выполнено при подборе',
    'selected_kind':'Выбранное правило','selected_price_multiplier':'Множитель цены',
    'selected_growth_multiplier':'Множитель роста','holdout_mae_difference':'Разница MAE после калибровки, мест',
    'improves_holdout_mae':'Ошибка на проверке уменьшилась','holdout_mae':'MAE на проверке, мест',
    'holdout_budget_wasserstein':'Расхождение бюджетов на проверке, ₽',
    'training_mae':'MAE при подборе, мест','training_budget_wasserstein':'Расхождение бюджетов при подборе, ₽',
    'successful':'Успешных конфигураций','bootstrap_successful':'Успешных повторных оценок',
    'primary_seats':'Первичный спрос, мест','purchase_seats':'Куплено до отмен, мест',
    'cancel_seats':'Отменено мест','hidden_seats':'Без покупки, мест','offer_intervals':'Интервалов предложений',
    'gross':'Куплено до отмен','canceled':'Отменено мест','net':'Баланс мест',
    'rejected':'Без покупки, мест','price_rejected':'Отказ по бюджету','capacity_rejected':'Отказ из-за мест',
    'closed_rejected':'Отказ при закрытых продажах','pareto':'На границе компромисса',
    'reference_gross':'Покупок в эталоне, мест','reference_hidden_seats':'Скрытых мест в эталоне',
    'reference_primary_seats':'Первичных мест в эталоне','known_hidden_cabin_seats':'Скрытых мест с известным салоном',
    'budget_reference_seats':'Мест с эталоном бюджета','budget_estimated_seats':'Мест с оценкой бюджета',
    'budget_quantiles':'Сравнение бюджетов','quantile':'Квантиль','min_estimated':'Минимальная оценка, мест',
    'max_estimated':'Максимальная оценка, мест','min_conditional_lower':'Минимальная условная граница',
    'max_conditional_upper':'Максимальная условная граница','median_budget':'Медианный бюджет, ₽',
    'replicate':'Повторная оценка','date':'Дата','dtd':'Дней до вылета','status':'Статус',
    'elapsed_seconds':'Время, с','ok':'Успешно','v3_mae':'MAE v3, мест',
    'baseline_mae':'MAE базового метода, мест','mae_difference':'Разница MAE, мест',
    'v3_wins':'Победа v3','tie':'Одинаковая ошибка','seed_split':'Разделение seed',
    'seeds':'Seed','phase':'Этап','demand':'Внешний N','historical_N_assumption':'Предположение о прошлом N',
    'future_purchase_mean':'Ожидаемые будущие покупки','future_cancellation_mean':'Ожидаемые будущие отмены',
}
STATUS_LABELS={'queued':'В очереди','running':'Выполняется','cancelling':'Остановка…','completed':'Завершено',
               'failed':'Ошибка','cancelled':'Отменено','interrupted':'Прервано при остановке сервера'}
VALUE_LABELS={'baseline':'Базовый метод','v3':'Модель v3','current':'Текущая модель','selected':'После калибровки',
              'ok':'Выполнено','failed':'Ошибка','revenue':'Выручка','profit':'Прибыль',
              'dynamic':'Динамическая цена','static':'Постоянная цена','gated':'Управление RBD',
              'training':'Подбор','holdout':'Независимая проверка'}
MAIN_TABLES={'by_scenario','overall','holdout','comparison','flights','periods','budgets','cabins','forecasts','sensitivity_envelope'}
PRIORITY=['paired_runs','v3_win_pct','mean_mae_difference','difference_ci_low','difference_ci_high','failed_fits',
          'flights','training_runs','validation_runs','improves_holdout_mae','holdout_mae_difference','selected_sigma',
          'selected_mixture','selected_beta_time','selected_beta_qty','successful','bootstrap_successful','configurations',
          'cutoffs','mean_final_absolute_error','paired_difference','candidate_count','feasible_training_policy',
          'primary_seats','purchase_seats','hidden_seats']


def label(key):
    return LABELS.get(str(key),str(key).replace('_',' '))


def format_value(value):
    if value is None: return '—'
    if isinstance(value,bool): return 'Да' if value else 'Нет'
    if isinstance(value,(int,float)):
        if not math.isfinite(float(value)): return '—'
        return f'{value:,.3f}'.rstrip('0').rstrip('.').replace(',','\u202f').replace('.',',') if isinstance(value,float) else f'{value:,}'.replace(',','\u202f')
    if isinstance(value,(dict,list,tuple)): return json.dumps(value,ensure_ascii=False)
    return VALUE_LABELS.get(str(value),str(value))


def summary_items(summary):
    if not isinstance(summary,dict): return []
    values=[(key,value) for key,value in summary.items() if not isinstance(value,(dict,list,tuple))]
    return sorted(values,key=lambda row: PRIORITY.index(row[0]) if row[0] in PRIORITY else len(PRIORITY))


def display_frame(rows):
    """Localise a copy and preserve numeric columns and unique Arrow names."""
    frame=pd.DataFrame(rows).copy()
    if frame.empty: return frame
    if 'scenario_label' in frame and 'scenario' in frame: frame=frame.drop(columns='scenario')
    for column in frame.columns:
        if frame[column].map(lambda value:isinstance(value,(dict,list,tuple))).any():
            frame[column]=frame[column].map(lambda value:json.dumps(value,ensure_ascii=False) if isinstance(value,(dict,list,tuple)) else value)
        elif column in ('method','status','objective','selected_kind','phase'):
            frame[column]=frame[column].map(lambda value:VALUE_LABELS.get(str(value),STATUS_LABELS.get(str(value),value)))
    columns=[];seen={}
    for key in frame.columns:
        name=label(key);seen[name]=seen.get(name,0)+1
        columns.append(name if seen[name]==1 else f'{name} ({seen[name]})')
    frame.columns=columns
    return frame


def tables_of(data):
    if isinstance(data,dict):
        return [{'name':name,'rows':value} if isinstance(value,list) else {'name':name,**value}
                for name,value in data.items() if isinstance(value,(dict,list))]
    if isinstance(data,list): return [table for table in data if isinstance(table,dict)]
    return []


def render_summary(summary,key_prefix='lab'):
    items=summary_items(summary)
    for start in range(0,min(6,len(items)),3):
        columns=st.columns(3)
        for column,(key,value) in zip(columns,items[start:start+3]):
            with column: st.metric(label(key),format_value(value),border=True)
    if len(items)>6:
        with st.expander('Все показатели'):
            st.dataframe(pd.DataFrame([{'Показатель':label(k),'Значение':format_value(v)} for k,v in items[6:]]),hide_index=True,width='stretch')


def render_tables(data,key_prefix='lab'):
    for number,table in enumerate(tables_of(data)):
        rows=table.get('rows',table.get('data',[]))
        if not isinstance(rows,list): continue
        name=table.get('name',table.get('title','Таблица'))
        title='Парные сравнения' if name=='paired_runs' else label(name)
        if name not in MAIN_TABLES or len(rows)>100:
            with st.expander(f'{title} · {len(rows)} строк'):
                _render_table(rows,title,f'{key_prefix}_{number}')
        else:
            with st.container(border=True):
                st.subheader(title)
                _render_table(rows,title,f'{key_prefix}_{number}')


def _render_table(rows,title,key):
    if not rows:
        st.caption('Нет строк для отображения.');return
    if len(rows)>50:
        pages=(len(rows)+49)//50
        page=st.number_input('Страница таблицы',min_value=1,max_value=pages,value=1,step=1,key=key+'_page')
        selected=rows[(page-1)*50:page*50]
        st.caption(f'Показаны строки {(page-1)*50+1}–{min(page*50,len(rows))} из {len(rows)}')
    else: selected=rows
    st.dataframe(display_frame(selected),hide_index=True,width='stretch',height='auto' if len(selected)<12 else 360)
    st.download_button('Скачать всю таблицу CSV',csv_bytes(rows),file_name=re.sub(r'[^\w-]','_',title)+'.csv',
                       mime='text/csv;charset=utf-8',on_click='ignore',key=key+'_csv')
