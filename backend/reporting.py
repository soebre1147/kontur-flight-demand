"""Standalone, escaped HTML reports and reproducible ZIP run bundles."""
import csv
from html import escape
import io
import json
import re
import zipfile

LABELS={'paired_runs':'Парных сравнений','independent_seed_clusters':'Независимых seed','mean_mae_difference':'Разница MAE: v3 − базовый, мест','difference_ci_low':'Нижняя граница 95%','difference_ci_high':'Верхняя граница 95%','v3_win_pct':'Побед v3, %','tie_pct':'Равная ошибка, %','scenario_count':'Сценариев','runs_per_scenario':'Запусков на сценарий','generated_realizations':'Реализаций','fits_attempted':'Оценок методов','successful_fits':'Успешных оценок','failed_fits':'Ошибок расчёта','seed_start':'Начальный seed','scenario':'Сценарий','scenario_label':'Название сценария','method':'Метод','runs':'Запусков','successes':'Успешных оценок','failures':'Ошибок оценки','hidden_period_mae':'MAE по периодам, мест','hidden_period_wape':'WAPE по периодам, %','hidden_cabin_mae':'MAE по салонам, мест','budget_wasserstein':'Расхождение бюджетов, ₽','interval_coverage_80':'Покрытие интервалов 80%, %','by_scenario':'Сравнение по сценариям','overall':'Общие показатели','all_runs':'Все расчёты','flights':'Рейсы','periods':'Спрос по периодам','budgets':'Квантили бюджетов','cabins':'Скрытый спрос по салонам','comparison':'Сравнение тарифных политик','holdout':'Независимая проверка','holdout_runs':'Все проверочные запуски','candidates':'Кандидаты параметров','training_candidates':'Подбор тарифных правил','forecasts':'Прогнозы до вылета','sensitivity_envelope':'Диапазон чувствительности','configurations':'Варианты предположений','bootstrap':'Повторные оценки','flight':'Рейс','seed':'Seed','actual':'Эталон','estimated':'Оценка','period':'Период','absolute_error':'Абсолютная ошибка','lower':'Нижняя граница 10%','upper':'Верхняя граница 90%','policy':'Политика','revenue':'Выручка, ₽','profit':'Прибыль, ₽','load_pct':'Загрузка, %','training_runs':'Обучающих запусков','validation_runs':'Проверочных запусков','selected_sigma':'Выбранный σ','selected_mixture':'Смесь бюджетов','selected_beta_time':'Временной эффект β','selected_beta_qty':'Эффект размера группы β','cutoff_dtd':'Дней до вылета','predicted_net':'Прогноз занятых мест','actual_net':'Фактически занято мест','mean_final_absolute_error':'Средняя ошибка итога, мест','candidate_count':'Кандидатов правил','paired_difference':'Изменение результата к текущему правилу'}


def _label(value): return LABELS.get(str(value),str(value).replace('_',' '))


def _json(value):
    return json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2)


def _text(value):
    if value is None: return '—'
    if isinstance(value,float): return f'{value:,.4g}'
    if isinstance(value,(dict,list)): return _json(value)
    return str(value)


def _tables(result):
    tables=result.get('tables',[]) if isinstance(result,dict) else []
    if isinstance(tables,dict): tables=[{'name':name,'rows':rows} for name,rows in tables.items()]
    if isinstance(tables,list):
        return [table for table in tables if isinstance(table,dict)]
    return []


def _columns(table):
    columns=table.get('columns') or list(dict.fromkeys(key for row in table.get('rows',[]) if isinstance(row,dict) for key in row))
    return [(column.get('key'),column.get('label',column.get('key'))) if isinstance(column,dict) else (column,column) for column in columns]


def csv_bytes(rows,columns=None):
    if not isinstance(rows,list): raise ValueError('Таблица должна быть списком строк.')
    columns=columns or list(dict.fromkeys(key for row in rows for key in row))
    stream=io.StringIO(newline='');writer=csv.writer(stream,delimiter=';',quoting=csv.QUOTE_ALL,lineterminator='\r\n')
    writer.writerow(columns)
    writer.writerows([_json(row.get(key)) if isinstance(row.get(key),(dict,list)) else row.get(key) for key in columns] for row in rows)
    return stream.getvalue().encode('utf-8-sig')


def build_report(record):
    result=record.get('result') or {};title=record.get('title') or result.get('title') or 'Результаты исследования'
    parts=['<!doctype html><html lang="ru"><meta charset="utf-8"><title>'+escape(title)+'</title>',
      '''<style>body{font:15px system-ui,sans-serif;max-width:1180px;margin:36px auto;padding:0 24px;color:#243c34;line-height:1.6}h1{color:#0b7467}h2{margin-top:32px}table{border-collapse:collapse;width:100%;font-size:12px}td,th{padding:8px;border:1px solid #dce5df;text-align:left;vertical-align:top}th{background:#eef6f1}pre{background:#f5f8f6;padding:14px;white-space:pre-wrap;overflow-wrap:anywhere;font-size:11px}.muted{color:#6c8076}.scroll{overflow:auto}svg{max-width:100%;height:auto}@media print{body{margin:0;font-size:11px}tr{break-inside:avoid}}</style>''',
      '<h1>'+escape(title)+'</h1>',
      '<p class="muted">Запуск '+escape(str(record.get('id','')))+' · '+escape(str(record.get('created_at','')))+' · '+escape(str(record.get('status','')))+'<br>Версия '+escape(str(record.get('app_version','')))+' · код '+escape(str(record.get('source_hash','')))+'</p>']
    if isinstance(result.get('results'),dict) and all(isinstance(data,dict) and 'periods' in data and 'summary' in data for data in result['results'].values()):
        tables=[]
        for flight,data in result['results'].items():
            tables.append({'name':flight+' — показатели','rows':[data['summary']]})
            tables.append({'name':flight+' — спрос по периодам','rows':data['periods']})
        result={'title':title,'tables':tables,'summary':'Наблюдаемые операции и математическая оценка скрытого спроса. N задано извне.'}
    summary=result.get('summary',{})
    if summary:
        if isinstance(summary,dict):
            parts.append('<h2>Результат</h2><table>'+''.join('<tr><th>'+escape(_label(key))+'</th><td>'+escape(_text(value))+'</td></tr>' for key,value in summary.items())+'</table>')
        else: parts.append('<h2>Результат</h2><p>'+escape(_text(summary))+'</p>')
    for table in _tables(result):
        columns=_columns(table);rows=table.get('rows',[])
        detailed=table.get('name') in ('all_runs','paired_runs','holdout_runs','training_candidates','configurations','bootstrap','daily','candidates')
        if detailed: parts.append('<details><summary>'+escape(_label(table.get('name','Таблица')))+f' · {len(rows)} строк</summary>')
        parts.append('<h2>'+escape(_label(table.get('name','Таблица')))+'</h2><div class="scroll"><table><thead><tr>'+''.join('<th>'+escape(_label(label))+'</th>' for key,label in columns)+'</tr></thead><tbody>')
        for row in rows:
            if isinstance(row,dict): parts.append('<tr>'+''.join('<td>'+escape(_text(row.get(key)))+'</td>' for key,label in columns)+'</tr>')
        parts.append('</tbody></table></div>')
        if detailed: parts.append('</details>')
    warnings=result.get('warnings',[])
    if warnings: parts.append('<h2>Допущения и ограничения</h2><ul>'+''.join('<li>'+escape(str(w))+'</li>' for w in warnings)+'</ul>')
    parts.append('<details><summary>Конфигурация и источники</summary><pre>'+escape(_json(record.get('config',{})))+'</pre></details>')
    parts.append('<p class="muted">Скрытые обращения и бюджеты являются модельными оценками. Совпадение заданного общего N не является самостоятельной проверкой точности.</p></html>')
    return '\n'.join(parts).encode('utf-8')


def _safe_name(value):
    return re.sub(r'[^\w.\-]','_',str(value))[:140] or 'data'


def build_run_archive(record):
    stream=io.BytesIO();result=record.get('result') or {}
    with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('report.html',build_report(record))
        metadata={k:v for k,v in record.items() if k not in ('payload','result')}
        archive.writestr('run.json',_json(metadata));archive.writestr('result.json',_json(result))
        for number,table in enumerate(_tables(result),1):
            archive.writestr(f'tables/{number:02d}-{_safe_name(table.get("name","table"))}.csv',csv_bytes(table.get('rows',[]),[key for key,label in _columns(table)]))
        for field in ('files','offers_files','reference_files'):
            for number,file in enumerate((record.get('payload') or {}).get(field) or [],1):
                archive.writestr(f'inputs/{field}/{number:02d}-{_safe_name(file["name"])}',file['text'].encode('utf-8-sig'))
        bundle=result.get('bundle') or {}
        for key in ('bookings','offers','passengers'):
            if isinstance(bundle.get(key),list): archive.writestr('generated/'+key+'.csv',csv_bytes(bundle[key]))
        if bundle: archive.writestr('generated/config.json',_json({k:v for k,v in bundle.items() if k not in ('bookings','offers','passengers','events','requests')}))
    return stream.getvalue()
