"""Server-side downloads for an already calculated result.

Exports format the supplied result only; they never rerun simulation or fitting.
Structured provenance is JSON inside the quoted CSV cell, so it can be parsed
without relying on Python's repr or losing the original source fields.
"""
import csv
import io
import json
import re


RESULT_COLUMNS = ['timestamp','id','cabin','rbd','qty','price','amount','amount_source',
    'outcome','flight','dtd','period','source_file','source_row','id_is_synthetic',
    'reconstruction_status','record_level','row_source','signed_qty',
    'model_preferred_cabin','reason','reason_is_modelled']

_KINDS = {
    'bookings':('observed','bookings',['timestamp','cabin','rbd','commercial','price','flight','source_file','source_row']),
    'reconstructed':('reconstructed','passengers-reconstructed',RESULT_COLUMNS),
    'events':('events','events',RESULT_COLUMNS+['cancelRef']),
    'daily':('curve','daily-bookings',['date','dtd','gross','net','canceled','purchases_delta','cancellations_delta','net_delta']),
    'periods':('periods','demand-periods',['period','purchased','canceled','estimated','actual','lower','upper','baseline']),
    'offers':('offers','offers',['flight','cabin','start','end','price','open','available_seats','rbd','offer_source']),
}


def _json(value):
    return json.dumps(value,ensure_ascii=False,allow_nan=False,separators=(',',':'))


def _cell(value):
    if value is None: return ''
    if isinstance(value,(dict,list,tuple)): return _json(value)
    if isinstance(value,bool): return 'true' if value else 'false'
    return value


def _is_proxy(row):
    source=str(row.get('offer_source') or '').lower()
    return source.startswith(('assumed_','inferred_')) or source in ('purchase_derived_assumption','proxy')


def _filename_part(value,fallback):
    # Service validates these fields; this also keeps the utility safe when
    # handed an older result rather than a current service response.
    return re.sub(r'[^A-Za-z0-9_.-]','_',str(value or fallback))[:80] or fallback


def serialize_export(result,kind):
    """Return (bytes, filename, MIME) for a single calculated flight result.

    CSV uses UTF-8 with BOM, CRLF, semicolon and quoted cells. Every field found
    in the selected journal is included after the stable leading columns.
    Actual-offer export excludes booking-derived price assumptions and raises
    ValueError if no supplied/simulated state history exists.
    """
    if not isinstance(result,dict): raise ValueError('Для экспорта требуется рассчитанный результат.')
    params=result.get('params') or {}
    flight=_filename_part(params.get('flight'),'flight')
    departure=_filename_part(params.get('departure'),'date')
    if kind=='params':
        document={'model':'kontur-python-v3','params':params,'model_config':result.get('model',{}),
            'mode':result.get('mode'),'diagnostics':result.get('diagnostics',{}),
            'export_schema':'kontur-export-v1'}
        return (json.dumps(document,ensure_ascii=False,allow_nan=False,indent=2).encode('utf-8'),
            f'scenario-{flight}.json','application/json')
    if kind not in _KINDS: raise ValueError('Неизвестный вид экспорта.')
    source,prefix,leading=_KINDS[kind]
    raw=result.get(source,[])
    if not isinstance(raw,list) or any(not isinstance(row,dict) for row in raw):
        raise ValueError('Экспортируемый журнал имеет некорректный формат.')
    rows=[dict(row) for row in raw]
    if kind=='bookings':
        for row in rows:
            # In an observed operation commercial is signed. Do not export
            # modeled qty as a new purchase or link a cancellation to an ID.
            row['commercial']=row.get('signed_qty',row.get('commercial'))
    elif kind=='offers':
        assumptions=(result.get('diagnostics') or {}).get('assumptions') or {}
        inferred=assumptions.get('offer_source')=='purchase_derived_assumption'
        rows=[row for row in rows if not _is_proxy(row)
              and not (inferred and not row.get('offer_source'))]
        if not rows:
            raise ValueError('Реальная история offers не загружена. Предполагаемая цена из покупок не экспортируется как история предложений.')
    columns=list(leading)
    seen=set(columns)
    for row in rows:
        for column in row:
            if column not in seen: columns.append(column);seen.add(column)
    stream=io.StringIO(newline='')
    writer=csv.writer(stream,delimiter=';',quoting=csv.QUOTE_ALL,lineterminator='\r\n')
    writer.writerow(columns)
    writer.writerows([_cell(row.get(column)) for column in columns] for row in rows)
    return (stream.getvalue().encode('utf-8-sig'),f'{prefix}-{flight}-{departure}.csv','text/csv;charset=utf-8')
