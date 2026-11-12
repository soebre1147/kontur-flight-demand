"""Validated in-memory CSV import. No user-supplied filesystem paths are read."""
import csv
from decimal import Decimal, InvalidOperation
from hashlib import sha256
import io
import math
from pathlib import PureWindowsPath
import re

import pandas as pd
from .simulation import period_of

MAX_BYTES=10*1024*1024
MAX_ROWS=25000


def validate_files(files):
    if not isinstance(files,list) or len(files)>50:
        raise ValueError('Передайте не более 50 CSV-файлов.')
    size=0;seen=set()
    for f in files:
        if not isinstance(f,dict) or not isinstance(f.get('text'),str) or not str(f.get('name','')).strip():
            raise ValueError('У файла должны быть имя и текст CSV.')
        body=f['text'].encode('utf-8');size+=len(body)
        fingerprint=sha256(body).hexdigest()
        if fingerprint in seen: raise ValueError('Один и тот же CSV передан несколько раз.')
        seen.add(fingerprint)
    if size>MAX_BYTES: raise ValueError('Общий размер CSV не должен превышать 10 МБ.')


def parse_table(file,required):
    name=PureWindowsPath(str(file['name'])).name
    text=file['text'].lstrip('\ufeff')
    if not text.strip(): raise ValueError(f'{name}: CSV пуст.')
    head=next(line for line in text.splitlines() if line.strip())
    separator=';' if ';' in head else '\t' if '\t' in head else ','
    try:
        reader=csv.reader(io.StringIO(text),delimiter=separator,strict=True)
        rows=[]
        while True:
            start_line=reader.line_num+1
            try: row=next(reader)
            except StopIteration: break
            if any(x.strip() for x in row): rows.append((row,start_line))
    except csv.Error as e: raise ValueError(f'{name}: некорректный CSV: {e}') from None
    if not rows: raise ValueError(f'{name}: CSV пуст.')
    header=[h.strip() for h in rows[0][0]]
    if len(header)!=len(set(header)): raise ValueError(f'{name}: повторяются названия столбцов.')
    if set(required)-set(header):
        raise ValueError(f'{name}: нет столбцов {", ".join(sorted(set(required)-set(header)))}.')
    if len(rows)-1>MAX_ROWS: raise ValueError(f'{name}: максимум {MAX_ROWS} строк.')
    output=[]
    for values,n in rows[1:]:
        if len(values)!=len(header): raise ValueError(f'{name}, строка {n}: число полей не совпадает с заголовком.')
        output.append(({k:v.strip() for k,v in zip(header,values)},n))
    if not output: raise ValueError(f'{name}: нет строк данных.')
    return name,output


def _number(value,name,row,label,integer=False,nonnegative=False,positive=False):
    normalized=str(value).replace('\u00a0','').replace(' ','').replace(',','.')
    try:
        exact=Decimal(normalized)
        result=int(exact) if integer and exact.is_finite() and exact==exact.to_integral_value() and abs(exact)<=9007199254740991 else float(exact)
    except (InvalidOperation,ValueError,OverflowError): result=float('nan')
    if (not math.isfinite(result) or integer and (not isinstance(result,int))
            or nonnegative and result<0 or positive and result<=0):
        raise ValueError(f'{name}, строка {row}: некорректное значение {label}.')
    return int(result) if integer else float(result)


def _time(raw,name,row,label):
    try:
        value=pd.Timestamp(str(raw).replace(' UTC','+00:00'))
        if pd.isna(value): raise ValueError()
        return value.tz_localize('UTC') if value.tzinfo is None else value.tz_convert('UTC')
    except (ValueError,TypeError,OverflowError):
        raise ValueError(f'{name}, строка {row}: некорректное время {label}.') from None


def parse_bookings(files,departure):
    validate_files(files)
    if not files: raise ValueError('Выберите хотя бы один bookings CSV.')
    date=pd.Timestamp(departure,tz='UTC');events=[];seen=set()
    for file in files:
        name,rows=parse_table(file,{'timestamp','cabin','rbd','commercial','price'})
        inferred=re.search(r'bookings-([A-Za-z0-9_.]+)-',name)
        for row,n in rows:
            flight=row.get('flight') or (inferred.group(1) if inferred else '')
            if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,39}',flight):
                raise ValueError(f'{name}, строка {n}: нет корректного flight и его нельзя определить из имени.')
            timestamp=_time(row['timestamp'],name,n,'timestamp')
            dtd=int((date-timestamp.normalize()).days)
            if not 0<=dtd<=3650: raise ValueError(f'{name}, строка {n}: проверьте дату вылета; допустимый горизонт 0–3650 дней.')
            commercial=_number(row['commercial'],name,n,'commercial',integer=True)
            if commercial==0 or abs(commercial)>9007199254740991:
                raise ValueError(f'{name}, строка {n}: commercial должен быть ненулевым целым числом с точным представлением.')
            price=_number(row['price'],name,n,'price',positive=True)
            cabin=row['cabin'];rbd=row['rbd']
            if not re.fullmatch(r'[A-Za-z]',cabin) or not re.fullmatch(r'[A-Za-z]',rbd):
                raise ValueError(f'{name}, строка {n}: cabin и RBD должны быть непустыми буквенными кодами.')
            cabin=cabin.upper();rbd=rbd.upper()
            if cabin not in ('C','Y') or rbd not in ('JOCIZ' if cabin=='C' else 'YXDSBMUPKHLQTENRGV'):
                raise ValueError(f'{name}, строка {n}: RBD не относится к кабине C/Y.')
            key=(timestamp,flight,cabin,rbd,commercial,price)
            if key in seen: raise ValueError(f'{name}, строка {n}: повторяется неотличимая операция бронирования.')
            seen.add(key)
            events.append({'timestamp':timestamp,'flight':flight,'cabin':cabin,'rbd':rbd,
                'commercial':commercial,'price':price,'source_file':name,'source_row':n,'source_values':dict(row)})
    if len(events)>MAX_ROWS: raise ValueError(f'Во всех файлах вместе допускается до {MAX_ROWS} операций.')
    return pd.DataFrame(events).sort_values(['flight','timestamp','source_file','source_row']).reset_index(drop=True)


def parse_offers(files):
    validate_files(files)
    records=[]
    for file in files:
        name,rows=parse_table(file,{'flight','cabin','start','end','price','open','available_seats'})
        for row,n in rows:
            if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,39}',row['flight']):
                raise ValueError(f'{name}, строка {n}: некорректный flight.')
            if row['cabin'].upper() not in ('C','Y'):
                raise ValueError(f'{name}, строка {n}: cabin должен быть C или Y.')
            if row.get('rbd') and row['rbd'].upper() not in ('JOCIZ' if row['cabin'].upper()=='C' else 'YXDSBMUPKHLQTENRGV'):
                raise ValueError(f'{name}, строка {n}: RBD не относится к указанной кабине.')
            opened=row['open'].lower()
            if opened not in ('true','false','1','0'): raise ValueError(f'{name}, строка {n}: open — true/false либо 1/0.')
            opened=opened in ('true','1')
            price=_number(row['price'],name,n,'price',positive=True) if opened or row['price'] else float('nan')
            records.append({'flight':row['flight'],'cabin':row['cabin'].upper(),
                'start':_time(row['start'],name,n,'start'),'end':_time(row['end'],name,n,'end'),
                'price':price,'open':opened,'available_seats':_number(row['available_seats'],name,n,'available_seats',integer=True,nonnegative=True),
                'rbd':row.get('rbd','').upper() or None,'source_file':name,'source_row':n})
    if len(records)>MAX_ROWS: raise ValueError(f'Допускается до {MAX_ROWS} интервалов предложений.')
    return pd.DataFrame(records)
