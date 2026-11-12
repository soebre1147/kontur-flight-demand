"""Local reproducible run history; raw uploaded datasets are deduplicated."""
from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
import uuid

APP_VERSION='kontur-lab-v4'
ACTIVE_STATUSES=('queued','running','cancelling')


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def dumps(value):
    return json.dumps(value,ensure_ascii=False,allow_nan=False,separators=(',',':'))


def code_fingerprint(root):
    root=Path(root);digest=sha256()
    files=[root/'app.py',root/'requirements.txt']
    for directory in ('backend','model','dist'):
        files.extend(p for p in (root/directory).rglob('*') if p.is_file() and p.suffix in ('.py','.mjs','.html','.css'))
    for file in sorted(files):
        digest.update(file.relative_to(root).as_posix().encode());digest.update(file.read_bytes())
    return digest.hexdigest()


class RunStore:
    def __init__(self,path,source_hash=''):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.source_hash=source_hash
        with self.connection() as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.executescript('''
                CREATE TABLE IF NOT EXISTS datasets (sha TEXT PRIMARY KEY, text TEXT NOT NULL, bytes INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS runs (
                  id TEXT PRIMARY KEY, action TEXT NOT NULL, title TEXT NOT NULL,
                  created_at TEXT NOT NULL, updated_at TEXT NOT NULL, status TEXT NOT NULL,
                  completed INTEGER NOT NULL DEFAULT 0, total INTEGER NOT NULL DEFAULT 0,
                  message TEXT NOT NULL DEFAULT '', error TEXT,
                  payload_json TEXT NOT NULL, result_json TEXT,
                  app_version TEXT NOT NULL, source_hash TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS runs_created ON runs(created_at DESC);
            ''')

    @contextmanager
    def connection(self):
        db=sqlite3.connect(self.path,timeout=30)
        db.row_factory=sqlite3.Row
        try:
            with db: yield db
        finally: db.close()

    def create(self,action,title,payload,status='queued'):
        value=json.loads(dumps(payload));identifier=str(uuid.uuid4());now=utc_now()
        with self.connection() as db:
            for field in ('files','offers_files','reference_files'):
                files=value.get(field)
                if files:
                    references=[]
                    for file in files:
                        raw=file['text'].encode('utf-8');digest=sha256(raw).hexdigest()
                        db.execute('INSERT OR IGNORE INTO datasets VALUES (?,?,?)',(digest,file['text'],len(raw)))
                        references.append({'name':file['name'],'dataset_sha256':digest,'bytes':len(raw)})
                    value[field]=references
            db.execute('''INSERT INTO runs(id,action,title,created_at,updated_at,status,payload_json,app_version,source_hash)
                          VALUES(?,?,?,?,?,?,?,?,?)''',
                       (identifier,action,title,now,now,status,dumps(value),APP_VERSION,self.source_hash))
        return identifier

    def update(self,identifier,**fields):
        allowed={'status','completed','total','message','error','title','result_json'}
        if set(fields)-allowed: raise ValueError('Unsupported history update.')
        fields={**fields,'updated_at':utc_now()}
        with self.connection() as db:
            db.execute('UPDATE runs SET '+','.join(f'{k}=?' for k in fields)+' WHERE id=?',(*fields.values(),identifier))

    def complete(self,identifier,result,status='completed'):
        with self.connection() as db:
            row=db.execute('SELECT completed,total FROM runs WHERE id=?',(identifier,)).fetchone()
        total=row['total'] if row else 0
        self.update(identifier,status=status,result_json=dumps(result),completed=total if status=='completed' else (row['completed'] if row else 0),message='Расчёт завершён' if status=='completed' else 'Расчёт отменён')

    def get(self,identifier,with_result=True,with_payload=False):
        with self.connection() as db:
            row=db.execute('SELECT * FROM runs WHERE id=?',(str(identifier),)).fetchone()
        if row is None: raise ValueError('Запуск не найден в истории.')
        value=dict(row);config=json.loads(value.pop('payload_json'))
        value['config']=config
        raw=value.pop('result_json')
        if with_result: value['result']=json.loads(raw) if raw else None
        if with_payload:
            payload=json.loads(dumps(config))
            with self.connection() as db:
                for field in ('files','offers_files','reference_files'):
                    for file in payload.get(field) or []:
                        digest=file.get('dataset_sha256')
                        if digest:
                            name=file['name']
                            data=db.execute('SELECT text FROM datasets WHERE sha=?',(digest,)).fetchone()
                            if data is None: raise ValueError('Исходный набор данных запуска недоступен.')
                            file.clear();file.update({'name':name,'text':data['text']})
            value['payload']=payload
        return value

    def history(self,limit=100):
        limit=max(1,min(500,int(limit)))
        with self.connection() as db:
            rows=db.execute('''SELECT id,action,title,created_at,updated_at,status,completed,total,message,error,app_version,source_hash
                               FROM runs ORDER BY rowid DESC LIMIT ?''',(limit,)).fetchall()
        return [dict(row) for row in rows]

    def recover_interrupted(self):
        with self.connection() as db:
            db.execute("UPDATE runs SET status='interrupted',message='Сервер перезапущен; запустите расчёт повторно',updated_at=? WHERE status IN ('queued','running','cancelling')",(utc_now(),))
