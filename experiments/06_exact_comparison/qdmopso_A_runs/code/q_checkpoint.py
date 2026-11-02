"""Versioned JSON checkpoints; legacy pickle tables are intentionally rejected."""
import json,math,os
from pathlib import Path
SCHEMA='raw-hv-ewma-sp-valid3-bounded-reward-v1'
def save_checkpoint(path,q_table,epsilon):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    payload={'schema':SCHEMA,'actions':list(range(7)),'epsilon':float(epsilon),
             'q_table':[{'state':list(s),'values':[float(v) for v in row]} for s,row in sorted(q_table.items())]}
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf8')
    os.replace(temporary,path)
def load_checkpoint(path):
    payload=json.loads(Path(path).read_text(encoding='utf8'))
    if payload.get('schema')!=SCHEMA or payload.get('actions')!=list(range(7)):
        raise ValueError('Incompatible state/reward checkpoint schema; retrain with this version.')
    seen=set()
    for row in payload['q_table']:
        s=row['state'];q=row['values']
        if len(s)!=5 or tuple(s) in seen or len(q)!=7 or not all(math.isfinite(v) for v in q):raise ValueError('Invalid Q table')
        if not all(type(v) is int and v in domain for v,domain in zip(s,[range(3),range(-1,2),range(5),range(7),range(3)])):raise ValueError('Invalid state')
        seen.add(tuple(s))
    if not math.isfinite(payload['epsilon']) or not 0<=payload['epsilon']<=1:raise ValueError('Invalid epsilon')
    return payload
