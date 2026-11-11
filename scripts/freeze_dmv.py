"""Download only the eleven research columns; preserve failed partial snapshots."""
import csv, datetime, json, time
from pathlib import Path
import requests

ROOT=Path(__file__).resolve().parents[1]
COLS=['record_type','registration_class','state','county','body_type','fuel_type','reg_valid_date','color','scofflaw_indicator','suspension_indicator','revocation_indicator']
out=ROOT/'data'/('snapshot_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
out.mkdir(parents=True,exist_ok=False)
s=requests.Session(); s.trust_env=False
base='https://data.ny.gov/resource/w4pv-hbkt.json'
def get(url,params=None):
    for attempt in range(3):
        try:
            r=s.get(url,params=params,timeout=(15,120));r.raise_for_status();return r.json()
        except requests.RequestException:
            if attempt==2:raise
            time.sleep(2)
meta=get('https://data.ny.gov/api/views/w4pv-hbkt.json')
count=int(get(base,{'$select':'count(*)'})[0]['count'])
receipt={'network_policy':'DIRECT_ONLY_NO_PROXY','status':'RUNNING','source':base,'rowsUpdatedAt':meta['rowsUpdatedAt'],'expected_rows':count,'columns':COLS,'rows_written':0,'started':datetime.datetime.now(datetime.timezone.utc).isoformat()}
def save(): (out/'manifest.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
save()
try:
    with (out/'dmv.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.writer(f);writer.writerow(COLS)
        # Missing key is NULL; encode explicitly to distinguish a literal empty string.
        for offset in range(0,count,100000):
            rows=get(base,{'$select':','.join(COLS),'$order':':id','$limit':100000,'$offset':offset})
            if len(rows)!=min(100000,count-offset):raise RuntimeError('Unexpected page length')
            for row in rows:
                vals=[row.get(c) for c in COLS]
                if any(v=='__NARU_NULL__' for v in vals):raise RuntimeError('NULL sentinel collision')
                writer.writerow(['__NARU_NULL__' if v is None else v for v in vals])
            receipt['rows_written']+=len(rows);f.flush();save()
            print(receipt['rows_written'],count,flush=True)
    after=get('https://data.ny.gov/api/views/w4pv-hbkt.json')
    final_count=int(get(base,{'$select':'count(*)'})[0]['count'])
    if after['rowsUpdatedAt']!=meta['rowsUpdatedAt'] or final_count!=count:raise RuntimeError('Source changed during download')
    receipt.update(status='COMPLETE',null_sentinel='__NARU_NULL__',bytes=(out/'dmv.csv').stat().st_size)
except Exception as e:
    receipt.update(status='FAILED',error=str(e));raise
finally:
    receipt['finished']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
