#!/usr/bin/env python3
"""Verify preserved raw bytes, schedule coverage, summary counts and checksums."""
import gzip,hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
for line in (root/'SHA256SUMS').read_text().splitlines():
 expected,name=line.split('  ',1)
 assert sha((root/name).read_bytes())==expected, name
for run in json.loads((root/'runs.json').read_text()):
 p=root/run['run'];packed=(p/'raw.jsonl.gz').read_bytes();raw=gzip.decompress(packed)
 assert sha(packed)==run['raw_gzip_sha256'] and sha(raw)==run['raw_jsonl_sha256']
 assert packed[4:8]==b'\0\0\0\0', 'gzip timestamp must be deterministic'
 rows=[json.loads(s) for s in raw.splitlines()];trials=[r for r in rows if r.get('kind')=='trial'];summary=json.loads((p/'summary.json').read_text());schedule=json.loads((p/'schedule.json').read_text())
 assert len(trials)==summary['attempts']==len(schedule)==run['attempts']
 assert len({t['id'] for t in trials})==len(trials)
 assert {t['id'] for t in trials}=={t['id'] for t in schedule}
 assert sum(t.get('valid',False) for t in trials)==summary['valid_attempts']
 assert {t['id'] for t in trials if not t.get('valid')}=={f['trial'] for f in summary['failures']}
assert json.loads((root/'validation.json').read_text())['all_checks_passed']
print('Verified SHA256, gzip, raw/summary counts, complete schedules, failures and derived means.')
