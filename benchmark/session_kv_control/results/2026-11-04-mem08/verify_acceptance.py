#!/usr/bin/env python3
"""Read-only integrity and expected-outcome verification of this measured archive."""
import gzip,hashlib,json,sys
from pathlib import Path
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root.parent.parent))
from analyze_stress import Evidence,numeric_differences,resource_inputs,verify_rows
from stress_bench import output_equal,summarize

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
for line in (root/'SHA256SUMS').read_text().splitlines():
 expected,name=line.split('  ',1);assert sha(root/name)==expected,name
analysis=json.loads((root/'analysis/analysis.json').read_text())
assert analysis['counts']['completed_formal_cells']==analysis['counts']['verified_cells']==8
assert analysis['counts']['valid_cells']==6
expected_invalid={
 'l3-c32-natural-baseline':set(json.loads((root/'provenance/baseline-reviewed.json').read_text())['expected_invalid_ids']),
 'l3-c32-natural-optimized':set(json.loads((root/'provenance/optimized-reviewed.json').read_text())['expected_invalid_ids'])}
assert len(expected_invalid['l3-c32-natural-baseline'])==1
assert len(expected_invalid['l3-c32-natural-optimized'])==4
calls=outputs=prompt=resume=0;signatures={};cells=0
for cell in analysis['cells']:
 if cell['state']!='completed':continue
 evidence=Evidence(root/cell['name']);manifest=evidence.load('manifest.json');schedule=evidence.load('schedule.json');arms=[evidence.load(x['id']+'.json') for x in schedule]
 assert not numeric_differences(summarize(arms),evidence.load('summary.json')),cell['name']
 errors,_=verify_rows(arms,manifest);assert not errors,(cell['name'],errors)
 invalid=set()
 signature=(manifest['input']['prefix_sha256'],manifest['input']['suffix_sha256'],manifest['args']['seed'],manifest['args']['output_tokens'])
 for arm in arms:
  for row in arm['rows']:
   generation=row.get('generation');assert generation and (row.get('phase')!='resume' or row.get('output_equal') is True)
   calls+=1;meta=generation['meta_info'];prompt+=meta['prompt_tokens'];outputs+=meta['completion_tokens']
   if not row['valid']:
    invalid.add(row['id']);op=row['prefetch']['operation'];assert not op['public_result']['ready_scope']
    assert op['update']['code']==('whole_scope_capacity' if cell['name'].endswith('optimized') else 'incomplete_storage_read')
   if row.get('phase')=='resume':
    prior=signatures.setdefault(signature,generation);assert output_equal(prior,generation);resume+=1
 assert invalid==expected_invalid.get(cell['name'],set()),(cell['name'],invalid)
 source=evidence.load('source-publication.json',False)
 for row in (source or {}).get('rows',[]):
  assert row['valid'];calls+=1;meta=row['generation']['meta_info'];prompt+=meta['prompt_tokens'];outputs+=meta['completion_tokens']
 cells+=1
assert (cells,calls,resume,prompt,outputs)==(8,5724,4656,93879792,150060)
parts=resource_inputs(root);digest=hashlib.sha256();records=0
for part in parts:
 with gzip.open(part,'rb') as handle:
  for line in handle:json.loads(line);digest.update(line);records+=1
index=json.loads((root/'resources-index.json').read_text())
assert records==index['records']==4570
assert digest.hexdigest()==index['uncompressed_full_sha256']
cleanup=json.loads((root/'provenance/final-cleanup.json').read_text())
assert {r['role'] for r in cleanup['closed']}=={'engine_a','engine_b','master'}
assert cleanup.get('resource_observer_additional_closed')
print('PASS: all file hashes; 8 measured cells; 5724 completed calls; 4656 cross-cell matching outputs; 1 known baseline failure + 4 capacity deferrals retained; 4570 losslessly reconstructed samples; task cleanup recorded.')
