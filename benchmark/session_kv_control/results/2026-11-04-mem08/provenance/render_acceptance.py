"""Render measured comparisons, CPU-counter costs and cross-cell output proof."""
import bisect,csv,gzip,hashlib,json,math,sys
from pathlib import Path
root=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(root.parent.parent))
from analyze_stress import read_json,locate
from stress_bench import output_equal
analysis=json.loads((root/'analysis/analysis.json').read_text())
cells={c['name']:c for c in analysis['cells'] if c['state']=='completed'}
rows=[]
for name,c in cells.items():
 a=c['groups']['on_demand'];key='pin' if c['args']['command']=='l2' else 'prefetch';b=c['groups'][key]
 rows.append({'cell':name,'candidate':key,'mean_ttft_delta_percent':100*(b['resume_ttft_ms']['mean']/a['resume_ttft_ms']['mean']-1),
  'p95_ttft_delta_percent':100*(b['resume_ttft_ms']['p95']/a['resume_ttft_ms']['p95']-1),
  'work_throughput_delta_percent':100*(b['work_excluding_flush_completed_requests_per_s']/a['work_excluding_flush_completed_requests_per_s']-1),
  'client_cycle_delta_percent':100*(b['client_cycle_ms']['mean']/a['client_cycle_ms']['mean']-1) if a['client_cycle_ms']['mean'] else None,
  'invalid_control_rows':b['invalid_requests'],'completed_per_arm':b['completed_requests']})
(root/'analysis/paired-relative.json').write_text(json.dumps(rows,indent=2)+'\n')
with (root/'analysis/paired-relative.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
# Every matching actual input must retain the same generated IDs/text across code snapshots and load levels.
expected={};checked=0;prefixes={};total_input=total_output=0
for name,c in cells.items():
 p=root/name;manifest=read_json(locate(p,'manifest.json'));args=manifest['args']
 signature=(manifest['input']['prefix_sha256'],manifest['input']['suffix_sha256'],args['seed'],args['output_tokens'])
 for spec in read_json(locate(p,'schedule.json')):
  arm=read_json(locate(p,spec['id']+'.json'))
  for r in arm['rows']:
   if r.get('generation'):
    meta=r['generation']['meta_info'];total_input+=meta['prompt_tokens'];total_output+=meta['completion_tokens']
   if r.get('phase')=='resume' and r.get('generation'):
    prior=expected.setdefault(signature,r['generation']);assert output_equal(prior,r['generation']),(name,r['id'])
    checked+=1;prefixes[name]=prefixes.get(name,0)+1
 pub=locate(p,'source-publication.json')
 if pub:
  for r in read_json(pub)['rows']:
   if r.get('generation'):
    m=r['generation']['meta_info'];total_input+=m['prompt_tokens'];total_output+=m['completion_tokens']
proof={'matched_resume_outputs_checked':checked,'distinct_input_signatures':len(expected),'per_cell':prefixes,'all_outputs_equal':True,'total_prompt_tokens_including_source':total_input,'total_completion_tokens_including_source':total_output}
(root/'analysis/output-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
# CPU counters are per process, not children_user/children_system, so tree deltas do not double count.
with gzip.open(root/'process-resources.jsonl.gz','rt') as f: cpu=[json.loads(line) for line in f]
times=[r['monotonic'] for r in cpu]
results=[]
def counters(sample,role):
 out={}
 for p in sample['roles'].get(role,{}).get('processes',[]):
  if all(k in p for k in ['pid','created','user_s','system_s']):out[(p['pid'],p['created'])]=p['user_s']+p['system_s']
 return out
for name,c in cells.items():
 for window in c['arm_windows']:
  start=window['work_start'] if window['work_start'] is not None else window['start'];end=window['end']
  lo=bisect.bisect_left(times,start);hi=bisect.bisect_right(times,end)-1
  rec={'cell':name,'arm_id':window['id'],'group':window['arm'],'work_seconds':end-start,'roles':{}}
  if 0<=lo<hi<len(cpu):
   duration=times[hi]-times[lo];rec.update(observed_seconds=duration,coverage=duration/(end-start))
   for role in ['engine_a','engine_b','master']:
    before,after=counters(cpu[lo],role),counters(cpu[hi],role);common=before.keys()&after.keys()
    complete=bool(common) and before.keys()==after.keys()
    delta=sum(max(0,after[k]-before[k]) for k in common)
    rec['roles'][role]={'cpu_seconds':delta,'sampled_average_cores':delta/duration,'stable_pids':len(common),'complete_endpoint_process_set':complete}
  else:rec.update(observed_seconds=0,coverage=0)
  results.append(rec)
aggregate=[]
for name,c in cells.items():
 for group,g in c['groups'].items():
  selected=[r for r in results if r['cell']==name and r['group']==group]
  for role in ['engine_a','engine_b','master']:
   usable=[r for r in selected if r['roles'].get(role,{}).get('complete_endpoint_process_set')]
   secs=sum(r['observed_seconds'] for r in usable);work=sum(r['work_seconds'] for r in selected);cpu_s=sum(r['roles'][role]['cpu_seconds'] for r in usable)
   aggregate.append({'cell':name,'group':group,'role':role,'observed_seconds':secs,'work_seconds':work,'coverage':secs/work if work else 0,
    'sampled_average_cores':cpu_s/secs if secs else None,'cpu_seconds_observed':cpu_s,
    'estimated_cpu_seconds_per_completed_call':cpu_s/secs*work/g['completed_requests'] if secs and secs/work>=.9 else None})
(root/'analysis/cpu-cost.json').write_text(json.dumps({'arms':results,'groups':aggregate,'definition':'Cumulative user+system deltas for stable PID/create-time process sets, using samples inside approximate work-only windows. 2s cadence omits boundaries. CPU/call is an estimate only when >=90% window coverage; no inference of unique physical host RSS.'},indent=2)+'\n')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,3,figsize=(12,4.5));colors=['#616b76','#138a72']
for j,(metric,label) in enumerate([('ttft','Mean TTFT (ms)'),('rate','Work completed requests/s'),('p95','Empirical p95 TTFT (ms)')]):
 for k,group in enumerate(['on_demand','prefetch']):
  values=[]
  for n in [8,16,32]:
   g=cells[f'l3-c{n}-natural-optimized']['groups'][group]
   values.append(g['resume_ttft_ms']['mean'] if metric=='ttft' else g['resume_ttft_ms']['p95'] if metric=='p95' else g['work_excluding_flush_completed_requests_per_s'])
  axes[j].bar([i+(k-.5)*.36 for i in range(3)],values,width=.36,label=group,color=colors[k])
 axes[j].set_xticks(range(3),['8','16','32']);axes[j].set_xlabel('Concurrent clients');axes[j].set_ylabel(label);axes[j].grid(axis='y',alpha=.2)
axes[0].legend();fig.suptitle('mem_fraction_static=0.8; 64 x 16K prefixes, 3 paired blocks, 384 resumes/condition')
fig.text(.5,.01,'All completed requests included. C32: 4 capacity deferrals; p95 worsens. Work rate includes 3s tool waits and control drain; excludes initial lab flush.',ha='center',fontsize=8)
fig.tight_layout(rect=(0,.05,1,.94));fig.savefig(root/'analysis/optimized-comparison.png',dpi=180);fig.savefig(root/'analysis/optimized-comparison.svg');plt.close(fig)
print(json.dumps({'relative_comparisons':rows,'output_proof':proof},ensure_ascii=False))
