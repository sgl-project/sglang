"""Capture completed task state and stop only verified task-owned services."""
import json,os,pathlib,psutil,signal,subprocess,time,urllib.request
base=pathlib.Path('/tmp/sglang-rfc-build/mem08')

def save(name,value):(base/name).write_text(json.dumps(value,indent=2)+'\n')
def get(port,path):
 with urllib.request.urlopen(f'http://127.0.0.1:{port}{path}',timeout=15) as r:return json.load(r)
for c in [8,16,32]:
 p=base/f'l3-c{c}-natural-optimized'
 assert (p/'checksums.json').is_file(),'optimized matrix not finished'
 d=json.loads((p/'summary.json').read_text())
 assert len(d['arms'])==6, 'optimized matrix incomplete'
 if not all(a['valid'] for a in d['arms']):
  review=json.loads((base/'optimized-reviewed.json').read_text());actual=[]
  for spec in json.loads((p/'schedule.json').read_text()):
   arm=json.loads((p/(spec['id']+'.json')).read_text());assert all(r.get('generation') and r.get('output_equal') is True for r in arm['rows'])
   actual.extend(r['id'] for r in arm['rows'] if not r['valid'])
  assert sorted(actual)==sorted(review['expected_invalid_ids']), 'optimized failures differ from reviewed capacity deferrals'
roles=json.loads((base/'task-processes.json').read_text())
loads={str(p):get(p,'/v1/loads?include=core,queues') for p in [51801,51803]}
assert all(s['num_running_reqs']==0 and s['num_waiting_reqs']==0 and all(v==0 for v in s['queues'].values()) for d in loads.values() for s in d['loads']),'not idle'
save('final-idle.json',loads)
save('final-capabilities.json',{str(p):get(p,'/v1/kv/capabilities') for p in [51801,51803]})
save('final-server-info.json',{str(p):get(p,'/server_info') for p in [51801,51803]})
(base/'hardware-final.csv').write_text(subprocess.check_output(['nvidia-smi','-i','0,1','--query-gpu=index,uuid,name,memory.total,memory.used,driver_version,temperature.gpu','--format=csv'],text=True))
observers=[]
for p in psutil.process_iter(['pid','cmdline']):
 cmd=p.info['cmdline'] or []
 if any(a.endswith('/benchmark/session_kv_control/observe_resources.py') for a in cmd) and str(base/'resources.jsonl') in cmd:
  observers.append(p)
 if len(cmd)==2 and cmd[1]==str(base/'cpu_observer.py'):observers.append(p)
for p in observers:p.terminate()
for p in observers:
 try:p.wait(timeout=15)
 except psutil.TimeoutExpired:raise RuntimeError('resource observer did not close')
print('Closed observers', [p.pid for p in observers],flush=True)

def stop(role,port):
 p=psutil.Process(roles[role]);cmd=p.cmdline()
 if role=='master':assert any(a.endswith('mooncake_master') for a in cmd) and '--port=52091' in cmd
 else:assert 'sglang.launch_server' in cmd and cmd[cmd.index('--port')+1]==str(port)
 owned=[(q.pid,q.create_time()) for q in [p]+p.children(recursive=True)]
 p.terminate();end=time.monotonic()+60
 alive=[]
 while time.monotonic()<end:
  alive=[]
  for pid,created in owned:
   try:
    q=psutil.Process(pid)
    if q.create_time()==created and q.status()!=psutil.STATUS_ZOMBIE:alive.append(q)
   except psutil.NoSuchProcess:pass
  if not alive:break
  time.sleep(.5)
 forced=[]
 for q in alive:
  try:q.kill();forced.append(q.pid)
  except psutil.NoSuchProcess:pass
 return {'role':role,'owned':owned,'forced':forced,'command':cmd}
closed=[]
for role,port in [('engine_a',51801),('engine_b',51803),('master',52091)]:
 closed.append(stop(role,port));print('Stopped',role,flush=True)
save('final-cleanup.json',{'closed':closed,'observer_pids':[p.pid for p in observers],'wall_time':time.time()})
