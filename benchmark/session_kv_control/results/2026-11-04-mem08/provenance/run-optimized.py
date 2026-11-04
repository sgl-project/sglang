"""Wait for complete baseline, verify on real GPU/Mooncake, replace only owned Engines, run matched matrix."""
import json,os,pathlib,psutil,re,signal,subprocess,sys,time,urllib.request
root=pathlib.Path('/home/youwei/bzh/project/TokenCake/RFC/sglang-rfc')
base=pathlib.Path('/tmp/sglang-rfc-build/mem08')
snapshot=pathlib.Path('/tmp/sglang-rfc-build/mem08-runtime-optimized')

def save(path,record):
 temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(record,indent=2)+'\n');temp.replace(path)

def get(port,path):
 with urllib.request.urlopen(f'http://127.0.0.1:{port}{path}',timeout=10) as r:
  data=r.read().decode()
 return json.loads(data) if not path.endswith('/metrics') else data

print('WAIT baseline c32 completion',flush=True)
cell=base/'l3-c32-natural-baseline'
while not (cell/'checksums.json').is_file():
 if (cell/'failure.json').exists(): raise RuntimeError('baseline c32 failed; inspect before transition')
 time.sleep(2)
summary=json.loads((cell/'summary.json').read_text())
assert len(summary['arms'])==6, 'baseline matrix incomplete'
if not all(a['valid'] for a in summary['arms']):
 review=json.loads((base/'baseline-reviewed.json').read_text())
 actual=[]
 for spec in json.loads((cell/'schedule.json').read_text()):
  arm=json.loads((cell/(spec['id']+'.json')).read_text())
  assert len(arm['rows'])==128 and all(r.get('generation') and r.get('output_equal') is True for r in arm['rows'])
  actual.extend(r['id'] for r in arm['rows'] if not r['valid'])
 assert sorted(actual)==sorted(review['expected_invalid_ids']), 'baseline failures differ from reviewed evidence'
 print('KEEP reviewed baseline control failure',actual,flush=True)
loads={str(p):get(p,'/v1/loads?include=core,queues') for p in [51801,51803]}
assert all(s['num_running_reqs']==0 and s['num_waiting_reqs']==0 and all(v==0 for v in s['queues'].values()) for d in loads.values() for s in d['loads'])
save(base/'idle-before-transition.json',loads)
print('START real GPU/file/Mooncake integration proof',flush=True)
env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','PYTHONPATH':'/home/youwei/.cache/sglang-rfc-tools/mooncake:'+str(snapshot/'python'),'KV_CONTROL_MOONCAKE_CONFIG':json.dumps({'master_server_address':'127.0.0.1:52091','metadata_server':'http://127.0.0.1:52093/metadata','master_metrics_port':52092,'local_hostname':'127.0.0.1','protocol':'tcp','standalone_storage':False,'check_server':False,'tenant_id':'default','global_segment_size':67108864})}
with (base/'native-gpu-integration.log').open('x') as out:
 proof=subprocess.run([str(root/'.venv/bin/python'),'-m','pytest','-q','--tb=short',str(snapshot/'test/registered/hicache/test_programmable_kv_cache.py')],cwd=snapshot,env=env,stdout=out,stderr=subprocess.STDOUT)
assert proof.returncode==0,'real GPU proof failed; do not replace baseline'
print('PASS real GPU integration; stopping owned idle Engines',flush=True)
old=json.loads((base/'task-processes.json').read_text())
owned=[]
for role,port in [('engine_a',51801),('engine_b',51803)]:
 p=psutil.Process(old[role]);cmd=p.cmdline()
 assert '-m' in cmd and 'sglang.launch_server' in cmd and cmd[cmd.index('--port')+1]==str(port), 'process identity changed'
 owned.extend([{'pid':x.pid,'created':x.create_time()} for x in [p]+p.children(recursive=True)])
 p.terminate()
end=time.monotonic()+60
while time.monotonic()<end:
 alive=[]
 for item in owned:
  try:
   p=psutil.Process(item['pid'])
   if p.create_time()==item['created'] and p.status()!=psutil.STATUS_ZOMBIE: alive.append(p)
  except psutil.NoSuchProcess: pass
 if not alive:break
 time.sleep(1)
for p in alive:p.kill()
save(base/'old-engine-shutdown.json',{'owned':owned,'forced_remaining_pids':[p.pid for p in alive],'wall_time':time.time()})
processes={}
for role,gpu,port,eid in [('engine_a',0,51801,'worker-a'),('engine_b',1,51803,'worker-b')]:
 out=(base/(role+'-optimized.log')).open('x')
 cmd=[str(root/'.venv/bin/python'),str(base/'start-engine-optimized.py'),'--gpu',str(gpu),'--port',str(port),'--engine-id',eid]
 p=subprocess.Popen(cmd,cwd=root,stdout=out,stderr=subprocess.STDOUT);out.close();processes[role]=p
 save(base/'task-processes.json',{**old,**{k:v.pid for k,v in processes.items()}})
 print('START',role,'PID',p.pid,flush=True)
end=time.monotonic()+300
ready={}
while time.monotonic()<end:
 for role,port in [('engine_a',51801),('engine_b',51803)]:
  if role in ready:continue
  assert processes[role].poll() is None,role+' exited during startup'
  try:
   caps=get(port,'/v1/kv/capabilities');info=get(port,'/server_info');metrics=get(port,'/metrics')
   match=re.search(r'^sglang:hicache_host_total_tokens(?:\{[^\n]*\})?\s+(\S+)',metrics,re.M)
   if not caps.get('success') or not caps['data'].get('enabled') or match is None or float(match[1])<=0:continue
   assert info['mem_fraction_static']==.8 and info['max_total_tokens'] is None
   ready[role]={'capabilities':caps,'server':info,'host_capacity_tokens':int(float(match[1])),'command':pathlib.Path(f'/proc/{processes[role].pid}/cmdline').read_bytes().decode().split('\0')[:-1]}
  except (OSError,ValueError): pass
 if len(ready)==2:break
 time.sleep(2)
assert len(ready)==2,'optimized Engines not ready in startup window'
save(base/'optimized-boot.json',{'engine_source':subprocess.check_output(['git','rev-parse','HEAD'],cwd=snapshot,text=True).strip(),'engines':ready,'wall_time':time.time()})
host=ready['engine_b']['host_capacity_tokens']
for concurrency in [8,16,32]:
 name=f'l3-c{concurrency}-natural-optimized'
 cmd=[str(root/'.venv/bin/python'),str(root/'benchmark/session_kv_control/stress_bench.py'),'l3','--engine-a','http://127.0.0.1:51801','--engine-b','http://127.0.0.1:51803','--model','/home/youwei/bzh/model/Qwen/Qwen3-8B','--host-capacity-tokens',str(host),'--length','16384','--output-tokens','32','--seed','1729','--timeout-s',str(180 if concurrency==32 else 120),'--retention-mode','natural','--blocks','3','--sessions','64','--passes','2','--concurrency',str(concurrency),'--tool-wait-s','3','--lead-s','2.5','--output',str(base/name)]
 record={'name':name,'command':cmd,'started_wall_time':time.time(),'engine_source':subprocess.check_output(['git','rev-parse','HEAD'],cwd=snapshot,text=True).strip(),'tool_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()}
 save(base/(name+'-command.json'),record);print('START',name,flush=True)
 with (base/(name+'.log')).open('x') as out:result=subprocess.run(cmd,cwd=root,stdout=out,stderr=subprocess.STDOUT)
 record.update(exit_code=result.returncode,finished_wall_time=time.time());save(base/(name+'-command.json'),record)
 print('END',name,'exit',result.returncode,'elapsed',record['finished_wall_time']-record['started_wall_time'],flush=True)
 if result.returncode!=0:sys.exit(result.returncode)
print('OPTIMIZED MATRIX COMPLETE; services left idle for root inspection',flush=True)
