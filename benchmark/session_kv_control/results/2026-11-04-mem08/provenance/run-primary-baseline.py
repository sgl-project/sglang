import json,subprocess,time,pathlib,sys,psutil
root=pathlib.Path('/home/youwei/bzh/project/TokenCake/RFC/sglang-rfc')
base=pathlib.Path('/tmp/sglang-rfc-build/mem08')
previous=1342087
while psutil.pid_exists(previous):
 try:
  process=psutil.Process(previous)
  if process.status()==psutil.STATUS_ZOMBIE or not any('stress_bench.py' in a for a in process.cmdline()): break
 except psutil.NoSuchProcess: break
 time.sleep(2)
old=base/'l2-pressure-baseline-command.json'
record=json.loads(old.read_text())
summary=json.loads((base/'l2-pressure-baseline/summary.json').read_text())
assert len(summary['arms'])==6, 'prior cell did not complete all six arms'
record.update(exit_code=0 if all(a['valid'] for a in summary['arms']) else 2,finished_wall_time=(base/'l2-pressure-baseline/checksums.json').stat().st_mtime,completion_observation='root recovered result from complete six-arm summary and final checksums after parent runner transition')
old.write_text(json.dumps(record,indent=2)+'\n')
common=[str(root/'.venv/bin/python'),str(root/'benchmark/session_kv_control/stress_bench.py')]
args=['--engine-a','http://127.0.0.1:51801','--engine-b','http://127.0.0.1:51803','--model','/home/youwei/bzh/model/Qwen/Qwen3-8B','--host-capacity-tokens','693952','--length','16384','--output-tokens','32','--seed','1729','--timeout-s','120','--retention-mode','natural']
cells=[('l2-natural-baseline','l2',['--rounds','3','--hot-sessions','4','--pressure-prefixes','53','--concurrency','8'])]+[(f'l3-c{c}-natural-baseline','l3',['--blocks','3','--sessions','64','--passes','2','--concurrency',str(c),'--tool-wait-s','3','--lead-s','2.5']) for c in (8,16)]
for name,mode,extra in cells:
 cmd=common+[mode]+args+extra+['--output',str(base/name)]
 record={'name':name,'command':cmd,'started_wall_time':time.time(),'engine_source':'f65f0393c4c916184c6c9017fafb9fa389ec2f4a','tool_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()}
 (base/(name+'-command.json')).write_text(json.dumps(record,indent=2)+'\n')
 print('START',name,flush=True)
 with (base/(name+'.log')).open('x') as output: result=subprocess.run(cmd,cwd=root,stdout=output,stderr=subprocess.STDOUT)
 record.update(exit_code=result.returncode,finished_wall_time=time.time())
 (base/(name+'-command.json')).write_text(json.dumps(record,indent=2)+'\n')
 print('END',name,'exit',result.returncode,'elapsed',record['finished_wall_time']-record['started_wall_time'],flush=True)
 if result.returncode != 0: sys.exit(result.returncode)
 if (base/'STOP_AFTER_CURRENT').exists(): sys.exit(3)
