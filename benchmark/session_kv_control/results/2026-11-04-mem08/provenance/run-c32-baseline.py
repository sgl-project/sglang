import psutil,json,pathlib,time,subprocess,sys
root=pathlib.Path('/home/youwei/bzh/project/TokenCake/RFC/sglang-rfc')
base=pathlib.Path('/tmp/sglang-rfc-build/mem08')
pid=json.loads((base/'primary-runner-pid.json').read_text())['pid']
while psutil.pid_exists(pid):
 try:
  p=psutil.Process(pid)
  if p.status()==psutil.STATUS_ZOMBIE or not any('run-primary-baseline.py' in c for c in p.cmdline()): break
 except psutil.NoSuchProcess: break
 time.sleep(2)
p=base/'l3-c16-natural-baseline'
assert (p/'checksums.json').is_file(), 'primary baseline matrix incomplete'
assert all(a['valid'] for a in json.loads((p/'summary.json').read_text())['arms']), 'primary baseline invalid; inspect before next test'
name='l3-c32-natural-baseline'
cmd=[str(root/'.venv/bin/python'),str(root/'benchmark/session_kv_control/stress_bench.py'),'l3','--engine-a','http://127.0.0.1:51801','--engine-b','http://127.0.0.1:51803','--model','/home/youwei/bzh/model/Qwen/Qwen3-8B','--host-capacity-tokens','693952','--length','16384','--output-tokens','32','--seed','1729','--timeout-s','180','--retention-mode','natural','--blocks','3','--sessions','64','--passes','2','--concurrency','32','--tool-wait-s','3','--lead-s','2.5','--output',str(base/name)]
record={'name':name,'command':cmd,'started_wall_time':time.time(),'engine_source':'f65f0393c4c916184c6c9017fafb9fa389ec2f4a','tool_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()}
(base/(name+'-command.json')).write_text(json.dumps(record,indent=2)+'\n')
print('START',name,flush=True)
with (base/(name+'.log')).open('x') as output:result=subprocess.run(cmd,cwd=root,stdout=output,stderr=subprocess.STDOUT)
record.update(exit_code=result.returncode,finished_wall_time=time.time())
(base/(name+'-command.json')).write_text(json.dumps(record,indent=2)+'\n')
print('END',name,'exit',result.returncode,'elapsed',record['finished_wall_time']-record['started_wall_time'],flush=True)
sys.exit(result.returncode)
