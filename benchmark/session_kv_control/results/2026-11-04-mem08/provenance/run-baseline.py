import json,subprocess,time,pathlib,sys
root=pathlib.Path('/home/youwei/bzh/project/TokenCake/RFC/sglang-rfc')
base=pathlib.Path('/tmp/sglang-rfc-build/mem08')
common=[str(root/'.venv/bin/python'),str(root/'benchmark/session_kv_control/stress_bench.py')]
args=['--engine-a','http://127.0.0.1:51801','--engine-b','http://127.0.0.1:51803','--model','/home/youwei/bzh/model/Qwen/Qwen3-8B','--host-capacity-tokens','693952','--length','16384','--output-tokens','32','--seed','1729','--timeout-s','120']
cells=[('l2-pressure-baseline','l2',['--rounds','3','--hot-sessions','4','--pressure-prefixes','53','--concurrency','8'])]+[(f'l3-c{c}-baseline','l3',['--blocks','3','--sessions','64','--passes','2','--concurrency',str(c),'--tool-wait-s','3','--lead-s','2.5','--retention-mode','natural']) for c in (8,16)]
for name,mode,extra in cells:
 cmd=common+[mode]+args+extra+['--output',str(base/name)]
 record={'name':name,'command':cmd,'started_wall_time':time.time(),'engine_source':'f65f0393c4c916184c6c9017fafb9fa389ec2f4a','tool_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()}
 (base/(name+'-command.json')).write_text(json.dumps(record,indent=2)+'\n')
 print('START',name,flush=True)
 with (base/(name+'.log')).open('x') as output: result=subprocess.run(cmd,cwd=root,stdout=output,stderr=subprocess.STDOUT)
 record.update(exit_code=result.returncode,finished_wall_time=time.time())
 (base/(name+'-command.json')).write_text(json.dumps(record,indent=2)+'\n')
 print('END',name,'exit',result.returncode,'elapsed',record['finished_wall_time']-record['started_wall_time'],flush=True)
 if result.returncode not in (0,2): sys.exit(result.returncode)
