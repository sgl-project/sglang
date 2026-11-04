"""Sample cumulative CPU counters for explicitly owned service process trees."""
import json,psutil,signal,threading,time,pathlib
base=pathlib.Path('/tmp/sglang-rfc-build/mem08')
stop=threading.Event()
for s in [signal.SIGINT,signal.SIGTERM]: signal.signal(s,lambda *_:stop.set())
with (base/'process-resources.jsonl').open('x',buffering=1) as out:
 while not stop.is_set():
  row={'wall_time':time.time(),'monotonic':time.monotonic(),'roles':{}}
  for name,pid in json.loads((base/'task-processes.json').read_text()).items():
   try:
    parent=psutil.Process(pid)
    rows=[]
    for p in [parent]+parent.children(recursive=True):
     try:
      with p.oneshot():
       cpu=p.cpu_times();m=p.memory_info()
       rows.append({'pid':p.pid,'created':p.create_time(),'user_s':cpu.user,'system_s':cpu.system,'rss':m.rss,'vms':m.vms,'threads':p.num_threads(),'status':p.status()})
     except psutil.Error as e: rows.append({'pid':p.pid,'error':str(e)})
    row['roles'][name]={'root_pid':pid,'processes':rows}
   except psutil.Error as e:row['roles'][name]={'root_pid':pid,'error':str(e)}
  out.write(json.dumps(row)+'\n')
  stop.wait(2)
