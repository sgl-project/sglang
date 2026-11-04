import gzip,hashlib,json,pathlib,shutil
root=pathlib.Path('/home/youwei/bzh/project/TokenCake/RFC/sglang-rfc')
base=pathlib.Path('/tmp/sglang-rfc-build/mem08')
dest=root/'benchmark/session_kv_control/results/2026-11-04-mem08'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def packed(src,dst):
 with src.open('rb') as inp,dst.open('wb') as out:
  with gzip.GzipFile(filename='',mode='wb',fileobj=out,mtime=0) as z:shutil.copyfileobj(inp,z)
for name in [f'l3-c{c}-natural-baseline' for c in [8,16,32]]:
 source=base/name;target=dest/name
 assert not target.exists()
 original=json.loads((source/'checksums.json').read_text())
 for file,expected in original.items():assert sha(source/file)==expected,(name,file)
 target.mkdir()
 for p in source.iterdir():
  if not p.is_file() or p.name=='checksums.json':continue
  if p.name=='raw.jsonl' or (p.suffix=='.json' and p.stat().st_size>32768):packed(p,target/(p.name+'.gz'))
  else:shutil.copy2(p,target/p.name)
 shutil.copy2(source/'checksums.json',target/'checksums-original.json')
 (target/'checksums.json').write_text(json.dumps({p.name:sha(p) for p in sorted(target.iterdir()) if p.is_file()},indent=2)+'\n')
 for suffix in ['-command.json','.log']:shutil.copy2(base/(name+suffix),dest/'provenance'/(name+suffix))
 print('Archived',name,flush=True)
for name in ['baseline-reviewed.json','native-queued-regressions-root.log','native-proof-manifest.json','native-queued-root-comparison.json','native-queued-before-root.log','native-queued-after-root.log','native_queued_demand_repro.py','native-gpu-integration.log','idle-before-transition.json','old-engine-shutdown.json','optimized-runner-initial-stop.log']:
 if (base/name).exists():shutil.copy2(base/name,dest/'provenance'/name)
for name in ['engine-a-metrics.log','engine-b-metrics.log','master.log']:
 packed(base/name,dest/'provenance'/(name+'.gz'))
