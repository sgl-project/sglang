import gzip,hashlib,json,pathlib,shutil
root=pathlib.Path('/home/youwei/bzh/project/TokenCake/RFC/sglang-rfc')
src=pathlib.Path('/tmp/sglang-rfc-build/mem08')
dst=root/'benchmark/session_kv_control/results/2026-11-04-mem08'
dst.mkdir(exist_ok=True)

def packed(source,target):
 with source.open('rb') as inp,target.open('wb') as out:
  with gzip.GzipFile(filename='',mode='wb',fileobj=out,mtime=0) as z:shutil.copyfileobj(inp,z)

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for name in ['l2-pressure-baseline','l2-natural-baseline']:
 source=src/name;target=dst/name;target.mkdir(exist_ok=False)
 original=json.loads((source/'checksums.json').read_text())
 for file,expected in original.items():assert sha(source/file)==expected,(name,file)
 for p in source.iterdir():
  if not p.is_file() or p.name=='checksums.json':continue
  if p.name=='raw.jsonl' or (p.suffix=='.json' and p.stat().st_size>32768):packed(p,target/(p.name+'.gz'))
  else:shutil.copy2(p,target/p.name)
 shutil.copy2(source/'checksums.json',target/'checksums-original.json')
 (target/'checksums.json').write_text(json.dumps({p.name:sha(p) for p in sorted(target.iterdir()) if p.is_file()},indent=2)+'\n')
 provenance=dst/'provenance';provenance.mkdir(exist_ok=True)
 shutil.copy2(src/(name+'-command.json'),provenance/(name+'-command.json'))
 shutil.copy2(src/(name+'.log'),provenance/(name+'.log'))
# Exact raw sampled records within the union of completed L2 arm windows.
analysis=json.loads(pathlib.Path('/tmp/sglang-rfc-build/mem08-analysis-l2-complete/analysis.json').read_text())
windows=[w for c in analysis['cells'] if c['state']=='completed' for w in c['arm_windows']]
lo=min(w['start'] for w in windows)-2;hi=max(w['end'] for w in windows)+2
count=0
with (src/'resources.jsonl').open() as inp,(dst/'resources.jsonl.gz').open('wb') as out:
 with gzip.GzipFile(filename='',mode='wb',fileobj=out,mtime=0) as z:
  for line in inp:
   try:r=json.loads(line)
   except json.JSONDecodeError:continue
   if lo<=r['monotonic']<=hi:z.write(line.encode());count+=1
(dst/'provenance/l2-resource-capture.json').write_text(json.dumps({'source':str(src/'resources.jsonl'),'selection':'exact original sampled JSONL records within union outer L2 bounds plus 2s; complete observer remains live for pending L3','monotonic_interval':[lo,hi],'samples':count,'sha256_compressed':sha(dst/'resources.jsonl.gz')},indent=2)+'\n')
for name in ['launch-commands.json','start-engine.py','harness-tests-main.log','harness-natural-tests-root.log','analysis-reset-tests-root.log']:
 shutil.copy2(src/name,dst/'provenance'/name)
print('Archived two verified L2 cells with',count,'sampled resource records',flush=True)
