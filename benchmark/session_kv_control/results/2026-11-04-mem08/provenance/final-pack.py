"""Deterministically archive completed cells and all sampled resource records."""
import gzip,hashlib,importlib.metadata,json,pathlib,shutil,subprocess,sys,time
root=pathlib.Path('/home/youwei/bzh/project/TokenCake/RFC/sglang-rfc')
base=pathlib.Path('/tmp/sglang-rfc-build/mem08')
dest=root/'benchmark/session_kv_control/results/2026-11-04-mem08'
prov=dest/'provenance'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def pack(source,target):
 with source.open('rb') as inp,target.open('wb') as out:
  with gzip.GzipFile(filename='',mode='wb',fileobj=out,mtime=0,compresslevel=6) as z:shutil.copyfileobj(inp,z)
for name in [f'l3-c{c}-natural-optimized' for c in [8,16,32]]:
 source=base/name;target=dest/name;assert not target.exists()
 original=json.loads((source/'checksums.json').read_text())
 for file,expected in original.items():assert sha(source/file)==expected,(name,file)
 target.mkdir()
 for p in source.iterdir():
  if not p.is_file() or p.name=='checksums.json':continue
  if p.name=='raw.jsonl' or (p.suffix=='.json' and p.stat().st_size>32768):pack(p,target/(p.name+'.gz'))
  else:shutil.copy2(p,target/p.name)
 shutil.copy2(source/'checksums.json',target/'checksums-original.json')
 (target/'checksums.json').write_text(json.dumps({p.name:sha(p) for p in sorted(target.iterdir()) if p.is_file()},indent=2)+'\n')
 print('Archived',name,flush=True)
# The first commit's L2-only view is retained verbatim; full capture uses bounded parts.
clip=dest/'resources.jsonl.gz'
assert clip.is_file()
clip.rename(prov/'resources-l2-clipped.jsonl.gz')
parts=[];full=hashlib.sha256();out=None;z=None;records=0;first=None;last=None;name=None
with (base/'resources.jsonl').open('rb') as inp:
 for line in inp:
  row=json.loads(line);full.update(line)
  if records%512==0:
   if z is not None:
    z.close();out.close();parts.append({'file':name,'sha256':sha(dest/name),'records':512,'first_monotonic':first,'last_monotonic':last})
   name=f'resources-{len(parts):03d}.jsonl.gz';out=(dest/name).open('wb');z=gzip.GzipFile(filename='',mode='wb',fileobj=out,mtime=0,compresslevel=6);first=row['monotonic']
  z.write(line);last=row['monotonic'];records+=1
if z is not None:
 z.close();out.close();parts.append({'file':name,'sha256':sha(dest/name),'records':records-512*len(parts),'first_monotonic':first,'last_monotonic':last})
assert full.hexdigest()==sha(base/'resources.jsonl')
(dest/'resources-index.json').write_text(json.dumps({'parts':parts,'records':records,'uncompressed_full_sha256':full.hexdigest(),'definition':'all original complete JSONL observer records in original order; deterministic gzip split at 512 records; no metric filtering'},indent=2)+'\n')
# Prove lossless reconstruction, independently of the part checksum index.
reconstructed=hashlib.sha256()
for part in parts:
 with gzip.open(dest/part['file'],'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):reconstructed.update(b)
assert reconstructed.hexdigest()==full.hexdigest()
pack(base/'process-resources.jsonl',dest/'process-resources.jsonl.gz')
for p in sorted(base.iterdir()):
 if p.is_file() and (p.suffix in ['.py','.log','.json'] or p.name=='CHECKPOINT.md') and p.name not in ['resources.jsonl','process-resources.jsonl']:
  if p.name.startswith('engine') or p.name=='master.log':pack(p,prov/(p.name+'.gz'))
  else:shutil.copy2(p,prov/p.name)
# Separate startup/function-only observations from formal cells.
observations=dest/'startup-observations';observations.mkdir()
for name in ['calibration-0','l3-smoke','l3-smoke-v2','l2-smoke']:
 source=base/name;target=observations/name
 if not source.exists():continue
 target.mkdir()
 for p in source.iterdir():
  if not p.is_file() or p.name=='checksums.json':continue
  if p.name=='raw.jsonl' or (p.suffix=='.json' and p.stat().st_size>32768):pack(p,target/(p.name+'.gz'))
  else:shutil.copy2(p,target/p.name)
 if (source/'checksums.json').exists():shutil.copy2(source/'checksums.json',target/'checksums-original.json')
 (target/'checksums.json').write_text(json.dumps({p.name:sha(p) for p in sorted(target.iterdir()) if p.is_file()},indent=2)+'\n')
versions={}
for name in ['torch','transformers','msgspec','numpy','matplotlib','mooncake-transfer-engine-cuda13','sgl-kernel']:
 try:versions[name]=importlib.metadata.version(name)
 except importlib.metadata.PackageNotFoundError:versions[name]=None
(prov/'pack-environment.json').write_text(json.dumps({'python':sys.version,'versions_distribution_lookup':versions,'runtime_old':'f65f0393c4c916184c6c9017fafb9fa389ec2f4a','runtime_optimized':'ff764308fbb40de10f8e3983cd01d3f54946863e','pack_tool_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'observer_records':records,'observer_parts':len(parts),'source_global_resource_sha256':full.hexdigest(),'lossless_reconstructed_sha256':reconstructed.hexdigest(),'wall_time':time.time()},indent=2)+'\n')
print('All captures archived losslessly:',records,'samples in',len(parts),'parts',flush=True)
