import asyncio,aiohttp,json,time,statistics,sys,os
from pathlib import Path
from transformers import AutoTokenizer
D=Path(os.environ["MODEL_DIR"])
T=AutoTokenizer.from_pretrained(str(D/'Qwen3-4B'))
base=T.encode("This is a reproducible counting exercise. Continue writing consecutive integers separated by commas. ",add_special_tokens=False)
tail=T.encode("\nCount from 1 to 1000. Output only the numbers, separated by commas.\n1,2,3,",add_special_tokens=False)
prompts=[]
for i in range(128):
 prefix=T.encode(f"Exercise {i:04d}. ",add_special_tokens=False)
 ids=prefix+(base*40)[:256-len(prefix)-len(tail)]+tail
 assert len(ids)==256
 prompts.append(ids)
inert=["velvet platypus sentinel "+str(i).zfill(3) for i in range(100)]
def words_for(case,i):
 if case=='varied100':return [('velvet '*(j%16+1))+'platypus sentinel '+str(j).zfill(3) for j in range(100)]
 if case=='none':return []
 if case=='inert1':return inert[:1]
 if case in ['inert100','ids100']:return inert
 if case=='mixed100':return inert if i%2==0 else []
 if case=='active1':return ['10']
 raise ValueError(case)
def encode(words):
 seqs=[]
 for word in words:
  plain=T.encode(word.lstrip(),add_special_tokens=False);seqs.append(plain)
  spaced=T.encode(' '+word.lstrip(),add_special_tokens=False)
  if spaced and spaced[0]!=plain[0] and len(spaced)==len(plain):seqs.append(spaced)
 return seqs
def pct(values,p):
 v=sorted(values);x=(len(v)-1)*p;lo=int(x);hi=min(lo+1,len(v)-1)
 return v[lo]+(v[hi]-v[lo])*(x-lo)
from sglang.srt.sampling.custom_logit_processor import BadWordsLogitsProcessor
ID_WORDS=encode(inert)
SERIALIZED=BadWordsLogitsProcessor.to_str()
async def batch(case,c,n,output_tokens):
 sem=asyncio.Semaphore(c);results=[]
 async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180),connector=aiohttp.TCPConnector(limit=c)) as session:
  async def req(i):
   async with sem:
    words=words_for(case,i)
    sampling=dict(temperature=0,max_new_tokens=output_tokens,ignore_eos=True)
    if words and case!='ids100':sampling['bad_words']=words
    if case=='ids100':sampling['custom_params']={'bad_words_token_ids':ID_WORDS}
    payload=dict(input_ids=prompts[i%len(prompts)],sampling_params=sampling,stream=True)
    if case=='ids100':payload['custom_logit_processor']=SERIALIZED
    start=time.perf_counter();first=None;last=None;events=[]
    async with session.post('http://127.0.0.1:'+os.environ.get('TEST_PORT','31082')+'/generate',json=payload) as response:
     if response.status!=200:raise RuntimeError((response.status,await response.text()))
     async for line in response.content:
      if not line.startswith(b'data: ') or line.strip()==b'data: [DONE]':continue
      data=json.loads(line[6:]);now=time.perf_counter()
      count=data.get('meta_info',{}).get('completion_tokens',0)
      if count>0:
       if first is None:first=now
       events.append([now-start,count])
       last=data
    end=time.perf_counter()
    assert last and first is not None
    ids=last['output_ids'];meta=last['meta_info']
    assert len(ids)==output_tokens,(len(ids),output_tokens,meta)
    assert meta['prompt_tokens']==256,meta
    return dict(index=i,has_bad_words=bool(words),ttft_ms=(first-start)*1000,e2e_ms=(end-start)*1000,tpot_ms=(end-first)*1000/(len(ids)-1),output_tokens=len(ids),ids=ids,meta=meta,events=events)
  start=time.perf_counter();results=await asyncio.gather(*(req(i) for i in range(n)));elapsed=time.perf_counter()-start
 # Validate after stopping the clock; tokenization/checking must not reduce client load.
 seqs=encode(words_for(case,0))
 for row in results:
  if not row['has_bad_words']:continue
  ids=row['ids']
  check_seqs=encode(words_for(case,row['index'])) if case=='cold100' else seqs
  for seq in check_seqs:
   assert all(ids[j:j+len(seq)]!=seq for j in range(len(ids)-len(seq)+1)),(case,seq,ids)
 return results,elapsed

async def main():
 label=sys.argv[1];repeat=int(sys.argv[2])
 import subprocess
 gpu_start=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid,memory.used,utilization.gpu,temperature.gpu,clocks.sm,power.draw','--format=csv']).decode()
 OUT=Path(os.environ.get("RESULT_DIR", "./badwords-results"))/label;OUT.mkdir(parents=True,exist_ok=True)
 (OUT/f'gpu-before-{repeat}.txt').write_text(gpu_start)
 configs=[(c,n,case) for c,n in [(1,16),(8,64),(32,192)] for case in ['none','inert100','mixed100','varied100','active1']]
 for c,n,case in (configs if repeat==1 else list(reversed(configs))):
  await batch(case,c,c,32)
  rows,elapsed=await batch(case,c,n,128)
  summary=dict(label=label,case=case,concurrency=c,repeat=repeat,requests=len(rows),wall_s=elapsed,
    output_tok_s=sum(r['output_tokens'] for r in rows)/elapsed,
    ttft_ms_p95=pct([r['ttft_ms'] for r in rows],.95),
    tpot_ms_mean=statistics.mean(r['tpot_ms'] for r in rows),
    tpot_ms_p95=pct([r['tpot_ms'] for r in rows],.95),
    tokens_per_verify=(sum(r['output_tokens'] for r in rows)/sum(r['meta'].get('spec_verify_ct',0) for r in rows) if sum(r['meta'].get('spec_verify_ct',0) for r in rows) else None))
  (OUT/f'{case}-c{c}-{repeat}.json').write_text(json.dumps(dict(summary=summary,requests=rows),indent=2))
  print(json.dumps(summary),flush=True)
if __name__ == '__main__':
 asyncio.run(main())
