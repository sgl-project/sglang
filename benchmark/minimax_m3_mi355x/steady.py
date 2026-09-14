"""Closed-loop steady-state decode throughput from the server's generation counter.

`N` workers each keep one long-form request in flight over a cached document; after the
warm-up the `generation_tokens_total` delta is measured over the window.
Usage: python steady.py URL N SECONDS [WARM=15]"""
import requests, glob, time, sys, re, threading, concurrent.futures as cf, random, os
url=sys.argv[1]; N=int(sys.argv[2]); WINDOW_SEC=float(sys.argv[3]); WARM_SEC=float(sys.argv[4]) if len(sys.argv)>4 else 15.0
files=sorted(glob.glob("/sgl-workspace/sglang/python/sglang/srt/**/*.py", recursive=True))
corpus="\n".join(open(f).read() for f in files)
# ~4.2 chars per token
CTX=int(os.environ.get("CTX_CHARS","300000"))
ASKS=["Write detailed documentation for every class and function in the dump above: one paragraph each, in order.",
      "Explain, function by function, what the code above does and list potential bugs with reasoning.",
      "Produce an exhaustive code review of the dump above, file by file, with concrete suggestions.",
      "Describe the control flow of the code above in depth, then write unit test plans for each module."]
def doc(i): off=i*(CTX+10000); return f"Code dump #{i}.\n\n"+corpus[off:off+CTX]+"\n\n"
def metric(name):
    m=requests.get(url+"/metrics", timeout=10).text
    return [float(x) for x in re.findall(r'sglang:%s\{[^}]*\} ([0-9.e+]+)'%name, m)]
def gen(i, ask, n):
    r=requests.post(url+"/generate", json=dict(text=doc(i)+ask, sampling_params=dict(max_new_tokens=n, temperature=0.7, top_p=0.95)), timeout=3600)
    return r.json()["meta_info"]["completion_tokens"]
if not os.environ.get("NOFLUSH"): requests.post(url+"/flush_cache")
with cf.ThreadPoolExecutor(N) as ex: list(ex.map(lambda i: gen(i, ASKS[0], 4), range(N)))
stop=threading.Event(); done=[0]*N; lens=[]
def worker(i):
    k=0
    while not stop.is_set():
        lens.append(gen(i, ASKS[(i+k)%len(ASKS)], 1200)); k+=1; done[i]+=1
threads=[threading.Thread(target=worker,args=(i,),daemon=True) for i in range(N)]
for t in threads: t.start()
time.sleep(WARM_SEC)
gen_before=sum(metric("generation_tokens_total")); t_start=time.time(); running_samples=[]
while time.time()-t_start<WINDOW_SEC:
    time.sleep(2.0); running_samples.append(max(metric("num_running_reqs")+[0]))
gen_after=sum(metric("generation_tokens_total")); t_end=time.time()
stop.set()
tps=(gen_after-gen_before)/(t_end-t_start); accept=metric("spec_accept_length")
print(f"STEADY N={N} gen_tok/s={tps:.0f} per_stream={tps/N:.1f} running(mean/min)={sum(running_samples)/len(running_samples):.1f}/{min(running_samples):.0f} accept_len(cum)={accept[0] if accept else 'na':.3} completions={sum(done)} mean_len={sum(lens)/max(1,len(lens)):.0f} window={t_end-t_start:.0f}s", flush=True)
