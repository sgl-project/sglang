import argparse, json, os
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--gpu',required=True,type=int)
p.add_argument('--port',required=True,type=int)
p.add_argument('--engine-id',required=True)
a=p.parse_args()
root=Path('/home/youwei/bzh/project/TokenCake/RFC/sglang-rfc')
snapshot=Path('/tmp/sglang-rfc-build/runtime-worktree')
os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
os.environ['SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND']='python'
os.environ['PYTHONPATH']='/home/youwei/.cache/sglang-rfc-tools/mooncake:'+str(snapshot/'python')
backend={'local_hostname':'127.0.0.1','metadata_server':'http://127.0.0.1:52093/metadata','master_server_address':'127.0.0.1:52091','master_metrics_port':52092,'protocol':'tcp','global_segment_size':'4GB','check_server':False,'standalone_storage':False}
control={'namespace':'qwen3-8b-bf16-v1','engine_id':a.engine_id,'policies':[]}
command=[str(root/'.venv/bin/python'),'-m','sglang.launch_server','--model-path','/home/youwei/bzh/model/Qwen/Qwen3-8B','--host','127.0.0.1','--port',str(a.port),'--tp-size','1','--attention-backend','triton','--sampling-backend','pytorch','--disable-overlap-schedule','--cuda-graph-backend-decode','disabled','--cuda-graph-backend-prefill','disabled','--mem-fraction-static','0.45','--max-total-tokens','32768','--context-length','32768','--max-running-requests','16','--chunked-prefill-size','4096','--page-size','64','--random-seed','0','--enable-hierarchical-cache','--hicache-ratio','2','--hicache-io-backend','direct','--hicache-mem-layout','page_first_direct','--hicache-write-policy','write_through','--hicache-storage-backend','mooncake','--hicache-storage-prefetch-policy','wait_complete','--hicache-storage-backend-extra-config',json.dumps(backend),'--kv-control-config',json.dumps(control)]
print('Starting',a.engine_id,'GPU',a.gpu,'port',a.port,'source',snapshot,flush=True)
os.execv(command[0],command)
