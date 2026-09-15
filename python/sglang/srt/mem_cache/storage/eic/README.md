# EIC as sglang HiCache Storage
EIC(Elastic Instant Cache) is a distributed database designed for LLM KV Cache. It supports RDMA, GDR and has the capabilities of distributed disaster tolerance and expansion.
You can understand the principles and architecture of EIC through these articles: https://mp.weixin.qq.com/s/tasDqXf0Gxr3o_WCJ2IJUQ https://mp.weixin.qq.com/s/b_4YhTa96Zeklh23lv8qBw


## Deploy EIC
You can visit the official link https://console.volcengine.com/eic and deploy EIC KVCache on your compute cluster with web UI.In addition, we provide particular image in volcano engine, which integrates various optimizations based on the official image.
You may use test_unit.py to detect the connectivity of EIC.



## Deploy Model With EIC
You can enable EIC KVCache offload with the official interface, such as

```bash
python -m sglang.launch_server \
    --model-path [model_path]
    --enable-hierarchical-cache \
    --hicache-storage-backend eic \
    --hicache-write-policy 'write_through' \
    --hicache-mem-layout 'page_first' \

```
For more details, you can see https://www.volcengine.com/docs/85848/1749188 .
