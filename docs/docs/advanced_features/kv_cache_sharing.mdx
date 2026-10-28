---
title: "KV Cache Sharing"
metatags:
    description: "Page-based KV cache sharding for prefill context parallelism, CPU HiCache, and P/D serving."
---

KV cache sharing reduces duplicate KV storage across GPUs in a prefill worker. It keeps one persistent copy of each cache page within the shard group, allowing the same GPU memory budget to retain more unique tokens.

[Prefill context parallelism (CP)](/docs/advanced_features/prefill_cp) distributes prompt computation, but can still leave the cached history replicated across ranks. KV cache sharing distributes those stored pages as well. This is useful for long-context and multi-turn workloads that reuse large prefixes.

## How it works

### Shared logical tree, distributed pages

Every rank keeps the same logical radix tree and request-to-token mapping. A prefix match therefore identifies the same cached tokens on all ranks, while each GPU stores only the pages it owns.

For example, a CP4 group can distribute eight logical pages as follows:

| Rank | Stored pages |
| --- | --- |
| 0 | A, E |
| 1 | B, F |
| 2 | C, G |
| 3 | D, H |

A request that reuses A–D matches the same logical prefix on every rank. New pages continue the cached path's ownership sequence, and prefixes remain reusable at the original page size.

### Gather one layer at a time

Before attention reads a cached prefix, each rank contributes its owned pages to an all-gather. Attention reads the assembled prefix from a temporary layer buffer, using a page-table mapping to preserve the logical token order.

Two buffers alternate across layers so the next gather can overlap with current-layer computation. Their memory is included in the cache budget. A prefix hit saves recomputation, but still incurs the cost of gathering the distributed KV.

### HiCache and MTP integration

[HiCache](/docs/advanced_features/hicache) preserves page ownership during GPU-to-host backup and host-to-GPU reload. Each rank backs up and restores its own pages. For each layer, execution follows this order:

```text Read order
Restore local pages → Gather the prefix → Run attention
```

With GLM MTP, the target and draft KV/indexer state follow the same ownership plan and are restored together. During [P/D transfer](/docs/advanced_features/pd_disaggregation), each prefill rank sends the pages it owns; decode receives complete, unsharded cache entries.

## Usage example

Enable KV cache sharing with `--enable-kv-cache-sharding` on the prefill worker only.

The following example uses Linux with NVIDIA CUDA on two GB300 nodes, with four GPUs per worker. Prefill uses TP4/CP4 and CPU HiCache; decode uses TP4/EP4 with attention DP4. Both workers use the same GLM-5.3 NVFP4 checkpoint and model revision.

### Environment variables

Create `kv-shard.env` on both nodes. Replace the example IP addresses and RDMA NIC names with values for your environment. This example uses Mooncake RDMA for P/D transfer.

```bash Environment variables
cat > kv-shard.env <<'EOF'
export MODEL_PATH=RadixArk/GLM-5.3-NVFP4
export MODEL_REVISION=6e389189d564d9d3e4ea7fa284fe91f136d8ae2a
export PREFILL_HOST=192.0.2.10
export DECODE_HOST=192.0.2.20
export PREFILL_IB_DEVICES=mlx5_0,mlx5_1,mlx5_2,mlx5_3
export DECODE_IB_DEVICES=mlx5_0,mlx5_1,mlx5_2,mlx5_3
export CUDA_VISIBLE_DEVICES=0,1,2,3
export SGLANG_DISAGG_STAGING_BUFFER=0
export MC_TCP_ENABLE_CONNECTION_POOL=true
unset MC_FORCE_MNNVL SGLANG_MOONCAKE_CUSTOM_MEM_POOL
EOF
```

### Prefill worker: TP4/CP4

Run this command on the prefill node. It enables interleave CP, page-based KV sharding, CPU HiCache, and EAGLE MTP.

```bash Prefill worker
source ./kv-shard.env
export SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python
python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --revision "$MODEL_REVISION" \
  --served-model-name glm-kv-shard \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 30001 \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-bootstrap-port 30200 \
  --disaggregation-ib-device "$PREFILL_IB_DEVICES" \
  --nccl-port 29501 \
  --tp 4 \
  --base-gpu-id 0 \
  --enable-prefill-cp \
  --attn-cp-size 4 \
  --cp-strategy interleave \
  --enable-kv-cache-sharding \
  --dsa-prefill-backend trtllm \
  --dsa-decode-backend trtllm \
  --kv-cache-dtype fp8_e4m3 \
  --page-size 64 \
  --mem-fraction-static 0.9 \
  --chunked-prefill-size 16384 \
  --max-running-requests 32 \
  --enable-hierarchical-cache \
  --hicache-size 240 \
  --hicache-host-memory-mode cache \
  --hicache-write-policy write_through \
  --hicache-io-backend kernel \
  --hicache-mem-layout page_first \
  --bf16-gemm-backend cutedsl \
  --moe-runner-backend flashinfer_cutlass \
  --disable-shared-experts-fusion \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --enable-metrics \
  --enable-metrics-for-all-schedulers \
  --enable-cache-report
```

### Decode worker: TP4/EP4

Run this command on the decode node. The target uses attention DP4 and FlashInfer A2A with EP4; the EAGLE draft uses the TP-only path. Decode does not enable KV cache sharding.

```bash Decode worker
source ./kv-shard.env
python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --revision "$MODEL_REVISION" \
  --served-model-name glm-kv-shard \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 30003 \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-bootstrap-port 30200 \
  --disaggregation-ib-device "$DECODE_IB_DEVICES" \
  --nccl-port 31000 \
  --tp 4 \
  --base-gpu-id 0 \
  --attn-dp-size 4 \
  --ep-size 4 \
  --enable-dp-lm-head \
  --enable-dp-attention-local-control-broadcast \
  --dsa-prefill-backend trtllm \
  --dsa-decode-backend trtllm \
  --kv-cache-dtype fp8_e4m3 \
  --page-size 64 \
  --mem-fraction-static 0.9 \
  --max-running-requests 512 \
  --disable-radix-cache \
  --bf16-gemm-backend cutedsl \
  --moe-a2a-backend flashinfer \
  --moe-runner-backend flashinfer_trtllm_routed \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --speculative-moe-a2a-backend none \
  --speculative-moe-runner-backend flashinfer_trtllm \
  --enable-metrics \
  --enable-metrics-for-all-schedulers \
  --enable-cache-report
```

### Router

After both workers are healthy, start the router on a host that can reach them. The worker HTTP endpoints, prefill bootstrap port, and RDMA transport must be reachable between nodes.

```bash Router
source ./kv-shard.env
curl --fail "http://${PREFILL_HOST}:30001/health"
curl --fail "http://${DECODE_HOST}:30003/health"

python -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill "http://${PREFILL_HOST}:30001" 30200 \
  --decode "http://${DECODE_HOST}:30003" \
  --policy cache_aware \
  --host 0.0.0.0 \
  --port 30000
```

### Send a request

From the router host, send a request using the shared served model name:

```bash Request
curl --fail http://127.0.0.1:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm-kv-shard","messages":[{"role":"user","content":"What is 17 + 25?"}],"temperature":0,"max_tokens":128}'
```

## Related documentation

- [Prefill context parallelism](/docs/advanced_features/prefill_cp)
- [HiCache design](/docs/advanced_features/hicache_design)
- [P/D disaggregation](/docs/advanced_features/pd_disaggregation)
- [GLM-5.3 cookbook](/cookbook/autoregressive/GLM/GLM-5.3)
