# Mooncake MNNVL/EGM and SGLang Direct Linker Deployment and Validation Guide

## 1. Overview

This guide describes the validated integration of Mooncake MNNVL/EGM with the
SGLang unified-cache external linker. It covers the complete path from NVIDIA
IMEX and NVLink Fabric Memory to Mooncake Store and SGLang GPU KV-cache reuse.

The integration provides two capabilities:

1. A distributed Mooncake EGM memory pool with directed cross-node object
   placement and MNNVL data transfer.
2. A native SGLang Direct Linker path that offloads GPU KV cache to Mooncake and
   restores it directly into the SGLang GPU KV pool.

```text
SGLang GPU KV Cache
        ↓ Direct Linker offload
Mooncake Store
        ↓ MNNVL / NVLink Fabric
Local or remote EGM Segment
        ↓ Direct Linker lookup + load-back
SGLang GPU KV Cache
```

The validated hardware consisted of two servers with four NVIDIA GB300 GPUs per
server. The SGLang functional validation used DeepSeek-V4.1-Flash with tensor
parallel size four.

## 2. Community dependencies

Mooncake support is provided by:

- [kvcache-ai/Mooncake#5161](https://github.com/kvcache-ai/Mooncake/pull/5161)
- [Mooncake #5161 commits](https://github.com/kvcache-ai/Mooncake/pull/5161/commits)

Mooncake PR #5161 provides:

- the strict MNNVL EGM Store;
- the `mnnvl` and `nvlink_host` Store protocols;
- physical requester placement;
- the `MC_PHYSICAL_GPU` interface;
- one EGM owner process per physical GPU;
- request-scoped CUDA producer-event ordering.

The SGLang integration is tracked by:

- [sgl-project/sglang#48088](https://github.com/sgl-project/sglang/pull/48088)

The SGLang change resolves and injects a physical GPU identity in every Direct
Linker worker before Mooncake Store initialization.

## 3. Architecture

### 3.1 Control plane

The control plane contains:

```text
Mooncake Master RPC
Mooncake Metadata service
Mooncake metrics endpoint
SGLang HTTP API
```

The Master maintains keys, replicas, allocations, and segment membership. The
Metadata service publishes Transfer Engine segment descriptors and fabric
handles.

### 3.2 Data plane

The KV-cache data plane is:

```text
SGLang Worker GPU buffer
        ↕
Mooncake MNNVL Transport
        ↕
EGM host-memory segment exported through NVLink Fabric
```

Master RPC and Metadata HTTP traffic remain control-plane traffic. KV-cache
payload movement uses the selected Mooncake data protocol.

### 3.3 EGM owner topology

Each physical GPU has one owner process. The owner:

1. selects one physical GPU;
2. allocates host memory on the GPU-nearest NUMA node;
3. exports the allocation through NVIDIA Fabric Memory;
4. publishes the segment metadata;
5. mounts the segment into Mooncake Master;
6. keeps the exported segment alive for clients.

An owner is ready when its log contains:

```text
MNNVL_SEGMENT_READY
exported=1
metadata_published=1
master_mounted=1
OWNER_READY
```

### 3.4 SGLang Direct Linker topology

The Direct Linker connects `UnifiedRadixCache` directly to Mooncake:

```text
GPU KV cache
    ↕
MooncakeDirectLinker
    ↕
MooncakeDistributedStore
```

The path has no SGLang host-cache tier. Direct Linker performs page existence
queries, asynchronous offload, layer-wise loading, and GPU-buffer registration.

## 4. Physical GPU identity

### 4.1 Placement identity

Mooncake MNNVL placement uses:

```text
MC_PHYSICAL_GPU
MOONCAKE_FABRIC_DOMAIN
MC_RACK
MC_TRAY
MOONCAKE_HOST_ID
```

`MC_PHYSICAL_GPU` is the host-local physical GPU ordinal. It is distinct from a
CUDA logical index after visibility remapping.

### 4.2 Worker-local resolution

SGLang worker processes inherit a common container environment and select their
own CUDA logical devices. Each worker performs the following sequence:

```text
torch.cuda.current_device()
        ↓
CUDA_VISIBLE_DEVICES mapping
        ↓
host-local physical GPU ordinal
        ↓
process-local MC_PHYSICAL_GPU
        ↓
Mooncake Store initialization
```

The mapping uses the active CUDA device rather than the TP rank. It applies to
TP2, TP4, TP8, and other one-process-per-GPU deployments.

### 4.3 TP2 example

```text
CUDA_VISIBLE_DEVICES=1,3

Worker cuda:0 → physical GPU 1
Worker cuda:1 → physical GPU 3
```

### 4.4 TP8 reordered-device example

```text
CUDA_VISIBLE_DEVICES=7,2,5,1,6,0,4,3

cuda:0 → physical GPU 7
cuda:1 → physical GPU 2
cuda:2 → physical GPU 5
cuda:3 → physical GPU 1
cuda:4 → physical GPU 6
cuda:5 → physical GPU 0
cuda:6 → physical GPU 4
cuda:7 → physical GPU 3
```

### 4.5 UUID and MIG visibility

For GPU UUID or MIG visibility, configure the host-local physical ordinal in
the worker environment:

```bash
export MC_PHYSICAL_GPU=<PHYSICAL_GPU_ORDINAL>
```

For numeric visibility, an explicit `MC_PHYSICAL_GPU` value must match the
worker's `CUDA_VISIBLE_DEVICES` mapping.

## 5. Mooncake MNNVL/EGM deployment

### 5.1 Verify NVIDIA IMEX

Run on every participating server:

```bash
nvidia-smi -L
nvidia-imex-ctl -N
systemctl status nvidia-imex
```

Required state:

```text
READY
CONNECTED
Fabric State: Completed
Fabric Status: Success
Fabric Health: Healthy
Fabric Bandwidth: Full
```

### 5.2 Build Mooncake

Build Mooncake PR #5161 with CUDA and TENT enabled. The build must produce:

```text
mooncake_master
Mooncake Python package
Mooncake native libraries
scripts/launch_egm_owners.py
```

### 5.3 Start Mooncake Master

```bash
./build-mnnvl/mooncake-store/src/mooncake_master \
  --rpc_address=0.0.0.0 \
  --rpc_port=<MASTER_PORT>
```

Verify:

```bash
ss -lntp | grep <MASTER_PORT>
timeout 2 bash -c '</dev/tcp/<MASTER_HOST>/<MASTER_PORT>' && echo MASTER_OK
```

### 5.4 Start EGM owners on node A

```bash
python3 scripts/launch_egm_owners.py \
  --host <NODE_A_HOST> \
  --rack <RACK> \
  --fabric-domain <FABRIC_DOMAIN> \
  --master <MASTER_HOST>:<MASTER_PORT> \
  --gpus 0,1,2,3 \
  --base-port <NODE_A_BASE_PORT> \
  --segment-gib 4 \
  --log-dir /tmp/mooncake-egm-owners-a
```

### 5.5 Start EGM owners on node B

```bash
python3 scripts/launch_egm_owners.py \
  --host <NODE_B_HOST> \
  --rack <RACK> \
  --fabric-domain <FABRIC_DOMAIN> \
  --master <MASTER_HOST>:<MASTER_PORT> \
  --gpus 0,1,2,3 \
  --base-port <NODE_B_BASE_PORT> \
  --segment-gib 4 \
  --log-dir /tmp/mooncake-egm-owners-b
```

### 5.6 Verify owners

```bash
grep -R 'MNNVL_SEGMENT_READY\|OWNER_READY' \
  /tmp/mooncake-egm-owners-a

grep -R 'MNNVL_SEGMENT_READY\|OWNER_READY' \
  /tmp/mooncake-egm-owners-b
```

Record every complete segment name and Transfer Engine endpoint.

## 6. Directed cross-node Store validation

### 6.1 Configure the client

```bash
export MOONCAKE_STORE_BACKEND=cpp
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export MC_PHYSICAL_GPU=0
export MC_RACK=<RACK>
export MC_TRAY=<CLIENT_HOST>
export MOONCAKE_HOST_ID=<CLIENT_HOST>
export MOONCAKE_FABRIC_DOMAIN=<FABRIC_DOMAIN>
export MC_USE_TENT=1
export MC_ENABLE_MNNVL=1
export MC_MNNVL_EGM=1
```

### 6.2 Create a directed request

```python
from mooncake.store import MooncakeDistributedStore, ReplicateConfig

store = MooncakeDistributedStore()
ret = store.setup(
    "<CLIENT_HOST>:<CLIENT_PORT>",
    "<METADATA_SERVER>",
    0,
    64 * 1024 * 1024,
    "mnnvl",
    "",
    "<MASTER_HOST>:<MASTER_PORT>",
)
assert ret == 0

config = ReplicateConfig()
config.replica_num = 1
config.preferred_segments = ["<REMOTE_SEGMENT_NAME>"]
config.fabric_domain = "<FABRIC_DOMAIN>"

payload = b"x" * (16 * 1024 * 1024)
assert store.put("mnnvl-directed-object", payload, config) == 0
assert store.get("mnnvl-directed-object") == payload
```

### 6.3 Verify remote placement

Check `get_replica_desc()` and confirm:

```text
Replica Status: COMPLETE
Replica Type: MEMORY
Replica Segment: selected remote segment
Replica Endpoint: remote node endpoint
GET length: payload length
SHA256 Match: YES
```

Run the same test in both directions.

## 7. SGLang Direct Linker deployment

### 7.1 Install the Mooncake runtime

Keep the SGLang image's PyTorch and NCCL packages on the default Python path.
Expose the Mooncake package and native libraries separately:

```bash
export PYTHONPATH=<MOONCAKE_PYTHON_PACKAGE_PARENT>:$PYTHONPATH
export LD_LIBRARY_PATH=<MOONCAKE_NATIVE_LIBRARY_DIR>:$LD_LIBRARY_PATH
```

### 7.2 Configure Mooncake

```bash
export MOONCAKE_STORE_BACKEND=cpp
export MOONCAKE_PROTOCOL=mnnvl
export MC_USE_TENT=1
export MC_ENABLE_MNNVL=1
export MC_MNNVL_EGM=1
export MOONCAKE_FABRIC_DOMAIN=<FABRIC_DOMAIN>
export MC_RACK=<RACK>
export MC_TRAY=<HOST_ID>
export MOONCAKE_HOST_ID=<HOST_ID>
export MOONCAKE_MASTER=<MASTER_HOST>:<MASTER_PORT>
export MOONCAKE_TE_META_DATA_SERVER=http://<METADATA_HOST>:<METADATA_PORT>/metadata
```

### 7.3 Launch SGLang

```bash
python3 -m sglang.launch_server \
  --model-path <MODEL_PATH> \
  --served-model-name <MODEL_NAME> \
  --host 0.0.0.0 \
  --port 30000 \
  --tp <TP_SIZE> \
  --page-size 256 \
  --chunked-prefill-size 4096 \
  --max-total-tokens 8192 \
  --max-running-requests 1 \
  --enable-cache-report \
  --enable-metrics \
  --enable-unified-cache-external-linker \
  --unified-cache-external-linker-backend mooncake \
  --unified-cache-external-linker-config \
    '{"enable_group_semantics":true,"protocol":"mnnvl"}'
```

### 7.4 Verify startup

```bash
curl -i http://127.0.0.1:30000/health
```

Expected log entries:

```text
Mooncake direct linker MNNVL rank identity
Mooncake store setup successfully
Mooncake store warmup successfully
Mooncake direct linker storage topology
MnnvlTransport: EGM host memory over NVLink enabled (strict)
```

## 8. KV-cache offload and load-back validation

Size the GPU KV pool to retain only one of two unrelated long prefixes at a
time.

1. Send long prefix A and confirm a cold request.
2. Send unrelated long prefix B and fill the local GPU KV pool.
3. Send prefix A again and confirm external cache recovery.

Validation evidence:

```text
Mooncake direct linker offload: tokens=<TOKENS>
Mooncake direct linker lookup hit: pages=<PAGES>
cached_tokens_details.host = <PAGES> × <PAGE_SIZE>
Mooncake Master Batch Get: success
Mooncake Master ExistKey: success
```

The validated DeepSeek-V4.1-Flash TP4 run produced:

```text
TP0 → physical_gpu=0
TP1 → physical_gpu=1
TP2 → physical_gpu=2
TP3 → physical_gpu=3

Prompt tokens: 6001
External cached tokens: 5632
Lookup pages: 22
Page size: 256 tokens
22 × 256 = 5632 tokens restored
```

The MNNVL EGM path completed:

```text
GPU → MNNVL/NVLink → EGM offload: PASS
EGM → MNNVL/NVLink → GPU load-back: PASS
```

## 9. RDMA transport validation

Use the same Direct Linker with an RDMA Store configuration:

```json
{
  "enable_group_semantics": true,
  "protocol": "rdma",
  "device_name": "mlx5_0,mlx5_1"
}
```

The validated RDMA run produced:

```text
Prompt tokens: 6301
External cached tokens: 6144
Lookup pages: 24
Page size: 256 tokens
24 × 256 = 6144 tokens restored

GPU → RDMA Store offload: PASS
RDMA Store → GPU load-back: PASS
```

The physical GPU identity setup runs only for `mnnvl` and `nvlink_host`. TCP,
RDMA, and classic `nvlink` use their existing initialization paths.

## 10. Stress-test procedure

Use one process per worker. Each operation performs:

```text
put
→ remote replica placement check
→ get
→ byte-for-byte validation
→ remove
```

Reference configuration:

```text
Directions: node A → node B and node B → node A
Workers: 8 per direction, 16 total
Object size: 16 MiB
Duration: 600 seconds
Local buffer: 64 MiB per worker
Remote pool: 1 GiB per node
Placement check interval: 10 operations
```

Validated result:

```text
Operations: 85,869
Operations/s: 142.591
PUT+GET data: 2.621 TiB
Combined PUT throughput: 2.228 GiB/s
Combined GET throughput: 2.228 GiB/s
Remote placement checks: 8,593
Consistency errors: 0
Placement errors: 0
Transfer errors: 0
```

## 11. Monitoring

Monitor the integration with:

```text
DCGM Exporter on every GPU node
Prometheus
Grafana
Mooncake Master metrics
SGLang metrics
```

Key signals:

- `DCGM_FI_DEV_NVLINK_BANDWIDTH_TOTAL`;
- Exporter UP state;
- NVLink error counters;
- Mooncake Storage Usage;
- Mooncake key count;
- Mooncake allocation failures;
- Direct Linker lookup, load, and offload counters.

## 12. Protocol selection

MNNVL EGM:

```json
{
  "enable_group_semantics": true,
  "protocol": "mnnvl"
}
```

RDMA:

```json
{
  "enable_group_semantics": true,
  "protocol": "rdma",
  "device_name": "mlx5_0,mlx5_1"
}
```

TCP:

```json
{
  "enable_group_semantics": true,
  "protocol": "tcp"
}
```

Each SGLang instance selects one transport protocol.

## 13. Test coverage

Hardware-independent tests cover:

- TP2 numeric device mapping;
- TP8 arbitrary device ordering;
- non-contiguous GPUs;
- explicit physical GPU values;
- explicit-versus-derived conflict detection;
- UUID/MIG visibility;
- `mnnvl` and `nvlink_host` selection;
- classic `nvlink` and RDMA isolation;
- per-worker environment configuration.

Result:

```text
15 tests passed
```

Repository checks:

```text
isort: PASS
ruff: PASS
ruff-format: PASS
codespell: PASS
Python AST: PASS
test registry: PASS
indirect-import: PASS
py_compile: PASS
```

## 14. Operations

```bash
docker ps --filter name=mooncake
docker logs -f <SGLANG_CONTAINER>
docker logs -f <MASTER_CONTAINER>
docker logs -f <OWNER_CONTAINER>
curl -i http://127.0.0.1:30000/health
```

Stop the deployment in this order:

1. stop new inference requests;
2. drain in-flight Direct Linker transfers;
3. stop SGLang;
4. stop EGM owners;
5. stop Mooncake Master and Metadata services.

## 15. Acceptance checklist

| Acceptance item | Result |
|---|---|
| IMEX nodes READY and CONNECTED | PASS |
| NVLink Fabric Memory transfer | PASS |
| Strict MNNVL EGM owners | PASS |
| Dual-node EGM memory pool | PASS |
| Bidirectional directed put/get | PASS |
| SHA256 and byte-level consistency | PASS |
| 10-minute bidirectional stress test | PASS |
| 2.621 TiB total transferred data | PASS |
| 8,593 remote placement checks | PASS |
| Zero transfer, placement, and consistency errors | PASS |
| SGLang Direct Linker MNNVL EGM | PASS |
| SGLang Direct Linker RDMA | PASS |
| TP4 hardware mapping | PASS |
| TP2 and TP8 generalized mapping tests | PASS |
| Repository formatting and static checks | PASS |

## 16. Community contribution workflow

Synchronize the branch:

```bash
git fetch origin main
git rebase origin/main
```

Run repository checks:

```bash
SKIP=no-commit-to-branch \
pre-commit run --all-files --show-diff-on-failure
```

Push to the shared fork:

```bash
git push --force-with-lease bruce \
  HEAD:gaochiyifeng/mooncake-mnnvl-physical-gpu
```

Pull request layout:

```text
Base repository: sgl-project/sglang
Base branch: main
Head repository: voidxb/sglang-bugfix
Compare branch: gaochiyifeng/mooncake-mnnvl-physical-gpu
```

CI commands:

```text
/tag-and-rerun-ci
/rerun-failed-ci
/run-ci nv
/run-ci amd
```

Merge requirements:

```text
Required checks PASS
At least one CODEOWNER approval
Merge Oncall completes the merge process
```

## 17. References

- [Mooncake PR #5161](https://github.com/kvcache-ai/Mooncake/pull/5161)
- [Mooncake PR #5161 commits](https://github.com/kvcache-ai/Mooncake/pull/5161/commits)
- [SGLang PR #48088](https://github.com/sgl-project/sglang/pull/48088)
- [Mooncake backend README](./README.md)

## 18. Conclusion

Mooncake MNNVL/EGM provides a validated path across IMEX, NVLink Fabric Memory,
the Transfer Engine, the EGM memory pool, and bidirectional cross-node Store
operations. SGLang Direct Linker resolves a physical GPU identity for every
worker and completes KV-cache offload, lookup, and load-back through both MNNVL
EGM and RDMA transports.

This guide defines the complete deployment, validation, monitoring,
stress-testing, and community contribution workflow for the integration.
