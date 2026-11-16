# DeepEP × NCCL 2.31.2 derisk on non-EFA (InfiniBand) — reproduction

Test collateral for sgl-project/sglang#48678 (EFA in main image, NCCL 2.30.7 → 2.31.2)
and sgl-project/DeepEP#12 (unordered hybrid kernels).

Validates three claims, on **non-EFA** hardware (the regression target for both PRs):

1. **R1**: the currently published `sgl-deep-ep 0.1.2.post1` wheel hard-crashes under
   runtime NCCL 2.31.2 (compile-time/runtime version assert), i.e. the NCCL bump and
   the DeepEP wheel upgrade are atomic.
2. **R2/R3**: DeepEP#12 ordered mode (the default) passes functionally on IB but
   regresses dispatch scale-out bandwidth −11..−22% vs base; unordered mode passes
   and is faster than ordered in every config.
3. **R7**: NCCL 2.31.2 vs 2.30.7 collectives perf on IB is within noise (<0.5%).

## Hardware used

- DeepEP matrix: 2 nodes × 4 B300 (sm_103), 13× mlx5 IB, 800Gb/s XDR. Any 2-node,
  4-GPU-per-node IB or RoCE cluster with mlx5 NICs works.
- NCCL microbench: 2 nodes × 4 H200, 8× mlx5 IB, 400Gb/s NDR.
- Image checks (scripts included but not required for the three claims above):
  1× H200 node.
- Image everywhere: `lmsysorg/sglang:dev` (CUDA 13.0.88 toolchain, pip
  `nvidia-nccl-cu13 2.30.7`, `sgl-deep-ep 0.1.2.post1`).

## Prerequisites

- Two nodes that can reach each other over RDMA; set `MASTER_ADDR` to the rank-0
  node and open TCP port 8361 between them.
- On both nodes: `git clone` this sglang checkout (or just copy this directory).

## 1. Setup (both nodes)

```bash
export D=/workspace/derisk; mkdir -p $D; cd $D
bash cluster_setup.sh <rank>          # 0 on first node, 1 on second
```

Captures env (GPU, NCCL runtime version, RDMA devices) and clones DeepEP at two refs:
`deepep-base` (sgl-deepep branch, tests matching the installed 0.1.2.post1 wheel) and
`deepep-pr12` (DeepEP#12 head).

## 2. Build the DeepEP#12 wheel against NCCL 2.31.2 headers (both nodes)

⚠️ The wheel **must** be compiled against NCCL ≥2.31 headers or it fails the same
`nccl.cu` compile-time==runtime assert it is supposed to fix. Building against 2.30.7
headers was verified to fail identically. `EP_NCCL_ROOT_DIR` is the knob
(CPATH/C_INCLUDE_PATH alone are ignored by setup.py).

```bash
bash build_wheel.sh                   # deepep-pr12  -> /workspace/derisk/wheels-pr12/
bash build_base_wheel.sh              # deepep-base  -> /workspace/derisk/wheels-base2312/  (NCCL isolation control)
```

Both side-stage NCCL 2.31.2 into `/workspace/derisk/nccl2312` without touching the
runtime environment. Look for `> NCCL path: /workspace/derisk/nccl2312/...` in the
setup.py output and `WHEEL_BUILD_OK` at the end (~3 min each).

## 3. Test matrix (both nodes; start rank 1, then rank 0 immediately)

NCCL runtime swap used below:

```bash
pip install "nvidia-nccl-cu13==2.31.2" --force-reinstall --no-deps -q   # or ==2.30.7
python3 -c "import torch,ctypes; v=ctypes.c_int(); \
  ctypes.CDLL('libnccl.so.2').ncclGetVersion(ctypes.byref(v)); print(v.value)"
# expect 23102 (2.31.2) or 23007 (2.30.7); the torch import is required first
```

| label | wheel under test | runtime NCCL | purpose |
|---|---|---|---|
| A2 | `sgl-deep-ep 0.1.2.post1` (preinstalled) | 2.30.7 | today's production baseline |
| B  | `sgl-deep-ep 0.1.2.post1` | 2.31.2 | **R1**: expect hard assert, 0/96 configs |
| B' | base wheel rebuilt vs 2.31.2 (`wheels-base2312`) | 2.31.2 | isolates NCCL-bump effect on base perf |
| C  | DeepEP#12 wheel (`wheels-pr12`), `EP_HYBRID_KERNEL=ordered` | 2.31.2 | **R2** |
| D  | DeepEP#12 wheel, `EP_HYBRID_KERNEL=unordered` | 2.31.2 | **R3** |

```bash
export MASTER_ADDR=<rank0-address>
# A2 example (preinstalled wheel, no env overrides):
bash run_ep_test.sh <rank> A2 /workspace/derisk/deepep-base
# B:
pip install "nvidia-nccl-cu13==2.31.2" --force-reinstall --no-deps -q
bash run_ep_test.sh <rank> B /workspace/derisk/deepep-base
# B':
pip uninstall -y sgl-deep-ep && pip install /workspace/derisk/wheels-base2312/*.whl
bash run_ep_test.sh <rank> Bp /workspace/derisk/deepep-base
# C:
pip uninstall -y sgl-deep-ep && pip install /workspace/derisk/wheels-pr12/*.whl
bash run_ep_test.sh <rank> C /workspace/derisk/deepep-pr12 EP_HYBRID_KERNEL=ordered
# D:
bash run_ep_test.sh <rank> D /workspace/derisk/deepep-pr12 EP_HYBRID_KERNEL=unordered
```

Each run is ~96 configs, ~10 min. Check `EXIT=0` in `/workspace/derisk/logs/<label>_rank0.log`.
Run B is expected to fail fast with:

```
RuntimeError: Assertion exception (csrc/kernels/backend/nccl.cu:104):
NCCL_VERSION_CODE == nccl_runtime_version and
"Prior to NCCL 2.31, NCCL compile-time and runtime versions must be the same. ..."
```

## 4. NCCL microbench (R7, both nodes)

```bash
pip install "nvidia-nccl-cu13==2.30.7" --force-reinstall --no-deps -q
MASTER_ADDR=<rank0> WORLD_SIZE=2 RANK=<rank> python3 bench_allreduce.py | tee nccl2307.log
pip install "nvidia-nccl-cu13==2.31.2" --force-reinstall --no-deps -q
MASTER_ADDR=<rank0> WORLD_SIZE=2 RANK=<rank> python3 bench_allreduce.py | tee nccl2312.log
```

## 5. CUDA 13.0 JIT gotcha (base wheel only)

Base DeepEP cannot JIT on the CUDA 13.0.88 toolchain (`ptxas error: Arguments
mismatch for instruction 'mov'`, from the `st.bulk` PTX emitted by
`internode_ll.cu`). DeepEP#12 ships the fix; for the A2/B/B' baseline runs, apply the
equivalent one-line patch to the installed sources:

```diff
# csrc/kernels/device/ptx.cuh, in st_bulk():
     asm volatile("st.bulk.weak.shared::cta [%0], %1, 0;\n" ::
                  "r"(static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr))),
-                 "r"(kNumBytes)
+                 "l"(static_cast<uint64_t>(kNumBytes))
                  : "memory");
```

## 6. Image checks (optional; R4/R5/R6 of the sglang PR)

`build_image.sh` (needs an amd64 docker host): builds sglang#48678's main Dockerfile
with the same args as the publish workflow, saves a tarball. `check_image.sh` (run
inside the image on an IB/RoCE node): verifies `/opt/amazon/efa` libfabric shadowing,
`fi_info -p verbs`, NCCL net selection (expect internal `NET/IB`, ofi plugin unused),
Mooncake `TransferEngine` RDMA init, NIXL LIBFABRIC backend availability.

## Results we measured (reference)

Dispatch scale-out GB/s (stable ±1 within a run; ordered reproduced twice):

| config | A2 base+2.30.7 | B' base+2.31.2 | C ordered | D unordered |
|---|---|---|---|---|
| fp8 dispatch  | 71  | 81  | 63  | 120 |
| bf16 dispatch | 129 | 144 | 114 | 140 |

- R1: confirmed broken (hard assert).
- R2: functional pass, dispatch perf regression vs B' (63 vs 81; 114 vs 144).
- R3: unordered faster than ordered everywhere; also faster than today's stack.
- R7: all_reduce/all_to_all 2.30.7 vs 2.31.2 within <0.5%.
