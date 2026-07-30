# Kimi-K3 Standalone Kernels Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port every standalone Kimi-K3 kernel from the Day-0 branch to current SGLang `main`, with direct correctness tests and without model or serving integration.

**Architecture:** Copy the final kernel implementations from the locked `origin/kimi-k3` source revision into a branch based on current `origin/main`. Keep tensor-level kernel APIs isolated under `python/sglang/kernels`, add only the minimal generic JIT/environment/communicator plumbing needed to compile them, and validate each family through direct single- or multi-GPU tests.

**Tech Stack:** Python, PyTorch, Triton, CUDA C++ JIT, CuTeDSL/CUTLASS DSL, PyTorch distributed, SGLang CI registration.

---

## Locked Revisions and File Boundaries

- Base: `origin/main` at `f4e0ac382e4e5d644f2fbe4a15c20da53500bbca`
- Source: `origin/kimi-k3` at `578edb240a6d6f6f2fa4c31497276955d7f73432`
- Worktree: `/Users/bbuf/工作目录/Common/sglang-kimi-k3-standalone-kernels`
- Branch: `bbuf/kimi-k3-standalone-kernels`

Allowed implementation areas:

- `python/sglang/kernels/jit/`
- `python/sglang/kernels/ops/`
- selected generic entries in `python/sglang/srt/environ.py`
- selected generic communicator support in
  `python/sglang/srt/distributed/device_communicators/custom_all_reduce_v2.py`
- tests under `test/registered/jit/` and `test/registered/kernels/`

Explicitly excluded:

- `python/sglang/srt/models/`
- `python/sglang/srt/managers/`
- `python/sglang/srt/speculative/`
- `python/sglang/srt/multimodal/`
- Kimi parser, server-argument, engine, scheduler, and disaggregation wiring
- all development-only benchmark files introduced by the Day-0 branch

### Task 1: Record the Kernel Inventory

**Files:**
- Create: `docs/superpowers/plans/kimi-k3-kernel-inventory.md`

- [ ] **Step 1: Generate the implementation inventory**

Run:

```bash
git diff --name-status \
  f4e0ac382e4e5d644f2fbe4a15c20da53500bbca...578edb240a6d6f6f2fa4c31497276955d7f73432 \
  -- python/sglang/kernels
```

Expected: only files below `python/sglang/kernels`; no model or scheduler files.

- [ ] **Step 2: Generate the existing-test inventory**

Run:

```bash
git diff --name-status \
  f4e0ac382e4e5d644f2fbe4a15c20da53500bbca...578edb240a6d6f6f2fa4c31497276955d7f73432 \
  -- test/registered/jit test/registered/kernels
```

Expected: correctness tests plus no surviving added benchmark files at source
revision `578edb240a`.

- [ ] **Step 3: Write the inventory**

The inventory must group every implementation and test into:

```text
generic-jit
generic-attention-mla-vlm
generic-moe-gemm-elementwise
kda-decode-prefill-state
kimi-k3-fused-compute
kimi-k3-distributed
excluded-runtime-integration
```

For every public Python wrapper, record its direct test file or the exact new
test added in Tasks 4–7.

- [ ] **Step 4: Commit the inventory**

```bash
git add docs/superpowers/plans/kimi-k3-kernel-inventory.md
git commit -m "docs: inventory Kimi-K3 standalone kernels"
```

Expected: one documentation-only commit.

### Task 2: Port Shared JIT Infrastructure

**Files:**
- Modify: `python/sglang/kernels/jit/utils/compile.py`
- Modify: `python/sglang/kernels/jit/include/sgl_kernel/distributed/communicator.cuh`
- Modify: `python/sglang/kernels/jit/include/sgl_kernel/math.cuh`
- Modify: `python/sglang/kernels/jit/include/sgl_kernel/warp.cuh`
- Modify: `python/sglang/kernels/jit/csrc/gemm/per_token_group_quant.cuh`
- Modify: `python/sglang/kernels/ops/__init__.py`

- [ ] **Step 1: Apply the final source versions**

```bash
git restore --source=578edb240a6d6f6f2fa4c31497276955d7f73432 -- \
  python/sglang/kernels/jit/utils/compile.py \
  python/sglang/kernels/jit/include/sgl_kernel/distributed/communicator.cuh \
  python/sglang/kernels/jit/include/sgl_kernel/math.cuh \
  python/sglang/kernels/jit/include/sgl_kernel/warp.cuh \
  python/sglang/kernels/jit/csrc/gemm/per_token_group_quant.cuh \
  python/sglang/kernels/ops/__init__.py
```

- [ ] **Step 2: Audit the diff**

```bash
git diff --check
git diff -- python/sglang/kernels/jit python/sglang/kernels/ops/__init__.py
```

Expected: JIT/compiler/header support only; no Kimi model imports.

- [ ] **Step 3: Compile Python files**

```bash
python3 -m compileall -q \
  python/sglang/kernels/jit/utils/compile.py \
  python/sglang/kernels/ops/__init__.py
```

Expected: exit code 0.

- [ ] **Step 4: Commit**

```bash
git add python/sglang/kernels/jit python/sglang/kernels/ops/__init__.py
git commit -m "feat(kernels): add shared JIT support for Kimi-K3 kernels"
```

### Task 3: Port Generic Attention, Elementwise, GEMM, MoE, and Sampling Kernels

**Files:**
- Create/modify: `python/sglang/kernels/jit/csrc/attention/`
- Create/modify: `python/sglang/kernels/jit/csrc/elementwise/`
- Create/modify: `python/sglang/kernels/jit/csrc/gemm/`
- Create/modify: `python/sglang/kernels/jit/csrc/moe/`
- Create/modify: `python/sglang/kernels/ops/attention/`
- Create: `python/sglang/kernels/ops/elementwise/add3.py`
- Create: `python/sglang/kernels/ops/gemm/tiny_gemm.py`
- Create/modify: `python/sglang/kernels/ops/moe/`
- Create: `python/sglang/kernels/ops/sampling/top_p_renorm_triton.py`
- Create: `python/sglang/kernels/ops/mm/process/image.py`

- [ ] **Step 1: Port only changed generic kernel files**

Run:

```bash
git diff --name-only \
  f4e0ac382e4e5d644f2fbe4a15c20da53500bbca...578edb240a6d6f6f2fa4c31497276955d7f73432 \
  -- \
  python/sglang/kernels/jit/csrc/attention \
  python/sglang/kernels/jit/csrc/elementwise \
  python/sglang/kernels/jit/csrc/gemm \
  python/sglang/kernels/jit/csrc/moe \
  python/sglang/kernels/ops/attention \
  python/sglang/kernels/ops/elementwise \
  python/sglang/kernels/ops/gemm \
  python/sglang/kernels/ops/moe \
  python/sglang/kernels/ops/sampling \
  python/sglang/kernels/ops/mm |
while IFS= read -r kernel_file; do
  git restore --source=578edb240a6d6f6f2fa4c31497276955d7f73432 -- "$kernel_file"
done
```

Expected: generic kernel changes only. The KDA files copied by this command are
left uncommitted for Task 5.

- [ ] **Step 2: Stage only non-KDA files**

```bash
git add \
  python/sglang/kernels/jit/csrc/elementwise \
  python/sglang/kernels/jit/csrc/gemm \
  python/sglang/kernels/jit/csrc/moe \
  python/sglang/kernels/ops/elementwise \
  python/sglang/kernels/ops/gemm \
  python/sglang/kernels/ops/moe \
  python/sglang/kernels/ops/sampling \
  python/sglang/kernels/ops/mm \
  python/sglang/kernels/ops/attention/concat_mla.py \
  python/sglang/kernels/ops/attention/set_mla_kv_concat_q.py \
  python/sglang/kernels/ops/attention/vision_rope.py \
  python/sglang/kernels/jit/csrc/elementwise/concat_mla.cuh \
  python/sglang/kernels/jit/csrc/elementwise/set_mla_kv_buffer.cuh \
  python/sglang/kernels/jit/csrc/elementwise/set_mla_kv_concat_q.cuh \
  python/sglang/kernels/jit/csrc/elementwise/set_mla_kv_concat_q_fp8.cuh
```

- [ ] **Step 3: Verify staged scope**

```bash
git diff --cached --name-only | rg 'kda|kimi_k3|srt/models|server_args'
```

Expected: no output.

- [ ] **Step 4: Commit**

```bash
git commit -m "feat(kernels): port generic Kimi-K3 prerequisite kernels"
```

### Task 4: Port and Complete Generic Kernel Tests

**Files:**
- Create: `test/registered/jit/test_set_mla_kv_concat_q.py`
- Create: `test/registered/jit/test_set_mla_kv_concat_q_fp8.py`
- Create: `test/registered/kernels/ops/test_add3.py`
- Create: `test/registered/kernels/ops/test_moe_route_quant_fused.py`
- Create: `test/registered/kernels/ops/test_moe_route_radix.py`
- Create: `test/registered/kernels/ops/test_tiny_gemm.py`
- Create/modify: `test/registered/kernels/ops/test_vision_rope.py`
- Create: `test/registered/kernels/ops/test_moe_auxiliary.py`
- Create: `test/registered/kernels/ops/test_mla_output_gate.py`
- Create: `test/registered/kernels/ops/test_top_p_renorm_triton.py`

- [ ] **Step 1: Port existing direct correctness tests**

```bash
for test_file in \
  test/registered/jit/test_set_mla_kv_concat_q.py \
  test/registered/jit/test_set_mla_kv_concat_q_fp8.py \
  test/registered/kernels/ops/test_add3.py \
  test/registered/kernels/ops/test_moe_route_quant_fused.py \
  test/registered/kernels/ops/test_moe_route_radix.py \
  test/registered/kernels/ops/test_tiny_gemm.py \
  test/registered/kernels/ops/test_vision_rope.py; do
  git restore --source=578edb240a6d6f6f2fa4c31497276955d7f73432 -- "$test_file"
done
```

- [ ] **Step 2: Add auxiliary MoE reference tests**

`test_moe_auxiliary.py` must directly check:

```python
sorted_ids, expert_ids, num_post = moe_align_single_token(topk_ids, block_size)
assert torch.equal(expert_ids.cpu(), torch.sort(topk_ids[0].int()).values.cpu())
assert num_post.item() == topk_ids.numel() * block_size

actual = moe_topk_sum(x, torch.empty_like(x[:, 0]))
expected = x.float().sum(dim=1).to(torch.bfloat16)
torch.testing.assert_close(actual, expected, rtol=0, atol=0)
```

Use distinct expert ids, block sizes 16 and 64, top-k values 8 and 16, and
token dimensions 128 and 7168.

- [ ] **Step 3: Add MLA output-gate bitwise test**

`test_mla_output_gate.py` must use:

```python
expected = x * torch.sigmoid(gate).to(torch.bfloat16)
actual = kimi_k3_mla_output_gate(x, gate)
torch.testing.assert_close(actual, expected, rtol=0, atol=0)
```

Cover shapes `(1, 7168)`, `(8, 7168)`, and a small shape whose element count is
divisible by eight. Assert `covered()` is false for FP16, mismatched shapes,
non-contiguous inputs, and zero elements.

- [ ] **Step 4: Add top-p reference test**

The reference keeps every probability greater than or equal to the ascending
CDF pivot and renormalizes:

```python
sorted_probs = probs.float().sort(dim=-1).values
cdf = sorted_probs.cumsum(dim=-1)
cutoff = torch.searchsorted(cdf, (1.0 - top_p).unsqueeze(1)).squeeze(1)
pivot = sorted_probs.gather(1, cutoff[:, None])
expected = torch.where(probs.float() >= pivot, probs.float(), 0)
expected /= expected.sum(dim=-1, keepdim=True)
```

Cover scalar and per-row `top_p`, tied probabilities, vocabulary sizes 7,
1024, and 157184, plus all documented validation failures.

- [ ] **Step 5: Run collection and direct tests**

```bash
python3 -m compileall -q test/registered/jit test/registered/kernels/ops
python3 test/registered/kernels/ops/test_moe_auxiliary.py
python3 test/registered/kernels/ops/test_mla_output_gate.py
python3 test/registered/kernels/ops/test_top_p_renorm_triton.py
```

Expected: files collect; GPU tests pass on supported NVIDIA hardware or skip
with an explicit architecture reason.

- [ ] **Step 6: Commit**

```bash
git add test/registered/jit test/registered/kernels/ops
git commit -m "test(kernels): cover generic Kimi-K3 prerequisite kernels"
```

### Task 5: Port the KDA Kernel Family and Tests

**Files:**
- Create/modify: `python/sglang/kernels/jit/csrc/attention/kda_fused_decode.cuh`
- Create: `python/sglang/kernels/jit/csrc/attention/kda_packed_decode.cuh`
- Create: `python/sglang/kernels/jit/csrc/attention/kda_prefill.cu`
- Create/modify: `python/sglang/kernels/ops/attention/kda_fused_decode.py`
- Create: `python/sglang/kernels/ops/attention/kda_packed_decode.py`
- Create: `python/sglang/kernels/ops/attention/fla/kda_replayssm_spec_decode.py`
- Create: `python/sglang/kernels/ops/attention/linear/kda_nvidia_prefill/`
- Create: `python/sglang/kernels/ops/attention/linear/kda_ptx_prefill/`
- Modify: `python/sglang/kernels/ops/attention/cutedsl_kda.py`
- Modify: `python/sglang/kernels/ops/attention/fla/kda.py`
- Modify: `python/sglang/kernels/ops/mamba/mamba_state_scatter_triton.py`
- Create: `test/registered/kernels/test_kda_mtp_cutedsl_replayssm_ring.py`
- Create: `test/registered/kernels/test_kda_replayssm_fold.py`
- Create: `test/registered/kernels/test_kda_replayssm_fold_batched.py`
- Create: `test/registered/kernels/test_kda_replayssm_ring_fused.py`
- Create: `test/registered/kernels/test_kda_replayssm_ring_ragged.py`
- Create: `test/registered/unit/layers/attention/linear/kernels/test_kda_nvidia.py`

- [ ] **Step 1: Stage the final KDA implementation already copied in Task 3**

```bash
git add \
  python/sglang/kernels/jit/csrc/attention/kda_fused_decode.cuh \
  python/sglang/kernels/jit/csrc/attention/kda_packed_decode.cuh \
  python/sglang/kernels/jit/csrc/attention/kda_prefill.cu \
  python/sglang/kernels/ops/attention/kda_fused_decode.py \
  python/sglang/kernels/ops/attention/kda_packed_decode.py \
  python/sglang/kernels/ops/attention/fla \
  python/sglang/kernels/ops/attention/linear/kda_nvidia_prefill \
  python/sglang/kernels/ops/attention/linear/kda_ptx_prefill \
  python/sglang/kernels/ops/attention/cutedsl_kda.py \
  python/sglang/kernels/ops/mamba/mamba_state_scatter_triton.py
```

- [ ] **Step 2: Port KDA tests**

```bash
for test_file in \
  test/registered/kernels/test_kda_mtp_cutedsl_replayssm_ring.py \
  test/registered/kernels/test_kda_replayssm_fold.py \
  test/registered/kernels/test_kda_replayssm_fold_batched.py \
  test/registered/kernels/test_kda_replayssm_ring_fused.py \
  test/registered/kernels/test_kda_replayssm_ring_ragged.py \
  test/registered/unit/layers/attention/linear/kernels/test_kda_nvidia.py; do
  git restore --source=578edb240a6d6f6f2fa4c31497276955d7f73432 -- "$test_file"
done
```

- [ ] **Step 3: Verify direct API boundaries**

```bash
rg -n 'ServerArgs|Scheduler|KimiK3ForCausalLM|launch_server' \
  python/sglang/kernels/ops/attention \
  test/registered/kernels/test_kda_* \
  test/registered/unit/layers/attention/linear/kernels/test_kda_nvidia.py
```

Expected: no output.

- [ ] **Step 4: Run KDA tests**

```bash
python3 test/registered/kernels/test_kda_replayssm_fold.py
python3 test/registered/kernels/test_kda_replayssm_fold_batched.py
python3 test/registered/kernels/test_kda_replayssm_ring_fused.py
python3 test/registered/kernels/test_kda_replayssm_ring_ragged.py
python3 test/registered/unit/layers/attention/linear/kernels/test_kda_nvidia.py
```

Expected: exact or declared-tolerance agreement with the reference path.

- [ ] **Step 5: Commit**

```bash
git add python/sglang/kernels test/registered/kernels \
  test/registered/unit/layers/attention/linear/kernels
git commit -m "feat(kernels): port standalone KDA kernels"
```

### Task 6: Port Kimi-K3 Fused Compute Kernels and Tests

**Files:**
- Create: `python/sglang/kernels/jit/csrc/kimi_k3/attn_res/fused_tma.cuh`
- Create: `python/sglang/kernels/jit/csrc/kimi_k3/mla_output_gate.cuh`
- Create: `python/sglang/kernels/jit/csrc/kimi_k3/situ_and_mul.cuh`
- Create: `python/sglang/kernels/jit/csrc/kimi_k3/situ_and_mul_masked_post_quant.cuh`
- Create: `python/sglang/kernels/ops/kimi_k3/__init__.py`
- Create: `python/sglang/kernels/ops/kimi_k3/activation.py`
- Create: `python/sglang/kernels/ops/kimi_k3/attn_res.py`
- Create: `python/sglang/kernels/ops/kimi_k3/kda_decode_mtp.py`
- Create: `python/sglang/kernels/ops/kimi_k3/mla_output_gate.py`
- Create: `python/sglang/kernels/ops/kimi_k3/moe.py`
- Create: `test/registered/kernels/ops/kimi_k3/test_attn_res_fused_tma.py`
- Create: `test/registered/kernels/ops/kimi_k3/test_situ_mul_quant.py`
- Create: `test/registered/kernels/test_attn_res_aggregate_stream.py`

- [ ] **Step 1: Port the compute-only K3 files**

```bash
for kernel_file in \
  python/sglang/kernels/jit/csrc/kimi_k3/attn_res/fused_tma.cuh \
  python/sglang/kernels/jit/csrc/kimi_k3/mla_output_gate.cuh \
  python/sglang/kernels/jit/csrc/kimi_k3/situ_and_mul.cuh \
  python/sglang/kernels/jit/csrc/kimi_k3/situ_and_mul_masked_post_quant.cuh \
  python/sglang/kernels/ops/kimi_k3/__init__.py \
  python/sglang/kernels/ops/kimi_k3/activation.py \
  python/sglang/kernels/ops/kimi_k3/attn_res.py \
  python/sglang/kernels/ops/kimi_k3/kda_decode_mtp.py \
  python/sglang/kernels/ops/kimi_k3/mla_output_gate.py \
  python/sglang/kernels/ops/kimi_k3/moe.py; do
  git restore --source=578edb240a6d6f6f2fa4c31497276955d7f73432 -- "$kernel_file"
done
```

- [ ] **Step 2: Port direct tests**

```bash
for test_file in \
  test/registered/kernels/ops/kimi_k3/test_attn_res_fused_tma.py \
  test/registered/kernels/ops/kimi_k3/test_situ_mul_quant.py \
  test/registered/kernels/test_attn_res_aggregate_stream.py; do
  git restore --source=578edb240a6d6f6f2fa4c31497276955d7f73432 -- "$test_file"
done
```

- [ ] **Step 3: Add the MLA output-gate test from Task 4 to this commit**

Run:

```bash
python3 test/registered/kernels/ops/test_mla_output_gate.py
```

Expected: bit-exact agreement with the documented double-rounding reference.

- [ ] **Step 4: Run fused-compute tests**

```bash
python3 test/registered/kernels/ops/kimi_k3/test_attn_res_fused_tma.py
python3 test/registered/kernels/ops/kimi_k3/test_situ_mul_quant.py
python3 test/registered/kernels/test_attn_res_aggregate_stream.py
```

Expected: pass on supported SM100 hardware.

- [ ] **Step 5: Commit**

```bash
git add python/sglang/kernels/jit/csrc/kimi_k3 \
  python/sglang/kernels/ops/kimi_k3 \
  test/registered/kernels
git commit -m "feat(kernels): port Kimi-K3 fused compute kernels"
```

### Task 7: Port Kimi-K3 Distributed Kernels with Direct Tests

**Files:**
- Create: `python/sglang/kernels/jit/csrc/kimi_k3/comm/ar_fusion.cuh`
- Create: `python/sglang/kernels/jit/csrc/kimi_k3/comm/gemm_ag.cuh`
- Create: `python/sglang/kernels/jit/csrc/kimi_k3/comm/gemm_ar.cuh`
- Create: `python/sglang/kernels/jit/csrc/kimi_k3/comm/sp_collective.cuh`
- Create: `python/sglang/kernels/ops/kimi_k3/all_reduce.py`
- Create: `python/sglang/kernels/ops/kimi_k3/gemm_ag.py`
- Create: `python/sglang/kernels/ops/kimi_k3/gemm_ar.py`
- Create: `python/sglang/kernels/ops/kimi_k3/sp_collective.py`
- Create: `python/sglang/kernels/ops/kimi_k3/configs/sp_collective/world=4,H=7168,device_name=NVIDIA_GB300.json`
- Create: `python/sglang/kernels/ops/kimi_k3/configs/sp_collective/world=8,H=7168,device_name=NVIDIA_GB300.json`
- Modify: `python/sglang/srt/environ.py`
- Modify: `python/sglang/srt/distributed/device_communicators/custom_all_reduce_v2.py`
- Create: `test/registered/kernels/ops/kimi_k3/test_ar_fusion.py`
- Create: `test/registered/kernels/ops/kimi_k3/test_gemm_ag.py`
- Create: `test/registered/kernels/ops/kimi_k3/test_sp_collective.py`
- Create: `test/registered/kernels/ops/kimi_k3/test_symm_buffers_direct.py`

- [ ] **Step 1: Port distributed implementation and tuning data**

```bash
for kernel_file in \
  python/sglang/kernels/jit/csrc/kimi_k3/comm/ar_fusion.cuh \
  python/sglang/kernels/jit/csrc/kimi_k3/comm/gemm_ag.cuh \
  python/sglang/kernels/jit/csrc/kimi_k3/comm/gemm_ar.cuh \
  python/sglang/kernels/jit/csrc/kimi_k3/comm/sp_collective.cuh \
  python/sglang/kernels/ops/kimi_k3/all_reduce.py \
  python/sglang/kernels/ops/kimi_k3/gemm_ag.py \
  python/sglang/kernels/ops/kimi_k3/gemm_ar.py \
  python/sglang/kernels/ops/kimi_k3/sp_collective.py \
  'python/sglang/kernels/ops/kimi_k3/configs/sp_collective/world=4,H=7168,device_name=NVIDIA_GB300.json' \
  'python/sglang/kernels/ops/kimi_k3/configs/sp_collective/world=8,H=7168,device_name=NVIDIA_GB300.json'; do
  git restore --source=578edb240a6d6f6f2fa4c31497276955d7f73432 -- "$kernel_file"
done
```

- [ ] **Step 2: Port only generic environment and communicator support**

From `python/sglang/srt/environ.py`, add only:

```python
SGLANG_TRTLLM_GEN_MOE_CUBIN_POOL = EnvStr(None)
SGLANG_FORCE_CUSTOM_ALL_REDUCE_V2_PULL_SIZE_KB = EnvInt(None)
SGLANG_FORCE_CUSTOM_ALL_REDUCE_V2_PUSH_SIZE_KB = EnvInt(None)
SGLANG_ENABLE_CUSTOM_ALL_REDUCE_V2_MULTINODE = EnvBool(False)
```

From the Kimi source version of `custom_all_reduce_v2.py`, port:

- forced pull/push workspace sizing;
- `mc_base_ptr` and `pull_sem_mc_ptr`;
- multi-node symmetric-memory capability gating;
- disabling graph zero-copy registration on multi-node groups.

Do not port K3 model feature flags or `GroupCoordinator` dispatch changes.

- [ ] **Step 3: Port reusable distributed correctness tests**

```bash
for test_file in \
  test/registered/kernels/ops/kimi_k3/test_ar_fusion.py \
  test/registered/kernels/ops/kimi_k3/test_gemm_ag.py; do
  git restore --source=578edb240a6d6f6f2fa4c31497276955d7f73432 -- "$test_file"
done
```

- [ ] **Step 4: Replace runtime-coupled symmetric-buffer testing**

Do not port `test_symm_buffers.py`. Create `test_symm_buffers_direct.py` with a
multi-process fixture that:

```python
dist.init_process_group("nccl")
comm = CustomAllReduceV2(group=dist.group.WORLD, device=torch.device("cuda"))
assert comm.mc_base_ptr >= 0
assert comm.pull_sem_mc_ptr >= 0
dist.barrier()
```

It must register a named persistent buffer twice, assert pointer stability,
run one eager and one CUDA-graph replay, and destroy the process group in
`finally`.

- [ ] **Step 5: Extract SP-collective correctness from the deleted benchmark**

Use the reference source:

```bash
git show 578edb240a^:benchmark/kernels/kimi_k3/bench_sp_collective.py
```

Create `test_sp_collective.py` that checks `reduce_scatter_res`,
`reduce_scatter_pull`, `all_gather`, and `all_gather_direct` against
`torch.distributed` for world sizes 4 and 8, hidden size 7168, BF16 inputs,
and token counts 1, 8, and 64. Do not retain timing, warmup loops, CLI parsing,
CSV, or throughput reporting.

- [ ] **Step 6: Run multi-GPU tests**

```bash
torchrun --standalone --nproc-per-node=4 \
  test/registered/kernels/ops/kimi_k3/test_ar_fusion.py
torchrun --standalone --nproc-per-node=4 \
  test/registered/kernels/ops/kimi_k3/test_gemm_ag.py
torchrun --standalone --nproc-per-node=4 \
  test/registered/kernels/ops/kimi_k3/test_sp_collective.py
torchrun --standalone --nproc-per-node=4 \
  test/registered/kernels/ops/kimi_k3/test_symm_buffers_direct.py
```

Expected: direct agreement with unfused PyTorch distributed references and no
leaked process groups or symmetric-memory registrations.

- [ ] **Step 7: Commit**

```bash
git add python/sglang/kernels/jit/csrc/kimi_k3/comm \
  python/sglang/kernels/ops/kimi_k3 \
  python/sglang/srt/environ.py \
  python/sglang/srt/distributed/device_communicators/custom_all_reduce_v2.py \
  test/registered/kernels/ops/kimi_k3
git commit -m "feat(kernels): port Kimi-K3 distributed kernels"
```

### Task 8: Remove Benchmark and Runtime Leakage

**Files:**
- Modify: all files staged by Tasks 2–7 as indicated by audit output

- [ ] **Step 1: Assert no Day-0 benchmark was ported**

```bash
git diff --name-only origin/main...HEAD |
  rg '(^benchmark/|/benchmark/bench_|bench_kimi|bench_kda|bench_hicache)'
```

Expected: no output.

- [ ] **Step 2: Assert no model or serving integration leaked**

```bash
git diff --name-only origin/main...HEAD |
  rg 'srt/(models|managers|speculative|multimodal|entrypoints|server_args|disaggregation)'
```

Expected: no output.

- [ ] **Step 3: Audit kernel imports**

```bash
rg -n \
  'sglang\\.srt\\.(models|managers|speculative|multimodal|entrypoints|server_args|disaggregation)' \
  python/sglang/kernels test/registered/kernels test/registered/jit
```

Expected: no output.

- [ ] **Step 4: Run formatting and static checks**

```bash
git diff --check
pre-commit run --files $(git diff --name-only origin/main...HEAD)
python3 -m compileall -q python/sglang/kernels test/registered/kernels test/registered/jit
```

Expected: all commands exit 0.

- [ ] **Step 5: Commit cleanup if needed**

```bash
git add python/sglang/kernels python/sglang/srt test/registered
git commit -m "test(kernels): decouple Kimi-K3 tests and remove benchmark artifacts"
```

Skip the commit only when the cleanup produces no diff.

### Task 9: Validate on B300 and Hopper

**Files:**
- Create: `docs/superpowers/plans/kimi-k3-kernel-validation.md`

- [ ] **Step 1: Run the single-GPU suite on B300**

Run every new one-GPU test file directly with `CUDA_VISIBLE_DEVICES=0`.

Expected: all supported tests pass; unsupported tests skip with precise SM or
toolchain reasons.

- [ ] **Step 2: Run the distributed suite on B300**

Run Task 7 commands with 4 GPUs, then the SP and all-reduce cases with 8 GPUs.

Expected: all ranks exit 0 and reference comparisons pass.

- [ ] **Step 3: Run architecture-specific Hopper cases**

Run tests whose `covered()` path supports SM90 on an H200 node.

Expected: Hopper paths pass or skip because the kernel is explicitly SM100-only;
no compile failure may masquerade as a skip.

- [ ] **Step 4: Compare representative outputs with the source branch**

For each family, record the source and port results for one production shape:

```text
set_mla_kv_concat_q
tiny_gemm
vision_rope
moe_route_quant_fused
kda_fused_decode
kda_ptx_prefill
kimi_k3_mla_output_gate
kimi_k3_attn_res
kimi_k3_all_reduce
kimi_k3_gemm_ag
kimi_k3_gemm_ar
kimi_k3_sp_collective
```

Expected: bitwise equality where documented; otherwise agreement within the
test's declared tolerance.

- [ ] **Step 5: Write and commit the validation report**

```bash
git add docs/superpowers/plans/kimi-k3-kernel-validation.md
git commit -m "docs: record Kimi-K3 standalone kernel validation"
```

### Task 10: Final Review

**Files:**
- Review: all files in `git diff origin/main...HEAD`

- [ ] **Step 1: Verify commit structure**

```bash
git log --oneline --decorate origin/main..HEAD
```

Expected: design, inventory, generic kernels/tests, KDA, K3 compute,
distributed kernels, cleanup if needed, and validation.

- [ ] **Step 2: Verify final scope**

```bash
git diff --stat origin/main...HEAD
git diff --check origin/main...HEAD
```

Expected: a kernel-only diff with tests and documentation.

- [ ] **Step 3: Verify clean worktree**

```bash
git status --short --branch
```

Expected: no unstaged or untracked files.
