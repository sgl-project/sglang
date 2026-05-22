# Step-03 Attention Init Coverage Tests

Python unittest equivalents of the bash smoke tests in
`test/registered/step03_coverage/`. Placed in `test/manual/` (not CI-registered)
because they require GPU cluster hardware and architecture-specific kernels.

## Files

| File | What it tests | Format |
|------|--------------|--------|
| `step03_test_utils.py` | Shared `MockModelRunner` factories, batch builders, helpers | utility (not a test file) |
| `test_step03_mha_backends.py` | Standard MHA backends via `MockModelRunner`: eager decode/extend, graph capture/replay, replay consistency | Python unittest — **no model weights** |
| `test_step03_mla_backends.py` | MLA backends via `MockModelRunner`: FlashInferMLA, FlashMLA (H100), CutlassMLA (B200), TRTLLM MLA | Python unittest — **no model weights** |
| `test_step03_e2e_runners.py` | Full-server E2E tests: all backend × runner combinations via `popen_launch_server --load-format dummy` | Python E2E — **needs GPU + model config in HF cache** |

## Unit tests (test_step03_mha_backends.py, test_step03_mla_backends.py)

These use `MockModelRunner` — no model weights needed. They test that
`init_forward_metadata` + `forward_decode`/`forward_extend` produces valid
(non-NaN) outputs for each backend. They also simulate the graph path:

```
init_cuda_graph_state(max_bs, max_tokens)
  → init_forward_metadata_capture_cuda_graph(bs, num_tokens, req_pool, seq_lens, ...)
  → init_forward_metadata_replay_cuda_graph(bs, req_pool, seq_lens, ...)
  → forward_decode(q, k, v, layer, fb)   # must not produce NaN
```

Run on the cluster (mounts `sgl-workspace/sglang` as the code source):

```bash
# Single file
python test_step03_mha_backends.py

# Specific test class
python test_step03_mha_backends.py TestTritonInit

# Specific test
python test_step03_mha_backends.py TestTritonInit.test_graph_replay_consistent
```

## E2E server tests (test_step03_e2e_runners.py)

Each test class launches a server with `--load-format dummy`, sends one
`/generate` request, and verifies the response contains a non-empty `text`
field with no error markers. Model weights are **not** loaded (dummy), but
the model config and tokenizer must be in the HF offline cache.

Overridable env vars (all have sensible defaults):

| Var | Default | Used by |
|-----|---------|---------|
| `SMALL_MODEL` | `Qwen/Qwen3-0.6B` | most MHA/runner tests |
| `DSV3_MODEL` | `lmsys/sglang-ci-dsv3-test` | MLA tests |
| `DSV4_MODEL` | `/flash_model` | DSV4 tests |
| `DSA_MODEL` | `deepseek-ai/DeepSeek-V3.2` | DSA test |
| `HYBRID_MODEL` | `nvidia/Nemotron-H-8B-Base-8K` | hybrid mamba test |
| `SWA_MODEL` | `openai/gpt-oss-20b` | SWA test |
| `EAGLE_TARGET` | `meta-llama/Llama-2-7b-chat-hf` | EAGLE test |
| `EAGLE_DRAFT` | `lmsys/sglang-EAGLE-llama2-chat-7B` | EAGLE test |
| `EAGLE3_TARGET` | `meta-llama/Llama-3.1-8B-Instruct` | EAGLE3 test |
| `TP_SIZE` | `1` | single-GPU tests |
| `TP_SIZE_LARGE` | `4` | multi-GPU tests (DSV4, DSA, DP idle) |
| `TEST_PORT` | `30000` | server port |

Run a specific E2E test on the cluster:

```bash
sudo srun --partition=gb300 --nodes=1 --gpus-per-node=4 --mem=0 --time=01:00:00 --mpi=pmix \
  --container-image '/mnt/vast/squash/lmsysorg+sglang+dev-cu13.sqsh' \
  --container-mounts '/mnt/home/cheng.wan/project/sglang:/sgl-workspace/sglang,...' \
  bash -c 'pip install sglang-kernel --upgrade --break-system-packages -q && \
           python /sgl-workspace/sglang/test/manual/attention/test_step03_e2e_runners.py TestTritonEager'
```

## Coverage matrix

### Unit tests (no model weights)

| Backend | Eager decode | Eager extend | Graph capture | Graph replay | Replay consistent |
|---------|:---:|:---:|:---:|:---:|:---:|
| Triton | ✓ | ✓ | ✓ | ✓ | ✓ |
| FlashInfer | ✓ | ✓ | ✓ | ✓ | ✓ |
| FA3 (SM≥80) | ✓ | ✓ | ✓ | ✓ | ✓ |
| TRTLLM MHA | ✓ | ✓ | ✓ | ✓ | — |
| TorchNative | ✓ | ✓ | — | — | — |
| FlashInfer MLA | ✓ | ✓ | ✓ | ✓ | ✓ |
| FlashMLA (SM90) | ✓ | — | ✓ | ✓ | — |
| CutlassMLA (SM100) | ✓ | — | ✓ | ✓ | — |
| TRTLLM MLA | ✓ | ✓ | ✓ | ✓ | ✓ |

### E2E server tests (--load-format dummy)

| Test class | Backend | Runner | Spec | Notes |
|------------|---------|--------|------|-------|
| `TestTritonEager` | triton | eager | — | |
| `TestFlashInferEager` | flashinfer | eager | — | |
| `TestFA3Eager` | fa3 | eager | — | skip SM≥100 |
| `TestTRTLLMMHAEager` | trtllm | eager | — | |
| `TestTritonCudaGraph` | triton | full CG | — | |
| `TestFlashInferCudaGraph` | flashinfer | full CG | — | |
| `TestFA3CudaGraph` | fa3 | full CG | — | skip SM≥100 |
| `TestTRTLLMMHACudaGraph` | trtllm | full CG | — | |
| `TestTritonPCG` | triton | PCG | — | |
| `TestFlashInferPCG` | flashinfer | PCG | — | step-03 regression guard |
| `TestFA3PCG` | fa3 | PCG | — | skip SM≥100 |
| `TestTRTLLMMHAPCG` | trtllm | PCG | — | step-03 regression guard |
| `TestBreakableCudaGraph` | triton | breakable CG | — | |
| `TestFlashInferMLACudaGraph` | flashinfer MLA | full CG | — | |
| `TestFlashMLACudaGraph` | flashmla | full CG | — | Hopper only |
| `TestCutlassMLACudaGraph` | cutlass_mla | full CG | — | B200 only |
| `TestTRTLLMMLACudaGraph` | trtllm MLA | full CG | — | |
| `TestDSV4CudaGraph` | dsv4 | full CG | — | needs /flash_model |
| `TestDSACudaGraph` | dsa | full CG | — | needs DeepSeek-V3.2 |
| `TestHybridMambaCudaGraph` | auto (mamba+attn) | full CG | — | |
| `TestSWACudaGraph` | auto (SWA) | full CG | — | |
| `TestTritonEAGLE` | triton | full CG | EAGLE | Llama-2-7b |
| `TestDSV4EAGLE` | dsv4 | full CG | EAGLE | motivating workload |
| `TestFA3EAGLE3` | fa3/triton | full CG | EAGLE3 | multi-layer draft |
| `TestDPIdleRank` | triton | full CG | — | DP=4 IDLE path |

## Step-03 migration note

When step-03 lands, update the unit tests to use the new API:
- `init_forward_metadata(fb)` → `init_forward_data(fb)` (eager)
- `init_forward_metadata_capture_cuda_graph(bs, num_tokens, ...)` → `init_forward_data_out_graph(fb)`
- `init_forward_metadata_replay_cuda_graph(bs, ...)` → `init_forward_data_out_graph(fb_view)`
The E2E tests need no changes (they test behavior not API).
