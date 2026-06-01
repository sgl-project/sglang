# SGLang & SGLang Omni — Comprehensive Study Plan

## How to Use This Plan

- **Prerequisites**: Python 3.10+, PyTorch, CUDA, familiarity with transformer architectures and LLM serving concepts.
- **Time estimate**: ~12–16 weeks at 10–15 hrs/week (adjustable by skipping optional sections).
- Each module has **Learn** (reading/watching) and **Do** (coding/lab) tasks. Complete them in order.
- The "Deep Dive" sections are for contributors; skip them if your goal is just usage.

---

## Module 0: Orientation & Setup (Week 1)

| Topic | What to Learn | Resources |
|-------|---------------|-----------|
| What is SGLang? | Mission, history, key differentiators (RadixAttention, SGLang frontend language, zero-overhead scheduler) | `README.md`, [sglang-0.2 blog](https://lmsys.org/blog/2024-07-25-sglang-llama3/), [sglang-0.4 blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/) |
| Repository layout | Top-level structure: `python/sglang/`, `sgl-kernel/`, `test/`, `docs/`, `rust/`, `docker/`, `.github/workflows/` | `python/sglang/README.md`, browse the repo |
| Install from source & run | `pip install -e "python[all]"`, launch `sglang/serve.py` with a small model (e.g., Llama-3.2-1B) | `docs/get_started/install.md`, `docs/basic_usage/send_request.html` |
| Environment & dependencies | `pyproject.toml` deps: torch, flashinfer, flash-attn, xgrammar, outlines, sglang-kernel | `python/pyproject.toml`, `python/sglang/check_env.py` |
| CLI entry points | `sglang` CLI, `sglang/serve.py`, `killall_sglang` | `python/sglang/cli/main.py`, `python/sglang/cli/serve.py` |

**Do**: Install SGLang from source, launch a tiny model, send a completion request via curl. Observe the startup logs.

---

## Module 1: SGLang Runtime (SRT) Architecture (Weeks 2–3)

### 1.1 Server Lifecycle & Entry Points

| Topic | What to Learn | Key Files |
|-------|---------------|-----------|
| Launch flow | From CLI to `launch_server.py` to `SRT SRTTorchLaunchEngine` | `python/sglang/cli/serve.py` → `python/sglang/launch_server.py` |
| Server arguments | All CLI flags: model, tp, dp, ep, schedule policy, cache, mem fraction | `python/sglang/srt/server_args.py` (~8000 lines — reference) |
| Entry points | OpenAI-compatible API, Anthropic, Ollama, gRPC | `python/sglang/srt/entrypoints/` (16 files) |

### 1.2 Core Managers

| Topic | What to Learn | Key Files |
|-------|---------------|-----------|
| TokenizerManager | Tokenization, detokenization, conversation templates, interleaving text/images | `python/sglang/srt/managers/tokenizer_manager.py` (30-manager directory) |
| Scheduler | Batching policy, prefill/decode scheduling, RadixAttention-based prefix cache, TTI | `python/sglang/srt/managers/scheduler.py` |
| Data parallelism controller | DP coordination across replicas | `python/sglang/srt/managers/dp_controller.py` |
| Detokenizer | Streaming output detokenization | `python/sglang/srt/managers/detokenizer_manager.py` |

### 1.3 Memory & KV Cache

| Topic | What to Learn | Key Files |
|-------|---------------|-----------|
| RadixAttention | Prefix caching via radix tree; how it reuses KV across requests | `python/sglang/srt/mem_cache/radix_cache.py` |
| Hybrid cache | Hierarchical caching across GPU/CPU | `python/sglang/srt/mem_cache/hybrid_cache.py` |
| Memory pool | KV cache allocation, paged attention integration | `python/sglang/srt/mem_cache/base_prefix_cache.py`, `python/sglang/srt/mem_cache/memory_pool.py` |
| Storage backends | Disk, MoonCake, Tuna | `python/sglang/srt/mem_cache/storage_backend/` |

**Do**: Set `--log-level debug`, read the scheduler logs. Trace a single request through TokenizerManager → Scheduler → Model → Detokenizer.

**Deep Dive**: Walk through `python/sglang/srt/managers/scheduler.py` — understand the `event_loop`, `get_next_batch`, and how eviction works in the radix cache.

---

## Module 2: Model Support & Inference (Weeks 4–6)

### 2.1 Model Architecture Registry

| Topic | What to Learn | Key Files |
|-------|---------------|-----------|
| Model loader | Weight loading from HuggingFace, sharding, device mapping | `python/sglang/srt/model_loader/` (6 files) |
| Model execution | Forward pass orchestration, CUDA graph runner | `python/sglang/srt/model_executor/` (14 files) |
| 194+ model implementations | Directory of supported architectures (Llama, Qwen, DeepSeek, Gemma, etc.) | `python/sglang/srt/models/` (194 files) |
| Model configs | Config overrides for each model family | `python/sglang/srt/configs/` (51 files) |

### 2.2 Attention Backends

| Topic | What to Learn | Key Files |
|-------|---------------|-----------|
| FlashInfer attention | Default backend, page attention, MLA | `python/sglang/srt/layers/attention/` (39 entries) |
| Radix attention | Prefix-cached attention variant | `python/sglang/srt/layers/radix_attention.py` |
| FlashAttention integration | flash-attn-4 wrapper | `python/sglang/srt/layers/attention/` |
| MLA (Multi-head Latent Attention) | DeepSeek's MLA optimization | `python/sglang/srt/layers/attention/mla.py` |
| DP attention | Data-parallel attention | `python/sglang/srt/layers/dp_attention.py` |

### 2.3 Sampling & Logits

| Topic | What to Learn | Key Files |
|-------|---------------|-----------|
| Logits processor | Top-k/top-p, temperature, penalty | `python/sglang/srt/layers/logits_processor.py` |
| Sampler | Token sampling strategies | `python/sglang/srt/layers/sampler.py` |
| Constrained decoding | Grammar-guided: xgrammar, outlines, llguidance | `python/sglang/srt/constrained/` (10 files) |

### 2.4 Quantization

| Topic | What to Learn | Key Files |
|-------|---------------|-----------|
| Quant backends | FP4, FP8, INT4, AWQ, GPTQ, TorchAO, ModelOpt, QoQ | `python/sglang/srt/layers/quantization/` (46 files) |

**Do**: Pick two models (e.g., Llama-3.2 and Qwen2.5). Trace how `model_loader` picks the right class from `srt/models/`. Run them both with `--quant fp8` and compare memory.

**Deep Dive**: Read `model_executor/forward_batch.py` — understand how a batch of requests flows through a single forward pass.

---

## Module 3: Parallelism & Distributed Serving (Weeks 5–6)

### 3.1 Parallelism Strategies

| Topic | What to Learn | Key Files |
|-------|---------------|-----------|
| Tensor Parallelism (TP) | Sharding across GPUs per layer | `python/sglang/srt/distributed/communication_op.py` |
| Pipeline Parallelism (PP) | Layer pipelining across devices | Via `--pp` flag |
| Expert Parallelism (EP) | MoE expert sharding (DeepSeek, Mixtral) | `python/sglang/srt/distributed/` |
| Data Parallelism (DP) | Replicating model across workers | `python/sglang/srt/managers/dp_controller.py` |
| Elastic EP | Dynamic expert scaling | `python/sglang/srt/elastic_ep/` (3 files) |

### 3.2 Distributed Runtime

| Topic | What to Learn | Key Files |
|-------|---------------|-----------|
| Ray integration | Multi-node orchestration | `python/sglang/srt/ray/` (5 files) |
| gRPC layer | Rust extension for inter-node comm | `rust/sglang-grpc/`, `proto/` |
| Communication ops | Allreduce, send/recv, custom collectives | `python/sglang/srt/distributed/` |

**Do**: Run a model with `--tp 2` (on 2 GPUs). Compare throughput with `--tp 1`. Then run with `--dp 2` and observe request distribution.

---

## Module 4: Advanced Features (Weeks 7–8)

### 4.1 Prefill-Deploy Disaggregation

| Topic | What to Learn | Key Files |
|-------|---------------|-----------|
| PD disaggregation | Split prefill and decode across separate nodes | `python/sglang/srt/disaggregation/` (16 files) |
| MoonCake | KV cache transfer fabric | `python/sglang/srt/disaggregation/mooncake/` |
| Encode server | Prefill-only server for disaggregation | `python/sglang/srt/disaggregation/encode_server.py` |
| Protocol | gRPC-based KV transfer between prefill/decode | `proto/` |

### 4.2 Speculative Decoding

| Topic | What to Learn | Key Files |
|-------|---------------|-----------|
| Eagle | Draft model-based speculation | `python/sglang/srt/speculative/eagle/` |
| N-gram | n-gram based speculation | `python/sglang/srt/speculative/ngram/` |
| MTP | Multi-token prediction | `python/sglang/srt/speculative/mtp/` |
| D-Flash | Flash-based speculation | `python/sglang/srt/speculative/dflash/` |

### 4.3 Multi-LoRA & Plugins

| Topic | What to Learn | Key Files |
|-------|---------------|-----------|
| LoRA serving | Adapter loading, batching, scheduling | `python/sglang/srt/lora/` (16 files) |
| Plugin system | Runtime plugin architecture | `python/sglang/srt/plugins/` (2 files) |
| Function calling | Model-specific FC detectors (30 model families) | `python/sglang/srt/function_call/` (30 files) |

### 4.4 Observability

| Topic | What to Learn | Key Files |
|-------|---------------|-----------|
| Prometheus metrics | Latency, throughput, KV cache hit rate, queue depth | `python/sglang/srt/observability/` |
| Tracing | OpenTelemetry integration | `python/sglang/srt/observability/tracing/` |
| Profiling | PyTorch profiler integration | `python/sglang/profiler.py` |

**Do**: Enable disaggregation between two processes on the same node. Then set up speculative decoding with Eagle.

**Deep Dive**: Read the speculative decoding orchestrator in `python/sglang/srt/speculative/` — understand how the draft model and target model interact.

---

## Module 5: Model Accelerators & Kernels (Weeks 7–8)

### 5.1 JIT Kernels

| Topic | What to Learn | Key Files |
|-------|---------------|-----------|
| Triton kernels | SGLang's own JIT Triton kernels | `python/sglang/jit_kernel/` (54 entries) |
| FlashInfer kernels | Attention, MoE, norm, activation | Third-party, via `flashinfer` |
| Deep GEMM | DeepSeek-optimized GEMM kernels | `sgl-kernel/` |

### 5.2 CUDA Graph & Compilation

| Topic | What to Learn | Key Files |
|-------|---------------|-----------|
| CUDA graph capture | Capturing forward pass into CUDA graphs | `python/sglang/srt/model_executor/cuda_graph_runner.py` |
| Compilation passes | Model-level compilation and caching | `python/sglang/srt/compilation/` (13 files) |
| Piecewise CUDA graph | Graph capture for variable-length sequences | `python/sglang/srt/compilation/piecewise_cuda_graph.py` |

**Do**: Run a model with and without `--disable-cuda-graph`. Observe the throughput difference. Read the CUDA graph runner to understand how it works.

---

## Module 6: SGLang Frontend Language (Week 9)

| Topic | What to Learn | Key Files |
|-------|---------------|-----------|
| SGLang programming model | `sglang.function`, `gen`, `select`, `fork` | `python/sglang/lang/` |
| Interpreter | How SGLang programs are executed | `python/sglang/lang/interpreter.py` |
| Backend | Runtime backend for frontend | `python/sglang/lang/backend/` |
| Chat templates | Jinja-based template engine | `python/sglang/lang/chat_template.py` |
| Tracer | Program tracing and optimization | `python/sglang/lang/tracer.py` |

**Do**: Write 3 SGLang programs: (1) simple Q&A, (2) multi-turn conversation, (3) parallel generation with `fork`. Run them against the server.

---

## Module 7: SGLang Omni (Weeks 10–11)

### 7.1 What is SGLang Omni?

SGLang Omni refers to the framework's support for **omni-modal models** — models that process and generate multiple modalities beyond text: audio, speech, images, video, and their combinations. The term "Omni" in the codebase primarily centers on **Qwen3-Omni** and **Qwen2.5-Omni** architectures.

| Resource | Location |
|----------|----------|
| Omni redirect | `docs_new/docs.json` lines 31–38 redirect `/sglang-omni` → `https://sgl-project.github.io/sglang-omni/` |
| Omni cookbook | `docs_new/cookbook/omni/intro.mdx` (FishAudio TTS, speech) |
| Omni config | `python/sglang/srt/configs/qwen3_omni.py` (609 lines) |
| Omni model | `python/sglang/srt/models/qwen3_omni_moe.py` (735 lines) |
| ASR model | `python/sglang/srt/models/qwen3_asr.py` (199 lines, reuses audio encoder from omni) |

### 7.2 Omni Model Architecture (Qwen3-Omni-MoE)

| Component | What to Learn | Key File/Area |
|-----------|---------------|---------------|
| Audio Encoder | Speech/audio feature extraction, Qwen3 audio frontend | `qwen3_omni.py` config's `AudioEncoderConfig` |
| Vision Encoder | Image/video encoding | `qwen3_omni.py` config's `VisionEncoderConfig` |
| Thinker | Language model core (text reasoning) | `qwen3_omni.py` config's `TextConfig` |
| Talker (Code Predictor) | Non-autoregressive speech code prediction | `qwen3_omni.py` config's `TalkerCodePredictorConfig` |
| Talker (Text) | Text output from talker | `qwen3_omni.py` config's `TalkerTextConfig` |
| Full MoE | Mixture-of-Experts in the backbone | `python/sglang/srt/models/qwen3_omni_moe.py` |

### 7.3 Multimodal Processing Pipeline

| Topic | What to Learn | Key Files |
|-------|---------------|-----------|
| Multimodal processor | Feature extraction, input construction for images/audio/video | `python/sglang/srt/multimodal/processors/qwen_vl.py` (lines 584–629 handle omni audio) |
| Position encoding | MRoPE (Multi-dimensional RoPE) for omni models | `python/sglang/srt/layers/rotary_embedding/mrope.py` (function `get_rope_index_qwen3_omni`) |
| Input ID construction | Building input_ids with audio timestamps | `python/sglang/srt/multimodal/processors/qwen_vl.py` — `build_input_ids_with_timestamps` |
| Attention mask | `concat_v3_mm_proj_attn_mask` for multimodal projector | `python/sglang/srt/multimodal/processors/qwen_vl.py` |

### 7.4 Omni-Specific Features

| Topic | What to Learn | Key Files |
|-------|---------------|-----------|
| Disaggregation for Omni | How omni models are split for encode-only serving | `python/sglang/srt/disaggregation/encode_server.py` (handles `qwen2_5_omni`, `qwen3_omni_moe`, `qwen3_asr`) |
| EPD disaggregation test | Test covering image/video/audio input for omni | `test/registered/disaggregation/test_epd_disaggregation.py` (`TestEPDDisaggregationOmni`) |
| Rotary embedding dispatch | Router selecting omni-specific RoPE | `python/sglang/srt/layers/rotary_embedding/mrope_rope_index.py` |
| Config updates | Vision and audio encoder config auto-fixes | `python/sglang/srt/configs/update_config.py` (lines 274, 282) |

### 7.5 Related Omni/Omni-like Models

| Model | Description | Key File |
|-------|-------------|----------|
| Qwen2.5-Omni | Earlier generation omni model | Referenced in `encode_server.py`, `qwen_vl.py` processor |
| Qwen3-ASR | Speech recognition (reuses omni audio encoder) | `python/sglang/srt/models/qwen3_asr.py` |
| MiniCPM-o | Omni-capable MiniCPM variant with `get_omni_embedding()` | `python/sglang/srt/models/minicpmo.py` (lines 1775, 1782) |

### 7.6 Speech & Audio Generation (Omni Cookbook)

| Topic | What to Learn | Resources |
|-------|---------------|-----------|
| FishAudio S2-Pro | TTS serving with SGLang | `docs_new/cookbook/omni/FishAudio/` |
| Diffusion for audio/video | Multimodal generation pipeline | `python/sglang/multimodal_gen/` |

### 7.7 Omni in CI & Testing

| Topic | What to Learn | Key Files |
|-------|---------------|-----------|
| EPD disaggregation CI test | Omni-specific test image/video/audio | `test/registered/disaggregation/test_epd_disaggregation.py` |
| Diffusion CI | Omni reference in diffusion comparison benchmarks | `scripts/ci/utils/diffusion/run_comparison.py` |

**Do**: Run Qwen3-Omni (or Qwen2.5-Omni if available) with both text and audio inputs. Observe the multimodal processing log. Then run the `test_epd_disaggregation.py` test to see omni disaggregation in action.

**Deep Dive**: Read `qwen3_omni.py` and `qwen3_omni_moe.py` side by side — understand how the five sub-configs (AudioEncoer, VisionEncoer, Text, TalkerCodePredictor, TalkerText) compose into a single model. Trace the `get_rope_index_qwen3_omni` function to understand positional encoding for omni inputs.

---

## Module 8: Testing & CI (Week 12)

| Topic | What to Learn | Key Files |
|-------|---------------|-----------|
| Test infrastructure | `test/registered/` (automated CI), `test/manual/` (manual), `test/srt/` | `test/`, `test/README.md` |
| CustomTestCase | SGLang's test runner/assertions | `python/sglang/test/` (44 entries) |
| CI workflow orchestration | PR stages, nightly, weekly, hardware-specific | `.github/workflows/pr-test.yml`, `_pr-test-stage.yml` |
| Model-specific tests | E2E model validation | `test/registered/models_e2e/`, `test/registered/models/` |
| Performance tests | Latency/throughput benchmarks | `test/registered/perf/`, `benchmark/` |

**Do**: Run `pytest test/registered/models_e2e/` for a small model. Write a minimal test using `CustomTestCase` that launches a server and sends a request.

---

## Module 9: Custom Kernel Development (Week 13, Optional)

| Topic | What to Learn | Key Files |
|-------|---------------|-----------|
| JIT Kernel guide | Step-by-step for adding Triton JIT kernels | `add-jit-kernel` skill |
| sgl-kernel guide | Heavyweight AOT CUDA/C++ kernels | `add-sgl-kernel` skill |
| FlashInfer integration | Custom attention kernels | `python/sglang/srt/layers/attention/` |
| Deep GEMM wrapper | DeepSeek-optimized GEMM | `python/sglang/srt/layers/deep_gemm_wrapper/` |

---

## Module 10: Production Deployment (Week 14)

| Topic | What to Learn | Key Files |
|-------|---------------|-----------|
| Docker deployment | `docker/Dockerfile`, `docker/compose.yaml`, k8s configs | `docker/` |
| Multi-node | gRPC-based inter-node, Ray | `rust/sglang-grpc/`, `python/sglang/srt/ray/` |
| Monitoring | Prometheus metrics, tracing | `python/sglang/srt/observability/` |
| Router | SGL Router for load balancing | `experimental/sgl-router/` |
| Connector | External system integration (Azure, Redis, S3) | `python/sglang/srt/connector/` |
| Security | Env vars, API key management | `python/sglang/srt/environ.py`, `python/sglang/global_config.py` |

---

## Appendix A: Key Benchmark Scripts

| Script | Purpose | Location |
|--------|---------|----------|
| `bench_offline_throughput.py` | Offline throughput benchmark | `python/sglang/bench_offline_throughput.py` |
| `bench_one_batch.py` | Single-batch latency benchmark | `python/sglang/bench_one_batch.py` |
| `bench_serving.py` | Online serving with dynamic requests | `python/sglang/bench_serving.py` |
| `auto_benchmark.py` | Automated benchmark suite | `python/sglang/auto_benchmark.py` |
| Perf tests | CI performance regression tests | `test/registered/perf/` |

## Appendix B: Key Environment Variables

| Variable | Purpose | Defined In |
|----------|---------|------------|
| `SGLANG_*` | All SGLang env vars | `python/sglang/srt/environ.py` |
| Various legacy `SGL_*` | Deprecated aliases | Same file |

## Appendix C: Key CI Workflows

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `pr-test.yml` | PR + schedule | Main PR validation (592 lines, multi-stage) |
| `pr-test-{xpu,xeon,npu,musa,amd}.yml` | PR | Hardware-specific testing |
| `nightly-test-nvidia.yml` | Nightly | Full NVIDIA regression suite |
| `nightly-72-gpu-gb200.yml` | Nightly | Large-scale GB200 testing |
| `release-docker.yml` | Release | Docker image build & publish |
| `ci-auto-bisect.yml` | On failure | Automatic regression bisect |

## Appendix D: Contribution Path

1. **Start with Module 0–1** to understand the codebase.
2. **Pick a small model** (e.g., Llama-3.2-1B) and trace its full lifecycle.
3. **Fix a documentation issue** in `docs_new/`.
4. **Add a test** for an existing model following patterns in `test/registered/models_e2e/`.
5. **Add a model config** for a new model variant in `python/sglang/srt/configs/`.
6. **Contribute a JIT kernel** following the `add-jit-kernel` skill.
7. **Contribute an sgl-kernel** following the `add-sgl-kernel` skill.

---

*Generated from the SGLang repository at `/root/xigu01/sglang/`.*
