Issue #18077 — Next Steps Plan (Updated, as of issue_18077 branch)

Context: Benchmark + Optimize GLM-Image inference efficiency (SGLang-D vs. Diffusers).
Current repo status: 已经在 issue_18077 分支整理完成（rebase/冲突处理完成，基线文档也在本地）。
目标:

Step 2：修复 GLM-Image 的 Sequence Parallelism（SP）运行时错误（einops shape mismatch），让 2 GPU / 5 GPU 都能跑通。

Step 3：跑出 SP enabled 的性能证据（speedup / latency / throughput / memory），贴回 issue。

0. Workspace / Branch 状态确认（你已完成，但这里给统一入口）
Commands
cd /data/users/yandache/workspaces/sglang/repo/sglang-src
git checkout issue_18077
git status
git log -1 --oneline


确保你跑的是本地代码（任选一个）：

Option A: editable install (推荐)

pip install -e .


Option B: PYTHONPATH

export PYTHONPATH=$PWD/python:$PYTHONPATH

1. ✅ Completed: Benchmarking Baselines (Step 1) — Done

你已经完成并在 issue/本地报告中记录了 baseline：

SGLang-D vs Diffusers：SGLang-D ~8–13% faster；memory + ~0.45GB

Single GPU vs 2 GPU (--enable-cfg-parallel)：~1.19–1.27× speedup

Bench infra：bench_serving.py + 相关 offline scripts（PR #18154 还在 review/merge 中，但 Step1 你已完成）

Optional re-run (only if maintainers ask)
# server
CUDA_VISIBLE_DEVICES=0 sglang serve --model-path zai-org/GLM-Image --backend sglang --port 30000
# bench
python -m sglang.multimodal_gen.benchmarks.bench_serving \
  --model zai-org/GLM-Image --dataset random \
  --num-prompts 20 --width 512 --height 512 --max-concurrency 1

1.5 Single-GPU 启动时的 text encoder 问题（A01 报错）— 已修

现象：用 A01_single_gpu.sh 启动单卡时出现：
- `AttributeError: 'EncoderConfig' object has no attribute 'parallel_folding'`
- 随后 fallback 到 native T5，并出现大量 decoder 权重 MISSING，提示 "performance may be sub-optimal"。

原因：GlmImagePipelineConfig 没有覆盖 text_encoder_configs，继承的是基类默认 `(EncoderConfig(),)`。T5 加载器里会调用 `_get_folding_tp_group(config)`，该函数要求 config 是 TextEncoderConfig（有 parallel_folding）；传入 EncoderConfig 就会报错并走 fallback，fallback 用的 native T5 与 GLM-Image 的 encoder-only 结构不一致，所以出现大量 MISSING。

为何 01/03 可能没遇到：从代码路径看，01_start_server.sh 和 A01_single_gpu.sh 对 GLM-Image 的加载逻辑是一样的（都是 sglang serve + 同模型）。若之前 01/03 没报错，可能是：当时跑的代码版本不同、或当时并未用 01 真正起 GLM-Image 服务器、或只跑了 03 的对比脚本（不启动 serve）。

修复：在 GlmImagePipelineConfig 中显式设置 text_encoder_configs 为 T5Config（见 glm_image.py），这样 T5 走定制加载路径，不再报 parallel_folding 且权重正确。修复后单卡启动应不再出现上述错误与 fallback。

2. 🚧 Next: Fix Sequence Parallelism (SP) for GLM-Image (Step 2)
2.1 Problem（已确认）

当开启 SP（e.g. sp_degree=2, ulyess_degree=2）时，GLM-Image 在 runtime 报 einops error。

Base 实现 shard_latents_for_sp 默认假设 latents 是 3D（B, seq, C）

但 GLM-Image latents 实际是 4D（B, C, H, W），例如 512×512 → [1, 16, 64, 64]

你贴的 GlmImagePipelineConfig 当前没有 override shard_latents_for_sp → 会继承 base → 必挂

结论：GLM-Image SP broken 的直接原因就是 latent layout mismatch（NCHW vs BSC）。
（当然也可能存在 “token order / rope / attention layout” 的更深层问题，但第一步必须先修 shape contract）

2.2 Required Work（你要做的改动）
Step 2.2.1 实现 GLM 专用 shard_latents_for_sp（NCHW → shard）

目标：让 (B, C, H, W) 支持 sp_degree 切分。

推荐策略：沿 H 维 shard（最直观、可控；后续如需更均衡再做 flatten H×W shard）。

同时必须保证 unshard/gather 对称，否则 VAE decode 会炸。

Commands（定位代码）

rg -n "def shard_latents_for_sp" -S python/sglang/multimodal_gen/configs/pipeline_configs
rg -n "GlmImagePipelineConfig" -S python/sglang/multimodal_gen/configs/pipeline_configs


Edit file（通常是 glm_image.py）

vim python/sglang/multimodal_gen/configs/pipeline_configs/glm_image.py
python -m py_compile python/sglang/multimodal_gen/configs/pipeline_configs/glm_image.py

Step 2.2.2 Align with model expectations（sequence ordering / RoPE）

GLM-Image 的 freqs_cis/rotary embedding 是基于 (H, W) 生成的（你代码里 get_freqs_cis() 就是按 height/width 建的 hidden_states）。
因此 shard 的时候要保证：

每个 rank 看到的 latent patch 对应的 RoPE / position mapping 不被破坏

或者：freqs 也要按相同 shard 逻辑切分（如果模型 forward 是 per-rank 局部）

这里属于 “可能需要进一步验证”的点：先让不报错跑通，再看输出是否合理、性能是否提升。

Step 2.2.3 Minimal regression test（至少给一个可复现命令）

CI 真正加 2×GPU test 可能很难，但可以先做到：

issue 里贴一条命令：2 GPU + SP + 512×512 能跑通

后续如 maintainers 要求，再补 unit/integration test

2.3 Verification plan（本地/服务器怎么测）
2.3.1 Sanity: 1 GPU 能跑（避免你改坏正常路径）
pkill -f "sglang serve" || true
CUDA_VISIBLE_DEVICES=0 sglang serve \
  --model-path zai-org/GLM-Image --backend sglang --port 30000

python -m sglang.multimodal_gen.benchmarks.bench_serving \
  --model zai-org/GLM-Image --dataset random \
  --num-prompts 1 --width 512 --height 512 --max-concurrency 1

2.3.2 Core: 2 GPU + SP 跑通（原先必挂，现在必须过）
pkill -f "sglang serve" || true
CUDA_VISIBLE_DEVICES=0,1 sglang serve \
  --model-path zai-org/GLM-Image --backend sglang \
  --sp-degree 2 --ulyess-degree 2 \
  --port 30000

python -m sglang.multimodal_gen.benchmarks.bench_serving \
  --model zai-org/GLM-Image --dataset random \
  --num-prompts 1 --width 512 --height 512 --max-concurrency 1

2.3.3 Performance evidence: 1GPU vs 2GPU(cfg) vs 2GPU(SP)
mkdir -p logs


(A) 1 GPU

pkill -f "sglang serve" || true
CUDA_VISIBLE_DEVICES=0 sglang serve --model-path zai-org/GLM-Image --backend sglang --port 30000
python -m sglang.multimodal_gen.benchmarks.bench_serving \
  --model zai-org/GLM-Image --dataset random \
  --num-prompts 20 --width 512 --height 512 --max-concurrency 1 \
  | tee logs/bench_1gpu_512.log


(B) 2 GPU cfg-parallel

pkill -f "sglang serve" || true
CUDA_VISIBLE_DEVICES=0,1 sglang serve --model-path zai-org/GLM-Image --backend sglang --enable-cfg-parallel --port 30000
python -m sglang.multimodal_gen.benchmarks.bench_serving \
  --model zai-org/GLM-Image --dataset random \
  --num-prompts 20 --width 512 --height 512 --max-concurrency 1 \
  | tee logs/bench_2gpu_cfg_512.log


(C) 2 GPU SP

pkill -f "sglang serve" || true
CUDA_VISIBLE_DEVICES=0,1 sglang serve --model-path zai-org/GLM-Image --backend sglang --sp-degree 2 --ulyess-degree 2 --port 30000
python -m sglang.multimodal_gen.benchmarks.bench_serving \
  --model zai-org/GLM-Image --dataset random \
  --num-prompts 20 --width 512 --height 512 --max-concurrency 1 \
  | tee logs/bench_2gpu_sp_512.log


Optional: 1024×1024（更能体现 SP 的价值）
把 width/height 换成 1024，再跑一轮。

2.4 Multi-GPU on server (5×A100) — after 2 GPU stable

在 5×A100 上建议先尝试 sp_degree=5（如果支持）或 sp_degree=4（更常见能整除 H 或 token 数）。

Commands

pkill -f "sglang serve" || true
CUDA_VISIBLE_DEVICES=0,1,2,3,4 sglang serve \
  --model-path zai-org/GLM-Image --backend sglang \
  --sp-degree 5 --ulyess-degree 5 \
  --port 30000


如果不支持 5（或不整除导致问题），先试：

CUDA_VISIBLE_DEVICES=0,1,2,3 sglang serve \
  --model-path zai-org/GLM-Image --backend sglang \
  --sp-degree 4 --ulyess-degree 4 \
  --port 30000


然后同样用 bench_serving 测：

python -m sglang.multimodal_gen.benchmarks.bench_serving \
  --model zai-org/GLM-Image --dataset random \
  --num-prompts 20 --width 1024 --height 1024 --max-concurrency 1 \
  | tee logs/bench_5gpu_sp_1024.log

3. After Step 2: Re-benchmark + Report (Step 3)

当 SP 跑通之后，你需要在 issue 里贴：

“SP crash fixed ✅”

“SP speedup numbers (2 GPU and ideally 4/5 GPU)”

“Memory impact”

“Commands to reproduce”

Commands（从 log 抽关键指标）
rg -n "throughput|Latency|P99|Request throughput|Peak Memory" logs/*.log


（你把这些 grep 输出贴给我，我可以帮你自动整理成 issue 可贴的 markdown table。）

4. Optional Follow-ups (after SP works)

Profiling (torch profiler / nsight) → 找 attention/kernel/comm bottleneck

Router / concurrency scaling issue（你之前观察到 concurrency=1→2 吞吐不涨）

Docs update（说明 GLM-Image SP support + recommended config）

Summary Table (Updated)
Step	Description	Status
1	Baselines: SGLang-D vs Diffusers; 1GPU vs 2GPU cfg-parallel	✅ Done
2	Fix GLM-Image SP: override shard_latents_for_sp for NCHW + symmetric unshard; verify 2 GPU SP works	🚧 In progress (current focus)
3	Re-benchmark with SP enabled (2 GPU + 4/5 GPU A100), report speedup + memory	Next
4	Profiling + further optimizations + docs	Optional
References

Issue: #18077

Base implementation: python/sglang/multimodal_gen/configs/pipeline_configs/base.py

GLM config: python/sglang/multimodal_gen/configs/pipeline_configs/glm_image.py (GlmImagePipelineConfig)

Benchmark: python/sglang/multimodal_gen/benchmarks/bench_serving.py

(Optional) offline scripts from PR #18154 (pending merge)