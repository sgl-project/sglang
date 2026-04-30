---
doc_type: roadmap
slug: ug-official-alignment
status: active
created: 2026-04-29
last_reviewed: 2026-05-01
tags: [ug, bagel, srt, diffusion, official-alignment]
related_requirements: []
related_architecture: [ug-runtime]
---

# UG Official Alignment Roadmap

## 1. 背景

UG runtime 已经从 fake G runtime proof 推进到 SRT-owned session、native BAGEL/Qwen2-MoT shell、native SRT G velocity、experimental interleave API 和真权重 smoke。下一阶段的首要目标不是继续扩大调度能力，而是先证明 SGLang UG 在用户可见结果上对齐官方 BAGEL 框架。

本 roadmap 从当前代码状态重新起步，优先覆盖三条结果线：VLM 文本理解结果、文生图结果、图像编辑结果。只有这些结果能和官方框架稳定对齐后，才继续做更深的 native model hardening、server/CLI 产品化和 batching。

## 2. 范围与明确不做

### 本 roadmap 覆盖

- 建立官方 BAGEL 与 SGLang UG 的可重复对齐 harness。
- 对齐 VLM image+text 到 text 的短 greedy 输出；完整 logits parity 另作后续加严项，不作为当前 Phase 2 的完成口径。
- 对齐 text-to-image 的 latent、velocity、decode image 和最终图像结果。
- 对齐 image-edit 的输入图、文本指令、生成图结果。
- 把对齐结果接到 experimental API、server 和 CLI smoke。
- 在结果对齐后继续 native BAGEL/Qwen2-MoT model path hardening。
- 最后再做单卡 multi-session guard/batching 的产品化闭环。

### 明确不做（本 roadmap 不覆盖）

- 不在结果对齐前追求吞吐、并发或多卡。
- 不支持 disaggregation、CFG parallel、multi-GPU BAGEL。
- 不把官方 BAGEL/seed 仓库函数 import 到 SGLang runtime；官方框架只能作为离线对照 harness 的 reference。
- 不先承诺 OpenAI-compatible 公共 API；UG 入口仍以 experimental internal API 为准。
- 不做模型质量调参；先做同权重、同 seed、同输入的实现对齐。

## 3. 阶段边界

本文里的“阶段”按 roadmap item 分组定义，后续讨论默认使用这里的口径。

- **Phase 1: parity harness** — 只包含 `ug-official-parity-harness`。目标是建立官方 BAGEL 与 SGLang UG 的离线对齐基座，固定 checkpoint、seed、prompt、image、sampling params 和 artifact 协议。Phase 1 结束条件是该 feature 通过 acceptance 并在 items.yaml 标为 `done`。
- **Phase 2: VLM official result parity** — 只包含 `ug-vlm-official-parity`。目标是先把 image+text -> text 的 U/VLM clean path 和官方 BAGEL 对齐；这一步未通过前不进入 G 正式验收。
- **Phase 3: VLM entrypoint smoke** — 只包含 `ug-vlm-entrypoint-official-smoke`。目标是用真权重通过 experimental Python API、HTTP 和 CLI 跑通 VLM smoke，让 U 这条线先独立闭环。
- **Phase 4: G official result parity** — 包含 `ug-sampling-cfg-parity`、`ug-t2i-official-parity`、`ug-edit-official-parity`。目标是在 VLM 已对齐的 SRT-owned session 之上，再对齐 CFG、文生图和图像编辑。当前三项均已完成，下一步进入 Phase 5 的 interleave 对齐。
- **Phase 5: interleave and full entrypoints** — 包含 `ug-interleaved-official-parity`、`ug-entrypoint-official-smoke`。目标是把已对齐的 U 和 G 串成 image -> text -> image -> text，并跑通完整 experimental 入口。目录 slug 仍保留 `ug-interleaved-official-parity`，但新 task/API 命名统一使用 `interleave`。
- **Phase 6: engineering hardening** — 包含 `ug-native-model-hardening`、`ug-single-card-batching-guards`、`ug-regression-and-docs`。目标是在结果对齐后继续收敛 native model path、单卡隔离/batching guard 和长期回归文档。

## 4. 子 feature 清单

1. **ug-official-parity-harness** — 建一个官方 BAGEL 与 SGLang UG 的离线对齐 harness，固定同 checkpoint、seed、prompt、image、sampling 参数和中间张量 dump。
   - 依赖：无
   - 状态：done
   - 对应 feature：2026-04-29-ug-official-parity-harness
   - 备注：Acceptance 已完成；live harness probe 在 `/tmp/ug-parity-phase1-codex-20260430` 通过，官方代码只允许在测试/对照脚本中使用，不能进入 runtime import 链。

2. **ug-vlm-official-parity** — 对齐 image+text 到 text 的 VLM 路径，证明 SRT U forward 的短 greedy generated token ids/text 与官方框架一致。
   - 依赖：ug-official-parity-harness
   - 状态：done
   - 对应 feature：2026-04-30-ug-vlm-official-parity
   - 备注：已完成 VLM text/token parity 收口；root cause 是 SRT U decode 没按官方 BAGEL `do_sample=False` 走 greedy，而是沿用默认 sampling。修复后远端真权重 `/tmp/ug-vlm-align-greedy` 通过，official/SGLang token ids 和 text 均为 `Audrey Hepburn in a red`；完整 logits parity 暂不要求。

3. **ug-vlm-entrypoint-official-smoke** — 用真权重跑通 experimental Python API、HTTP 和 CLI 的 VLM-only smoke。
   - 依赖：ug-vlm-official-parity
   - 状态：done
   - 对应 feature：2026-04-30-ug-vlm-entrypoint-official-smoke
   - 备注：入口 contract、Python API、HTTP `/v1/ug/vlm`、CLI `--ug-vlm-input`、pipeline/worker VLM-only 分流已实现；diffusion worker 现在会为 real BAGEL `UGPipeline` 自动创建并注入 native SRT Scheduler。远端 CPU 单测 21 tests OK；2026-04-30 真权重 Python API、CLI、HTTP server 均输出 `Audrey Hepburn in a red`，token ids 与官方短 greedy exact match，且 `prefill_count=1`、`velocity_count=0`、`append_image_count=0`。

4. **ug-sampling-cfg-parity** — 对齐 BAGEL denoise sampling 语义，包括 timestep shift、CFG text/img branch、cfg interval 和 renorm。
   - 依赖：ug-vlm-entrypoint-official-smoke
   - 状态：done
   - 对应 feature：2026-04-30-ug-sampling-cfg-parity
   - 备注：已完成 BAGEL official shifted denoise schedule 和 effective CFG scale helper；`UGDenoiseStage`、native BAGEL executor、opt-in G parity harness 已共用同一套语义。CPU 单测覆盖 timestep、interval、image CFG gating、`text_channel` renorm；远端 GPU2 真权重 2-step 256x256 T2I+Edit G parity probe 通过，输出目录 `/tmp/ug-phase4-g-parity`。

5. **ug-t2i-official-parity** — 对齐 text-to-image 路径，证明 init latent、多步 velocity、VAE decode 和最终图像与官方框架匹配。
   - 依赖：ug-vlm-official-parity, ug-sampling-cfg-parity
   - 状态：done
   - 对应 feature：2026-04-30-ug-t2i-official-parity
   - 备注：已完成 T2I official parity；harness 可通过 `SGLANG_TEST_BAGEL_G_TASKS=text_to_image` 单独运行 T2I。远端 GPU2 真权重 512x512/50-step quality probe report passed、diffs=0，输出图在 `/Users/bytedance/Desktop/ug-t2i-official-parity`。过程中捕获并修复 native SRT G velocity 手动 forward 未关闭 autograd 导致长步数 OOM 的 stop signal。

6. **ug-edit-official-parity** — 对齐 image-edit 路径，证明输入图像理解、编辑 prompt、G denoise 和 append image 的结果与官方框架匹配。
   - 依赖：ug-t2i-official-parity
   - 状态：done
   - 对应 feature：2026-04-30-ug-edit-official-parity
   - 备注：已完成 image-edit official parity；harness 可通过 `SGLANG_TEST_BAGEL_G_TASKS=image_edit` 单独运行 edit。远端 GPU2 真权重 1024x800/50-step quality probe report passed、diffs=0，counter 显示 `prefill_count=1`、`velocity_count=49`、`srt_sidecar_request_count=1`、`context_length=7065`，证明图像条件进入 SRT-owned session 且 CFG image sidecar 由 SRT 准备。输出图在 `/Users/bytedance/Desktop/ug-edit-official-parity`。

7. **ug-interleaved-official-parity** — 对齐多轮 interleave 流程，覆盖 image -> text -> image -> text 的官方结果顺序和 session 状态。
   - 依赖：ug-vlm-official-parity, ug-edit-official-parity
   - 状态：done
   - 对应 feature：2026-04-30-ug-interleaved-official-parity
   - 备注：已完成；G parity harness 新 task 名为 `SGLANG_TEST_BAGEL_G_TASKS=interleave`，保留 `interleaved` 读入 alias。远端 GPU2 真权重 1024x800/50-step quality probe report passed、diffs=0；post-image text/token ids exact match，counter 显示同一 `session_id=bagel-g-parity-interleave` 中 `prefill_count=1`、`velocity_count=49`、`append_image_count=1`、`srt_u_decode_request_count=4`。2026-05-01 远端 GPU2 2-step `interleave` task smoke 复验 OK。

8. **ug-entrypoint-official-smoke** — 用真权重跑通 experimental Python API、HTTP `/v1/ug/interleave` 和 CLI 的 VLM/生图/编辑/interleave smoke。
   - 依赖：ug-interleaved-official-parity
   - 状态：done
   - 对应 feature：2026-04-30-ug-entrypoint-official-smoke
   - 备注：已完成；opt-in true-weight smoke 显式覆盖 Python API 的 `t2i`、`edit`、`interleave`、`vlm`、`think_t2i` 五个 case；新增 `DiffGenerator.generate_interleave_serializable`，CLI 使用 `--ug-interleave-input/--ug-interleave-output`，HTTP 使用 `POST /v1/ug/interleave`，旧 `interleaved` 名称保留 alias。2026-05-01 远端 GPU2 Python 五 case OK：T2I/Edit 只返回 `image`，Interleave 返回 `image,text`，VLM 只返回 `text` 且 `velocity_count=0`，`think_t2i` 是 `mode=t2i + think` 开关而非独立模式；CLI/HTTP interleave 单 case OK。

9. **ug-native-model-hardening** — 基于 parity 证据继续收敛 BAGEL/Qwen2-MoT native model path，清理临时 shell/observer 边界并补齐 loader/权重映射。
   - 依赖：ug-entrypoint-official-smoke
   - 状态：done
   - 对应 feature：2026-04-30-ug-native-model-hardening
   - 备注：已完成；1) post-image U decode 改为 SRT-owned BAGEL decode loop，文本由 SRT tokenizer decode 后通过 SRT metadata 返回，不再由 BAGEL adapter 拼 token-id 字符串；2) BAGEL logical kv_len/rope 通过 `UGSessionRuntime.ug_model_state` 保存和传播，BAGEL native helpers 在 text/image/denoise 前从 SRT-owned state resync；3) CFG text/img logical context、token_count、sidecar marker 也进入 `ug_model_state`，`prepare_latents`/`predict_velocity` 从 session view resync CFG state，同时 full KV length 仍来自 SRT token binding。CPU runtime+BAGEL 56 tests、adapter+diffusion 29 tests 均通过；远端 GPU2 真权重 Python entry smoke 输出 `image,text`，text 为 `'<think>\n'`，stats 为 `prefill_count=1`、`velocity_count=3`、`append_image_count=1`、`srt_u_decode_request_count=5`。下一步进入 `ug-single-card-batching-guards`。

10. **ug-single-card-batching-guards** — 产品化单卡 multi-session 基础隔离和 batch guard，保证 UG non-causal/G segment 不污染普通 SRT 请求。
   - 依赖：ug-native-model-hardening
   - 状态：done
   - 对应 feature：2026-04-30-ug-single-card-batching-guards
   - 备注：已完成；保留现有 scheduler/ScheduleBatch compatibility key 作为边界，新增测试证明 mixed ordinary causal + UG non-causal-query extend batch 在分配前早失败；新增带 sidecar 的两 UG session close/release 隔离测试，关闭一个 session 只释放 owner 和自己的 sidecar，另一个 session 可继续 prefill。远端 CPU runtime/diffusion/prefill-adder 回归通过；未触发 scheduler 重写、KV 暴露或 official runtime import stop signal。

11. **ug-regression-and-docs** — 固化官方对齐回归、真权重 opt-in 测试、开发文档和 roadmap/feature acceptance 回写。
    - 依赖：ug-entrypoint-official-smoke, ug-single-card-batching-guards
    - 状态：done
    - 对应 feature：未启动
    - 备注：2026-05-01 完成当前回写：entrypoint smoke 显式覆盖 `t2i/edit/interleave/vlm/think_t2i`；G parity harness task 名切到 `interleave` 并保留 legacy alias；roadmap/items 记录最新真权重复验结果。后续若要把大权重 opt-in smoke 接成 CI gate，应另起 feature。

**最小结果闭环**：第 2 条 `ug-vlm-official-parity` 已完成。给定同一张图和同一段文本，SGLang UG 能通过 SRT-owned session 得到和官方 BAGEL 对齐的短 greedy generated token ids/text；这证明后续生图/编辑共享的 U 上下文没有走偏。注意：这不是完整 logits parity，也不是 Phase 1 的结束条件，Phase 1 只到 harness acceptance。当前 Phase 3 已完成真实 Python API、CLI、HTTP VLM-only smoke，Phase 4 已完成 CFG/T2I/Edit official result parity，Phase 5 已完成 interleave official parity 和完整 experimental entrypoint smoke，Phase 6 已完成 native hardening 与单卡 batching guards；本轮 `ug-regression-and-docs` 完成当前 roadmap 回写。

## 5. 排期思路

拆分按“先工具、先 VLM、再 G、再 interleave、最后工程化”的顺序推进。Phase 1 只做 parity harness，因为没有稳定对照基座，后续 VLM/文生图/编辑对齐会各写各的比较逻辑。Phase 2 只追 VLM 对齐，因为如果 U/image understanding 结果对不上，继续对齐 G 或重写 scheduler 只会把错误上下文产品化。Phase 3 先把 VLM 入口打通，确保 U 线能被真实入口复现。Phase 4 先把 CFG/sampling 语义钉住，再分别关文生图和编辑结果对齐。Phase 5 再把 U/G 串成 interleave 和完整入口。Phase 6 才做 native hardening 和 batching guard，让实现变得更像 SGLang 的长期形态。

最大的技术卡点是 CFG 与 image feature/native U forward。VLM 对齐能最早暴露 image preprocessing、VIT/VAE feature、rope/session state 是否正确；CFG 对齐则是公平比较生图和编辑前的必要条件。

## 6. 观察项

- 当前 codestable 里只有早期 `ug-g-runtime-proof` feature 文档，没有正式 UG roadmap；这份 roadmap 是从当前代码状态重新建的规划层。
- 早期 feature design 仍写着“不接真实 BAGEL”，已明显过时；后续 acceptance 时应补一份新的架构现状文档，而不是继续改旧 proof 文档。
- 当前代码已有 BAGEL/Qwen2-MoT native model pieces，但真权重结果对齐还没有成为自动化 gate；这正是本 roadmap 前半段要补的东西。
- `codestable/tools/validate-yaml.py` 当前仓库里不存在，因此 items.yaml 只能做 YAML 语法和人工 schema 自查。

## 7. 变更日志

- 2026-04-29：将 `ug-official-parity-harness` 标为 `in-progress`，对应 feature 为 `2026-04-29-ug-official-parity-harness`；当前只完成 design draft，尚未完成实现与验收。
- 2026-04-29：完成 `ug-official-parity-harness` implementation，状态保持 `in-progress`，等待 feature acceptance 后再标 `done`。
- 2026-04-30：补充分阶段定义，明确 Phase 1 只包含 `ug-official-parity-harness`；VLM/T2I/Edit/Interleaved parity 归入 Phase 2。
- 2026-04-30：完成 Phase 1 acceptance，`ug-official-parity-harness` 标为 `done`，新增 `codestable/architecture/ug-runtime.md` 记录 parity harness 边界；下一步进入 Phase 2 的 `ug-vlm-official-parity`。
- 2026-04-30：启动 Phase 2 第一项 `ug-vlm-official-parity`，对应 feature 为 `2026-04-30-ug-vlm-official-parity`。
- 2026-04-30：`ug-vlm-official-parity` 跑到真权重 stop signal；text-only 已 exact match，VIT packed sequence 已 bitwise match，并修复 UG non-causal image block 被 chunked prefill 切开的问题；image+text greedy token 仍未对齐，暂停进入 T2I/Edit。
- 2026-04-30：按用户确认重排阶段边界：Phase 2 只收 VLM，Phase 3 做 VLM-only 入口 smoke，Phase 4 才正式对齐 G/CFG/T2I/Edit；已有 G spike 不计入 roadmap 验收。
- 2026-04-30：`ug-vlm-official-parity` 完成短 greedy live parity 复验：cold max_new_tokens=8 官方与 SGLang token/text exact match；完整 logits parity 尚未验证，G/CFG/T2I/Edit 仍保持 Phase 4。
- 2026-04-30：启动 Phase 3 `ug-vlm-entrypoint-official-smoke`，目标是把已对齐的 VLM U path 接到 Python API、HTTP 和 CLI 的 VLM-only smoke，先不进入生图/编辑。
- 2026-04-30：Phase 3 已完成 VLM-only 入口代码与 CPU smoke，但真权重 entrypoint 输出 `Audrey Hepburn is in the`，direct parity 复跑也出现 `Audrey Hepburn wearing a red` vs 官方 `Audrey Hepburn in a red`；将 `ug-vlm-entrypoint-official-smoke` 标为 blocked，并把 Phase 2 VLM text/token 对齐打回继续收口。
- 2026-04-30：完成 VLM text/token parity regression 收口；root cause 是 SRT U decode 仍在默认 sampling，和官方 BAGEL `do_sample=False` 不一致。新增 greedy decode 开关并让 BAGEL VLM path 使用 greedy 后，远端真权重 direct parity 与 `UGPipeline.forward_vlm` 均输出 `Audrey Hepburn in a red`，`ug-vlm-official-parity` 标为 done，`ug-vlm-entrypoint-official-smoke` 解除 blocked，剩余真实 server/CLI 进程级 smoke。
- 2026-04-30：重跑 Phase 3 真权重 smoke：手工挂 SRT Scheduler 的 `UGPipeline.forward_vlm` PASS，输出 `Audrey Hepburn in a red`，token ids 与官方短 greedy exact match；真实 CLI `--ug-vlm-input` 和 HTTP server 启动均失败在 diffusion worker 构造 `UGPipeline` 时缺少 SRT Scheduler，BAGEL real backend 无法创建 native SRT denoise executor。`ug-vlm-entrypoint-official-smoke` 标回 blocked，下一步先补 CLI/server 启动链路里的 SRT scheduler wiring。
- 2026-04-30：完成 Phase 3 入口 wiring：diffusion worker 在 real BAGEL `UGPipeline` 启动前自动创建 BAGEL language-model view 和 native SRT Scheduler，并注入 `server_args.ug_srt_scheduler`。远端真权重 CLI `--ug-vlm-input` 与 HTTP `/v1/ug/vlm` 均输出 `Audrey Hepburn in a red`，token ids 与官方短 greedy exact match；`ug-vlm-entrypoint-official-smoke` 标为 done。
- 2026-04-30：完成 Phase 4 第一项 `ug-sampling-cfg-parity`：抽出 BAGEL official denoise schedule 和 CFG gating helper，补齐 CPU 单测；远端 GPU2 真权重 2-step 256x256 T2I+Edit G parity probe 通过，输出目录 `/tmp/ug-phase4-g-parity`。下一步进入 `ug-t2i-official-parity`。
- 2026-04-30：完成 Phase 4 第二项 `ug-t2i-official-parity`：G parity harness 支持单独运行 text_to_image，并补齐 T2I counter 验收。远端 GPU2 真权重 512x512/50-step T2I official-vs-SGLang report passed、diffs=0，图像已拉到 `/Users/bytedance/Desktop/ug-t2i-official-parity`；同时修复 native SRT G velocity 未关闭 autograd 导致长步数 OOM 的 runtime bug。下一步进入 `ug-edit-official-parity`。
- 2026-04-30：完成 Phase 4 第三项 `ug-edit-official-parity`：G parity harness 支持单独运行 image_edit，并补齐 edit counter 验收。远端 GPU2 真权重 1024x800/50-step edit official-vs-SGLang report passed、diffs=0，图像已拉到 `/Users/bytedance/Desktop/ug-edit-official-parity`；Phase 4 收口，下一步进入 `ug-interleaved-official-parity`。
- 2026-04-30：启动 Phase 5 第一项 `ug-interleaved-official-parity`，对应 feature 为 `2026-04-30-ug-interleaved-official-parity`；第一版目标是复用 G official parity harness 证明同 session 的 image/text -> generated image -> post-image greedy text 对齐。
- 2026-04-30：完成 Phase 5 第一项 `ug-interleaved-official-parity`：G parity harness 支持 interleave task，远端 GPU2 真权重 1024x800/50-step official-vs-SGLang report passed、diffs=0，post-image greedy token/text exact match，图像已拉到 `/Users/bytedance/Desktop/ug-interleaved-official-parity`；下一步进入 `ug-entrypoint-official-smoke`。
- 2026-04-30：启动 Phase 5 第二项 `ug-entrypoint-official-smoke`，对应 feature 为 `2026-04-30-ug-entrypoint-official-smoke`；目标是把已证明的 interleave 路径跑过 Python API、CLI 和 HTTP experimental 入口。
- 2026-04-30：完成 Phase 5 第二项 `ug-entrypoint-official-smoke`：远端 GPU2 真权重 Python API、CLI、HTTP `/v1/ug/interleaved` 4-step smoke 均通过，输出 `image,text`，stats 证明同 session interleave 与 SRT sidecar 路径；该旧路径后续保留为 alias，主路径改为 `/v1/ug/interleave`；Phase 5 收口，下一步进入 Phase 6 `ug-native-model-hardening`。
- 2026-04-30：启动 Phase 6 第一项 `ug-native-model-hardening`，对应 feature 为 `2026-04-30-ug-native-model-hardening`；第一闭环是把 post-image U decode 的文本解码收回 SRT runtime。
- 2026-04-30：完成 `ug-native-model-hardening` 第一闭环：post-image U decode 走 SRT-owned BAGEL decode loop，输出文本由 SRT tokenizer decode 后返回；CPU runtime/adapter/BAGEL 与 diffusion pipeline 单测通过，远端 GPU2 真权重 Python entry smoke 输出 `image,text`，post-image text 为 `'<think>\n'`，stats 为 `prefill_count=1`、`velocity_count=3`、`append_image_count=1`、`srt_u_decode_request_count=5`。该 roadmap item 继续 in-progress，下一步处理 native state/rope 和 executor metadata 边界。
- 2026-04-30：完成 `ug-native-model-hardening` 第二闭环：新增 SRT-owned `ug_model_state` 窄口保存 BAGEL logical kv_len/rope，BAGEL native text/image/denoise helpers 从 runtime state resync，单测证明即使 adapter 本地 `gen_context` 被污染，append-image curr_rope 仍取 SRT-owned state；CPU runtime+BAGEL 56 tests、adapter+diffusion 29 tests 通过，远端 GPU2 真权重 Python entry smoke 继续输出 `image,text` 与 `'<think>\n'`。
- 2026-04-30：完成 `ug-native-model-hardening` 第三闭环并收口该 roadmap item：CFG text/img logical context、token_count、sidecar marker 进入 `ug_model_state`，sidecar request view 保留 owner model state，G latent/velocity preparation 从 session view resync CFG state，同时 full KV length 继续来自 SRT token binding；CPU runtime+BAGEL 56 tests、adapter+diffusion 29 tests 通过，远端 GPU2 真权重 Python entry smoke 继续输出 `image,text` 与 `'<think>\n'`。下一步进入 `ug-single-card-batching-guards`。
- 2026-04-30：启动 Phase 6 第二项 `ug-single-card-batching-guards`，对应 feature 为 `2026-04-30-ug-single-card-batching-guards`；范围限定为单卡 multi-session 正确性 guard，不做 scheduler 重写或吞吐产品化。
- 2026-04-30：完成 `ug-single-card-batching-guards`：新增 `ScheduleBatch.prepare_for_extend` mixed ordinary causal + UG non-causal-query 早失败测试，新增 sidecar-aware 多 session close/release 隔离测试，并把 sidecar close 顺序稳定化。远端 CPU `test_ug_session_runtime.py` 22 tests、`test_ug_diffusion_pipeline.py` 24 tests、`test_prefill_adder.py` 11 tests 通过；下一步进入 `ug-regression-and-docs`。
- 2026-05-01：完成当前 `ug-regression-and-docs` 回写：入口 smoke 明确四种模式 `t2i/edit/interleave/vlm`，`think` 作为 T2I 开关而非独立模式；Python 真权重五 case、CLI/HTTP interleave、G parity `interleave` task 均在远端 GPU2 复验通过。G parity task/API 新命名统一使用 `interleave`，旧 `interleaved` 只作为兼容 alias。
