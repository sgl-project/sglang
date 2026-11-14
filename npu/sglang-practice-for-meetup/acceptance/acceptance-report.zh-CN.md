# 验收报告 — DeepSeek-V4-Flash 在昇腾 950PR 上的 SGLang 部署与性能实测

**提交人**：（填你的名字 / 社区 ID）
**日期**：2026-11-14
**关联活动**：昇腾开源软件极客营 · SGLang Meetup（上海，2026-11-14）
**代码仓 / PR**：`sgl-project/sglang`，目标分支 `ifmn/npu/sglang-practice-for-meetup`（材料位于 `npu/sglang-practice-for-meetup/`）
**实践目录**：`ascend-950/`（连接与诊断）、`sglang-practice-pr/`（PR 材料与经验归档）

---

## 一、验收结论

在 **4 × 昇腾 950PR（每 die 128GB HBM，共 512GB）** 的容器环境（CANN 9.1.0 + torch_npu 2.10 +
SGLang 0.5.21+npu）中，完成 **DeepSeek-V4-Flash（284B 总参 / 13B 激活，原生 FP4+FP8 权重，156GB）**
的端到端部署与性能实测：

- 服务可用：`/health = 200`，`/generate` 输出正确（「The capital of France is → Paris. …」）
- 图模式（NPU Graph）开启成功：capture 13.25s，占用 5.06GB，**解码较 eager 模式提速 2.9–3.7 倍**
- 单流：**19.58 ms/token（≈51 tok/s）**；8k 上下文：**20.66 ms/token**（几乎不随上下文退化）
- 并发：c=8 输出 **178.9 tok/s**；饱和扫描 c=64 达 **803.2 tok/s**
- 对照厂商 A3（16 die、W8A8、图模式）公开的 50 ms TPOT 目标：本机 **4 die 上 20.66 ms（8k/1k）**，
  按单位算力约 **2.4 倍优于参考值**（target-class 对比，非严格 A/B）

**结论：达到 L3（单机调优级：图模式 + 调优工作点 + 容量包络已测量）。**

---

## 二、环境与版本（实测）

| 项 | 值 |
| --- | --- |
| 硬件 | 4 × Ascend950PR，131072 MB HBM/die，健康状态 OK，共 512GB |
| 宿主机 | Ubuntu 22.04 容器（镜像 `sglang-950_26.2`），754GB RAM，256 核，499GB 可用磁盘 |
| CANN | 9.1.0（ascend-toolkit / cann-9.1.0 / nnal / extra-ops） |
| Python | `/usr/local/python3.12.13`（注意：`/usr/bin/python3` 无 torch） |
| torch / torch_npu | 2.10.0+cpu / 2.10.0.post6，`torch.npu.is_available()=True`，4 devices |
| SGLang | 0.5.21+npu，源码 `/sgl-workspace/sglang`，分支 `release/2026930`，HEAD `545ed1c6`（支持 DeepSeek-V4 IndexCache） |
| 依赖 | sgl_kernel_npu 2026.9.0、memfabric_hybrid 1.3.0、sglang-router 0.3.2 |
| 模型 | `/workspace/shared_assets/OssSGLangMeetup/model/DeepSeek-V4-Flash-0731`（156GB，48 shards，FP4 experts + FP8，含 MTP） |
| 部署形态 | 单机 PD-mixed，`--tp-size 4 --dp-size 4 --enable-dp-attention --enable-dp-lm-head` |

---

## 三、关键问题与修复（部署过程中实测遇到，逐一解决）

| # | 现象 | 根因 | 修复 |
| --- | --- | --- | --- |
| 1 | `ValueError: Unexpected UE8M0 scale shape: got (8,8,32), expected (64,32)` | DeepSeek-V4 分组 `wo_a` 的 block scale 为 3D，950 MXFP8 权重重排前需展平 | 在 `w8a8_mxfp8.py::_layout_npu_arch35_ue8m0_weights` 校验前 reshape；已提供 `deploy/00_patch_npu_mxfp8.py` |
| 2 | `aclnnKvQuantSparseAttnSharedkvMetadata … not in libopapi.so` | CANN `custom_transformer` 等 vendor 未加入 op 路径 | source 各 vendor `set_env.bash`，并设置 `ASCEND_CUSTOM_OPP_PATH` |
| 3 | decode 图捕获崩溃 `npu_cudagraph_backend → deepseek_v4.py:5107` | **实为问题 2 的症状**（aclnn 算子在图捕获内同样需要） | 修复 op 路径后图捕获成功（13.25s / 5.06GB） |
| 4 | `ModuleNotFoundError: No module named 'tilelang'`（mHC layernorm） | 镜像未安装 tilelang | `SGLANG_OPT_USE_TILELANG_MHC_PRE/POST=False` |
| 5 | `Failed to build JIT module sgl_kernel_jit_dpsk_v4_hash_topk…` | hash top-k 融合 kernel JIT 构建失败 | `SGLANG_OPT_USE_FUSED_HASH_TOPK=False`（可辅以 `pip install ninja`） |
| 6 | `[CamMoeDispatchNormal] HCCL_BUFFSIZE is too SMALL … 214MB > 200MB` | DeepEP normal 模式 HCCL 缓冲不足 | `HCCL_BUFFSIZE=1200`、`DEEPEP_HCCL_BUFFSIZE=2500` |
| 7 | 8k prompt × c=8 触发 `deep_ep.cpp:130 num_tokens(16384) > capacity(8192)` | DeepEP long-seq 每轮容量默认 8192 | `DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=1`、`DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=2048`、`DEEPEP_NORMAL_LONG_SEQ_ROUND=8`、`--chunked-prefill-size 8192`（已定位，待复测） |

---

## 四、实测性能（全部为现场实测，非估算）

### 4.1 eager 与图模式对比（同一模型/配置）

| 场景 | eager TPOT p50 | 图模式 TPOT p50 | 提升 | 图模式 out tok/s | 图模式 TTFT p50 |
| --- | --- | --- | --- | --- | --- |
| in128/out512, c=1 | 57.1 ms | **19.58 ms** | 2.92× | 50.7 | 111.9 ms |
| in8192/out1024, c=1 | 59.2 ms | **20.66 ms** | 2.87× | 46.1 | 1,214.3 ms |
| in2048/out256, c=8 | 86.3 ms | **23.55 ms** | 3.66× | 178.9 | 731.7 ms |

### 4.2 吞吐饱和（in512/out256，64 prompts，图模式）

| 并发 | req/s | out tok/s | TTFT p50 | TPOT p50 |
| --- | --- | --- | --- | --- |
| c=16 | 1.41 | 360.8 | 3,926 ms | 26.8 ms |
| c=32 | 1.88 | 482.5 | 5,398 ms | 40.8 ms |
| c=64 | 3.14 | **803.2** | 3,724 ms | 65.3 ms |

按 TPOT p50 ≤ 50ms 的 SLO，**调优工作点为 c≈32（约 482 tok/s）**；拐点在 c=32 以上。

### 4.3 长上下文

TPOT 与上下文长度基本无关：512 tokens 时 19.58 ms，8k tokens 时 20.66 ms。
长 prompt 的成本体现在 TTFT（128 tokens: 111.9 ms → 8k tokens: 1,214.3 ms），
与 V4 混合 CSA/HCA 压缩 KV 的设计预期一致。

### 4.4 精度（MMLU 5-shot）

| 项 | 值 |
| --- | --- |
| 数据集 | `cais/mmlu`（本机 `/workspace/shared_assets/datasets/cais/mmlu`，57 科目，test/dev parquet） |
| 提示 | 标准 HELM 风格 5-shot（每科取 dev 5 例），`Answer:` 结尾 |
| 评分 | `/v1/completions` 的 `logprobs=20` 对 `" A"…" D"` 取最大对数概率，非字符串解析 |
| 样本 | 每科 20 题 = **1,140 题**，实际作答 **1,015 题**（作答率 89.0%） |
| **准确率（作答口径）** | **89.04%**（904 / 1,015） |
| 准确率（保守口径，未作答记为错） | **79.3%**（904 / 1,140） |
| 最弱科目 | virology 60%、college_chemistry 65%、college_mathematics 65%、formal_logic 70%、high_school_physics 70% |
| 产物 | `/workspace/v4run/results/accuracy/mmlu_shot.json` |

未作答的 125 题为并发评测下的客户端超时（已重试 3 次），非错误答案；保守口径给出偏差上界。

### 4.5 与厂商参考对比

厂商最佳实践（A3 ×8 卡=16 die，W8A8，图模式，in8k/out1k）公开 TPOT **50 ms**；
本机 **4 die**、原生 FP4+FP8、in8k/out1k 实测 **20.66 ms** —— target-class 对比约 **2.4 倍** 单位算力优势
（参考行未注明并发，故不做严格 A/B 结论）。

---

## 五、可复现步骤

1. 连接：`ascend-950/connect.sh`（jump 代理 + 容器内 bash；token 有时效，需在控制台刷新）
2. 环境：source CANN 各 `set_env.sh` + `custom_transformer/customize` vendor，设置 `ASCEND_CUSTOM_OPP_PATH`
3. 修复：`python3 deploy/00_patch_npu_mxfp8.py`
4. 启动：见 `deepseek-v4-flash/docs/experiment-log.md` 的「validated launch env」与完整命令
5. 压测：`bench/run_matrix.sh`（或单条 `python3 -m sglang.benchmark.serving …`）
6. 汇总：`bench/summarize.py` → `docs/measurements.md`

仓库材料：
- `npu/sglang-practice-for-meetup/`：runbook、部署脚本、压测脚本、修复补丁、实验日志、故障表
- `sglang-practice-pr/`：`PR_DESCRIPTION.md`、`EXPERIENCE_ARCHIVE.md`、`practice-package.tar.gz`、`ACCEPTANCE.md`

---

## 六、证据清单（提交时一并附上）

- [ ] `/workspace/v4run/results/dsv4-*.jsonl`（原始压测数据）与 `summary` 输出
- [ ] `/workspace/v4run/logs/server-v8graph.log`（含图捕获成功行 13.25s/5.06GB、ready to roll）
- [ ] `npu-smi info` 截图（4 × Ascend950PR，HBM 占用 ≈94%）
- [ ] `/generate` 正确输出截图（Paris）
- [ ] `mmlu_shot.json` 与 MMLU 评测终端输出（89.04% / 1015 answered）
- [ ] 压测终端输出截图（4.1/4.2 表格对应）
- [ ] PR 链接（推送后）；未推送前可先附 `PR_DESCRIPTION.md`

---

## 七、局限与未完成项（如实说明）

1. **PD 分离未实测**：需要两个资源池（≥8 die），4 die 切片无法拆分；本报告已将其列为 scale-blocked。
2. **8k × c>1 未复测**：问题 7 的配置修复已给出但未再次验证；当前验证包络为「短/2k 提示 c≤64；8k 提示 c=1」。
3. **精度**：已完成 MMLU 5-shot（89.04% 作答口径 / 79.3% 保守口径）；未做 GSM8K/多模态/长上下文精度。
4. **参考对比非严格 A/B**：厂商 50ms 行未注明并发；本机为单流/闭环保留实验，已在正文标注。
5. **实例时效**：平台 token 有时效性，证据需在实例回收前打包留存。

---

## 八、申请验收

以上材料满足「可部署、可测量、可复现、问题可追溯」的实践验收标准。请按活动要求提交（PR / 群内表单 / 现场演示任选其一），
并在提交时注明：**4 × 950PR、图模式、单流 19.6 ms/token、c=64 达 803 tok/s、对照厂商参考 2.4×/单位算力**。
