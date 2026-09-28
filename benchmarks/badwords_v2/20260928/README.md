# SGLang badwords：ModelRunner V2 流程适配实验

实验日期：2026-09-28。状态：**实现与本轮实验完成；性能与行为等价性尚未验收**。

本分支按实验包发布，包含已测试代码、分层补丁、测试、合成负载和指标。它**没有将实现移植到本 fork 的 main**。实验运行基线是 SGLang 0.5.20 的 `59a0eb4d843c18b23ecb2c77dc7dfbaea3b0a543` 加当时已有改动；本分支父提交是公开 fork 的 `fae90abf6e15aaffb6fd924a439253674771487d`，不包含实验使用的 DFlash V2 文件。不得把本报告的测试结果解释为此 fork main 上的验证结果。

未推送实验基线的内部 Git 历史或整棵源码树。复现需要相应基线及其已有依赖；只有本公开分支不足以从零构建完全相同的实验服务。原始源码快照、全部请求记录和服务日志另有独立实验归档。这里保留审阅代码与分析结论所需的文件。

## 结论

- 完成了 CUDA badwords 路径的请求状态、采样上下文和 accepted-token 提交接口适配；普通 decode 与 DFlash 使用同一提交入口。
- 原有 Triton `_match` 和 `_commit` 内核未改。本轮主要是 Python 状态/生命周期接口重构，并非新的 GPU matcher 或完整 vLLM ModelRunner 移植。
- 测试副本和写回后的工作目录均为 **202 passed + 33 subtests passed**。已测流量未发现禁词 token 序列泄漏。
- 没有证明性能提升。并发 32 的吞吐相对原 CUDA 实现下降约 3%～6%；全矩阵最差为 DFlash 并发 8、mixed100，下降 12.67%。
- 基线与最终候选 5,440 对请求中，5,419 对输出 token IDs 完全一致，21 对不同。不能声称逐位等价、随机分布等价，或已具备充分生产验证。

## 代码与补丁结构

| 文件/目录 | 内容 |
|---|---|
| `patches/01-cuda-badwords-baseline.patch` | 本次重构依赖的前序 badwords API、CPU/CUDA 路径和测试；排除无关的多模态和 Transformers 修改 |
| `patches/02-v2-state-context-commit.patch` | 本次适配本身：5 个生产源码文件、1 个更新的测试、3 个新增测试/辅助文件 |
| `snapshots/baseline/bad_words_device.py` | 性能对照中的原 CUDA 模块 |
| `snapshots/candidate/bad_words_device.py` | 最终测试过的 CUDA 模块 |
| `tests/` | 更新后的设备测试、新上下文测试、vLLM mask 对照，以及前序 processor 测试 |
| `reference/` | 锁定版本的 vLLM V2 参考源码及来源 SHA256 |
| `scripts/benchmark.py` | 本轮测量脚本，发布时仅将模型目录和结果目录参数化 |
| `results/` | 全矩阵指标、微基准、输出一致性、环境、数据集和测试报告 |

补丁 01 是前序实现的依赖快照，**不是全部由本轮新增的代码**。它包含可选的 Rust backend 分支，但本轮使用 CUDA backend，未发布或验证 Rust 扩展。补丁 02 才是本轮适配的增量。

### 五个生产文件的改动

下列路径相对于实验源码根目录。

| 文件 | 改动 | 目的 |
|---|---|---|
| `python/sglang/srt/sampling/bad_words_device.py` | 增加不可变请求配置、状态获取入口、固定批次视图、stream fence、采样上下文、提交保护和参数检查 | 明确状态及每次调用的归属 |
| `python/sglang/srt/managers/schedule_batch.py` | 新增显式 `Req.bad_words_state`；retraction 清除句柄 | 将重建前后设备状态分开 |
| `python/sglang/srt/sampling/sampling_batch_info.py` | 新增 `bad_words_context`；forward copy 清除一次性上下文 | 防止复制旧采样事务 |
| `python/sglang/srt/layers/sampler.py` | 普通 decode 取走局部上下文，采样后调用统一提交入口 | 提交对象绑定到本次采样 |
| `python/sglang/srt/speculative/dflash_worker_v2.py` | verify 后取走上下文；accept 后提交 `out_tokens/commit_lens` | 用实际接受结果更新历史 |

### 请求级设备状态

`BadWordsRequestSpec` 是 frozen dataclass，保存去重后的 token 序列、flattened IDs、offsets 和最大前缀长度；构建时做非空、token 类型及资源上限检查。`DeviceState` 持有这份配置和 CUDA 后缀 ring。

设备状态从隐式 `req._bad_words_device` 改为显式 `req.bad_words_state`。`BadWordsStateManager.acquire()` 是一个轻量状态获取入口，并不是 worker 全局 slot allocator：首次使用或 retraction generation 变化时，从请求的 CPU `output_ids` 重建，其余步骤复用设备状态。它不会每步重新从 CPU 输出历史做匹配。

ring 只保存最长禁词前缀需要的后缀，prompt 不进入新请求的生成历史。没有移植 vLLM 的完整 `all_token_ids` 表，也没有新增全局 slot 复用机制。

### 固定批次视图与事件依赖

`DeviceBatch` 改成 frozen dataclass；请求状态引用保存为 tuple，同时记录 batch size 和 commit eligibility。每次调用生成自己的 pointer table，后续 request 重排、过滤或列表修改不能改变该调用已构造的映射。tensor 本身并非深度不可变，内部代码仍需遵守接口约束。

GPU event 负责跨 stream 的先后顺序；`record_stream` 负责 CUDA allocator 的内存生命周期，两者不能互相替代。固定视图内部的 `_StreamFence` 允许更新完成事件，而不修改请求到行的映射。第一版曾同时保留上传事件和新的完成事件，产生多余等待；最终版恢复用完成事件覆盖已经包含的上传依赖。

pointer table 仍每次分配 pinned staging 并上传，**尚未实现可复用 staging pool**。

### 一次采样的上下文和统一提交

原实现将本次 `DeviceBatch` 动态挂到 `sampling_info._bad_words_device_batch`，提交时再查找并清空。现在流程为：

```python
# mask 完成后，移出此次调用的上下文
context = take_bad_words_context(sampling_info)

# ordinary decode：采样及 TP 同步之后
commit_accepted_bad_words(context, sampled_tokens)

# DFlash：acceptance 得出接受结果之后
commit_accepted_bad_words(context, out_tokens, commit_lens)
```

同一 `BadWordsSamplingContext` 再次提交会报错；旧上下文尚未取走时再次创建也会报错。它提供“防止同一对象重复提交”的保护，不能单靠一个布尔值证明所有路径都必然提交了一次。`None` 上下文仍允许无操作。

`copy_for_forward()` 清除一次调用专属的上下文，避免下一次 forward 复制旧上下文。普通 decode 和 DFlash 的提交位置总体沿用原有逻辑，本轮没有重写 acceptance 算法。

### DFlash 匹配与提交语义

```text
已提交历史 H = [a, b]
verify 输入   = [b, c, d]     # 第 0 列 b 已提交

row 0 匹配历史 [a, b]
row 1 匹配历史 [a, b, c]
row 2 匹配历史 [a, b, c, d]

若接受 c、拒绝 d，replacement 为 x：只提交 [c, x]
```

形式上 row j 使用 `H + candidates[1:j+1]`。mask 阶段只读取历史，commit 阶段只写接受结果；rejected drafts 不进入 ring。中间 prefill chunk 的伪采样由已有 commit mask 跳过。bonus/replacement 属于实际提交结果。

retraction 时清掉请求的状态句柄。旧上下文仍持有旧状态，新调用创建新状态，旧提交不能写进新对象。正常结束/取消时依赖请求及在途视图释放引用，不存在新增的全局状态注册表；session 长期保留的 Req 可能保留其有界设备状态直到 session 清理。

## 实验配置和方法

- 单张 NVIDIA H100 80GB HBM3；SGLang 0.5.20；Torch 2.13.0+cu130；Triton 3.7.1。
- Qwen3-4B；DFlash 使用 Qwen3-4B-DFlash-b16。模型 revision 和容器 digest 未采集。
- TP/PP/DP=1，FA3，dtype=auto，context length=4096，mem fraction=0.4，max running=32，chunked prefill=512。
- decode CUDA Graph batch sizes `[1,2,4,8,16,24,32]`，prefill eager，overlap 开启，radix cache 关闭，seed=42。
- 两策略均设置 `SGLANG_BAD_WORDS_BACKEND=cuda`、`SGLANG_BAD_WORDS_KEEP_SYNC=0`。默认 backend 没有改为 CUDA。
- 输入 256 tokens，输出 128 tokens，temperature=0，ignore_eos=True；每个批次先预热，停止计时后再检查实际 output IDs 中的禁词。
- 并发 1/8/32，每场景每轮分别 16/64/192 请求；每轮共 1,360 请求，两轮第二轮反转场景顺序。

| 场景 | 含义 |
|---|---|
| none | 无禁词，对照整体运行变化 |
| inert100 | 100 条通常不会命中的短语 |
| mixed100 | 半数请求携带 100 条禁词，其余无禁词 |
| varied100 | 100 条不同长度前缀的短语 |
| active1 | 禁止 `10`，实际改变连续数字生成过程 |

### 运行顺序与数据选择

最初的候选首轮在正确性、生命周期、取消测试之后运行，基线则从新服务直接开始测量。候选 DFlash 首轮连 none 场景也明显变慢，重启后恢复，不能将该轮差异直接归因于补丁。该组结果被保留作诊断，不进入正式对比。

最终对比使用 `baseline-*` 与 `candidate-clean-*`：各自从新服务开始，先运行相同预热和 round 1，再在同进程运行反向顺序的 round 2。最终候选的 HTTP 正确性测试安排在性能测试后，避免改变测量前状态。

全部实验共保留 16,320 个测量请求；正式对比使用 10,880 个请求，另外 5,440 个属于前期候选诊断。公开包保存汇总和差异计数，未将约 117 MB 的逐请求原始记录全部加入 Git；完整原始记录在独立实验归档中。

## 性能结果及分析

吞吐按两轮总输出 tokens 除以两轮总时间计算。下表包含全部并发 32 场景及下降超过 10% 的场景；[完整指标](results/full-metrics.md) 和 [全部对比](results/performance-summary.md) 包含其余结果及延迟。

| 模式 | 并发 | 场景 | 基线 tok/s | 候选 tok/s | 变化 |
|---|---:|---|---:|---:|---:|
| regular | 32 | none | 4157.8 | 4034.8 | -2.96% |
| regular | 32 | inert100 | 4177.1 | 3927.2 | -5.98% |
| regular | 32 | mixed100 | 4190.7 | 3961.1 | -5.48% |
| regular | 32 | varied100 | 4137.8 | 3892.8 | -5.92% |
| regular | 32 | active1 | 4188.1 | 3980.3 | -4.96% |
| dflash | 8 | inert100 | 6714.9 | 5878.5 | -12.46% |
| dflash | 8 | mixed100 | 6697.2 | 5848.9 | -12.67% |
| dflash | 8 | varied100 | 6470.0 | 5713.5 | -11.69% |
| dflash | 32 | none | 9124.8 | 8781.0 | -3.77% |
| dflash | 32 | inert100 | 8905.6 | 8590.4 | -3.54% |
| dflash | 32 | mixed100 | 9018.0 | 8735.1 | -3.14% |
| dflash | 32 | varied100 | 8809.1 | 8458.3 | -3.98% |
| dflash | 32 | active1 | 5248.1 | 5094.1 | -2.93% |

### 新增开销

请求创建/重建时新增不可变配置对象、逐 token 校验和 CPU 元数据保存。每步新增上下文对象、固定视图构造、shape/dtype/device 检查和提交状态检查。没有新增匹配内核，但 Python 调度与元数据成本有所增加。

100 条禁词下，独立微基准测得的中位数如下；每组四次交错重复，每次 300 步。计时包含 CPU 准备和 GPU 执行，**不能当作纯 kernel 时间**。

| 请求数 | Verify width | 基线 µs/步 | 候选 µs/步 | 增量 |
|---:|---:|---:|---:|---:|
| 1 | 1 | 112.9 | 118.8 | +5.9 |
| 1 | 16 | 114.4 | 123.2 | +8.8 |
| 32 | 1 | 294.3 | 303.2 | +8.9 |
| 32 | 16 | 294.1 | 306.2 | +12.1 |

可以确认整个 badwords 路径有新增成本，但未逐项消融，不能把增量全部归到某个检查、对象或事件。第一版多余事件等待已修正，修正前后的原始微基准数据都保留。

### 端到端下降的归因边界

none 场景也下降约 3%～4%，说明需要进一步排查整体运行差异；这不是忽略禁词场景下降的理由。同机其他六张 GPU 持续有负载，CPU 亲和性未固定，基线和最终候选不是同时或严格交错运行。两轮数据不足以证明小幅差异的统计显著性。

因此当前结论是“测得下降，尚未充分归因”，不是“全部来自代码”，也不是“全部属于噪声”。尤其 DFlash 并发 8 的 11.7%～12.7% 下降，需要保留为待调查项。不能以架构更清楚为理由宣称性能验收通过。

## 正确性证据和剩余问题

### 已验证

- 完整测试：独立副本与写回工作目录均 202 项 + 33 子测试通过。多数为原有回归，不能理解成 202 项全新的并发正确性证明。
- 完整 mask 对照：100 组随机输入与锁定 vLLM V2 Triton kernel 一致，覆盖 prompt、slot 重排、verify width 1/2/4/16；另有 CPU reference 随机 ring/acceptance 检查。
- 状态测试：重复提交、未消费上下文、零/部分/全部接受、replacement/bonus、CPU 输出历史滞后、chunk 跳过、行重排、跨 stream、retraction generation 隔离、引用释放。
- 最终候选每种模式：100 native 请求（含 temperature=0.7）、2 OpenAI API 请求、96 个生命周期请求、16 个取消请求、1 次超限拒绝、4 个 EOS/stop/logprob 检查均通过。
- native 及性能脚本检查实际 output IDs 的禁词序列；OpenAI HTTP 测试主要验证接口兼容，不能冒充完整 token 级验证。

### 21 对输出差异没有闭环

| 并发 8 / active1 | 基线与候选不同 | 基线自身跨轮不同 | 候选自身跨轮不同 |
|---|---:|---:|---:|
| ordinary decode | 3/128 | 0/64 | 3/64 |
| DFlash | 18/128 | 8/64 | 16/64 |

未发现禁词泄漏，但不代表全部行为等价：错误历史或过度 mask 也可能不泄漏禁词。DFlash 基线本身存在变化，提示需要控制调度和数值因素；普通 decode 基线一致、候选出现差异，更不能直接归为正常波动。

尚未捕获分歧请求在首个不同 token 处的历史、完整 mask 和 logits，因此没有排除实现问题，也没有证明逐位或随机分布等价。

### 未覆盖或依赖的条件

- 未通过 HTTP 强制触发真实显存压力 retraction；对应隔离主要有单元测试证据。
- 未穷尽所有 overlap/cancel 竞态，未做长期压力或 compute-sanitizer 验证。
- GPU accepted-length 数值信任内部 acceptance 输出，热路径只检查元数据，没有读回数值逐项校验。
- 仅单 GPU、非 PD、普通 decode 与 DFlash uniform-width chain；未验证多 GPU、ragged/tree speculation。
- 没有新增全局 slot 管理、可复用上传池或 admission 阶段的一体化 processor 框架；这是 V2 工作流思想的局部适配。

优先后续工作：先定位 21 对输出的首个分歧，区分 mask、历史和 logits 差异；再固定 CPU 条件、交错重复，逐项消融新对象与校验成本。完成这些之前，不建议将本实验认定为可以无条件替换基线的生产版本。

## 复现与审阅

请在**具有对应 DFlash V2 及前序依赖的实验基线**副本上操作。不要在本 fork main 上强行应用补丁或用模糊匹配覆盖失败的 hunk。

```bash
# 先确认基线与文件前置条件；已有 CUDA 实验基线时只需要补丁 02。
git apply --check /path/to/01-cuda-badwords-baseline.patch
git apply /path/to/01-cuda-badwords-baseline.patch
git apply --check /path/to/02-v2-state-context-commit.patch
git apply /path/to/02-v2-state-context-commit.patch
```

完整启动参数见 [launch-config.json](results/launch-config.json)。其中路径为占位符；普通 decode 删除 DFlash 的两个参数。使用不同进程顺序测试基线和候选，正确性流量放在性能测试之后。

```bash
export SOURCE_DIR=/path/to/patched-experiment-source
export MODEL_DIR=/path/to/models  # 含 Qwen3-4B 与 Qwen3-4B-DFlash-b16
export RESULT_DIR=/path/to/new-results
export PYTHONPATH="$SOURCE_DIR/python"
export CUDA_VISIBLE_DEVICES=0
export SGLANG_BAD_WORDS_BACKEND=cuda
export SGLANG_BAD_WORDS_KEEP_SYNC=0
# 按 launch-config.json 启动服务，默认端口 31082。
python scripts/benchmark.py candidate 1
python scripts/benchmark.py candidate 2

# 在相应实验源码中执行；需其已有的测试与依赖。
python -m pytest -q   test/registered/unit/sampling/test_bad_words_processor.py   test/registered/unit/sampling/test_custom_logit_processor.py   test/registered/unit/sampling/test_sampling_batch_info.py   test/registered/unit/sampling/test_sampling_params.py   test/registered/unit/sampling/test_bad_words_device.py   test/registered/unit/sampling/test_bad_words_context.py   test/registered/unit/sampling/test_vllm_parity.py
```

发布版 benchmark 只参数化了机器路径；模型负载、预热、并发、请求数、禁词检查和计时方法保持原实验。发布包未重新部署该 fork main 或运行新的 GPU 测试。

## 证据索引

- [完整代码补丁](patches/02-v2-state-context-commit.patch)，[前序依赖补丁](patches/01-cuda-badwords-baseline.patch)
- [最终设备模块](snapshots/candidate/bad_words_device.py)，[原设备模块](snapshots/baseline/bad_words_device.py)
- [全部指标](results/full-metrics.md)，[机器可读比较](results/performance-summary.json)
- [输出一致性](results/output-parity.json)，[最终微基准](results/microbench.json)，[第一版微基准](results/microbench-v1.json)
- [测试副本报告](results/unit-final.xml)，[写回后报告](results/integrated-unit.xml)
- [合成负载](results/dataset.json)，[软件环境](results/environment.json)，[源码哈希](results/source-audit.json)
- [vLLM 固定来源](reference/source.json)：V2 `bad_words.py` commit `ccfd1cea75eff8956c5d08b15a7eaec8891642e9`

发布物中不包含访问凭证、远端 Git 配置、内部仓库祖先历史、模型权重，或与 badwords 无关的多模态/Transformers 修改。测试 XML 的机器标识和绝对路径已脱敏，计数与结果保持不变。文件校验清单见 `SHA256SUMS.json`。
