# 第 21 章 二次开发与贡献：从改文档到改内核

## 21.1 先建立正确的预期

SGLang 是"读起来有意思、改起来有门槛"的项目：

- 门槛最低：文档、示例、测试、benchmark 脚本；
- 门槛中等：新增模型、新增 processor、修 bug；
- 门槛最高：调度器、内存池、注意力 kernel、分布式通信——每一处改动都可能影响正确性（KV 缓存）或全集群性能。

从低到高走，每个层级都在用前面章节的知识。

## 21.2 新增一个模型：完整的代码路径

这是最经典的入门任务，对应第 9、12 章的知识：

```text
1. 写模型实现      python/sglang/srt/models/xxx.py
   - forward() 复用 layers/ 的 RadixAttention、Linear、RotaryEmbedding
   - 依赖 hf_config 判断架构，不要写死
2. 注册 config     python/sglang/srt/configs/ 或 model_config.py
   - 让 --model-path 能识别新架构
3. 验证            --load-format dummy 先跑通链路（假权重）
4. 加测试          test/srt/ 下补 smoke test（用 mock model）
5. 多模态模型？    多一步：srt/multimodal/processors/ 加 processor
```

质量红线：

- 前向结果必须和 HuggingFace 参考实现一致（有专门的对比测试）；
- 复用 `layers/` 而不是复制粘贴别的模型的代码；
- 显存/耗时不能明显劣于同类模型（贡献指南要求附 benchmark）。

## 21.3 新增一个 kernel 或 attention backend

第 9 章讲过 attention 是插件化的。新增 backend 的路径：

```text
1. 继承 AttentionBackend（layers/attention/base_attn_backend.py）
2. 实现 init_forward_metadata / 前向内核调用
3. 在 attention_registry.py 注册
4. --attention-backend xxx 显式指定测试
5. 覆盖正确性（kv_canary）+ 性能对比
```

Triton kernel 的开发流程见 `docs_new/docs/developer_guide/development_jit_kernel_guide.mdx`，仓库里 `python/sglang/kernels/` 有大量现成算子可以模仿。

## 21.4 测试体系：改动质量的守护

```text
test/
├── srt/          # 运行时测试（按模块：radix、scheduler、sampler...）
├── registered/   # 注册式测试
├── manual/       # 大模型手动测试
└── run_suite.py  # 套件入口
```

提交前的最低标准：

1. `pre-commit run --all-files`（ruff/black/isort/mypy/codespell，配置在 `.pre-commit-config.yaml`）；
2. 改动的模块相关 pytest 通过；
3. 涉及缓存/注意力的改动跑 kv_canary；
4. 涉及行为变化的改动附 benchmark 数据。

仓库有大量 CI 工作流（`.github/workflows/`）：`pr-test.yml`、`lint.yml`、各硬件的 `pr-test-*`。PR 标题/描述会被机器人读取，规范很重要。

## 21.5 读代码的顺序建议（重新出发版）

如果你认真读完了本册，建议按这个顺序做"代码考古"：

```text
1. 通读 managers/scheduler.py 的 __init__（看一个 Scheduler 世界需要哪些零件）
2. 通读 mem_cache/radix_cache.py 全部（前缀树的每个方法都读一遍）
3. 通读 model_executor/forward_batch_info.py（ForwardBatch 每个字段追到使用处）
4. 选一个模型文件（models/llama.py），把 forward 追到 layers/ 的实现
5. 选一个高级特性（投机/PD/LoRA），把它的 mixin 追到调度器里的挂载点
```

做完这五步，你已经能看懂 80% 的日常 PR 在改什么。

## 21.6 贡献流程速查

1. 大改动先去 GitHub Issues / Slack 讨论（调度和缓存改动尤其要先说方案）；
2. 小改动直接 PR，描述写清楚：动机、改动点、测试、性能影响；
3. 保持 PR 小且聚焦，方便 review；
4. 参考 `docs_new/CONTRIBUTING.md` 与仓库根 `CONTRIBUTING.md`；
5. 社区活跃：官方 docs、roadmap.sglang.io、每周 dev meeting。

## 21.7 学习路线的终点

本册的结构本身就是一条学习路线：

```text
第一部分：建立心智模型（餐厅图）→ 会用（启动/调用/API）
第二部分：读懂实现（调度器/缓存/执行/并行/采样/多模态的代码）
第三部分：理解权衡（PD/投机/LoRA/Rust/路由/调优/排障/RL）
再之后：动手改（先模型/测试，再调度/内核）
```

当你能把第 18 章的三个案例讲给同事听、能对着 `scheduler.py` 指出"这里改了会影响什么"时，你就是一名合格的 AI Infra 推理工程师了。
