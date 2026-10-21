# KV 控制与 HiCache 验证记录

本记录覆盖本地 SGLang 实现的控制入口、副本优先级和持久 host 层搬运。默认关闭控制；开启后执行显式命令。本阶段支持单个 TP/PP/CP/DP owner，HiCache 搬运限于 FULL 状态、非 overlap 调度和本地持久 host 池。

## CPU 验证

各文件在独立进程运行，使用 `CUDA_VISIBLE_DEVICES=999` 隐藏 GPU，并显式选择 Python TreeCore。缓存测试使用真实的 tensor 池、allocator、radix tree 和 HybridCacheController。搬运测试只控制 L2 DMA 及其完成通知，实际复制 K/V tensor。

| 测试文件 | 通过项 |
| --- | ---: |
| `managers/test_kv_control.py` | 78 |
| `mem_cache/test_kv_control_cache.py` | 21 |
| `mem_cache/test_kv_control_service.py` | 12 |
| `mem_cache/test_kv_control_transfers.py` | 14 |
| `mem_cache/test_unified_hicache_transfer_lifecycle.py` | 8 |
| `managers/test_prefill_adder.py` | 53 |
| `managers/test_schedule_policy.py` | 11 |
| `spec/test_weight_update_runners.py` | 1 |
| `managers/test_scheduler_weight_version_tracking.py` | 20 |

共 218 项。前 217 项的逐文件输出保存在 `/tmp/sglang-kv-host-cpu-proof.txt`；增加 flush 后资源批准回调的回归测试后，搬运与服务两个文件合计 26 项通过。

覆盖行为包括：共享前缀只计费一次、保留运行中请求的 decode 增长估计、整个可复用前缀就绪前持续持有引用、DMA 期间 radix 分裂、单个与全部等待者取消、部分 MOVE、部分提交失败、失败结果的 ACK/重放、独立重试，以及分配前的范围和结果预算检查。超过目标池总容量的操作返回终态 `deferred`，避免留下无法完成的活跃记录。临时容量不足采用可取消的 `queued` 状态。

缓存 flush 会更换 epoch，同时保留调度器的资源批准回调。权重更新的受控流程已另行实现，见文末记录；低层未启用实际 key 命名空间的调用模式仍拒绝参数修改。

## 单卡验证

使用 GPU 1：NVIDIA A100-SXM4-80GB。目标虚拟环境安装了 PyTorch 2.14.1+cu130、torchvision 0.29.1、sglang-kernel 0.4.10；其他项目的环境和已有进程保持原状。

真实 CUDA 缓存测试通过优先级淘汰、KV 字节保留、请求引用保护、压力下回收、副本重建与重放，以及 allocator 余额检查。独立 HiCache 测试使用 direct I/O，验证 GPU→host MOVE 和 host→GPU COPY。

恢复测试会先将已经释放的 GPU 缓冲区填成错误值，再检查从 host 恢复的非零 K/V 数据。MOVE 后 64 个 token 槽全部可用；COPY 后保留 4 个有效缓存 token，可用槽为 60。结束时请求与 host 引用计数归零，搬运队列和控制任务均已退出。该微型缓存 fixture 的峰值 tensor 分配为 9,728 字节；它不代表服务内存占用或吞吐结果。

```bash
PYTHONPATH=python CUDA_VISIBLE_DEVICES=1 .venv/bin/python \
  test/manual/entrypoints/programmable_kv_gpu_smoke.py

PYTHONPATH=python CUDA_VISIBLE_DEVICES=1 .venv/bin/python \
  test/manual/entrypoints/programmable_kv_hicache_smoke.py
```

`/tmp/sglang-kv-gpu-hicache-poison-smoke.log` 保存了恢复检查结果。微型 fixture 的 16 字节行宽会触发可选 JIT kernel 的不支持提示；实际运行的是显式选择的 direct 后端。

## 服务级 smoke test

本地 OPT-125m 配置配合 `--load-format dummy` 完成了真实 GPU forward、HTTP、typed IPC 与缓存执行。测试请求先通过附带指令设置优先级，再独立 MOVE 整个已提交前缀，检查目标 GPU 副本消失且 host 副本可用，随后 COPY 回 GPU。重复生成命中 3 个缓存 token，输出 token 与原请求一致。恢复副本的优先级为零；flush 更换 epoch，旧操作引用查询返回 HTTP 409。

```bash
PYTHONPATH=python CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 \
  .venv/bin/python test/manual/entrypoints/programmable_kv_server_smoke.py \
  --model-path LOCAL_OPT_CONFIG --load-format dummy --hicache
```

脚本为每次运行生成独立的 owner 身份和命名空间，保存 `server.log`、`result.json`，只清理本次启动的进程。已通过的结果位于 `/tmp/sglang-kv-opt-hicache-final-smoke/`、`/tmp/sglang-kv-opt-hicache-unique-smoke/` 和 `/tmp/sglang-kv-opt-hicache-rebind-smoke/`。末次运行还检查了 flush 后仍由调度器批准 GPU 预取容量。随机权重的结果用于接口与执行路径验证，不能据此评价模型质量。

tiny Llama 的本地 checkpoint 路径仍未通过启动验证：环境中的 CUTLASS DSL 4.8.0 分发未提供所需的 `cutlass.cute` 模块，现有 DualGemm 在缓存初始化前报错。初始 PyTorch/kernel ABI 和 torchvision 不匹配的失败日志也已保留。未绕过版本检查或替换模型 kernel 来生成通过结果。

## 后续实现范围

真实 L3/linker 搬运、混合状态与多 rank 发布、可组合策略与请求 admission 协作仍需开发。当前 capabilities 会明确报告已支持的方向、粒度和限制。此处没有 SGLang 吞吐提升或训练权重输出质量结论。

## 参数更新与版本发布

参数修改前会隔离旧 epoch 和实际 lookup namespace；多步更新期间，新推理直接返回 503，KV 提交返回 busy。参数工作同步完成后才执行所需的 cache flush，并发布版本对应的新命名空间。`flush_cache=False` 时保留的旧物理副本也无法命中新版本查找。

23 项新增 CPU 测试修改实际 tensor，覆盖 disk/IPC、begin/tensor/distributed/end、Python Engine 真实序列化及 typed IPC、版本声明、回滚与部分失败、同步失败、最终 flush 重试、错误线程和内存所有权。相关 CPU 检查合计 143 项通过，2 项跳过，输出在 `/tmp/sglang-kv-weight-cpu-proof.log`。

GPU 1 上的 OPT smoke test 实际修改了 attention LayerNorm bias 参数：修改前后 checksum 不同，namespace 和 epoch 更新；同一输入在新版本首次生成命中 0 个缓存 token，随后重复生成命中 3 个，输出 token 一致。修改的是随机权重 fixture 中的有限偏置参数；这不构成训练质量或完整训练更新结论。结果保存在 `/tmp/sglang-kv-opt-weights-smoke/result.json`。

现有 `test_weight_checker.py` 按其单卡配置运行得到 63 项通过、1 项失败；失败的 FP8 路径缺少 `deep_gemm`。在隐藏 GPU 环境中误运行该套件的失败记录保留在 `/tmp/sglang-kv-weight-checker-cpu-env-failed.log`，单卡输出在 `/tmp/sglang-kv-weight-checker-gpu-proof.log`。未修改既有测试以掩盖依赖问题。
