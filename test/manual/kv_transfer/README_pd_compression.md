# r4：固定块 Host pool 的 KV 压缩实验

本版将 r3.4 的连续区间分配器替换为 4 KiB 固定块存储。HiCache 继续管理前缀、备份、恢复、淘汰和 ACK；P/D 继续管理传输范围及远端完成。压缩层只管理数据表示和执行资源。

交付仅表示源码及本地测试完成。新镜像的 GPU、RDMA、真实生成和在线故障验收必须重新运行，不沿用旧镜像结果。

固定基线为 `8ae4a39b50ffcdb44175cd2fc5594032740ba888`，实验目录为 `/private/tmp/sglang-pd-hicache-compression`。修改包含未提交文件，仅切换分支不能取得完整实现，应使用交付 overlay 或补丁。保留 nvCOMP `5.3.0.16` 及既有依赖，v2 网络协议和 LZ4 格式未改变。

## 支持范围

Qwen3-8B、BF16、FlashInfer、1P1D、TP/PP/DP/CP/DCP=1、page=1、chunk=1024。共享 GPU workspace 512 MiB，Prefill L2 8 GB，Decode L2 关闭，VERIFY=1。实验压缩存储仅支持 `write_through`、Python UnifiedRadixCache；关闭 overlap、CUDA graph、HiSparse、推测解码、LoRA、session、L3。

一份压缩对象仍是一个 token 的全部 K/V 层。4 KiB 只是 Host 存储块，不改变 KV 页、attention 布局或压缩算法。普通模式允许 raw 回退；强制模式允许 LZ4 略大于原始字节。它不是 KVServe 算法或控制器的复现，不宣称带宽或吞吐收益。

## 数据和控制路径

```text
Prefill GPU KV ──共享编码执行器─── P/D wire buffer ── RDMA ── Decode 恢复与校验
                         │               ▲
                         ▼               │ 只读租约，复制后释放
                HiCache 固定块 Host pool ┘
                         │
                         └── 恢复与校验 ── HiCache ticket ── 发布 GPU 页映射
```

GPU 上仍保留正常计算使用的 KV。临时编码结果是连续 GPU 字节；Host 对象可以跨多个不连续块。压缩后长度为 148224 字节时占 37 块；197632 字节输出上界预留 49 块，进入写入阶段后归还 12 块。块无需连续，因此归还的小块可以立刻组合用于新对象。

`alloc(n)` 在 HiCache 提交备份前原子预留逻辑句柄和全部上界块。`available_size()` 始终返回可预留的逻辑页数；HiCache 按页数缺口选择淘汰节点，不接收物理块数量。读写租约未排空时，逻辑淘汰不会释放物理块。

```text
FREE → RESERVED → WRITING → READY → RETIRED → FREE
```

`prepare_write()` 每批最多 64 页，验证长度、冻结映射、归还多余块并取得写入租约。D2H 完成后按块批量散写，再发布对象；所有必要对象完成后 HiCache 才处理成功 ACK。取消只撤销逻辑所有权，不能移动或回收仍在访问的块。

`acquire()` 仅取得描述符和读租约。恢复或发送线程通过 `materialize_pages()` 聚集数据，复制到 GPU 后确认读取完成，才释放 Host 暂存区。P/D 的 wire buffer 独立保留到 RDMA 完成。

两条固定 pinned Host 通道分别用于写入及读取，读取由恢复和 P/D 共用；每条最多 64 页。需要两条时先读后写。编码结果和 GPU 资源先就绪，再取得暂存区。scheduler 不等待暂存区、Future 或 GPU 事件，后台待推进时沿用 1 ms 让出机制。

## 预算与诊断

8 GB 包含 arena、对象记录、SHA256、哈希索引、逻辑空闲栈、三个 int32 块数组、两条 pinned 通道，以及有界 CPU 散写 scratch 和索引预算。Python 对象、临时 NumPy 索引、日志、框架分配器和 GPU 内存不属于这个 Host pool 预算；它不是进程 RSS 上限。

`KV_COMPRESSION_STATS.l2` 记录 `free_blocks / reserved_blocks / live_blocks / retired_blocks`、实际编码字节、块内余量、元数据、暂存区、活跃读写者和隔离对象。分配、prepare_write、free 各自记录等待与持锁；聚集、散写、通道等待单独记录。`allocation_seconds` 包含等锁和持锁，`backup_d2h_seconds` 包含提交及同步，GPU event 时间是 CUDA stream 上两个 event 的间隔，可能包含提交空隙；不能将这些嵌套值相加。

开启 `SGLANG_KV_COMPRESSION_TRACE_STORE=1` 后，`KV_COMPRESSION_STORE` 记录节点提交、淘汰选择、物理释放和租约最后释放。使用 handle/generation/page_ref 关联 P/D 既有 rid/room 记录；共享缓存操作不虚构请求 ID。诊断关闭时不输出完整身份数组。

## 本地检查

在仓库根目录运行，独立 CPU 环境可使用 `SGLANG_COMPRESSION_STANDALONE_TEST=1`。此开关只跳过 package 初始化；存储和执行器是真实 CPU 实现，部分 serving 测试执行真实方法体并使用受控协作者，不等于完整服务。

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 SGLANG_COMPRESSION_STANDALONE_TEST=1 PYTHONPATH=python \
python -m pytest -q \
  test/registered/unit/mem_cache/test_compressed_hicache.py \
  test/registered/unit/mem_cache/test_compressed_block_pool.py \
  test/registered/unit/disaggregation/test_pd_compression_protocol.py \
  test/registered/unit/disaggregation/test_pd_compression_validation.py
```

旧的小型控制流夹具为固定块最小单元和暂存区增加了测试内预算。8 GB 容量测试直接实例化生产 pool，不使用这些夹具；长度来源必须明确标为模拟或真实采集。

```bash
python test/manual/kv_transfer/replay_block_capacity.py \
  --workload workload.json --order request-order.jsonl \
  --node-pages 1024 8192 --output capacity.json
```

回放保留原 token IDs 和请求顺序，模拟节点 LRU、部分命中拆分及原对象身份；不执行 GPU、复制或实际树调度。默认长度 147900–148220 字节是模拟数据。`--lengths` 可提供以逐前缀 SHA256 为键的真实长度 JSON，缺失样本会失败，不能自动补模拟值。目标存储身份为 8192 页，线上实际采用门槛仍为 8191 页。

## 新镜像与预检

将完整 overlay 覆盖到固定基线后构建镜像，两端同步更新。不使用旧源码热修复挂载。镜像内先执行：

```bash
python test/manual/kv_transfer/check_pd_compression_image.py --manifest /path/to/manifest.json
python -m pytest -q test/manual/kv_transfer/test_pd_compression_gpu.py --junitxml=/tmp/gpu.xml
```

交付 `render_phase.py` 生成的预检会将 JUnit 用例名称与 manifest 的实际收集清单逐项核对，拒绝缺失、额外、失败或跳过。GPU 用例覆盖真实 nvCOMP、非连续块读写、校验、恢复和写入失败后的排空；不是 RDMA 端到端证明。

启动器保留 `MODEL_PATH`、`IB_DEVICE`、`ROLE` 等现有配置与 GPU 绑定。GPU workspace、wire buffer、Decode staging 和验证内存分别计量，512 MiB 不等于压缩子系统总显存上限。

| 组别 | COMPRESSION_MODE | Prefill ENABLE_HICACHE | HICACHE_COMPRESSION | FORCE |
|---|---|---:|---|---:|
| native | off | 1 | off | 0 |
| passthrough | passthrough | 1 | passthrough | 0 |
| force-l2 | lz4 | 1 | lz4 | 1 |
| off | off | 0 | off | 0 |
| force | lz4 | 0 | off | 1 |
| lz4 | lz4 | 1 | lz4 | 0 |

Decode 的 ENABLE_HICACHE=0、HICACHE_COMPRESSION=off，P/D 模式、FORCE、VERIFY 与 Prefill 一致。完整启动命令由 `DRY_RUN=1 bash test/manual/kv_transfer/launch_pd_compression.sh` 输出。

## 在线验收

顺序固定为 `native → passthrough → force-l2 → off → force → lz4 → 故障专项`。每组使用新镜像、新 Pod 建立该组状态；native 必须重新运行。组内不得靠清缓存、重启或跳过备份获得排空。任一门槛失败立即保留现场，停止后续组。

用同一份冻结 workload。以下命令中的日志命令须返回当前组当前 Pod 的日志，不能混入旧 Pod；Router 使用真实请求入口：

```bash
python test/manual/kv_transfer/validate_pd_compression.py run \
  --workload workload.json --output results/force-l2 --phase force-l2 \
  --router "$ROUTER" \
  --prefill-log-command "$PREFILL_LOG_COMMAND" \
  --decode-log-command "$DECODE_LOG_COMMAND"
python test/manual/kv_transfer/validate_pd_compression.py audit results/force-l2
python test/manual/kv_transfer/validate_pd_compression.py compare results/native results/force-l2
```

缓存组各完整 38 请求，off/force 各 14 请求。passthrough/lz4/force-l2 与新 native 比较；force 与新 off 比较。同路径完整 token 相同；三轮恢复并采用 `device=0, host=8191`；四并发正常；每轮 180 秒内自然排空且连续三次新记录归零；无请求阶段健康超时或 Router 503。缺失/null 的缓存来源明确作为 Host 未命中，保留原响应，不降低 8191 门槛。

force-l2 必须同时证明真实 LZ4 发送、接收校验、L2 保存、稳定前缀旧对象复用、GPU 淘汰后恢复。不能用输出相同或总编码计数代替这些证据。保存启动配置、image digest、两端哈希/GPU JUnit、Router 响应、完整 token、工作负载指纹、诊断窗口和健康事件。

## 故障验证

正常启动默认不注入故障。内部测试注入只接受一个明确操作，禁止通配符，例如：

```text
SGLANG_KV_COMPRESSION_TEST_FAULT={"operation":"write:4294967296","point":"after_scatter","kind":"error"}
```

`write:<首 handle>` 和 `restore:<首 handle>` 可由身份日志确定，handle 包含 generation。仅精确匹配的操作触发一次；重放时必须核对实际命中，否则该测试不算执行。支持 after_scatter、before_publish、before_restore、after_restore_copy；kind 为 error、checksum 或 undrained。普通六组必须把此变量清空。它是受控异常注入，不模拟真实 CUDA/RDMA 硬件故障。

| 场景 | 方法与必须保留的证据 |
|---|---|
| 取消、取消后晚到完成 | 在独立故障轮取消明确 rid；检查 ticket 未发布、引用排空、安全回收及下一请求。对应本地测试覆盖晚到写入和 ticket。 |
| 部分散写、发布失败 | after_scatter/before_publish 精确注入；无成功 ACK、无无效对象命中；安全回收后下一请求正常。 |
| 摘要或恢复失败 | before_restore/after_restore_copy 精确注入；明确请求失败、不得发布目标页；真实目标字节损坏由 GPU/控制流校验测试覆盖。 |
| 描述符损坏、缺失摘要 | 运行既有协议拒绝测试并在隔离的 P/D 故障测试中对指定 room 重放损坏描述符，保留双方错误和目标未放行证据。正常 Router 请求不能制造这一故障，本交付未声称提供远程注入 API。 |
| 无法确认排空 | undrained 精确注入后，buffer/lease/ticket 保持隔离，idle 不得归零，不要求同 worker 下一请求成功。实际 RDMA 断连仍需独立跨节点测试。 |

本地异常注入与 GPU 预检不能替代在线故障矩阵。没有执行的项目明确记录为待验收。
