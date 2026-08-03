# 第 7 章 调度器 Scheduler：连续批处理与事件循环

## 7.1 为什么调度器是灵魂

GPU 做一次前向，代价与 batch 大小几乎无关（小 batch 时），因此**能塞多少请求进一个 batch，吞吐就有多大**。但请求到达是随机的：有刚来的长 prompt（prefill）、有进行到一半的短尾巴（decode）、有要中止的、有带不同采样约束的。调度器就是回答三件事的引擎：

1. **现在跑谁**（等待队列 → running batch）；
2. **跑成什么样**（prefill batch 还是 decode batch，还是混合）；
3. **显存不够怎么办**（谁被抢占、谁被淘汰）。

SGLang 的调度器位于 `python/sglang/srt/managers/scheduler.py`，核心类是 `Scheduler`（第 359 行），注释写着："A scheduler that manages a tensor parallel GPU worker."

## 7.2 Scheduler 的初始化：一个进程就是一套世界

`Scheduler.__init__` 要做的事极多，重要的一批：

- `init_ipc_channels`：建立与 TokenizerManager/Detokenizer 的 ZMQ 通道；
- `init_model_config` / `init_tp_model_worker`：解析模型配置、初始化 GPU worker（ModelRunner）；
- `init_memory_pools`：创建 req_to_token 池与 KV 池（`mem_cache/memory_pool.py`）；
- `init_all_attention_backends` / `init_all_cuda_graphs`：准备注意力内核与 CUDA graph 缓存；
- `init_chunked_prefill`：决定长 prefill 是否分块；
- `init_schedule_policy`：选择调度策略；
- `init_disaggregation`：PD 分离模式下注册传输队列；
- `init_overlap`：CPU 调度与 GPU 执行重叠模式。

读代码时建议从 `run_event_loop` 倒着往上读：先看"循环里做什么"，再回头理解每个初始化步骤为什么存在。

## 7.3 事件循环：两种形态

### event_loop_normal（第 1632 行）

最简单的形态，循环体只有四步：

```python
while True:
    recv_reqs = self.request_receiver.recv_requests()   # 从 ZMQ 收新请求
    self.process_input_requests(recv_reqs)              # 解析并放入 waiting queue
    plan = self.get_next_batch_to_run(running_batch, last_batch)
    self.running_batch = plan.running_batch
    batch = plan.batch_to_run
    if batch:
        result = self.run_batch(batch)                  # 同步执行 GPU 前向
        self.process_batch_result(batch, result)
    else:
        self.on_idle()                                  # 空闲自检
    self.last_batch = batch
```

这个循环是**单线程阻塞式**的：调度决策和 GPU 前向交替进行。简单但浪费——GPU 在跑的时候 CPU 在等。

### event_loop_overlap（第 1666 行）

重叠模式把"上一批的结果处理"与"下一批的 GPU 执行"并行：

```python
while True:
    # 处理上一批结果
    tmp_batch, tmp_result = self.result_queue.popleft()
    self.process_batch_result(tmp_batch, tmp_result)

    # 收请求、组批、启动这一批（异步）
    ...
    self.run_batch_async(batch)   # GPU 前向交给 stream，不阻塞
    self.result_queue.append((batch, result_future))
```

实现细节依赖 CUDA stream 的并发：调度器在 `schedule_stream` 上做 CPU 工作，模型在 `forward_stream` 上做 GPU 工作，中间用 WAR barrier（写后读屏障）保证共享内存池不冲突。这个机制是"零开销调度器"卖点的来源，也是 `scheduler.py` 里最微妙的部分之一。

## 7.4 队列与请求对象

- **waiting_queue**：`List[Req]`，新请求先在这里排队；
- **running_batch**：`ScheduleBatch`（`managers/schedule_batch.py` 第 1923 行），当前正在执行的请求集合；
- **Req**（`schedule_batch.py` 第 771 行）：调度器内部的请求表示，字段包括 `rid`、`origin_input_ids`、`output_ids`、`sampling_params`、`kv_committed_len`、`lora_id`、`priority`、`session` 等。

`Req` 与 HTTP 层的 `GenerateReqInput` 是两套对象：前者是"调度器的视图"，后者是"网络的视图"。

## 7.5 组批：PrefillAdder 与 DecodeAdder

调度策略定义在 `managers/schedule_policy.py`：

- `SchedulePolicy.calc_priority()` 给 waiting queue 排序。策略包括：
  - **LPM**（Longest Prefix Match）：优先跑能命中前缀缓存最多的请求（cache-aware）；
  - **FCFS**（先来先服务）；
  - **LOF**（Longest Output First）、**RANDOM**、**DFS_WEIGHT** 等。
  - 一个工程细节：LPM 在队列超过 128 时会自动降级为 FCFS，避免排序开销过大（`_determine_active_policy`）。

- `PrefillAdder`（`schedule_policy.py` 第 444 行）把 waiting queue 的请求装进 prefill batch，受以下约束：
  - `max_running_requests`：batch 上限；
  - `max_prefill_tokens`：单批 prefill 总 token 上限；
  - `chunked_prefill_size`：单个超长请求被切成多少 token 一块；
  - 显存预算：由 `token_to_kv_pool_allocator` 决定还能分配多少页。

```python
adder = PrefillAdder(
    self.page_size,
    self.tree_cache,
    self.token_to_kv_pool_allocator,
    running_batch,
    ...
)
for req in self.waiting_queue:
    # 判断能否加入、要不要抢占、是否截断为 chunk
    ...
```

`scheduler.py` 的 `get_new_batch_prefill`（第 3014 行）与 decode 分支共同构成 `get_next_batch_to_run`（第 2872 行）的核心。

## 7.6 Chunked Prefill：长请求不被饿死

一个 100k token 的请求如果整段 prefill，会独占 GPU 很久，导致后续请求 TTFT 爆炸。SGLang 的 chunked prefill 把它切成 `chunked_prefill_size` 的小块（`--chunked-prefill-size` 可指定，`-1` 表示禁用；默认值按显存自动确定，T4 为 2k、H100 为 8k、B200 为 16k，见 `server_args.py` 的 `_validate_cuda_graph_config` 中按 `gpu_mem` 分档的逻辑），块之间可以插入 decode 或别的 prefill。实现上，`Req` 有 `chunked_prefill_size` 与 `chunked_req` 状态，调度器在 `_get_new_batch_prefill_raw` 里用 `add_chunked_req` 把未完成块重新放回调度。

## 7.7 抢占与显存不足

当显存不足（allocator 返回可分配页数为 0）时：

1. **CPU 抢占**：把某些请求从 running batch 里移出，释放其 KV 页（`mem_cache` 的 free 操作）；
2. 被抢占请求的进度保留在前缀缓存中，等空间恢复后可从断点继续（因为 KV 是按 token 提交的，天然支持续跑）；
3. `min_free_slots_delayer`、`prefill_delayer` 等机制则反过来：宁可让 prefill 等一等，也要保住 decode 的延迟。

这种"运行中请求可被摘除并续跑"的能力，依赖第 8 章的 Radix Cache 设计。

## 7.8 结果处理：process_batch_result

`process_batch_result`（第 3756 行）做四类事：

- **更新 Req 状态**：把新生成的 token 追加到 `output_ids`，判断是否 EOS/达 max_tokens；
- **KV 落缓存**：`tree_cache.cache_finished_req`（或 `insert`）把完成的请求写进 radix 树；
- **产出响应**：组装 `GenerationBatchResult` 发给 detokenizer（走 `output_sender`/ZMQ）；
- **维护指标**：更新 pool 用量、token 统计等（`metrics_reporter`、`new_token_ratio_tracker`）。

## 7.9 与 DP Controller 的关系

`managers/data_parallel_controller.py` 运行一个调度器之上的"调度器"：当 `--dp-size > 1` 时，请求先在 DP Controller 上按负载/缓存感知策略分发给某个 Scheduler（数据并行副本），每个副本各有自己的 `running_batch`。这是第 17 章路由话题的前奏。

## 7.10 本章小结

- Scheduler 是独立进程 + 无限事件循环，核心是"收请求 → 组批 → 执行 → 处理结果"。
- 重叠模式下 CPU 调度与 GPU 前向并发，是 SGLang 吞吐优势的关键。
- 组批由 SchedulePolicy 排序 + PrefillAdder/DecodeAdder 装填，受显存、token 预算、请求上限约束。
- Chunked prefill 与抢占机制共同保证长请求和短请求公平共存。
- 下一章看调度器依赖的地基：KV Cache 与 RadixAttention。
