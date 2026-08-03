# 第 7 章 调度器代码走读：下一批跑什么、为什么

> 前置要求：先读第 3 章的心智模型，或至少知道 waiting queue / running batch / prefill / decode 是什么。
> 本章代码全部来自 `python/sglang/srt/managers/scheduler.py` 与 `schedule_policy.py`。

## 7.1 先找到主角

`Scheduler` 类在 `scheduler.py` 第 359 行，类的文档注释只有一句：*"A scheduler that manages a tensor parallel GPU worker."*——它管理着一个 TP 组的 GPU worker。

调度器是**独立进程**（由 `entrypoints/engine.py` 用 `run_scheduler_process` 拉起），与 HTTP 层通过 ZMQ 通信。它内部持有：

- `waiting_queue`：等待中的请求列表；
- `running_batch`：正在跑的 `ScheduleBatch`；
- `tree_cache`：前缀缓存（第 8 章的主角）；
- `token_to_kv_pool_allocator`：KV 显存分配器。

## 7.2 事件循环：调度器的"心跳"

最基础的模式是 `event_loop_normal`（第 1632 行），完整循环只有四步：

```python
def event_loop_normal(self):
    """A normal scheduler loop."""
    while True:
        if self.gracefully_exit:
            break

        # Receive requests
        recv_reqs = self.request_receiver.recv_requests()   # ① 从 ZMQ 收请求
        self.process_input_requests(recv_reqs)              # ② 解析并放入 waiting_queue
        if self._engine_paused:
            continue

        # Get the next batch to run
        plan = self.get_next_batch_to_run(
            running_batch=self.running_batch, last_batch=self.last_batch
        )                                                   # ③ 决策：下一批跑什么
        self.running_batch = plan.running_batch
        batch = plan.batch_to_run

        # Launch the current batch
        if batch:
            result = self.run_batch(batch)                  # ④ GPU 前向（同步阻塞）
            self.process_batch_result(batch, result)        # ⑤ 处理结果
        else:
            self.on_idle()                                  # 空闲自检

        self.last_batch = batch
```

读这段代码要注意一个反直觉的点：**循环里的"每一轮"不一定对应一个请求，而是对应一次 GPU 前向**。`batch` 可能是 prefill 批、decode 批，也可能为空（什么都不做）。

⑤ 处理结果时会发生什么？看 `process_batch_result`（第 3756 行）的职责：

1. 把新 token 追加到每个请求的 `output_ids`，判断是否 EOS / 达到 `max_new_tokens`；
2. 完成的请求把 KV 写回前缀缓存（`tree_cache.cache_finished_req`）；
3. 组一个"结果消息"发给 Detokenizer 进程；
4. 更新指标（pool 用量、token 统计）。

## 7.3 快照：事件循环的"状态机"本质

把 ③④⑤ 连起来看，调度器其实在反复执行：

```text
读请求 → 更新队列状态 → 决策下一批 → 执行 → 根据结果更新状态 → 循环
```

`get_next_batch_to_run`（第 2872 行）开头还会做几件容易被忽略的事：

- `_abort_on_waiting_timeout` / `_abort_on_running_timeout`：请求等待/运行超时直接中止；
- chunked 请求的特殊处理：未完成的 chunk 请求要"插队"继续跑；
- PD 分离模式下，prefill 批可能要先发缓存前缀的 KV。

这说明调度循环不只是"组批"，还是一个**生命周期管理器**：请求的中止、超时、续跑都发生在这里。

## 7.4 决策是怎么做出来的：组批的三层检查

`get_new_batch_prefill`（第 3014 行）→ `_get_new_batch_prefill_raw` → `PrefillAdder`（`schedule_policy.py` 第 444 行）。

`PrefillAdder` 构造时先算好"这轮还有多少预算"：

```python
class PrefillAdder:
    def __init__(self, page_size, tree_cache, token_to_kv_pool_allocator,
                 running_batch, new_token_ratio, rem_input_tokens, rem_chunk_tokens,
                 num_mixed_decode_tokens=0, ...):
        self.rem_input_tokens = rem_input_tokens - num_mixed_decode_tokens
        self.rem_chunk_tokens = rem_chunk_tokens
        ...
```

然后主循环遍历 `waiting_queue`，对每个请求做三类检查（见 `_get_new_batch_prefill_raw` 中的循环）：

```text
对每个请求：
  ① 显存检查：allocator 还能分配足够的 KV 页吗？
     → 不够：要么抢占 running batch 里优先级低的请求，要么停手
  ② token 预算检查：加上这个请求，prefill 总 token 数超 max_prefill_tokens 吗？
     → 超了：把它切成 chunk（chunked prefill），下一轮继续
  ③ 请求数检查：batch 人数达到 max_running_requests 吗？
     → 到了：标记 batch full，后面的等下轮
```

顺序很重要：**显存检查在最前**，因为显存是硬约束，token 数是软约束。

`calc_priority`（`schedule_policy.py` 第 186 行）决定遍历顺序。策略枚举：

```python
class SchedulePolicy:
    Policy = Union[CacheAwarePolicy, CacheAgnosticPolicy]
```

- **LPM**（Longest Prefix Match，缓存感知）：先算每个请求能命中多少缓存前缀（`_compute_prefix_matches`），按命中量从大到小排。命中越多，这轮要算的越少，GPU 越省。
- **FCFS**：先来先服务，不看缓存。
- **LOF**：输出越长的越优先（避免短请求把长请求饿死，用于 decode 混合场景）。
- 工程细节：LPM 在等待队列超过 128 个请求时会自动退化成 FCFS（`_determine_active_policy`）——因为对 128+ 个请求逐个做前缀匹配的 CPU 开销，可能比省下的 GPU 时间还多。

## 7.5 Chunked prefill 的循环细节

`_get_new_batch_prefill_raw` 里有这样一段：

```python
if self.chunked_req is not None:
    self.chunked_req.init_next_round_input()
    self.chunked_req = adder.add_chunked_req(self.chunked_req)
```

含义：上一个 chunk 跑完后，调度器记得"这个请求还有一半没算"（`chunked_req` 非空），下一轮**先**把它接着算完，再处理新请求。`init_next_round_input` 把剩余段变成这一轮的输入。

这样长请求（比如 100k token）就不会独占 GPU：它每次只算一小块，中间可以穿插其他请求的 decode。

## 7.6 抢占：显存不够时的不变量

当 allocator 说"没页了"，调度器有两种选择：这轮不加新请求，或者**把 running batch 里的请求移出去**（CPU preemption）。

被抢占的请求不是从头再来——它的 KV 已经按 token 提交到前缀缓存里了，之后可以**从断点续跑**。这个能力依赖第 8 章的缓存设计，先记住不变量：

> **不变量：任何时刻，一个请求的 KV 都是"从开头到某个位置"的连续前缀。** 有了这个不变量，抢占 = 摘下来，续跑 = 接着算，永远不需要重算。

## 7.7 重叠模式：让 CPU 调度和 GPU 计算并行

`event_loop_normal` 是"算完再调度"，GPU 干活时 CPU 闲着。`event_loop_overlap`（第 1666 行）把两步叠起来：

```python
def event_loop_overlap(self):
    ...
    while True:
        def pop_and_process():
            # 处理上一批的结果（CPU 活）
            tmp_batch, tmp_result = self.result_queue.popleft()
            self.process_batch_result(tmp_batch, tmp_result)
        ...
        # 把这一批的 GPU 前向丢到另一个 CUDA stream 上异步执行
        # 不阻塞，立刻回到循环继续做下一批的调度准备
```

实现细节依赖 CUDA stream：调度器在 `schedule_stream` 上做 CPU 准备，模型在 `forward_stream` 上做 GPU 前向，中间用"写后读屏障"（WAR barrier）保证两者不踩到同一块显存缓冲。`run_batch` 里能看到：

```python
with self.forward_stream_ctx:
    self.forward_stream.wait_stream(self.schedule_stream)
    ...
```

这就是 SGLang "零开销调度器"的来源：GPU 算这一批的时候，CPU 已经把下一批准备好了。

## 7.8 自己动手的实验

1. **观察两种策略的差别**：同一批 50 个请求（其中 10 个共享前缀），分别用 `--schedule-policy lpm` 和 `fcfs` 跑，对比总耗时与 TTFT 分布。
2. **看 chunked prefill**：发一个 10k token 的请求，`--log-level debug`，观察日志里它是被切成了多少块跑的。
3. **看抢占**：小显存（`--mem-fraction-static 0.3`）+ 高并发，观察日志里的 preempt 相关输出。
4. **开 overlap 对比**：`--enable-overlap-schedule` 开关前后，对比同负载下的吞吐。

## 7.9 本章小结

- 调度器 = 无限循环：收请求 → 组批 → 执行 → 处理结果。
- 组批是三层预算检查（显存 / token / 请求数），顺序体现硬软约束。
- LPM 策略让"缓存命中多的请求"优先，是 SGLang 吞吐的独门武器。
- Chunked prefill 与抢占依赖同一个不变量：请求 KV 永远是连续前缀。
- 重叠模式用 CUDA stream 让 CPU 调度和 GPU 计算并行。

> 下一章深入那个"被反复使用的缓存"：KV Cache 与 RadixAttention 的代码实现。
