# 为什么“攒 Batch”会让 TTFT 变大？（排队等待详解）

## 1. 你卡住的句子是什么意思

> “但为了攒 batch 会引入排队等待 → TTFT 增加”

这里的 **排队等待（queueing delay）** 指的是：**请求到了以后，不立刻上 GPU 跑 prefill，而是在调度器里等一会儿，等更多请求一起凑成一个更大的 batch 再跑**。

因此 **TTFT = 等待时间 + 真正开始 prefill 的计算时间 + 直到第一个 token 产生/回传的时间**。  
攒 batch 的“等待时间”变大，TTFT 自然就变大。

---

## 2. 用时间线直觉理解（最重要）

### 2.1 不攒 batch（低吞吐，低等待）

```
t0: 请求到达
t0~t0+ε: 几乎立刻被调度上 GPU
t0+Δprefill: prefill 结束/开始产生第一个 token
=> TTFT ≈ ε + Δprefill
```

优点：TTFT 小  
缺点：GPU 利用率可能很差（batch 太小，算子 launch 多，吞吐低）

---

### 2.2 攒 batch（高吞吐，但会多一段等待）

```
t0: 请求到达
t0~t0+W: 在 waiting_queue 里等（为了凑 batch / 等资源预算）
t0+W: 才开始上 GPU 跑 prefill
t0+W+Δprefill: 才产生第一个 token
=> TTFT ≈ W + Δprefill
```

这就是“攒 batch 会让 TTFT 增加”的核心：**多了一个 W**。

---

## 3. 为什么要攒 batch（为什么值得等）

攒 batch 的目标是 **提高吞吐 / 降低单位 token 成本**，常见收益来自：

- **更高 GPU 利用率**：prefill 通常更像大矩阵乘，batch 大更容易吃满 GPU
- **更少 kernel launch / 更少调度开销**：小 batch 会导致频繁 launch、开销占比高
- **更好 amortize（摊销）**：一次跑更多 token，总体更划算

所以系统经常在做一个 trade-off：

\[
\text{吞吐} \uparrow \quad \text{vs} \quad \text{TTFT} \downarrow
\]

---

## 4. 在 SGLang 里，“等待”具体发生在哪

### 4.1 等待发生在 Scheduler 的队列里（不是 Router）

Router 的职责是“选哪个 worker”；**是否等待、等多久、怎么凑 batch** 是 **worker 内 Scheduler** 的职责。

你可以对应到你之前的理解纠正文档：
- `yc_self_learn/md/22_SGLang完整请求流程详解_纠正版.md`
  - **Step 7/8**：Scheduler 做匹配/构建 batch（这里会产生等待）

---

### 4.2 等待最典型的“物理位置”

在 worker 内部，Scheduler 维护类似：
- `waiting_queue`：新来的请求先进来
- `get_new_batch_prefill()`：从 waiting_queue 里挑哪些 req 进下一轮 prefill batch

**等待产生的常见原因**（你可以把它当成 W 的组成部分）：
- **凑 batch**：还没凑到“足够划算”的 batch（例如 batch_tokens/req 数没到目标）
- **等资源**：KV cache/内存预算不够、需要等上一轮释放，或需要抢占/回收
- **策略约束**：为了稳定尾延迟/避免抖动，调度器会更保守（例如 conservativeness）

---

## 5. 你去旧文档哪里看（推荐）

如果你想回到之前更细的解释：
- **最推荐**：`yc_self_learn/md/12_调度器waiting_queue与get_new_batch详解.md`
  - 重点找 `waiting_queue`、`get_new_batch_prefill`、`PrefillAdder` 相关小节（这些就是“为什么会等/怎么挑”的核心）
- **其次**：`yc_self_learn/md/22_SGLang完整请求流程详解_纠正版.md`
  - 重点看 Step 7/8（调度器开始介入的地方）

---

## 6. 用一句话记住

**攒 batch = 用“多等一会儿（W）”换“GPU 跑得更满（吞吐更高/成本更低）”；TTFT 会把这段等待算进去，所以可能变大。**


