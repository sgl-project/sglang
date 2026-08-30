# 创新点 2 方案：Agent 容忍期内 KV Cache 保留（Score 驱逐）

> 状态:执行中(2026-08-30 开工)
> 分支:`paper/mxfp4-kv-sm86`

## 问题定义

Agent 会话以工具调用结束后,KV 前缀进入 radix cache,按 **LRU** 驱逐
(`radix_cache.py:92` 的 `__lt__` 只比 `last_access_time`)。但工具执行有一个
"容忍窗口"(0.1s~30s):窗口内若前缀被驱逐,工具结果返回时就要全量重算。
**方案 = 驱逐策略从 LRU 换成 Score(返回预测 × 重算成本 ÷ 压力成本)**。

## 代码修改点(v0.5.2,已核实)

| 改动 | 位置 | 内容 | 量 |
|---|---|---|---|
| A. 驱逐策略 | `radix_cache.py` TreeNode(:43)+`__lt__`(:92)+`evict`(:294) | TreeNode 加 `retain_score`/`tool_return_eta`/`ctx_len`;evict key 从 LRU 改 Score;分级软保护 | ~150 行 |
| B. 工具结束信号 | `serving_chat.py`(:593/:745)+`io_struct.py`+scheduler(:2347 记 rid→last_node) | TM 检测 `finish_reason=tool_calls` → 控制消息 → scheduler 打分 | ~100 行 |
| C. Score 计算 | scheduler 内新模块 | per-工具名 EMA 耗时表 + 公式 | ~150 行 |

**Score 公式**:`retain_score(n) = P_return(Δt|tool) · C_recompute(L_n) / C_pressure(pool)`
- P_return: 该工具耗时分布(lognormal CDF,EMA 均值方差)
- C_recompute ∝ 上下文长度 × prefill 速率
- C_pressure ∝ evictable_size / total

**v0 简化**:`evict_key = last_access_time − α·E[tool_time]`(把"刚以工具调用结束"
的会话视作最后访问时间是未来的工具返回时刻)。

## 对比实验

| 组 | 配置 | 问题 |
|---|---|---|
| ① No-radix | `--disable-radix-cache` | 最差下界 |
| ② Radix-LRU | 生产配置 | **要打赢的 baseline** |
| ③ Radix-Oracle | harness 真实延迟标签神谕 | 上界 |
| ④ Radix-Score (ours) | 完整策略 | 主结果 |

指标:续轮 TTFT(主)、prefill token 总量、驱逐后悔数、端到端时延、池占用轨迹。
压力扫描:N ∈ {16,32,64,128};工具延迟 4 类 lognormal(快 80%/慢 20%)。
交叉消融:BF16 vs MXFP4 × ②/④(连接创新点 1)。

## 数据集(三层)

1. **主实验:自建 trace-driven harness**(`agent_kv_bench/`)。现有 benchmark
   (BFCL/SWE-bench)的工具执行都是进程内瞬时,无真实延迟分布、无持续压力——
   评测不了本创新点,这个空白本身可写。会话剧本用 BFCL 风格工具集,上下文
   fabrication 精确控长(2k→26k),延迟按表采样,sleep 后发续轮。
2. Sanity:BFCL multi_turn_base 原版(不伤害标准分数)。
3. 加分项:SWE-bench-lite 轨迹回放。

延迟表(手工标定 + 敏感性分析 ×2/×0.5):
quick_fs 0.5 权重 ln(0.3)s / web_search 0.2 ln(2)s / run_tests 0.2 ln(10)s / code_edit 0.1 ln(4)s

## 预期收益

3090+4B:8k 前缀 prefill ≈2-4s,命中只算增量 ≈0.1-0.3s → 续轮 TTFT 改善
空间 10 倍级。保守预期(饱和压力 N=32-64,Score vs LRU):续轮 TTFT 中位数
-30~-60%,prefill tokens -20~-40%,后悔数 -50%+。

## 时间表

- D1-D2: harness → D3: v0 Score + 基线 → D4-D5: 完整 Score + 扫描 + MXFP4 交叉 → D6: 入库
