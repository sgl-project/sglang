# KV Canary 与 HiCache L2 集成修改方案（待确认）

## 0. 本文状态

- 本文件只记录计划修改的文件和内容。
- 当前尚未修改任何 Python、CUDA/C++ 或测试代码。
- 本方案以“复用原始 `--kv-canary`”为原则：只补充 canary 在 HiCache L2 中的保存与恢复，不新增另一套 L2 校验算法。

## 1. 需求一的准确定义

需要防护的链路是：

```text
L1 生成 KV + 原始 canary
        ↓ evict（使用同一组 device_indices → host_indices）
L2 保存 KV + 原始 canary
        ↓ reload（使用同一组 host_indices → device_indices）
L1 恢复 KV + 原始 canary
        ↓
复用原始 --kv-canary 校验
```

目标是发现 KV 在以下环节发生的损坏或错配：

- L1→L2 evict 传输；
- L2 host slot 保存；
- host/device 索引映射；
- L2→L1 reload 传输；
- reload 后到 KV 被使用前的 L1 数据。

### 1.1 当前代码是否已经实现

没有完整实现。

- 当前原始 `--kv-canary` 已经能在 L1 生成 canary，并按配置记录 token、position、chain hash 和 real-KV hash。
- 当前 HiCache L1→L2 只复制真实 KV，没有把对应 canary 保存到 L2。
- 当前 HiCache L2→L1 只恢复真实 KV，没有恢复对应 canary。
- `get_contiguous_buf_infos/get_state_buf_infos` 中已有的 canary 拼接主要服务于 PD disaggregation，不能等同于 HiCache L2 已支持 canary。

因此需要修改的是 HiCache 的 canary 保存/恢复通道；原始 canary 的生成和校验逻辑不需要重写。

## 2. 需求一计划修改的生产代码

### 新建：`python/sglang/srt/kv_canary/hicache/__init__.py`

- 导出 HiCache canary bridge 的安装入口。
- 不放具体传输逻辑，避免 `CanaryManager` 直接依赖不同 HiCache backend 的实现细节。

### 新建：`python/sglang/srt/kv_canary/hicache/bridge.py`

新增 `CanaryHiCacheBridge`，负责原始 canary 的 L2 生命周期：

- 接收现有 `CanaryBufferGroup`，不创建新的 canary 格式。
- 按 HiCache host pool 的 token slot 数量分配 pinned-host canary buffer。
- MHA/FULL 保存现有 `k_head`、`k_tail`、`v_head`、`v_tail`。
- SWA 分别保存 FULL group 和 SWA group 的现有 canary buffer。
- L2 canary 与 L2 KV 共用同一套 host slot 编号；`host_indices[i]` 的 KV 和 canary必须描述同一个 token。
- 提供：
  - `backup(device_indices, host_indices)`：把原始 L1 canary 保存到 L2。
  - `restore(host_indices, device_indices)`：把 L2 中的原始 canary恢复到新的 L1 slot。
  - `destroy()`：释放或注销 pinned-host canary buffer。
- 检查索引数量、范围和 FULL/SWA 索引空间，索引不合法时直接报错，不能静默跳过。
- 不提供 `verify_restored()`，也不生成 L2 专用 verify plan；恢复后继续走原始 `--kv-canary`。

### 新建：`python/sglang/srt/kv_canary/hicache/transfer.py`

- 封装 canary sidecar 的 indexed D2H/H2D 复制。
- 使用与真实 KV 相同的 `device_indices/host_indices` 顺序。
- 优先复用 HiCache 已有的 indexed-transfer 能力；不使用逐 token Python 循环。
- 支持当前使用的 `kernel` 和 `direct` IO backend。
- 传输异步提交到当前 HiCache stream，不在内部执行全局 `synchronize()`。
- 对传输使用的索引 tensor 执行正确的 stream 生命周期管理。

### 修改：`python/sglang/srt/kv_canary/runner/canary_manager.py`

修改 `CanaryManager.attach_radix_cache()`：

- 保留当前 sweep 和 perturb 安装逻辑。
- 检测到 HiCache `cache_controller` 后创建 `CanaryHiCacheBridge`。
- 将 bridge 注册给 cache controller。
- 保存 bridge 引用，保证 L2 canary buffer 在 server 生命周期内有效。
- 防止重复 attach 和重复传输同一组 canary。

### 修改：`python/sglang/srt/managers/cache_controller.py`

为普通 `HiCacheController` 增加可选 canary sidecar hook：

- 新增 `register_canary_hicache_bridge(bridge)`。
- `start_writing()` 中：
  1. 使用现有逻辑把真实 KV 从 L1 复制到 L2；
  2. 使用完全相同的 `device_indices/host_indices` 调用 `bridge.backup(...)`；
  3. KV 和 canary 都提交完成后才 `finish_event.record()`。
- `start_loading()` 中：
  1. 使用现有逻辑把真实 KV 从 L2 恢复到 L1；
  2. 使用完全相同的 `host_indices/device_indices` 调用 `bridge.restore(...)`；
  3. KV 和 canary 都提交完成后才完成 load event，随后由原始 `--kv-canary` 在正常使用路径校验。
- bridge 未注册时保持原有行为不变。

### 修改：`python/sglang/srt/mem_cache/hybrid_cache/hybrid_cache_controller.py`

- 添加与普通 `HiCacheController` 相同的 bridge 注册和调用点。
- FULL 与 SWA group 必须使用各自已经解析完成的 pool-transfer 索引，不能错误地共用 FULL 索引。
- KV、SWA state 和对应 canary 都提交后才能完成 transfer event。

## 3. 明确不修改的原始 KV Canary 代码

按照“复用原始 `--kv-canary`”的要求，第一版不修改以下文件的校验语义：

- `python/sglang/jit_kernel/kv_canary/verify.py`
- `python/sglang/jit_kernel/csrc/kv_canary/canary_verify.cuh`
- `python/sglang/srt/kv_canary/endpoint.py`
- `python/sglang/srt/kv_canary/runner/violation_reporter.py`
- `python/sglang/srt/kv_canary/runner/stats_logger.py`
- `python/sglang/srt/kv_canary/state.py`
- `python/sglang/srt/kv_canary/config.py`

具体含义：

- 不新增 `L2_RESTORE_*` launch tag。
- 不新增 L2 专用 verify kernel。
- 不改变现有 violation ring ABI 和日志格式。
- 不改变现有 `log`/`raise` 行为。
- 不在本次修改中实现自动重算。

## 4. reload 后如何复用原始 `--kv-canary`

reload 完成后，L1 的真实 KV 和原始 canary 已一起恢复到新的 device slot。后续请求正常进入原始校验路径：

- real-KV hash 不一致：发现 KV 内容损坏或 KV/canary 错配；
- position 不一致：发现 canary 被恢复到错误逻辑位置；
- chain hash 不一致：发现前缀链或 slot 顺序错配；
- 开启原有 verify-token assert 时，还会比较 token id。

原始校验路径的覆盖范围保持不变：

- per-forward 校验请求当前使用的 head/tail 边界；
- 配置 `--kv-canary-sweep-interval=N` 后，按原有 sweep 机制检查 radix cache 中的 slot。

本方案不会额外承诺“reload 完成时立即全量校验所有 slot”，因为这会引入新的 L2 专用校验流程，不符合本次“复用原始 `--kv-canary`”的要求。

### 4.1 `--kv-canary-real-data` 的必要说明

- `--kv-canary-real-data=partial`：保存并比较真实 KV 的部分指纹，可发现被覆盖范围内的 KV 字节损坏，开销较低。
- `--kv-canary-real-data=all`：保存并比较完整 real-KV source 的指纹，覆盖更强、开销更高。
- `--kv-canary-real-data=none`：只能检查 token、position、chain/index 等一致性，不能发现“索引完全正确但 KV 字节发生改变”的纯数据损坏。

因此，若目标明确包含 KV 内容损坏检测，运行时必须使用 `partial` 或 `all`。本次不修改默认参数，也不改变原始配置语义。

## 5. 第一版支持边界

- 支持：普通 MHA HiCache L1↔L2。
- 计划支持：hybrid/SWA HiCache 的 FULL 与 SWA group，前提是分别使用正确的 pool-transfer 索引。
- L3 storage 不属于本次需求。为避免 L3 reload 使用缺失或陈旧的 canary，第一版在 `kv-canary + HiCache L3` 组合下 fail-fast；L3 若要提供同等保护，需要单独把 canary 纳入 L2↔L3 保存、索引和恢复链路。
- DeepSeek V4 当前 adapter 未提供 real-KV fingerprint；在补齐其 real-KV source 前，不能宣称支持 DSV4 的 KV 内容损坏检测。
- PD disaggregation 现有 canary buffer 拼接逻辑保持不变，并加入回归测试防止破坏。

## 6. 计划新增或修改的测试

### 新建：`test/registered/kv_canary/test_self_unit_hicache_bridge.py`

覆盖：

- L1→L2 后，L2 canary 与原始 L1 canary 按指定索引逐字节一致。
- 清空或污染对应 L1 slot 后，L2→L1 能把原始 canary 恢复到新的 device slot。
- 打乱但保持配对的 `device_indices/host_indices` 时仍正确映射。
- 故意制造 KV/canary 索引错配后，原始 verify 路径能够记录 violation。
- 未参与本次 evict/reload 的 slot 不被覆盖。
- FULL 与 SWA 使用各自的索引空间。

### 修改：`test/registered/kv_canary/test_self_unit_pool_patcher.py`

- 保留现有 PD layout 测试。
- 增加断言：HiCache bridge 不改变 PD buffer 列表顺序，也不会重复插入 canary。

### 新建：`test/registered/kv_canary/test_self_e2e_hicache.py`

至少包含：

1. clean path：第一次请求生成 KV/canary 并 evict 到 L2，驱逐 L1 后通过相同前缀 reload；断言 L2 命中且原始 kv-canary 无 violation。
2. L2 KV corruption：破坏 L2 KV、保留 L2 canary，reload 后断言原始 `verify_real_kv_hash` violation。
3. index mismatch：故意让 KV 与 canary 使用不一致的恢复索引，断言原始 position、chain 或 real-KV hash violation。
4. canary corruption：破坏 L2 canary 后 reload，断言原始校验能够报告 violation。

测试运行时显式使用 `--kv-canary-real-data=partial` 或 `all`；需要扩大 radix 覆盖时使用原有 `--kv-canary-sweep-interval=1`。

## 7. 推荐实施顺序

1. 编写 HiCache bridge 的失败单元测试。
2. 实现 L2 pinned-host canary buffer 和对称 D2H/H2D indexed copy。
3. 接入普通 `HiCacheController`，验证 MHA clean/corruption path。
4. 接入 `HybridCacheController`，验证 FULL/SWA 索引映射。
5. 回归现有 kv-canary、HiCache 和 PD disaggregation 测试。
6. 比较 `partial` 与 `all` 的延迟、带宽和 L2 内存开销。

在你确认上述文件范围和行为前，不进行代码实现。
