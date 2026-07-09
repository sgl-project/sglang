# KV Canary HiCache L2 修改摘要

相较于原始代码，本次主要修改如下：

- 新增 HiCache canary bridge，为原始 K/V head/tail canary 分配 L2 host buffer。
- L1→L2 evict 时，使用与 KV 相同的 `device_indices → host_indices` 保存 canary。
- L2→L1 reload 时，使用与 KV 相同的 `host_indices → device_indices` 恢复 canary。
- 普通 MHA HiCache 和 hybrid/SWA HiCache 均已接入；SWA 使用自己的 `PoolTransfer` 索引。
- canary 传输加入原有 HiCache stream/event 顺序，避免 KV 与 canary 状态不同步。
- `CanaryManager` 在绑定 HiCache radix cache 时自动注册 bridge。
- 未修改原始 `--kv-canary` 校验算法、日志格式和参数语义；reload 后继续复用原始校验流程。
- `--kv-canary-real-data=partial/all` 可检测 KV 内容损坏；`none` 只能检测位置、索引和链错配。
- 当前不支持 L3 canary 保存；同时启用 kv-canary 与 HiCache L3 时会 fail-fast。
- 新增 L2 canary 往返、FULL/SWA 索引及 controller 接入测试。

主要新增文件：

- `python/sglang/srt/kv_canary/hicache/bridge.py`
- `python/sglang/srt/kv_canary/hicache/transfer.py`
- `test/registered/kv_canary/test_self_unit_hicache_bridge.py`
