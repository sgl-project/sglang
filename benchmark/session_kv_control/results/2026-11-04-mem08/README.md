# 0.8 显存预算下的持续 KV 压力测试

已完成的 L2 记录与分析见 [L2 验收](L2_ACCEPTANCE.zh-CN.md)：732 次真实生成调用全部有效，实际 L1/L2 缓存容量接近满载，热点恢复延迟下降，总吞吐约持平。

L3 的 64×16K 会话循环矩阵使用自然 GPU 驱逐、8/16/32 并发，比较原生按需读取和工具等待期间的显式预取。优化前后快照分别记录；完整 L3 结果仍在执行，会在后续提交补充。

使用仓库内 `benchmark/session_kv_control/analyze_stress.py` 可重算所有已完成、已封存的实验；脚本不访问在线服务。各目录 `checksums-original.json` 保留压缩前索引，`checksums.json` 校验实际归档文件。
