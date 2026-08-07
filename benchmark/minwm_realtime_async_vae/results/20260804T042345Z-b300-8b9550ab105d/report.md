# MinWM 异步 VAE 端到端测试报告

## 结论

- 同步基线最高稳定并发：2
- 异步 VAE 最高稳定并发：2
- 并发 1 下 P95：452.2 ms → 370.9 ms
- 端到端 P95 改善：17.97%

## 硬件与部署

| 模式 | Denoiser | VAE | 实例/容量 | GPU 使用数 |
|---|---|---|---|---:|
| baseline | NVIDIA B300 SXM6 AC | TAEHV taew2_2 / denoiser GPU 0 | 1×p6-b300.48xlarge（物理 8 卡）/ capacity-block | 1 |
| async | NVIDIA B300 SXM6 AC | TAEHV taew2_2 / dedicated VAE worker | 1×p6-b300.48xlarge（物理 8 卡）/ capacity-block | 2 |

## 并发压测

| 模式 | 并发 | P95 action→首帧 (ms) | P95 chunk (ms) | 最低单会话 FPS | 集群 FPS | 错误率 |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 1 | 452.2 | 393.0 | 41.38 | 41.38 | 0.00% |
| baseline | 2 | 755.0 | 703.0 | 22.90 | 45.79 | 0.00% |
| baseline | 4 | 1407.8 | 1350.0 | 11.88 | 47.51 | 0.00% |
| baseline | 8 | 2705.6 | 2646.0 | 6.06 | 48.49 | 0.00% |
| async | 1 | 370.9 | 347.0 | 46.53 | 46.53 | 0.00% |
| async | 2 | 698.8 | 647.0 | 25.21 | 50.42 | 0.00% |
| async | 4 | 1316.2 | 1240.0 | 12.96 | 51.85 | 0.00% |
| async | 8 | 2495.2 | 2440.0 | 6.59 | 52.73 | 0.00% |

## 异步收益

| 并发 | action P95 降低 | chunk P95 降低 | 集群吞吐提升 |
|---:|---:|---:|---:|
| 1 | 17.97% | 11.70% | 12.46% |
| 2 | 7.45% | 7.97% | 10.11% |
| 4 | 6.51% | 8.15% | 9.14% |
| 8 | 7.77% | 7.79% | 8.74% |

## 单用户关键阶段 P95

- Denoising：305.7 ms
- 远端 TAEHV decode：27.9 ms
- WebP encode：121.5 ms
- Latent send：0.330 ms
- VAE queue wait：0.099 ms
- 与下一 chunk denoising overlap：35.8 ms

## 容量与配额验证

- 配置上限：Denoiser 8 个活跃会话、VAE 8 个活跃会话、每会话 1 个有界等待 chunk、准入最长等待 10 秒。
- 同时发起 16 个标准会话（每会话 2 个 warmup chunk + 6 个测量 chunk）：8 个成功，8 个明确返回 `CAPACITY_EXHAUSTED`，错误率 50%。
- 结论：本次单副本硬并发上限为 8；按 action→首帧 P95 < 1 秒且最低单会话 FPS >= 16 的实时 SLO，稳定并发上限为 2。
- 16 个短会话可通过排队全部完成，但这不代表 16 个请求同时占用 GPU；长会话压测证明有界准入能阻止无界堆积。

## 浏览器端到端验证

- WebUI 显式启用 I2V/T2V，并以 T2V 模式完成 121 帧生成，状态正常进入 `Closed / generation complete`。
- 页面收到 121 帧，WebP 解码约 2 ms，最后一帧 display lag 约 0.2 秒；Trace 页成功显示 Scheduler、Denoising、TAEHV Decode、Transport 和 Frontend 耗时。
- WASD 按键事件在浏览器中以约 100 ms 周期持续发送；服务内环回压测验证 action event 能被后续 chunk 采样并反映到首帧延迟。
- Mac 到 EKS 的 `kubectl port-forward` 在浏览器验证中出现约 100 秒的客户端侧异常滞后，而同一 Pod 内环回完整生成仅数秒；性能结论使用 Pod 内环数据，未把 port-forward 延迟计入服务性能。

## 边界与成本说明

- 正式测试开始时，H100 Spot 与 B200 Spot 在目标 AZ 的 placement score 均为 1/10，H100 实际申请持续返回 `UnfulfillableCapacity`。
- 资源释放后的复查中，B200 仍为 1/10；H100 有一个 AZ 短暂升到 4/10、其余仍为 1/10。Placement score 是动态容量概率，不等同于实例已申请成功。
- 为避免继续等待并产生额外实例费用，本次使用 aws03 已有 Capacity Block 的 B300 节点；同步模式实际使用 1 张 B300，异步模式使用 1 张 Denoiser B300 + 1 张 VAE B300。
- 架构已支持把 VAE worker 独立调度到低成本 GPU，但本轮没有可用的 L4/L20/L40 节点，因此低成本 VAE GPU 的绝对 decode 性能仍需单独验证。
