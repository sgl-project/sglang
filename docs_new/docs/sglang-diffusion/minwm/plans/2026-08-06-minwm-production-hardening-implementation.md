# MinWM Realtime Production Hardening Implementation Plan

**Goal:** 在保留 T2V/I2V、动态 Action/Prompt、Trace、Dump 和同步 VAE 回滚路径的前提下，
把当前异步 VAE 验证链路补齐为可灰度上线的多用户生产实现，并用 H100 Spot + L4 完成真实
端到端验证。

**Scope:** 安全认证和签名由上游内部服务负责，本计划不实现公网身份认证。其余容量、调度、
故障、弹性、可观测和发布语义必须与
`2026-08-03-minwm-async-vae-multiuser-design.md` 及
`2026-08-04-minwm-runtime-transport-latency-optimization-design.md` 一致。

## Current Model Deployment Target

- Model ID: `wan22-5b-varlen-multishot-product-ws1-afreeze-0805-caaedfe`
- Artifact revision: `gs4000-dmd31-step2800-full-v1`
- Source checkpoint:
  `s3://leap-world-us-east-2/world-model/minwm/checkpoints/dmd-merged/Wan21/Action2V/bidirectional/wan22-5B-varlen-multishot-product-ws1-afreeze-0805-caaedfe/global_step_004000/dmd31-step2800-full/model.pt`
- Source S3 VersionId: `ZsIeQtL0YiGeNEg1mG28XzaaZs813i1u`
- Serving artifact:
  `s3://leap-world-us-west-2/world-model/minwm/serving-artifacts/wan22-5b-varlen-multishot-product-ws1-afreeze-0805-caaedfe/gs4000-dmd31-step2800-full-v1/model/`

## Stage 1: Production Stability

- [x] Gateway 使用有界准入等待队列；容量不足最多等待 10 秒，队列满或超时返回可重试错误。
- [x] Worker heartbeat 携带不可复用的 epoch、lifecycle、活动/排队/阻塞 Session 与服务时间。
- [x] Drain Worker 停止新准入，已有 Session 在 deadline 内完成；Kubernetes preStop 触发 Drain。
- [x] Coordinator 在提交分配后向 Denoiser/VAE 执行幂等 ReserveSession；部分成功必须回滚。
- [x] Worker SessionOpen 消费匹配的 reservation token 和 epoch；旧 Pod/旧 epoch 消息被拒绝。
- [x] VAE 使用跨 Session Round Robin，单 Session 严格有序，队列深度和 credit 全部有界。
- [x] VAE 只有在 latent 进入 Decode 后才发下一张 credit；持续背压终止受影响 Session。
- [x] GPU NodePool 保持 Spot-only、固定上限、空闲缩容与成本告警；不隐式回退 On-Demand。

## Stage 2: Scale And Failure Recovery

- [x] Coordinator 路由同时考虑兼容性、AZ、free slot、queue depth、近期服务时间与负载。
- [x] Worker 心跳丢失、epoch 变化或进入 failed 状态后，Lease renew 主动失败并关闭绑定 Session。
- [x] 浏览器刷新采用“旧连接释放、新 generation 在有界队列等待”的明确语义，不迁移 GPU state。
- [x] GPU scaler 根据共享 Coordinator capacity snapshot 的 waiting/free/active 信号扩容。
- [x] 缩容只在无活动 Session 时执行；Worker 先 Drain，连续低负载后才缩容，支持 scale-to-zero。
- [ ] Spot interruption/Drain 的真实云事件注入尚未执行；慢客户端、主动断连、Worker kill、重连和故障后恢复已完成实测。

## Stage 3: Operations And Release Governance

- [x] Trace 媒体 WebSocket 保持纯视频/控制数据；CloudWatch Query API 提供最近 5 分钟聚合。
- [x] Trace 查询合并、并发有界、缓存 15 秒，CloudWatch 暂无新结果时保留上次成功值。
- [x] 生产与 benchmark 配置显式分层；生产资源不携带测试 TTL/cleanup 标签。
- [x] 发布前验证不可变镜像 digest、模型 `_READY`、DynamoDB schema、Trace 5 天保留和清单占位符。
- [x] 发布过程支持预检、rollout status、失败自动回滚；回滚只切角色镜像和模型 revision。
- [x] 运维文档列出容量、告警、故障定位、扩缩容、回滚和人工释放资源的标准命令。

## Verification Gate

- [x] 相关单元/清单/脚本测试全部通过，`git diff --check` 无错误；本机未安装 Torch，因此未运行整个仓库的全量 Pytest。
- [x] 从公网 NLB 经 Gateway、Coordinator、H100 Denoiser、L4 VAE 完成 T2V/I2V。
- [x] 1/2/4/8 并发记录端到端、chunk、denoise、VAE queue/decode/encode、display lag 和 overlap。
- [x] 验证 Action/Prompt 更新、Trace HTTP、Dump、断连、Worker kill、重连、慢消费者和容量满拒绝。
- [x] 代码与两份设计文档逐条核对，未实测的 Spot interruption/Drain 云事件保留为后续项。
- [x] 测试完成后保留 H100 Spot、L4、控制面和 NLB，供人工验证，不执行自动清理。
