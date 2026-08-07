# MinWM Realtime Production Topology

本目录是可重复部署的生产链路，不是直连 Denoiser 的验证拓扑：

```text
NLB -> Gateway CPU Pool -> Coordinator CPU Pool -> H100 Spot Denoiser
                                             \-> L4 Spot TAEHV
TAEHV -> owning Gateway -> Browser
```

Gateway 和 Coordinator 各至少两个跨 AZ CPU Pod。Coordinator 使用 DynamoDB
On-Demand 保存短期用户、Session 和 Worker slot Lease；GPU KV、latent history 和
TAEHV context 只保存在绑定 Worker 本地。H100 与 L4 使用独立 Spot NodePool，并可独立
缩容到 0。L4 不满足延迟门禁时，使用不在 base kustomization 中的 `l40s-vae.yaml`。

当前模型的生产准入上限为 `4 Session/H100`，8 个 H100 Worker 共 32 个
Denoiser slot。Coordinator 会按 Worker 的有效占用率、队列深度和服务耗时选择 slot，
使同一批新 Session 尽量平均分布到 8 张 H100。单 L4 VAE 广播 16 个 Context slot；
如果只部署 1 个 L4 VAE Worker，整体链路会先受 VAE 的 16 个 slot 限制，需要 2 个
L4 VAE Worker 才能完整释放 32 个 Denoiser slot。

## 不可变依赖与模型

- Gateway、Coordinator、Denoiser、VAE 都使用 ECR digest，禁止使用可变 tag。
- 容器启动时禁止 `pip install`、`git clone`、`curl` 或外网下载。
- TAEHV 包和校验过的 `taew2_2.pth` 在 VAE 镜像构建时固化。
- 原始 checkpoint 先由一次性 CPU Spot Publisher 转换为版本化、带 SHA-256 manifest
  的独立 S3 serving artifact；`_READY` 最后写入。Denoiser 只挂载只读 serving
  artifact，启动时不再转换 checkpoint。
- 集群复用已有 NVIDIA device plugin、S3 CSI Driver 和 EC2NodeClass，不重复安装集群级
  组件。

## 标准发布顺序

所有 AWS/Kubernetes 写操作执行前，必须先展示精确资源、范围、费用影响和清理方案，并按
仓库规则获得当次人工确认。

1. `provision_aws.sh` 创建 CloudFormation 控制面：DynamoDB、5 天 CloudWatch Log
   Group、不可变 ECR 和最小权限 IRSA。
2. `docker/build_and_push.sh` 构建并推送四个角色镜像，生成 `.env.images` digest 清单。
3. `publish_model_artifact.sh` 在一次性 `r7i.8xlarge Spot` 节点发布模型制品并写
   `_READY`；Publisher 使用独立 300Gi 加密 gp3 NodeClass，结束后随 NodePool 删除。
4. `deploy_production.sh` 只读检查 DDB、Log Group、模型 `_READY` 和所有镜像 digest，
   然后 server-side apply 完整拓扑。
   Denoiser StatefulSet 固定使用 `podManagementPolicy: Parallel` 和
   `updateStrategy: OnDelete`；GPU 版本变更时默认按 2 个 Pod 一批执行
   `2 -> 2 -> 2 -> 2` 滚动替换。每批两个旧 Pod 完全删除、同名新 Pod 创建并全部
   Ready 后才进入下一批，
   因此更新期间至少保留 6 张卡服务，不做逐卡串行更新，也不一次删空整台节点。
5. 从 NLB 运行 `browser_probe.cjs` 与 `e2e_production_chain.py`；测试必须证明真实
   Coordinator 配对、VAE 直传、独立 Trace HTTP 查询和 Display Lag 门禁。
6. 本轮测试完成后保持服务运行供人工验证，不调用清理脚本。收到明确清理指令后才运行
   `cleanup_production.sh --execute`，并验证 GPU Node、NLB 和命名空间全部消失。版本化
   模型制品和 ECR 镜像默认保留，便于下次复用。

## 回滚

部署脚本在写入前保存 workload spec。发布失败时使用原 spec 执行原地 Server-Side Apply；
Deployment 保持原地滚动恢复，Denoiser StatefulSet 则恢复原 spec 后按相同的 2 Pod
批次回滚。模型版本
由 `MODEL_ARTIFACT_REVISION` 固定。DynamoDB Schema 保持向后兼容；若控制面发布失败，现有
Session 不迁移，受影响用户重试。Spot Worker 不复制状态，节点中断时仅终止绑定 Session，
Coordinator TTL 自动回收 Lease。

### GPU 分批重启与冷加载

- 默认 `DENOISER_RESTART_BATCH_SIZE=2`，同批两个 Pod 并行重建，批次之间有严格
  Ready 屏障。
- 发布期间临时给承载 Denoiser 的节点加 `karpenter.sh/do-not-disrupt`，完成后移除，
  避免 Pod 短暂减少时触发 Spot 节点整机回收。
- 宿主机模型制品只由一个 `model-stager` 写入，其他 Pod 通过 `flock`
  复用本地缓存，避免重复下载和覆盖。
- 同批最多两个 Pod 进入冷加载，槽位锁在 Denoiser `/health` 成功后立即释放。
- 验收时必须检查每批两个 Pod 都恢复 Ready，最终 8 个 Pod 全部位于新 revision，
  restart count 为 0。

## 验收门禁

- 单用户和 2/4/8 并发 Session 全部完成且错误率为 0；当部署 8 个 H100 Denoiser
  Worker 和 2 个 L4 VAE Worker 时，32 并发应全部准入，超过 32 的请求明确返回
  `CAPACITY_EXHAUSTED`。
- 视频 WebSocket 不包含 Trace payload；Trace 仅通过 OTLP 和独立 HTTP Query API。
- Warm session Display Lag P95 不高于 250ms。
- 每个 Session 的 latent、Gateway 输出队列和 Trace 查询并发均有硬上限。
- 容量控制器能够从 Coordinator 的 shared-capacity 快照扩容，并在 active、queued 或
  draining 非零时拒绝缩容；定时 scale-to-zero 与事件扩容可以独立暂停。
- 故障测试完成后记录 H100/L4 Pod、Node、Spot 生命周期、NLB 与访问 URL，并保持服务运行
  供人工验证。只有收到明确清理指令后，才验证标签
  `seedleap.ai/test-run=minwm-async-vae-benchmark` 的 Node 归零。
