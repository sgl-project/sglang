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
5. 从 NLB 运行 `browser_probe.cjs` 与 `e2e_production_chain.py`；测试必须证明真实
   Coordinator 配对、VAE 直传、独立 Trace HTTP 查询和 Display Lag 门禁。
6. 测试完成后运行 `cleanup_production.sh --execute`，验证 GPU Node、NLB 和命名空间
   全部消失。版本化模型制品和 ECR 镜像默认保留，便于下次秒级复用。

## 回滚

应用回滚只替换各工作负载中四个角色的镜像 digest，并重新执行 server-side apply。
模型回滚只替换 `MODEL_ARTIFACT_REVISION`。DynamoDB Schema 保持向后兼容；若控制面发布
失败，现有 Session 不迁移，受影响用户重试。Spot Worker 不复制状态，节点中断时仅终止
绑定 Session，Coordinator TTL 自动回收 Lease。

## 验收门禁

- 单用户和 4 并发 Session 全部完成且错误率为 0。
- 视频 WebSocket 不包含 Trace payload；Trace 仅通过 OTLP 和独立 HTTP Query API。
- Warm session Display Lag P95 不高于 250ms。
- 每个 Session 的 latent、Gateway 输出队列和 Trace 查询并发均有硬上限。
- GPU 资源清理后，标签 `seedleap.ai/test-run=minwm-async-vae-benchmark` 的 Node 为 0。
