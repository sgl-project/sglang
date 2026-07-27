# MinWM 0724 DMD 实时服务部署记录

更新时间：2026-07-27

部署状态：已运行并完成 Web UI、NLB、WebSocket、832×480 视频和吞吐验收

## 1. 结果摘要

本次部署使用 SGLang 的 `MinWMCausalDMDPipeline`，运行最新的 MinWM
0724 bidirectional DMD checkpoint。实际获得的计算节点是
`p6-b200.48xlarge` Spot，服务 Pod 只申请并看到其中 1 张 B200。

内网入口：

- Web UI：
  `http://k8s-default-minwmdmd-68fa5a3b8e-1e3897a685e566f1.elb.us-west-2.amazonaws.com/`
- WebSocket：
  `ws://k8s-default-minwmdmd-68fa5a3b8e-1e3897a685e566f1.elb.us-west-2.amazonaws.com:30000/v1/realtime_video/generate`
- 健康检查：
  `http://k8s-default-minwmdmd-68fa5a3b8e-1e3897a685e566f1.elb.us-west-2.amazonaws.com:30000/health`
- 模型列表：
  `http://k8s-default-minwmdmd-68fa5a3b8e-1e3897a685e566f1.elb.us-west-2.amazonaws.com:30000/v1/models`

这是 internal NLB，只允许 `10.0.0.0/8` 和 `172.16.0.0/12`，不能从普通公网访问。
当前没有配置 TLS，因此内网入口使用 HTTP/WS，而不是 HTTPS/WSS。

稳态吞吐结果：

| 指标 | 结果 |
| --- | ---: |
| SGLang scheduler forward | 37.906 FPS |
| 服务端完整 chunk | 37.541 FPS |
| 客户端通过 NLB 实收 | 37.874 FPS |
| 目标实时帧率 | 24 FPS |
| 余量（客户端实收 / 目标） | 1.58× |

Web UI 开启 24 FPS pacing 时，服务端每 16 帧约 0.665 秒，约等于
24.06 FPS。浏览器验收进入 `Live` 状态并实际收到 87 帧。

## 2. 部署拓扑

```text
公司内网浏览器
    │
    ├── HTTP :80 ───────┐
    │                   ▼
    └── WS :30000 ── internal NLB
                         │  IP target，cross-zone
                         ▼
                 EKS Pod 10.82.3.52
                    ├── :18080 SGLang Realtime Web UI
                    └── :30000 SGLang MinWM API/WebSocket
                              │
                              ▼
                        1 × NVIDIA B200
```

NLB 在 `us-west-2a/2b/2d` 启用，Pod 位于 `us-west-2d`。Web UI 会根据
当前页面 hostname 自动构造：

```text
ws://<当前 hostname>:30000/v1/realtime_video/generate
```

因此从 NLB 打开 UI 时不需要手工填写 WebSocket 地址。

## 3. 代码、模型和运行环境

### 3.1 代码

- SGLang 分支：`codex/minwm-realtime-api`
- SGLang commit：`8de158c6e9d4fae849cc15cc77b1185db1d20511`
- 参考的 MinWM main commit：
  `c6432ac1ef5ecadf04cb772394e5bef06d0ef4a8`
- 容器镜像：
  `829115578968.dkr.ecr.us-east-2.amazonaws.com/leap-world/minwm-training@sha256:bedc07ea3ba55059a8c1c569c3b177c4d00d41f37d4fa9105375531534ef5f2a`

部署运行时不会 import、mount 或 clone `~/workspace/minWM`。MinWM 仓库仅用于实现时
核对 main 分支语义；线上运行只依赖 SGLang、转换后的 checkpoint 和 donor
Diffusers 组件。

### 3.2 Checkpoint

用户指定的源：

```text
s3://leap-world-us-west-2/world-model/minwm/checkpoints/run-archive/rolling/Wan21/Action2V/bidirectional/wan22-5B-varlen-multishot-texiao-addsplithq-da25148-dmd-0724-5eba381389f-merge/global_step_011000/generator/model.pt
```

为让 us-west-2 EKS 使用现有 S3 CSI claim，服务端复制到：

```text
s3://leap-world-us-east-2/world-model/evals/minwm/checkpoint-staging/wan22-5B-varlen-multishot-texiao-addsplithq-da25148-dmd-0724-5eba381389f-merge/global_step_011000/generator/model.pt
```

复制完全在 AWS 服务端执行，没有经本机下载再上传。校验结果：

| 字段 | 值 |
| --- | --- |
| ContentLength | 20,014,135,255 bytes |
| CRC64NVME | `8KS+pbHYSjY=` |
| ETag | `d76d7a982b7eca7908e4b2a0fb4a4f6b-2386` |
| 源 VersionId | `Byk70ZwuVy96DMkNuwe_dAHTtvICQCcC` |
| 目标 VersionId | `yX2Wlgz70mNgNUa0maBe0yyXShIo2wjk` |
| SHA-256 | `6fa23f07a9b912c76724d14d2b217904bba3854c026d7e0dffcd43861e7c4486` |

源和目标的长度、CRC64NVME、ETag 完全一致。

转换后的模型：

- action type：`primitive_token_residual`
- 30 个 transformer blocks
- 5,003,467,456 parameters
- 841 tensors
- 5 个 safetensors shards
- 权重总大小：20,013,869,824 bytes
- 原生几何：832×480
- `num_frames_per_block=4`
- `local_attn_size=-1`
- `sink_size=0`
- `sliding_window_num_frames=128`

转换结果、checkpoint 和 donor 文件保存在 200 GiB GP3 PVC 上。Pod 重启不会重复
复制或转换 checkpoint。

## 4. 计算和 Kubernetes 资源

### 4.1 实际实例

| 字段 | 值 |
| --- | --- |
| Cluster | `minwm-test-phx2` |
| Region / AZ | `us-west-2 / us-west-2d` |
| EC2 | `i-0e2f487faad8413a7` |
| Instance type | `p6-b200.48xlarge` |
| Lifecycle | Spot |
| Node private IP | `10.82.3.245` |
| Pod GPU request | `nvidia.com/gpu: 1` |
| Pod 内可见 GPU | 1 × NVIDIA B200，183,359 MiB |
| NodePool | `minwm-test-usw2d-p6-spot` |
| NodeClaim | `minwm-test-usw2d-p6-spot-jxfbm` |

NodePool 只允许 Spot、`p6-b300.48xlarge` 或 `p6-b200.48xlarge`，并限制在
`us-west-2d`。它不会使用已有的 B300 Capacity Block。

### 4.2 新建子网

- Subnet：`subnet-05f94519b218b0f37`
- VPC：`vpc-00727fc3526cbf033`
- AZ：`us-west-2d`
- CIDR：`10.82.3.0/24`
- Route table：`rtb-08b9f539ebd9bfa1a`
- Route table association：`rtbassoc-003afcc9c2c52d886`

这个子网用于 `2d` 的 EKS Auto Mode Spot worker，同时也被加入 internal NLB，
保证 `2d` IP target 能进入健康状态。

## 5. 推理配置

服务启动参数的关键部分：

```text
pipeline              MinWMCausalDMDPipeline
attention backend     fa
performance mode      speed
torch.compile         true
GPU count             1
SP degree             1
CFG parallel          false
warmup mode           off
```

环境配置：

```text
MINWM_ATTENTION_IMPL=dense
MINWM_PACKED_ATTENTION_DETERMINISTIC=false
TORCHINDUCTOR_CACHE_DIR=/work/torchinductor
```

请求合同：

```text
size                   832x480
fps                    24
seed                   42
steps                  4
guidance_scale         0.0
action_type            primitive_token_residual
action label           9（forward）
latent frames/chunk    4
pixel frames/chunk     16
```

DMD student 没有 CFG 的 uncond lane，因此 guidance 被规范化为 0，CFG parallel
关闭。

## 6. 验收结果

### 6.1 NLB 和 API

滚动重启后，两组 NLB target 均为 healthy：

| 端口 | Target | 状态 |
| --- | --- | --- |
| 18080（Web UI） | `10.82.3.52:18080` | healthy |
| 30000（API/WS） | `10.82.3.52:30000` | healthy |

通过 NLB 验证：

- `GET :30000/health` → 200，`{"status":"ok"}`
- `GET :30000/v1/models` → 200，模型 `/work/model`
- `GET :80/` → 200，SGLang Realtime Studio
- WebSocket 真正生成 832×480 视频成功

### 6.2 视频

测试 case：`00_forward_pottery`

| 字段 | 值 |
| --- | --- |
| 分辨率 | 832×480 |
| 编码 FPS | 24 |
| 总帧数 | 65（1 reference + 64 generated） |
| 时长 | 2.708333 秒 |
| MP4 大小 | 467,587 bytes |
| MP4 SHA-256 | `15c106e90b0cb3b7dfc32799b53e51d5ca1097488594d3f9ca273df1ad9699e6` |

产物：

```text
benchmark/minwm_realtime_parity/results/latest-checkpoint-0724-spot-b200/
├── cases/00_forward_pottery/
│   ├── sglang_spot_b200_final.mp4
│   └── sglang_spot_b200_final.json
├── inputs/00_forward_pottery.png
├── sglang_spot_b200_final_run.json
└── throughput_final.json
```

最终 Pod 的视频 SHA 与前一个 Pod 的视频 SHA 不同。当前部署选择
`performance_mode=speed`、whole-DiT `torch.compile` 和非确定性 FA 路径，
目标是吞吐而不是 bitwise 重现；同 prompt、seed 和首帧不保证跨进程 bitwise
一致。若要做 parity 回归，必须改用仓库中的 bitwise profile。

### 6.3 吞吐

吞吐测试采用同一条 NLB WebSocket、10 个 warmup chunks 和 20 个 measured
chunks，共计量 320 个生成帧。

| 指标 | 均值/结果 | P95 | P99 |
| --- | ---: | ---: | ---: |
| scheduler forward | 422.10 ms/chunk | 466 ms | 475 ms |
| 完整 chunk | 426.20 ms/chunk | 472 ms | 478 ms |
| NLB 客户端 chunk 间隔 | 422.45 ms/chunk | 467.94 ms | 472.95 ms |
| scheduler FPS | 37.906 | — | — |
| 完整服务端 FPS | 37.541 | — | — |
| NLB 客户端 FPS | 37.874 | — | — |

这里的 38 FPS 是“不做输出 pacing 时模型能生产多少帧”。Web UI 默认启用
24 FPS pacing，所以 UI 路径会主动等待并按约 24 FPS 推送，不能把 pacing 后的
24 FPS误判成模型吞吐下降。

首次部署的第一个 cold compile chunk 用时 126.64 秒。最终声明式 Pod 重启后的
首个验证 case 仍有 44.71 秒 compile/cache 恢复开销，最后两个 chunk 分别是
396 ms 和 402 ms。完成 10 个 warmup chunks 后，最终吞吐测试的首 payload
时间为 2.959 秒，之后进入上述约 38 FPS 的稳定窗口。

### 6.4 Web UI

浏览器实际检查了以下状态：

- 自动识别模型 `/work/model`
- 默认分辨率 832×480
- 默认目标 24 FPS
- WebSocket 连接成功
- 状态进入 `Live`
- 实际收到 87 帧
- WebP preview 正常解码和展示
- `Close session` 正常释放会话

由于执行验收的桌面不在该 VPC 路由内，浏览器视觉检查通过临时
`kubectl port-forward` 连接同一个 Service。NLB HTTP、模型列表、健康检查、
WebSocket 视频和吞吐则全部从集群内通过真正的 internal NLB 地址完成。

## 7. 重大决策和与预期不同之处

### 7.1 Spot region 不是最初尝试的 region

最初尝试的 us-east-2、ATL Local Zone 和 us-east-1d 都返回
`UnfulfillableCapacity`。Spot placement score 显示 us-west-2d 为 9/10，因此
新增精确的 `2d` worker 子网并部署到 `minwm-test-phx2`。最终确实获得了
p6-b200 Spot。

### 7.2 Checkpoint 不从本机中转

20 GB checkpoint 使用 S3 服务端跨区复制，并校验长度、CRC64NVME、ETag 和
SHA-256。这样避免本机带宽成为瓶颈，也留下可重复验证的 staging object。

### 7.3 Web UI 的 LingBot 默认 KV 参数不能直接用于 MinWM

配套 UI 原默认 `sink=9/window=18`，它来自 LingBot。这个 MinWM checkpoint 是：

```text
local_attn_size=-1
sink_size=0
sliding_window_num_frames=128
```

浏览器第一次生成时，LingBot override 触发了一条不同的动态 compile 图，
TorchInductor 生成代码报：

```text
NameError: name 's22' is not defined
```

部署版 UI 现在不下发 sink/window override，让 SGLang 使用 MinWM 对齐语义：
sink 取模型的 0，因 `local_attn_size=-1` 保留完整因果历史。修正后 Web UI
生成成功。

### 7.4 NLB 必须启用 Pod 所在 AZ

最初 NLB 只在 `2a/2b`，Pod 在 `2d`，target 状态是：

```text
unused / Target.NotInUse /
Target is in an Availability Zone that is not enabled for the load balancer
```

将 `subnet-05f94519b218b0f37` 加入 NLB 后，target 变为 healthy。EKS Auto
Mode load-balancing role 还缺少 `ec2:DeleteTags`，导致控制器更新 subnet 时失败；
本次用部署角色对该 NLB 执行等价的 `set-subnets`。Manifest 已声明三个子网，
但长期方案仍应补齐集群 load-balancing role 的权限。

### 7.5 不能让 `sglang` 成为 PID 1

若容器执行 `exec sglang serve`，CLI 成为 PID 1，它创建的 scheduler worker 的
parent PID 也是 1。SGLang 的 orphan guard 会把这种 worker 判定为孤儿并主动
SIGKILL，表现为 scheduler 无错误退出、return code `-9`。

部署脚本保留 Bash 为 PID 1，在后台启动 `sglang serve` 并 `wait`。这样 scheduler
的 parent 是 SGLang CLI，而不是 PID 1。

### 7.6 Rust gRPC extension 在当前镜像中不可构建

当前 SGLang editable install 包含 Rust/PyO3 gRPC extension，但 MinWM 镜像没有
Rust toolchain。MinWM WebSocket serving 不使用这个 extension，因此启动脚本仅将
它标记为 optional，再安装 diffusion extras。

长期方案应该构建一份固定镜像，预装 SGLang wheel 和完全锁定的依赖；当前 cold
Pod 启动仍会花约 2 分钟安装依赖，GP3 冷读模型时还可能额外花 4 分钟。

## 8. 运维命令

设置上下文：

```bash
export AWS_PROFILE=wms
export KUBECONFIG=/tmp/kubeconfig-minwm-test-phx2-wms
```

检查状态：

```bash
kubectl get pod,service,pvc -n default -l app=minwm-dmd-0724-p6 -o wide
kubectl logs -n default -l app=minwm-dmd-0724-p6 -f
kubectl rollout status deployment/minwm-dmd-0724-p6 -n default
```

内网健康检查：

```bash
curl http://k8s-default-minwmdmd-68fa5a3b8e-1e3897a685e566f1.elb.us-west-2.amazonaws.com:30000/health
```

部署或更新：

```bash
kubectl apply -f benchmark/minwm_realtime_parity/k8s/minwm_dmd_0724_p6_spot_service.yaml
```

本机临时调试 UI：

```bash
kubectl port-forward -n default service/minwm-dmd-0724-p6 \
  18080:80 30000:30000
```

然后打开 `http://127.0.0.1:18080/`。

## 9. 生命周期和风险

- 这是 Spot 实例，AWS 可以随时回收。
- NodePool `expireAfter=8h`；到期会尝试替换节点，不等于删除 Deployment。
- 模型位于 `us-west-2d` 的 RWO EBS。替换节点必须仍在 `2d` 才能挂载。
- Spot 无容量时服务会 Pending，internal NLB 会暂时没有 healthy target。
- Node 重建后的 cold start 当前约 2–6 分钟，主要来自运行时 pip install 和
  GP3 冷读；应通过固定镜像和服务 warmup 降低。
- NLB 没有 TLS，适合当前公司内网验证，不适合直接暴露到公网。

## 10. 清理记录

us-east-1d 的重复测试部署后来才获得一台 p6-b200 Spot，但 Pod 已连续重启，且
美西服务已经验收完成。已删除该测试部署的：

- `deployment/minwm-dmd-0724-p6`
- `service/minwm-dmd-0724-p6` 及其 internal NLB
- `pvc/minwm-dmd-0724-p6-work` 及 200 GiB EBS
- `nodepool/minwm-test-use1d-p6-spot`
- `nodeclass/minwm-test-use1d`
- `storageclass/minwm-dmd-0724-p6-gp3`

没有删除或修改任何共享 NodePool。

## 11. 理解检查

1. 为什么吞吐测试报告约 38 FPS，而 Web UI 的服务端 chunk 被 pacing 到约
   24 FPS？这两者分别测量什么？
2. 为什么 internal NLB 仅放在 `2a/2b` 时，即使打开 cross-zone，`2d` 的 IP
   target 仍是 `unused`？
3. `local_attn_size=-1`、`sink_size=0` 和 `window=18` override 同时出现时，
   哪一项背离了这个 checkpoint 的 MinWM main 语义？
4. 为什么本部署关闭 CFG parallel，并强制 `guidance_scale=0`？
5. checkpoint 转换为什么需要 donor Diffusers 目录，但运行时不依赖
   `~/workspace/minWM`？
6. 为什么把 `sglang serve` 直接 `exec` 成容器 PID 1 会杀死 scheduler worker？
7. 38 FPS 的结论为什么必须同时固定 checkpoint、GPU 数、分辨率、steps、
   action、seed、attention backend 和请求 payload？
8. Spot 节点被回收后，哪些 Kubernetes/AWS 资源能保留，哪些必须重新创建或
   重新加载？
