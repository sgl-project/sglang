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

这是 internal NLB。NLB 安全组允许 VPC/私网网段 `10.0.0.0/8`、
`172.16.0.0/12`，并参考现有 `lingbot2-h200-public` 服务加入公司出口
`5.34.217.44/32`、`45.8.204.4/32`；没有开放 `0.0.0.0/0`。
但安全组放行不等于网络路径可达：当前 VPC 没有到公司网络的
TGW/VPN/peering，internal NLB 也没有公网入口，因此普通办公网络仍不能直接访问。
当前没有配置 TLS，入口使用 HTTP/WS，而不是 HTTPS/WSS。

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

这三个值由 SGLang checkpoint converter 写入服务目录，并不储存在 `model.pt`
权重张量中。converter 现在将其作为显式参数写入 conversion manifest，避免再把
serving 默认值误记成 checkpoint 训练合同。

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

桌面上的代理/TUN DNS 还会把 NLB hostname 映射到 `198.18.0.0/15` 的 fake-IP；
即使 NLB 安全组已放行公司出口 `/32`，该 fake-IP 和缺失的 VPC 公司网路由仍会
使桌面请求失败。要获得可直接打开的公司入口，需要二选一：

- 将 Service 改为 internet-facing NLB，并继续仅允许公司出口 `/32`；这会重建
  NLB，hostname 会变化。
- 保持 internal NLB，为 VPC 建立到公司网络的 TGW/VPN/peering，并确保公司 DNS
  返回 NLB 的真实 `10.82.x.x` 地址。

## 7. 重大决策和与预期不同之处

### 7.1 Spot region 不是最初尝试的 region

最初尝试的 us-east-2、ATL Local Zone 和 us-east-1d 都返回
`UnfulfillableCapacity`。Spot placement score 显示 us-west-2d 为 9/10，因此
新增精确的 `2d` worker 子网并部署到 `minwm-test-phx2`。最终确实获得了
p6-b200 Spot。

### 7.2 Checkpoint 不从本机中转

20 GB checkpoint 使用 S3 服务端跨区复制，并校验长度、CRC64NVME、ETag 和
SHA-256。这样避免本机带宽成为瓶颈，也留下可重复验证的 staging object。

### 7.3 MinWM 和 LingBot 都支持 sink/window，默认 profile 不同

此前把 UI 的 `sink=9/window=18` 解释成“LingBot 专属，不能用于 MinWM”，这个结论
不正确。2026-07-27 核对的 seedleap/minWM 最新 `main`
`0796bc201fae4c86f100620cb23402ae21c8f3b5` 中，5B DMD 仍通过
`UnifiedDiffusionInferencePipelineV3` 调用模型自持的 `CausalKVCache`；该 cache
原生同时实现有限 `local_attn_size` window 和 `sink_size`。

需要区分两套 profile：

- MinWM exact：不下发 override，复现 base config 的
  `local_attn_size=-1/sink_size=0` 完整历史；
- bounded serving：当前部署默认使用 `sink=4/window=20`，限制长 session 的 KV
  容量；Web UI 与 `sglang serve` 默认值保持一致，单次 WebSocket 请求仍可覆盖。
  它是 MinWM 和 LingBot 都支持的性能/内存策略，但历史开始淘汰后不会与完整历史
  baseline bitwise 一致。此前完成的 `sink=9/window=18` parity 属于历史验收合同，
  结果不会因当前 serving preset 改变而重命名或覆盖。

第一次用 9/18 失败的真正原因是 whole-DiT `torch.compile` 编译了有状态 cache
滚动代码，Inductor 生成的动态 slice kernel 引用了未绑定 symbol：

```text
NameError: name 's22' is not defined
```

这不是 KV 参数不兼容。修复只在 MinWM self-attention 调用 cache update 的边界
graph break，把 cursor、滚动 copy 和动态 slice 留在 eager；通用 cache 和 DiT
其余计算路径仍可编译。同时修复 MinWM stage 的请求间 override 泄漏。部署 manifest
保留 UI 的 4/20 bounded preset，exact benchmark 仍显式留空。UI 也不再把 LingBot
的 `0.05/frame`、`4/6 deg/frame` 写成 MinWM 的物理动作幅度，而显示为
`checkpoint-relative`。

B200 实际验证使用 832×480、24 FPS metadata、8 chunks，输出 129 帧并跨过
window=18 的滚动边界。修复后 8/8 chunks 完成；排除冷编译及第二个过渡 chunk，
chunks 2–7 的 scheduler 均值 379.3 ms/chunk，对应约 42.2 generated FPS。
此前同机短测的完整历史与 bounded profile 分别约 40.0/39.9 FPS，未观察到 cache
边界 graph break 的稳态吞吐惩罚；这仍是单 case 冒烟数据，不应当作正式容量结论。

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

### 7.7 实时预览和动作验收需要不同的丢帧策略

人工复现 `W 5s → S 5s → idle 5s → W 5s → S 5s` 时，旧 Web UI 会出现
“卡顿后直接跳到最新画面”。2026-07-27 对运行中的同一实例做了拆分测量：

| 层级 | 实测 |
| --- | ---: |
| 模型端 source | 23.8–24.0 FPS |
| 浏览器 WebP decode | 13–23 ms/chunk |
| 浏览器 render（卡顿窗口） | 7–10 FPS |
| display lag | 4.2–9.8 秒 |

因此这次现象不是 B200/DiT 吞吐下降，而是浏览器播放线程落后。旧播放器为保证
“按键后尽快看到最新结果”，会在两个位置主动丢旧帧：

1. 解码队列超过约 2 秒时丢旧的 encoded frame batch；
2. 播放 backlog 过长或新的 action/prompt event 生效时，按 chunk 裁掉旧 event
   的排队帧。

第二条会让 action-to-display latency 变小，却破坏人工验收的完整时间线：人按了
5 秒，画面中不一定还能看到完整的 5 秒区间。为此 Web UI 增加两种明确模式：

- `Low latency (may skip)`：保留原策略，用于交互体验；允许追赶和丢旧帧。
- `Full timeline (no frame skipping)`：解码、backlog 和 event cutover 都不丢帧，
  用于动作时长、prompt switch 和 parity 观察。浏览器若跟不上，代价是显示延迟和
  内存队列增长，而不是时间线被压缩。

通用 Realtime Studio 仍默认低延迟模式，以免无界积压改变产品语义；本 MinWM
部署默认完整时间线模式。UI 会持续显示模式、队列深度和累计 dropped frame 数，
也可用 `?playback=timeline` 显式选择。记录 MP4 的路径仍在 preview 丢帧之前接收
source frames，因此两种模式都不会把预览丢帧误写成模型输出缺帧。

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
3. 为什么 MinWM exact profile 使用 `-1/0`，但线上 bounded profile 仍可以使用
   `sink=9/window=18`？两者从第几个 latent frame 起可能分叉？
4. 为什么本部署关闭 CFG parallel，并强制 `guidance_scale=0`？
5. checkpoint 转换为什么需要 donor Diffusers 目录，但运行时不依赖
   `~/workspace/minWM`？
6. 为什么把 `sglang serve` 直接 `exec` 成容器 PID 1 会杀死 scheduler worker？
7. 38 FPS 的结论为什么必须同时固定 checkpoint、GPU 数、分辨率、steps、
   action、seed、attention backend 和请求 payload？
8. Spot 节点被回收后，哪些 Kubernetes/AWS 资源能保留，哪些必须重新创建或
   重新加载？
9. 为什么人工验证 5 秒 action 区间时应使用 `Full timeline`，而追求按键后尽快
   看到新画面时应使用 `Low latency`？两种模式各自牺牲了什么？
