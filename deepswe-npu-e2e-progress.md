# NPU 多机 DeepSWE 测试支持 — 进展与交接文档

> 分支：`npu-deepswe-multi-node-eval`（fork: `shun8686/sglang`）
> PR：https://github.com/sgl-project/sglang/pull/53104
> 最后更新：2026-11-28
> 状态：**端到端已跑通**（run `45975067300` 成功，accuracy 0.0）

---

## 1. 目标

在现有 NPU 多机（1P1D，16P）k8s 测试框架上支持 **DeepSWE** 评测：

- NPU 的 prefill/decode/router Pod 只负责推理；
- 由 **router Pod** 在用例执行时，按 `kubernetes.io/arch=amd64` 动态拉起一个**临时评测 Pod**（`dind` sidecar + 评测容器），在 amd64 节点上跑 evalscope `deep_swe`；
- 结果（accuracy）经日志回传，汇总进 `metrics.json`；
- 不改动 workflow 的整体编排（仍是 volcano Job + 复用 multi-node workflow）。

---

## 2. 当前状态（2026-11-28）

### 2.1 已打通的链路

run [`45975067300`](https://github.com/sgl-project/sglang/actions/runs/45975067300) **成功**（job 2h14m46s，`limit=1`）：

- 评测 Pod 构建任务镜像、跑 agent、调 router（80 次 `chat/completions` **全部 200**）、跑 verifier、写出 `reward.txt`；
- 回传 `[DEEPSWE_METRICS] {"accuracy": 0.0}` → `metrics.json`（`status=pass`，因 `accuracy=None` 暂无基线）；
- trial 目录完整回传：`agent/mini-swe-agent.trajectory.json`、`agent/mini-swe-agent.txt`、`artifacts/model.patch`（32KB）、`verifier/reward.txt`、`verifier/test-stdout.txt`、`reports/*.json/html`。

### 2.2 本次结果

单任务 `superjson-error-stack-serialization`，accuracy **0.0**：

| 项 | 结果 |
|---|---|
| agent | 80 次 API 调用 / 77 steps，**被 5400s（90min）官方 agent 预算掐断**（`exit_status` 为空、无 submission） |
| baseline 测试 | **81 passed**（改动未破坏既有行为） |
| new 测试 | **112/116 passed，4 failed**：`maxCauseDepth` 非整数回退、`normalizeNewlines` CRLF 两个、`normalizeErrorStackOptions` 默认值 |
| 结论 | 功能基本实现，差 4 个边界；不是 harness 问题 |

### 2.3 未推送提交

- `4b23625500` trial 日志 glob 修复（补时间戳层级）
- `770478dc5b` 环境变量/启动参数对齐 128k1k 性能用例

---

## 3. 总体架构

```
GitHub Actions (nightly-test-npu.yml)
  └─ reusable: nightly-test-npu-e2e-multi-node.yml  ("Run test")
       └─ python run_npu_e2e_test.py  (render jinja → volcano Job + ConfigMap)
            ├─ prefill / decode Pod  ── run_npu_testcase.sh → python <case>（只推理）
            └─ router Pod            ── play role "router"
                  └─ run_accuracy()  @check_role("router")
                       └─ run_evalscope_with_retry()
                            └─ deep_swe → run_evalscope_on_x86_pod()
                                 ├─ 创建 amd64 评测 Pod（dind + eval 两容器）
                                 ├─ 流式转发 eval 日志（前缀 [deepswe-eval]）
                                 ├─ 轮询退出码 / 解析 [DEEPSWE_METRICS]
                                 └─ finally 删除 Pod
                  └─ assert_metrics() → [METRIC] accuracy=...
       └─ monitor_pod_logs(router) → generate_metrics_json() → metrics.json
```

评测 Pod 内（两容器共享同一个 `share` 卷）：

```
dind (docker:27-dind, privileged)
  ├─ /var/lib/docker ← emptyDir（每次运行空仓库，镜像需重拉）
  └─ /root/.cache    ← share 卷（与 eval 容器同路径，见问题时序）
eval
  ├─ /root/.cache    ← share 卷（evalscope 快照 / trial 目录）
  ├─ /root/sglang    ← share 卷 subPath（sglang 源码）
  └─ run_deepswe_eval.sh
       ├─ 装 docker CLI + compose 插件
       ├─ 等 dind (tcp://127.0.0.1:2375)
       └─ run_deepswe_eval.py
            ├─ run_evalscope.sh（venv --system-site-packages + datacurve-pier）
            ├─ patch pier：uv 源 / 网络策略
            ├─ 心跳：每 60s 打印 docker ps / images
            └─ evalscope run_task(deep_swe, api_url=http://<router>:6677/v1, limit=1)
                 └─ pier → docker compose build/up（用 dind）→ agent 调 router → verifier
```

关键点：pier 把 trial 的 `agent/`、`verifier/`、`artifacts/` 作为 **bind mount** 交给 dind，source 在 `/root/.cache/tests/...`。因此 **dind 必须把 `share` 卷挂到同一路径 `/root/.cache`**，否则写入落在 dind 私有目录，eval 容器读不到（曾导致 trajectory/reward/model.patch 全部“写了但不存在”）。

---

## 4. 关键文件

| 文件 | 作用 |
|---|---|
| `test/registered/npu/accuracy/deepseek_v4_flash/test_npu_deepseek_v4_flash_0731_w8a8_1p1d_16p_deepswe.py` | 用例：1P1D 16P 部署 + `deep_swe` 评测（env/args 已对齐性能用例） |
| `test/registered/npu/performance/deepseek_v4_flash/test_npu_deepseek_v4_flash_w8a8_1p1d_16p_in128k_out1k_prefix90_20ms.py` | 部署参考（env/args 对齐来源，含 `expert-layout-ais-replicas16.json`） |
| `python/sglang/test/ascend/e2e/test_npu_accuracy_utils.py` | 评测框架；`run_evalscope_on_x86_pod` / `_build_deepswe_eval_pod` 等 |
| `python/sglang/test/ascend/e2e/run_deepswe_eval.sh` | 评测 Pod 入口（docker CLI + compose + 等 daemon） |
| `python/sglang/test/ascend/e2e/run_deepswe_eval.py` | 评测驱动（evalscope/pier、uv 源补丁、网络策略补丁、心跳、trial 日志抓取） |
| `python/sglang/test/ascend/e2e/run_evalscope.sh` | 评测环境构建（解释器选择 / venv / pier 安装） |
| `python/sglang/test/ascend/e2e/run_npu_e2e_test.py` | 多机编排（建 Job、存活检查、日志监控、metrics） |
| `python/sglang/test/ascend/e2e/k8s_multi_pd_separation.yaml.jinja2` | Pod 模板 |
| `.github/workflows/nightly-test-npu.yml` | nightly 入口；当前只保留 DeepSWE 多机用例（临时加 `pull_request`） |
| `.github/workflows/nightly-test-npu-e2e-multi-node.yml` | 多机复用 workflow（Run test / Clear / Post process） |

---

## 5. 提交记录（`8833aa2478` 之后）

| commit | 内容 | 状态 |
|---|---|---|
| `76da87d222` | dind 用 daocloud docker.io mirror | 已推 |
| `3ea9074a69` ~ `7d7372aea1` | ubuntu:24.04 apt 签名补丁系列（后被 dind 27 方案移除） | 已推 |
| `bc1eb12a31` | 换 `docker:27-dind`（SWR） | 已推 |
| `0d3d8c55c5` | uv 改从 OBS 下载（GitHub release 不可达） | 已推 |
| `6ea1625cbc` | 删除 ubuntu:24.04 apt 补丁 | 已推 |
| `629d6864f0` | 精简 entrypoint（dind 27 后无需 compose 兜底/API 协商） | 已推 |
| `a9f894b1c4` | 绕过 pier 的 no-internet squid 代理，让 agent 能连 router | 已推 |
| `bdda6d54b1` | `uv tool install` 显式 `--index-url`（避免 pythonhosted） | 已推 |
| `5d8348382a` | dind 心跳日志 + 单任务 smoke（`limit=1`） | 已推 |
| `ca6696ebc4` | dind 资源 2C/4Gi → limits 8C/8Gi | 已推 |
| `ee6d19e2ad` | 传输后端固定 `ascend`（绕开 mooncake KV layout 拒绝） | 已推 |
| `8ffd3db503` | dind 共享 `share` 卷 + 失败时 dump trial 日志 | 已推 |
| `d8a9df4414` | 启用 `--tool-call-parser deepseekv4 --reasoning-parser deepseek-v4` | 已推 |
| `4b23625500` | trial 日志 glob 补时间戳层级 | **本地** |
| `770478dc5b` | env/args 对齐 128k1k 性能用例 | **本地** |

早期提交（`8833aa2478` 及之前：dind 引入、SWR 镜像、Pod 名/清理/存活检查修复、Python 3.12、BuildKit 关闭等）见 git 历史。

---

## 6. 问题与修复记录

| 现象 | 根因 | 修复 |
|---|---|---|
| 重试创建 Pod 409 | 固定 Pod 名冲突 | 名字加 6 位随机后缀 |
| Run test 提前失败、eval Pod 泄漏 | 存活检查把 eval Pod 当 volcano Pod | `check_pods_running` 跳过 `app=sgl-deepswe-eval` |
| Post process 卡住 | 删除命令漏 `-n` + `while true` | 指定 namespace + 有界重试 + force delete |
| `./k8s_multi_pd_*.yaml does not exist` | runner 已删文件 | 遍历匹配文件，无匹配静默 |
| `ImagePullBackOff` | SWR 路径不存在 / 镜像不可拉 | 改用可拉镜像并写死 |
| `client version 1.52 too new. Maximum 1.38` | 节点缓存旧 dind | 升级 `docker:27-dind` + `Always` |
| `datacurve-pier Requires-Python >=3.12` | 镜像 python 3.11 | 换多架构 sglang 镜像（3.12） |
| evalscope 重装慢 | 每次全量装 | `--system-site-packages` 复用镜像自带 |
| `libascend_hal.so` / `torch_npu` 崩溃 | amd64 节点无 Ascend 驱动 | `TORCH_DEVICE_BACKEND_AUTOLOAD=0` |
| `moby/buildkit` 拉取超时 | Compose Bake/BuildKit | `DOCKER_BUILDKIT=0`、`COMPOSE_BAKE=false` |
| `ubuntu:24.04` apt `NO_PUBKEY` | 旧 dind 下的 apt 签名校验 | 升级 dind 27 后彻底解决，补丁已删 |
| dind 拉 docker.io 超时 | 无 Hub 访问 | `--registry-mirror https://docker.m.daocloud.io`（生效） |
| uv 下载失败（astral/GitHub release 不可达） | GitHub 资产不可达 | uv 二进制走 OBS 直链 |
| `Failed to fetch files.pythonhosted.org` | uv 未走镜像 | `uv tool install --index-url <tuna>` |
| agent 0 请求、router 收不到 | pier `allow_internet=false` → squid 只放行 80/443 + 域名 | 运行时补丁强制 `allow_internet=True` |
| `PD KV layout mismatch (prefill=4096 decode=32768)` | mooncake 注册校验 + DSPARK + prefill PP>1 | 传输后端固定 `ascend` |
| 首任务构建每步 ~10 分钟 | dind 仅 2 CPU | limits 提到 8C/8Gi |
| trajectory / reward / model.patch “写了但不存在” | bind source 在 dind 容器不可见 | dind 同路径挂载 `share` 到 `/root/.cache` |
| agent 3 次请求后退出 | 无 tool-call 解析器，模型返回纯文本 → 连续 FormatError | 加 `--tool-call-parser deepseekv4 --reasoning-parser deepseek-v4` |
| accuracy 0.0 | agent 被 90min 预算掐断（见 §7） | 待定（保预算 / 提吞吐 / 换任务） |
| trial 日志 dump 不触发 | glob 少了 `<timestamp>/` 层级 | `**/deep_swe_jobs/**/...` |

---

## 7. 单任务耗时与超时分析

`superjson-error-stack-serialization`（run `45975067300`）：

| 项 | 数值 |
|---|---|
| `task.toml` `[agent].timeout_sec` | **5400s（90min，官方预算）** |
| agent 执行区间 | 10:19:57 → 11:49:57（正好 5400s，被 `asyncio.wait_for` 掐断） |
| 环境构建（拉镜像 + 装 agent 层） | 10:00:31 → 10:19:56（~20min） |
| 模型调用 | 80 次，全部 200；耗时合计 **2896s ≈ 48min** |
| 请求延迟 | min 2.7s / median 8.3s / p90 55.6s / **max 524s** |
| 输出 token | 49,694（单次最大 8,807） |
| 输入 token | 3,924,094（上下文 1,969 → **83,306**） |
| 有效解码速度 | ≈ 17 tok/s |
| 工具执行 + 其他 | ≈ 42min |

结论：90 分钟主要被「少数超长生成（写整文件，最长 8.7min）+ 上下文膨胀到 83k」吃掉；不是卡死。`limit=10` 会远超 workflow `timeout-minutes: 300`。

---

## 8. 关键配置

`test_npu_accuracy_utils.py`：

| 常量 | 值 |
|---|---|
| `DEEPSWE_EVAL_IMAGE` | `swr.cn-southwest-2.myhuaweicloud.com/base_image/dockerhub/lmsysorg/sglang:cann9.2.0-a3-B050` |
| `DEEPSWE_DIND_IMAGE` | `swr.cn-southwest-2.myhuaweicloud.com/base_image/dockerhub/library/docker:27-dind` |
| `DEEPSWE_DIND_REGISTRY_MIRRORS` | `("https://docker.m.daocloud.io",)` |
| dind 资源 | requests 2C/4Gi，limits 8C/8Gi |
| `DEEPSWE_EVAL_NODE_ARCH` | `amd64`（可用 env 覆盖） |
| `DEEPSWE_EVAL_SCHEDULE_TIMEOUT` | 1200s |
| `DEEPSWE_EVAL_RUN_TIMEOUT` | 14400s |
| `DEEPSWE_EVAL_ACTIVE_DEADLINE` | schedule + run + 900s（Pod 级兜底） |
| `REMOTE_EVAL_DATASETS` | `{"deep_swe"}` |

评测 Pod 环境（`_build_deepswe_eval_env`）：`DOCKER_BUILDKIT=0`、`COMPOSE_BAKE=false`、`TORCH_DEVICE_BACKEND_AUTOLOAD=0`、`DOCKER_HOST=tcp://127.0.0.1:2375`、`HF_ENDPOINT`、`EVALSCOPE_PIP_MIRROR`、`DEEPSWE_UV_TARBALL_URL`（OBS）、`DEEPSWE_UV_INDEX_URL`（tuna）。

用例参数：`datasets=["deep_swe"]`、`limit=1`、`timeout=7200`、`eval_batch_size=1`、`max_retries=1`、`accuracy=None`。

用例部署（已对齐 128k1k 性能用例）：

- 传输后端 `DISAGGREGATION_TRANSFER_BACKEND = "ascend"`（唯一有意差异，性能用例是 mooncake-if-installed）；
- 保留 `--tool-call-parser deepseekv4 --reasoning-parser deepseek-v4`（agent 需要，性能用例不需要）；
- 不写 `--disaggregation-bootstrap-port`（多机 harness 注入 `8995 + pod_index`）；
- prefill `tp16`（无 pp）、`attn-dp 4`、`cp-strategy zigzag`、`chunked-prefill 32768`、`mem-fraction 0.68`、`--init-expert-location <performance>/expert-layout-ais-replicas16.json`；
- decode `tp16`、`attn-dp 16`、`cuda-graph-bs 1,2,3,4,6,7,8,10`、`swa-full-tokens-ratio 0.05`；
- 两侧 DSPARK 含 `--speculative-dspark-block-size 5`。

### trial 日志抓取

`run_deepswe_eval.py` 在 evalscope 结束后（无论成败）按以下 glob 打印并保存到 `deepswe_eval.log`（也会进 artifact）：

```
**/deep_swe_jobs/**/agent/mini-swe-agent.txt
**/deep_swe_jobs/**/verifier/test-stdout.txt
**/deep_swe_jobs/**/verifier/run.log
**/deep_swe_jobs/**/exception.txt
```

---

## 9. 下一步

1. **推送** `4b23625500` + `770478dc5b`，再跑一轮验证性能对齐后的耗时/吞吐（关注 17 tok/s 是否提升、agent 是否能在 90min 内完成）。
2. 决定后续策略：
   - 保持官方 5400s（结果即真实能力）；
   - 或给 evalscope `AgentConfig.override_timeout_sec` 打补丁放宽（改变基准语义，CI 时间上升）；
   - 或提升解码吞吐 / 选更小任务做 smoke。
3. 正式 `limit`：单任务 ~2h15m，`limit>1` 需并行或加长 step 超时。
4. 首次拿到非零结果后回填用例 `accuracy` 基线。
5. 合并前移除 nightly 的 `pull_request` 触发；更新 PR 描述。
6. 可选：性能对齐后 prefill 已是 pp1 + DSPARK，可再试 mooncake 后端（之前 pp>1 触发 KV layout 拒绝）。

---

## 10. 排查手册

- 评测 Pod 名：`<router Job 前缀>-deepswe-eval-<6位hex>`，label `app=sgl-deepswe-eval`
  ```bash
  kubectl get pods -n <NAMESPACE> | grep deepswe-eval
  kubectl describe pod <name> -n <NAMESPACE>
  kubectl logs <name> -c eval -n <NAMESPACE>
  kubectl exec <name> -c eval -- docker ps -a      # 构建中间容器
  kubectl exec <name> -c eval -- docker images
  ```
- CI 日志里看进度与试次：
  - `[deepswe-eval] [diag] +Ns docker ps/images: ...`（心跳）
  - `[deepswe-eval] ===== trial log: ...`（失败时自动 dump）
  - `[DEEPSWE_METRICS] {...}` / `DeepSWE accuracy: ...`
- CI artifact：`npu-metrics-<case>-attempt-N`
  ```bash
  gh run download <run_id> -R sgl-project/sglang -n <artifact> -D out
  ```
  含 `metrics.json`、`deepswe_eval.log`、`deepswe_evalscope/<ts>/{predictions,reports}`、`deep_swe_jobs/<task>/<trial>/{agent,verifier,artifacts}`、`node_logs/{router,prefill,decode}`。
- 判读顺序：`node_logs` 看 router `chat/completions` 状态 → `verifier/test-stdout.txt` 看测试 → `agent/mini-swe-agent.trajectory.json` 看 `exit_status`/usage。

---

## 11. 运行前提 checklist

- [x] 可调度的 amd64 节点（`kubernetes.io/arch=amd64`），允许 `privileged` Pod
- [x] 集群可拉取 dind / eval 镜像（amd64）
- [x] dind 可通过 daocloud mirror 访问 docker.io；任务镜像走 `public.ecr.aws`
- [x] 评测 Pod 可出网：pypi(tuna)、ModelScope（数据集快照）、OBS（uv 包）
- [x] 模型部署：传输后端 ascend，OpenAI tool-call/reasoning parser 已配置
- [ ] 超时配置：`limit=1` 尚可；扩量前需统一（见 §9）
