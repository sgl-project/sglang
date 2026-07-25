# SGLang 嵌入式负载上报器

## 概述

在 SGLang Python HTTP/TokenizerManager 进程内实现的 per-worker 负载上报器，通过 gRPC client-streaming 向多个 Router 实时推送调度器负载快照。

**运行约束：**
- 单 tokenizer 模式从 `TokenizerManager.get_loads(include=["core"])` 采集
- 多 tokenizer 模式由唯一的 `MultiTokenizerRouter` 持有 runtime，HTTP worker 通过 IPC 注册并合并刷新通知
- 使用 `grpc.aio.insecure_channel`（h2c，无 TLS/认证）
- `POST /v1/start_reporting` 必须使用 `--admin-api-key` 对应的 Bearer token

### 运行时依赖

负载上报依赖通过各平台 pyproject 中统一的 `load-reporter` optional extra 提供，
不会增加普通 SGLang wheel 的默认依赖。使用该功能的环境应安装：

```bash
pip install "sglang[load-reporter]"
```

该 extra 约束 `grpcio>=1.78.0` 与 `protobuf>=6.31.1,<7`；`test`/`dev` extra
也会安装它，以保证社区 CI 执行负载上报测试。缺少依赖或版本不兼容时普通服务
仍可启动，注册接口返回 HTTP 501。

## 架构

```
FastAPI lifespan
 ├─ 单 tokenizer: LoadReporterRuntime (composition root)
 └─ 多 tokenizer: HTTP worker proxy/notifier ─IPC→ MultiTokenizerRouter
                                           └─ LoadReporterRuntime (唯一 owner)
     ├─ LatestSnapshotStore (atomic latest-wins)
     ├─ ReportBuilder (SnapshotView → protobuf)
     ├─ LoadSampler (single-flight get_loads 循环)
     └─ MonitorManager (MonitorKey → MonitorTask map)
         └─ MonitorTask × N (每个 Router target 一个独立 gRPC stream)
```

**时序边界：**
- Request-end 只刷新 store（同步、非阻塞）
- gRPC 发送只在 stream 初次连接和该 Monitor 自己的 `report_interval_ms` deadline 触发
- Timer 与 request-end 共用一个 sampler 状态机，任一时刻最多一个 `get_loads()` 在飞

## 模块组成

| 文件 | 职责 |
|-----|-----|
| `config.py` | `LoadReporterConfig` / `WorkerMetadata` 从 `ServerArgs` 固化；内部 transport 常量 |
| `store.py` | `LatestSnapshotStore`：校验 `LoadSnapshot`、timestamp fallback、latest-wins、不可变 `SnapshotView` |
| `report_builder.py` | `ReportBuilder`：纯函数转换 `SnapshotView` → `pb.LoadReport`，附加 status 和全局 sequence |
| `sampler.py` | `LoadSampler`：唯一调用 `get_loads()` 的 single-flight 后台 task，coalescing refresh 通知 |
| `registration.py` | 严格 Pydantic schema、`MonitorKey`/`MonitorRegistration` value objects、origin 规范化、`POST /v1/start_reporting` 路由 |
| `monitor.py` | `MonitorManager` (map ownership + identity-safe upsert)、`MonitorTask` (每目标一条 gRPC stream + fixed-rate + lease/reconnect 状态机) |
| `runtime.py` | `LoadReporterRuntime`：顶层对象装配、`start_reporting` 控制面、`notify_request_finished` 同步钩子、有界 `close()` |
| `ipc.py` | 多 tokenizer 的控制请求/响应关联、刷新事件合并及稳定错误映射 |
| `proto/load_monitor.proto` | Canonical `model_gateway.loadmonitor.v1` IDL（字段/enum 值逐字等于 Router 协议，不增减） |

### 重新生成 Python 协议代码

从仓库根目录使用固定工具链生成，确保生成代码兼容项目支持的最低运行时版本：

```bash
codegen_dir=$(mktemp -d /tmp/sglang-load-reporter-codegen.XXXXXX)
python3 -m venv "$codegen_dir/venv"
"$codegen_dir/venv/bin/python" -m pip install \
  grpcio==1.78.0 grpcio-tools==1.78.0 protobuf==6.31.1
cd python/sglang/srt/load_reporter/proto
"$codegen_dir/venv/bin/python" -m grpc_tools.protoc \
  -I. --python_out=. --grpc_python_out=. load_monitor.proto
```

`grpc_tools.protoc` 会为同目录模块生成绝对导入；提交前须将
`import load_monitor_pb2 as load__monitor__pb2` 改为包内相对导入
`from . import load_monitor_pb2 as load__monitor__pb2`。不要删除或修改生成器写入的
protobuf/gRPC 运行时版本校验。

## 控制流

### 启动
1. **Lifespan setup**
   - 单 tokenizer：构造 `LoadReporterRuntime(snapshot_source, server_args)` 并安装 request-finished hook
   - 多 tokenizer：每个 HTTP worker 构造 control proxy / refresh notifier；Router 在首次注册时惰性构造唯一 runtime
   - 写入 `app.state.load_reporter_runtime` / `load_reporter_unsupported_reason`

### 注册与上报
2. **Router 注册**
   - `POST /v1/start_reporting` → `runtime.start_reporting(payload, worker_addr)` → `MonitorManager.upsert`
   - 首次注册：创建 `MonitorTask`，启动 `run()` task；调用 `sampler.activate()`
   - 重复注册（相同 origin）：更新 `MonitorRegistration` (revision++)、lease、interval；唤醒 task 重算 deadline
   - 冲突（不同 origin）：返回 HTTP 409

3. **采集循环** (`LoadSampler`)
   - Activation 后立即刷新；随后 wait(wake_event, timeout=min_interval_ms)
   - Request-end hook 调用 `notify_refresh()` 设置 wake 事件
   - `MonitorManager` interval 改变时调用 `notify_schedule_changed()`
   - 每次 `get_loads()` 完成后，`LatestSnapshotStore.apply_full_snapshot` 原子发布新 view（失败时 `record_error`）
   - Coalescing 规则：采集进行中到达的通知只触发至多一次额外刷新

4. **流发送** (`MonitorTask`)
   - 每 target 一个 `grpc.aio` channel + client stream (`LoadMonitorServiceStub.Report`)
   - (Re)connect 后立即发送当前 snapshot；随后按 `report_interval_ms` 固定速率发送
   - 并发等待：stop event / registration update / lease expiry / call completion / report deadline
   - Update 到达时用 `updated_at + interval` 重新锚定 deadline
   - Write/backpressure 恢复后跳过已错过周期（不补发历史）

5. **重连与错误分类**
   - **Retryable** (`UNAVAILABLE` / `DEADLINE_EXCEEDED` / `RESOURCE_EXHAUSTED`)：exponential backoff (0.25s–5s, ±20% jitter)
   - **Wait-for-renewal** (`INVALID_ARGUMENT` / `UNAUTHENTICATED` / `PERMISSION_DENIED` / `UNIMPLEMENTED`)：记录错误，阻塞等待下一次 registration update (revision 增加)
   - 成功 epoch (至少发送一条) 后重置 backoff 为初始值
   - Lease 到期：task 自行退出，调用 `on_stopped` 从 manager map 移除

### 关闭
6. **Shutdown**
   - HTTP worker lifespan：先 detach hook / IPC 组件，再关闭 proxy 与 notifier
   - 父进程 finally：在 Router event loop 上关闭唯一 runtime，再清理 Router socket
   - `close()` 顺序：sampler.close() → manager.close() (stop 所有 task 并 await 收敛)
   - Timeout 后调用 `cancel_remaining()` 强制取消未收敛的 task

## 配置参数

| `ServerArgs` 字段 | 默认值 | 说明 |
|------------------|-------|-----|
| `load_reporter_snapshot_stale_after_ms` | 3000 | 超过此阈值报告 `REPORT_STATUS_STALE` |
| `load_reporter_zone` | `None` | 可选 zone 元数据（空字符串规范化为 `None`） |

**内部常量** (`config.py`，不生成 CLI 参数)：
- `GRPC_CONNECT_TIMEOUT_SECONDS = 3.0`
- `RECONNECT_INITIAL_SECONDS = 0.25`
- `RECONNECT_MAX_SECONDS = 5.0`
- `SHUTDOWN_TIMEOUT_SECONDS = 5.0`

## 协议约束

- **Wire contract**：`proto/load_monitor.proto` 的字段号/类型/enum 值逐字等于 Router 提供的 canonical IDL，不增加 normalized load / SDK metadata / `worker_id`
- **`Worker.worker_addr`**：从注册 HTTP request origin 规范化（`scheme://host:port`），不读 `Forwarded` / `X-Forwarded-*`
- **`RankLoad.snapshot_time_unix_ms`**：优先使用 `LoadSnapshot.timestamp * 1000`；无效时 fallback 到 `collected_at_unix_ms`
- **Latest-wins 合并**：同一 DP rank 的多次 snapshot 按 timestamp 保留较新值；timestamp 相同时使用本次完整 sample 的 raw metrics
- **Status 逻辑**：
  - `HEALTHY`：所有 rank 的 `(report_time - snapshot_time)` ≤ `snapshot_stale_after_ms`
  - `STALE`：有 rank 超过阈值（report 仍包含 ranks）
  - `UNREACHABLE`：无权威 rank snapshot（store 从未成功 `apply_full_snapshot`）

## 线程与异步模型

- **单 event loop**：Reporter 所有组件与 FastAPI / TokenizerManager 在同一 asyncio loop
- **Single-flight sampler**：任一时刻最多一个 `get_loads()` 在飞；通知只设置 wake event，不创建 task
- **Per-target task**：每 Monitor 一个独立 `asyncio.Task`；manager 不引入 coordinator / reconcile loop / session generation
- **Request-end hook**：同步调用 `sampler.notify_refresh()`（设置 event），不 await / 不创建 task / 不调用 `get_loads()`
- **Error isolation**：Reporter 的采集 / 校验 / 连接 / 写入 / 后台 task / 关闭错误不传播到推理请求或 FastAPI 主生命周期

## 测试与验证

单元测试覆盖 store/builder/sampler/monitor deadline、鉴权、可选依赖边界、IPC 关联与合并、
多 worker 唯一 owner、退出清理和 msgpack round-trip。GPU E2E 验证两个 tokenizer/HTTP
worker 启动后仅建立一条 Router gRPC stream。

## 已知限制

- 无 TLS / mTLS / 认证 / ACK / 重放 / exactly-once / 持久化
- 不自定义 gRPC keepalive / message size（使用 grpcio 默认值）
- Router 协议不支持 SDK metadata / normalized load / `worker_id` 等扩展字段
