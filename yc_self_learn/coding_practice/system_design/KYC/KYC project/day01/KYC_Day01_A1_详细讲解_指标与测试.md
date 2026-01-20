# Day 1｜指标体系详细讲解：L0/L1/L2、错误预算、测试设计

**基于项目**：KYC Identity Verification System (PoV)  
**目标**：深入理解三层指标体系、错误预算，以及如何设计测试来测量 p95/p99

---

## 📊 第一部分：什么是 L0/L1/L2 三层指标？

### 核心理念

**三层指标不是随意的分类，而是按照"谁关心、关心什么"来划分的**：

- **L0 稳定性**：运维/On-Call 关心 → "系统现在健康吗？"
- **L1 业务收益**：业务/PM 关心 → "系统创造价值了吗？"
- **L2 长期健康**：工程/Arch 关心 → "系统能持续进化吗？"

---

### L0 稳定性（监控/可用/延迟）- "系统现在健康吗？"

**定义**：实时监控系统是否正常工作的指标。

**谁关心**：
- On-Call 工程师（半夜告警时最关心）
- 运维团队（日常监控）
- 用户（直接感知到的问题）

**核心指标**：

#### 1. 成功率（Success Rate）

**定义**：正常响应的请求数 / 总请求数

**KYC 项目示例**：
- **计算方式**：`_summary.json` 中 `status: "success"` 的文档数 / 总文档数
- **目标**：99%（Production）
- **当前值**：95%（PoV 阶段）
- **告警阈值**：< 98% 触发告警

**为什么重要**：
- 直接反映系统可用性
- 用户最直观的感受
- On-Call 最关心的指标

#### 2. 延迟指标（Latency）- p50/p95/p99

**定义**：请求从发起到响应完成的时间。

**分位数解释**：

```
p50 (中位数) = 50% 的请求都在这个时间以内
p95 = 95% 的请求都在这个时间以内
p99 = 99% 的请求都在这个时间以内

例如：
- p50 = 3秒：50% 的请求在 3 秒内完成
- p95 = 8秒：95% 的请求在 8 秒内完成（5% 的请求超过 8 秒）
- p99 = 15秒：99% 的请求在 15 秒内完成（1% 的请求超过 15 秒）
```

**为什么用 p95/p99 而不是平均值？**

```
假设 100 个请求的延迟：
[1s, 2s, 3s, ..., 99s, 100s]

平均值 = 50.5 秒（被极端值拉高）
p95 = 95 秒（更真实反映大部分用户的体验）
p99 = 99 秒（捕获最坏情况的用户体验）
```

**平均值的问题**：
- 被极端值（outliers）拉高
- 无法反映"大部分用户的真实体验"
- 无法捕获"长尾问题"（那 5% 或 1% 的用户体验）

**KYC 项目示例**：
- **p50**：3-5 秒（单文档处理时间的中位数）
- **p95**：8-10 秒（SLO 目标：< 15 秒）
- **p99**：15-20 秒（SLO 目标：< 30 秒）

**组成分析**（单文档处理时间）：
- Preprocess（图片预处理）：100-200ms
- Rate Limiter Acquire：0-1000ms（如果有排队）
- Fireworks API Call：2000-8000ms（模型推理时间）
- Schema Validation：50-100ms
- Deterministic Rules：10-50ms
- Save Result：20-50ms

**告警阈值**：
- p95 > 15 秒 → Warning（触发告警）
- p99 > 30 秒 → Critical（立即回滚）

##### A1_a1：分位数指标的设计原理

**为什么使用分位数而不是平均值？**

**问题 1：平均值被极端值"拉高"**

```
假设 100 个请求的延迟（秒）：
[1, 2, 3, 4, 5, ..., 98, 99, 1000]  ← 最后一个请求异常慢（1000秒）

平均值 = (1+2+3+...+99+1000) / 100 = 59.5 秒
p95 = 95 秒（排除最慢的 5%）
p99 = 99 秒（排除最慢的 1%）

问题：平均值 59.5 秒被一个 1000 秒的异常请求拉高了
结论：平均值无法反映"大部分用户的真实体验"
```

**问题 2：长尾问题（Long Tail Problem）**

**什么是长尾问题？**
- 大部分请求很快（1-5秒）
- 少数请求很慢（10-30秒）
- 这"少数慢请求"就是"长尾"

**为什么长尾重要？**
```
假设 1000 个用户：
- 950 个用户：1-5 秒（体验好）
- 45 个用户：10-20 秒（体验差）
- 5 个用户：30+ 秒（体验极差）

平均值 = 3.5 秒（看起来很好）
p95 = 15 秒（捕获了 45 个慢用户）
p99 = 25 秒（捕获了 5 个极慢用户）

结论：p95/p99 能捕获"长尾问题"，平均值不能
```

**分位数的实际意义**：

- **p50（中位数）**：
  - **意义**：一半用户的体验
  - **用途**：了解"典型用户"的体验
  - **示例**：p50 = 3秒 → 50% 的用户在 3 秒内完成

- **p95**：
  - **意义**：95% 用户的体验（排除最慢的 5%）
  - **用途**：SLO 目标（Service Level Objective）
  - **示例**：p95 = 8秒 → 95% 的用户在 8 秒内完成，5% 的用户超过 8 秒

- **p99**：
  - **意义**：99% 用户的体验（捕获最慢的 1%）
  - **用途**：捕获"极端情况"（长尾问题）
  - **示例**：p99 = 15秒 → 99% 的用户在 15 秒内完成，1% 的用户超过 15 秒

**如何计算分位数？**

```python
import numpy as np

def calculate_percentiles(latencies):
    """
    计算 p50/p95/p99
    
    Args:
        latencies: 延迟列表（单位：秒）
    
    Returns:
        {"p50": 3.5, "p95": 8.2, "p99": 15.0}
    """
    
    # 使用 numpy 计算分位数
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    
    return {
        "p50": p50,
        "p95": p95,
        "p99": p99
    }

# 示例
latencies = [1.2, 2.5, 3.1, 3.8, 4.2, 5.0, 6.5, 8.0, 10.0, 15.0]
percentiles = calculate_percentiles(latencies)
# 结果：{"p50": 4.2, "p95": 10.0, "p99": 15.0}
```

**分位数的计算原理**：

```
步骤 1：排序
latencies = [1.2, 2.5, 3.1, 3.8, 4.2, 5.0, 6.5, 8.0, 10.0, 15.0]
sorted = [1.2, 2.5, 3.1, 3.8, 4.2, 5.0, 6.5, 8.0, 10.0, 15.0]

步骤 2：计算索引
p50 索引 = 50% × 10 = 5.0 → 取第 5 和第 6 个值的平均 = (4.2 + 5.0) / 2 = 4.6
p95 索引 = 95% × 10 = 9.5 → 取第 9 和第 10 个值的插值 = 10.0 + 0.5 × (15.0 - 10.0) = 12.5
p99 索引 = 99% × 10 = 9.9 → 取第 9 和第 10 个值的插值 = 10.0 + 0.9 × (15.0 - 10.0) = 14.5

步骤 3：返回结果
p50 = 4.6 秒（50% 的请求在 4.6 秒内完成）
p95 = 12.5 秒（95% 的请求在 12.5 秒内完成）
p99 = 14.5 秒（99% 的请求在 14.5 秒内完成）
```

##### A1_a2：请求的底层实现（TCP/HTTP）

**网络协议栈（Protocol Stack）**

**KYC 项目的请求流程**：

```
应用层（Application Layer）
    ↓ HTTP/HTTPS
传输层（Transport Layer）
    ↓ TCP（可靠传输）
网络层（Network Layer）
    ↓ IP（路由）
数据链路层（Data Link Layer）
    ↓ Ethernet/WiFi
物理层（Physical Layer）
    ↓ 光缆/网线
```

**HTTP 请求的完整流程**：

**KYC 项目中的 HTTP 请求**（基于代码分析）：

```python
# KYC 项目调用 Fireworks API 的流程

1. 应用层（Python 代码）
   ↓
   requests.post(url, json=data)  # 或 urllib.request.urlopen()
   ↓
2. HTTP 层
   - 构建 HTTP 请求：
     POST /v1/chat/completions HTTP/1.1
     Host: api.fireworks.ai
     Content-Type: application/json
     Authorization: Bearer <api_key>
     Content-Length: 1234
     
     {"model": "qwen2.5-vl-32b", "messages": [...]}
   ↓
3. TLS/SSL 层（HTTPS）
   - 建立 TLS 连接（加密）
   - 证书验证
   - 密钥交换
   ↓
4. TCP 层
   - 建立 TCP 连接（三次握手）
   - 数据分段（Segment）
   - 可靠传输（确认、重传）
   ↓
5. IP 层
   - 路由选择
   - 数据包（Packet）
   ↓
6. 物理层
   - 通过网线/光缆传输
```

**TCP 连接详解**：

**TCP 三次握手**（建立连接）：

```
客户端（KYC 系统）                   服务器（Fireworks API）
     |                                      |
     |--- SYN (seq=x) -------------------->|
     |                                      |
     |<-- SYN-ACK (seq=y, ack=x+1) --------|
     |                                      |
     |--- ACK (seq=x+1, ack=y+1) -------->|
     |                                      |
     |     TCP 连接建立成功                 |
```

**TCP 数据传输**：

```
客户端                             服务器
  |                                  |
  |--- HTTP Request (数据包) ------->|
  |                                  |
  |<-- ACK (确认收到) --------------|
  |                                  |
  |<-- HTTP Response (数据包) -------|
  |                                  |
  |--- ACK (确认收到) -------------->|
```

**TCP 四次挥手**（关闭连接）：

```
客户端                             服务器
  |                                  |
  |--- FIN (关闭请求) -------------->|
  |                                  |
  |<-- ACK (确认) -------------------|
  |                                  |
  |<-- FIN (关闭请求) ---------------|
  |                                  |
  |--- ACK (确认) ------------------>|
  |                                  |
  |     TCP 连接关闭                 |
```

**KYC 项目的实际代码**：

```python
# KYC 项目使用 urllib 或 requests 库
# 底层都是 TCP + HTTP

def http_request(url, json=None, api_key=None):
    """HTTP 请求（底层使用 TCP）"""
    
    # 1. 构建 HTTP 请求头
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Authorization": f"Bearer {api_key}"
    }
    
    # 2. 创建 HTTP 请求对象
    req = urllib.request.Request(url, headers=headers, method="POST")
    
    # 3. 发送数据（JSON）
    data = bytes(json.dumps(json), encoding="utf-8")
    
    # 4. 发送请求（底层：TCP 连接 + HTTP 协议）
    resp = urllib.request.urlopen(req, data=data)
    # ↑ 这一步会：
    #   - 建立 TCP 连接（三次握手）
    #   - 发送 HTTP 请求
    #   - 接收 HTTP 响应
    #   - 关闭 TCP 连接（四次挥手）
    
    return resp
```

**延迟的组成部分**：

**单次 HTTP 请求的延迟分解**：

```
总延迟 = TCP 连接建立 + HTTP 请求发送 + 服务器处理 + HTTP 响应接收

1. TCP 连接建立（三次握手）
   - 延迟：10-50ms（本地）或 100-300ms（跨地区）
   - 如果使用 HTTP Keep-Alive：可以复用连接，延迟 = 0ms

2. TLS/SSL 握手（HTTPS）
   - 延迟：50-200ms（首次连接）
   - 如果使用 TLS Session Resumption：延迟 = 0ms

3. HTTP 请求发送
   - 延迟：取决于请求大小和网络带宽
   - 示例：1KB 请求，100Mbps 带宽 → 0.08ms

4. 服务器处理（Fireworks API）
   - 延迟：2000-8000ms（模型推理时间）
   - 这是 KYC 项目的主要延迟来源

5. HTTP 响应接收
   - 延迟：取决于响应大小和网络带宽
   - 示例：10KB 响应，100Mbps 带宽 → 0.8ms

6. TCP 连接关闭（四次挥手）
   - 延迟：10-50ms
   - 如果使用 HTTP Keep-Alive：延迟 = 0ms（不关闭连接）
```

**KYC 项目的实际延迟分解**（来自文档）：

```
单文档处理时间 = 3-10 秒（p50-p95）

组成：
- Preprocess（图片预处理）：100-200ms
- Rate Limiter Acquire：0-1000ms（如果有排队）
- Fireworks API Call：2000-8000ms ← 主要延迟（模型推理）
  ├─ TCP 连接建立：10-50ms
  ├─ TLS 握手：50-200ms（首次）
  ├─ HTTP 请求发送：< 1ms
  ├─ 服务器处理（模型推理）：2000-8000ms ← 最大延迟
  └─ HTTP 响应接收：< 10ms
- Schema Validation：50-100ms
- Deterministic Rules：10-50ms
- Save Result：20-50ms
```

##### A1_a3：这些底层协议是自动的吗？

**重要说明：开发者不需要从零实现这些协议！**

**这些协议由谁实现？**

1. **TCP/IP 协议**：
   - **实现者**：操作系统内核（Windows/Linux/macOS）
   - **开发者做什么**：什么都不用做，操作系统自动处理
   - **示例**：当你调用 `requests.post()` 时，操作系统自动处理 TCP 三次握手、数据传输、四次挥手

2. **TLS/SSL 协议**：
   - **实现者**：Python 的 `ssl` 库（基于 OpenSSL）
   - **开发者做什么**：使用 `https://` URL，库自动处理加密
   - **示例**：`requests.post("https://api.fireworks.ai/...")` 自动使用 TLS 加密

3. **HTTP 协议**：
   - **实现者**：Python 的 `requests` 或 `urllib` 库
   - **开发者做什么**：调用库的 API，库自动构建 HTTP 请求
   - **示例**：`requests.post(url, json=data)` 自动构建 HTTP 请求头

**实际开发中的代码**：

```python
# KYC 项目的实际代码（非常简单！）

import requests

# 只需要这一行代码！
response = requests.post(
    url="https://api.fireworks.ai/v1/chat/completions",
    json={"model": "qwen2.5-vl-32b", "messages": [...]},
    headers={"Authorization": "Bearer <api_key>"}
)

# 底层发生了什么（自动处理，你不需要写代码）：
# 1. 操作系统：TCP 三次握手（自动）
# 2. ssl 库：TLS 握手（自动）
# 3. requests 库：构建 HTTP 请求（自动）
# 4. 操作系统：TCP 数据传输（自动）
# 5. requests 库：解析 HTTP 响应（自动）
# 6. 操作系统：TCP 四次挥手（自动）

# 你只需要关心业务逻辑，不需要关心底层协议！
```

**为什么还要理解这些原理？**

虽然不需要从零实现，但理解原理很重要：

1. **问题排查**：
   ```
   问题：请求很慢（10秒）
   
   如果不理解 TCP/HTTP：
   - 不知道是网络问题还是服务器问题
   - 不知道是连接建立慢还是数据传输慢
   
   如果理解 TCP/HTTP：
   - 可以分析：TCP 连接建立慢？TLS 握手慢？服务器处理慢？
   - 可以定位问题：网络延迟？服务器负载高？
   ```

2. **性能优化**：
   ```
   优化 1：使用 HTTP Keep-Alive
   - 复用 TCP 连接，避免每次请求都建立连接
   - 节省 10-50ms 的连接建立时间
   
   优化 2：使用连接池
   - 复用多个 TCP 连接
   - 提高并发性能
   ```

3. **系统设计**：
   ```
   设计决策：
   - 什么时候使用 HTTP Keep-Alive？
   - 什么时候使用连接池？
   - 如何设置超时时间？
   - 如何处理网络错误？
   
   理解底层原理 → 做出更好的设计决策
   ```

**实际开发中的最佳实践**：

```python
# KYC 项目的实际代码（使用连接池优化）

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# 创建 Session（复用 TCP 连接）
session = requests.Session()

# 配置重试策略
retry_strategy = Retry(
    total=3,  # 最多重试 3 次
    backoff_factor=1,  # 指数退避
    status_forcelist=[429, 500, 502, 503, 504]  # 这些状态码会重试
)

# 配置连接池
adapter = HTTPAdapter(
    max_retries=retry_strategy,
    pool_connections=10,  # 连接池大小
    pool_maxsize=10
)

session.mount("https://", adapter)

# 使用 Session 发送请求（自动复用 TCP 连接）
response = session.post(
    url="https://api.fireworks.ai/v1/chat/completions",
    json={"model": "qwen2.5-vl-32b", "messages": [...]},
    headers={"Authorization": "Bearer <api_key>"},  # ← headers 是什么？
    timeout=30  # 超时时间
)

# 底层自动处理：
# - TCP 连接建立（首次请求）
# - TCP 连接复用（后续请求，节省时间）
# - TLS 加密（自动）
# - HTTP 请求构建（自动）
# - 错误重试（自动）
```

**总结**：

| 协议层 | 谁实现？ | 开发者需要做什么？ |
|--------|---------|------------------|
| **TCP/IP** | 操作系统内核 | 什么都不用做，自动处理 |
| **TLS/SSL** | Python `ssl` 库 | 使用 `https://` URL，自动加密 |
| **HTTP** | Python `requests`/`urllib` 库 | 调用库的 API，自动构建请求 |
| **应用层** | **你自己** | 写业务逻辑代码 |

**关键点**：
- ✅ **不需要**从零实现 TCP/IP/TLS/HTTP
- ✅ **只需要**使用高级库（`requests`、`urllib`）
- ✅ **但需要**理解原理，用于问题排查和性能优化

##### A1_a4：HTTP Headers 是什么？

**Headers（请求头）是什么？**

**简单理解**：Headers 是 HTTP 请求的"元数据"，告诉服务器"这个请求的额外信息"。

**类比**：
```
写信：
- 信封上的地址 = URL（去哪里）
- 信的内容 = Body（数据）
- 邮票、寄件人 = Headers（元数据）
```

**HTTP 请求的完整结构**：

```
POST /v1/chat/completions HTTP/1.1          ← 请求行（方法 + 路径 + 协议版本）
Host: api.fireworks.ai                       ← Headers（请求头）
Content-Type: application/json               ← Headers
Authorization: Bearer fw-xxxxx               ← Headers（API Key）
Content-Length: 1234                         ← Headers

{"model": "qwen2.5-vl-32b", "messages": [...]}  ← Body（请求体，实际数据）
```

**常见的 Headers**：

| Header 名称 | 作用 | 示例 |
|------------|------|------|
| **Authorization** | 身份验证（API Key） | `Bearer fw-xxxxx` |
| **Content-Type** | 告诉服务器数据格式 | `application/json` |
| **Content-Length** | 数据大小（字节） | `1234` |
| **User-Agent** | 客户端信息 | `Python-requests/2.31.0` |
| **Accept** | 期望的响应格式 | `application/json` |

**KYC 项目中的实际代码**：

```python
import requests

# Headers 是一个字典（Dictionary）
headers = {
    "Authorization": "Bearer fw-xxxxx",  # API Key（身份验证）
    "Content-Type": "application/json"   # 告诉服务器：我发送的是 JSON 格式
}

# 发送请求时，headers 会自动添加到 HTTP 请求头中
response = requests.post(
    url="https://api.fireworks.ai/v1/chat/completions",
    json={"model": "qwen2.5-vl-32b", "messages": [...]},
    headers=headers  # ← 这里传入 headers
)

# 底层发生了什么？
# requests 库会自动将 headers 转换为 HTTP 请求头：
# 
# POST /v1/chat/completions HTTP/1.1
# Host: api.fireworks.ai
# Authorization: Bearer fw-xxxxx          ← 从 headers 字典转换而来
# Content-Type: application/json          ← 从 headers 字典转换而来
# 
# {"model": "qwen2.5-vl-32b", ...}        ← 从 json 参数转换而来
```

**为什么需要 Authorization Header？**

**身份验证**：
- Fireworks API 需要知道"你是谁"
- API Key 就像"密码"，证明你有权限使用这个 API
- 没有 API Key → 服务器拒绝请求（401 Unauthorized）

**示例**：

```python
# ❌ 没有 Authorization Header（会失败）
response = requests.post(
    url="https://api.fireworks.ai/v1/chat/completions",
    json={"model": "qwen2.5-vl-32b", "messages": [...]}
    # 没有 headers → 服务器返回 401 Unauthorized
)

# ✅ 有 Authorization Header（成功）
response = requests.post(
    url="https://api.fireworks.ai/v1/chat/completions",
    json={"model": "qwen2.5-vl-32b", "messages": [...]},
    headers={"Authorization": "Bearer fw-xxxxx"}  # ← 身份验证
    # 服务器验证 API Key → 返回结果
)
```

**Headers 的完整示例**：

```python
# KYC 项目的完整 Headers 示例

headers = {
    # 1. 身份验证（必须）
    "Authorization": "Bearer fw-xxxxx",
    
    # 2. 内容类型（告诉服务器数据格式）
    "Content-Type": "application/json",
    
    # 3. 可选：自定义 Headers
    "X-Request-ID": "trace_abc123",  # 追踪 ID（用于日志）
    "User-Agent": "KYC-Project/1.0"   # 客户端标识
}

response = requests.post(
    url="https://api.fireworks.ai/v1/chat/completions",
    json={"model": "qwen2.5-vl-32b", "messages": [...]},
    headers=headers
)
```

**实际 HTTP 请求（抓包看到的）**：

```
POST /v1/chat/completions HTTP/1.1
Host: api.fireworks.ai
Authorization: Bearer fw-xxxxx
Content-Type: application/json
Content-Length: 1234
User-Agent: Python-requests/2.31.0
X-Request-ID: trace_abc123

{"model": "qwen2.5-vl-32b", "messages": [...]}
```

**总结**：

- **Headers 是什么**：HTTP 请求的"元数据"（额外信息）
- **为什么需要**：
  - 身份验证（Authorization）
  - 告诉服务器数据格式（Content-Type）
  - 传递追踪信息（X-Request-ID）
- **如何设置**：使用 Python 字典 `{"Header-Name": "value"}`
- **底层处理**：`requests` 库自动将字典转换为 HTTP 请求头

##### A1_a5：应用层 vs AI Infra 的区别

**对于 KYC 项目（应用层）：只需要调用 API 和计算指标**

**你的工作**：

```python
# 1. 调用 API（就这么简单！）
import requests
import time

def process_document(image_path):
    start_time = time.perf_counter()
    
    # 调用 Fireworks API
    response = requests.post(
        url="https://api.fireworks.ai/v1/chat/completions",
        json={"model": "qwen2.5-vl-32b", "messages": [...]},
        headers={"Authorization": "Bearer <api_key>"}
    )
    
    end_time = time.perf_counter()
    latency = end_time - start_time
    
    return {
        "result": response.json(),
        "latency_ms": latency * 1000
    }

# 2. 计算指标（从 _summary.json）
import numpy as np

def calculate_p99(latencies):
    return np.percentile(latencies, 99)

# 就这些！不需要关心 TCP/HTTP 底层实现
```

**总结：对于应用层开发者**：
- ✅ 调用 `requests.post()` → 完成
- ✅ 测量延迟 → 完成
- ✅ 计算 p50/p95/p99 → 完成
- ❌ **不需要**实现 TCP/IP/TLS/HTTP
- ❌ **不需要**实现 HTTP 服务器
- ❌ **不需要**实现负载均衡

---

**如果是 AI Infra（基础设施层）：需要更多工作**

**AI Infra 的工作**（SGLang 项目）：

```
应用层（KYC 项目）
    ↓ 调用 API
AI Infra（SGLang）
    ├─ HTTP 服务器（接收请求）
    ├─ 请求路由（Load Balancer）
    ├─ 模型推理引擎（KV Cache、Batching）
    ├─ 资源调度（GPU 分配）
    ├─ 监控系统（Metrics、Logging）
    └─ 性能优化（Prefill/Decode 分离）
```

**SGLang 项目需要做什么**：

1. **实现 HTTP 服务器**：
   ```rust
   // sgl-router/src/routers/http/router.rs
   // 需要自己实现 HTTP 服务器，接收请求
   ```

2. **实现请求路由**：
   ```rust
   // 需要实现 Load Balancer，分配请求到不同的 Worker
   ```

3. **实现模型推理引擎**：
   ```python
   // 需要实现 KV Cache、Batching、Prefill/Decode 分离
   ```

4. **实现资源调度**：
   ```rust
   // 需要实现 GPU 分配、任务调度
   ```

**对比**：

| 工作内容 | 应用层（KYC 项目） | AI Infra（SGLang） |
|---------|------------------|------------------|
| **HTTP 请求** | 使用 `requests.post()` | 需要实现 HTTP 服务器 |
| **TCP/IP** | 操作系统自动处理 | 操作系统自动处理 |
| **延迟测量** | 测量 API 调用时间 | 测量整个推理流程 |
| **指标计算** | 计算 p50/p95/p99 | 计算 p50/p95/p99 |
| **负载均衡** | 不需要（API 提供商处理） | 需要实现（自己处理） |
| **模型推理** | 调用 API（Fireworks 处理） | 需要实现（自己处理） |

**结论**：

- **对于 KYC 项目（应用层）**：
  - ✅ 只需要 `requests.post()` + 计算 p99
  - ✅ AI Infra 的工作到这里就结束了（因为用的是第三方 API）
  - ✅ 不需要关心底层实现

- **对于 SGLang 项目（AI Infra）**：
  - ❌ 需要实现 HTTP 服务器、路由、推理引擎等
  - ❌ 需要更多底层工作

**你的 KYC 项目**：
```python
# 就这么简单！
response = requests.post(url, json=data)
latency = measure_latency()
p99 = calculate_p99(latencies)

# 完成！不需要更多了
```

##### A1_a4：延迟测量的原理

**如何测量延迟？**

**Python 代码示例**：

```python
import time
import requests

def measure_latency():
    """测量 HTTP 请求的延迟"""
    
    # 开始时间
    start_time = time.perf_counter()  # 或 time.time()（更精确）
    
    # 发送 HTTP 请求
    response = requests.post(
        url="https://api.fireworks.ai/v1/chat/completions",
        json={"model": "qwen2.5-vl-32b", "messages": [...]},
        headers={"Authorization": "Bearer <api_key>"}
    )
    
    # 结束时间
    end_time = time.perf_counter()
    
    # 计算延迟
    latency = end_time - start_time  # 单位：秒
    
    return latency
```

**延迟测量的时间点**：

```
时间轴：
t0: 开始测量（time.perf_counter()）
    ↓
t1: TCP 连接建立完成
    ↓
t2: TLS 握手完成（如果是 HTTPS）
    ↓
t3: HTTP 请求发送完成
    ↓
t4: 服务器开始处理
    ↓
t5: 服务器处理完成（模型推理完成）
    ↓
t6: HTTP 响应接收完成
    ↓
t7: 结束测量（time.perf_counter()）

总延迟 = t7 - t0
```

**KYC 项目的延迟测量**（基于 `_summary.json` 结构）：

```python
# KYC 项目在 pipeline 中测量延迟

import time

def process_document(image_path):
    """处理单个文档（测量延迟）"""
    
    # 开始时间
    start_time = time.perf_counter()  # 高精度计时器
    
    try:
        # 1. 预处理
        preprocessed = preprocess_image(image_path)
        
        # 2. 调用 Fireworks API（HTTP 请求）
        result = call_fireworks_api(preprocessed)
        # ↑ 这一步包含：
        #   - TCP 连接建立
        #   - TLS 握手（HTTPS）
        #   - HTTP 请求发送
        #   - 服务器处理（模型推理）
        #   - HTTP 响应接收
        
        # 3. 验证和保存
        validated = validate_result(result)
        save_result(validated)
        
        status = "success"
        
    except Exception as e:
        status = "fail"
        error_code = str(e)
    
    # 结束时间
    end_time = time.perf_counter()
    
    # 计算延迟（毫秒）
    latency_ms = (end_time - start_time) * 1000
    
    return {
        "status": status,
        "latency_ms": latency_ms,
        "result": result if status == "success" else None
    }
```

#### 3. 错误率（Error Rate）

**定义**：错误响应的请求数 / 总请求数

**KYC 项目示例**：
- **目标**：< 1%（Production）
- **当前值**：5%（PoV 阶段）
- **错误分类**（基于 `src/errors.py`）：
  - `IMAGE_FORMAT_UNSUPPORTED`：1%
  - `SCHEMA_VALIDATION_FAILED`：2%
  - `API_TIMEOUT`：1%
  - `RATE_LIMIT_EXCEEDED`：1%
  - > **设计说明**（如何做、为何这些类）：见 [A1_B1：错误分类的设计](./KYC_Day01_A1_B1_error_classification.md)。

**告警阈值**：
- Error Rate > 2% → Warning
- Error Rate > 5% → Critical（立即回滚）
- > **设计说明**（为什么不是所有错误都告警）：见 [A1_B3：告警阈值设计](./KYC_Day01_A1_B3_alert_threshold.md)。
- > **设计说明**（重试如何影响错误率、如何规避概率累积）：见 [A1_B4：重试与错误率](./KYC_Day01_A1_B4_retry_error_rate.md)。
- > **设计说明**（如何平衡 Error Rate 与 fragile system）：见 [A1_B5：校验严格度权衡](./KYC_Day01_A1_B5_validation_tradeoff.md)。
- > **设计说明**（分层容错设计：如何通过多层 99% 通过率保证系统不 down，类比 Chrome 的设计思路）：见 [A1_B6：分层容错设计](./KYC_Day01_A1_B6_layered_fault_tolerance.md)。
- > **设计说明**（错误影响分级：什么时候会造成公司/客户/人力损失，什么时候触发 on-call）：见 [A1_B7：错误影响分级与损失评估](./KYC_Day01_A1_B7_error_impact_classification.md)。
- > **设计说明**（如何用 A/B Test 验证设计思路，特别是用户体验阈值）：见 [A1_B8：A/B Test 验证设计思路](./KYC_Day01_A1_B8_ab_testing_validation.md)。

#### 4. 回退率（Fallback Rate）

**定义**：触发降级/回退的请求数 / 总请求数

**KYC 项目示例**：
- **当前值**：0%（PoV 阶段无降级）
- **未来规划**：低质量图片 → OCR-only fallback

#### 5. 可用性（Availability）

**定义**：系统可用的时间 / 总时间

**计算方式**：
- 99% = 每月 < 7.3 小时不可用
- 99.9% = 每月 < 43 分钟不可用
- 99.99% = 每月 < 4.3 分钟不可用

**KYC 项目示例**：
- **目标**：99.9%（Production = 每月 < 43 分钟不可用）
- **当前值**：99%（PoV 阶段）

**SLA vs SLO vs SLI**：
- **SLA**（Service Level Agreement）：对用户的承诺（如 99.9% 可用性）
- **SLO**（Service Level Objective）：内部目标（如 99.95% 可用性，比 SLA 更严格）
- **SLI**（Service Level Indicator）：实际测量的指标（如当前的 99% 可用性）

---

### L1 业务收益（ROI）- "系统创造价值了吗？"

**定义**：系统为业务创造的实际价值（省钱、省时、降低风险）。

**谁关心**：
- 业务团队/PM（证明系统的价值）
- 财务部门（成本分析）
- 决策层（是否继续投资）

**核心指标**：

#### 1. 每单节省的人审分钟数

**KYC 项目示例**：
- **基线**：5-10 分钟/单（人工审核一张 ID 文档）
- **当前值**：3-5 秒/单（AI 处理时间）
- **节省**：5 分钟/单
- **ROI**：`5 分钟/单 × 1000 单/月 = 5000 分钟/月 = 83 小时/月`
- **节省效率**：`> 99%` 的时间节省

**为什么重要**：
- 直接证明系统的价值
- 可以计算人力成本节省
- 业务团队最关心的指标

#### 2. 错误拦截率带来的风险降低

**KYC 项目示例**：
- **关键拦截场景**（基于 `src/rules.py` 的 fraud markers）：
  - `expiry:expired`：过期文档拦截
  - `missing_critical:full_name,date_of_birth`：关键字段缺失拦截
  - `low_confidence_critical:document_number`：低置信度关键字段拦截
- **基线**：0%（无自动化拦截）
- **当前值**：15-20%（PoV 阶段）
- **风险降低估算**：200 件/月（避免的潜在合规风险）

**为什么重要**：
- 降低合规风险（KYC 是强监管领域）
- 减少人工审核的漏检
- 降低潜在的法律风险

#### 3. 吞吐提升带来的成本节省

**KYC 项目示例**：
- **成本分解**：
  - **Fireworks API 调用**：$0.001-0.002 / request（Qwen2.5-VL-32B）
  - **人工审核成本**：$5-10 / request（按小时工资计算）
- **基线成本**：$7.5 / request（平均人工成本）
- **当前成本**：$0.0015 / request（AI 成本）
- **节省**：`$7.5 / request × 1000 requests/月 = $7500 / 月`
- **ROI 倍数**：`5000x` 成本降低

**为什么重要**：
- 直接的成本节省
- 可以量化 ROI
- 财务部门最关心的指标

#### 4. 自动化率

**KYC 项目示例**：
- **定义**：无需人工介入的请求数 / 总请求数
- **自动化判断**（基于 `src/rules.py`）：
  - 所有关键字段提取成功（`full_name, date_of_birth, document_number, expiry_date, issuing_country`）
  - 置信度 > 阈值（如 `> 0.85`）
  - 通过确定性规则检查（expiry valid, quality good）
- **目标**：80%（20% 需要人工 review）
- **当前值**：60-70%（PoV 阶段）

**为什么重要**：
- 反映系统的成熟度
- 直接关联人力成本
- 业务团队关心的效率指标

---

### L2 长期健康（可维护/可扩展）- "系统能持续进化吗？"

**定义**：系统能否持续改进、降低维护成本、提高可扩展性的指标。

**谁关心**：
- 工程团队/架构师（系统的可持续性）
- Tech Lead（技术债务管理）
- CTO（技术战略）

**核心指标**：

#### 1. 变更失败率（Change Failure Rate）

**定义**：导致回滚/问题的发布数 / 总发布数

**KYC 项目示例**：
- **目标**：< 5%
- **当前值**：0%（PoV 阶段，尚未有 production releases）
- **基于**：Schema-First 设计（`src/schemas.py`）和确定性规则（`src/rules.py`）降低变更风险

**为什么重要**：
- 反映系统设计的稳定性
- 降低变更的风险成本
- 工程团队关心的稳定性指标

#### 2. 回滚频率（Rollback Frequency）

**定义**：回滚次数 / 总发布次数

**KYC 项目示例**：
- **目标**：< 2%
- **当前值**：0%（PoV 阶段）
- **近30天**：0 次回滚

**为什么重要**：
- 反映发布流程的成熟度
- 降低生产事故风险
- 工程团队关心的发布稳定性

#### 3. 回归门禁通过率（Regression Gate Pass Rate）

**定义**：通过回归测试的发布数 / 总发布数

**KYC 项目示例**：
- **回归测试覆盖**：
  - Unit tests (`tests/test_rules.py`, `tests/test_validators.py`)
  - Schema validation tests (`tests/test_validators.py`)
  - Error handling tests (`tests/test_errors.py`)
- **目标**：> 95%
- **当前值**：100%（PoV 阶段，所有单元测试通过）

**为什么重要**：
- 反映测试覆盖的完整性
- 降低回归问题的风险
- 工程团队关心的质量指标

#### 4. 告警噪音（Alert Noise - Precision）

**定义**：有效告警数 / 总告警数

**KYC 项目示例**：
- **目标**：> 80%（减少误报）
- **当前值**：N/A（PoV 阶段，无生产告警系统）
- **平均告警数/周**：0（PoV 阶段）

**为什么重要**：
- 降低 On-Call 的疲劳（减少误报）
- 提高告警的响应效率
- 运维团队关心的告警质量

#### 5. Toil（重复劳动）趋势

**定义**：每周花在重复性任务上的时间

**KYC 项目示例**：
- **Toil 来源**：
  - 手动运行 batch processing
  - 手动检查 `_summary.json`
  - 手动处理错误文档
- **目标**：< 5 小时/周
- **当前值**：2-3 小时/周（PoV 阶段）
- **趋势**：→（稳定）
- **自动化机会**：CI/CD 集成、自动化 batch scheduling、错误自动重试

**为什么重要**：
- 降低维护成本（重复劳动 = 浪费工程师时间）
- 提高工程效率
- 工程团队关心的效率指标

#### 6. Schema 兼容性

**定义**：Schema 变更导致的 breaking changes 数 / 总 schema 变更数

**KYC 项目示例**：
- **目标**：0 breaking changes（通过 versioning：`schema_version = "v1"`）
- **当前值**：0（PoV 阶段）
- **基于**：`src/schemas.py` 的版本化设计

**为什么重要**：
- 反映 API 设计的稳定性
- 降低下游系统的影响
- 架构师关心的 API 稳定性

#### 7. Auditability 覆盖率

**定义**：包含完整 trace_id 的文档数 / 总文档数

**KYC 项目示例**：
- **目标**：100%
- **当前值**：100%（PoV 阶段）
- **基于**：每个请求都有 `trace_id`（Privacy & Logging section）

**为什么重要**：
- 支持合规审计（KYC 是强监管领域）
- 支持问题定位
- 合规团队关心的可审计性

#### 8. PII 泄漏事件

**定义**：PII 泄漏事件数

**KYC 项目示例**：
- **目标**：0
- **当前值**：0
- **基于**：Logging rules（Never log: base64 image, prompt content, extracted PII fields）

**为什么重要**：
- 降低合规风险（PII 泄漏 = 严重合规问题）
- 降低法律风险
- 合规/安全团队最关心的指标

---

## 🎯 第二部分：什么是错误预算（Error Budget）？

### 核心理念

**错误预算不是"允许失败"，而是"平衡发布速度 vs 稳定性"的控制机制**。

### 错误预算的定义

**Error Budget = 100% - SLO**

**示例**：
- SLO = 99%（成功率）
- Error Budget = 1% = 1000 文档中允许 10 个失败（月度）

### 错误预算的作用

#### 1. 平衡"发布速度 vs 稳定性"

```
如果没有错误预算：
- 产品团队："我们要快速发布新功能！"
- 工程团队："不行，新功能可能引入 bug，影响稳定性！"
- 结果：双方争论不休，决策困难

有了错误预算：
- 错误预算充足（> 50%）→ 可以快速发布（产品团队开心）
- 错误预算不足（< 25%）→ 冻结发布，专注稳定性（工程团队开心）
- 结果：客观的决策机制，双方都能接受
```

#### 2. 量化稳定性成本

```
错误预算 = 稳定性成本

如果错误预算消耗太快：
- 说明系统稳定性有问题
- 需要投入更多资源在稳定性上
- 暂停新功能开发，专注稳定性修复
```

### 错误预算的计算

**L0 稳定性 Error Budget**：

**月度计算**：
- **总请求数**：1000 文档/月
- **SLO**：99%（成功率）
- **Error Budget**：1% = 10 个文档/月 可以失败

**当前状态**（PoV 阶段）：
- **成功率**：95%
- **实际失败数**：50 个文档/月（5% × 1000）
- **Error Budget 消耗**：50 / 10 = 500%（远超预算）

**决策**：
- 当前状态：`冻结`（Error Budget 严重超标）
- 需要：立即修复稳定性问题

### 错误预算的状态

#### 1. 正常状态（Error Budget > 50%）

**状态**：✅ 健康

**决策**：
- ✅ 可以继续发布新功能
- ✅ 可以承担一定风险

**示例**：
- Error Budget 剩余：80%
- 可以快速发布新功能

#### 2. 警告状态（Error Budget 25% - 50%）

**状态**：⚠️ 警告

**决策**：
- ⚠️ 限制高风险发布
- ⚠️ 增加审查流程

**示例**：
- Error Budget 剩余：30%
- 只能发布低风险功能，高风险功能需要更严格的测试

#### 3. 冻结状态（Error Budget < 25%）

**状态**：🛑 冻结

**决策**：
- 🛑 **冻结所有新功能发布**
- 🛑 **只允许稳定性修复和优化**
- 🛑 **全力修复稳定性债务**

**示例**：
- Error Budget 剩余：10%
- 禁止所有新功能发布，只允许 bug fix 和稳定性优化

### KYC 项目的错误预算示例

**月度计算**：
- **总请求数**：1000 文档/月
- **SLO**：99%（成功率）
- **Error Budget**：1% = 10 个文档/月 可以失败

**当前状态**（PoV 阶段）：
- **成功率**：95%
- **实际失败数**：50 个文档/月（5% × 1000）
- **Error Budget 消耗**：50 / 10 = 500%（严重超标）

**决策**：
- 🛑 **冻结发布**，专注稳定性修复
- **优先级**：
  1. 修复 Schema Validation Failures（2% = 20 个文档/月）
  2. 优化错误处理逻辑
  3. 增加重试机制

### 错误预算策略（Error Budget Policy）

**规则**：

1. **错误预算消耗 > 50%** → 正常状态，可以快速发布
2. **错误预算消耗 25% - 50%** → 警告状态，限制高风险发布
3. **错误预算消耗 < 25%** → 冻结状态，禁止新功能发布

**实施**：

```python
# 伪代码示例
def can_release_new_feature(error_budget_remaining_percent):
    if error_budget_remaining_percent > 50:
        return True  # 正常状态，可以发布
    elif error_budget_remaining_percent > 25:
        return "require_strict_review"  # 警告状态，需要严格审查
    else:
        return False  # 冻结状态，禁止发布
```

---

## 🧪 第三部分：如何设计测试来测量 p95/p99？

### 大厂思维：测试的时机和策略

#### 1. 测试的时机（When to Test）

**什么时候测试 p95/p99？**

| 时机 | 测试类型 | 目的 | 频率 |
|------|---------|------|------|
| **开发阶段** | Unit Test | 验证单文档处理时间 | 每次提交 |
| **集成阶段** | Integration Test | 验证 E2E 流程延迟 | 每次 PR |
| **发布前** | Performance Test | 验证 p95/p99 是否达标 | 每次发布前 |
| **生产环境** | Continuous Monitoring | 实时监控 p95/p99 | 实时 |
| **定期** | Load Test | 验证系统容量 | 每月/每季度 |

**KYC 项目的测试时机**：

1. **开发阶段**（每次代码提交）
   - **Unit Test**：`tests/test_pipeline.py`（单文档处理时间）
   - **目的**：快速反馈，验证代码变更没有引入性能退化
   - **频率**：每次 `git commit`

2. **集成阶段**（每次 PR）
   - **Integration Test**：`tests/test_integration.py`（E2E 流程）
   - **目的**：验证完整流程的延迟
   - **频率**：每次 Pull Request

3. **发布前**（每次发布前）
   - **Performance Test**：专门的性能测试脚本
   - **目的**：验证 p95/p99 是否达标（SLO：p95 < 15s, p99 < 30s）
   - **频率**：每次发布前（必须通过才能发布）

4. **生产环境**（实时监控）
   - **Continuous Monitoring**：基于 `_summary.json` 的实时分析
   - **目的**：实时监控生产环境的 p95/p99
   - **频率**：实时（每次 batch 运行后）

5. **定期**（每月/每季度）
   - **Load Test**：压力测试
   - **目的**：验证系统容量（如 1000 文档/小时的处理能力）
   - **频率**：每月/每季度

---

### 2. 测试的设计（How to Design Tests）

#### 测试设计原则

1. **真实场景**：使用真实的数据分布（不是均匀分布）
2. **样本量足够**：至少 100+ 样本才能计算 p95/p99
3. **覆盖边界情况**：包括正常、异常、边界场景
4. **隔离环境**：避免外部因素影响（如网络波动）

#### KYC 项目的测试设计

**测试场景设计**：

```python
# tests/test_performance.py (伪代码示例)

class TestLatency:
    """测试延迟指标（p50/p95/p99）"""
    
    def test_single_document_latency(self):
        """单文档处理时间（Unit Test）"""
        # 测试单个文档的处理时间
        # 目标：p95 < 10s, p99 < 15s
        
        latencies = []
        for i in range(100):  # 100 个样本
            start_time = time.time()
            result = pipeline.process_document(test_image)
            latency = time.time() - start_time
            latencies.append(latency)
        
        # 计算 p95/p99
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)
        
        assert p95 < 10, f"p95 {p95}s exceeds 10s threshold"
        assert p99 < 15, f"p99 {p99}s exceeds 15s threshold"
    
    def test_batch_processing_latency(self):
        """Batch 处理时间（Integration Test）"""
        # 测试 100 个文档的 batch 处理时间
        # 目标：每个文档的平均时间 < 8s
        
        start_time = time.time()
        results = pipeline.process_batch(test_images, batch_size=100)
        total_time = time.time() - start_time
        
        avg_latency_per_doc = total_time / 100
        
        assert avg_latency_per_doc < 8, f"Average latency {avg_latency_per_doc}s exceeds 8s threshold"
    
    def test_p95_p99_distribution(self):
        """测试延迟分布（Performance Test）"""
        # 测试不同场景的延迟分布
        # 正常场景、边界场景、异常场景
        
        scenarios = {
            "normal": 50,  # 50 个正常场景
            "edge": 30,    # 30 个边界场景
            "anomaly": 20  # 20 个异常场景
        }
        
        all_latencies = []
        for scenario, count in scenarios.items():
            for i in range(count):
                image = load_test_image(scenario, i)
                start_time = time.time()
                result = pipeline.process_document(image)
                latency = time.time() - start_time
                all_latencies.append(latency)
        
        # 计算 p50/p95/p99
        p50 = np.percentile(all_latencies, 50)
        p95 = np.percentile(all_latencies, 95)
        p99 = np.percentile(all_latencies, 99)
        
        # 验证是否达标
        assert p50 < 5, f"p50 {p50}s exceeds 5s threshold"
        assert p95 < 15, f"p95 {p95}s exceeds 15s threshold"
        assert p99 < 30, f"p99 {p99}s exceeds 30s threshold"
```

**测试数据设计**：

```python
# 测试数据分布（模拟真实场景）

test_data_distribution = {
    "normal_cases": {
        "count": 50,  # 50% 正常场景
        "examples": ["清晰ID", "标准格式", "高分辨率"]
    },
    "edge_cases": {
        "count": 30,  # 30% 边界场景
        "examples": ["模糊图片", "部分遮挡", "低分辨率"]
    },
    "anomaly_cases": {
        "count": 20,  # 20% 异常场景
        "examples": ["版式变化", "多页文档", "特殊字符"]
    }
}
```

---

### 3. 测试的实现（Implementation）

#### CI/CD 集成（发布前必跑）

```yaml
# .github/workflows/performance_test.yml (示例)

name: Performance Test

on:
  pull_request:
    branches: [main]
  workflow_dispatch:

jobs:
  performance_test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      
      - name: Install dependencies
        run: pip install -r requirements.txt
      
      - name: Run Performance Test
        run: |
          python -m pytest tests/test_performance.py -v
          # 验证 p95 < 15s, p99 < 30s
      
      - name: Performance Test Report
        run: |
          # 生成性能测试报告
          python scripts/generate_performance_report.py
```

#### 生产环境监控（实时）

```python
# scripts/monitor_latency.py (示例)

import json
import numpy as np
from pathlib import Path

def calculate_p95_p99_from_summary(summary_path):
    """从 _summary.json 计算 p95/p99"""
    
    with open(summary_path) as f:
        summary = json.load(f)
    
    latencies = []
    for result in summary.get("results", []):
        if result.get("status") == "success":
            latency = result.get("latency_ms", 0) / 1000  # 转换为秒
            latencies.append(latency)
    
    if len(latencies) < 10:
        print("Not enough samples for p95/p99 calculation")
        return None
    
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    
    print(f"Latency Statistics:")
    print(f"  p50: {p50:.2f}s")
    print(f"  p95: {p95:.2f}s (SLO: < 15s)")
    print(f"  p99: {p99:.2f}s (SLO: < 30s)")
    
    # 检查是否达标
    if p95 > 15:
        print(f"⚠️  WARNING: p95 {p95:.2f}s exceeds SLO threshold 15s")
    if p99 > 30:
        print(f"🛑 CRITICAL: p99 {p99:.2f}s exceeds SLO threshold 30s")
    
    return {"p50": p50, "p95": p95, "p99": p99}

if __name__ == "__main__":
    summary_path = Path("output_results/_summary.json")
    calculate_p95_p99_from_summary(summary_path)
```

---

### 4. 测试的门禁（Release Gate）

**发布前必须通过的性能测试**：

```python
# tests/test_release_gate.py (示例)

class TestReleaseGate:
    """发布门禁：性能测试（Release Gate）"""
    
    def test_p95_latency_gate(self):
        """p95 延迟门禁（必须 < 15s 才能发布）"""
        latencies = self.run_performance_test(sample_size=100)
        p95 = np.percentile(latencies, 95)
        
        assert p95 < 15, f"p95 {p95}s exceeds release gate threshold 15s. Cannot release."
    
    def test_p99_latency_gate(self):
        """p99 延迟门禁（必须 < 30s 才能发布）"""
        latencies = self.run_performance_test(sample_size=100)
        p99 = np.percentile(latencies, 99)
        
        assert p99 < 30, f"p99 {p99}s exceeds release gate threshold 30s. Cannot release."
```

---

## 📝 总结

### 三层指标的关系

```
L0 稳定性（实时监控）
    ↓
L1 业务收益（价值证明）
    ↓
L2 长期健康（可持续发展）
```

### 错误预算的作用

- **平衡"发布速度 vs 稳定性"**
- **量化稳定性成本**
- **客观的决策机制**

### 测试的设计原则

1. **真实场景**：使用真实的数据分布
2. **样本量足够**：至少 100+ 样本
3. **覆盖边界情况**：正常、异常、边界场景
4. **隔离环境**：避免外部因素影响

### 测试的时机

- **开发阶段**：Unit Test（每次提交）
- **集成阶段**：Integration Test（每次 PR）
- **发布前**：Performance Test（每次发布前，Release Gate）
- **生产环境**：Continuous Monitoring（实时）
- **定期**：Load Test（每月/每季度）

---

## 🎯 下一步行动

1. **理解你的 KYC 项目的当前指标**
   - 分析 `_summary.json` 计算当前的 p50/p95/p99
   - 计算当前的 L0/L1/L2 指标

2. **设计性能测试**
   - 创建 `tests/test_performance.py`
   - 实现 p95/p99 测试（至少 100 个样本）

3. **集成到 CI/CD**
   - 添加 Performance Test 到 GitHub Actions
   - 设置 Release Gate（p95 < 15s, p99 < 30s）

4. **建立监控 Dashboard**
   - 从 `_summary.json` 实时计算 p95/p99
   - 设置告警（p95 > 15s → Warning, p99 > 30s → Critical）

---

## 📚 参考

- KYC 项目：https://github.com/Nickcp39/kyc_pov/tree/main
- Google SRE Book: [SLO, SLI, SLAs](https://sre.google/workbook/slo/)
- Error Budget Policy: https://sre.google/workbook/error-budget-policy/
- Performance Testing: https://sre.google/workbook/testing-for-reliability/
