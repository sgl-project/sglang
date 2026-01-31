# B04: GPU调度成熟方案与API（K8s简化版）
# B04: GPU Scheduling Solutions and APIs (K8s-based, Simplified)

**作者**：Yanda Cheng  
**项目**：Rad-Linter  
**创建日期**：2025-01-25  
**设计原则**：基于K8s，只做核心功能，简单实用

---

## 📋 目录

1. [K8s原生GPU调度方案](#k8s原生gpu调度方案)
2. [Service负载均衡（K8s原生）](#service负载均衡k8s原生)
3. [GPU资源管理（Device Plugin）](#gpu资源管理device-plugin)
4. [直接可用的API](#直接可用的api)
5. [简化设计说明](#简化设计说明)

---

## K8s原生GPU调度方案

### 核心设计

**基于K8s，只做核心功能，无需复杂Router**

```
┌─────────────────────────────────────────────────────────┐
│  K8s Service (负载均衡，K8s原生)                         │
│  • 自动负载均衡 (Round Robin)                            │
│  • 自动健康检查 (Readiness Probe)                        │
│  • 自动服务发现 (无需额外组件)                           │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│  K8s Scheduler (Pod调度，K8s原生)                         │
│  • 决定Pod在哪个Node运行                                 │
│  • 基于GPU资源可用性                                      │
│  • 自动调度                                               │
└─────────────────────────────────────────────────────────┘
```

**设计原则**：用K8s原生组件，不做复杂设计

---

## Service负载均衡（K8s原生）

### 方案1：Envoy Proxy + Prometheus（生产级）⭐

**这是业界最成熟的方案，直接可用**

#### 核心组件

1. **Envoy Proxy**：负载均衡器（CNCF项目，生产级）
2. **Prometheus**：指标存储和查询
3. **Envoy Admin API**：动态更新权重

#### 直接可用的API

**1. Prometheus Query API**（获取Worker指标）

```python
import requests

# 查询Worker指标
def get_worker_metrics(worker_id):
    prometheus_url = "http://localhost:9090"
    query = f'sglang_inflight_requests{{worker_id="{worker_id}"}}'
    
    response = requests.get(
        f"{prometheus_url}/api/v1/query",
        params={"query": query}
    )
    return response.json()["data"]["result"][0]["value"][1]

# 直接调用
metrics = get_worker_metrics("worker-1")
```

**2. Envoy Admin API**（更新权重）

```python
import requests

# 更新Worker权重
def update_envoy_weight(worker_host, weight):
    envoy_admin_url = "http://localhost:9901"
    
    # 方法1：通过Admin API更新（需要Envoy配置支持）
    response = requests.post(
        f"{envoy_admin_url}/clusters/llm_judge_cluster/update",
        json={
            "host": worker_host,
            "weight": weight
        }
    )
    return response.status_code == 200

# 直接调用
update_envoy_weight("worker-1:30001", 20)
```

**3. Envoy xDS API**（动态配置，推荐）

```python
# 使用Envoy的xDS (x Discovery Service) API
# 这是Envoy官方推荐的动态配置方式

from envoy.config.cluster.v3 import cluster_pb2
from envoy.service.cluster.v3 import cds_pb2_grpc

# 通过gRPC更新集群配置
# 需要Envoy配置支持xDS
```

#### 完整示例（开箱即用）

```python
# router_with_envoy.py
import requests
import time
from prometheus_client import Counter

class EnvoyGPURouter:
    """基于Envoy + Prometheus的GPU Router"""
    
    def __init__(self, prometheus_url="http://localhost:9090", 
                 envoy_admin_url="http://localhost:9901"):
        self.prometheus_url = prometheus_url
        self.envoy_admin_url = envoy_admin_url
    
    def get_worker_score(self, worker_id):
        """从Prometheus获取Worker得分（SECT算法）"""
        # 查询多个指标
        queries = {
            "inflight": f'sglang_inflight_requests{{worker_id="{worker_id}"}}',
            "latency": f'sglang_recent_latency_p95_ms{{worker_id="{worker_id}"}}',
            "free_vram": f'sglang_free_vram_mb{{worker_id="{worker_id}"}}',
            "queue": f'sglang_queue_length{{worker_id="{worker_id}"}}'
        }
        
        metrics = {}
        for key, query in queries.items():
            response = requests.get(
                f"{self.prometheus_url}/api/v1/query",
                params={"query": query}
            )
            if response.json()["status"] == "success":
                metrics[key] = float(response.json()["data"]["result"][0]["value"][1])
        
        # SECT算法计算得分
        base_latency = metrics.get("latency", 3000)
        queue_wait = metrics.get("queue", 0) * base_latency
        load_penalty = metrics.get("inflight", 0) / 16 * 100
        
        score = base_latency + queue_wait + load_penalty
        return score
    
    def select_best_worker(self, workers):
        """选择最佳Worker"""
        scores = {}
        for worker_id in workers:
            scores[worker_id] = self.get_worker_score(worker_id)
        
        # 返回得分最低的Worker
        best_worker = min(scores, key=scores.get)
        return best_worker, scores[best_worker]
    
    def update_envoy_weights(self, workers):
        """更新Envoy权重（基于得分）"""
        scores = {}
        for worker_id in workers:
            scores[worker_id] = self.get_worker_score(worker_id)
        
        # 归一化权重（得分越低，权重越高）
        max_score = max(scores.values())
        weights = {
            worker_id: int(100 * (max_score - score) / max_score)
            for worker_id, score in scores.items()
        }
        
        # 更新Envoy（通过xDS或Admin API）
        # 这里需要根据你的Envoy配置选择方式
        return weights

# 使用示例
router = EnvoyGPURouter()
workers = ["worker-0", "worker-1", "worker-2"]
best_worker, score = router.select_best_worker(workers)
print(f"Best worker: {best_worker}, score: {score}")

# 定期更新权重
while True:
    weights = router.update_envoy_weights(workers)
    # 更新Envoy配置
    time.sleep(10)
```

**优势**：
- ✅ Envoy是CNCF项目，生产级
- ✅ Prometheus是标准监控方案
- ✅ 有完整的API文档
- ✅ 社区活跃，问题容易解决

---

### 方案2：Traefik + Prometheus（快速落地）

**更简单，但功能相对有限**

#### 直接可用的API

**1. Traefik API**（查询服务状态）

```python
import requests

# Traefik API
traefik_url = "http://localhost:8080"

# 获取服务列表
response = requests.get(f"{traefik_url}/api/http/services")
services = response.json()

# 获取服务健康状态
response = requests.get(f"{traefik_url}/api/http/services/llm-judge-service/health")
health = response.json()
```

**2. 动态配置更新**

```yaml
# Traefik支持动态配置文件
# 修改配置文件，Traefik会自动重载

# dynamic.yml
http:
  services:
    llm-judge-service:
      weighted:
        services:
          - name: worker-0
            weight: 10  # 可以动态修改
          - name: worker-1
            weight: 20
```

**Python脚本更新配置**：

```python
import yaml

def update_traefik_weights(weights):
    """更新Traefik权重配置"""
    config = {
        "http": {
            "services": {
                "llm-judge-service": {
                    "weighted": {
                        "services": [
                            {"name": f"worker-{i}", "weight": w}
                            for i, w in enumerate(weights)
                        ]
                    }
                }
            }
        }
    }
    
    with open("/etc/traefik/dynamic.yml", "w") as f:
        yaml.dump(config, f)
    
    # Traefik会自动检测文件变化并重载

# 使用
update_traefik_weights([10, 20, 15])
```

**优势**：
- ✅ 配置简单
- ✅ 自动服务发现
- ✅ 文件更新自动重载

**劣势**：
- ⚠️ 动态权重更新需要写文件（不如API直接）

---

### 方案3：Ray Serve（如果使用Ray框架）

**Ray官方提供的GPU调度方案**

#### 直接可用的API

```python
from ray import serve
from ray.serve import Application

# Ray Serve自动处理GPU调度
@serve.deployment(
    num_replicas=3,  # 3个Worker
    ray_actor_options={
        "num_gpus": 1,
        "resources": {"gpu_priority": "high"}
    }
)
class LLMJudgeDeployment:
    def __init__(self):
        self.model = load_model()
    
    async def __call__(self, request):
        result = self.model.inference(request)
        return result

# Ray Serve自动负载均衡
app = LLMJudgeDeployment.bind()
serve.run(app)

# 直接调用
response = requests.post("http://localhost:8000/generate", json=data)
```

**Ray Serve的优势**：
- ✅ 自动负载均衡（内置）
- ✅ 自动扩缩容
- ✅ 支持GPU资源管理
- ✅ 有完整的Python API

**劣势**：
- ⚠️ 需要整个系统迁移到Ray
- ⚠️ 学习曲线较陡

---

## 基础设施层调度方案

### 方案1：Kueue（Kubernetes原生）⭐

**这是Kubernetes官方的作业队列系统**

#### 直接可用的API

**1. Kubernetes CRD API**

```python
from kubernetes import client, config

# 加载K8s配置
config.load_incluster_config()  # 在Pod内
# 或 config.load_kube_config()  # 本地

# 创建Kueue Workload
v1 = client.CustomObjectsApi()

workload = {
    "apiVersion": "kueue.x-k8s.io/v1beta1",
    "kind": "Workload",
    "metadata": {
        "name": "llm-judge-job-001",
        "namespace": "default"
    },
    "spec": {
        "queueName": "llm-judge-queue",
        "priority": 100,  # 优先级
        "podSets": [
            {
                "name": "llm-judge",
                "count": 1,
                "template": {
                    "spec": {
                        "containers": [
                            {
                                "name": "sglang",
                                "resources": {
                                    "requests": {
                                        "nvidia.com/gpu": "1"
                                    }
                                }
                            }
                        ]
                    }
                }
            }
        ]
    }
}

# 提交Workload
v1.create_namespaced_custom_object(
    group="kueue.x-k8s.io",
    version="v1beta1",
    namespace="default",
    plural="workloads",
    body=workload
)
```

**2. 查询Workload状态**

```python
# 查询Workload状态
workload = v1.get_namespaced_custom_object(
    group="kueue.x-k8s.io",
    version="v1beta1",
    namespace="default",
    plural="workloads",
    name="llm-judge-job-001"
)

status = workload["status"]
print(f"Admitted: {status.get('admission', {}).get('clusterQueue')}")
```

**优势**：
- ✅ Kubernetes官方支持
- ✅ 完整的CRD API
- ✅ 支持优先级、抢占、配额
- ✅ 与cluster-autoscaler集成

**适用场景**：
- 适合决定**Pod在哪个节点运行**
- 不适合**应用层请求路由**

---

### 方案2：Ray + Kueue集成

**如果使用Ray框架，可以用Ray + Kueue**

#### 直接可用的API

```python
from ray.job_submission import JobSubmissionClient

# 提交Ray Job（自动与Kueue集成）
client = JobSubmissionClient("http://ray-head:8265")

job_id = client.submit_job(
    entrypoint="python llm_judge.py",
    runtime_env={
        "env_vars": {"CUDA_VISIBLE_DEVICES": "0"}
    }
)

# 查询Job状态
status = client.get_job_status(job_id)
print(f"Job status: {status}")
```

**优势**：
- ✅ Ray官方支持
- ✅ 自动与Kueue集成
- ✅ 支持Gang调度

---

## 直接可用的API总结

### 应用层Router（你的需求）

| 方案 | API类型 | 直接可用 | 推荐度 |
|-----|---------|---------|--------|
| **Envoy + Prometheus** | HTTP REST API | ✅ 是 | ⭐⭐⭐⭐⭐ |
| **Traefik + Prometheus** | HTTP REST API + 文件配置 | ✅ 是 | ⭐⭐⭐⭐ |
| **Ray Serve** | Python API | ✅ 是 | ⭐⭐⭐ |

### 基础设施层调度

| 方案 | API类型 | 直接可用 | 推荐度 |
|-----|---------|---------|--------|
| **Kueue** | Kubernetes CRD API | ✅ 是 | ⭐⭐⭐⭐⭐ |
| **Ray + Kueue** | Ray Job API | ✅ 是 | ⭐⭐⭐⭐ |

---

## 推荐方案组合

### 方案A：应用层Router（推荐，符合你的需求）

```
医生请求
    ↓
Traefik/Envoy (负载均衡)
    ↓ (基于Prometheus指标选择)
SGLang Worker (GPU推理)
```

**使用的API**：
1. **Prometheus Query API**：`GET /api/v1/query`
2. **Envoy Admin API**：`POST /clusters/{cluster}/update`
3. **或Traefik文件配置**：修改`dynamic.yml`

**代码示例**（完整可用）：

```python
# 完整的Router实现（基于Envoy + Prometheus）
import requests
import time
from typing import List, Dict

class GPURouter:
    def __init__(self, prometheus_url="http://localhost:9090"):
        self.prometheus_url = prometheus_url
        self.workers = ["worker-0", "worker-1", "worker-2"]
    
    def query_prometheus(self, query: str) -> float:
        """查询Prometheus指标"""
        response = requests.get(
            f"{self.prometheus_url}/api/v1/query",
            params={"query": query}
        )
        data = response.json()
        if data["status"] == "success" and data["data"]["result"]:
            return float(data["data"]["result"][0]["value"][1])
        return 0.0
    
    def get_worker_metrics(self, worker_id: str) -> Dict:
        """获取Worker指标"""
        return {
            "inflight": self.query_prometheus(
                f'sglang_inflight_requests{{worker_id="{worker_id}"}}'
            ),
            "latency": self.query_prometheus(
                f'sglang_recent_latency_p95_ms{{worker_id="{worker_id}"}}'
            ),
            "free_vram": self.query_prometheus(
                f'sglang_free_vram_mb{{worker_id="{worker_id}"}}'
            ),
            "queue": self.query_prometheus(
                f'sglang_queue_length{{worker_id="{worker_id}"}}'
            )
        }
    
    def calculate_score(self, metrics: Dict) -> float:
        """SECT算法计算得分"""
        base_latency = metrics.get("latency", 3000)
        queue_wait = metrics.get("queue", 0) * base_latency
        load_penalty = metrics.get("inflight", 0) / 16 * 100
        return base_latency + queue_wait + load_penalty
    
    def select_best_worker(self) -> str:
        """选择最佳Worker"""
        scores = {}
        for worker_id in self.workers:
            metrics = self.get_worker_metrics(worker_id)
            scores[worker_id] = self.calculate_score(metrics)
        
        return min(scores, key=scores.get)
    
    def route_request(self, request_data: dict) -> dict:
        """路由请求到最佳Worker"""
        best_worker = self.select_best_worker()
        
        # 通过Envoy/Traefik发送请求
        worker_url = f"http://{best_worker}:30001/generate"
        response = requests.post(worker_url, json=request_data)
        
        return {
            "worker": best_worker,
            "result": response.json()
        }

# 使用
router = GPURouter()
result = router.route_request({
    "report_facts": [...],
    "visual_facts": [...]
})
print(f"Routed to {result['worker']}")
```

---

### 方案B：基础设施层 + 应用层（完整方案）

```
医生请求
    ↓
Traefik/Envoy (应用层负载均衡)
    ↓
Kueue (队列管理，决定何时运行)
    ↓
Kubernetes (决定Pod在哪个节点)
    ↓
SGLang Worker (GPU推理)
```

**使用的API**：
1. **Prometheus Query API**：应用层路由
2. **Kueue CRD API**：基础设施层调度
3. **Kubernetes API**：Pod管理

---

## 总结

### 你的需求：Router选择最佳Worker

**推荐方案**：**Envoy/Traefik + Prometheus**

**直接可用的API**：
1. ✅ **Prometheus Query API**：`GET /api/v1/query`（获取Worker指标）
2. ✅ **Envoy Admin API**：`POST /clusters/{cluster}/update`（更新权重）
3. ✅ **或Traefik文件配置**：修改`dynamic.yml`（更新权重）

**无需自己造轮子**：
- ✅ Envoy/Traefik是成熟的负载均衡器
- ✅ Prometheus是标准的监控方案
- ✅ 都有完整的API文档
- ✅ 代码示例可以直接使用

### 如果要用基础设施层调度

**推荐方案**：**Kueue**

**直接可用的API**：
- ✅ **Kubernetes CRD API**：通过`kubernetes` Python库调用

**但注意**：Kueue是决定Pod在哪个节点运行，不是应用层请求路由。

---

**文档版本**：v1.0  
**最后更新**：2025-01-25  
**维护者**：Yanda Cheng
