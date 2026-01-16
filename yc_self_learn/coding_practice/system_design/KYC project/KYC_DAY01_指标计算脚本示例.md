# Day 1｜指标计算：如何从 _summary.json 计算 Success Rate

**基于项目**：KYC Identity Verification System  
**目标**：展示如何读取和分析 `_summary.json`，计算 L0/L1/L2 指标

---

## 📊 _summary.json 的结构

### 典型的 _summary.json 格式

基于 KYC 项目的 `io_utils.py` 和 `BatchSummary` 设计，`_summary.json` 的结构如下：

```json
{
  "batch_id": "batch_20250101_120000",
  "start_time": "2025-01-01T12:00:00Z",
  "end_time": "2025-01-01T12:05:00Z",
  "total_documents": 100,
  "results": [
    {
      "file_id": "doc_001.jpg",
      "status": "success",
      "trace_id": "trace_abc123",
      "fw_request_id": "fw_req_xyz789",
      "latency_ms": 3500,
      "tokens_used": 1500,
      "cost_usd": 0.0015,
      "extracted_fields": {
        "full_name": {
          "value": "John Doe",
          "confidence": 0.95,
          "reason_if_null": null
        },
        "date_of_birth": {
          "value": "1990-01-01",
          "confidence": 0.92,
          "reason_if_null": null
        }
      },
      "fraud_markers": [],
      "needs_review": false
    },
    {
      "file_id": "doc_002.jpg",
      "status": "fail",
      "trace_id": "trace_def456",
      "error_code": "SCHEMA_VALIDATION_FAILED",
      "error_msg": "Missing required field: document_number",
      "latency_ms": 2800,
      "tokens_used": 1200,
      "cost_usd": 0.0012
    },
    {
      "file_id": "doc_003.jpg",
      "status": "success",
      "trace_id": "trace_ghi789",
      "fw_request_id": "fw_req_uvw456",
      "latency_ms": 8200,
      "tokens_used": 1800,
      "cost_usd": 0.0018,
      "extracted_fields": {
        "full_name": {
          "value": "Jane Smith",
          "confidence": 0.88,
          "reason_if_null": null
        },
        "date_of_birth": {
          "value": "1985-05-15",
          "confidence": 0.90,
          "reason_if_null": null
        },
        "document_number": {
          "value": null,
          "confidence": 0.45,
          "reason_if_null": "blur"
        }
      },
      "fraud_markers": ["missing_critical:document_number"],
      "needs_review": true
    }
  ],
  "summary": {
    "success_count": 85,
    "fail_count": 15,
    "total_processed": 100,
    "avg_latency_ms": 4500,
    "p50_latency_ms": 3800,
    "p95_latency_ms": 8500,
    "p99_latency_ms": 12000,
    "total_tokens": 150000,
    "total_cost_usd": 0.15,
    "automated_count": 60,
    "review_count": 25,
    "fraud_markers_count": 10
  }
}
```

---

## 🔧 Python 脚本：计算 Success Rate 和其他指标

### 📋 脚本说明

#### 1. 输入（Input）：`_summary.json` 文件

**文件结构**：
- **位置**：`output_results/_summary.json`（KYC 项目 batch processing 后生成）
- **格式**：JSON 格式
- **核心数据**：`results` 数组，包含每个文档的处理结果

**关键字段**：
- `results[].status`：`"success"` 或 `"fail"`（用于计算成功率）
- `results[].latency_ms`：处理延迟（毫秒，用于计算 p95/p99）
- `results[].cost_usd`：成本（美元，用于计算成本指标）
- `results[].needs_review`：是否需要人工审核（用于计算自动化率）
- `results[].trace_id`：追踪 ID（用于验证 Auditability 覆盖率）

**为什么需要这个文件？**
- KYC 项目的 `io_utils.py` 在 batch processing 后会自动生成 `_summary.json`
- 这个文件包含了所有文档的处理结果，是计算指标的数据源
- 统一的数据格式便于后续的统计分析和监控

#### 2. 输出（Output）：L0/L1/L2 三层指标

**L0 稳定性指标**：
- **成功率（Success Rate）**：`0.95`（95%，0.0-1.0）
- **错误率（Error Rate）**：`0.05`（5%，1.0 - Success Rate）
- **延迟分位数（Latency Percentiles）**：`{"p50": 4.5, "p95": 8.5, "p99": 12.0}`（秒）
- **错误分类（Error Breakdown）**：`{"SCHEMA_VALIDATION_FAILED": 2, "API_TIMEOUT": 1}`

**L1 业务收益指标**：
- **自动化率（Automation Rate）**：`0.65`（65%，无需人工介入的请求比例）
- **成本指标（Cost Metrics）**：`{"total_cost_usd": 0.15, "avg_cost_per_request_usd": 0.0015}`
- **时间节省（Time Savings）**：`{"time_saved_per_request_minutes": 4.92, "total_time_saved_hours": 81.0}`

**L2 长期健康指标**：
- **Fraud Markers 数量**：`10`（检测到的欺诈标记总数）
- **Auditability 覆盖率**：`1.0`（100%，包含 trace_id 的文档比例）

**为什么输出这些指标？**
- **L0 稳定性**：On-Call 工程师关心的实时指标（系统现在健康吗？）
- **L1 业务收益**：业务团队关心的价值指标（系统创造价值了吗？）
- **L2 长期健康**：工程团队关心的可持续性指标（系统能持续进化吗？）

#### 3. 计算逻辑（Why）：为什么这么算？

##### 3.1 成功率（Success Rate）的计算

**公式**：
```
Success Rate = (成功文档数) / (总文档数)
             = count(status == "success") / count(total)
```

**为什么这样算？**
- **定义**：成功率 = 正常响应的请求数 / 总请求数
- **数据来源**：`_summary.json` 中每个 `result` 的 `status` 字段
- **分类标准**：`"success"` 表示成功，`"fail"` 表示失败
- **意义**：直接反映系统可用性，是 L0 稳定性的核心指标

**示例计算**：
- 总文档数：100
- 成功文档数：95（`status == "success"`）
- 成功率 = 95 / 100 = 0.95（95%）

##### 3.2 延迟分位数（p95/p99）的计算

**公式**：
```
p95 = percentile(latencies, 95)  # 95% 的请求都在这个时间以内
p99 = percentile(latencies, 99)  # 99% 的请求都在这个时间以内
```

**为什么用 p95/p99 而不是平均值？**
- **平均值的问题**：被极端值（outliers）拉高，无法反映真实用户体验
- **p95/p99 的优势**：
  - p95 反映 95% 用户的真实体验（排除最慢的 5%）
  - p99 反映 99% 用户的真实体验（捕获长尾问题）
  - 更真实地反映系统性能

**计算步骤**：
1. 提取所有成功请求的 `latency_ms`（转换为秒）
2. 使用 `numpy.percentile()` 计算 p50/p95/p99
3. 样本量要求：至少 10+ 样本才能计算分位数

**示例计算**：
- 100 个成功请求的延迟：[2s, 3s, 4s, ..., 15s, 20s]
- p50（中位数）= 5s（50% 的请求在 5 秒内完成）
- p95 = 10s（95% 的请求在 10 秒内完成）
- p99 = 15s（99% 的请求在 15 秒内完成）

##### 3.3 自动化率（Automation Rate）的计算

**公式**：
```
Automation Rate = (无需人工审核的文档数) / (总文档数)
                = count(status == "success" AND needs_review == false) / count(total)
```

**为什么这样算？**
- **定义**：自动化率 = 无需人工介入的请求数 / 总请求数
- **分类标准**：`status == "success"` 且 `needs_review == false`
- **意义**：反映系统成熟度，直接关联人力成本（L1 业务收益）

**示例计算**：
- 总文档数：100
- 成功且无需审核：65（`status == "success" AND needs_review == false`）
- 自动化率 = 65 / 100 = 0.65（65%）

##### 3.4 成本指标（Cost Metrics）的计算

**公式**：
```
Total Cost = sum(cost_usd for all success requests)
Avg Cost per Request = Total Cost / Success Count
```

**为什么这样算？**
- **定义**：成本 = Fireworks API 调用成本（基于 tokens 使用量）
- **数据来源**：每个成功请求的 `cost_usd` 字段
- **意义**：量化系统成本，对比人工成本计算 ROI（L1 业务收益）

**示例计算**：
- 成功请求数：95
- 总成本：$0.1425（所有成功请求的 `cost_usd` 之和）
- 平均成本/请求 = $0.1425 / 95 = $0.0015

---

### 完整脚本示例

**使用前必读**：
- **输入**：需要 `output_results/_summary.json` 文件（KYC 项目 batch processing 后生成）
- **输出**：打印格式化的 L0/L1/L2 指标报告
- **依赖**：需要安装 `numpy`：`pip install numpy`

```python
#!/usr/bin/env python3
"""
计算 KYC 项目的 L0/L1/L2 指标
从 _summary.json 读取数据并计算各项指标

输入（Input）：
    - _summary.json 文件路径（例如：output_results/_summary.json）
    - 包含 results 数组，每个 result 有 status, latency_ms, cost_usd 等字段

输出（Output）：
    - L0 稳定性指标：成功率、错误率、延迟分位数（p50/p95/p99）、错误分类
    - L1 业务收益指标：自动化率、成本指标、时间节省
    - L2 长期健康指标：Fraud Markers 数量、Auditability 覆盖率

计算逻辑（Why）：
    1. 成功率 = count(status == "success") / count(total)
    2. p95/p99 = percentile(latencies, 95/99)  # 排除极端值，反映真实用户体验
    3. 自动化率 = count(status == "success" AND needs_review == false) / count(total)
    4. 成本 = sum(cost_usd) / count(success)  # 平均每请求成本
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime


class MetricsCalculator:
    """指标计算器：从 _summary.json 计算 L0/L1/L2 指标"""
    
    def __init__(self, summary_path: str):
        """
        初始化指标计算器
        
        Args:
            summary_path: _summary.json 的路径
        """
        self.summary_path = Path(summary_path)
        self.data = self._load_summary()
        self.results = self.data.get("results", [])
    
    def _load_summary(self) -> Dict[str, Any]:
        """加载 _summary.json"""
        if not self.summary_path.exists():
            raise FileNotFoundError(f"Summary file not found: {self.summary_path}")
        
        with open(self.summary_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    # ========== L0 稳定性指标 ==========
    
    def calculate_success_rate(self) -> float:
        """
        计算成功率（Success Rate）
        
        Returns:
            成功率（0.0 - 1.0）
        """
        total = len(self.results)
        if total == 0:
            return 0.0
        
        success_count = sum(1 for r in self.results if r.get("status") == "success")
        success_rate = success_count / total
        
        return success_rate
    
    def calculate_error_rate(self) -> float:
        """
        计算错误率（Error Rate）
        
        Returns:
            错误率（0.0 - 1.0）
        """
        return 1.0 - self.calculate_success_rate()
    
    def calculate_latency_percentiles(self) -> Dict[str, float]:
        """
        计算延迟分位数（p50/p95/p99）
        
        Returns:
            包含 p50, p95, p99 的字典（单位：秒）
        """
        # 提取所有成功的请求的延迟
        latencies = []
        for result in self.results:
            if result.get("status") == "success":
                latency_ms = result.get("latency_ms", 0)
                if latency_ms > 0:
                    latencies.append(latency_ms / 1000.0)  # 转换为秒
        
        if len(latencies) < 10:
            return {
                "p50": 0.0,
                "p95": 0.0,
                "p99": 0.0,
                "sample_count": len(latencies)
            }
        
        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)
        
        return {
            "p50": p50,
            "p95": p95,
            "p99": p99,
            "sample_count": len(latencies)
        }
    
    def calculate_error_breakdown(self) -> Dict[str, int]:
        """
        计算错误分类统计
        
        Returns:
            错误类型 -> 数量的字典
        """
        error_breakdown = {}
        
        for result in self.results:
            if result.get("status") == "fail":
                error_code = result.get("error_code", "UNKNOWN_ERROR")
                error_breakdown[error_code] = error_breakdown.get(error_code, 0) + 1
        
        return error_breakdown
    
    # ========== L1 业务收益指标 ==========
    
    def calculate_automation_rate(self) -> float:
        """
        计算自动化率
        
        定义：无需人工介入的请求数 / 总请求数
        条件：needs_review == false 的请求
        
        Returns:
            自动化率（0.0 - 1.0）
        """
        total = len(self.results)
        if total == 0:
            return 0.0
        
        automated_count = sum(
            1 for r in self.results 
            if r.get("status") == "success" and not r.get("needs_review", True)
        )
        
        automation_rate = automated_count / total
        return automation_rate
    
    def calculate_cost_metrics(self) -> Dict[str, Any]:
        """
        计算成本指标
        
        Returns:
            包含总成本、平均成本的字典
        """
        total_cost = 0.0
        total_tokens = 0
        success_count = 0
        
        for result in self.results:
            if result.get("status") == "success":
                cost = result.get("cost_usd", 0.0)
                tokens = result.get("tokens_used", 0)
                total_cost += cost
                total_tokens += tokens
                success_count += 1
        
        avg_cost_per_request = total_cost / success_count if success_count > 0 else 0.0
        avg_tokens_per_request = total_tokens / success_count if success_count > 0 else 0
        
        return {
            "total_cost_usd": total_cost,
            "avg_cost_per_request_usd": avg_cost_per_request,
            "total_tokens": total_tokens,
            "avg_tokens_per_request": avg_tokens_per_request,
            "success_count": success_count
        }
    
    def calculate_time_savings(self, manual_time_minutes: float = 5.0) -> Dict[str, Any]:
        """
        计算时间节省
        
        Args:
            manual_time_minutes: 人工审核每单所需时间（分钟）
        
        Returns:
            包含时间节省的字典
        """
        total_success = sum(1 for r in self.results if r.get("status") == "success")
        
        # AI 处理时间（秒转分钟）
        ai_times = []
        for result in self.results:
            if result.get("status") == "success":
                latency_s = result.get("latency_ms", 0) / 1000.0
                ai_times.append(latency_s / 60.0)  # 转换为分钟
        
        avg_ai_time_minutes = np.mean(ai_times) if ai_times else 0.0
        
        # 时间节省
        time_saved_per_request = manual_time_minutes - avg_ai_time_minutes
        total_time_saved_minutes = time_saved_per_request * total_success
        total_time_saved_hours = total_time_saved_minutes / 60.0
        
        return {
            "manual_time_per_request_minutes": manual_time_minutes,
            "avg_ai_time_per_request_minutes": avg_ai_time_minutes,
            "time_saved_per_request_minutes": time_saved_per_request,
            "total_time_saved_minutes": total_time_saved_minutes,
            "total_time_saved_hours": total_time_saved_hours,
            "efficiency_improvement_percent": (time_saved_per_request / manual_time_minutes) * 100
        }
    
    # ========== L2 长期健康指标 ==========
    
    def calculate_fraud_markers_count(self) -> int:
        """
        计算 Fraud Markers 数量
        
        Returns:
            Fraud Markers 总数
        """
        total_markers = 0
        
        for result in self.results:
            if result.get("status") == "success":
                markers = result.get("fraud_markers", [])
                total_markers += len(markers)
        
        return total_markers
    
    def calculate_auditability_coverage(self) -> float:
        """
        计算 Auditability 覆盖率
        
        定义：包含完整 trace_id 的文档数 / 总文档数
        
        Returns:
            覆盖率（0.0 - 1.0）
        """
        total = len(self.results)
        if total == 0:
            return 0.0
        
        covered_count = sum(
            1 for r in self.results 
            if r.get("trace_id") is not None and r.get("trace_id") != ""
        )
        
        coverage = covered_count / total
        return coverage
    
    # ========== 综合报告 ==========
    
    def generate_full_report(self) -> Dict[str, Any]:
        """
        生成完整的指标报告
        
        Returns:
            包含所有 L0/L1/L2 指标的字典
        """
        success_rate = self.calculate_success_rate()
        error_rate = self.calculate_error_rate()
        latency_percentiles = self.calculate_latency_percentiles()
        error_breakdown = self.calculate_error_breakdown()
        
        automation_rate = self.calculate_automation_rate()
        cost_metrics = self.calculate_cost_metrics()
        time_savings = self.calculate_time_savings()
        
        fraud_markers_count = self.calculate_fraud_markers_count()
        auditability_coverage = self.calculate_auditability_coverage()
        
        return {
            "batch_info": {
                "batch_id": self.data.get("batch_id"),
                "start_time": self.data.get("start_time"),
                "end_time": self.data.get("end_time"),
                "total_documents": len(self.results)
            },
            "l0_stability": {
                "success_rate": success_rate,
                "success_rate_percent": success_rate * 100,
                "error_rate": error_rate,
                "error_rate_percent": error_rate * 100,
                "latency_percentiles_seconds": latency_percentiles,
                "error_breakdown": error_breakdown
            },
            "l1_business_value": {
                "automation_rate": automation_rate,
                "automation_rate_percent": automation_rate * 100,
                "cost_metrics": cost_metrics,
                "time_savings": time_savings
            },
            "l2_long_term_health": {
                "fraud_markers_count": fraud_markers_count,
                "auditability_coverage": auditability_coverage,
                "auditability_coverage_percent": auditability_coverage * 100
            }
        }
    
    def print_report(self):
        """打印格式化的报告"""
        report = self.generate_full_report()
        
        print("=" * 60)
        print("KYC 项目指标报告")
        print("=" * 60)
        
        # Batch 信息
        batch_info = report["batch_info"]
        print(f"\n📦 Batch 信息:")
        print(f"  Batch ID: {batch_info.get('batch_id')}")
        print(f"  开始时间: {batch_info.get('start_time')}")
        print(f"  结束时间: {batch_info.get('end_time')}")
        print(f"  总文档数: {batch_info.get('total_documents')}")
        
        # L0 稳定性
        l0 = report["l0_stability"]
        print(f"\n📊 L0 稳定性指标:")
        print(f"  成功率: {l0['success_rate_percent']:.2f}%")
        print(f"  错误率: {l0['error_rate_percent']:.2f}%")
        
        latency = l0["latency_percentiles_seconds"]
        if latency["sample_count"] >= 10:
            print(f"  延迟分位数（秒）:")
            print(f"    p50: {latency['p50']:.2f}s")
            print(f"    p95: {latency['p95']:.2f}s (SLO: < 15s)")
            print(f"    p99: {latency['p99']:.2f}s (SLO: < 30s)")
        else:
            print(f"  ⚠️  样本不足（{latency['sample_count']} 个），无法计算分位数")
        
        error_breakdown = l0["error_breakdown"]
        if error_breakdown:
            print(f"  错误分类:")
            for error_code, count in error_breakdown.items():
                print(f"    {error_code}: {count} 次")
        
        # L1 业务收益
        l1 = report["l1_business_value"]
        print(f"\n💰 L1 业务收益指标:")
        print(f"  自动化率: {l1['automation_rate_percent']:.2f}%")
        
        cost = l1["cost_metrics"]
        print(f"  成本指标:")
        print(f"    总成本: ${cost['total_cost_usd']:.4f}")
        print(f"    平均成本/请求: ${cost['avg_cost_per_request_usd']:.4f}")
        print(f"    平均 Tokens/请求: {cost['avg_tokens_per_request']:.0f}")
        
        time_savings = l1["time_savings"]
        print(f"  时间节省:")
        print(f"    每单节省: {time_savings['time_saved_per_request_minutes']:.2f} 分钟")
        print(f"    总节省: {time_savings['total_time_saved_hours']:.2f} 小时")
        print(f"    效率提升: {time_savings['efficiency_improvement_percent']:.2f}%")
        
        # L2 长期健康
        l2 = report["l2_long_term_health"]
        print(f"\n🔧 L2 长期健康指标:")
        print(f"  Fraud Markers 总数: {l2['fraud_markers_count']}")
        print(f"  Auditability 覆盖率: {l2['auditability_coverage_percent']:.2f}%")
        
        print("\n" + "=" * 60)


def main():
    """主函数：演示如何使用指标计算器"""
    
    # 示例：使用 _summary.json 文件
    summary_path = "output_results/_summary.json"
    
    # 如果文件不存在，创建一个示例文件用于演示
    if not Path(summary_path).exists():
        print(f"⚠️  文件不存在: {summary_path}")
        print("创建一个示例 _summary.json 用于演示...")
        
        # 创建示例数据
        example_data = {
            "batch_id": "batch_example_20250101",
            "start_time": "2025-01-01T12:00:00Z",
            "end_time": "2025-01-01T12:05:00Z",
            "total_documents": 100,
            "results": []
        }
        
        # 生成示例结果
        import random
        for i in range(100):
            is_success = random.random() > 0.05  # 95% 成功率
            
            if is_success:
                latency = random.uniform(2000, 10000)  # 2-10 秒
                result = {
                    "file_id": f"doc_{i:03d}.jpg",
                    "status": "success",
                    "trace_id": f"trace_{random.randint(1000, 9999)}",
                    "fw_request_id": f"fw_req_{random.randint(1000, 9999)}",
                    "latency_ms": int(latency),
                    "tokens_used": random.randint(1000, 2000),
                    "cost_usd": round(latency / 1000 * 0.001, 4),
                    "needs_review": random.random() > 0.65  # 65% 自动化
                }
            else:
                error_codes = [
                    "SCHEMA_VALIDATION_FAILED",
                    "API_TIMEOUT",
                    "IMAGE_FORMAT_UNSUPPORTED"
                ]
                result = {
                    "file_id": f"doc_{i:03d}.jpg",
                    "status": "fail",
                    "trace_id": f"trace_{random.randint(1000, 9999)}",
                    "error_code": random.choice(error_codes),
                    "error_msg": "Example error message",
                    "latency_ms": random.randint(1000, 5000),
                    "tokens_used": random.randint(500, 1000),
                    "cost_usd": 0.0
                }
            
            example_data["results"].append(result)
        
        # 保存示例文件
        Path("output_results").mkdir(exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(example_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 已创建示例文件: {summary_path}")
    
    # 使用指标计算器
    calculator = MetricsCalculator(summary_path)
    
    # 计算并打印报告
    calculator.print_report()
    
    # 也可以单独计算指标
    print("\n" + "=" * 60)
    print("单独指标计算示例:")
    print("=" * 60)
    
    success_rate = calculator.calculate_success_rate()
    print(f"\n成功率: {success_rate * 100:.2f}%")
    
    latency = calculator.calculate_latency_percentiles()
    if latency["sample_count"] >= 10:
        print(f"p95 延迟: {latency['p95']:.2f} 秒")
        print(f"p99 延迟: {latency['p99']:.2f} 秒")


if __name__ == "__main__":
    main()
```

---

## 📝 使用方法

### 1. 基本使用

```python
from pathlib import Path
from metrics_calculator import MetricsCalculator

# 创建指标计算器
calculator = MetricsCalculator("output_results/_summary.json")

# 计算成功率
success_rate = calculator.calculate_success_rate()
print(f"成功率: {success_rate * 100:.2f}%")

# 计算延迟分位数
latency = calculator.calculate_latency_percentiles()
print(f"p95: {latency['p95']:.2f}s")
print(f"p99: {latency['p99']:.2f}s")

# 生成完整报告
report = calculator.generate_full_report()
calculator.print_report()
```

### 2. 与 KYC 项目集成

在 `main.py` 或测试脚本中使用：

```python
# main.py 末尾添加
from metrics_calculator import MetricsCalculator

# 处理完 batch 后，计算指标
if __name__ == "__main__":
    # ... 原有的 batch processing 代码 ...
    
    # 计算并打印指标
    summary_path = args.output_dir / "_summary.json"
    if summary_path.exists():
        calculator = MetricsCalculator(str(summary_path))
        calculator.print_report()
```

### 3. CI/CD 集成（Release Gate）

```python
# tests/test_release_gate.py
import pytest
from metrics_calculator import MetricsCalculator

def test_success_rate_gate():
    """成功率门禁：必须 > 95% 才能发布"""
    calculator = MetricsCalculator("output_results/_summary.json")
    success_rate = calculator.calculate_success_rate()
    
    assert success_rate >= 0.95, f"Success rate {success_rate*100:.2f}% below 95% threshold. Cannot release."

def test_p95_latency_gate():
    """p95 延迟门禁：必须 < 15s 才能发布"""
    calculator = MetricsCalculator("output_results/_summary.json")
    latency = calculator.calculate_latency_percentiles()
    
    if latency["sample_count"] >= 10:
        assert latency["p95"] < 15, f"p95 latency {latency['p95']:.2f}s exceeds 15s threshold. Cannot release."
```

---

## 🔍 关键函数解析

### `calculate_success_rate()` 详细解析

```python
def calculate_success_rate(self) -> float:
    """
    计算成功率的核心逻辑：
    
    1. 遍历所有 results
    2. 统计 status == "success" 的数量
    3. 除以总数量
    """
    total = len(self.results)  # 总文档数
    if total == 0:
        return 0.0
    
    # 统计成功的数量
    success_count = sum(
        1 for r in self.results 
        if r.get("status") == "success"  # 关键：检查 status 字段
    )
    
    # 计算成功率
    success_rate = success_count / total
    
    return success_rate  # 返回 0.0 - 1.0 的浮点数
```

### 文件交互流程

```
_summary.json (JSON 文件)
    ↓
json.load() (Python 读取)
    ↓
self.data (Python 字典)
    ↓
self.results (结果列表)
    ↓
遍历 results，检查每个 result 的 "status" 字段
    ↓
统计 "status" == "success" 的数量
    ↓
计算成功率 = success_count / total_count
```

---

## 📊 实际运行示例

运行脚本后的输出：

```
============================================================
KYC 项目指标报告
============================================================

📦 Batch 信息:
  Batch ID: batch_20250101_120000
  开始时间: 2025-01-01T12:00:00Z
  结束时间: 2025-01-01T12:05:00Z
  总文档数: 100

📊 L0 稳定性指标:
  成功率: 95.00%
  错误率: 5.00%
  延迟分位数（秒）:
    p50: 4.50s
    p95: 8.50s (SLO: < 15s)
    p99: 12.00s (SLO: < 30s)
  错误分类:
    SCHEMA_VALIDATION_FAILED: 2 次
    API_TIMEOUT: 2 次
    IMAGE_FORMAT_UNSUPPORTED: 1 次

💰 L1 业务收益指标:
  自动化率: 65.00%
  成本指标:
    总成本: $0.1425
    平均成本/请求: $0.0015
    平均 Tokens/请求: 1500
  时间节省:
    每单节省: 4.92 分钟
    总节省: 81.00 小时
    效率提升: 98.40%

🔧 L2 长期健康指标:
  Fraud Markers 总数: 10
  Auditability 覆盖率: 100.00%

============================================================
```

---

## 🎯 总结

### 如何计算 Success Rate

1. **读取文件**：使用 `json.load()` 读取 `_summary.json`
2. **遍历结果**：遍历 `results` 列表
3. **检查状态**：检查每个 `result` 的 `"status"` 字段
4. **统计成功**：统计 `"status" == "success"` 的数量
5. **计算比率**：`success_count / total_count`

### 与文件如何交互

- **读取**：`json.load()` → Python 字典
- **访问**：`data["results"]` → 结果列表
- **遍历**：`for result in results` → 逐个检查
- **写入**：KYC 项目的 `io_utils.py` 负责写入

### 实用建议

1. **处理边界情况**：`total == 0`、`latency_ms == None`
2. **样本量检查**：计算 p95/p99 需要至少 10+ 样本
3. **错误处理**：文件不存在、格式错误
4. **性能考虑**：大文件使用流式读取

---

## 📚 参考

- KYC 项目 `io_utils.py`：查看如何写入 `_summary.json`
- KYC 项目 `BatchSummary`：查看批量汇总的实现
- Python `json` 模块文档：https://docs.python.org/3/library/json.html
