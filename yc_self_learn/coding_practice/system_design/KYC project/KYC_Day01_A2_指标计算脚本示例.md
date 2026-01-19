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

## ⏰ 指标计算的时机：什么时候计算？自动还是手动？

### A2_a1：指标计算的三种时机

**指标计算发生在什么时候？**

| 时机 | 触发方式 | 适用场景 | 代码位置 |
|------|---------|---------|---------|
| **1. 批量处理完成后** | 自动触发 | 处理完一个 batch 后立即计算 | `main.py` 末尾 |
| **2. 定时任务** | 自动触发（定时） | 每天/每周汇总统计 | Cron 任务 / 定时脚本 |
| **3. CI/CD 门禁** | 自动触发（测试） | 发布前检查指标是否达标 | `tests/test_release_gate.py` |

#### 整体结构与逻辑

**（1）整体结构：三种时机并列，共同指向指标计算**

```
                         ┌─────────────────────────┐
                         │      指标计算            │
                         │ (MetricsCalculator +     │
                         │  _summary.json)         │
                         └────────────┬────────────┘
                                      │
          ┌───────────────────────────┼───────────────────────────┐
          │                           │                           │
          ▼                           ▼                           ▼
  ┌───────────────┐         ┌───────────────┐         ┌───────────────┐
  │ 时机 1        │         │ 时机 2        │         │ 时机 3        │
  │ 批量处理完成后 │         │ 定时任务      │         │ CI/CD 门禁    │
  │               │         │               │         │               │
  │ 触发：batch 结束│         │ 触发：Cron 到点│         │ 触发：测试跑完 │
  │ 位置：main.py │         │ 位置：定时脚本 │         │ 位置：test_*  │
  └───────────────┘         └───────────────┘         └───────────────┘
```

**（2）逻辑设计：各时机的 输入 → 触发条件 → 计算 → 输出**

| 时机 | 输入 | 触发条件 | 计算行为 | 输出 |
|------|------|----------|----------|------|
| **1. 批量处理完成后** | 本 batch 的 `_summary.json` | `process_batch()` 结束、`_summary.json` 已写入 | 读本批 `_summary.json`，算 L0/L1/L2 | 当批指标（含 success_rate、p95 等）；可选打印/落库 |
| **2. 定时任务** | 多个 batch 的 `_summary.json`（如当日/当周） | Cron 到点（如每天 2:00） | 汇总多个 `_summary.json`，再算 | 日/周汇总指标；写入 `daily_metrics_*.json` 等 |
| **3. CI/CD 门禁** | 测试生成的 `_summary.json` 或固定 fixture | CI 跑 `test_release_gate.py` | 算 success_rate、p95 等，与阈值比较 | 通过则继续发布；不达标则**阻断** pipeline |

**（3）关系小结**

- **互斥的只是「谁在何时调」**：三种时机在不同场景下由不同入口调用，彼此不互斥（可同时存在：batch 完算一次 + 定时汇总 + CI 门禁）。
- **计算逻辑统一**：都依赖 `_summary.json` 的 `results[].status` 等，用同一套 MetricsCalculator 或等价公式。
- **输出用途不同**：时机 1 偏实时反馈/调试；时机 2 偏监控/报表；时机 3 偏发布质量门禁。

**（4）A2_a1 下钻（children）**

| 概念 | 节点 | 所属时机 |
|------|------|----------|
| Cron | [KYC_Day01_A2_B1_cron](./KYC_Day01_A2_B1_cron.md) | 时机 2 定时任务 |
| CI/CD、CI/CD 门禁 | [KYC_Day01_A2_B2_ci_cd](./KYC_Day01_A2_B2_ci_cd.md) | 时机 3 CI/CD 门禁 |
| 从开发到用户使用的完整流程 | [KYC_Day01_A2_B3_从开发到用户使用的完整流程](./KYC_Day01_A2_B3_从开发到用户使用的完整流程.md) | 整体流程概览 |

---

### A2_a2：方式一：批量处理完成后自动计算（推荐）

**什么时候计算？**

**处理完一个 batch 后立即计算**（自动触发）

**代码位置**：`main.py` 末尾

**完整示例**：

```python
# main.py
import argparse
from pathlib import Path
from metrics_calculator import MetricsCalculator

def process_batch(input_dir, output_dir):
    """处理一个 batch 的文档"""
    # ... 原有的处理逻辑 ...
    # 1. 读取文档
    # 2. 调用 API
    # 3. 保存结果到 _summary.json
    
    # 4. 【自动触发】处理完成后，立即计算指标
    summary_path = output_dir / "_summary.json"
    if summary_path.exists():
        calculator = MetricsCalculator(str(summary_path))
        report = calculator.generate_full_report()
        
        # 打印指标（可选）
        calculator.print_report()
        
        # 检查是否达标（可选）
        if report["success_rate"] < 0.95:
            print("⚠️  警告：成功率低于 95%！")
        
        return report
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    
    # 处理 batch（自动触发指标计算）
    report = process_batch(args.input_dir, args.output_dir)
```

**执行流程**：

```
1. 运行 main.py
   ↓
2. 处理文档（调用 API、保存结果）
   ↓
3. 写入 _summary.json
   ↓
4. 【自动触发】立即计算指标
   ↓
5. 打印报告（或检查门禁）
```

**优点**：
- ✅ 实时反馈：处理完立即知道结果
- ✅ 自动化：不需要手动运行
- ✅ 适合开发/测试阶段

### A2_a3：方式二：定时任务自动计算（生产环境）

**什么时候计算？**

**每天/每周定时计算**（自动触发，定时执行）

**代码位置**：独立的定时脚本 `scripts/daily_metrics.py`

**完整示例**：

```python
# scripts/daily_metrics.py
"""
定时任务：每天凌晨 2 点计算所有 batch 的指标
"""
import schedule
import time
from pathlib import Path
from metrics_calculator import MetricsCalculator
from datetime import datetime

def calculate_daily_metrics():
    """计算当天的所有 batch 指标"""
    today = datetime.now().strftime("%Y%m%d")
    output_base = Path("output_results")
    
    # 找到今天的所有 batch
    today_batches = list(output_base.glob(f"batch_{today}_*"))
    
    if not today_batches:
        print(f"今天（{today}）没有 batch 需要计算指标")
        return
    
    # 汇总所有 batch 的指标
    all_results = []
    for batch_dir in today_batches:
        summary_path = batch_dir / "_summary.json"
        if summary_path.exists():
            calculator = MetricsCalculator(str(summary_path))
            report = calculator.generate_full_report()
            all_results.append(report)
    
    # 计算总体指标
    total_success = sum(r["success_count"] for r in all_results)
    total_count = sum(r["total_documents"] for r in all_results)
    overall_success_rate = total_success / total_count if total_count > 0 else 0.0
    
    # 保存到文件（用于监控系统读取）
    metrics_file = output_base / f"daily_metrics_{today}.json"
    with open(metrics_file, "w") as f:
        json.dump({
            "date": today,
            "overall_success_rate": overall_success_rate,
            "total_batches": len(today_batches),
            "total_documents": total_count,
            "batches": all_results
        }, f, indent=2)
    
    print(f"✅ 已计算 {today} 的指标：成功率 {overall_success_rate*100:.2f}%")

# 定时任务：每天凌晨 2 点执行
schedule.every().day.at("02:00").do(calculate_daily_metrics)

if __name__ == "__main__":
    print("定时任务已启动：每天凌晨 2 点计算指标")
    while True:
        schedule.run_pending()
        time.sleep(60)  # 每分钟检查一次
```

**如何运行定时任务？**

**方式 A：使用 Python schedule 库（开发环境）**：

```bash
# 后台运行
nohup python scripts/daily_metrics.py > logs/metrics.log 2>&1 &
```

**方式 B：使用 Cron（生产环境，推荐）**：

```bash
# 编辑 crontab
crontab -e

# 添加一行：每天凌晨 2 点执行
0 2 * * * cd /path/to/kyc_project && python scripts/daily_metrics.py >> logs/metrics.log 2>&1
```

**方式 C：使用 Windows 任务计划程序（Windows）**：

1. 打开"任务计划程序"
2. 创建基本任务
3. 触发器：每天 2:00
4. 操作：启动程序 `python scripts/daily_metrics.py`

**优点**：
- ✅ 自动化：无需人工干预
- ✅ 适合生产环境：定期汇总统计
- ✅ 可以发送告警（如果指标异常）

### A2_a4：方式三：CI/CD 门禁自动检查（发布前）

**什么时候计算？**

**发布前自动检查**（CI/CD 触发）

**代码位置**：`tests/test_release_gate.py`

**完整示例**：

```python
# tests/test_release_gate.py
"""
CI/CD 门禁：发布前必须检查指标是否达标
"""
import pytest
from pathlib import Path
from metrics_calculator import MetricsCalculator

def test_success_rate_gate():
    """成功率门禁：必须 > 95% 才能发布"""
    summary_path = Path("output_results/_summary.json")
    
    if not summary_path.exists():
        pytest.skip("_summary.json 不存在，跳过门禁检查")
    
    calculator = MetricsCalculator(str(summary_path))
    success_rate = calculator.calculate_success_rate()
    
    # 门禁：成功率必须 >= 95%
    assert success_rate >= 0.95, (
        f"❌ 发布被阻止：成功率 {success_rate*100:.2f}% 低于 95% 阈值。"
        f"请修复问题后再发布。"
    )
    
    print(f"✅ 成功率门禁通过：{success_rate*100:.2f}%")

def test_p95_latency_gate():
    """p95 延迟门禁：必须 < 15s 才能发布"""
    summary_path = Path("output_results/_summary.json")
    
    if not summary_path.exists():
        pytest.skip("_summary.json 不存在，跳过门禁检查")
    
    calculator = MetricsCalculator(str(summary_path))
    latency = calculator.calculate_latency_percentiles()
    
    if latency["sample_count"] < 10:
        pytest.skip("样本量不足，跳过延迟门禁检查")
    
    # 门禁：p95 延迟必须 < 15s
    assert latency["p95"] < 15, (
        f"❌ 发布被阻止：p95 延迟 {latency['p95']:.2f}s 超过 15s 阈值。"
        f"请优化性能后再发布。"
    )
    
    print(f"✅ 延迟门禁通过：p95 = {latency['p95']:.2f}s")

def test_all_gates():
    """运行所有门禁检查"""
    test_success_rate_gate()
    test_p95_latency_gate()
    print("✅ 所有门禁检查通过，可以发布！")
```

**在 CI/CD 中运行**：

```yaml
# .github/workflows/release.yml
name: Release Gate

on:
  pull_request:
    branches: [main]

jobs:
  metrics-gate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Run KYC batch processing
        run: |
          python main.py --input_dir test_data --output_dir output_results
      
      - name: Run metrics gate tests
        run: |
          pytest tests/test_release_gate.py -v
      
      # 如果测试失败，PR 无法合并
```

**执行流程**：

```
1. 开发者提交 PR
   ↓
2. CI/CD 自动运行测试
   ↓
3. 处理测试数据，生成 _summary.json
   ↓
4. 【自动触发】运行门禁检查
   ↓
5. 如果指标不达标 → PR 被阻止
   如果指标达标 → PR 可以合并
```

**优点**：
- ✅ 防止发布低质量代码
- ✅ 自动化：无需人工检查
- ✅ 强制要求：不达标无法发布

### A2_a5：方式四：实时监控（高级，可选）

**什么时候计算？**

**每处理一个文档就更新指标**（实时计算）

**代码位置**：在 `main.py` 的处理循环中

**完整示例**：

```python
# main.py（实时监控版本）
from collections import deque
from metrics_calculator import MetricsCalculator

class RealTimeMetrics:
    """实时指标计算器"""
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.recent_latencies = deque(maxlen=window_size)  # 最近 100 个请求的延迟
        self.success_count = 0
        self.total_count = 0
    
    def update(self, status: str, latency_ms: float):
        """每处理一个文档，更新指标"""
        self.total_count += 1
        if status == "success":
            self.success_count += 1
            self.recent_latencies.append(latency_ms)
    
    def get_current_metrics(self):
        """获取当前指标"""
        success_rate = self.success_count / self.total_count if self.total_count > 0 else 0.0
        
        if len(self.recent_latencies) >= 10:
            sorted_latencies = sorted(self.recent_latencies)
            p95_idx = int(len(sorted_latencies) * 0.95)
            p99_idx = int(len(sorted_latencies) * 0.99)
            p95 = sorted_latencies[p95_idx]
            p99 = sorted_latencies[p99_idx]
        else:
            p95 = p99 = None
        
        return {
            "success_rate": success_rate,
            "p95_latency_ms": p95,
            "p99_latency_ms": p99,
            "total_processed": self.total_count
        }

def process_batch_with_realtime_metrics(input_dir, output_dir):
    """处理 batch，实时计算指标"""
    metrics = RealTimeMetrics()
    
    for doc_file in input_dir.glob("*.jpg"):
        # 处理文档
        result = process_document(doc_file)
        
        # 【实时更新】每处理一个文档，更新指标
        metrics.update(result["status"], result["latency_ms"])
        
        # 每处理 10 个文档，打印一次指标
        if metrics.total_count % 10 == 0:
            current = metrics.get_current_metrics()
            print(f"实时指标：成功率 {current['success_rate']*100:.2f}%, "
                  f"p95={current['p95_latency_ms']:.0f}ms")
    
    # 处理完成后，保存最终指标
    final_metrics = metrics.get_current_metrics()
    print(f"✅ 最终指标：{final_metrics}")
```

**优点**：
- ✅ 实时反馈：立即知道当前状态
- ✅ 适合调试：快速发现问题
- ✅ 适合演示：实时展示指标

**缺点**：
- ❌ 计算开销：每处理一个文档都要计算
- ❌ 内存占用：需要保存历史数据

### A2_a6：总结：指标计算的时机

**四种方式对比**：

| 方式 | 触发时机 | 自动化程度 | 适用场景 | 代码位置 |
|------|---------|----------|---------|---------|
| **批量完成后** | 处理完 batch 后 | ✅ 自动 | 开发/测试 | `main.py` 末尾 |
| **定时任务** | 每天/每周定时 | ✅ 自动 | 生产环境 | `scripts/daily_metrics.py` |
| **CI/CD 门禁** | 发布前 | ✅ 自动 | 发布流程 | `tests/test_release_gate.py` |
| **实时监控** | 每处理一个文档 | ✅ 自动 | 调试/演示 | `main.py` 处理循环中 |

**推荐方案**：

1. **开发阶段**：使用"批量完成后"方式（最简单）
2. **生产环境**：使用"定时任务"方式（定期汇总）
3. **发布流程**：使用"CI/CD 门禁"方式（强制检查）
4. **调试阶段**：使用"实时监控"方式（快速反馈）

**关键点**：

- ✅ **都是自动的**：不需要手动运行（除了第一次设置）
- ✅ **计算位置**：在代码中调用 `MetricsCalculator` 类
- ✅ **触发时机**：根据场景选择（批量/定时/CI/CD/实时）

### A2_a7：人工介入时机：什么时候需要人来处理？

**核心原则**：

```
自动化优先 → 自动处理失败 → 人工介入
```

**什么时候需要人工介入？**

| 场景 | 自动处理 | 人工介入 | 原因 |
|------|---------|---------|------|
| **Success Rate < 98%** | ✅ 自动重试 | ❌ 不需要 | 自动重试通常能恢复 |
| **Success Rate < 95%** | ✅ 自动降级 | ⚠️ 需要监控 | 降级后需要人工确认是否恢复 |
| **自动重试失败** | ❌ 无法自动 | ✅ **必须介入** | 需要人工调查原因 |
| **自动降级失败** | ❌ 无法自动 | ✅ **必须介入** | 需要人工决策（回滚/修复） |
| **自动回滚失败** | ❌ 无法自动 | ✅ **紧急介入** | 系统可能完全不可用 |
| **指标异常但原因不明** | ❌ 无法自动 | ✅ **必须介入** | 需要人工分析根因 |
| **业务规则变更** | ❌ 无法自动 | ✅ **必须介入** | 需要人工更新代码/配置 |

### A2_a8：人工介入的完整流程

**流程 1：告警触发 → 自动处理 → 人工确认**

```python
# 1. 告警触发（自动）
if success_rate < 0.95:
    # 2. 自动处理（自动）
    auto_fallback()  # 自动降级
    
    # 3. 等待一段时间，检查是否恢复
    time.sleep(300)  # 等待 5 分钟
    
    # 4. 重新计算指标
    new_metrics = calculate_metrics()
    
    # 5. 如果仍未恢复 → 通知人工介入
    if new_metrics.success_rate < 0.95:
        # 【人工介入点 1】发送告警通知
        send_alert_to_oncall(
            level="critical",
            message=f"自动降级后仍未恢复：Success Rate {new_metrics.success_rate*100:.2f}%",
            action_required=True,
            oncall_phone=True
        )
        
        # 【人工介入点 2】生成诊断报告
        diagnostic_report = generate_diagnostic_report()
        send_report_to_oncall(diagnostic_report)
```

**流程 2：人工介入的具体步骤**

**Step 1：接收告警通知**

```
告警通知（Slack / PagerDuty / 电话）
    ↓
On-Call 工程师收到通知
    ↓
查看告警详情：
- Success Rate: 92.5% (阈值: 95%)
- p95 Latency: 18.5s (阈值: 15s)
- 自动重试: 已执行，失败
- 自动降级: 已执行，仍未恢复
```

**Step 2：查看诊断报告**

```python
# 诊断报告包含：
diagnostic_report = {
    "metrics": {
        "success_rate": 0.925,
        "p95_latency": 18.5,
        "error_breakdown": {
            "API_TIMEOUT": 45,
            "SCHEMA_VALIDATION_FAILED": 12,
            "API_SERVER_ERROR": 8
        }
    },
    "auto_actions_taken": [
        "Auto-retry: 3 attempts, all failed",
        "Auto-fallback: Enabled OCR-only mode, still failing"
    ],
    "recent_changes": [
        "Deployment: v1.2.3 deployed 2 hours ago",
        "Config change: API timeout increased to 30s"
    ],
    "suggested_actions": [
        "Check Fireworks API status",
        "Review recent code changes",
        "Check if data format changed"
    ]
}
```

**Step 3：人工决策和操作**

**决策树**：

```
人工介入
    ↓
查看诊断报告
    ↓
判断问题类型
    ├─→ API 服务问题 → 联系 Fireworks 支持 / 切换备用 API
    ├─→ 代码 Bug → 回滚到上一个稳定版本
    ├─→ 数据格式问题 → 修复数据 / 更新 Schema
    ├─→ 配置错误 → 修复配置 / 回滚配置
    └─→ 未知问题 → 深入调查（查看日志、抓包等）
```

**具体操作示例**：

```python
# 人工操作：回滚到上一个稳定版本
def manual_rollback():
    """人工触发回滚"""
    # 1. 确认回滚目标版本
    stable_version = get_last_stable_version()
    print(f"准备回滚到版本: {stable_version}")
    
    # 2. 人工确认（防止误操作）
    confirm = input("确认回滚？(yes/no): ")
    if confirm != "yes":
        print("回滚已取消")
        return
    
    # 3. 执行回滚
    rollback_to_version(stable_version)
    
    # 4. 验证回滚效果
    time.sleep(300)  # 等待 5 分钟
    new_metrics = calculate_metrics()
    
    if new_metrics.success_rate > 0.95:
        print("✅ 回滚成功，指标已恢复")
        # 记录 Postmortem
        record_postmortem(
            incident="Success rate dropped to 92.5%",
            root_cause="API timeout issue",
            action_taken="Rollback to v1.2.2",
            resolution="Success rate recovered to 96.8%"
        )
    else:
        print("❌ 回滚失败，需要进一步调查")
        escalate_to_senior_engineer()
```

### A2_a9：人工介入的典型场景

**场景 1：自动重试失败**

```
触发条件：Success Rate < 98%
自动处理：自动重试失败的请求（3 次）
结果：重试后 Success Rate 仍然 < 98%

【人工介入】
1. 查看错误类型分布
2. 如果是 API 服务问题 → 联系 Fireworks 支持
3. 如果是代码问题 → 查看最近部署的代码
4. 如果是数据问题 → 检查输入数据格式
```

**场景 2：自动降级失败**

```
触发条件：Success Rate < 95%
自动处理：自动切换到 OCR-only 降级方案
结果：降级后 Success Rate 仍然 < 95%

【人工介入】
1. 检查降级方案是否正常工作
2. 如果降级方案也失败 → 考虑回滚
3. 如果降级方案正常但指标仍低 → 可能是业务逻辑问题
4. 需要人工决策：回滚 vs 修复 vs 等待
```

**场景 3：自动回滚失败**

```
触发条件：Success Rate < 90% 或 Error Rate > 10%
自动处理：自动回滚到上一个稳定版本
结果：回滚后 Success Rate 仍然 < 90%

【人工介入】（紧急）
1. 立即通知 Senior Engineer
2. 检查是否有基础设施问题（网络、数据库等）
3. 考虑切换到备用系统
4. 启动应急预案
```

**场景 4：指标异常但原因不明**

```
触发条件：Success Rate 突然下降，但自动处理都正常
自动处理：无（因为原因不明）

【人工介入】
1. 查看最近 24 小时的变更记录
2. 检查是否有外部依赖变化（API、数据库等）
3. 分析错误日志，找出异常模式
4. 可能需要深入调查（抓包、性能分析等）
```

**场景 5：业务规则变更**

```
触发条件：业务需求变更（如新的验证规则）
自动处理：无（需要人工更新代码）

【人工介入】
1. 理解新的业务需求
2. 更新代码 / 配置
3. 测试新功能
4. 部署到生产环境
5. 监控指标变化
```

### A2_a10：人工介入的工具和流程

**工具 1：告警通知系统**

```python
# 发送告警通知
def send_alert_to_oncall(level, message, action_required=False):
    """发送告警到 On-Call 工程师"""
    
    # 1. Slack 通知（非紧急）
    if level in ["info", "warning"]:
        send_slack_message(
            channel="#kyc-alerts",
            message=f"[{level.upper()}] {message}",
            mention_users=["@oncall"]
        )
    
    # 2. PagerDuty 通知（紧急）
    elif level in ["critical", "emergency"]:
        send_pagerduty_alert(
            severity=level,
            message=message,
            action_required=action_required
        )
    
    # 3. 电话通知（Emergency）
    if level == "emergency":
        call_oncall_phone()
```

**工具 2：诊断报告生成**

```python
# 生成诊断报告
def generate_diagnostic_report():
    """生成诊断报告，帮助人工快速定位问题"""
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "metrics": calculate_current_metrics(),
        "error_breakdown": analyze_error_types(),
        "recent_changes": get_recent_changes(hours=24),
        "auto_actions_taken": get_auto_actions_log(),
        "suggested_actions": suggest_actions(),
        "related_logs": get_relevant_logs(),
        "system_status": check_system_status()
    }
    
    return report
```

**工具 3：Postmortem 记录**

```python
# 记录 Postmortem
def record_postmortem(incident, root_cause, action_taken, resolution):
    """记录事故处理过程，用于后续分析"""
    
    postmortem = {
        "incident_id": generate_incident_id(),
        "timestamp": datetime.now().isoformat(),
        "incident": incident,
        "root_cause": root_cause,
        "action_taken": action_taken,
        "resolution": resolution,
        "metrics_before": get_metrics_before_incident(),
        "metrics_after": get_metrics_after_resolution(),
        "lessons_learned": [],
        "follow_up_actions": []
    }
    
    # 保存到数据库
    save_postmortem(postmortem)
    
    # 发送到团队
    send_postmortem_to_team(postmortem)
```

### A2_a11：人工介入的总结

**什么时候需要人工介入？**

| 情况 | 是否需要人工 | 原因 |
|------|------------|------|
| 自动重试成功 | ❌ 不需要 | 问题已自动解决 |
| 自动降级成功 | ⚠️ 需要监控 | 需要确认是否恢复正常 |
| 自动重试失败 | ✅ **必须介入** | 需要调查原因 |
| 自动降级失败 | ✅ **必须介入** | 需要决策（回滚/修复） |
| 自动回滚失败 | ✅ **紧急介入** | 系统可能完全不可用 |
| 指标异常但原因不明 | ✅ **必须介入** | 需要人工分析 |
| 业务规则变更 | ✅ **必须介入** | 需要人工更新代码 |

**人工介入的流程**：

```
1. 接收告警通知（Slack / PagerDuty / 电话）
   ↓
2. 查看诊断报告（了解问题详情）
   ↓
3. 判断问题类型（API / 代码 / 数据 / 配置）
   ↓
4. 执行操作（回滚 / 修复 / 联系支持）
   ↓
5. 验证恢复（检查指标是否恢复）
   ↓
6. 记录 Postmortem（用于后续分析）
```

**关键原则**：

- ✅ **自动化优先**：能自动处理的，不要等人工
- ✅ **快速响应**：Critical 告警 < 5 分钟响应
- ✅ **可追溯**：所有操作都要记录
- ✅ **持续改进**：从 Postmortem 中学习，改进自动化

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

## 🏗️ 设计原理与方案对比

---

> 📌 **设计原理学习路径**（从大到小、从高到低、从浅入深）已单独成文，见：[KYC_teaching_rules.md](./KYC_teaching_rules.md)

---

### A2_b1：当前设计的核心原理（逐条展开）

**为什么这样设计？**

**原理 1：批量计算（Batch Processing）**

```
设计：处理完一个 batch 后，一次性计算所有指标
原理：减少计算开销，提高效率
```

**为什么不用实时计算？**

| 方案 | 计算频率 | 开销 | 适用场景 |
|------|---------|------|---------|
| **实时计算** | 每处理一个文档就计算 | 高（频繁计算） | 调试/演示 |
| **批量计算** | 处理完 batch 后计算 | 低（一次性计算） | **生产环境（推荐）** |

**Trade-off**：
- ✅ **优点**：计算开销低，适合大规模处理
- ❌ **缺点**：无法实时看到指标（但可以接受，因为 batch 通常几分钟就完成）

**原理 2：文件存储（File-based Storage）**

```
设计：将结果保存到 _summary.json 文件
原理：简单、可追溯、易于调试
```

**为什么不用数据库？**

| 方案 | 存储方式 | 优点 | 缺点 |
|------|---------|------|------|
| **文件存储** | `_summary.json` | 简单、无需数据库、易于调试 | 不适合高并发 |
| **数据库存储** | MySQL/PostgreSQL | 支持高并发、查询灵活 | 需要数据库、复杂度高 |
| **时序数据库** | InfluxDB/Prometheus | 专为指标设计、查询快 | 需要额外基础设施 |

**Trade-off**：
- ✅ **优点**：简单、无需额外基础设施、易于调试
- ❌ **缺点**：不适合高并发场景（但 KYC 项目是批量处理，不是高并发）

**原理 3：分层指标（L0/L1/L2）**

```
设计：将指标分为三层（L0 核心、L1 业务、L2 长期）
原理：关注点分离，不同层级关注不同问题
```

**为什么分层？**

| 层级 | 关注点 | 响应时间 | 告警阈值 |
|------|--------|---------|---------|
| **L0** | 系统是否正常工作 | 立即 | Success Rate < 95% |
| **L1** | 业务是否高效 | 15分钟 | Cost > 预算 |
| **L2** | 长期健康度 | 1天 | Fraud Markers 增加 |

**Trade-off**：
- ✅ **优点**：清晰、易于管理、不同问题不同响应
- ❌ **缺点**：需要维护三层指标（但值得）

**原理 4：自动化优先（Automation First）**

```
设计：能自动处理的，不要等人工
原理：快速响应、减少人工成本
```

**为什么自动化优先？**

| 方案 | 响应时间 | 成本 | 可靠性 |
|------|---------|------|--------|
| **人工处理** | 5-30分钟 | 高（需要 On-Call） | 依赖人工 |
| **自动化处理** | < 1分钟 | 低（代码执行） | **高（推荐）** |

**Trade-off**：
- ✅ **优点**：快速响应、24/7 可用、成本低
- ❌ **缺点**：需要设计自动化逻辑（但一次设计，长期受益）

### A2_b2：其他设计方案

**方案 A：实时流式计算（Streaming）**

**设计**：
```python
# 每处理一个文档，立即更新指标
class StreamingMetrics:
    def __init__(self):
        self.success_count = 0
        self.total_count = 0
        self.latencies = []
    
    def update(self, status, latency):
        """每处理一个文档，立即更新"""
        self.total_count += 1
        if status == "success":
            self.success_count += 1
            self.latencies.append(latency)
        
        # 立即计算指标
        success_rate = self.success_count / self.total_count
        p95 = calculate_percentile(self.latencies, 0.95)
        
        # 立即检查告警
        if success_rate < 0.95:
            trigger_alert()
```

**Trade-off**：

| 维度 | 实时流式计算 | 批量计算（当前） |
|------|------------|----------------|
| **实时性** | ✅ 实时（立即） | ❌ 延迟（batch 完成后） |
| **计算开销** | ❌ 高（频繁计算） | ✅ 低（一次性计算） |
| **内存占用** | ❌ 高（需要保存所有数据） | ✅ 低（只保存文件） |
| **复杂度** | ❌ 高（需要流式处理逻辑） | ✅ 低（简单批量计算） |
| **适用场景** | 高并发、实时监控 | **批量处理（KYC 项目）** |

**结论**：KYC 项目是批量处理，不需要实时计算，批量计算更合适。

---

**方案 B：数据库存储（Database Storage）**

**设计**：
```python
# 将结果保存到数据库
class DatabaseMetrics:
    def __init__(self, db_connection):
        self.db = db_connection
    
    def save_result(self, result):
        """保存每个结果到数据库"""
        self.db.execute(
            "INSERT INTO kyc_results (file_id, status, latency_ms, ...) VALUES (?, ?, ?, ...)",
            (result["file_id"], result["status"], result["latency_ms"], ...)
        )
    
    def calculate_metrics(self):
        """从数据库查询计算指标"""
        success_rate = self.db.execute(
            "SELECT COUNT(*) FILTER (WHERE status = 'success') * 1.0 / COUNT(*) FROM kyc_results"
        ).fetchone()[0]
        return success_rate
```

**Trade-off**：

| 维度 | 数据库存储 | 文件存储（当前） |
|------|----------|----------------|
| **并发支持** | ✅ 高（数据库支持并发） | ❌ 低（文件锁） |
| **查询灵活性** | ✅ 高（SQL 查询） | ❌ 低（需要读取整个文件） |
| **基础设施** | ❌ 需要数据库 | ✅ 无需额外基础设施 |
| **复杂度** | ❌ 高（需要数据库设计） | ✅ 低（简单文件） |
| **调试难度** | ❌ 高（需要 SQL 查询） | ✅ 低（直接看 JSON） |
| **适用场景** | 高并发、多用户 | **单机批量处理（KYC 项目）** |

**结论**：KYC 项目是单机批量处理，文件存储更简单、更合适。

---

**方案 C：时序数据库（Time-Series Database）**

**设计**：
```python
# 使用 Prometheus 或 InfluxDB
from prometheus_client import Counter, Histogram

# 定义指标
success_counter = Counter('kyc_success_total', 'Total successful KYC verifications')
failure_counter = Counter('kyc_failure_total', 'Total failed KYC verifications')
latency_histogram = Histogram('kyc_latency_seconds', 'KYC processing latency')

# 每处理一个文档，更新指标
def process_document(doc):
    start_time = time.time()
    result = process(doc)
    latency = time.time() - start_time
    
    if result["status"] == "success":
        success_counter.inc()
    else:
        failure_counter.inc()
    
    latency_histogram.observe(latency)
```

**Trade-off**：

| 维度 | 时序数据库 | 文件存储（当前） |
|------|----------|----------------|
| **指标查询** | ✅ 专为指标设计、查询快 | ❌ 需要读取文件计算 |
| **可视化** | ✅ 内置 Grafana 支持 | ❌ 需要自己实现 |
| **告警集成** | ✅ 内置告警规则 | ❌ 需要自己实现 |
| **基础设施** | ❌ 需要 Prometheus/InfluxDB | ✅ 无需额外基础设施 |
| **复杂度** | ❌ 高（需要配置时序数据库） | ✅ 低（简单文件） |
| **适用场景** | 大规模监控、多服务 | **单机批量处理（KYC 项目）** |

**结论**：KYC 项目规模较小，文件存储更简单；如果未来需要大规模监控，再迁移到时序数据库。

---

**方案 D：完全人工处理（Manual Processing）**

**设计**：
```python
# 不自动计算，完全依赖人工
# 人工定期查看 _summary.json，手动计算指标
```

**Trade-off**：

| 维度 | 完全人工 | 自动化（当前） |
|------|---------|---------------|
| **响应时间** | ❌ 慢（需要人工） | ✅ 快（自动） |
| **成本** | ❌ 高（需要 On-Call） | ✅ 低（代码执行） |
| **可靠性** | ❌ 低（依赖人工） | ✅ 高（自动化） |
| **24/7 可用** | ❌ 否（需要人工值班） | ✅ 是（自动运行） |
| **复杂度** | ✅ 低（不需要代码） | ❌ 高（需要设计自动化） |

**结论**：自动化是必须的，完全人工不可行。

---

**方案 E：混合方案（Hybrid）**

**设计**：
```python
# 批量计算 + 实时监控
class HybridMetrics:
    def __init__(self):
        self.batch_calculator = BatchMetricsCalculator()  # 批量计算
        self.real_time_monitor = RealTimeMonitor()  # 实时监控
    
    def process_batch(self, batch):
        # 批量处理
        results = []
        for doc in batch:
            result = process_document(doc)
            results.append(result)
            
            # 实时监控（每 10 个文档更新一次）
            if len(results) % 10 == 0:
                self.real_time_monitor.update(results[-10:])
        
        # 批量计算（处理完成后）
        self.batch_calculator.calculate(results)
```

**Trade-off**：

| 维度 | 混合方案 | 纯批量计算（当前） |
|------|---------|------------------|
| **实时性** | ✅ 有实时监控 | ❌ 无实时监控 |
| **计算开销** | ❌ 高（批量 + 实时） | ✅ 低（只有批量） |
| **复杂度** | ❌ 高（两套逻辑） | ✅ 低（一套逻辑） |
| **适用场景** | 需要实时反馈 + 批量处理 | **纯批量处理（KYC 项目）** |

**结论**：KYC 项目不需要实时监控，纯批量计算更简单。

### A2_b3：方案对比总结

**完整对比表**：

| 方案 | 实时性 | 计算开销 | 复杂度 | 基础设施 | 适用场景 | 推荐度 |
|------|--------|---------|--------|---------|---------|--------|
| **批量计算（当前）** | ⚠️ 延迟 | ✅ 低 | ✅ 低 | ✅ 无 | 批量处理 | ⭐⭐⭐⭐⭐ |
| **实时流式计算** | ✅ 实时 | ❌ 高 | ❌ 高 | ✅ 无 | 高并发 | ⭐⭐ |
| **数据库存储** | ⚠️ 延迟 | ⚠️ 中 | ❌ 高 | ❌ 需要数据库 | 高并发 | ⭐⭐⭐ |
| **时序数据库** | ✅ 实时 | ✅ 低 | ❌ 高 | ❌ 需要时序数据库 | 大规模监控 | ⭐⭐⭐ |
| **完全人工** | ❌ 很慢 | ✅ 无 | ✅ 低 | ✅ 无 | 不推荐 | ⭐ |
| **混合方案** | ✅ 实时 | ❌ 高 | ❌ 高 | ⚠️ 可选 | 需要实时 + 批量 | ⭐⭐⭐ |

**为什么选择当前方案（批量计算 + 文件存储）？**

1. **KYC 项目特点**：
   - ✅ 批量处理（不是高并发）
   - ✅ 单机运行（不需要分布式）
   - ✅ 规模较小（不需要大规模监控）

2. **当前方案优势**：
   - ✅ **简单**：无需额外基础设施
   - ✅ **低成本**：文件存储，无需数据库
   - ✅ **易于调试**：直接看 JSON 文件
   - ✅ **足够用**：满足 KYC 项目需求

3. **未来扩展性**：
   - 如果未来需要高并发 → 迁移到数据库
   - 如果未来需要实时监控 → 添加实时计算
   - 如果未来需要大规模监控 → 迁移到时序数据库

**设计原则总结**：

1. **简单优先**：能用简单方案解决的，不要用复杂方案
2. **够用就好**：满足当前需求即可，不要过度设计
3. **可扩展**：保留未来扩展的可能性
4. **成本考虑**：基础设施成本 vs 开发成本

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
