# 评估报告模板和生成详解

---
doc_type: tutorial
layer: L2
scope_in:  评估报告模板（HTML/JSON/Markdown）、报告生成代码（Python）、报告可视化、报告分发机制
scope_out: 具体可视化库配置（见 reference）；邮件/Slack 集成（见 reference）
inputs:  (读者) 需求：理解如何创建评估报告模板，如何用 Python 代码生成报告，如何可视化和分发报告
outputs:  完整报告模板 + 生成代码 + 可视化示例 + 分发机制 + KYC 项目实际案例
entrypoints: [ 报告模板, 报告生成, 报告可视化, 报告分发 ]
children: []
related: [ 回归测试, 评估报告, Before/After 对比, KYC_Day03_A1_回归测试与门禁详解.md ]
---

## Definition（定义）

**核心问题**：**如何生成标准化的评估报告？什么是报告模板？什么是生成代码？**

**核心答案**：
- ✅ **详细模板**：报告的结构和格式（HTML/JSON/Markdown），定义了报告长什么样
- ✅ **生成代码**：Python 代码，将测试结果填充到模板中，生成最终报告
- ✅ **报告示例**：最终生成的报告文件（HTML/JSON/Markdown）
- ✅ **可视化**：图表、表格，让报告更易读
- ✅ **分发机制**：邮件、Slack、PR 评论，自动发送报告

---

## 📋 什么是"详细模板"和"生成代码"？

### 1. 详细模板（Report Template）

**定义**：**报告的结构和格式，定义了报告应该包含哪些内容、如何排版**

**类比**：
- Word 模板：定义了报告的格式（标题、表格、样式）
- 详细模板：定义了报告的结构（HTML/JSON/Markdown 格式）

**作用**：
- ✅ 定义报告结构（标题、章节、表格）
- ✅ 定义报告格式（样式、颜色、布局）
- ✅ 定义数据占位符（`{{ version }}`、`{{ metrics }}`）

---

### 2. 生成代码（Report Generation Code）

**定义**：**Python 代码，将测试结果数据填充到模板中，生成最终报告文件**

**类比**：
- Word 邮件合并：将 Excel 数据填充到 Word 模板
- 生成代码：将测试结果填充到 HTML 模板

**作用**：
- ✅ 读取测试结果数据（JSON）
- ✅ 填充到模板中（渲染模板）
- ✅ 生成报告文件（HTML/JSON/Markdown）
- ✅ 保存和分发报告

---

### 3. 完整流程

```
测试结果数据（JSON）
    ↓
生成代码（Python）
    ↓
详细模板（HTML）
    ↓
最终报告（HTML 文件）
    ↓
分发报告（邮件/Slack/PR）
```

---

## 📄 详细模板示例

### 模板 1：HTML 模板（可视化报告）

**文件位置**：`templates/regression_report.html`

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>回归测试报告 - {{ version }}</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .header {
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }
        
        .header h1 {
            color: #2c3e50;
            margin-bottom: 10px;
        }
        
        .header-info {
            display: flex;
            gap: 30px;
            color: #666;
            font-size: 14px;
        }
        
        .status-badge {
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 14px;
        }
        
        .status-passed {
            background-color: #4CAF50;
            color: white;
        }
        
        .status-failed {
            background-color: #f44336;
            color: white;
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }
        
        .metric-card {
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 20px;
            background: #fafafa;
        }
        
        .metric-card h3 {
            font-size: 14px;
            color: #666;
            margin-bottom: 10px;
        }
        
        .metric-value {
            font-size: 32px;
            font-weight: bold;
            margin-bottom: 5px;
        }
        
        .metric-value.passed {
            color: #4CAF50;
        }
        
        .metric-value.failed {
            color: #f44336;
        }
        
        .metric-threshold {
            font-size: 12px;
            color: #999;
        }
        
        .comparison-table {
            width: 100%;
            border-collapse: collapse;
            margin: 30px 0;
        }
        
        .comparison-table th,
        .comparison-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        
        .comparison-table th {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
        }
        
        .comparison-table tr:hover {
            background-color: #f5f5f5;
        }
        
        .delta-positive {
            color: #4CAF50;
            font-weight: bold;
        }
        
        .delta-negative {
            color: #f44336;
            font-weight: bold;
        }
        
        .delta-zero {
            color: #666;
        }
        
        .failed-cases {
            margin-top: 30px;
        }
        
        .failed-case {
            background-color: #ffebee;
            border-left: 4px solid #f44336;
            padding: 15px;
            margin-bottom: 10px;
            border-radius: 4px;
        }
        
        .failed-case h4 {
            color: #c62828;
            margin-bottom: 5px;
        }
        
        .failed-case .error {
            color: #666;
            font-size: 14px;
        }
        
        .footer {
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            text-align: center;
            color: #999;
            font-size: 12px;
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- 报告头部 -->
        <div class="header">
            <h1>回归测试报告</h1>
            <div class="header-info">
                <span><strong>版本:</strong> {{ version }}</span>
                <span><strong>测试时间:</strong> {{ test_date }}</span>
                <span><strong>Golden Set:</strong> {{ total_cases }} 条用例</span>
                <span class="status-badge {{ 'status-passed' if all_passed else 'status-failed' }}">
                    {{ '✅ 通过' if all_passed else '❌ 失败' }}
                </span>
            </div>
        </div>
        
        <!-- 变更信息 -->
        <div class="section">
            <h2>变更信息</h2>
            <p><strong>变更类型:</strong> {{ change_type }}</p>
            <p><strong>变更描述:</strong> {{ change_description }}</p>
            <p><strong>基准版本:</strong> {{ baseline_version }}</p>
            <p><strong>当前版本:</strong> {{ current_version }}</p>
        </div>
        
        <!-- 核心指标卡片 -->
        <div class="metrics-grid">
            <div class="metric-card">
                <h3>Schema Pass Rate</h3>
                <div class="metric-value {{ 'passed' if schema_pass_rate >= 95 else 'failed' }}">
                    {{ "%.2f"|format(schema_pass_rate) }}%
                </div>
                <div class="metric-threshold">阈值: ≥ 95%</div>
            </div>
            
            <div class="metric-card">
                <h3>字段级准确率</h3>
                <div class="metric-value {{ 'passed' if field_accuracy >= 90 else 'failed' }}">
                    {{ "%.2f"|format(field_accuracy) }}%
                </div>
                <div class="metric-threshold">阈值: ≥ 90%</div>
            </div>
            
            <div class="metric-card">
                <h3>字段级一致性</h3>
                <div class="metric-value {{ 'passed' if field_consistency >= 85 else 'failed' }}">
                    {{ "%.2f"|format(field_consistency) }}%
                </div>
                <div class="metric-threshold">阈值: ≥ 85%</div>
            </div>
            
            <div class="metric-card">
                <h3>Fallback 比例</h3>
                <div class="metric-value {{ 'passed' if fallback_rate < 5 else 'failed' }}">
                    {{ "%.2f"|format(fallback_rate) }}%
                </div>
                <div class="metric-threshold">阈值: < 5%</div>
            </div>
            
            <div class="metric-card">
                <h3>平均延迟</h3>
                <div class="metric-value {{ 'passed' if avg_latency_ms < 2000 else 'failed' }}">
                    {{ "%.0f"|format(avg_latency_ms) }} ms
                </div>
                <div class="metric-threshold">阈值: < 2000ms</div>
            </div>
            
            <div class="metric-card">
                <h3>平均成本</h3>
                <div class="metric-value {{ 'passed' if avg_cost_per_request < 0.002 else 'failed' }}">
                    ${{ "%.4f"|format(avg_cost_per_request) }}
                </div>
                <div class="metric-threshold">阈值: < $0.002</div>
            </div>
        </div>
        
        <!-- Before/After 对比表格 -->
        <div class="section">
            <h2>Before/After 对比</h2>
            <table class="comparison-table">
                <thead>
                    <tr>
                        <th>指标</th>
                        <th>Before</th>
                        <th>After</th>
                        <th>变化 (Delta)</th>
                        <th>状态</th>
                    </tr>
                </thead>
                <tbody>
                    {% for metric in comparison_metrics %}
                    <tr>
                        <td><strong>{{ metric.name }}</strong></td>
                        <td>{{ metric.before }}</td>
                        <td>{{ metric.after }}</td>
                        <td class="{% if metric.delta > 0 %}delta-positive{% elif metric.delta < 0 %}delta-negative{% else %}delta-zero{% endif %}">
                            {{ '+' if metric.delta > 0 else '' }}{{ "%.2f"|format(metric.delta) }}{{ metric.unit }}
                        </td>
                        <td>
                            {% if metric.passed %}
                                <span class="status-badge status-passed">✅ 通过</span>
                            {% else %}
                                <span class="status-badge status-failed">❌ 失败</span>
                            {% endif %}
                        </td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        
        <!-- 退化分析 -->
        {% if regressed_metrics %}
        <div class="section">
            <h2>⚠️ 退化分析</h2>
            <p><strong>退化的指标:</strong> {{ regressed_metrics|join(', ') }}</p>
            <p><strong>退化原因分析:</strong> {{ regression_analysis }}</p>
        </div>
        {% endif %}
        
        <!-- 失败用例详情 -->
        {% if failed_cases %}
        <div class="section failed-cases">
            <h2>❌ 失败用例详情</h2>
            <p><strong>失败用例数:</strong> {{ failed_cases|length }} / {{ total_cases }}</p>
            {% for case in failed_cases[:10] %}
            <div class="failed-case">
                <h4>用例 ID: {{ case.case_id }}</h4>
                <p><strong>分类:</strong> {{ case.category }}</p>
                <p><strong>错误信息:</strong> <span class="error">{{ case.error }}</span></p>
                <p><strong>预期:</strong> {{ case.expected }}</p>
                <p><strong>实际:</strong> {{ case.actual }}</p>
            </div>
            {% endfor %}
            {% if failed_cases|length > 10 %}
            <p><em>... 还有 {{ failed_cases|length - 10 }} 个失败用例，详见完整报告</em></p>
            {% endif %}
        </div>
        {% endif %}
        
        <!-- 门禁结论 -->
        <div class="section">
            <h2>🚪 Release Gate 结论</h2>
            {% if all_passed %}
            <div style="background-color: #e8f5e9; padding: 20px; border-radius: 8px; border-left: 4px solid #4CAF50;">
                <h3 style="color: #2e7d32; margin-bottom: 10px;">✅ 所有门禁指标通过</h3>
                <p><strong>决策:</strong> 允许发布</p>
                <p><strong>通过的门禁:</strong> {{ passed_gates|length }} / {{ total_gates }}</p>
            </div>
            {% else %}
            <div style="background-color: #ffebee; padding: 20px; border-radius: 8px; border-left: 4px solid #f44336;">
                <h3 style="color: #c62828; margin-bottom: 10px;">❌ 门禁指标未通过</h3>
                <p><strong>决策:</strong> 禁止发布</p>
                <p><strong>失败的门禁:</strong> {{ failed_gates|join(', ') }}</p>
                <p><strong>需要修复:</strong> 请修复上述问题后重新运行回归测试</p>
            </div>
            {% endif %}
        </div>
        
        <!-- 报告尾部 -->
        <div class="footer">
            <p>报告生成时间: {{ report_generated_at }}</p>
            <p>KYC 项目回归测试系统</p>
        </div>
    </div>
</body>
</html>
```

---

### 模板 2：JSON 模板（数据格式）

**文件位置**：`templates/regression_report.json`

```json
{
  "report_version": "1.0.0",
  "version": "{{ version }}",
  "test_date": "{{ test_date }}",
  "baseline_version": "{{ baseline_version }}",
  "current_version": "{{ current_version }}",
  "change_type": "{{ change_type }}",
  "change_description": "{{ change_description }}",
  "golden_set": {
    "total_cases": {{ total_cases }},
    "passed_cases": {{ passed_cases }},
    "failed_cases": {{ failed_cases }}
  },
  "metrics": {
    "schema_pass_rate": {
      "value": {{ schema_pass_rate }},
      "threshold": 95,
      "passed": {{ schema_pass_rate >= 95|lower }},
      "before": {{ before_schema_pass_rate }},
      "after": {{ schema_pass_rate }},
      "delta": {{ schema_pass_rate - before_schema_pass_rate }}
    },
    "field_accuracy": {
      "value": {{ field_accuracy }},
      "threshold": 90,
      "passed": {{ field_accuracy >= 90|lower }},
      "before": {{ before_field_accuracy }},
      "after": {{ field_accuracy }},
      "delta": {{ field_accuracy - before_field_accuracy }}
    }
  },
  "release_gate": {
    "all_passed": {{ all_passed|lower }},
    "passed_gates": {{ passed_gates|tojson }},
    "failed_gates": {{ failed_gates|tojson }},
    "decision": "{{ '允许发布' if all_passed else '禁止发布' }}"
  },
  "failed_cases": {{ failed_cases_list|tojson }},
  "regression_analysis": {
    "regressed_metrics": {{ regressed_metrics|tojson }},
    "analysis": "{{ regression_analysis }}"
  },
  "report_generated_at": "{{ report_generated_at }}"
}
```

---

### 模板 3：Markdown 模板（文本格式）

**文件位置**：`templates/regression_report.md`

```markdown
# 回归测试报告

## 基本信息

- **版本**: {{ version }}
- **测试时间**: {{ test_date }}
- **Golden Set**: {{ total_cases }} 条用例
- **状态**: {{ '✅ 通过' if all_passed else '❌ 失败' }}

## 变更信息

- **变更类型**: {{ change_type }}
- **变更描述**: {{ change_description }}
- **基准版本**: {{ baseline_version }}
- **当前版本**: {{ current_version }}

## 核心指标

| 指标 | 当前值 | 阈值 | 状态 |
|------|--------|------|------|
| Schema Pass Rate | {{ "%.2f"|format(schema_pass_rate) }}% | ≥ 95% | {{ '✅' if schema_pass_rate >= 95 else '❌' }} |
| 字段级准确率 | {{ "%.2f"|format(field_accuracy) }}% | ≥ 90% | {{ '✅' if field_accuracy >= 90 else '❌' }} |
| 字段级一致性 | {{ "%.2f"|format(field_consistency) }}% | ≥ 85% | {{ '✅' if field_consistency >= 85 else '❌' }} |
| Fallback 比例 | {{ "%.2f"|format(fallback_rate) }}% | < 5% | {{ '✅' if fallback_rate < 5 else '❌' }} |
| 平均延迟 | {{ "%.0f"|format(avg_latency_ms) }} ms | < 2000ms | {{ '✅' if avg_latency_ms < 2000 else '❌' }} |
| 平均成本 | ${{ "%.4f"|format(avg_cost_per_request) }} | < $0.002 | {{ '✅' if avg_cost_per_request < 0.002 else '❌' }} |

## Before/After 对比

| 指标 | Before | After | 变化 (Delta) | 状态 |
|------|--------|-------|-------------|------|
{% for metric in comparison_metrics %}
| {{ metric.name }} | {{ metric.before }} | {{ metric.after }} | {{ '+' if metric.delta > 0 else '' }}{{ "%.2f"|format(metric.delta) }}{{ metric.unit }} | {{ '✅' if metric.passed else '❌' }} |
{% endfor %}

## Release Gate 结论

{% if all_passed %}
### ✅ 所有门禁指标通过

**决策**: 允许发布

**通过的门禁**: {{ passed_gates|length }} / {{ total_gates }}
{% else %}
### ❌ 门禁指标未通过

**决策**: 禁止发布

**失败的门禁**: {{ failed_gates|join(', ') }}

**需要修复**: 请修复上述问题后重新运行回归测试
{% endif %}

{% if failed_cases %}
## ❌ 失败用例详情

失败用例数: {{ failed_cases|length }} / {{ total_cases }}

{% for case in failed_cases[:10] %}
### 用例 ID: {{ case.case_id }}

- **分类**: {{ case.category }}
- **错误信息**: {{ case.error }}
- **预期**: {{ case.expected }}
- **实际**: {{ case.actual }}

{% endfor %}
{% if failed_cases|length > 10 %}
... 还有 {{ failed_cases|length - 10 }} 个失败用例，详见完整报告
{% endif %}
{% endif %}

---

报告生成时间: {{ report_generated_at }}
```

---

## 🐍 生成代码（Python）

### 完整生成代码示例

**文件位置**：`scripts/generate_regression_report.py`

```python
"""
回归测试报告生成器

功能：
1. 读取测试结果数据
2. 填充到模板中
3. 生成 HTML/JSON/Markdown 报告
4. 保存和分发报告
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from jinja2 import Template, Environment, FileSystemLoader


class RegressionReportGenerator:
    """回归测试报告生成器"""
    
    def __init__(self, template_dir: str = "templates"):
        """初始化报告生成器"""
        self.template_dir = Path(template_dir)
        self.env = Environment(loader=FileSystemLoader(str(self.template_dir)))
        
        # 注册自定义过滤器
        self.env.filters['format_percent'] = lambda x: f"{x:.2f}%"
        self.env.filters['format_currency'] = lambda x: f"${x:.4f}"
        self.env.filters['format_latency'] = lambda x: f"{x:.0f} ms"
    
    def load_test_results(self, results_file: str) -> Dict:
        """加载测试结果数据"""
        with open(results_file, 'r') as f:
            return json.load(f)
    
    def prepare_report_data(self, test_results: Dict, baseline_results: Optional[Dict] = None) -> Dict:
        """准备报告数据"""
        
        # 基本信息
        report_data = {
            "version": test_results.get("version", "unknown"),
            "test_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "baseline_version": baseline_results.get("version", "unknown") if baseline_results else "unknown",
            "current_version": test_results.get("version", "unknown"),
            "change_type": test_results.get("change_type", "unknown"),
            "change_description": test_results.get("change_description", ""),
            "total_cases": test_results.get("total", 0),
            "passed_cases": test_results.get("passed", 0),
            "failed_cases": test_results.get("failed", 0),
            "report_generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # 核心指标
        report_data.update({
            "schema_pass_rate": test_results.get("schema_pass_rate", 0),
            "field_accuracy": test_results.get("field_accuracy", 0),
            "field_consistency": test_results.get("field_consistency", 0),
            "fallback_rate": test_results.get("fallback_rate", 0),
            "avg_latency_ms": test_results.get("avg_latency_ms", 0),
            "avg_cost_per_request": test_results.get("avg_cost_per_request", 0)
        })
        
        # Before/After 对比
        if baseline_results:
            report_data.update({
                "before_schema_pass_rate": baseline_results.get("schema_pass_rate", 0),
                "before_field_accuracy": baseline_results.get("field_accuracy", 0),
                "before_field_consistency": baseline_results.get("field_consistency", 0),
                "before_fallback_rate": baseline_results.get("fallback_rate", 0),
                "before_avg_latency_ms": baseline_results.get("avg_latency_ms", 0),
                "before_avg_cost_per_request": baseline_results.get("avg_cost_per_request", 0)
            })
        else:
            report_data.update({
                "before_schema_pass_rate": 0,
                "before_field_accuracy": 0,
                "before_field_consistency": 0,
                "before_fallback_rate": 0,
                "before_avg_latency_ms": 0,
                "before_avg_cost_per_request": 0
            })
        
        # 对比指标列表
        comparison_metrics = [
            {
                "name": "Schema Pass Rate",
                "before": report_data["before_schema_pass_rate"],
                "after": report_data["schema_pass_rate"],
                "delta": report_data["schema_pass_rate"] - report_data["before_schema_pass_rate"],
                "unit": "%",
                "passed": report_data["schema_pass_rate"] >= 95
            },
            {
                "name": "字段级准确率",
                "before": report_data["before_field_accuracy"],
                "after": report_data["field_accuracy"],
                "delta": report_data["field_accuracy"] - report_data["before_field_accuracy"],
                "unit": "%",
                "passed": report_data["field_accuracy"] >= 90
            },
            {
                "name": "字段级一致性",
                "before": report_data["before_field_consistency"],
                "after": report_data["field_consistency"],
                "delta": report_data["field_consistency"] - report_data["before_field_consistency"],
                "unit": "%",
                "passed": report_data["field_consistency"] >= 85
            },
            {
                "name": "Fallback 比例",
                "before": report_data["before_fallback_rate"],
                "after": report_data["fallback_rate"],
                "delta": report_data["fallback_rate"] - report_data["before_fallback_rate"],
                "unit": "%",
                "passed": report_data["fallback_rate"] < 5
            },
            {
                "name": "平均延迟",
                "before": report_data["before_avg_latency_ms"],
                "after": report_data["avg_latency_ms"],
                "delta": report_data["avg_latency_ms"] - report_data["before_avg_latency_ms"],
                "unit": " ms",
                "passed": report_data["avg_latency_ms"] < 2000
            },
            {
                "name": "平均成本",
                "before": report_data["before_avg_cost_per_request"],
                "after": report_data["avg_cost_per_request"],
                "delta": report_data["avg_cost_per_request"] - report_data["before_avg_cost_per_request"],
                "unit": "",
                "passed": report_data["avg_cost_per_request"] < 0.002
            }
        ]
        report_data["comparison_metrics"] = comparison_metrics
        
        # Release Gate 判断
        gates = {
            "schema_pass_rate": report_data["schema_pass_rate"] >= 95,
            "field_accuracy": report_data["field_accuracy"] >= 90,
            "field_consistency": report_data["field_consistency"] >= 85,
            "fallback_rate": report_data["fallback_rate"] < 5,
            "avg_latency_ms": report_data["avg_latency_ms"] < 2000,
            "avg_cost_per_request": report_data["avg_cost_per_request"] < 0.002
        }
        
        passed_gates = [name for name, passed in gates.items() if passed]
        failed_gates = [name for name, passed in gates.items() if not passed]
        all_passed = len(failed_gates) == 0
        
        report_data.update({
            "all_passed": all_passed,
            "passed_gates": passed_gates,
            "failed_gates": failed_gates,
            "total_gates": len(gates)
        })
        
        # 失败用例
        failed_cases = test_results.get("failed_cases", [])
        report_data["failed_cases"] = failed_cases
        report_data["failed_cases_list"] = failed_cases
        
        # 退化分析
        regressed_metrics = [
            metric["name"] for metric in comparison_metrics
            if metric["delta"] < 0 and not metric["passed"]
        ]
        report_data["regressed_metrics"] = regressed_metrics
        report_data["regression_analysis"] = self._analyze_regression(regressed_metrics, comparison_metrics)
        
        return report_data
    
    def _analyze_regression(self, regressed_metrics: List[str], comparison_metrics: List[Dict]) -> str:
        """分析退化原因"""
        if not regressed_metrics:
            return "无退化指标"
        
        analysis = []
        for metric_name in regressed_metrics:
            metric = next((m for m in comparison_metrics if m["name"] == metric_name), None)
            if metric:
                if metric["delta"] < -5:
                    analysis.append(f"{metric_name} 严重退化（下降 {abs(metric['delta']):.2f}{metric['unit']}）")
                else:
                    analysis.append(f"{metric_name} 轻微退化（下降 {abs(metric['delta']):.2f}{metric['unit']}）")
        
        return "；".join(analysis)
    
    def generate_html_report(self, report_data: Dict, output_file: str) -> str:
        """生成 HTML 报告"""
        template = self.env.get_template("regression_report.html")
        html_content = template.render(**report_data)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return output_file
    
    def generate_json_report(self, report_data: Dict, output_file: str) -> str:
        """生成 JSON 报告"""
        # 简化数据（移除 Jinja2 模板语法）
        json_data = {
            "report_version": "1.0.0",
            "version": report_data["version"],
            "test_date": report_data["test_date"],
            "baseline_version": report_data["baseline_version"],
            "current_version": report_data["current_version"],
            "change_type": report_data["change_type"],
            "change_description": report_data["change_description"],
            "golden_set": {
                "total_cases": report_data["total_cases"],
                "passed_cases": report_data["passed_cases"],
                "failed_cases": report_data["failed_cases"]
            },
            "metrics": {
                "schema_pass_rate": {
                    "value": report_data["schema_pass_rate"],
                    "threshold": 95,
                    "passed": report_data["schema_pass_rate"] >= 95,
                    "before": report_data["before_schema_pass_rate"],
                    "after": report_data["schema_pass_rate"],
                    "delta": report_data["schema_pass_rate"] - report_data["before_schema_pass_rate"]
                },
                "field_accuracy": {
                    "value": report_data["field_accuracy"],
                    "threshold": 90,
                    "passed": report_data["field_accuracy"] >= 90,
                    "before": report_data["before_field_accuracy"],
                    "after": report_data["field_accuracy"],
                    "delta": report_data["field_accuracy"] - report_data["before_field_accuracy"]
                }
            },
            "release_gate": {
                "all_passed": report_data["all_passed"],
                "passed_gates": report_data["passed_gates"],
                "failed_gates": report_data["failed_gates"],
                "decision": "允许发布" if report_data["all_passed"] else "禁止发布"
            },
            "failed_cases": report_data["failed_cases_list"],
            "regression_analysis": {
                "regressed_metrics": report_data["regressed_metrics"],
                "analysis": report_data["regression_analysis"]
            },
            "report_generated_at": report_data["report_generated_at"]
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        return output_file
    
    def generate_markdown_report(self, report_data: Dict, output_file: str) -> str:
        """生成 Markdown 报告"""
        template = self.env.get_template("regression_report.md")
        markdown_content = template.render(**report_data)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        return output_file
    
    def generate_all_reports(self, test_results_file: str, baseline_results_file: Optional[str] = None, output_dir: str = "reports") -> Dict[str, str]:
        """生成所有格式的报告"""
        # 加载数据
        test_results = self.load_test_results(test_results_file)
        baseline_results = None
        if baseline_results_file:
            baseline_results = self.load_test_results(baseline_results_file)
        
        # 准备报告数据
        report_data = self.prepare_report_data(test_results, baseline_results)
        
        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 生成报告文件名
        version = report_data["version"]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        html_file = output_path / f"regression_report_{version}_{timestamp}.html"
        json_file = output_path / f"regression_report_{version}_{timestamp}.json"
        markdown_file = output_path / f"regression_report_{version}_{timestamp}.md"
        
        # 生成所有格式的报告
        generated_files = {
            "html": self.generate_html_report(report_data, str(html_file)),
            "json": self.generate_json_report(report_data, str(json_file)),
            "markdown": self.generate_markdown_report(report_data, str(markdown_file))
        }
        
        return generated_files


# 使用示例
if __name__ == "__main__":
    # 初始化生成器
    generator = RegressionReportGenerator(template_dir="templates")
    
    # 生成报告
    generated_files = generator.generate_all_reports(
        test_results_file="regression_results.json",
        baseline_results_file="baseline_results.json",
        output_dir="reports"
    )
    
    print("✅ 报告生成完成:")
    print(f"  - HTML: {generated_files['html']}")
    print(f"  - JSON: {generated_files['json']}")
    print(f"  - Markdown: {generated_files['markdown']}")
```

---

## 📊 报告可视化（图表）

### 使用 Matplotlib 生成图表

```python
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
from matplotlib import font_manager

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def generate_comparison_chart(report_data: Dict, output_file: str):
    """生成 Before/After 对比图表"""
    
    metrics = report_data["comparison_metrics"]
    metric_names = [m["name"] for m in metrics]
    before_values = [m["before"] for m in metrics]
    after_values = [m["after"] for m in metrics]
    
    x = range(len(metric_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar([i - width/2 for i in x], before_values, width, label='Before', color='#90CAF9')
    bars2 = ax.bar([i + width/2 for i in x], after_values, width, label='After', color='#66BB6A')
    
    ax.set_xlabel('指标')
    ax.set_ylabel('数值')
    ax.set_title('Before/After 对比')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=45, ha='right')
    ax.legend()
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_file

def generate_trend_chart(historical_results: List[Dict], output_file: str):
    """生成趋势图表"""
    
    versions = [r["version"] for r in historical_results]
    schema_pass_rates = [r["schema_pass_rate"] for r in historical_results]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(versions, schema_pass_rates, marker='o', linewidth=2, markersize=8)
    ax.axhline(y=95, color='r', linestyle='--', label='阈值 (95%)')
    ax.set_xlabel('版本')
    ax.set_ylabel('Schema Pass Rate (%)')
    ax.set_title('Schema Pass Rate 趋势')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_file
```

---

## 📧 报告分发机制

### 1. 发送到 PR 评论（GitHub）

```python
from github import Github

def post_report_to_pr(report_content: str, pr_number: int, repo_name: str, github_token: str):
    """将报告发送到 PR 评论"""
    g = Github(github_token)
    repo = g.get_repo(repo_name)
    pr = repo.get_pull(pr_number)
    
    # 生成 Markdown 格式的报告摘要
    report_summary = f"""
## 📊 回归测试报告

{report_content}

[查看完整报告](./reports/regression_report.html)
"""
    
    pr.create_issue_comment(report_summary)
    print(f"✅ 报告已发送到 PR #{pr_number}")
```

### 2. 发送到 Slack

```python
import requests
import json

def send_report_to_slack(report_data: Dict, webhook_url: str):
    """将报告发送到 Slack"""
    
    # 生成 Slack 消息
    if report_data["all_passed"]:
        color = "good"  # 绿色
        emoji = "✅"
    else:
        color = "danger"  # 红色
        emoji = "❌"
    
    slack_message = {
        "attachments": [{
            "color": color,
            "title": f"{emoji} 回归测试报告 - {report_data['version']}",
            "fields": [
                {
                    "title": "Schema Pass Rate",
                    "value": f"{report_data['schema_pass_rate']:.2f}%",
                    "short": True
                },
                {
                    "title": "字段级准确率",
                    "value": f"{report_data['field_accuracy']:.2f}%",
                    "short": True
                },
                {
                    "title": "Release Gate",
                    "value": "✅ 通过" if report_data["all_passed"] else "❌ 失败",
                    "short": True
                }
            ],
            "footer": "KYC 项目回归测试系统",
            "ts": int(datetime.now().timestamp())
        }]
    }
    
    response = requests.post(webhook_url, json=slack_message)
    if response.status_code == 200:
        print("✅ 报告已发送到 Slack")
    else:
        print(f"❌ 发送失败: {response.text}")
```

### 3. 发送邮件

```python
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

def send_report_email(report_data: Dict, html_report_file: str, recipients: List[str], smtp_config: Dict):
    """发送报告邮件"""
    
    msg = MIMEMultipart()
    msg['From'] = smtp_config["from_email"]
    msg['To'] = ", ".join(recipients)
    msg['Subject'] = f"回归测试报告 - {report_data['version']} - {'✅ 通过' if report_data['all_passed'] else '❌ 失败'}"
    
    # 邮件正文
    body = f"""
回归测试报告已生成。

版本: {report_data['version']}
测试时间: {report_data['test_date']}
状态: {'✅ 通过' if report_data['all_passed'] else '❌ 失败'}

核心指标:
- Schema Pass Rate: {report_data['schema_pass_rate']:.2f}%
- 字段级准确率: {report_data['field_accuracy']:.2f}%

请查看附件中的完整报告。
"""
    msg.attach(MIMEText(body, 'plain', 'utf-8'))
    
    # 附件（HTML 报告）
    with open(html_report_file, 'rb') as f:
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', f'attachment; filename=regression_report_{report_data["version"]}.html')
        msg.attach(part)
    
    # 发送邮件
    server = smtplib.SMTP(smtp_config["smtp_server"], smtp_config["smtp_port"])
    server.starttls()
    server.login(smtp_config["username"], smtp_config["password"])
    server.send_message(msg)
    server.quit()
    
    print(f"✅ 报告已发送到: {', '.join(recipients)}")
```

---

## 📋 KYC 项目完整示例

### 完整工作流程

```python
"""
KYC 项目回归测试报告生成完整流程
"""

from scripts.generate_regression_report import RegressionReportGenerator
from scripts.visualization import generate_comparison_chart, generate_trend_chart
from scripts.distribution import post_report_to_pr, send_report_to_slack, send_report_email

def generate_and_distribute_report(
    test_results_file: str,
    baseline_results_file: str,
    pr_number: Optional[int] = None,
    slack_webhook: Optional[str] = None,
    email_recipients: Optional[List[str]] = None
):
    """生成并分发报告"""
    
    # 1. 初始化生成器
    generator = RegressionReportGenerator(template_dir="templates")
    
    # 2. 生成所有格式的报告
    generated_files = generator.generate_all_reports(
        test_results_file=test_results_file,
        baseline_results_file=baseline_results_file,
        output_dir="reports"
    )
    
    # 3. 加载报告数据（用于可视化）
    test_results = generator.load_test_results(test_results_file)
    baseline_results = generator.load_test_results(baseline_results_file)
    report_data = generator.prepare_report_data(test_results, baseline_results)
    
    # 4. 生成可视化图表
    chart_file = generate_comparison_chart(report_data, "reports/comparison_chart.png")
    
    # 5. 分发报告
    if pr_number:
        # 发送到 PR 评论
        with open(generated_files["markdown"], 'r', encoding='utf-8') as f:
            markdown_content = f.read()
        post_report_to_pr(markdown_content, pr_number, "your-org/kyc-project", os.environ["GITHUB_TOKEN"])
    
    if slack_webhook:
        # 发送到 Slack
        send_report_to_slack(report_data, slack_webhook)
    
    if email_recipients:
        # 发送邮件
        send_report_email(
            report_data,
            generated_files["html"],
            email_recipients,
            {
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "from_email": "kyc-test@example.com",
                "username": os.environ["SMTP_USERNAME"],
                "password": os.environ["SMTP_PASSWORD"]
            }
        )
    
    return generated_files

# 使用示例
if __name__ == "__main__":
    generated_files = generate_and_distribute_report(
        test_results_file="regression_results.json",
        baseline_results_file="baseline_results.json",
        pr_number=123,
        slack_webhook=os.environ.get("SLACK_WEBHOOK_URL"),
        email_recipients=["team@example.com"]
    )
    
    print("✅ 报告生成和分发完成")
    print(f"  - HTML: {generated_files['html']}")
    print(f"  - JSON: {generated_files['json']}")
    print(f"  - Markdown: {generated_files['markdown']}")
```

---

## 💡 总结

### 核心概念

1. **详细模板**：报告的结构和格式（HTML/JSON/Markdown）
   - 定义了报告应该包含哪些内容
   - 定义了报告的排版和样式
   - 使用占位符（`{{ variable }}`）标记数据位置

2. **生成代码**：Python 代码，将数据填充到模板中
   - 读取测试结果数据
   - 准备报告数据
   - 渲染模板（填充数据）
   - 生成报告文件

3. **报告示例**：最终生成的报告文件
   - HTML：可视化报告（网页格式）
   - JSON：数据报告（程序可读）
   - Markdown：文本报告（GitHub 友好）

### 完整流程

```
测试结果（JSON）
    ↓
生成代码（Python）
    ├─ 读取数据
    ├─ 准备数据
    └─ 渲染模板
    ↓
详细模板（HTML/JSON/Markdown）
    ↓
最终报告（文件）
    ↓
分发报告（PR/Slack/邮件）
```

---

**下一步**：
- 查看 [回归测试与门禁详解](./KYC_Day03_A1_回归测试与门禁详解.md)
- 查看 [Golden Set 存储和使用详解](./KYC_Day03_A1_B1_Golden_Set存储和使用详解.md)
