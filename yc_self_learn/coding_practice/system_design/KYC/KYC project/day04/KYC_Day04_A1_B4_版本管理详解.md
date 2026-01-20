# KYC_Day04_A1_B4: 版本管理和版本对比详解

## 📋 目录
1. [核心问题：如何管理版本和对比版本？](#核心问题如何管理版本和对比版本)
2. [版本管理设计](#版本管理设计)
3. [版本对比功能](#版本对比功能)
4. [版本回滚功能](#版本回滚功能)
5. [版本历史记录](#版本历史记录)
6. [KYC 项目实际案例](#kyc-项目实际案例)

---

## 核心问题：如何管理版本和对比版本？

### 问题场景

**需求**：
- 如何管理不同版本的 Feature Flag 配置？
- 如何对比不同版本的指标？
- 如何回滚到任意历史版本？

**挑战**：
- 需要版本化存储配置
- 需要对比不同版本的指标
- 需要支持快速回滚

---

## 版本管理设计

### 1. Feature Flag 版本管理

#### 1.1 版本号设计

**语义化版本号**：`MAJOR.MINOR.PATCH`

- **MAJOR**：重大变更（如切换模型架构）
- **MINOR**：功能新增（如新增 Prompt 版本）
- **PATCH**：Bug 修复或配置调整

#### 1.2 代码实现

```python
# src/version_manager.py
from typing import Dict, List, Optional
from datetime import datetime
import json
import yaml
from dataclasses import dataclass, asdict

@dataclass
class VersionInfo:
    """版本信息"""
    version: str
    timestamp: str
    description: str
    author: str
    changes: List[str]
    config: Dict

class VersionManager:
    """版本管理器"""
    
    def __init__(self, config_file: str = "config/feature_flags.yaml"):
        self.config_file = config_file
        self.version_history_file = "config/version_history.json"
        self.version_history = self.load_version_history()
        self.current_version = self.get_current_version()
    
    def load_version_history(self) -> List[Dict]:
        """加载版本历史"""
        try:
            with open(self.version_history_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return []
    
    def save_version_history(self):
        """保存版本历史"""
        with open(self.version_history_file, 'w') as f:
            json.dump(self.version_history, f, indent=2)
    
    def get_current_version(self) -> Optional[str]:
        """获取当前版本"""
        if self.version_history:
            return self.version_history[-1]['version']
        return None
    
    def create_version(
        self,
        version: str,
        description: str,
        author: str,
        changes: List[str]
    ) -> VersionInfo:
        """创建新版本"""
        # 加载当前配置
        with open(self.config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # 创建版本信息
        version_info = VersionInfo(
            version=version,
            timestamp=datetime.now().isoformat(),
            description=description,
            author=author,
            changes=changes,
            config=config
        )
        
        # 保存版本历史
        self.version_history.append(asdict(version_info))
        self.save_version_history()
        
        # 更新当前版本
        self.current_version = version
        
        print(f"✅ 创建版本 {version}: {description}")
        return version_info
    
    def get_version(self, version: str) -> Optional[Dict]:
        """获取指定版本"""
        for v in self.version_history:
            if v['version'] == version:
                return v
        return None
    
    def list_versions(self) -> List[Dict]:
        """列出所有版本"""
        return self.version_history
    
    def get_version_diff(self, version1: str, version2: str) -> Dict:
        """对比两个版本的差异"""
        v1 = self.get_version(version1)
        v2 = self.get_version(version2)
        
        if not v1 or not v2:
            return {"error": "版本不存在"}
        
        diff = {
            "version1": version1,
            "version2": version2,
            "differences": []
        }
        
        # 对比配置差异
        config1 = v1['config']
        config2 = v2['config']
        
        # 对比 feature_flags
        flags1 = config1.get('feature_flags', {})
        flags2 = config2.get('feature_flags', {})
        
        # 检查新增的功能
        for flag_name in flags2:
            if flag_name not in flags1:
                diff['differences'].append({
                    "type": "added",
                    "flag": flag_name,
                    "value": flags2[flag_name]
                })
        
        # 检查删除的功能
        for flag_name in flags1:
            if flag_name not in flags2:
                diff['differences'].append({
                    "type": "removed",
                    "flag": flag_name,
                    "value": flags1[flag_name]
                })
        
        # 检查修改的功能
        for flag_name in flags1:
            if flag_name in flags2:
                flag1 = flags1[flag_name]
                flag2 = flags2[flag_name]
                
                if flag1 != flag2:
                    diff['differences'].append({
                        "type": "modified",
                        "flag": flag_name,
                        "old_value": flag1,
                        "new_value": flag2
                    })
        
        return diff
```

---

## 版本对比功能

### 1. 指标对比

#### 1.1 对比维度

| 维度 | 说明 |
|------|------|
| **Schema Pass Rate** | 对比不同版本的 Schema Pass Rate |
| **p95 Latency** | 对比不同版本的 p95 Latency |
| **Error Rate** | 对比不同版本的 Error Rate |
| **Cost per Request** | 对比不同版本的成本 |
| **字段级准确率** | 对比不同版本的字段级准确率 |

#### 1.2 代码实现

```python
# src/version_comparator.py
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import json

class VersionComparator:
    """版本对比器"""
    
    def __init__(self, metrics_storage_file: str = "data/version_metrics.json"):
        self.metrics_storage_file = metrics_storage_file
        self.version_metrics = self.load_version_metrics()
    
    def load_version_metrics(self) -> Dict:
        """加载版本指标"""
        try:
            with open(self.metrics_storage_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
    
    def save_version_metrics(self):
        """保存版本指标"""
        with open(self.metrics_storage_file, 'w') as f:
            json.dump(self.version_metrics, f, indent=2)
    
    def record_version_metrics(self, version: str, metrics: Dict, time_range: Dict):
        """记录版本指标"""
        if version not in self.version_metrics:
            self.version_metrics[version] = []
        
        self.version_metrics[version].append({
            "timestamp": datetime.now().isoformat(),
            "time_range": time_range,
            "metrics": metrics
        })
        
        self.save_version_metrics()
    
    def compare_versions(
        self,
        version1: str,
        version2: str,
        time_range_minutes: int = 60
    ) -> Dict:
        """
        对比两个版本的指标
        
        Args:
            version1: 版本 1
            version2: 版本 2
            time_range_minutes: 对比的时间范围（分钟）
        
        Returns:
            对比结果
        """
        # 获取两个版本的指标
        metrics1 = self.get_version_metrics(version1, time_range_minutes)
        metrics2 = self.get_version_metrics(version2, time_range_minutes)
        
        if not metrics1 or not metrics2:
            return {"error": "版本指标数据不足"}
        
        # 聚合指标
        aggregated1 = self.aggregate_metrics(metrics1)
        aggregated2 = self.aggregate_metrics(metrics2)
        
        # 计算差异
        comparison = {
            "version1": version1,
            "version2": version2,
            "time_range_minutes": time_range_minutes,
            "metrics": {
                "schema_pass_rate": {
                    "version1": aggregated1.get('schema_pass_rate', 0),
                    "version2": aggregated2.get('schema_pass_rate', 0),
                    "diff": aggregated2.get('schema_pass_rate', 0) - aggregated1.get('schema_pass_rate', 0),
                    "diff_percent": (
                        (aggregated2.get('schema_pass_rate', 0) - aggregated1.get('schema_pass_rate', 0)) /
                        aggregated1.get('schema_pass_rate', 1.0) * 100
                    )
                },
                "p95_latency_seconds": {
                    "version1": aggregated1.get('p95_latency_seconds', 0),
                    "version2": aggregated2.get('p95_latency_seconds', 0),
                    "diff": aggregated2.get('p95_latency_seconds', 0) - aggregated1.get('p95_latency_seconds', 0),
                    "diff_percent": (
                        (aggregated2.get('p95_latency_seconds', 0) - aggregated1.get('p95_latency_seconds', 0)) /
                        aggregated1.get('p95_latency_seconds', 1.0) * 100
                    )
                },
                "error_rate": {
                    "version1": aggregated1.get('error_rate', 0),
                    "version2": aggregated2.get('error_rate', 0),
                    "diff": aggregated2.get('error_rate', 0) - aggregated1.get('error_rate', 0),
                    "diff_percent": (
                        (aggregated2.get('error_rate', 0) - aggregated1.get('error_rate', 0)) /
                        aggregated1.get('error_rate', 1.0) * 100
                        if aggregated1.get('error_rate', 0) > 0 else 0
                    )
                },
                "cost_per_request": {
                    "version1": aggregated1.get('cost_per_request', 0),
                    "version2": aggregated2.get('cost_per_request', 0),
                    "diff": aggregated2.get('cost_per_request', 0) - aggregated1.get('cost_per_request', 0),
                    "diff_percent": (
                        (aggregated2.get('cost_per_request', 0) - aggregated1.get('cost_per_request', 0)) /
                        aggregated1.get('cost_per_request', 1.0) * 100
                        if aggregated1.get('cost_per_request', 0) > 0 else 0
                    )
                }
            },
            "summary": self.generate_comparison_summary(aggregated1, aggregated2)
        }
        
        return comparison
    
    def get_version_metrics(self, version: str, time_range_minutes: int) -> List[Dict]:
        """获取版本的指标"""
        if version not in self.version_metrics:
            return []
        
        cutoff_time = datetime.now() - timedelta(minutes=time_range_minutes)
        return [
            m for m in self.version_metrics[version]
            if datetime.fromisoformat(m['timestamp']) >= cutoff_time
        ]
    
    def aggregate_metrics(self, metrics_list: List[Dict]) -> Dict:
        """聚合指标"""
        if not metrics_list:
            return {}
        
        # 合并所有指标
        all_metrics = []
        for m in metrics_list:
            all_metrics.extend(m.get('metrics', {}).get('data_points', []))
        
        if not all_metrics:
            return {}
        
        # 计算聚合值
        total_requests = len(all_metrics)
        passed_requests = sum(1 for m in all_metrics if m.get('schema_pass', False))
        schema_pass_rate = passed_requests / total_requests if total_requests > 0 else 0
        
        latencies = [m.get('latency_ms', 0) / 1000.0 for m in all_metrics]
        latencies.sort()
        p95_index = int(len(latencies) * 0.95)
        p95_latency = latencies[p95_index] if latencies else 0
        
        error_count = sum(1 for m in all_metrics if m.get('error', False))
        error_rate = error_count / total_requests if total_requests > 0 else 0
        
        total_cost = sum(m.get('cost', 0) for m in all_metrics)
        cost_per_request = total_cost / total_requests if total_requests > 0 else 0
        
        return {
            "schema_pass_rate": schema_pass_rate,
            "p95_latency_seconds": p95_latency,
            "error_rate": error_rate,
            "cost_per_request": cost_per_request,
            "total_requests": total_requests
        }
    
    def generate_comparison_summary(self, metrics1: Dict, metrics2: Dict) -> Dict:
        """生成对比摘要"""
        summary = {
            "better_metrics": [],
            "worse_metrics": [],
            "similar_metrics": []
        }
        
        # Schema Pass Rate
        schema_diff = metrics2.get('schema_pass_rate', 0) - metrics1.get('schema_pass_rate', 0)
        if abs(schema_diff) > 0.01:  # 1% 差异
            if schema_diff > 0:
                summary["better_metrics"].append("Schema Pass Rate 提升")
            else:
                summary["worse_metrics"].append("Schema Pass Rate 下降")
        else:
            summary["similar_metrics"].append("Schema Pass Rate 相似")
        
        # p95 Latency
        latency_diff = metrics2.get('p95_latency_seconds', 0) - metrics1.get('p95_latency_seconds', 0)
        if abs(latency_diff) > 1.0:  # 1秒差异
            if latency_diff < 0:
                summary["better_metrics"].append("p95 Latency 降低")
            else:
                summary["worse_metrics"].append("p95 Latency 增加")
        else:
            summary["similar_metrics"].append("p95 Latency 相似")
        
        # Error Rate
        error_diff = metrics2.get('error_rate', 0) - metrics1.get('error_rate', 0)
        if abs(error_diff) > 0.01:  # 1% 差异
            if error_diff < 0:
                summary["better_metrics"].append("Error Rate 降低")
            else:
                summary["worse_metrics"].append("Error Rate 增加")
        else:
            summary["similar_metrics"].append("Error Rate 相似")
        
        return summary
```

---

## 版本回滚功能

### 1. 回滚到历史版本

#### 代码实现

```python
# src/version_rollback.py
from typing import Dict, Optional
from datetime import datetime
import yaml
import json

class VersionRollback:
    """版本回滚管理器"""
    
    def __init__(self, version_manager, config_file: str = "config/feature_flags.yaml"):
        self.version_manager = version_manager
        self.config_file = config_file
        self.rollback_history_file = "config/rollback_history.json"
        self.rollback_history = self.load_rollback_history()
    
    def load_rollback_history(self) -> List[Dict]:
        """加载回滚历史"""
        try:
            with open(self.rollback_history_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return []
    
    def save_rollback_history(self):
        """保存回滚历史"""
        with open(self.rollback_history_file, 'w') as f:
            json.dump(self.rollback_history, f, indent=2)
    
    def rollback_to_version(self, target_version: str, reason: str) -> Dict:
        """
        回滚到指定版本
        
        Returns:
            回滚事件信息
        """
        # 1. 获取目标版本
        target_version_info = self.version_manager.get_version(target_version)
        if not target_version_info:
            raise ValueError(f"版本 {target_version} 不存在")
        
        # 2. 获取当前版本
        current_version = self.version_manager.get_current_version()
        
        # 3. 保存当前配置（备份）
        backup_file = f"config/backup_{current_version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
        with open(self.config_file, 'r') as f:
            current_config = yaml.safe_load(f)
        with open(backup_file, 'w') as f:
            yaml.dump(current_config, f)
        
        # 4. 恢复目标版本的配置
        with open(self.config_file, 'w') as f:
            yaml.dump(target_version_info['config'], f)
        
        # 5. 更新当前版本
        self.version_manager.current_version = target_version
        
        # 6. 记录回滚事件
        rollback_event = {
            "timestamp": datetime.now().isoformat(),
            "from_version": current_version,
            "to_version": target_version,
            "reason": reason,
            "backup_file": backup_file
        }
        self.rollback_history.append(rollback_event)
        self.save_rollback_history()
        
        print(f"✅ 回滚到版本 {target_version}: {reason}")
        return rollback_event
    
    def get_rollback_history(self) -> List[Dict]:
        """获取回滚历史"""
        return self.rollback_history
```

---

## 版本历史记录

### 1. 历史记录设计

#### 代码实现

```python
# src/version_history.py
from typing import Dict, List
from datetime import datetime
import json

class VersionHistory:
    """版本历史记录管理器"""
    
    def __init__(self, version_manager):
        self.version_manager = version_manager
    
    def generate_version_report(self) -> str:
        """生成版本报告"""
        versions = self.version_manager.list_versions()
        
        if not versions:
            return "没有版本历史记录"
        
        report = f"""
# 版本历史报告

## 总览
- **总版本数**: {len(versions)}
- **当前版本**: {self.version_manager.get_current_version()}

## 版本列表
"""
        for v in versions:
            report += f"""
### {v['version']} - {v['timestamp']}
- **作者**: {v['author']}
- **描述**: {v['description']}
- **变更**:
"""
            for change in v.get('changes', []):
                report += f"  - {change}\n"
        
        return report
    
    def get_version_timeline(self) -> List[Dict]:
        """获取版本时间线"""
        versions = self.version_manager.list_versions()
        return [
            {
                "version": v['version'],
                "timestamp": v['timestamp'],
                "description": v['description']
            }
            for v in versions
        ]
```

---

## KYC 项目实际案例

### 案例 1：版本管理和对比

#### 使用示例

```python
# src/kyc_version_service.py
from version_manager import VersionManager
from version_comparator import VersionComparator
from version_rollback import VersionRollback
from version_history import VersionHistory

class KYCVersionService:
    """KYC 版本管理服务"""
    
    def __init__(self):
        self.version_manager = VersionManager()
        self.comparator = VersionComparator()
        self.rollback = VersionRollback(self.version_manager)
        self.history = VersionHistory(self.version_manager)
    
    def create_new_version(
        self,
        version: str,
        description: str,
        author: str,
        changes: List[str]
    ):
        """创建新版本"""
        return self.version_manager.create_version(
            version, description, author, changes
        )
    
    def compare_versions(self, version1: str, version2: str) -> Dict:
        """对比两个版本"""
        return self.comparator.compare_versions(version1, version2)
    
    def rollback_to_version(self, target_version: str, reason: str):
        """回滚到指定版本"""
        return self.rollback.rollback_to_version(target_version, reason)
    
    def get_version_report(self) -> str:
        """获取版本报告"""
        return self.history.generate_version_report()
```

---

## 相关文档

- [KYC_Day04_A1_发布策略与回滚详解.md](./KYC_Day04_A1_发布策略与回滚详解.md) - Feature Flag 版本管理基础
- [KYC_Day03_A1_B7_测试用例版本管理和结果对比详解.md](../day03/KYC_Day03_A1_B7_测试用例版本管理和结果对比详解.md) - 测试用例版本管理

---

## 总结

### 核心要点

1. **版本管理设计**：
   - 语义化版本号（MAJOR.MINOR.PATCH）
   - 版本历史记录
   - 配置版本化存储

2. **版本对比功能**：
   - 对比不同版本的指标
   - 计算差异和百分比
   - 生成对比摘要

3. **版本回滚功能**：
   - 回滚到任意历史版本
   - 自动备份当前配置
   - 记录回滚历史

4. **版本历史记录**：
   - 版本时间线
   - 版本报告
   - 变更追踪
