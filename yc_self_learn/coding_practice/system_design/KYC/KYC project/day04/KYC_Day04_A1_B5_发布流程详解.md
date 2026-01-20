# KYC_Day04_A1_B5: 发布流程详解

## 📋 目录
1. [核心问题：完整的发布流程是什么？](#核心问题完整的发布流程是什么)
2. [发布流程设计](#发布流程设计)
3. [发布审批流程](#发布审批流程)
4. [发布通知机制](#发布通知机制)
5. [发布历史记录](#发布历史记录)
6. [KYC 项目实际案例](#kyc-项目实际案例)

---

## 核心问题：完整的发布流程是什么？

### 问题场景

**需求**：
- 从开发到生产的完整发布流程是什么？
- 谁可以发布？谁可以回滚？
- 如何通知相关团队？

**挑战**：
- 需要明确的流程和角色
- 需要审批机制
- 需要通知机制

---

## 发布流程设计

### 1. 完整发布流程

#### 1.1 流程阶段

```
开发 (Development)
    ↓
测试 (Testing)
    ↓
预发布 (Staging)
    ↓
Canary Release (1% → 5% → 25% → 100%)
    ↓
生产 (Production)
```

#### 1.2 详细流程

```python
# src/release_pipeline.py
from typing import Dict, List, Optional
from datetime import datetime
from enum import Enum
from dataclasses import dataclass

class ReleaseStage(Enum):
    """发布阶段"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    CANARY_1_PERCENT = "canary_1_percent"
    CANARY_5_PERCENT = "canary_5_percent"
    CANARY_25_PERCENT = "canary_25_percent"
    PRODUCTION = "production"
    ROLLBACK = "rollback"

@dataclass
class ReleaseInfo:
    """发布信息"""
    release_id: str
    version: str
    stage: ReleaseStage
    author: str
    description: str
    timestamp: str
    approver: Optional[str] = None
    approval_timestamp: Optional[str] = None

class ReleasePipeline:
    """发布流程管理器"""
    
    def __init__(self):
        self.release_history = []
        self.current_release = None
    
    def create_release(
        self,
        release_id: str,
        version: str,
        author: str,
        description: str
    ) -> ReleaseInfo:
        """创建发布"""
        release = ReleaseInfo(
            release_id=release_id,
            version=version,
            stage=ReleaseStage.DEVELOPMENT,
            author=author,
            description=description,
            timestamp=datetime.now().isoformat()
        )
        
        self.current_release = release
        self.release_history.append(release)
        
        print(f"✅ 创建发布 {release_id}: {description}")
        return release
    
    def advance_stage(
        self,
        release_id: str,
        target_stage: ReleaseStage,
        approver: Optional[str] = None
    ) -> bool:
        """
        推进到下一阶段
        
        Returns:
            True if successful, False otherwise
        """
        release = self.get_release(release_id)
        if not release:
            return False
        
        # 检查是否需要审批
        if self.requires_approval(release.stage, target_stage):
            if not approver:
                print("❌ 需要审批才能推进")
                return False
            
            release.approver = approver
            release.approval_timestamp = datetime.now().isoformat()
        
        # 推进阶段
        release.stage = target_stage
        release.timestamp = datetime.now().isoformat()
        
        print(f"✅ 发布 {release_id} 推进到 {target_stage.value}")
        return True
    
    def requires_approval(self, current_stage: ReleaseStage, target_stage: ReleaseStage) -> bool:
        """检查是否需要审批"""
        # 从 Staging 到 Canary 需要审批
        if current_stage == ReleaseStage.STAGING and target_stage == ReleaseStage.CANARY_1_PERCENT:
            return True
        
        # 从 Canary 25% 到 Production 需要审批
        if current_stage == ReleaseStage.CANARY_25_PERCENT and target_stage == ReleaseStage.PRODUCTION:
            return True
        
        return False
    
    def get_release(self, release_id: str) -> Optional[ReleaseInfo]:
        """获取发布信息"""
        for r in self.release_history:
            if r.release_id == release_id:
                return r
        return None
    
    def rollback_release(self, release_id: str, reason: str) -> bool:
        """回滚发布"""
        release = self.get_release(release_id)
        if not release:
            return False
        
        release.stage = ReleaseStage.ROLLBACK
        release.timestamp = datetime.now().isoformat()
        
        print(f"🔄 发布 {release_id} 已回滚: {reason}")
        return True
```

---

## 发布审批流程

### 1. 审批角色设计

#### 1.1 角色定义

| 角色 | 权限 | 可以审批的阶段 |
|------|------|----------------|
| **Developer** | 创建发布、推进到 Testing | Development → Testing |
| **QA** | 推进到 Staging | Testing → Staging |
| **Tech Lead** | 推进到 Canary、Production | Staging → Canary, Canary → Production |
| **PM** | 推进到 Production（业务审批） | Canary → Production |

#### 1.2 代码实现

```python
# src/release_approval.py
from typing import Dict, List, Optional
from datetime import datetime
from enum import Enum
from dataclasses import dataclass

class ApprovalRole(Enum):
    """审批角色"""
    DEVELOPER = "developer"
    QA = "qa"
    TECH_LEAD = "tech_lead"
    PM = "pm"

@dataclass
class ApprovalRequest:
    """审批请求"""
    request_id: str
    release_id: str
    from_stage: ReleaseStage
    to_stage: ReleaseStage
    requester: str
    timestamp: str
    approver: Optional[str] = None
    approval_status: Optional[str] = None
    approval_timestamp: Optional[str] = None
    comments: Optional[str] = None

class ReleaseApproval:
    """发布审批管理器"""
    
    def __init__(self):
        self.approval_rules = {
            (ReleaseStage.DEVELOPMENT, ReleaseStage.TESTING): [ApprovalRole.DEVELOPER],
            (ReleaseStage.TESTING, ReleaseStage.STAGING): [ApprovalRole.QA],
            (ReleaseStage.STAGING, ReleaseStage.CANARY_1_PERCENT): [ApprovalRole.TECH_LEAD],
            (ReleaseStage.CANARY_25_PERCENT, ReleaseStage.PRODUCTION): [ApprovalRole.TECH_LEAD, ApprovalRole.PM]
        }
        self.approval_requests = []
    
    def create_approval_request(
        self,
        release_id: str,
        from_stage: ReleaseStage,
        to_stage: ReleaseStage,
        requester: str
    ) -> ApprovalRequest:
        """创建审批请求"""
        request = ApprovalRequest(
            request_id=f"approval_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            release_id=release_id,
            from_stage=from_stage,
            to_stage=to_stage,
            requester=requester,
            timestamp=datetime.now().isoformat()
        )
        
        self.approval_requests.append(request)
        
        # 通知审批人
        self.notify_approvers(request)
        
        return request
    
    def approve(
        self,
        request_id: str,
        approver: str,
        role: ApprovalRole,
        comments: Optional[str] = None
    ) -> bool:
        """审批通过"""
        request = self.get_approval_request(request_id)
        if not request:
            return False
        
        # 检查是否有审批权限
        required_roles = self.approval_rules.get(
            (request.from_stage, request.to_stage), []
        )
        if role not in required_roles:
            print(f"❌ {role.value} 没有审批权限")
            return False
        
        # 检查是否已经审批
        if request.approval_status:
            print(f"⚠️ 审批请求已经处理: {request.approval_status}")
            return False
        
        # 审批通过
        request.approver = approver
        request.approval_status = "approved"
        request.approval_timestamp = datetime.now().isoformat()
        request.comments = comments
        
        print(f"✅ 审批通过: {approver} ({role.value})")
        return True
    
    def reject(
        self,
        request_id: str,
        approver: str,
        role: ApprovalRole,
        comments: str
    ) -> bool:
        """审批拒绝"""
        request = self.get_approval_request(request_id)
        if not request:
            return False
        
        # 检查是否有审批权限
        required_roles = self.approval_rules.get(
            (request.from_stage, request.to_stage), []
        )
        if role not in required_roles:
            print(f"❌ {role.value} 没有审批权限")
            return False
        
        # 审批拒绝
        request.approver = approver
        request.approval_status = "rejected"
        request.approval_timestamp = datetime.now().isoformat()
        request.comments = comments
        
        print(f"❌ 审批拒绝: {approver} ({role.value}) - {comments}")
        return True
    
    def get_approval_request(self, request_id: str) -> Optional[ApprovalRequest]:
        """获取审批请求"""
        for r in self.approval_requests:
            if r.request_id == request_id:
                return r
        return None
    
    def notify_approvers(self, request: ApprovalRequest):
        """通知审批人"""
        required_roles = self.approval_rules.get(
            (request.from_stage, request.to_stage), []
        )
        
        print(f"📧 通知审批人: {[r.value for r in required_roles]}")
        # 实际应该发送到 Slack、邮件等
```

---

## 发布通知机制

### 1. 通知设计

#### 1.1 通知时机

| 时机 | 通知内容 | 通知对象 |
|------|----------|----------|
| **创建发布** | 发布创建通知 | 开发团队 |
| **推进阶段** | 阶段推进通知 | 相关团队 |
| **审批请求** | 审批请求通知 | 审批人 |
| **审批结果** | 审批结果通知 | 请求人、相关团队 |
| **回滚** | 回滚通知 | 所有团队 |
| **异常** | 异常告警 | 运维团队、Tech Lead |

#### 1.2 代码实现

```python
# src/release_notification.py
from typing import Dict, List
from datetime import datetime
from enum import Enum

class NotificationChannel(Enum):
    """通知渠道"""
    SLACK = "slack"
    EMAIL = "email"
    WEBHOOK = "webhook"

class ReleaseNotification:
    """发布通知管理器"""
    
    def __init__(self):
        self.notification_channels = [
            NotificationChannel.SLACK,
            NotificationChannel.EMAIL
        ]
        self.notification_history = []
    
    def notify_release_created(self, release_info: ReleaseInfo):
        """通知发布创建"""
        message = f"""
🚀 新发布已创建

发布 ID: {release_info.release_id}
版本: {release_info.version}
描述: {release_info.description}
作者: {release_info.author}
时间: {release_info.timestamp}
"""
        self.send_notification("开发团队", message)
    
    def notify_stage_advanced(self, release_info: ReleaseInfo, from_stage: ReleaseStage, to_stage: ReleaseStage):
        """通知阶段推进"""
        message = f"""
✅ 发布阶段推进

发布 ID: {release_info.release_id}
版本: {release_info.version}
阶段: {from_stage.value} → {to_stage.value}
时间: {release_info.timestamp}
"""
        self.send_notification("相关团队", message)
    
    def notify_approval_request(self, approval_request: ApprovalRequest):
        """通知审批请求"""
        message = f"""
📋 审批请求

请求 ID: {approval_request.request_id}
发布 ID: {approval_request.release_id}
阶段: {approval_request.from_stage.value} → {approval_request.to_stage.value}
请求人: {approval_request.requester}
时间: {approval_request.timestamp}
"""
        self.send_notification("审批人", message)
    
    def notify_approval_result(self, approval_request: ApprovalRequest):
        """通知审批结果"""
        status_emoji = "✅" if approval_request.approval_status == "approved" else "❌"
        message = f"""
{status_emoji} 审批结果

请求 ID: {approval_request.request_id}
发布 ID: {approval_request.release_id}
状态: {approval_request.approval_status}
审批人: {approval_request.approver}
时间: {approval_request.approval_timestamp}
备注: {approval_request.comments or "无"}
"""
        self.send_notification("请求人、相关团队", message)
    
    def notify_rollback(self, release_info: ReleaseInfo, reason: str):
        """通知回滚"""
        message = f"""
🔄 发布回滚

发布 ID: {release_info.release_id}
版本: {release_info.version}
原因: {reason}
时间: {release_info.timestamp}
"""
        self.send_notification("所有团队", message)
    
    def notify_alert(self, alert_type: str, message: str):
        """通知告警"""
        alert_message = f"""
🚨 告警: {alert_type}

{message}
时间: {datetime.now().isoformat()}
"""
        self.send_notification("运维团队、Tech Lead", alert_message)
    
    def send_notification(self, recipients: str, message: str):
        """发送通知"""
        notification = {
            "timestamp": datetime.now().isoformat(),
            "recipients": recipients,
            "message": message,
            "channels": [c.value for c in self.notification_channels]
        }
        
        self.notification_history.append(notification)
        
        # 实际应该发送到 Slack、邮件等
        print(f"📧 发送通知到 {recipients}:")
        print(message)
```

---

## 发布历史记录

### 1. 历史记录设计

#### 代码实现

```python
# src/release_history.py
from typing import Dict, List
from datetime import datetime
import json

class ReleaseHistory:
    """发布历史记录管理器"""
    
    def __init__(self, release_pipeline):
        self.release_pipeline = release_pipeline
        self.history_file = "data/release_history.json"
        self.history = self.load_history()
    
    def load_history(self) -> List[Dict]:
        """加载历史记录"""
        try:
            with open(self.history_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return []
    
    def save_history(self):
        """保存历史记录"""
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def record_release(self, release_info: ReleaseInfo):
        """记录发布"""
        self.history.append({
            "release_id": release_info.release_id,
            "version": release_info.version,
            "stage": release_info.stage.value,
            "author": release_info.author,
            "description": release_info.description,
            "timestamp": release_info.timestamp,
            "approver": release_info.approver,
            "approval_timestamp": release_info.approval_timestamp
        })
        self.save_history()
    
    def generate_report(self, days: int = 30) -> str:
        """生成发布报告"""
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_releases = [
            r for r in self.history
            if datetime.fromisoformat(r['timestamp']) >= cutoff_date
        ]
        
        if not recent_releases:
            return f"最近 {days} 天没有发布记录"
        
        total_releases = len(recent_releases)
        production_releases = sum(
            1 for r in recent_releases
            if r['stage'] == 'production'
        )
        rollback_releases = sum(
            1 for r in recent_releases
            if r['stage'] == 'rollback'
        )
        
        report = f"""
# 发布历史报告

## 统计信息（最近 {days} 天）
- **总发布数**: {total_releases}
- **生产发布**: {production_releases}
- **回滚次数**: {rollback_releases}
- **成功率**: {(production_releases / total_releases * 100):.1f}%

## 最近发布记录
"""
        for r in recent_releases[-10:]:  # 最近 10 次
            report += f"""
### {r['release_id']} - {r['timestamp']}
- **版本**: {r['version']}
- **阶段**: {r['stage']}
- **作者**: {r['author']}
- **描述**: {r['description']}
"""
        
        return report
```

---

## KYC 项目实际案例

### 案例 1：完整发布流程

#### 使用示例

```python
# src/kyc_release_service.py
from release_pipeline import ReleasePipeline, ReleaseStage
from release_approval import ReleaseApproval, ApprovalRole
from release_notification import ReleaseNotification
from release_history import ReleaseHistory

class KYCReleaseService:
    """KYC 发布服务"""
    
    def __init__(self):
        self.pipeline = ReleasePipeline()
        self.approval = ReleaseApproval()
        self.notification = ReleaseNotification()
        self.history = ReleaseHistory(self.pipeline)
    
    def create_release(self, release_id: str, version: str, author: str, description: str):
        """创建发布"""
        release = self.pipeline.create_release(release_id, version, author, description)
        self.notification.notify_release_created(release)
        self.history.record_release(release)
        return release
    
    def advance_to_staging(self, release_id: str, approver: str):
        """推进到 Staging"""
        release = self.pipeline.get_release(release_id)
        if not release:
            return False
        
        # 创建审批请求
        approval_request = self.approval.create_approval_request(
            release_id, release.stage, ReleaseStage.STAGING, release.author
        )
        
        # 审批
        if self.approval.approve(approval_request.request_id, approver, ApprovalRole.QA):
            # 推进阶段
            if self.pipeline.advance_stage(release_id, ReleaseStage.STAGING, approver):
                release = self.pipeline.get_release(release_id)
                self.notification.notify_stage_advanced(
                    release, ReleaseStage.TESTING, ReleaseStage.STAGING
                )
                self.notification.notify_approval_result(approval_request)
                self.history.record_release(release)
                return True
        
        return False
    
    def rollback_release(self, release_id: str, reason: str):
        """回滚发布"""
        if self.pipeline.rollback_release(release_id, reason):
            release = self.pipeline.get_release(release_id)
            self.notification.notify_rollback(release, reason)
            self.history.record_release(release)
            return True
        return False
    
    def get_release_report(self) -> str:
        """获取发布报告"""
        return self.history.generate_report()
```

---

## 相关文档

- [KYC_Day04_A1_发布策略与回滚详解.md](./KYC_Day04_A1_发布策略与回滚详解.md) - 发布策略基础
- [KYC_Day04_A1_B2_Canary_Release监控详解.md](./KYC_Day04_A1_B2_Canary_Release监控详解.md) - Canary Release 监控
- [KYC_Day04_A1_B3_Rollback自动化详解.md](./KYC_Day04_A1_B3_Rollback自动化详解.md) - Rollback 自动化

---

## 总结

### 核心要点

1. **发布流程设计**：
   - Development → Testing → Staging → Canary → Production
   - 每个阶段都有明确的进入条件

2. **发布审批流程**：
   - 不同角色有不同的审批权限
   - 关键阶段需要审批才能推进

3. **发布通知机制**：
   - 创建、推进、审批、回滚都有通知
   - 支持多种通知渠道

4. **发布历史记录**：
   - 记录所有发布事件
   - 生成发布报告
   - 分析发布趋势
