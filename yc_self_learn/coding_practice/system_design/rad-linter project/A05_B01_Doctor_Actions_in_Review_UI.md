# A05_B01: Doctor Actions in Review UI
# A05_B01: 医生在审查UI中的操作详解

**Author**：Yanda Cheng  
**Project**：Rad-Linter  
**Purpose**：详解医生在审查UI中的三种操作类型及其在Post-Training中的应用  
**Related Document**：A05_Post_Training_Mechanism.md (Section: Feedback Collection Mechanism)

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Three Action Types](#three-action-types)
3. [Action Details & Use Cases](#action-details--use-cases)
4. [Data Structure & Schema](#data-structure--schema)
5. [UI Design Considerations](#ui-design-considerations)
6. [Feedback Collection Implementation](#feedback-collection-implementation)
7. [Data Flow to Post-Training](#data-flow-to-post-training)
8. [Best Practices](#best-practices)

---

## Overview

### Core Concept

**医生在审查UI中的操作**是Post-Training反馈数据的核心来源。医生通过三种操作类型（Adopt、Ignore、Edit）来表达对系统预测的反馈，这些反馈数据将用于：

1. **Adopt（接受）** → 正样本 → SFT训练
2. **Ignore（忽略）** → 误报 → 规则优化
3. **Edit（编辑）** → 专家标注 → DPO训练

### Feedback Loop

```
Production Environment (On-Prem)
    ↓
Doctor Reviews Issue in UI
    ↓
Doctor Action (Adopt/Ignore/Edit)
    ↓
Feedback Record Created
    ↓
Stored in PostgreSQL (On-Prem)
    ↓
Synced to S3 (Daily Batch)
    ↓
Classified for Post-Training
    ↓
Model Improvement (SFT/DPO/Rule Optimization)
```

---

## Three Action Types

### Action Type Summary

| Action | 中文 | 含义 | 使用场景 | Post-Training用途 |
|--------|------|------|----------|------------------|
| **Adopt** | 接受 | 医生同意系统预测 | 系统预测正确，医生认可 | SFT（监督微调） |
| **Ignore** | 忽略 | 医生认为系统预测错误（误报） | 系统预测不正确，但不需要修正 | 规则优化 |
| **Edit** | 编辑 | 医生提供专家修正 | 系统预测部分正确，需要专家修正 | DPO（偏好学习） |

### Visual Representation

```
┌─────────────────────────────────────────────────────────┐
│              Doctor Review UI                            │
│                                                          │
│  Issue: Laterality Mismatch                            │
│  Severity: High                                         │
│  Confidence: 0.95                                       │
│  Supporting Facts: [vf_001, rf_002]                     │
│                                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│  │  Adopt   │  │  Ignore   │  │   Edit   │            │
│  │  (接受)  │  │  (忽略)   │  │  (编辑)   │            │
│  └──────────┘  └──────────┘  └──────────┘            │
│                                                          │
│  [If Ignore: Reason required]                           │
│  [If Edit: Corrected content required]                 │
└─────────────────────────────────────────────────────────┘
```

---

## Action Details & Use Cases

### 1. Adopt（接受）

#### Definition

**Adopt** 表示医生完全同意系统的预测，认为系统检测到的问题是正确的。

#### Use Cases

✅ **典型场景**：
- 系统检测到"左右侧不匹配"（Laterality Mismatch），医生确认确实存在此问题
- 系统检测到"遗漏测量"（Missing Measurement），医生确认确实遗漏了
- 系统检测到"矛盾描述"（Contradiction），医生确认确实存在矛盾

✅ **特征**：
- 系统预测置信度高（confidence > 0.9）
- 医生快速确认，无需修改
- 系统预测与医生判断一致

#### Example

```json
{
  "case_id": "case_001",
  "issue_id": "issue_001",
  "action": "adopt",
  "reason": null,
  "corrected_content": null,
  "reviewer_id": "doctor_001",
  "timestamp": "2025-01-01T10:00:00Z",
  "model_version": "lora_model_v1.0",
  "original_prediction": {
    "issue_type": "laterality_mismatch",
    "severity": "high",
    "confidence": 0.95,
    "supporting_facts": ["vf_001", "rf_002"]
  },
  "doctor_feedback": {
    "action": "adopt",
    "reason": null,
    "corrected_severity": null
  }
}
```

#### Post-Training Usage

- **数据分类**：Accept Cases（正样本）
- **训练方法**：SFT (Supervised Fine-Tuning)
- **目的**：强化模型已经做对的地方，让模型更自信地预测类似问题

---

### 2. Ignore（忽略）

#### Definition

**Ignore** 表示医生认为系统预测是错误的（误报），但不需要提供修正内容。

#### Use Cases

✅ **典型场景**：
- 系统检测到"遗漏测量"，但该测量不适用于此检查类型
- 系统检测到"矛盾描述"，但实际上是正常的医学表达
- 系统检测到"左右侧不匹配"，但视觉事实置信度太低，不可靠

✅ **特征**：
- 系统预测置信度可能较低（confidence < 0.8）
- 医生认为这是误报（False Positive）
- 医生提供忽略原因（reason required）

#### Example

```json
{
  "case_id": "case_002",
  "issue_id": "issue_002",
  "action": "ignore",
  "reason": "False positive - visual fact confidence too low",
  "corrected_content": null,
  "reviewer_id": "doctor_001",
  "timestamp": "2025-01-01T10:05:00Z",
  "model_version": "lora_model_v1.0",
  "original_prediction": {
    "issue_type": "missing_measurement",
    "severity": "med",
    "confidence": 0.75,
    "supporting_facts": ["vf_003"]
  },
  "doctor_feedback": {
    "action": "ignore",
    "reason": "Not relevant for this exam type",
    "corrected_severity": null
  }
}
```

#### Common Ignore Reasons

| Reason Category | Example Reasons | Analysis Action |
|----------------|-----------------|-----------------|
| **False Positive** | "Visual fact confidence too low" | 调整置信度阈值 |
| **Not Relevant** | "Not relevant for this exam type" | 添加规则过滤条件 |
| **Edge Case** | "Normal medical expression variation" | 更新规则例外情况 |
| **Rule Too Strict** | "Threshold too sensitive" | 调整规则阈值 |

#### Post-Training Usage

- **数据分类**：Ignore Cases（误报）
- **训练方法**：Rule Optimization（规则优化，非模型训练）
- **目的**：降低误报率，调整规则阈值和过滤条件

---

### 3. Edit（编辑）

#### Definition

**Edit** 表示医生认为系统预测部分正确，但需要提供专家修正。

#### Use Cases

✅ **典型场景**：
- 系统检测到"矛盾描述"，但严重程度判断错误（应该是high，系统判断为med）
- 系统检测到"遗漏测量"，但遗漏的具体内容需要修正
- 系统检测到"左右侧不匹配"，但支持事实需要补充

✅ **特征**：
- 系统预测方向正确，但细节需要修正
- 医生提供专家级别的修正内容
- 形成"原始预测 vs 专家修正"的偏好对（Preference Pair）

#### Example

```json
{
  "case_id": "case_003",
  "issue_id": "issue_003",
  "action": "edit",
  "reason": "Severity should be high, not med",
  "corrected_content": {
    "issue_type": "contradiction",
    "severity": "high",
    "confidence": 0.95,
    "supporting_facts": ["vf_004", "rf_005", "rf_006"],
    "explanation": "Expert-corrected explanation: The contradiction is critical for patient safety"
  },
  "reviewer_id": "doctor_002",
  "timestamp": "2025-01-01T10:10:00Z",
  "model_version": "lora_model_v1.0",
  "original_prediction": {
    "issue_type": "contradiction",
    "severity": "med",
    "confidence": 0.85,
    "supporting_facts": ["vf_004", "rf_005"],
    "explanation": "Original model explanation"
  },
  "doctor_feedback": {
    "action": "edit",
    "reason": "Severity should be high, not med",
    "corrected_severity": "high",
    "corrected_supporting_facts": ["vf_004", "rf_005", "rf_006"],
    "corrected_explanation": "Expert-corrected explanation: The contradiction is critical for patient safety"
  }
}
```

#### Preference Pair Formation

Edit操作会形成偏好对（Preference Pair），用于DPO训练：

```json
{
  "input": {
    "visual_facts": [...],
    "report_facts": [...]
  },
  "preferred": {
    "issue_type": "contradiction",
    "severity": "high",
    "explanation": "Expert-corrected explanation"
  },
  "rejected": {
    "issue_type": "contradiction",
    "severity": "med",
    "explanation": "Original model explanation"
  },
  "case_id": "case_003"
}
```

#### Post-Training Usage

- **数据分类**：Review Cases（专家标注）
- **训练方法**：DPO (Direct Preference Optimization)
- **目的**：学习专家的偏好和修正，让模型输出更接近专家判断

---

## Data Structure & Schema

### Complete Feedback Record Schema

```python
from pydantic import BaseModel, Field
from typing import Optional, List, Literal
from datetime import datetime
from enum import Enum

class ActionType(str, Enum):
    """Doctor action types"""
    ADOPT = "adopt"
    IGNORE = "ignore"
    EDIT = "edit"

class IssueType(str, Enum):
    """Issue types"""
    LATERALITY_MISMATCH = "laterality_mismatch"
    MISSING_MEASUREMENT = "missing_measurement"
    CONTRADICTION = "contradiction"
    OMISSION = "omission"
    # ... more issue types

class Severity(str, Enum):
    """Severity levels"""
    LOW = "low"
    MED = "med"
    HIGH = "high"
    CRITICAL = "critical"

class OriginalPrediction(BaseModel):
    """Original model prediction"""
    issue_type: IssueType
    severity: Severity
    confidence: float = Field(ge=0.0, le=1.0)
    supporting_facts: List[str]
    explanation: Optional[str] = None
    recommended_action: Optional[str] = None

class DoctorFeedback(BaseModel):
    """Doctor feedback on the prediction"""
    action: ActionType
    reason: Optional[str] = Field(None, description="Required for 'ignore' action")
    corrected_severity: Optional[Severity] = None
    corrected_content: Optional[dict] = None  # Full corrected prediction if "edit"
    corrected_supporting_facts: Optional[List[str]] = None
    corrected_explanation: Optional[str] = None

class FeedbackRecord(BaseModel):
    """Complete feedback record from production"""
    
    # Case identification
    case_id: str = Field(..., description="Unique case identifier")
    issue_id: str = Field(..., description="Unique issue identifier within case")
    
    # Doctor action
    action: ActionType = Field(..., description="Doctor action: adopt, ignore, or edit")
    reason: Optional[str] = Field(
        None, 
        description="Reason for action (required for 'ignore')"
    )
    corrected_content: Optional[dict] = Field(
        None,
        description="Corrected content if action is 'edit'"
    )
    
    # Reviewer info
    reviewer_id: str = Field(..., description="Doctor/Reviewer identifier")
    reviewer_role: Literal["doctor", "expert", "supervisor"] = Field(
        default="doctor",
        description="Role of the reviewer"
    )
    timestamp: datetime = Field(default_factory=datetime.now)
    
    # Model version tracking
    model_version: str = Field(..., description="Model version that made the prediction")
    prompt_version: str = Field(..., description="Prompt version used")
    rule_version: str = Field(..., description="Rule version used")
    
    # Original prediction
    original_prediction: OriginalPrediction = Field(
        ...,
        description="Original model prediction"
    )
    
    # Doctor feedback
    doctor_feedback: DoctorFeedback = Field(
        ...,
        description="Doctor feedback on the prediction"
    )
    
    # Context (for analysis and training)
    visual_facts: List[dict] = Field(
        default_factory=list,
        description="Visual facts extracted from images"
    )
    report_facts: List[dict] = Field(
        default_factory=list,
        description="Report facts extracted from text"
    )
    report_text: str = Field(..., description="Full report text")
    
    # Metadata
    department: str = Field(..., description="Department name")
    exam_type: str = Field(..., description="Exam type (e.g., chest_xray, ct_chest)")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        json_schema_extra = {
            "example": {
                "case_id": "case_001",
                "issue_id": "issue_001",
                "action": "ignore",
                "reason": "False positive - visual fact confidence too low",
                "corrected_content": None,
                "reviewer_id": "doctor_001",
                "reviewer_role": "doctor",
                "timestamp": "2025-01-01T10:00:00Z",
                "model_version": "lora_model_v1.0",
                "prompt_version": "prompt_v1.2",
                "rule_version": "rules_v1.0",
                "original_prediction": {
                    "issue_type": "laterality_mismatch",
                    "severity": "high",
                    "confidence": 0.95,
                    "supporting_facts": ["vf_001", "rf_002"],
                    "explanation": "Left side mentioned in report but image shows right side"
                },
                "doctor_feedback": {
                    "action": "ignore",
                    "reason": "False positive - visual fact confidence too low",
                    "corrected_severity": None
                },
                "visual_facts": [
                    {"id": "vf_001", "type": "laterality", "value": "right", "confidence": 0.65}
                ],
                "report_facts": [
                    {"id": "rf_002", "type": "laterality", "value": "left", "confidence": 0.95}
                ],
                "report_text": "Chest X-ray shows...",
                "department": "radiology",
                "exam_type": "chest_xray",
                "metadata": {}
            }
        }
```

### Validation Rules

```python
from pydantic import validator

class FeedbackRecord(BaseModel):
    # ... fields ...
    
    @validator('reason')
    def reason_required_for_ignore(cls, v, values):
        """Reason is required for 'ignore' action"""
        if values.get('action') == ActionType.IGNORE and not v:
            raise ValueError("Reason is required for 'ignore' action")
        return v
    
    @validator('corrected_content')
    def corrected_content_required_for_edit(cls, v, values):
        """Corrected content is required for 'edit' action"""
        if values.get('action') == ActionType.EDIT and not v:
            raise ValueError("Corrected content is required for 'edit' action")
        return v
```

---

## UI Design Considerations

### Review UI Layout

```
┌─────────────────────────────────────────────────────────────┐
│  Case: case_001 | Issue: issue_001                          │
│  Department: Radiology | Exam Type: Chest X-ray              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─ System Prediction ─────────────────────────────────┐  │
│  │ Issue Type: Laterality Mismatch                      │  │
│  │ Severity: High                                       │  │
│  │ Confidence: 0.95                                     │  │
│  │                                                      │  │
│  │ Supporting Facts:                                    │  │
│  │ • Visual Fact vf_001: Right side (confidence: 0.65)│  │
│  │ • Report Fact rf_002: Left side (confidence: 0.95) │  │
│  │                                                      │  │
│  │ Explanation:                                         │  │
│  │ Left side mentioned in report but image shows       │  │
│  │ right side                                          │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌─ Doctor Action ─────────────────────────────────────┐  │
│  │                                                      │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────┐ │  │
│  │  │   ✓ Adopt    │  │   ✗ Ignore   │  │  ✏ Edit   │ │  │
│  │  │   (接受)     │  │   (忽略)     │  │  (编辑)   │ │  │
│  │  └──────────────┘  └──────────────┘  └──────────┘ │  │
│  │                                                      │  │
│  │  [If Ignore selected:]                              │  │
│  │  ┌──────────────────────────────────────────────┐  │  │
│  │  │ Reason (required):                           │  │  │
│  │  │ [Dropdown or Text Input]                      │  │  │
│  │  │ • False positive - visual fact confidence    │  │  │
│  │  │   too low                                      │  │  │
│  │  │ • Not relevant for this exam type             │  │  │
│  │  │ • Normal medical expression variation         │  │  │
│  │  │ • Other: [________________]                   │  │  │
│  │  └──────────────────────────────────────────────┘  │  │
│  │                                                      │  │
│  │  [If Edit selected:]                                │  │
│  │  ┌──────────────────────────────────────────────┐  │  │
│  │  │ Corrected Prediction:                         │  │  │
│  │  │ Severity: [High ▼]                            │  │  │
│  │  │ Supporting Facts: [Add/Remove facts]          │  │  │
│  │  │ Explanation: [Text area for correction]      │  │  │
│  │  └──────────────────────────────────────────────┘  │  │
│  │                                                      │  │
│  │  [Submit Button]                                     │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### UI Interaction Flow

```python
# Frontend UI State Machine
class ReviewUIState:
    """State machine for review UI"""
    
    def __init__(self):
        self.selected_action = None
        self.reason = None
        self.corrected_content = None
        self.is_submitting = False
    
    def select_action(self, action: ActionType):
        """Select an action (adopt/ignore/edit)"""
        self.selected_action = action
        
        # Clear previous inputs
        if action != ActionType.IGNORE:
            self.reason = None
        if action != ActionType.EDIT:
            self.corrected_content = None
    
    def set_reason(self, reason: str):
        """Set reason (required for ignore)"""
        if self.selected_action == ActionType.IGNORE:
            self.reason = reason
    
    def set_corrected_content(self, content: dict):
        """Set corrected content (required for edit)"""
        if self.selected_action == ActionType.EDIT:
            self.corrected_content = content
    
    def can_submit(self) -> bool:
        """Check if form can be submitted"""
        if not self.selected_action:
            return False
        
        if self.selected_action == ActionType.IGNORE:
            return bool(self.reason)
        
        if self.selected_action == ActionType.EDIT:
            return bool(self.corrected_content)
        
        return True  # Adopt can always be submitted
    
    async def submit_feedback(self, case_id: str, issue_id: str):
        """Submit feedback to backend"""
        if not self.can_submit():
            raise ValueError("Cannot submit: missing required fields")
        
        self.is_submitting = True
        
        try:
            feedback = {
                "case_id": case_id,
                "issue_id": issue_id,
                "action": self.selected_action.value,
                "reason": self.reason,
                "corrected_content": self.corrected_content
            }
            
            response = await api.post("/api/v1/feedback", json=feedback)
            return response
        finally:
            self.is_submitting = False
```

### UI Best Practices

#### 1. **Clear Action Buttons**

- **Adopt**: Green button with checkmark icon
- **Ignore**: Red button with X icon
- **Edit**: Blue button with edit icon

#### 2. **Contextual Help**

- Show tooltips explaining each action
- Provide examples of when to use each action
- Show what happens to the feedback (e.g., "This will be used for SFT training")

#### 3. **Quick Actions**

- Provide keyboard shortcuts (e.g., `A` for Adopt, `I` for Ignore, `E` for Edit)
- Support bulk actions for similar issues

#### 4. **Feedback Confirmation**

- Show confirmation dialog before submitting
- Display feedback summary before final submission
- Show success message after submission

---

## Feedback Collection Implementation

### Backend API Endpoint

```python
from fastapi import APIRouter, HTTPException, Depends
from typing import Optional
from datetime import datetime

router = APIRouter(prefix="/api/v1/feedback", tags=["feedback"])

@router.post("/", response_model=FeedbackRecord)
async def submit_feedback(
    feedback: FeedbackRecord,
    current_user: dict = Depends(get_current_user)
):
    """
    Submit doctor feedback on a system prediction
    
    Args:
        feedback: Feedback record with action, reason, etc.
        current_user: Current authenticated user (doctor)
    
    Returns:
        Created feedback record
    """
    # Validate feedback
    if feedback.action == ActionType.IGNORE and not feedback.reason:
        raise HTTPException(
            status_code=400,
            detail="Reason is required for 'ignore' action"
        )
    
    if feedback.action == ActionType.EDIT and not feedback.corrected_content:
        raise HTTPException(
            status_code=400,
            detail="Corrected content is required for 'edit' action"
        )
    
    # Set reviewer info
    feedback.reviewer_id = current_user["id"]
    feedback.reviewer_role = current_user.get("role", "doctor")
    feedback.timestamp = datetime.now()
    
    # Get current model versions
    feedback.model_version = get_current_model_version()
    feedback.prompt_version = get_current_prompt_version()
    feedback.rule_version = get_current_rule_version()
    
    # Get original prediction and context
    original_prediction = get_original_prediction(feedback.case_id, feedback.issue_id)
    feedback.original_prediction = original_prediction
    
    visual_facts = get_visual_facts(feedback.case_id)
    feedback.visual_facts = visual_facts
    
    report_facts = get_report_facts(feedback.case_id)
    feedback.report_facts = report_facts
    
    report_text = get_report_text(feedback.case_id)
    feedback.report_text = report_text
    
    # Get metadata
    case_metadata = get_case_metadata(feedback.case_id)
    feedback.department = case_metadata["department"]
    feedback.exam_type = case_metadata["exam_type"]
    feedback.metadata = case_metadata.get("metadata", {})
    
    # Store in PostgreSQL
    feedback_id = save_feedback_to_database(feedback)
    
    # Queue for S3 sync (async, non-blocking)
    queue_feedback_for_s3_sync(feedback)
    
    # Return created feedback
    feedback.id = feedback_id
    return feedback
```

### Database Storage

```python
from sqlalchemy import Column, String, DateTime, JSON, Index
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class FeedbackRecordDB(Base):
    """Feedback record database model"""
    __tablename__ = "feedback_records"
    
    # Primary keys
    case_id = Column(String, primary_key=True)
    issue_id = Column(String, primary_key=True)
    
    # Action fields
    action = Column(String, nullable=False)  # "adopt" | "ignore" | "edit"
    reason = Column(String, nullable=True)
    corrected_content = Column(JSON, nullable=True)
    
    # Reviewer info
    reviewer_id = Column(String, nullable=False)
    reviewer_role = Column(String, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    
    # Model version tracking
    model_version = Column(String, nullable=False)
    prompt_version = Column(String, nullable=False)
    rule_version = Column(String, nullable=False)
    
    # Original prediction and feedback (stored as JSONB)
    original_prediction = Column(JSON, nullable=False)
    doctor_feedback = Column(JSON, nullable=False)
    
    # Context (for analysis)
    visual_facts = Column(JSON, nullable=False)
    report_facts = Column(JSON, nullable=False)
    report_text = Column(String, nullable=False)
    
    # Metadata
    department = Column(String, nullable=False)
    exam_type = Column(String, nullable=False)
    metadata = Column(JSON, nullable=True)
    
    # Indexes for efficient querying
    __table_args__ = (
        Index('idx_action_timestamp', 'action', 'timestamp'),
        Index('idx_model_version_timestamp', 'model_version', 'timestamp'),
        Index('idx_reviewer_timestamp', 'reviewer_id', 'timestamp'),
    )
```

### S3 Sync Process

```python
import boto3
from datetime import datetime, timedelta
from typing import List

s3_client = boto3.client('s3')
S3_BUCKET = "rad-linter-data"

def sync_feedback_to_s3(days_back: int = 1):
    """
    Sync feedback data from PostgreSQL to S3 (daily batch)
    
    Args:
        days_back: Number of days to sync (default: 1, sync yesterday's data)
    """
    # Query feedback from last N days
    cutoff_time = datetime.now() - timedelta(days=days_back)
    feedback_records = query_feedback_since(cutoff_time)
    
    # Classify by action type
    accept_cases = [f for f in feedback_records if f.action == ActionType.ADOPT]
    ignore_cases = [f for f in feedback_records if f.action == ActionType.IGNORE]
    review_cases = [f for f in feedback_records if f.action == ActionType.EDIT]
    
    # Get current model version for S3 path
    model_version = get_current_model_version()
    date_str = datetime.now().strftime("%Y%m%d")
    
    # Upload to S3
    upload_to_s3(
        bucket=S3_BUCKET,
        key=f"feedback/accept_cases/{model_version}/accept_cases_{date_str}.jsonl",
        data=accept_cases
    )
    
    upload_to_s3(
        bucket=S3_BUCKET,
        key=f"feedback/ignore_cases/{model_version}/ignore_cases_{date_str}.jsonl",
        data=ignore_cases
    )
    
    upload_to_s3(
        bucket=S3_BUCKET,
        key=f"feedback/review_cases/{model_version}/review_cases_{date_str}.jsonl",
        data=review_cases
    )
    
    # Generate metadata
    metadata = {
        "date": date_str,
        "model_version": model_version,
        "accept_count": len(accept_cases),
        "ignore_count": len(ignore_cases),
        "review_count": len(review_cases),
        "total_count": len(feedback_records)
    }
    
    upload_to_s3(
        bucket=S3_BUCKET,
        key=f"feedback/accept_cases/{model_version}/metadata_{date_str}.json",
        data=metadata
    )
```

---

## Data Flow to Post-Training

### Complete Data Flow

```
┌─────────────────────────────────────────────────────────┐
│ Phase 1: Feedback Collection (On-Prem)                 │
│                                                          │
│ Doctor Review UI                                         │
│   ↓                                                      │
│ Doctor Action (Adopt/Ignore/Edit)                       │
│   ↓                                                      │
│ FeedbackRecord Created                                   │
│   ↓                                                      │
│ Stored in PostgreSQL                                     │
│   ↓                                                      │
│ Daily S3 Sync                                           │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│ Phase 2: Data Classification (AWS)                    │
│                                                          │
│ S3: feedback/accept_cases/v1.0/                        │
│ S3: feedback/ignore_cases/v1.0/                        │
│ S3: feedback/review_cases/v1.0/                        │
│                                                          │
│ Classify by action type:                                │
│   • Adopt → Accept Cases                                │
│   • Ignore → Ignore Cases                               │
│   • Edit → Review Cases                                 │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│ Phase 3: Post-Training Preparation (AWS)              │
│                                                          │
│ Accept Cases → SFT Training Data                       │
│   • Format: (input, output) pairs                      │
│   • Purpose: Reinforce correct predictions              │
│                                                          │
│ Ignore Cases → Rule Optimization                        │
│   • Analyze false positive patterns                    │
│   • Update rules_v1.0.py → rules_v1.1.py               │
│                                                          │
│ Review Cases → DPO Training Data                       │
│   • Format: (input, preferred, rejected) pairs         │
│   • Purpose: Learn from expert corrections             │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│ Phase 4: Post-Training (AWS)                           │
│                                                          │
│ SFT Training (Accept Cases)                             │
│   • Base: lora_model_v1.0.pt                           │
│   • Output: lora_model_v1.1.pt                         │
│                                                          │
│ DPO Training (Review Cases)                             │
│   • Base: lora_model_v1.1.pt                           │
│   • Output: lora_model_v1.1.pt (updated)               │
│                                                          │
│ Rule Optimization (Ignore Cases)                        │
│   • Update: rules_v1.0.py → rules_v1.1.py               │
└─────────────────────────────────────────────────────────┘
```

### Data Transformation Examples

#### Adopt → SFT Training Data

```python
def transform_adopt_to_sft(accept_cases: List[FeedbackRecord]) -> List[dict]:
    """Transform accept cases to SFT training format"""
    sft_data = []
    
    for case in accept_cases:
        # Input: visual_facts + report_facts
        input_data = {
            "visual_facts": case.visual_facts,
            "report_facts": case.report_facts,
            "report_text": case.report_text
        }
        
        # Output: doctor-accepted prediction (original_prediction)
        output_data = case.original_prediction
        
        sft_data.append({
            "input": input_data,
            "output": output_data,
            "case_id": case.case_id,
            "issue_id": case.issue_id
        })
    
    return sft_data
```

#### Ignore → Rule Optimization Analysis

```python
def analyze_ignore_cases(ignore_cases: List[FeedbackRecord]) -> dict:
    """Analyze ignore cases to identify false positive patterns"""
    analysis = {
        "total_count": len(ignore_cases),
        "reason_distribution": {},
        "issue_type_distribution": {},
        "confidence_distribution": {},
        "patterns": []
    }
    
    # Group by reason
    for case in ignore_cases:
        reason = case.reason or "unknown"
        analysis["reason_distribution"][reason] = \
            analysis["reason_distribution"].get(reason, 0) + 1
        
        issue_type = case.original_prediction["issue_type"]
        analysis["issue_type_distribution"][issue_type] = \
            analysis["issue_type_distribution"].get(issue_type, 0) + 1
        
        confidence = case.original_prediction["confidence"]
        confidence_bucket = f"{int(confidence * 10) * 10}%"
        analysis["confidence_distribution"][confidence_bucket] = \
            analysis["confidence_distribution"].get(confidence_bucket, 0) + 1
    
    # Identify patterns
    if "False positive - visual fact confidence too low" in analysis["reason_distribution"]:
        analysis["patterns"].append({
            "pattern": "low_visual_confidence",
            "suggestion": "Increase visual fact confidence threshold",
            "affected_count": analysis["reason_distribution"]["False positive - visual fact confidence too low"]
        })
    
    return analysis
```

#### Edit → DPO Training Data

```python
def transform_edit_to_dpo(review_cases: List[FeedbackRecord]) -> List[dict]:
    """Transform review cases to DPO training format"""
    dpo_data = []
    
    for case in review_cases:
        # Input: visual_facts + report_facts
        input_data = {
            "visual_facts": case.visual_facts,
            "report_facts": case.report_facts,
            "report_text": case.report_text
        }
        
        # Preferred: expert-corrected output
        preferred = case.corrected_content or case.doctor_feedback.get("corrected_content")
        
        # Rejected: original model output
        rejected = case.original_prediction
        
        dpo_data.append({
            "input": input_data,
            "preferred": preferred,
            "rejected": rejected,
            "case_id": case.case_id,
            "issue_id": case.issue_id
        })
    
    return dpo_data
```

---

## Best Practices

### 1. **Action Selection Guidelines**

#### When to Use Adopt

✅ **Use Adopt when**:
- System prediction is completely correct
- No modifications needed
- High confidence in system prediction
- Quick review, no hesitation

❌ **Don't use Adopt when**:
- You have any doubts about the prediction
- Prediction needs any modification
- Confidence is low

#### When to Use Ignore

✅ **Use Ignore when**:
- System prediction is clearly wrong (false positive)
- No need to provide corrected content
- Can explain why it's wrong in reason field

❌ **Don't use Ignore when**:
- Prediction is partially correct (use Edit instead)
- You want to provide corrected content (use Edit instead)

#### When to Use Edit

✅ **Use Edit when**:
- System prediction is partially correct
- You want to provide expert correction
- Prediction needs refinement (severity, facts, explanation)

❌ **Don't use Edit when**:
- Prediction is completely wrong (use Ignore instead)
- Prediction is completely correct (use Adopt instead)

### 2. **Data Quality Requirements**

#### Minimum Feedback Thresholds

```yaml
Before Starting Post-Training:
  Accept Cases: 1000+ examples (for SFT)
  Ignore Cases: 500+ examples (for rule analysis)
  Review Cases: 500+ preference pairs (for DPO)
  
  Total: 2000+ feedback cases
  Collection Period: 4+ weeks
```

#### Data Quality Checks

```python
def validate_feedback_quality(feedback_records: List[FeedbackRecord]) -> dict:
    """Validate feedback data quality"""
    checks = {
        "minimum_samples": len(feedback_records) >= 2000,
        "action_distribution": {
            "adopt": len([f for f in feedback_records if f.action == ActionType.ADOPT]),
            "ignore": len([f for f in feedback_records if f.action == ActionType.IGNORE]),
            "edit": len([f for f in feedback_records if f.action == ActionType.EDIT])
        },
        "reason_completeness": all(
            f.reason for f in feedback_records 
            if f.action == ActionType.IGNORE
        ),
        "corrected_content_completeness": all(
            f.corrected_content for f in feedback_records 
            if f.action == ActionType.EDIT
        ),
        "diversity": check_diversity(feedback_records)  # All issue types, departments
    }
    
    return checks
```

### 3. **UI/UX Best Practices**

#### Make Actions Clear

- Use clear, descriptive button labels
- Provide tooltips explaining each action
- Show examples of when to use each action
- Display what happens to the feedback (training purpose)

#### Reduce Friction

- Support keyboard shortcuts
- Provide quick actions for common scenarios
- Auto-save draft feedback
- Show feedback history

#### Ensure Data Quality

- Require reason for Ignore (validation)
- Require corrected content for Edit (validation)
- Provide reason templates/dropdown for Ignore
- Show confirmation before submission

### 4. **Monitoring & Analytics**

#### Key Metrics to Track

```yaml
Feedback Collection Metrics:
  - Daily feedback count (by action type)
  - Action distribution (Adopt/Ignore/Edit ratio)
  - Average time to review
  - Feedback quality score
  - Reviewer engagement rate
  
Feedback Quality Metrics:
  - Reason completeness (for Ignore)
  - Corrected content completeness (for Edit)
  - Feedback diversity (issue types, departments)
  - Reviewer consistency
```

#### Dashboard Example

```python
def generate_feedback_dashboard() -> dict:
    """Generate feedback collection dashboard"""
    dashboard = {
        "summary": {
            "total_feedback": get_total_feedback_count(),
            "adopt_count": get_action_count(ActionType.ADOPT),
            "ignore_count": get_action_count(ActionType.IGNORE),
            "edit_count": get_action_count(ActionType.EDIT),
            "collection_period_days": get_collection_period_days()
        },
        "trends": {
            "daily_feedback": get_daily_feedback_trend(),
            "action_distribution": get_action_distribution(),
            "issue_type_distribution": get_issue_type_distribution()
        },
        "quality": {
            "reason_completeness": get_reason_completeness(),
            "corrected_content_completeness": get_corrected_content_completeness(),
            "diversity_score": get_diversity_score()
        },
        "readiness": {
            "ready_for_post_training": check_post_training_readiness(),
            "missing_data": get_missing_data_breakdown()
        }
    }
    
    return dashboard
```

---

## Summary

### Key Takeaways

1. ✅ **Three Action Types**: Adopt (接受), Ignore (忽略), Edit (编辑)
2. ✅ **Action → Post-Training Mapping**:
   - Adopt → SFT (Supervised Fine-Tuning)
   - Ignore → Rule Optimization
   - Edit → DPO (Direct Preference Optimization)
3. ✅ **Data Structure**: Complete FeedbackRecord schema with validation
4. ✅ **UI Design**: Clear action buttons, contextual help, validation
5. ✅ **Data Flow**: PostgreSQL (On-Prem) → S3 (AWS) → Post-Training
6. ✅ **Quality Requirements**: 2000+ feedback cases, 4+ weeks collection

### Quick Reference

| Action | 中文 | Required Fields | Post-Training Use |
|--------|------|----------------|------------------|
| **Adopt** | 接受 | None | SFT Training |
| **Ignore** | 忽略 | Reason (required) | Rule Optimization |
| **Edit** | 编辑 | Corrected Content (required) | DPO Training |

### Related Documents

- **A05**: Post-Training Mechanism (完整Post-Training流程)
- **A06**: Fine-Tuning vs Post-Training (Fine-Tuning与Post-Training的区别)

---

**Remember**: 医生在审查UI中的操作是Post-Training反馈数据的核心来源。设计良好的UI和清晰的操作流程对于收集高质量的反馈数据至关重要。
