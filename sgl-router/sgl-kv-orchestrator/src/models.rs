//! Immutable logical identities and version fences, independent of operations.
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Compatibility {
    pub namespace: String,
    pub cache_salt: Option<String>,
    pub layout_id: String,
    pub page_size: u64,
    pub components: Vec<String>,
    pub hash_scheme: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Block {
    #[serde(with = "decimal_i64")]
    pub block_hash: i64,
    pub component: String,
    pub start_token: u64,
    pub end_token: u64,
    #[serde(default)]
    pub dependency_start: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrefixTarget {
    pub kind: String,
    pub compatibility: Compatibility,
    pub start_token: u64,
    pub end_token: u64,
    pub token_ids: Option<Vec<u32>>,
    #[serde(default)]
    pub blocks: Vec<Block>,
}

/// Identity survives worker changes; a worker epoch is deliberately absent.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct SessionIdentity {
    pub caller_id: String,
    pub session_id: String,
    pub incarnation: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BranchSnapshot {
    pub session: SessionIdentity,
    pub branch_id: String,
    pub revision: u64,
    pub prefix: PrefixTarget,
    pub parent_frontier: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClaimStatus {
    Active,
    Released,
    Expired,
    Invalidated,
    Unknown,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskIdentity {
    pub session_id: String,
    pub incarnation: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClaimState {
    pub caller_id: String,
    pub task_incarnation: String,
    pub claim_id: String,
    pub revision: u64,
    pub state: ClaimStatus,
    pub owner_epoch: String,
    #[serde(default)]
    pub tombstone: bool,
    #[serde(default = "result_version")]
    pub protection_version: String,
    pub effective_protection: Option<String>,
    pub remaining_ttl: Option<f64>,
    pub priority: Option<i32>,
    pub owner_task: Option<TaskIdentity>,
    #[serde(default)]
    pub actual_scope: Vec<Value>,
    pub location: Option<crate::protocol::Location>,
}

/// Carries observations, never creates claims or confirms queued placement.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicResult {
    #[serde(default = "result_version")]
    pub result_version: String,
    #[serde(default)]
    pub requested_scope: Vec<Value>,
    #[serde(default)]
    pub accepted_scope: Vec<Value>,
    #[serde(default)]
    pub ready_scope: Vec<Value>,
    pub ready_location: Option<crate::protocol::Location>,
    #[serde(default)]
    pub preserved_scope: Vec<Value>,
    #[serde(default)]
    pub reclaimed_scope: Vec<Value>,
    #[serde(default)]
    pub missing_scope: Vec<Value>,
    #[serde(default)]
    pub blocked_scope: Vec<Value>,
    pub owner_epoch: String,
    pub state_version: u64,
    pub outcome_sequence: u64,
    pub cache_generation: Option<u64>,
    pub source_observed_at: Option<f64>,
    pub claim: Option<ClaimState>,
}
fn result_version() -> String {
    "1.0".into()
}

/// Pending placement has its own type; submission never creates a replica.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PlacementExpectation {
    pub operation_id: String,
    pub worker_id: String,
    pub prefix: PrefixTarget,
    pub expires_at: f64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObservationState {
    Confirmed,
    Missing,
    Unknown,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReplicaObservation {
    pub prefix: PrefixTarget,
    pub owner_id: String,
    pub owner_epoch: String,
    pub tier: String,
    pub state: ObservationState,
    pub state_version: u64,
    pub observed_at: f64,
    pub provenance: String,
}

/// P5/P6 configuration names only. No unmeasured defaults or runtime allocation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MetadataLimits {
    pub max_sessions: u64,
    pub max_branches: u64,
    pub max_token_material_bytes: u64,
    pub max_history_revisions: u64,
    pub max_pending_intents: u64,
    pub max_operations: u64,
    pub max_claims: u64,
    pub max_event_buffer: u64,
    pub max_tombstones: u64,
}

mod decimal_i64 {
    use serde::{Deserialize, Deserializer, Serializer};
    pub fn serialize<S: Serializer>(value: &i64, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&value.to_string())
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<i64, D::Error> {
        let text = String::deserialize(deserializer)?;
        let value = text.parse::<i64>().map_err(serde::de::Error::custom)?;
        if value.to_string() != text {
            return Err(serde::de::Error::custom("expected canonical decimal i64"));
        }
        Ok(value)
    }
}
