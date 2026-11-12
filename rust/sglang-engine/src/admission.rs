//! Bounded request admission state for the Rust scheduler.
//!
//! This owns queue/running transitions and prefill-token budget accounting.
//! KV allocation, prefix-cache matching, decode batching, and preemption are
//! separate scheduler components and are not modeled by this first port.

use std::collections::{BTreeMap, HashSet};

use crate::model::ExecutionTask;
use crate::scheduler::{ScheduleOrderError, WaitingOrder, WaitingRequest, order_waiting};

/// Request accepted into the scheduler's waiting queue.
#[derive(Clone, Debug, PartialEq)]
pub struct AdmissionRequest {
    /// Ordering metadata and stable request ID.
    pub waiting: WaitingRequest,
    /// Tokenized model work carried with the scheduler request.
    pub task: ExecutionTask,
    /// Estimated uncached prefill tokens charged to the next admission batch.
    pub prefill_tokens: u64,
}

/// Result of canceling a request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CancelledRequest {
    /// Request was removed before execution.
    Pending,
    /// Request was marked canceled while active; execution owner must observe it.
    Running,
}

/// Limits for bounded pending and running request state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AdmissionLimits {
    /// Maximum requests held in the pending queue.
    pub max_pending: usize,
    /// Maximum requests admitted but not completed.
    pub max_running: usize,
}

/// Scheduler admission state.
pub struct AdmissionQueue {
    limits: AdmissionLimits,
    pending: Vec<AdmissionRequest>,
    running: BTreeMap<String, AdmissionRequest>,
    cancelled_running: HashSet<String>,
}

impl AdmissionQueue {
    /// Create an empty admission queue.
    pub fn new(limits: AdmissionLimits) -> Self {
        Self {
            limits,
            pending: Vec::new(),
            running: BTreeMap::new(),
            cancelled_running: HashSet::new(),
        }
    }

    /// Add a request to the bounded pending queue.
    pub fn enqueue(&mut self, request: AdmissionRequest) -> Result<(), AdmissionError> {
        let id = &request.waiting.id;
        if request.task.request_id != *id {
            return Err(AdmissionError::IdentityMismatch(id.clone()));
        }
        if self.running.contains_key(id) || self.pending.iter().any(|r| r.waiting.id == *id) {
            return Err(AdmissionError::DuplicateId(id.clone()));
        }
        if self.pending.len() >= self.limits.max_pending {
            return Err(AdmissionError::QueueFull);
        }
        self.pending.push(request);
        Ok(())
    }

    /// Admit requests in policy order while respecting running and token limits.
    ///
    /// Requests that do not fit remain pending; later smaller requests may fit
    /// in the same batch. `prefill_budget` is a per-admission-round limit.
    pub fn admit(
        &mut self,
        order: WaitingOrder,
        prefill_budget: u64,
    ) -> Result<Vec<AdmissionRequest>, AdmissionError> {
        if self.running.len() >= self.limits.max_running {
            return Ok(Vec::new());
        }
        let mut metadata: Vec<_> = self.pending.iter().map(|r| r.waiting.clone()).collect();
        order_waiting(&mut metadata, order).map_err(AdmissionError::Order)?;

        // Policies return request identities/order. IDs are unique by the
        // enqueue invariant, so reconstructing the ordered payloads is safe.
        let mut by_id: BTreeMap<_, _> = self
            .pending
            .drain(..)
            .map(|request| (request.waiting.id.clone(), request))
            .collect();
        let mut selected = Vec::new();
        let mut retained = Vec::new();
        let mut spent = 0_u64;
        for item in metadata {
            let request = by_id
                .remove(&item.id)
                .expect("metadata was derived from queued requests");
            let has_slot = self.running.len() + selected.len() < self.limits.max_running;
            let fits = request.prefill_tokens <= prefill_budget.saturating_sub(spent);
            if has_slot && fits {
                spent += request.prefill_tokens;
                selected.push(request);
            } else {
                retained.push(request);
            }
        }
        for request in &selected {
            self.running
                .insert(request.waiting.id.clone(), request.clone());
        }
        self.pending = retained;
        Ok(selected)
    }

    /// Mark a running request complete and release its concurrency slot.
    pub fn complete(&mut self, id: &str) -> Result<AdmissionRequest, AdmissionError> {
        self.cancelled_running.remove(id);
        self.running
            .remove(id)
            .ok_or_else(|| AdmissionError::UnknownRunningId(id.into()))
    }

    /// Cancel a pending request or signal cancellation for a running request.
    pub fn cancel(&mut self, id: &str) -> Result<CancelledRequest, AdmissionError> {
        if let Some(index) = self.pending.iter().position(|r| r.waiting.id == id) {
            self.pending.remove(index);
            return Ok(CancelledRequest::Pending);
        }
        if self.running.contains_key(id) {
            self.cancelled_running.insert(id.into());
            return Ok(CancelledRequest::Running);
        }
        Err(AdmissionError::UnknownId(id.into()))
    }

    /// Whether a running request has a cancellation signal pending.
    pub fn is_cancelled(&self, id: &str) -> bool {
        self.cancelled_running.contains(id)
    }

    /// Number of queued requests.
    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }

    /// Number of admitted requests not yet completed.
    pub fn running_len(&self) -> usize {
        self.running.len()
    }
}

/// Admission state or policy failure.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum AdmissionError {
    #[error("pending request queue is full")]
    QueueFull,
    #[error("request id already exists: {0}")]
    DuplicateId(String),
    #[error("scheduler request and model task IDs differ for request {0}")]
    IdentityMismatch(String),
    #[error("unknown request id: {0}")]
    UnknownId(String),
    #[error("request is not running: {0}")]
    UnknownRunningId(String),
    #[error("waiting policy could not order requests: {0:?}")]
    Order(ScheduleOrderError),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(id: &str, cost: u64, arrival: u64) -> AdmissionRequest {
        AdmissionRequest {
            waiting: WaitingRequest {
                id: id.into(),
                priority: None,
                arrival_sequence: arrival,
                wait_started_micros: arrival,
                uncached_input_tokens: cost,
                matched_prefix_tokens: 0,
                output_tokens: 0,
                max_new_tokens: 8,
                arrival_processed_tokens: 0,
                temporarily_deprioritized: false,
                routing_key: None,
                prefix_node: 0,
            },
            task: ExecutionTask {
                request_id: id.into(),
                prompt_token_ids: vec![1, 2],
                sampling: crate::model::SamplingConfig {
                    max_new_tokens: 8,
                    temperature: 0.0,
                    top_p: 1.0,
                    top_k: 0,
                },
            },
            prefill_tokens: cost,
        }
    }

    #[test]
    fn admission_obeys_policy_budget_and_running_limit() {
        let mut queue = AdmissionQueue::new(AdmissionLimits {
            max_pending: 4,
            max_running: 2,
        });
        queue.enqueue(request("large", 8, 1)).unwrap();
        queue.enqueue(request("small-a", 3, 2)).unwrap();
        queue.enqueue(request("small-b", 2, 3)).unwrap();

        let admitted = queue.admit(WaitingOrder::Fcfs, 5).unwrap();
        assert_eq!(
            admitted
                .iter()
                .map(|r| r.waiting.id.as_str())
                .collect::<Vec<_>>(),
            ["small-a", "small-b"]
        );
        assert_eq!(queue.pending_len(), 1);
        assert_eq!(queue.running_len(), 2);
        assert!(queue.admit(WaitingOrder::Fcfs, 99).unwrap().is_empty());

        queue.complete("small-a").unwrap();
        let next = queue.admit(WaitingOrder::Fcfs, 8).unwrap();
        assert_eq!(next[0].waiting.id, "large");
    }

    #[test]
    fn enqueue_rejects_duplicate_and_over_capacity() {
        let mut queue = AdmissionQueue::new(AdmissionLimits {
            max_pending: 1,
            max_running: 1,
        });
        queue.enqueue(request("a", 1, 1)).unwrap();
        assert_eq!(
            queue.enqueue(request("a", 1, 2)),
            Err(AdmissionError::DuplicateId("a".into()))
        );
        assert_eq!(
            queue.enqueue(request("b", 1, 2)),
            Err(AdmissionError::QueueFull)
        );
    }

    #[test]
    fn enqueue_rejects_task_id_mismatch() {
        let mut queue = AdmissionQueue::new(AdmissionLimits {
            max_pending: 1,
            max_running: 1,
        });
        let mut mismatched = request("a", 1, 1);
        mismatched.task.request_id = "b".into();
        assert_eq!(
            queue.enqueue(mismatched),
            Err(AdmissionError::IdentityMismatch("a".into()))
        );
        assert_eq!(queue.pending_len(), 0);
    }

    #[test]
    fn cancellation_removes_pending_and_marks_running_until_completion() {
        let mut queue = AdmissionQueue::new(AdmissionLimits {
            max_pending: 2,
            max_running: 1,
        });
        queue.enqueue(request("pending", 1, 1)).unwrap();
        assert_eq!(queue.cancel("pending"), Ok(CancelledRequest::Pending));
        queue.enqueue(request("active", 1, 2)).unwrap();
        queue.admit(WaitingOrder::Fcfs, 1).unwrap();
        assert_eq!(queue.cancel("active"), Ok(CancelledRequest::Running));
        assert!(queue.is_cancelled("active"));
        queue.complete("active").unwrap();
        assert!(!queue.is_cancelled("active"));
        assert_eq!(queue.running_len(), 0);
    }

    #[test]
    fn failed_ordering_preserves_pending_queue() {
        let mut queue = AdmissionQueue::new(AdmissionLimits {
            max_pending: 2,
            max_running: 1,
        });
        let mut a = request("a", 1, 1);
        a.waiting.priority = Some(1);
        queue.enqueue(a).unwrap();
        queue.enqueue(request("b", 1, 2)).unwrap();
        assert!(matches!(
            queue.admit(
                WaitingOrder::PriorityFcfs {
                    lower_value_first: true
                },
                2
            ),
            Err(AdmissionError::Order(_))
        ));
        assert_eq!(queue.pending_len(), 2);
        assert_eq!(queue.running_len(), 0);
    }
}
