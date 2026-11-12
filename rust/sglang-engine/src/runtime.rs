//! Rust scheduler orchestration over admission and model execution.

use crate::admission::{AdmissionError, AdmissionLimits, AdmissionQueue, AdmissionRequest};
use crate::model::{ExecutionResult, ModelContractError, ModelExecutor, validate_batch};
use crate::scheduler::WaitingOrder;
use crate::{EngineConfig, RuntimeError};

/// Runtime errors surfaced by Rust scheduler orchestration.
#[derive(Debug, thiserror::Error)]
pub enum SchedulerError {
    #[error(transparent)]
    Admission(#[from] AdmissionError),
    #[error(transparent)]
    Runtime(#[from] RuntimeError),
    #[error(transparent)]
    ModelContract(#[from] ModelContractError),
}

/// Scheduler that owns admission state and delegates model batches to an executor.
pub struct SchedulerRuntime<E> {
    queue: AdmissionQueue,
    executor: E,
}

impl<E: ModelExecutor> SchedulerRuntime<E> {
    /// Load the model executor and create a scheduler runtime.
    pub fn start(
        config: &EngineConfig,
        limits: AdmissionLimits,
        mut executor: E,
    ) -> Result<Self, SchedulerError> {
        config
            .validate()
            .map_err(|error| RuntimeError::new("scheduler startup", error.to_string()))?;
        executor.load(config)?;
        Ok(Self {
            queue: AdmissionQueue::new(limits),
            executor,
        })
    }

    /// Enqueue one request for model execution.
    pub fn submit(&mut self, request: AdmissionRequest) -> Result<(), SchedulerError> {
        self.queue.enqueue(request)?;
        Ok(())
    }

    /// Admit and execute one batch; completed requests release their slots.
    pub fn execute_round(
        &mut self,
        order: WaitingOrder,
        prefill_budget: u64,
    ) -> Result<Vec<ExecutionResult>, SchedulerError> {
        let admitted = self.queue.admit(order, prefill_budget)?;
        if admitted.is_empty() {
            return Ok(Vec::new());
        }
        let tasks: Vec<_> = admitted
            .iter()
            .map(|request| request.task.clone())
            .collect();
        let results = match self.executor.execute_batch(&tasks) {
            Ok(results) => results,
            Err(error) => {
                self.release_failed(&admitted);
                return Err(error.into());
            }
        };
        if let Err(error) = validate_batch(&tasks, &results) {
            self.release_failed(&admitted);
            return Err(error.into());
        }
        for result in &results {
            self.queue.complete(&result.request_id)?;
        }
        Ok(results)
    }

    fn release_failed(&mut self, admitted: &[AdmissionRequest]) {
        for request in admitted {
            let _ = self.queue.complete(&request.waiting.id);
        }
    }

    /// Cancel a queued or active request.
    pub fn cancel(
        &mut self,
        request_id: &str,
    ) -> Result<crate::admission::CancelledRequest, SchedulerError> {
        Ok(self.queue.cancel(request_id)?)
    }

    /// Current number of queued requests.
    pub fn pending_len(&self) -> usize {
        self.queue.pending_len()
    }

    /// Current number of active requests.
    pub fn running_len(&self) -> usize {
        self.queue.running_len()
    }

    /// Stop model execution and release its owned resources.
    pub fn shutdown(&mut self) -> Result<(), SchedulerError> {
        self.executor.shutdown()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::admission::CancelledRequest;
    use crate::model::{ExecutionTask, SamplingConfig};
    use crate::scheduler::WaitingRequest;

    #[derive(Default)]
    struct FakeState {
        loaded: bool,
        shutdown: bool,
        executed_batches: usize,
        fail_execution: bool,
    }

    struct FakeExecutor(Arc<Mutex<FakeState>>);

    impl ModelExecutor for FakeExecutor {
        fn load(&mut self, _: &EngineConfig) -> Result<(), RuntimeError> {
            self.0.lock().unwrap().loaded = true;
            Ok(())
        }

        fn execute_batch(
            &mut self,
            tasks: &[ExecutionTask],
        ) -> Result<Vec<ExecutionResult>, RuntimeError> {
            let mut state = self.0.lock().unwrap();
            state.executed_batches += 1;
            if state.fail_execution {
                return Err(RuntimeError::new("execute", "injected error"));
            }
            Ok(tasks
                .iter()
                .map(|task| ExecutionResult {
                    request_id: task.request_id.clone(),
                    output_token_ids: vec![9],
                    output_logprobs: None,
                    finished: true,
                })
                .collect())
        }

        fn shutdown(&mut self) -> Result<(), RuntimeError> {
            self.0.lock().unwrap().shutdown = true;
            Ok(())
        }
    }

    fn request(id: &str, arrival: u64) -> AdmissionRequest {
        AdmissionRequest {
            waiting: WaitingRequest {
                id: id.into(),
                priority: None,
                arrival_sequence: arrival,
                wait_started_micros: arrival,
                uncached_input_tokens: 1,
                matched_prefix_tokens: 0,
                output_tokens: 0,
                max_new_tokens: 2,
                arrival_processed_tokens: 0,
                temporarily_deprioritized: false,
                routing_key: None,
                prefix_node: 0,
            },
            task: ExecutionTask {
                request_id: id.into(),
                prompt_token_ids: vec![1, 2],
                sampling: SamplingConfig {
                    max_new_tokens: 2,
                    temperature: 0.0,
                    top_p: 1.0,
                    top_k: 0,
                },
            },
            prefill_tokens: 1,
        }
    }

    fn config() -> EngineConfig {
        EngineConfig {
            model: "test-model".into(),
            tokenizer: None,
            cache_dir: None,
        }
    }

    #[test]
    fn scheduler_loads_batches_completes_and_shuts_down_executor() {
        let shared = Arc::new(Mutex::new(FakeState::default()));
        let mut scheduler = SchedulerRuntime::start(
            &config(),
            AdmissionLimits {
                max_pending: 4,
                max_running: 2,
            },
            FakeExecutor(shared.clone()),
        )
        .unwrap();
        assert!(shared.lock().unwrap().loaded);
        scheduler.submit(request("first", 1)).unwrap();
        scheduler.submit(request("second", 2)).unwrap();
        let results = scheduler.execute_round(WaitingOrder::Fcfs, 2).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(scheduler.pending_len(), 0);
        assert_eq!(scheduler.running_len(), 0);
        scheduler.shutdown().unwrap();
        assert!(shared.lock().unwrap().shutdown);
    }

    #[test]
    fn scheduler_releases_slots_after_backend_failure() {
        let shared = Arc::new(Mutex::new(FakeState {
            fail_execution: true,
            ..FakeState::default()
        }));
        let mut scheduler = SchedulerRuntime::start(
            &config(),
            AdmissionLimits {
                max_pending: 2,
                max_running: 1,
            },
            FakeExecutor(shared),
        )
        .unwrap();
        scheduler.submit(request("a", 1)).unwrap();
        assert!(matches!(
            scheduler.execute_round(WaitingOrder::Fcfs, 1),
            Err(SchedulerError::Runtime(_))
        ));
        assert_eq!(scheduler.running_len(), 0);
    }

    #[test]
    fn scheduler_cancellation_updates_queue_state() {
        let shared = Arc::new(Mutex::new(FakeState::default()));
        let mut scheduler = SchedulerRuntime::start(
            &config(),
            AdmissionLimits {
                max_pending: 2,
                max_running: 1,
            },
            FakeExecutor(shared),
        )
        .unwrap();
        scheduler.submit(request("a", 1)).unwrap();
        assert_eq!(scheduler.cancel("a").unwrap(), CancelledRequest::Pending);
        assert_eq!(scheduler.pending_len(), 0);
    }
}
