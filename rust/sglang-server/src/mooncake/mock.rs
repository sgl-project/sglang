use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use crate::mooncake::{
    BatchId, EngineError, EngineFactory, EngineOperation, NativeOperation, OperationProgress,
    OperationState, PeerDescriptor, PeerId, RegionDescriptor, RegionId, TransferEngine,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MockFailurePoint {
    Create,
    LocalPeerDescriptor,
    RegisterRegion,
    UnregisterRegion,
    OpenPeer,
    ClosePeer,
    AllocateBatch,
    SubmitBatch,
    Poll,
    FreeBatch,
    Shutdown,
}

impl MockFailurePoint {
    fn operation(self) -> NativeOperation {
        match self {
            Self::Create => NativeOperation::CreateEngine,
            Self::LocalPeerDescriptor => NativeOperation::GetLocalEndpoint,
            Self::RegisterRegion => NativeOperation::RegisterRegion,
            Self::UnregisterRegion => NativeOperation::UnregisterRegion,
            Self::OpenPeer => NativeOperation::OpenPeer,
            Self::ClosePeer => NativeOperation::ClosePeer,
            Self::AllocateBatch => NativeOperation::AllocateBatch,
            Self::SubmitBatch => NativeOperation::SubmitBatch,
            Self::Poll => NativeOperation::Poll,
            Self::FreeBatch => NativeOperation::FreeBatch,
            Self::Shutdown => NativeOperation::DestroyEngine,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MockEvent {
    Failure {
        point: MockFailurePoint,
        raw_code: i32,
    },
    Create,
    LocalPeerDescriptor,
    RegisterRegion {
        id: RegionId,
    },
    UnregisterRegion {
        id: RegionId,
    },
    OpenPeer {
        id: PeerId,
    },
    ClosePeer {
        id: PeerId,
    },
    AllocateBatch {
        id: BatchId,
        operation_count: usize,
    },
    SubmitBatch {
        id: BatchId,
    },
    Poll {
        id: BatchId,
        operation_index: usize,
        state: OperationState,
    },
    FreeBatch {
        id: BatchId,
    },
    CancelBatch {
        id: BatchId,
    },
    Shutdown,
}

#[derive(Debug, Clone)]
pub struct MockPlan {
    status_script: Vec<Vec<OperationState>>,
    failures: HashMap<MockFailurePoint, VecDeque<i32>>,
    delays: HashMap<MockFailurePoint, Duration>,
}

impl Default for MockPlan {
    fn default() -> Self {
        Self::with_status_script(vec![vec![OperationState::Completed]])
    }
}

impl MockPlan {
    pub fn with_status_script(status_script: Vec<Vec<OperationState>>) -> Self {
        Self {
            status_script: if status_script.is_empty() {
                vec![vec![OperationState::Completed]]
            } else {
                status_script
            },
            failures: HashMap::new(),
            delays: HashMap::new(),
        }
    }

    pub fn fail_once(mut self, point: MockFailurePoint, raw_code: i32) -> Self {
        self.failures.entry(point).or_default().push_back(raw_code);
        self
    }

    pub fn delay(mut self, point: MockFailurePoint, duration: Duration) -> Self {
        self.delays.insert(point, duration);
        self
    }
}

#[derive(Clone)]
pub struct MockEngineFactory {
    plan: MockPlan,
    events: Arc<Mutex<Vec<MockEvent>>>,
}

impl MockEngineFactory {
    pub fn new(plan: MockPlan) -> Self {
        Self {
            plan,
            events: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn events(&self) -> Arc<Mutex<Vec<MockEvent>>> {
        Arc::clone(&self.events)
    }
}

impl EngineFactory for MockEngineFactory {
    fn create(&self) -> Result<Box<dyn TransferEngine>, EngineError> {
        let mut failures = self.plan.failures.clone();
        if let Some(raw_code) = take_failure(&mut failures, MockFailurePoint::Create) {
            self.events
                .lock()
                .map_err(|_| EngineError::LockPoisoned)?
                .push(MockEvent::Failure {
                    point: MockFailurePoint::Create,
                    raw_code,
                });
            return Err(EngineError::native(
                MockFailurePoint::Create.operation(),
                raw_code,
            ));
        }
        if let Some(duration) = self.plan.delays.get(&MockFailurePoint::Create) {
            thread::sleep(*duration);
        }
        self.events
            .lock()
            .map_err(|_| EngineError::LockPoisoned)?
            .push(MockEvent::Create);
        Ok(Box::new(MockEngine {
            status_script: self.plan.status_script.clone(),
            failures,
            delays: self.plan.delays.clone(),
            events: Arc::clone(&self.events),
            batches: HashMap::new(),
            shutdown: false,
        }))
    }
}

struct MockBatch {
    operation_lengths: Vec<u64>,
    poll_cursors: Vec<usize>,
    latest: Vec<OperationState>,
    submitted: bool,
}

struct MockEngine {
    status_script: Vec<Vec<OperationState>>,
    failures: HashMap<MockFailurePoint, VecDeque<i32>>,
    delays: HashMap<MockFailurePoint, Duration>,
    events: Arc<Mutex<Vec<MockEvent>>>,
    batches: HashMap<BatchId, MockBatch>,
    shutdown: bool,
}

impl MockEngine {
    fn fail(&mut self, point: MockFailurePoint) -> Result<(), EngineError> {
        if let Some(raw_code) = take_failure(&mut self.failures, point) {
            self.event(MockEvent::Failure { point, raw_code })?;
            return Err(EngineError::native(point.operation(), raw_code));
        }
        Ok(())
    }

    fn delay(&self, point: MockFailurePoint) {
        if let Some(duration) = self.delays.get(&point) {
            thread::sleep(*duration);
        }
    }

    fn event(&self, event: MockEvent) -> Result<(), EngineError> {
        self.events
            .lock()
            .map_err(|_| EngineError::LockPoisoned)?
            .push(event);
        Ok(())
    }
}

impl TransferEngine for MockEngine {
    fn local_peer_descriptor(&mut self) -> Result<PeerDescriptor, EngineError> {
        self.fail(MockFailurePoint::LocalPeerDescriptor)?;
        self.event(MockEvent::LocalPeerDescriptor)?;
        self.delay(MockFailurePoint::LocalPeerDescriptor);
        PeerDescriptor::new("127.0.0.1:19000")
    }

    fn register_region(
        &mut self,
        id: RegionId,
        _descriptor: &RegionDescriptor,
    ) -> Result<(), EngineError> {
        self.fail(MockFailurePoint::RegisterRegion)?;
        self.event(MockEvent::RegisterRegion { id })?;
        self.delay(MockFailurePoint::RegisterRegion);
        Ok(())
    }

    fn unregister_region(&mut self, id: RegionId) -> Result<(), EngineError> {
        self.fail(MockFailurePoint::UnregisterRegion)?;
        self.event(MockEvent::UnregisterRegion { id })?;
        self.delay(MockFailurePoint::UnregisterRegion);
        Ok(())
    }

    fn open_peer(&mut self, id: PeerId, _descriptor: &PeerDescriptor) -> Result<(), EngineError> {
        self.fail(MockFailurePoint::OpenPeer)?;
        self.event(MockEvent::OpenPeer { id })?;
        self.delay(MockFailurePoint::OpenPeer);
        Ok(())
    }

    fn close_peer(&mut self, id: PeerId) -> Result<(), EngineError> {
        self.fail(MockFailurePoint::ClosePeer)?;
        self.event(MockEvent::ClosePeer { id })?;
        self.delay(MockFailurePoint::ClosePeer);
        Ok(())
    }

    fn allocate_batch(&mut self, id: BatchId, operation_count: usize) -> Result<(), EngineError> {
        self.fail(MockFailurePoint::AllocateBatch)?;
        self.event(MockEvent::AllocateBatch {
            id,
            operation_count,
        })?;
        self.batches.insert(
            id,
            MockBatch {
                operation_lengths: vec![0; operation_count],
                poll_cursors: vec![0; operation_count],
                latest: vec![OperationState::Waiting; operation_count],
                submitted: false,
            },
        );
        self.delay(MockFailurePoint::AllocateBatch);
        Ok(())
    }

    fn submit_batch(
        &mut self,
        id: BatchId,
        operations: &[EngineOperation],
    ) -> Result<(), EngineError> {
        self.fail(MockFailurePoint::SubmitBatch)?;
        let batch = self
            .batches
            .get_mut(&id)
            .ok_or(EngineError::ResourceClosed {
                kind: "batch",
                id: id.get(),
            })?;
        batch.operation_lengths = operations.iter().map(EngineOperation::length).collect();
        batch.submitted = true;
        self.event(MockEvent::SubmitBatch { id })?;
        self.delay(MockFailurePoint::SubmitBatch);
        Ok(())
    }

    fn poll(
        &mut self,
        id: BatchId,
        operation_index: usize,
    ) -> Result<OperationProgress, EngineError> {
        self.fail(MockFailurePoint::Poll)?;
        let batch = self
            .batches
            .get_mut(&id)
            .ok_or(EngineError::ResourceClosed {
                kind: "batch",
                id: id.get(),
            })?;
        let cursor =
            batch
                .poll_cursors
                .get_mut(operation_index)
                .ok_or(EngineError::InvalidDescriptor {
                    field: "operation_index",
                    detail: operation_index.to_string(),
                })?;
        let script_index = (*cursor).min(self.status_script.len() - 1);
        let states = &self.status_script[script_index];
        let state = states
            .get(operation_index)
            .copied()
            .or_else(|| states.first().copied())
            .unwrap_or(OperationState::Unknown(i32::MIN));
        *cursor = cursor.saturating_add(1);
        batch.latest[operation_index] = state;
        let transferred_bytes = if state == OperationState::Completed {
            batch.operation_lengths[operation_index]
        } else {
            0
        };
        self.event(MockEvent::Poll {
            id,
            operation_index,
            state,
        })?;
        self.delay(MockFailurePoint::Poll);
        Ok(OperationProgress {
            state,
            transferred_bytes,
        })
    }

    fn free_batch(&mut self, id: BatchId) -> Result<(), EngineError> {
        self.fail(MockFailurePoint::FreeBatch)?;
        let batch = self.batches.get(&id).ok_or(EngineError::ResourceClosed {
            kind: "batch",
            id: id.get(),
        })?;
        if batch.submitted && !batch.latest.iter().all(|state| state.is_terminal()) {
            return Err(EngineError::BatchNotTerminal { id: id.get() });
        }
        self.event(MockEvent::FreeBatch { id })?;
        self.batches.remove(&id);
        self.delay(MockFailurePoint::FreeBatch);
        Ok(())
    }

    fn shutdown(&mut self) -> Result<(), EngineError> {
        if self.shutdown {
            return Ok(());
        }
        self.fail(MockFailurePoint::Shutdown)?;
        self.shutdown = true;
        self.event(MockEvent::Shutdown)?;
        self.delay(MockFailurePoint::Shutdown);
        Ok(())
    }
}

fn take_failure(
    failures: &mut HashMap<MockFailurePoint, VecDeque<i32>>,
    point: MockFailurePoint,
) -> Option<i32> {
    failures.get_mut(&point).and_then(VecDeque::pop_front)
}
