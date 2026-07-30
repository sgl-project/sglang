use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use flume::Receiver;

use crate::mooncake::owner::{Command, ShutdownOutcome};
use crate::mooncake::{
    BatchId, BatchSnapshot, EngineError, EngineOperation, OperationProgress, PeerDescriptor,
    PeerId, RegionDescriptor, RegionId, TransferEngine, TransferOperation,
};

const MAX_INFLIGHT_BATCHES: usize = 4;

fn normalize_exact_progress(expected: u64, mut progress: OperationProgress) -> OperationProgress {
    if progress.state == crate::mooncake::OperationState::Completed
        && progress.transferred_bytes != expected
    {
        progress.state = crate::mooncake::OperationState::Pending;
    }
    progress
}

struct RegionState {
    descriptor: RegionDescriptor,
    release_requested: bool,
    release_attempted: bool,
}

struct PeerState {
    release_requested: bool,
    release_attempted: bool,
}

struct BatchState {
    operations: Vec<EngineOperation>,
    progress: Vec<OperationProgress>,
    submitted: bool,
    logical_aborted: bool,
    forgotten: bool,
    free_attempted: bool,
    safely_freed: bool,
    last_error: Option<EngineError>,
}

impl BatchState {
    fn snapshot(&self) -> BatchSnapshot {
        BatchSnapshot {
            operations: self.progress.clone(),
            logical_aborted: self.logical_aborted,
            safe_terminal: self.safely_freed,
        }
    }
}

pub(crate) struct Worker {
    engine: Box<dyn TransferEngine>,
    commands: Receiver<Command>,
    poll_interval: Duration,
    safe_shutdown: Arc<Mutex<Option<ShutdownOutcome>>>,
    accepting: bool,
    regions: BTreeMap<RegionId, RegionState>,
    peers: BTreeMap<PeerId, PeerState>,
    batches: BTreeMap<BatchId, BatchState>,
    region_order: Vec<RegionId>,
    peer_order: Vec<PeerId>,
    next_region_id: u64,
    next_peer_id: u64,
    next_batch_id: u64,
    poll_rotation: usize,
    cleanup_failure: Option<EngineError>,
    engine_shutdown_attempted: bool,
}

impl Worker {
    pub(crate) fn new(
        engine: Box<dyn TransferEngine>,
        commands: Receiver<Command>,
        poll_interval: Duration,
        safe_shutdown: Arc<Mutex<Option<ShutdownOutcome>>>,
    ) -> Self {
        Self {
            engine,
            commands,
            poll_interval,
            safe_shutdown,
            accepting: true,
            regions: BTreeMap::new(),
            peers: BTreeMap::new(),
            batches: BTreeMap::new(),
            region_order: Vec::new(),
            peer_order: Vec::new(),
            next_region_id: 1,
            next_peer_id: 1,
            next_batch_id: 1,
            poll_rotation: 0,
            cleanup_failure: None,
            engine_shutdown_attempted: false,
        }
    }

    pub(crate) fn run(mut self) {
        loop {
            let command = self.commands.recv_timeout(self.poll_interval);
            match command {
                Ok(command) => {
                    if self.handle(command) {
                        return;
                    }
                }
                Err(flume::RecvTimeoutError::Timeout) => {}
                Err(flume::RecvTimeoutError::Disconnected) => {
                    self.begin_shutdown();
                }
            }

            self.poll_batches();
            self.cleanup_requested_resources();
            if !self.accepting && self.try_finish_shutdown() {
                return;
            }
        }
    }

    fn handle(&mut self, command: Command) -> bool {
        match command {
            Command::LocalPeerDescriptor { reply } => {
                let _ = reply.send(self.engine.local_peer_descriptor());
            }
            Command::RegisterRegion { descriptor, reply } => {
                let _ = reply.send(self.register_region(descriptor));
            }
            Command::ReleaseRegion { id, reply } => {
                let result = self.request_region_release(id);
                if let Some(reply) = reply {
                    let _ = reply.send(result);
                }
            }
            Command::OpenPeer { descriptor, reply } => {
                let _ = reply.send(self.open_peer(descriptor));
            }
            Command::ReleasePeer { id, reply } => {
                let result = self.request_peer_release(id);
                if let Some(reply) = reply {
                    let _ = reply.send(result);
                }
            }
            Command::Submit { operations, reply } => {
                let _ = reply.send(self.submit(operations));
            }
            Command::Status { id, reply } => {
                let _ = reply.send(self.status(id));
            }
            Command::Abort { id, reply } => {
                let _ = reply.send(self.abort(id));
            }
            Command::ForgetBatch { id } => self.forget(id),
            Command::Shutdown { reply } => {
                self.begin_shutdown();
                self.poll_batches();
                self.cleanup_requested_resources();
                if self.try_finish_shutdown() {
                    let _ = reply.send(Ok(ShutdownOutcome::SafeTerminal));
                    return true;
                }
                let result = self.current_shutdown_outcome();
                let _ = reply.send(result);
            }
            Command::BeginShutdown => self.begin_shutdown(),
        }
        false
    }

    fn ensure_accepting(&self) -> Result<(), EngineError> {
        if self.accepting {
            Ok(())
        } else {
            Err(EngineError::WorkerClosed)
        }
    }

    fn register_region(&mut self, descriptor: RegionDescriptor) -> Result<RegionId, EngineError> {
        self.ensure_accepting()?;
        let id = RegionId::new(self.next_region_id);
        self.next_region_id = self.next_region_id.saturating_add(1);
        self.engine.register_region(id, &descriptor)?;
        self.regions.insert(
            id,
            RegionState {
                descriptor,
                release_requested: false,
                release_attempted: false,
            },
        );
        self.region_order.push(id);
        Ok(id)
    }

    fn open_peer(&mut self, descriptor: PeerDescriptor) -> Result<PeerId, EngineError> {
        self.ensure_accepting()?;
        let id = PeerId::new(self.next_peer_id);
        self.next_peer_id = self.next_peer_id.saturating_add(1);
        self.engine.open_peer(id, &descriptor)?;
        self.peers.insert(
            id,
            PeerState {
                release_requested: false,
                release_attempted: false,
            },
        );
        self.peer_order.push(id);
        Ok(id)
    }

    fn submit(&mut self, operations: Vec<TransferOperation>) -> Result<BatchId, EngineError> {
        self.ensure_accepting()?;
        if operations.is_empty() {
            return Err(EngineError::InvalidDescriptor {
                field: "operations",
                detail: "batch must contain at least one operation".into(),
            });
        }
        if self.inflight_count() >= MAX_INFLIGHT_BATCHES {
            return Err(EngineError::InFlightLimit {
                limit: MAX_INFLIGHT_BATCHES,
            });
        }

        let mut resolved = Vec::with_capacity(operations.len());
        for operation in operations {
            let region =
                self.regions
                    .get(&operation.region_id)
                    .ok_or(EngineError::ResourceClosed {
                        kind: "region",
                        id: operation.region_id.get(),
                    })?;
            if region.release_requested {
                return Err(EngineError::ResourceClosed {
                    kind: "region",
                    id: operation.region_id.get(),
                });
            }
            let peer = self
                .peers
                .get(&operation.peer_id)
                .ok_or(EngineError::ResourceClosed {
                    kind: "peer",
                    id: operation.peer_id.get(),
                })?;
            if peer.release_requested {
                return Err(EngineError::ResourceClosed {
                    kind: "peer",
                    id: operation.peer_id.get(),
                });
            }
            let local_address = region
                .descriptor
                .address()
                .checked_add(operation.local_offset)
                .ok_or(EngineError::RangeOverflow {
                    field: "local_address",
                })?;
            resolved.push(EngineOperation::new(
                operation.opcode,
                operation.region_id,
                operation.peer_id,
                local_address,
                operation.remote_address,
                operation.length,
            ));
        }

        let id = BatchId::new(self.next_batch_id);
        self.next_batch_id = self.next_batch_id.saturating_add(1);
        self.engine.allocate_batch(id, resolved.len())?;
        if let Err(error) = self.engine.submit_batch(id, &resolved) {
            return match self.engine.free_batch(id) {
                Ok(()) => Err(error),
                Err(cleanup) => {
                    let rollback = EngineError::Rollback {
                        operation: crate::mooncake::NativeOperation::SubmitBatch,
                        cleanup: cleanup.to_string(),
                    };
                    self.batches.insert(
                        id,
                        BatchState {
                            progress: vec![OperationProgress::default(); resolved.len()],
                            operations: resolved,
                            submitted: false,
                            logical_aborted: true,
                            forgotten: true,
                            free_attempted: true,
                            safely_freed: false,
                            last_error: Some(rollback.clone()),
                        },
                    );
                    Err(rollback)
                }
            };
        }
        self.batches.insert(
            id,
            BatchState {
                progress: vec![OperationProgress::default(); resolved.len()],
                operations: resolved,
                submitted: true,
                logical_aborted: false,
                forgotten: false,
                free_attempted: false,
                safely_freed: false,
                last_error: None,
            },
        );
        Ok(id)
    }

    fn status(&self, id: BatchId) -> Result<BatchSnapshot, EngineError> {
        let batch = self.batches.get(&id).ok_or(EngineError::ResourceClosed {
            kind: "batch",
            id: id.get(),
        })?;
        if let Some(error) = &batch.last_error {
            return Err(error.clone());
        }
        Ok(batch.snapshot())
    }

    fn abort(&mut self, id: BatchId) -> Result<(), EngineError> {
        let batch = self
            .batches
            .get_mut(&id)
            .ok_or(EngineError::ResourceClosed {
                kind: "batch",
                id: id.get(),
            })?;
        batch.logical_aborted = true;
        Ok(())
    }

    fn forget(&mut self, id: BatchId) {
        if let Some(batch) = self.batches.get_mut(&id) {
            batch.logical_aborted = true;
            batch.forgotten = true;
        }
        self.remove_forgotten_batches();
    }

    fn request_region_release(&mut self, id: RegionId) -> Result<(), EngineError> {
        let region = self
            .regions
            .get_mut(&id)
            .ok_or(EngineError::ResourceClosed {
                kind: "region",
                id: id.get(),
            })?;
        region.release_requested = true;
        self.cleanup_requested_resources();
        match self.regions.get(&id) {
            None => Ok(()),
            Some(_) if self.resource_in_use_region(id) => Ok(()),
            Some(_) => self.cleanup_failure.clone().map_or(Ok(()), Err),
        }
    }

    fn request_peer_release(&mut self, id: PeerId) -> Result<(), EngineError> {
        let peer = self.peers.get_mut(&id).ok_or(EngineError::ResourceClosed {
            kind: "peer",
            id: id.get(),
        })?;
        peer.release_requested = true;
        self.cleanup_requested_resources();
        match self.peers.get(&id) {
            None => Ok(()),
            Some(_) if self.resource_in_use_peer(id) => Ok(()),
            Some(_) => self.cleanup_failure.clone().map_or(Ok(()), Err),
        }
    }

    fn inflight_count(&self) -> usize {
        self.batches
            .values()
            .filter(|batch| !batch.safely_freed)
            .count()
    }

    fn poll_batches(&mut self) {
        let mut ids: Vec<_> = self
            .batches
            .iter()
            .filter_map(|(id, batch)| (batch.submitted && !batch.safely_freed).then_some(*id))
            .collect();
        if ids.is_empty() {
            return;
        }
        let rotation = self.poll_rotation % ids.len();
        ids.rotate_left(rotation);
        self.poll_rotation = self.poll_rotation.wrapping_add(1);

        for id in ids {
            self.poll_batch(id);
        }
        self.remove_forgotten_batches();
    }

    fn poll_batch(&mut self, id: BatchId) {
        let operation_count = match self.batches.get(&id) {
            Some(batch) => batch.operations.len(),
            None => return,
        };

        for operation_index in 0..operation_count {
            let already_terminal = self
                .batches
                .get(&id)
                .is_some_and(|batch| batch.progress[operation_index].state.is_terminal());
            if already_terminal {
                continue;
            }
            match self.engine.poll(id, operation_index) {
                Ok(progress) => {
                    let expected = self.batches[&id].operations[operation_index].length();
                    let progress = normalize_exact_progress(expected, progress);
                    let batch = self
                        .batches
                        .get_mut(&id)
                        .expect("batch exists while polling");
                    batch.progress[operation_index] = progress;
                    batch.last_error = None;
                }
                Err(error) => {
                    if let Some(batch) = self.batches.get_mut(&id) {
                        batch.last_error = Some(error);
                    }
                }
            }
        }

        let all_terminal = self
            .batches
            .get(&id)
            .is_some_and(|batch| batch.progress.iter().all(|value| value.state.is_terminal()));
        if !all_terminal {
            return;
        }
        let should_free = self
            .batches
            .get(&id)
            .is_some_and(|batch| !batch.free_attempted);
        if !should_free {
            return;
        }
        self.batches
            .get_mut(&id)
            .expect("batch exists")
            .free_attempted = true;
        match self.engine.free_batch(id) {
            Ok(()) => {
                let batch = self.batches.get_mut(&id).expect("batch exists");
                batch.safely_freed = true;
                batch.last_error = None;
            }
            Err(error) => {
                self.batches.get_mut(&id).expect("batch exists").last_error = Some(error);
            }
        }
    }

    fn resource_in_use_region(&self, id: RegionId) -> bool {
        self.batches.values().any(|batch| {
            !batch.safely_freed
                && batch
                    .operations
                    .iter()
                    .any(|operation| operation.region_id() == id)
        })
    }

    fn resource_in_use_peer(&self, id: PeerId) -> bool {
        self.batches.values().any(|batch| {
            !batch.safely_freed
                && batch
                    .operations
                    .iter()
                    .any(|operation| operation.peer_id() == id)
        })
    }

    fn cleanup_requested_resources(&mut self) {
        if self.cleanup_failure.is_some() {
            return;
        }
        let peer_ids: Vec<_> = self.peer_order.iter().rev().copied().collect();
        for id in peer_ids {
            let eligible = self
                .peers
                .get(&id)
                .is_some_and(|peer| peer.release_requested && !peer.release_attempted)
                && !self.resource_in_use_peer(id);
            if !eligible {
                continue;
            }
            self.peers
                .get_mut(&id)
                .expect("peer exists")
                .release_attempted = true;
            match self.engine.close_peer(id) {
                Ok(()) => {
                    self.peers.remove(&id);
                }
                Err(error) => {
                    self.cleanup_failure.get_or_insert(error);
                    return;
                }
            }
        }

        let region_ids: Vec<_> = self.region_order.iter().rev().copied().collect();
        for id in region_ids {
            let eligible = self
                .regions
                .get(&id)
                .is_some_and(|region| region.release_requested && !region.release_attempted)
                && !self.resource_in_use_region(id);
            if !eligible {
                continue;
            }
            self.regions
                .get_mut(&id)
                .expect("region exists")
                .release_attempted = true;
            match self.engine.unregister_region(id) {
                Ok(()) => {
                    self.regions.remove(&id);
                }
                Err(error) => {
                    self.cleanup_failure.get_or_insert(error);
                    return;
                }
            }
        }
    }

    fn remove_forgotten_batches(&mut self) {
        self.batches
            .retain(|_, batch| !(batch.forgotten && batch.safely_freed));
    }

    fn begin_shutdown(&mut self) {
        if !self.accepting {
            return;
        }
        self.accepting = false;
        for batch in self.batches.values_mut() {
            if !batch.safely_freed {
                batch.logical_aborted = true;
            }
        }
        for peer in self.peers.values_mut() {
            peer.release_requested = true;
        }
        for region in self.regions.values_mut() {
            region.release_requested = true;
        }
    }

    fn current_shutdown_outcome(&self) -> Result<ShutdownOutcome, EngineError> {
        if let Some(error) = &self.cleanup_failure {
            return Err(error.clone());
        }
        let batches = self
            .batches
            .iter()
            .filter_map(|(id, batch)| (!batch.safely_freed).then_some(*id))
            .collect();
        Ok(ShutdownOutcome::NotSafe { batches })
    }

    fn try_finish_shutdown(&mut self) -> bool {
        if self.accepting
            || self.cleanup_failure.is_some()
            || self.batches.values().any(|batch| !batch.safely_freed)
        {
            return false;
        }
        self.cleanup_requested_resources();
        if self.cleanup_failure.is_some() || !self.peers.is_empty() || !self.regions.is_empty() {
            return false;
        }
        if self.engine_shutdown_attempted {
            return false;
        }
        self.engine_shutdown_attempted = true;
        if let Err(error) = self.engine.shutdown() {
            self.cleanup_failure = Some(error);
            return false;
        }
        if let Ok(mut outcome) = self.safe_shutdown.lock() {
            *outcome = Some(ShutdownOutcome::SafeTerminal);
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mooncake::{OperationProgress, OperationState};

    #[test]
    fn completed_progress_stays_nonterminal_until_transferred_bytes_are_exact() {
        for actual in [65_536, 196_608] {
            assert_eq!(
                normalize_exact_progress(
                    131_072,
                    OperationProgress {
                        state: OperationState::Completed,
                        transferred_bytes: actual,
                    },
                ),
                OperationProgress {
                    state: OperationState::Pending,
                    transferred_bytes: actual,
                }
            );
        }
        assert_eq!(
            normalize_exact_progress(
                131_072,
                OperationProgress {
                    state: OperationState::Completed,
                    transferred_bytes: 131_072,
                },
            ),
            OperationProgress {
                state: OperationState::Completed,
                transferred_bytes: 131_072,
            }
        );
        assert_eq!(
            normalize_exact_progress(
                131_072,
                OperationProgress {
                    state: OperationState::Failed,
                    transferred_bytes: 0,
                },
            ),
            OperationProgress {
                state: OperationState::Failed,
                transferred_bytes: 0,
            }
        );
    }
}
