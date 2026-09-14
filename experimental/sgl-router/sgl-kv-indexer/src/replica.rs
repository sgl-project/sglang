//! Per-stream recovery and a full, flat materialized placement view.
//! All mutation and coverage reads share one lock; staging is never queryable.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use tokio::sync::Semaphore;
use tonic::{Request, Response, Status};

use crate::pb::kv_replica_server::{KvReplica, KvReplicaServer};
use crate::pb::*;
use crate::service::{
    validate_actions, BlockComponents, WorkerPrefixScanner, MAX_GRPC_DECODING_MESSAGE_SIZE,
};

type Key = (String, String, u32);
type Holdings = HashMap<(i64, i32), PlacementBlock>;
type Flat = HashMap<(String, i64), HashMap<(Key, i32), PlacementBlock>>;
const MAX_RECORDS: u64 = 10_000_000;
const MAX_CHUNK: usize = 4096;

#[derive(Debug)]
struct Stream {
    cut: SnapshotCut,
    session: String,
    next: u64,
    received: u64,
    staging: Option<Holdings>,
    holdings: Holdings,
    ready: bool,
    needs_snapshot: bool,
    touched: Instant,
}

impl Stream {
    fn progress(&self) -> StreamProgress {
        StreamProgress {
            session: self.session.clone(),
            next_sequence: self.next,
            ready: self.ready,
            needs_snapshot: self.needs_snapshot,
        }
    }
}

#[derive(Debug, Default)]
struct State {
    streams: HashMap<Key, Stream>,
    flat: Flat,
}

#[derive(Clone)]
pub struct ReplicaService {
    state: Arc<Mutex<State>>,
    lease: Duration,
    queries: Arc<Semaphore>,
}

impl ReplicaService {
    pub fn new(lease: Duration, max_queries: usize) -> Self {
        assert!(!lease.is_zero() && max_queries > 0);
        Self {
            state: Arc::default(),
            lease,
            queries: Arc::new(Semaphore::new(max_queries)),
        }
    }

    pub fn into_server(self) -> KvReplicaServer<Self> {
        KvReplicaServer::new(self)
            .max_decoding_message_size(MAX_GRPC_DECODING_MESSAGE_SIZE)
            .max_encoding_message_size(MAX_GRPC_DECODING_MESSAGE_SIZE)
    }

    /// Lease-expired streams are hidden immediately by queries; reclaim their
    /// soft state after a further lease so removed Workers do not accumulate.
    pub fn reap_expired(&self) -> Result<usize, Status> {
        let mut state = self.lock()?;
        let expired: Vec<_> = state
            .streams
            .iter()
            .filter(|(_, stream)| stream.touched.elapsed() >= self.lease.saturating_mul(2))
            .map(|(key, _)| key.clone())
            .collect();
        for key in &expired {
            if let Some(stream) = state.streams.remove(key) {
                remove_flat(&mut state.flat, key, &stream.holdings);
            }
        }
        Ok(expired.len())
    }

    fn lock(&self) -> Result<MutexGuard<'_, State>, Status> {
        self.state
            .lock()
            .map_err(|_| Status::internal("replica lock poisoned"))
    }

    fn owner<'a>(
        &self,
        state: &'a mut State,
        owner: Option<&StreamSession>,
    ) -> Result<(&'a mut Stream, Key), Status> {
        let owner = owner.ok_or_else(|| Status::invalid_argument("missing owner"))?;
        let key = key(owner.key.as_ref())?;
        let stream = state
            .streams
            .get_mut(&key)
            .ok_or_else(|| Status::failed_precondition("unknown stream; snapshot required"))?;
        if stream.session != owner.session || owner.session.is_empty() {
            return Err(Status::failed_precondition("obsolete recovery session"));
        }
        if stream.touched.elapsed() >= self.lease {
            stream.ready = false;
            stream.needs_snapshot = true;
        }
        Ok((stream, key))
    }
}

fn key(value: Option<&StreamKey>) -> Result<Key, Status> {
    let value = value.ok_or_else(|| Status::invalid_argument("missing stream key"))?;
    for s in [&value.namespace, &value.worker_id] {
        if s.is_empty() || s.len() > 256 {
            return Err(Status::invalid_argument("invalid stream identity"));
        }
    }
    Ok((
        value.namespace.clone(),
        value.worker_id.clone(),
        value.dp_rank,
    ))
}

fn valid_block(block: &PlacementBlock) -> Result<(), Status> {
    if !(1..=3).contains(&block.tier) || block.block_size == 0 || block.component_mask & !7 != 0 {
        return Err(Status::invalid_argument(
            "invalid tier, size or component mask",
        ));
    }
    if block.parent_block_hash == Some(block.block_hash) {
        return Err(Status::invalid_argument("self-parent block"));
    }
    Ok(())
}

fn remove_flat(flat: &mut Flat, key: &Key, holdings: &Holdings) {
    for &(hash, tier) in holdings.keys() {
        let flat_key = (key.0.clone(), hash);
        if let Some(placements) = flat.get_mut(&flat_key) {
            placements.remove(&(key.clone(), tier));
            if placements.is_empty() {
                flat.remove(&flat_key);
            }
        }
    }
}

fn insert_flat(flat: &mut Flat, key: &Key, holdings: &Holdings) {
    for (&(hash, tier), block) in holdings {
        flat.entry((key.0.clone(), hash))
            .or_default()
            .insert((key.clone(), tier), *block);
    }
}

fn mutate(holdings: &mut Holdings, actions: &[ExternalKvAction]) {
    for action in actions {
        match action.r#type {
            1 => {
                let mut parent = action.parent_block_hash;
                for (i, &hash) in action.hashes.iter().enumerate() {
                    holdings.insert(
                        (hash, action.tier),
                        PlacementBlock {
                            block_hash: hash,
                            parent_block_hash: parent,
                            tier: action.tier,
                            block_size: action.block_sizes.get(i).copied().unwrap_or(0),
                            component_mask: action.component_masks.get(i).copied().unwrap_or(0),
                        },
                    );
                    parent = Some(hash);
                }
            }
            2 => {
                for hash in &action.hashes {
                    holdings.remove(&(*hash, action.tier));
                }
            }
            3 => holdings.retain(|(_, tier), _| *tier != action.tier),
            _ => unreachable!("validated actions"),
        }
    }
}

#[tonic::async_trait]
impl KvReplica for ReplicaService {
    async fn begin_snapshot(
        &self,
        request: Request<BeginSnapshotRequest>,
    ) -> Result<Response<StreamProgress>, Status> {
        let request = request.into_inner();
        let cut = request
            .cut
            .ok_or_else(|| Status::invalid_argument("missing snapshot cut"))?;
        let descriptor = cut
            .stream
            .as_ref()
            .ok_or_else(|| Status::invalid_argument("missing descriptor"))?;
        let key = key(descriptor.key.as_ref())?;
        if cut.version != 2
            || cut.worker_epoch.is_empty()
            || cut.worker_epoch.len() > 256
            || cut.barrier_id.is_empty()
            || cut.barrier_id.len() > 256
            || cut.barrier_seq.checked_add(1) != Some(cut.resume_seq)
            || cut.record_count > MAX_RECORDS
            || descriptor.page_size == 0
            || descriptor.hash_schema_version != 1
            || descriptor.model.is_empty()
            || descriptor.worker_address.is_empty()
        {
            return Err(Status::invalid_argument("invalid snapshot v2 header"));
        }
        if let Some(spec) = descriptor.cache_spec {
            if spec.version != 1
                || spec.components & 1 == 0
                || spec.components & !7 != 0
                || (spec.components & 2 != 0 && spec.swa_window_tokens == 0)
            {
                return Err(Status::invalid_argument("unsupported worker cache spec"));
            }
        }
        let mut state = self.lock()?;
        if let Some(old) = state.streams.get(&key) {
            if old.touched.elapsed() < self.lease && old.session != request.session {
                return Err(Status::already_exists("stream has a live Bridge owner"));
            }
        }
        let old = state.streams.remove(&key);
        if let Some(old) = &old {
            remove_flat(&mut state.flat, &key, &old.holdings);
        }
        let stream = Stream {
            next: cut.resume_seq,
            received: 0,
            cut,
            session: uuid::Uuid::new_v4().to_string(),
            staging: Some(HashMap::new()),
            holdings: HashMap::new(),
            ready: false,
            needs_snapshot: false,
            touched: Instant::now(),
        };
        let progress = stream.progress();
        state.streams.insert(key, stream);
        Ok(Response::new(progress))
    }

    async fn snapshot_chunk(
        &self,
        request: Request<SnapshotChunkRequest>,
    ) -> Result<Response<StreamProgress>, Status> {
        let request = request.into_inner();
        if request.blocks.len() > MAX_CHUNK {
            return Err(Status::resource_exhausted("snapshot chunk too large"));
        }
        let mut seen = HashSet::new();
        for block in &request.blocks {
            valid_block(block)?;
            if !seen.insert((block.block_hash, block.tier)) {
                return Err(Status::invalid_argument("duplicate snapshot placement"));
            }
        }
        let mut state = self.lock()?;
        let (stream, _) = self.owner(&mut state, request.owner.as_ref())?;
        if stream.needs_snapshot {
            return Ok(Response::new(stream.progress()));
        }
        if stream.received != request.offset
            || stream.received + request.blocks.len() as u64 > stream.cut.record_count
        {
            return Err(Status::failed_precondition(
                "snapshot offset or count mismatch",
            ));
        }
        let staging = stream
            .staging
            .as_mut()
            .ok_or_else(|| Status::failed_precondition("no staging snapshot"))?;
        if request
            .blocks
            .iter()
            .any(|b| staging.contains_key(&(b.block_hash, b.tier)))
        {
            return Err(Status::invalid_argument("duplicate snapshot placement"));
        }
        stream.received += request.blocks.len() as u64;
        staging.extend(
            request
                .blocks
                .into_iter()
                .map(|b| ((b.block_hash, b.tier), b)),
        );
        stream.touched = Instant::now();
        Ok(Response::new(stream.progress()))
    }

    async fn apply_live(
        &self,
        request: Request<LiveBatchRequest>,
    ) -> Result<Response<StreamProgress>, Status> {
        let request = request.into_inner();
        validate_actions(&request.actions)?;
        for action in &request.actions {
            if action.r#type == 1 {
                if action.block_sizes.len() != action.hashes.len()
                    || action.block_sizes.contains(&0)
                    || action.component_masks.iter().any(|mask| mask & !7 != 0)
                {
                    return Err(Status::invalid_argument(
                        "v2 live report requires valid block sizes/components",
                    ));
                }
            }
        }
        let mut state = self.lock()?;
        let (stream, key) = self.owner(&mut state, request.owner.as_ref())?;
        if request.worker_epoch != stream.cut.worker_epoch {
            stream.ready = false;
            stream.needs_snapshot = true;
        }
        if stream.needs_snapshot {
            return Ok(Response::new(stream.progress()));
        }
        if request.sequence < stream.next {
            return Ok(Response::new(stream.progress()));
        }
        if request.sequence > stream.next {
            stream.ready = false;
            return Ok(Response::new(stream.progress()));
        }
        if stream.staging.is_some() && stream.received != stream.cut.record_count {
            return Err(Status::failed_precondition("snapshot chunks incomplete"));
        }
        let next = stream
            .next
            .checked_add(1)
            .ok_or_else(|| Status::out_of_range("sequence exhausted"))?;
        if let Some(staging) = &mut stream.staging {
            mutate(staging, &request.actions);
        } else {
            // Temporarily remove the stream so flat + reverse holdings can be
            // changed under the same lock without cloning the full placement.
            let mut owned = state.streams.remove(&key).expect("checked owner");
            // Only affected reverse holdings are removed from the flat map.
            let mut affected = Holdings::new();
            for action in &request.actions {
                if action.r#type == 3 {
                    affected.extend(
                        owned
                            .holdings
                            .iter()
                            .filter(|((_, t), _)| *t == action.tier)
                            .map(|(k, v)| (*k, *v)),
                    );
                } else {
                    for hash in &action.hashes {
                        if let Some(block) = owned.holdings.get(&(*hash, action.tier)) {
                            affected.insert((*hash, action.tier), *block);
                        }
                    }
                }
            }
            remove_flat(&mut state.flat, &key, &affected);
            mutate(&mut owned.holdings, &request.actions);
            for action in &request.actions {
                for hash in &action.hashes {
                    if let Some(block) = owned.holdings.get(&(*hash, action.tier)) {
                        state
                            .flat
                            .entry((key.0.clone(), *hash))
                            .or_default()
                            .insert((key.clone(), action.tier), *block);
                    }
                }
            }
            state.streams.insert(key.clone(), owned);
        }
        let stream = state.streams.get_mut(&key).expect("checked owner");
        stream.next = next;
        // Live events alone cannot renew readiness after an idle gap: the
        // Bridge periodically proves catch-up against the Worker replay head.
        Ok(Response::new(stream.progress()))
    }

    async fn confirm_stream(
        &self,
        request: Request<ConfirmStreamRequest>,
    ) -> Result<Response<StreamProgress>, Status> {
        let request = request.into_inner();
        let mut state = self.lock()?;
        let (stream, key) = self.owner(&mut state, request.owner.as_ref())?;
        if request.worker_epoch != stream.cut.worker_epoch {
            stream.ready = false;
            stream.needs_snapshot = true;
        }
        if stream.needs_snapshot {
            return Ok(Response::new(stream.progress()));
        }
        if request.resume_seq != stream.next {
            stream.ready = false;
            return Ok(Response::new(stream.progress()));
        }
        if stream.staging.is_some() {
            if stream.received != stream.cut.record_count
                || request.barrier_id != stream.cut.barrier_id
                || request.barrier_seq != stream.cut.barrier_seq
            {
                return Err(Status::failed_precondition(
                    "snapshot barrier/count not confirmed",
                ));
            }
            let mut owned = state.streams.remove(&key).expect("checked owner");
            owned.holdings = owned.staging.take().expect("checked staging");
            insert_flat(&mut state.flat, &key, &owned.holdings);
            state.streams.insert(key.clone(), owned);
        }
        let stream = state.streams.get_mut(&key).expect("checked owner");
        stream.ready = true;
        stream.touched = Instant::now();
        Ok(Response::new(stream.progress()))
    }

    async fn invalidate_stream(
        &self,
        request: Request<StreamSession>,
    ) -> Result<Response<StreamProgress>, Status> {
        let mut state = self.lock()?;
        let (stream, _) = self.owner(&mut state, Some(request.get_ref()))?;
        stream.ready = false;
        stream.needs_snapshot = true;
        stream.staging = None;
        Ok(Response::new(stream.progress()))
    }

    async fn remove_stream(
        &self,
        request: Request<StreamSession>,
    ) -> Result<Response<StreamProgress>, Status> {
        let mut state = self.lock()?;
        let (_, key) = self.owner(&mut state, Some(request.get_ref()))?;
        let old = state.streams.remove(&key).expect("checked owner");
        remove_flat(&mut state.flat, &key, &old.holdings);
        Ok(Response::new(StreamProgress::default()))
    }

    async fn match_prefix(
        &self,
        request: Request<ReplicaPrefixRequest>,
    ) -> Result<Response<ReplicaPrefixResponse>, Status> {
        let _permit = self
            .queries
            .try_acquire()
            .map_err(|_| Status::resource_exhausted("replica prefix queries overloaded"))?;
        let request = request.into_inner();
        if request.eligible_streams.is_empty() || request.eligible_streams.len() > 4096 {
            return Err(Status::invalid_argument("provide 1..4096 eligible streams"));
        }
        let state = self.lock()?;
        let mut response = ReplicaPrefixResponse {
            complete: true,
            ..Default::default()
        };
        let limit = if request.max_blocks == 0 {
            request.hashes.len()
        } else {
            request.hashes.len().min(request.max_blocks as usize)
        };
        for eligible in &request.eligible_streams {
            let key = key(Some(eligible))?;
            if key.0 != request.namespace {
                return Err(Status::invalid_argument("eligible namespace mismatch"));
            }
            let stream = state.streams.get(&key);
            let ready = stream.is_some_and(|s| {
                let d = s.cut.stream.as_ref().expect("validated descriptor");
                s.ready
                    && !s.needs_snapshot
                    && s.touched.elapsed() < self.lease
                    && d.model == request.model
                    && d.page_size == request.page_size
                    && d.hash_schema_version == request.hash_schema_version
                    && d.is_bigram == request.is_bigram
            });
            response.complete &= ready;
            response.coverage.push(StreamCoverage {
                stream: stream.and_then(|s| s.cut.stream.clone()).or_else(|| {
                    Some(StreamDescriptor {
                        key: Some(eligible.clone()),
                        ..Default::default()
                    })
                }),
                worker_epoch: stream
                    .map(|s| s.cut.worker_epoch.clone())
                    .unwrap_or_default(),
                watermark: stream.map(|s| s.next.saturating_sub(1)).unwrap_or(0),
                ready,
            });
            if !ready {
                continue;
            }
            let stream = stream.expect("ready stream");
            let descriptor = stream.cut.stream.as_ref().expect("validated descriptor");
            let mut scanner = WorkerPrefixScanner::new(descriptor.cache_spec.as_ref());
            for hash in &request.hashes[..limit] {
                let mut block = BlockComponents {
                    token_count: 0,
                    tier_masks: Vec::new(),
                };
                if let Some(placements) = state.flat.get(&(key.0.clone(), *hash)) {
                    for tier in [1, 2] {
                        if let Some(p) = placements.get(&(key.clone(), tier)) {
                            block.token_count = p.block_size;
                            block.tier_masks.push((tier, p.component_mask));
                        }
                    }
                }
                scanner.push((!block.tier_masks.is_empty()).then_some(&block));
            }
            if scanner.prefix() > 0 {
                response.matches.push(ReplicaPrefixMatch {
                    worker_address: descriptor.worker_address.clone(),
                    matched_prefix_blocks: scanner.prefix(),
                    worker_id: eligible.worker_id.clone(),
                    dp_rank: eligible.dp_rank,
                });
            }
        }
        response
            .matches
            .sort_by_key(|m| std::cmp::Reverse(m.matched_prefix_blocks));
        Ok(Response::new(response))
    }
}
