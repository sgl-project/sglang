//! Bounded source sequence state. The memory backend calls this only while its
//! placement write lock is held, so mutation and watermark visibility are atomic.
use crate::pb::{
    ApplyExternalKvBatchRequest, CacheWatermark, ReconcileSourceRequest, RegisterSourceRequest,
    SourceObservation,
};
use prost::Message;
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use tonic::Status;

#[derive(Clone, Copy, Debug)]
pub struct SourceLimits {
    pub max_sources: usize,
    pub max_retired_epochs: usize,
}
impl Default for SourceLimits {
    fn default() -> Self {
        Self {
            max_sources: 4096,
            max_retired_epochs: 8192,
        }
    }
}
#[derive(Debug)]
struct Pending {
    seq: u64,
    count: u32,
    fingerprints: Vec<[u8; 32]>,
    cache_watermark: Option<CacheWatermark>,
}
#[derive(Debug)]
struct Source {
    epoch: String,
    tiers: Vec<i32>,
    watermark: Option<u64>,
    next_seq: Option<u64>,
    highest_seen_seq: Option<u64>,
    pending: Option<Pending>,
    last_batch: Option<Pending>,
    baseline_complete: bool,
    reason: String,
    version: u64,
    suspended: bool,
    cache_watermark: Option<CacheWatermark>,
}
impl Source {
    fn view(&self, worker: &str, indexer_epoch: &str) -> SourceObservation {
        SourceObservation {
            worker_id: worker.into(),
            indexer_epoch: indexer_epoch.into(),
            source_epoch: self.epoch.clone(),
            tiers: self.tiers.clone(),
            watermark_seq: self.watermark,
            pending_seq: self.pending.as_ref().map(|p| p.seq),
            next_chunk_index: self
                .pending
                .as_ref()
                .map_or(0, |p| p.fingerprints.len() as u32),
            chunk_count: self.pending.as_ref().map_or(0, |p| p.count),
            complete: self.baseline_complete && self.pending.is_none() && !self.suspended,
            reason: if self.pending.is_some() {
                "partial_batch".into()
            } else {
                self.reason.clone()
            },
            source_version: self.version,
            suspended: self.suspended,
            cache_watermark: self.cache_watermark.clone(),
        }
    }
}
#[derive(Debug)]
pub(crate) struct SourceRegistry {
    sources: HashMap<String, Source>,
    retired: HashSet<(String, String)>,
    pub limits: SourceLimits,
    pub indexer_epoch: String,
}
/// Invalidation must still commit if the action is rejected as out of order.
pub(crate) struct Gate {
    pub clear: Vec<i32>,
    pub result: Result<bool, Status>,
}
impl Default for SourceRegistry {
    fn default() -> Self {
        Self {
            sources: HashMap::new(),
            retired: HashSet::new(),
            limits: SourceLimits::default(),
            indexer_epoch: uuid::Uuid::new_v4().to_string(),
        }
    }
}
impl SourceRegistry {
    pub fn contains(&self, worker: &str) -> bool {
        self.sources.contains_key(worker)
    }
    pub fn observations(&self) -> Vec<SourceObservation> {
        let mut values: Vec<_> = self
            .sources
            .iter()
            .map(|(worker, source)| source.view(worker, &self.indexer_epoch))
            .collect();
        values.sort_by(|a, b| a.worker_id.cmp(&b.worker_id));
        values
    }
    pub fn view(&self, worker: &str) -> Option<SourceObservation> {
        self.sources
            .get(worker)
            .map(|s| s.view(worker, &self.indexer_epoch))
    }
    pub fn register(
        &mut self,
        request: RegisterSourceRequest,
    ) -> Result<(Vec<i32>, SourceObservation), Status> {
        crate::service::validate_epoch(&request.source_epoch)?;
        if request.worker_id.is_empty()
            || request.worker_id.len() > 256
            || request.tiers.is_empty()
            || request.tiers.len() > 3
            || request.tiers.iter().any(|t| !(1..=3).contains(t))
            || request.tiers.iter().collect::<HashSet<_>>().len() != request.tiers.len()
        {
            return Err(Status::invalid_argument("invalid registration"));
        }
        let mut tiers = request.tiers;
        tiers.sort_unstable();
        if self
            .retired
            .contains(&(request.worker_id.clone(), request.source_epoch.clone()))
        {
            return Err(Status::failed_precondition("retired source epoch"));
        }
        if let Some(current) = self.sources.get(&request.worker_id) {
            if current.epoch == request.source_epoch {
                if current.tiers != tiers {
                    return Err(Status::already_exists(
                        "epoch registered with different tiers",
                    ));
                }
                return Ok((
                    vec![],
                    current.view(&request.worker_id, &self.indexer_epoch),
                ));
            }
            if request.previous_epoch.as_ref() != Some(&current.epoch) {
                return Err(Status::failed_precondition("source epoch CAS mismatch"));
            }
            if self.retired.len() >= self.limits.max_retired_epochs {
                return Err(Status::resource_exhausted("retired source epoch capacity"));
            }
        } else {
            if request.previous_epoch.is_some() {
                return Err(Status::failed_precondition(
                    "source absent after restart; explicit reconciliation required",
                ));
            }
            if self.sources.len() >= self.limits.max_sources {
                return Err(Status::resource_exhausted("source capacity"));
            }
        }
        let mut clear = tiers.clone();
        if let Some(old) = self.sources.remove(&request.worker_id) {
            clear.extend(old.tiers);
            self.retired.insert((request.worker_id.clone(), old.epoch));
        }
        clear.sort_unstable();
        clear.dedup();
        let source = Source {
            epoch: request.source_epoch,
            tiers,
            watermark: None,
            next_seq: None,
            highest_seen_seq: None,
            pending: None,
            last_batch: None,
            baseline_complete: false,
            reason: "registered_unknown".into(),
            version: 1,
            suspended: false,
            cache_watermark: None,
        };
        let view = source.view(&request.worker_id, &self.indexer_epoch);
        self.sources.insert(request.worker_id, source);
        Ok((clear, view))
    }
    pub fn reconcile(
        &mut self,
        request: ReconcileSourceRequest,
    ) -> Result<(Vec<i32>, SourceObservation), Status> {
        if request.expected_indexer_epoch != self.indexer_epoch {
            return Err(Status::failed_precondition(
                "indexer incarnation CAS mismatch",
            ));
        }
        if request.evidence_id.is_empty() || request.evidence_id.len() > 1024 {
            return Err(Status::invalid_argument(
                "bounded external evidence required",
            ));
        }
        let source = self
            .sources
            .get_mut(&request.worker_id)
            .ok_or_else(|| Status::failed_precondition("unregistered source"))?;
        if source.epoch != request.source_epoch || source.version != request.expected_source_version
        {
            return Err(Status::failed_precondition(
                "source reconciliation CAS mismatch",
            ));
        }
        if request.confirmed_empty
            && (source
                .highest_seen_seq
                .is_some_and(|seq| request.next_seq <= seq)
                || source.watermark.is_some_and(|seq| request.next_seq <= seq)
                || source
                    .pending
                    .as_ref()
                    .is_some_and(|p| request.next_seq <= p.seq)
                || source.next_seq.is_some_and(|seq| request.next_seq < seq))
        {
            return Err(Status::failed_precondition(
                "recovery sequence cut must follow all observed batches",
            ));
        }
        source.baseline_complete = request.confirmed_empty;
        source.suspended = !request.confirmed_empty;
        source.reason = if request.confirmed_empty {
            "confirmed_empty"
        } else {
            "invalidated"
        }
        .into();
        source.watermark = None;
        source.next_seq = Some(request.next_seq);
        source.pending = None;
        source.last_batch = None;
        source.cache_watermark = None;
        source.version += 1;
        Ok((
            source.tiers.clone(),
            source.view(&request.worker_id, &self.indexer_epoch),
        ))
    }
    pub fn replay_candidate(&self, r: &ApplyExternalKvBatchRequest) -> bool {
        self.sources.get(&r.worker_id).is_some_and(|s| {
            Some(&s.epoch) == r.source_epoch.as_ref() && is_historical_replay(s, r)
        })
    }
    pub fn apply(&mut self, request: &ApplyExternalKvBatchRequest) -> Gate {
        let fail = |s: &str| Gate {
            clear: vec![],
            result: Err(Status::failed_precondition(s.to_owned())),
        };
        if request.source_epoch.is_none() {
            return if self.contains(&request.worker_id) {
                fail("legacy apply cannot overwrite registered source")
            } else {
                Gate {
                    clear: vec![],
                    result: Ok(false),
                }
            };
        }
        let Some(source) = self.sources.get_mut(&request.worker_id) else {
            return fail("unregistered source epoch");
        };
        if Some(&source.epoch) != request.source_epoch.as_ref() {
            return fail("stale source epoch");
        }
        if source.suspended {
            return fail("source suspended; trusted reconciliation required");
        }
        if request
            .actions
            .iter()
            .any(|a| !source.tiers.contains(&a.tier))
        {
            return fail("action tier outside registered source domain");
        }
        let fingerprint: [u8; 32] = Sha256::digest(request.encode_to_vec()).into();
        if source.pending.as_ref().is_some_and(|p| {
            p.seq == request.seq
                && p.count == request.chunk_count
                && p.fingerprints.get(request.chunk_index as usize) == Some(&fingerprint)
        }) || source.last_batch.as_ref().is_some_and(|p| {
            p.seq == request.seq
                && p.count == request.chunk_count
                && p.fingerprints.get(request.chunk_index as usize) == Some(&fingerprint)
        }) {
            return Gate {
                clear: vec![],
                result: Ok(true),
            };
        }
        if request.observation_gap && self::is_historical_replay(source, request) {
            source.baseline_complete = false;
            source.reason = "reconnect_or_decode_gap".into();
            source.pending = None;
            source.version += 1;
            return Gate {
                clear: source.tiers.clone(),
                result: Ok(true),
            };
        }
        let duplicate = if let Some(p) = &source.pending {
            if request.seq == p.seq && request.chunk_index < p.fingerprints.len() as u32 {
                Some(
                    p.count == request.chunk_count
                        && p.fingerprints[request.chunk_index as usize] == fingerprint,
                )
            } else {
                None
            }
        } else {
            None
        };
        if let Some(matches) = duplicate {
            if matches {
                return Gate {
                    clear: vec![],
                    result: Ok(true),
                };
            }
            source.baseline_complete = false;
            source.reason = "conflicting_duplicate".into();
            source.pending = None;
            source.version += 1;
            return Gate {
                clear: source.tiers.clone(),
                result: Err(Status::data_loss("duplicate chunk payload conflict")),
            };
        }
        // Lower completed batches are duplicate historical replay within this
        // same epoch. They can never recreate a placement cleared by a gap.
        if source
            .watermark
            .is_some_and(|watermark| request.seq <= watermark)
        {
            if let Some(last) = &source.last_batch {
                if request.seq == last.seq
                    && (request.chunk_count != last.count
                        || last.fingerprints.get(request.chunk_index as usize)
                            != Some(&fingerprint))
                {
                    return Gate {
                        clear: vec![],
                        result: Err(Status::already_exists("completed chunk payload conflict")),
                    };
                }
            }
            return Gate {
                clear: vec![],
                result: Ok(true),
            };
        }
        if source.pending.as_ref().is_some_and(|p| request.seq < p.seq)
            || (source.pending.is_none()
                && source
                    .next_seq
                    .is_some_and(|expected| request.seq < expected))
        {
            return Gate {
                clear: vec![],
                result: Ok(true),
            };
        }
        if source.pending.is_none()
            && source
                .highest_seen_seq
                .is_some_and(|highest| request.seq < highest)
        {
            return Gate {
                clear: vec![],
                result: Ok(true),
            };
        }
        source.highest_seen_seq = Some(
            source
                .highest_seen_seq
                .map_or(request.seq, |old| old.max(request.seq)),
        );
        let mut cache_reset = false;
        if source.cache_watermark.is_some() && request.cache_watermark.is_none() {
            source.baseline_complete = false;
            source.reason = "cache_watermark_missing".into();
            source.pending = None;
            source.version += 1;
            return Gate {
                clear: source.tiers.clone(),
                result: Err(Status::failed_precondition(
                    "native cache watermark disappeared",
                )),
            };
        }
        if let Some(incoming) = &request.cache_watermark {
            if let Some(old) = &source.cache_watermark {
                if incoming.source_id != old.source_id || incoming.source_epoch != old.source_epoch
                {
                    source.baseline_complete = false;
                    source.suspended = true;
                    source.reason = "cache_source_epoch_changed".into();
                    source.pending = None;
                    source.version += 1;
                    return Gate {
                        clear: source.tiers.clone(),
                        result: Err(Status::failed_precondition(
                            "native cache source changed; trusted lifecycle reconciliation required",
                        )),
                    };
                }
                if incoming.generation < old.generation || incoming.version < old.version {
                    source.baseline_complete = false;
                    source.reason = "cache_watermark_regressed".into();
                    source.pending = None;
                    source.version += 1;
                    return Gate {
                        clear: source.tiers.clone(),
                        result: Err(Status::failed_precondition("stale native cache watermark")),
                    };
                }
                cache_reset = incoming.generation > old.generation
                    && source.pending.as_ref().is_none_or(|p| p.seq != request.seq);
            }
        }
        if source
            .pending
            .as_ref()
            .is_some_and(|p| p.seq == request.seq && p.cache_watermark != request.cache_watermark)
        {
            source.baseline_complete = false;
            source.reason = "chunk_cache_watermark_conflict".into();
            source.pending = None;
            source.version += 1;
            return Gate {
                clear: source.tiers.clone(),
                result: Err(Status::data_loss("chunk cache watermark conflict")),
            };
        }
        let order_ok = match &source.pending {
            Some(p) => {
                request.seq == p.seq
                    && request.chunk_count == p.count
                    && request.chunk_index == p.fingerprints.len() as u32
            }
            None => {
                request.chunk_index == 0
                    && source
                        .next_seq
                        .is_none_or(|expected| expected == request.seq)
            }
        };
        let mut clear = vec![];
        if cache_reset {
            clear = source.tiers.clone();
            source.baseline_complete = false;
            source.reason = "native_cache_generation_changed".into();
        }
        if request.observation_gap {
            clear = source.tiers.clone();
            source.baseline_complete = false;
            source.reason = "publisher_decode_gap".into();
        }
        if !order_ok {
            clear = source.tiers.clone();
            source.baseline_complete = false;
            source.reason = "sequence_or_chunk_gap".into();
            source.pending = None;
            source.version += 1;
            // Restart only from a complete chunk-zero boundary. A late missing
            // chunk can no longer rebuild observations from an abandoned batch.
            if request.chunk_index != 0 {
                source.next_seq = Some(request.seq);
                return Gate {
                    clear,
                    result: Err(Status::failed_precondition(
                        "missing/out-of-order chunk; restart batch at chunk zero",
                    )),
                };
            }
        }
        if source.pending.is_none() {
            source.pending = Some(Pending {
                seq: request.seq,
                count: request.chunk_count,
                fingerprints: Vec::new(),
                cache_watermark: request.cache_watermark.clone(),
            });
        }
        let pending = source.pending.as_mut().unwrap();
        pending.fingerprints.push(fingerprint);
        source.version += 1;
        if pending.fingerprints.len() == pending.count as usize {
            source.watermark = Some(request.seq);
            source.next_seq = request.seq.checked_add(1);
            if source.next_seq.is_none() {
                source.baseline_complete = false;
                source.suspended = true;
                source.reason = "sequence_exhausted".into();
            }
            source.cache_watermark = pending.cache_watermark.clone();
            source.last_batch = source.pending.take();
        }
        Gate {
            clear,
            result: Ok(false),
        }
    }
}

fn is_historical_replay(source: &Source, r: &ApplyExternalKvBatchRequest) -> bool {
    (source.pending.is_none() && source.highest_seen_seq.is_some_and(|seq| r.seq < seq))
        || source.watermark.is_some_and(|seq| r.seq <= seq)
        || source.pending.as_ref().is_some_and(|p| {
            r.seq < p.seq || (r.seq == p.seq && r.chunk_index < p.fingerprints.len() as u32)
        })
        || (source.pending.is_none() && source.next_seq.is_some_and(|seq| r.seq < seq))
}
