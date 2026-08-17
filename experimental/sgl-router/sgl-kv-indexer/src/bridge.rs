// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! SGLang KV event bridge.
//!
//! Subscribes to a worker's ZMQ KV-event stream, decodes each batch, and
//! forwards it to the indexer over gRPC.
//!
//! It keeps a reconnect supervisor but does not recover data: no sequence
//! tracking, no replay of missed batches, no incarnation token, no liveness
//! heartbeat. A sequence gap is logged and ignored, and events produced while
//! the bridge is disconnected are lost.

use std::io::Cursor;
use std::time::Duration;

use rmpv::decode::value::read_value;
use rmpv::Value;
use tonic::transport::{Channel, Endpoint};
use tonic::{Code, Status};
use tracing::{debug, info, warn};
use zeromq::{Socket, SocketRecv, SubSocket};

use crate::pb::kv_indexer_client::KvIndexerClient;
use crate::pb::{
    ApplyExternalKvBatchRequest, ExternalKvAction, ExternalKvActionType, TierType, WorkerCacheSpec,
};
use crate::service::{component_bit, COMPONENT_SWA, MAX_ACTIONS_PER_BATCH, MAX_HASHES_PER_REQUEST};

/// Backoff bounds for the reconnect supervisor loop.
const RECONNECT_MIN_DELAY: Duration = Duration::from_millis(500);
const RECONNECT_MAX_DELAY: Duration = Duration::from_secs(10);
const GRPC_CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
const GRPC_REQUEST_TIMEOUT: Duration = Duration::from_secs(10);

#[derive(Debug, Clone)]
pub struct BridgeConfig {
    pub worker_id: String,
    /// The worker's KV-transfer address, forwarded on every apply batch so the
    /// indexer can answer MatchExternalKv with an address. Empty if unset.
    pub worker_address: String,
    pub event_endpoint: String,
    pub event_topic: String,
    pub indexer_endpoint: String,
    pub clear_tiers: Vec<i32>,
    /// The worker's component cache spec, forwarded on every apply batch. `None`
    /// for a legacy / full-only worker that reports no component metadata.
    pub cache_spec: Option<WorkerCacheSpec>,
}

impl BridgeConfig {
    pub fn from_env() -> Result<Self, BridgeError> {
        let worker_id = std::env::var("KV_INDEXER_WORKER_ID")
            .map_err(|_| BridgeError::Config("KV_INDEXER_WORKER_ID is required".to_string()))?;
        let worker_address = std::env::var("KV_INDEXER_WORKER_ADDRESS").unwrap_or_default();
        let event_endpoint = std::env::var("SGLANG_KV_EVENT_ENDPOINT")
            .unwrap_or_else(|_| "tcp://127.0.0.1:5557".to_string());
        // Match SGLang's upstream ZMQ publisher default. Deployments that use a
        // non-empty topic must configure the same value on both sides.
        let event_topic = std::env::var("SGLANG_KV_EVENT_TOPIC").unwrap_or_default();
        let indexer_endpoint = std::env::var("KV_INDEXER_ENDPOINT")
            .unwrap_or_else(|_| "http://[::1]:50051".to_string());
        let clear_tiers = parse_clear_tiers(
            &std::env::var("KV_INDEXER_CLEAR_TIERS").unwrap_or_else(|_| "HBM,DRAM,SSD".to_string()),
        )?;
        let cache_spec = cache_spec_from_env()?;

        Ok(Self {
            worker_id,
            worker_address,
            event_endpoint,
            event_topic,
            indexer_endpoint,
            clear_tiers,
            cache_spec,
        })
    }
}

#[derive(Debug)]
pub enum BridgeError {
    Config(String),
    Decode(String),
    Rpc(tonic::Status),
    PermanentRpc(tonic::Status),
    Transport(tonic::transport::Error),
    Zmq(zeromq::ZmqError),
}

impl std::fmt::Display for BridgeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BridgeError::Config(message) => write!(f, "bridge config error: {message}"),
            BridgeError::Decode(message) => write!(f, "bridge decode error: {message}"),
            BridgeError::Rpc(status) => write!(f, "indexer rpc error: {status}"),
            BridgeError::PermanentRpc(status) => {
                write!(f, "permanent indexer rpc error: {status}")
            }
            BridgeError::Transport(error) => write!(f, "indexer transport error: {error}"),
            BridgeError::Zmq(error) => write!(f, "zmq error: {error}"),
        }
    }
}

impl std::error::Error for BridgeError {}

impl BridgeError {
    fn is_permanent(&self) -> bool {
        matches!(self, BridgeError::Config(_) | BridgeError::PermanentRpc(_))
    }
}

impl From<zeromq::ZmqError> for BridgeError {
    fn from(error: zeromq::ZmqError) -> Self {
        BridgeError::Zmq(error)
    }
}

impl From<tonic::transport::Error> for BridgeError {
    fn from(error: tonic::transport::Error) -> Self {
        BridgeError::Transport(error)
    }
}

fn classify_rpc(status: Status) -> BridgeError {
    match status.code() {
        Code::InvalidArgument
        | Code::FailedPrecondition
        | Code::NotFound
        | Code::AlreadyExists
        | Code::OutOfRange
        | Code::Unauthenticated
        | Code::PermissionDenied
        | Code::Unimplemented
        | Code::DataLoss => BridgeError::PermanentRpc(status),
        // RESOURCE_EXHAUSTED is the indexer shedding load or refusing an
        // oversized batch. Reconnecting loses that batch's events, which costs
        // routing accuracy; exiting loses every later batch too.
        _ => BridgeError::Rpc(status),
    }
}

/// A single indexer mutation, kept in the exact order it appeared in the event
/// batch so mutations on the same hash are never reordered.
///
/// `Report` carries per-hash component metadata in arrays index-aligned with
/// `hashes`: `masks[i]` is `None` for a legacy whole-block store, and
/// `block_sizes[i]` is the reported token count, `None` when none was supplied.
#[derive(Debug, PartialEq, Eq)]
enum Action {
    Report {
        tier: i32,
        hashes: Vec<i64>,
        masks: Vec<Option<u32>>,
        block_sizes: Vec<Option<u32>>,
    },
    Revoke {
        tier: i32,
        hashes: Vec<i64>,
    },
    ClearAll,
}

#[derive(Debug, Default)]
struct EventActions {
    actions: Vec<Action>,
}

impl EventActions {
    /// Append a store for the block hashes of one `BlockStored`, coalescing only
    /// with an immediately-preceding store to the same tier and never across a
    /// revoke/clear, so the final per-hash state is preserved. All hashes here
    /// share the event's component mask and block size.
    fn report(&mut self, tier: i32, hashes: Vec<i64>, mask: Option<u32>, block_size: Option<u32>) {
        if hashes.is_empty() {
            return;
        }
        let n = hashes.len();
        if let Some(Action::Report {
            tier: last_tier,
            hashes: last,
            masks,
            block_sizes,
        }) = self.actions.last_mut()
        {
            if *last_tier == tier {
                last.extend(hashes);
                masks.extend(std::iter::repeat_n(mask, n));
                block_sizes.extend(std::iter::repeat_n(block_size, n));
                return;
            }
        }
        self.actions.push(Action::Report {
            tier,
            hashes,
            masks: vec![mask; n],
            block_sizes: vec![block_size; n],
        });
    }

    fn revoke(&mut self, tier: i32, hashes: Vec<i64>) {
        if hashes.is_empty() {
            return;
        }
        if let Some(Action::Revoke {
            tier: last_tier,
            hashes: last,
        }) = self.actions.last_mut()
        {
            if *last_tier == tier {
                last.extend(hashes);
                return;
            }
        }
        self.actions.push(Action::Revoke { tier, hashes });
    }

    fn clear_all(&mut self) {
        self.actions.push(Action::ClearAll);
    }
}

pub async fn run_bridge(config: BridgeConfig) -> Result<(), BridgeError> {
    run_bridge_until(config, std::future::pending()).await
}

/// [`run_bridge`], but returns as soon as `shutdown` resolves. An in-flight apply
/// is dropped: acknowledgements are not tracked, so that batch is lost.
pub async fn run_bridge_until<F>(config: BridgeConfig, shutdown: F) -> Result<(), BridgeError>
where
    F: std::future::Future<Output = ()>,
{
    tokio::select! {
        result = supervise(config) => result,
        () = shutdown => {
            info!("bridge stopped by shutdown signal");
            Ok(())
        }
    }
}

async fn supervise(config: BridgeConfig) -> Result<(), BridgeError> {
    info!(
        worker_id = %config.worker_id,
        event_endpoint = %config.event_endpoint,
        event_topic = %config.event_topic,
        indexer_endpoint = %config.indexer_endpoint,
        "starting SGLang KV event bridge"
    );

    // Supervisor loop: (re)connect to both the indexer and the ZMQ publisher,
    // run until a connection-level error, then back off and retry. Decode-level
    // problems are handled inside the session and never tear down the bridge.
    // Reconnecting recovers the connection only: events published while
    // disconnected are lost.
    let mut delay = RECONNECT_MIN_DELAY;
    loop {
        match connect(&config).await {
            Ok((client, subscriber)) => {
                delay = RECONNECT_MIN_DELAY;
                match run_session(&config, client, subscriber).await {
                    Ok(()) => {
                        info!("bridge shut down cleanly");
                        return Ok(());
                    }
                    Err(error) => {
                        if error.is_permanent() {
                            return Err(error);
                        }
                        warn!(%error, retry_in = ?delay, "bridge session lost; reconnecting");
                    }
                }
            }
            Err(error) => {
                if error.is_permanent() {
                    return Err(error);
                }
                warn!(%error, retry_in = ?delay, "bridge connect failed; retrying");
            }
        }

        tokio::time::sleep(delay).await;
        delay = (delay * 2).min(RECONNECT_MAX_DELAY);
    }
}

async fn connect(
    config: &BridgeConfig,
) -> Result<(KvIndexerClient<Channel>, SubSocket), BridgeError> {
    let channel = Endpoint::from_shared(config.indexer_endpoint.clone())?
        .connect_timeout(GRPC_CONNECT_TIMEOUT)
        .timeout(GRPC_REQUEST_TIMEOUT)
        .connect()
        .await?;
    let client = KvIndexerClient::new(channel);
    let mut subscriber = SubSocket::new();
    subscriber.subscribe(&config.event_topic).await?;
    subscriber.connect(&config.event_endpoint).await?;
    info!("bridge session established");
    Ok((client, subscriber))
}

/// Runs a single connected session. Returns `Ok(())` only on a clean shutdown
/// (ctrl-c); any connection-level error is propagated so the supervisor can
/// reconnect.
async fn run_session(
    config: &BridgeConfig,
    mut client: KvIndexerClient<Channel>,
    mut subscriber: SubSocket,
) -> Result<(), BridgeError> {
    // Tracked only to log a discontinuity. Nothing acts on it.
    let mut last_seq: Option<u64> = None;

    loop {
        let message = tokio::select! {
            result = subscriber.recv() => result?,
            _ = tokio::signal::ctrl_c() => {
                info!("received ctrl-c; shutting down bridge");
                return Ok(());
            }
        };

        let (seq, payload) = match parse_zmq_frames(&message.into_vec()) {
            Ok((seq, payload)) => (seq, payload.to_vec()),
            Err(error) => {
                warn!(%error, "skipping malformed ZMQ message");
                continue;
            }
        };

        if let Some(previous) = last_seq {
            if seq != previous.wrapping_add(1) {
                warn!(
                    previous,
                    actual = seq,
                    "SGLang KV event sequence is not contiguous; this build does not recover the gap"
                );
            }
        }
        last_seq = Some(seq);

        forward_raw_batch(config, &mut client, seq, &payload).await?;
    }
}
fn parse_zmq_frames(frames: &[bytes::Bytes]) -> Result<(u64, &[u8]), BridgeError> {
    match frames.len() {
        2 => Ok((decode_seq(&frames[0])?, frames[1].as_ref())),
        3 => Ok((decode_seq(&frames[1])?, frames[2].as_ref())),
        n => Err(BridgeError::Decode(format!(
            "expected 2 or 3 ZMQ frames, got {n}"
        ))),
    }
}

fn decode_seq(bytes: &[u8]) -> Result<u64, BridgeError> {
    let seq_bytes: [u8; 8] = bytes
        .try_into()
        .map_err(|_| BridgeError::Decode("sequence frame must be 8 bytes".to_string()))?;
    Ok(u64::from_be_bytes(seq_bytes))
}

/// Decodes one raw batch and forwards it. A batch that cannot be decoded, or
/// that carries no supported mutation, is skipped without an RPC.
async fn forward_raw_batch(
    config: &BridgeConfig,
    client: &mut KvIndexerClient<Channel>,
    seq: u64,
    payload: &[u8],
) -> Result<(), BridgeError> {
    let actions = match decode_event_batch(payload) {
        Ok(actions) => actions,
        Err(error) => {
            warn!(seq, %error, "skipping undecodable event batch");
            return Ok(());
        }
    };

    let request = build_apply_request(config, seq, actions);
    // One ZMQ batch can hold more mutations than a single apply RPC admits, so
    // send the parts in order. A later failure leaves an applied prefix, which
    // beats rejecting and losing the whole event batch.
    for request in split_apply_request(request) {
        client
            .apply_external_kv_batch(request)
            .await
            .map_err(classify_rpc)?;
    }
    Ok(())
}

/// Maps a decoded `EventActions` into a single `ApplyExternalKvBatchRequest`,
/// preserving per-action order. A `ClearAll` is expanded in place into one
/// `CLEAR_ALL_AT_TIER` action per configured clear tier.
fn build_apply_request(
    config: &BridgeConfig,
    seq: u64,
    events: EventActions,
) -> ApplyExternalKvBatchRequest {
    let mut actions = Vec::with_capacity(events.actions.len());
    for action in events.actions {
        match action {
            Action::Report {
                tier,
                hashes,
                masks,
                block_sizes,
            } => actions.push(ExternalKvAction {
                r#type: ExternalKvActionType::ActionReport as i32,
                tier,
                hashes,
                // Emit the per-hash arrays only when some hash carries
                // component data; a fully-legacy report leaves them empty so
                // the backend keeps the whole-block fast path.
                component_masks: encode_component_masks(&masks),
                block_sizes: encode_block_sizes(&block_sizes),
            }),
            Action::Revoke { tier, hashes } => actions.push(ExternalKvAction {
                r#type: ExternalKvActionType::ActionRevoke as i32,
                tier,
                hashes,
                component_masks: Vec::new(),
                block_sizes: Vec::new(),
            }),
            Action::ClearAll => {
                for tier in &config.clear_tiers {
                    actions.push(ExternalKvAction {
                        r#type: ExternalKvActionType::ActionClearAllAtTier as i32,
                        tier: *tier,
                        hashes: Vec::new(),
                        component_masks: Vec::new(),
                        block_sizes: Vec::new(),
                    });
                }
            }
        }
    }

    ApplyExternalKvBatchRequest {
        worker_id: config.worker_id.clone(),
        seq,
        actions,
        worker_address: config.worker_address.clone(),
        cache_spec: config.cache_spec,
    }
}

/// Splits one decoded ZMQ batch into apply RPCs within the service's action and
/// hash bounds, preserving action order and per-hash index alignment.
///
/// Every part reuses the source `seq`, which is safe because `seq` is
/// observability only (see the proto): applies are never deduplicated or fenced,
/// so a repeated `seq` cannot get a part dropped as stale.
fn split_apply_request(request: ApplyExternalKvBatchRequest) -> Vec<ApplyExternalKvBatchRequest> {
    let mut template = request;
    let actions = std::mem::take(&mut template.actions);
    let mut batches = Vec::new();
    let mut current = Vec::new();
    let mut current_hashes = 0usize;

    let flush = |actions: &mut Vec<ExternalKvAction>,
                 batches: &mut Vec<ApplyExternalKvBatchRequest>| {
        if actions.is_empty() {
            return;
        }
        batches.push(ApplyExternalKvBatchRequest {
            actions: std::mem::take(actions),
            ..template.clone()
        });
    };

    for action in actions.into_iter().flat_map(split_action) {
        let action_hashes = action.hashes.len();
        if !current.is_empty()
            && (current.len() == MAX_ACTIONS_PER_BATCH
                || current_hashes + action_hashes > MAX_HASHES_PER_REQUEST)
        {
            flush(&mut current, &mut batches);
            current_hashes = 0;
        }
        current_hashes += action_hashes;
        current.push(action);
    }
    flush(&mut current, &mut batches);
    batches
}

fn split_action(action: ExternalKvAction) -> Vec<ExternalKvAction> {
    if action.hashes.len() <= MAX_HASHES_PER_REQUEST {
        return vec![action];
    }

    (0..action.hashes.len())
        .step_by(MAX_HASHES_PER_REQUEST)
        .map(|start| {
            let end = (start + MAX_HASHES_PER_REQUEST).min(action.hashes.len());
            ExternalKvAction {
                r#type: action.r#type,
                tier: action.tier,
                hashes: action.hashes[start..end].to_vec(),
                component_masks: slice_or_empty(&action.component_masks, start, end),
                block_sizes: slice_or_empty(&action.block_sizes, start, end),
            }
        })
        .collect()
}

/// Slices a per-hash array alongside its `hashes` slice. Empty is the legacy
/// "field absent" signal; non-empty arrays are aligned by `build_apply_request`.
fn slice_or_empty<T: Clone>(values: &[T], start: usize, end: usize) -> Vec<T> {
    if values.is_empty() {
        return Vec::new();
    }
    values[start..end].to_vec()
}

/// Maps per-hash component sets to the wire form: an empty vector (the legacy
/// signal) when no hash carries components, otherwise one mask per hash.
fn encode_component_masks(masks: &[Option<u32>]) -> Vec<u32> {
    if masks.iter().all(Option::is_none) {
        return Vec::new();
    }
    masks.iter().map(|mask| mask.unwrap_or_default()).collect()
}

/// Maps per-hash block sizes to the wire form. Returns an empty vector when no
/// hash carries a size, otherwise one entry per hash (`0` for a legacy hash).
fn encode_block_sizes(block_sizes: &[Option<u32>]) -> Vec<u32> {
    if block_sizes.iter().all(Option::is_none) {
        return Vec::new();
    }
    block_sizes
        .iter()
        .map(|size| size.unwrap_or_default())
        .collect()
}

fn decode_event_batch(payload: &[u8]) -> Result<EventActions, BridgeError> {
    decode_event_batch_impl(payload, true)
}

fn decode_event_batch_impl(
    payload: &[u8],
    log_event_errors: bool,
) -> Result<EventActions, BridgeError> {
    let mut cursor = Cursor::new(payload);
    let value = read_value(&mut cursor).map_err(|error| BridgeError::Decode(error.to_string()))?;
    let batch = expect_array(&value, "KVEventBatch")?;
    if batch.len() < 2 {
        return Err(BridgeError::Decode(
            "KVEventBatch must contain timestamp and events".to_string(),
        ));
    }

    let events = expect_array(&batch[1], "KVEventBatch.events")?;
    let mut actions = EventActions::default();
    for (event_index, event) in events.iter().enumerate() {
        if let Err(error) = decode_event(event, &mut actions) {
            if log_event_errors {
                warn!(
                    event_index,
                    %error,
                    "skipping one undecodable SGLang KV event; preserving valid siblings"
                );
            }
        }
    }
    Ok(actions)
}

fn decode_event(event: &Value, actions: &mut EventActions) -> Result<(), BridgeError> {
    let event = expect_array(event, "KV event")?;
    let event_type = expect_str(
        event
            .first()
            .ok_or_else(|| BridgeError::Decode("KV event is empty".to_string()))?,
        "KV event tag",
    )?;

    match event_type {
        "BlockStored" => {
            // At least 7 fields (the legacy schema); an 8th `component_types`
            // slot appears with `--enable-kv-events-component-types`. Both
            // shapes are accepted.
            if event.len() < 7 {
                return Err(BridgeError::Decode(
                    "BlockStored must have at least 7 array fields".to_string(),
                ));
            }
            let tier = medium_to_tier(expect_optional_str(&event[6], "BlockStored.medium")?)?;
            // `component_types` is the trailing slot: a list of component labels
            // folded into a bitmask, or nil/absent for a legacy whole-block store.
            let mask = match event.get(7) {
                Some(value) => decode_component_mask(value)?,
                None => None,
            };
            // The token count is only carried alongside component-aware stores,
            // where the query path needs it to accumulate trailing windows.
            let block_size = match mask {
                Some(_) => Some(decode_block_size(&event[4])?),
                None => None,
            };
            actions.report(tier, decode_hashes(&event[1])?, mask, block_size);
        }
        "BlockRemoved" => {
            if event.len() < 3 {
                return Err(BridgeError::Decode(
                    "BlockRemoved must have 3 array fields".to_string(),
                ));
            }
            let tier = medium_to_tier(expect_optional_str(&event[2], "BlockRemoved.medium")?)?;
            actions.revoke(tier, decode_hashes(&event[1])?);
        }
        "AllBlocksCleared" => {
            actions.clear_all();
        }
        other => {
            debug!(event_type = other, "ignoring unsupported SGLang KV event");
        }
    }
    Ok(())
}

fn decode_hashes(value: &Value) -> Result<Vec<i64>, BridgeError> {
    expect_array(value, "block_hashes")?
        .iter()
        .map(|value| {
            if let Some(value) = value.as_i64() {
                return Ok(value);
            }
            // SGLang folds the unsigned top 64 bits of the SHA-256 into the
            // signed range by subtracting 2^64 (`hash_str_to_int64`), which is
            // two's complement, so a producer that serialises the unsigned half
            // instead is carrying identical bits. Reinterpreting recovers the
            // hash the router queries for; refusing the value would instead skip
            // the whole event and lose every placement it carried.
            if let Some(value) = value.as_u64() {
                return Ok(value as i64);
            }
            Err(BridgeError::Decode(
                "block hash must be an integer".to_string(),
            ))
        })
        .collect()
}

/// Decodes the optional `component_types` slot of a `BlockStored` into a component
/// bitmask. `nil` maps to `None`, a legacy whole-block store; an array of labels
/// folds into a bitmask, and labels this build does not model are ignored.
fn decode_component_mask(value: &Value) -> Result<Option<u32>, BridgeError> {
    if matches!(value, Value::Nil) {
        return Ok(None);
    }
    let mut mask = 0u32;
    for item in expect_array(value, "BlockStored.component_types")? {
        let name = item
            .as_str()
            .ok_or_else(|| BridgeError::Decode("component type must be a string".to_string()))?;
        if let Some(bit) = component_bit(name) {
            mask |= bit;
        }
    }
    Ok(Some(mask))
}

/// Decodes the `block_size` (token count) slot of a `BlockStored`.
fn decode_block_size(value: &Value) -> Result<u32, BridgeError> {
    let raw = value
        .as_u64()
        .or_else(|| value.as_i64().and_then(|v| u64::try_from(v).ok()))
        .ok_or_else(|| {
            BridgeError::Decode("block_size must be a non-negative integer".to_string())
        })?;
    u32::try_from(raw).map_err(|_| BridgeError::Decode("block_size exceeds u32".to_string()))
}

fn medium_to_tier(medium: Option<&str>) -> Result<i32, BridgeError> {
    match medium {
        Some("GPU") => Ok(TierType::TierHbm as i32),
        Some("CPU_PINNED") => Ok(TierType::TierDram as i32),
        Some("DISK") => Ok(TierType::TierSsd as i32),
        Some("EXTERNAL") => Err(BridgeError::Decode(
            "EXTERNAL medium does not map to a local indexer tier".to_string(),
        )),
        Some(other) => Err(BridgeError::Decode(format!(
            "unsupported SGLang storage medium: {other}"
        ))),
        None => Err(BridgeError::Decode(
            "SGLang storage medium is missing".to_string(),
        )),
    }
}

/// Builds the worker's [`WorkerCacheSpec`] from the environment, `None` for a
/// legacy / full-only worker. Rules are fixed, so the config only declares which
/// components are present, the SWA window, and their servable tiers:
///
/// ```text
///   KV_INDEXER_CACHE_COMPONENTS = full,swa      (present components; FULL implied)
///   KV_INDEXER_SWA_WINDOW_TOKENS = 4096         (required when swa present)
///   KV_INDEXER_FULL_TIERS  = HBM,DRAM           (optional, default HBM,DRAM)
///   KV_INDEXER_SWA_TIERS   = HBM
///   KV_INDEXER_MAMBA_TIERS = HBM,DRAM
///   KV_INDEXER_CACHE_SPEC_VERSION = 1           (optional, default 1)
/// ```
fn cache_spec_from_env() -> Result<Option<WorkerCacheSpec>, BridgeError> {
    let Some(list) = env_nonempty("KV_INDEXER_CACHE_COMPONENTS") else {
        return Ok(None);
    };
    let mut components = 0u32;
    for name in list.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let bit = component_bit(name)
            .ok_or_else(|| BridgeError::Config(format!("unknown cache component: {name}")))?;
        components |= bit;
    }
    // FULL is the base component and always present on a stored block.
    components |= crate::service::COMPONENT_FULL;

    let swa_window_tokens = match env_nonempty("KV_INDEXER_SWA_WINDOW_TOKENS") {
        Some(v) => v.parse::<u32>().map_err(|_| {
            BridgeError::Config(format!("KV_INDEXER_SWA_WINDOW_TOKENS is not a u32: {v}"))
        })?,
        None => 0,
    };
    if components & COMPONENT_SWA != 0 && swa_window_tokens == 0 {
        return Err(BridgeError::Config(
            "swa component requires KV_INDEXER_SWA_WINDOW_TOKENS".to_string(),
        ));
    }

    let version = match env_nonempty("KV_INDEXER_CACHE_SPEC_VERSION") {
        Some(v) => v
            .parse::<u32>()
            .map_err(|_| BridgeError::Config(format!("cache spec version is not a u32: {v}")))?,
        None => 1,
    };

    Ok(Some(WorkerCacheSpec {
        version,
        components,
        swa_window_tokens,
        full_tier_mask: env_tier_mask("KV_INDEXER_FULL_TIERS")?,
        swa_tier_mask: env_tier_mask("KV_INDEXER_SWA_TIERS")?,
        mamba_tier_mask: env_tier_mask("KV_INDEXER_MAMBA_TIERS")?,
    }))
}

fn env_nonempty(key: &str) -> Option<String> {
    std::env::var(key)
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

/// Parses a `KV_INDEXER_*_TIERS` list into a `1 << TierType` bitmask, defaulting
/// to HBM+DRAM when unset.
fn env_tier_mask(key: &str) -> Result<u32, BridgeError> {
    let list = env_nonempty(key).unwrap_or_else(|| "HBM,DRAM".to_string());
    let mut mask = 0u32;
    for tier in list.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        mask |= 1u32 << tier_name_to_type(tier)?;
    }
    Ok(mask)
}

fn tier_name_to_type(name: &str) -> Result<i32, BridgeError> {
    match name {
        "HBM" | "GPU" => Ok(TierType::TierHbm as i32),
        "DRAM" | "CPU" | "CPU_PINNED" => Ok(TierType::TierDram as i32),
        "SSD" | "DISK" => Ok(TierType::TierSsd as i32),
        other => Err(BridgeError::Config(format!(
            "cache spec has unsupported tier: {other}"
        ))),
    }
}

fn parse_clear_tiers(value: &str) -> Result<Vec<i32>, BridgeError> {
    value
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|part| match part {
            "HBM" | "GPU" => Ok(TierType::TierHbm as i32),
            "DRAM" | "CPU" | "CPU_PINNED" => Ok(TierType::TierDram as i32),
            "SSD" | "DISK" => Ok(TierType::TierSsd as i32),
            other => Err(BridgeError::Config(format!(
                "unsupported clear tier: {other}"
            ))),
        })
        .collect()
}

fn expect_array<'a>(value: &'a Value, field: &str) -> Result<&'a [Value], BridgeError> {
    value
        .as_array()
        .map(Vec::as_slice)
        .ok_or_else(|| BridgeError::Decode(format!("{field} must be an array")))
}

fn expect_str<'a>(value: &'a Value, field: &str) -> Result<&'a str, BridgeError> {
    value
        .as_str()
        .ok_or_else(|| BridgeError::Decode(format!("{field} must be a string")))
}

fn expect_optional_str<'a>(value: &'a Value, field: &str) -> Result<Option<&'a str>, BridgeError> {
    if matches!(value, Value::Nil) {
        return Ok(None);
    }
    expect_str(value, field).map(Some)
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;
    use rmpv::Value;

    fn hbm() -> i32 {
        TierType::TierHbm as i32
    }
    fn dram() -> i32 {
        TierType::TierDram as i32
    }
    fn ssd() -> i32 {
        TierType::TierSsd as i32
    }

    fn encode(value: &Value) -> Vec<u8> {
        let mut buf = Vec::new();
        rmpv::encode::write_value(&mut buf, value).unwrap();
        buf
    }

    fn ints(values: &[i64]) -> Value {
        Value::Array(values.iter().map(|v| Value::from(*v)).collect())
    }

    fn stored(hashes: &[i64], medium: &str) -> Value {
        Value::Array(vec![
            Value::String("BlockStored".into()),
            ints(hashes),
            Value::Nil,         // parent_block_hash
            ints(&[1]),         // token_ids
            Value::from(1_i64), // block_size
            Value::Nil,         // lora_id
            Value::String(medium.into()),
        ])
    }

    /// A component-aware `BlockStored` (8-element schema): trailing
    /// `component_types` slot plus a concrete `block_size` token count.
    fn stored_c(hashes: &[i64], medium: &str, block_size: i64, components: Value) -> Value {
        Value::Array(vec![
            Value::String("BlockStored".into()),
            ints(hashes),
            Value::Nil, // parent_block_hash
            ints(&[1]), // token_ids
            Value::from(block_size),
            Value::Nil, // lora_id
            Value::String(medium.into()),
            components, // component_types (Nil or array of strings)
        ])
    }

    fn strv(items: &[&str]) -> Value {
        Value::Array(items.iter().map(|s| Value::String((*s).into())).collect())
    }

    /// Legacy (whole-block) report action expectation.
    fn rep(tier: i32, hashes: &[&str]) -> Action {
        Action::Report {
            tier,
            hashes: hashes.iter().map(|h| h.parse().unwrap()).collect(),
            masks: vec![None; hashes.len()],
            block_sizes: vec![None; hashes.len()],
        }
    }

    fn rev(tier: i32, hashes: &[&str]) -> Action {
        Action::Revoke {
            tier,
            hashes: hashes.iter().map(|h| h.parse().unwrap()).collect(),
        }
    }

    fn removed(hashes: &[i64], medium: &str) -> Value {
        Value::Array(vec![
            Value::String("BlockRemoved".into()),
            ints(hashes),
            Value::String(medium.into()),
        ])
    }

    fn cleared() -> Value {
        Value::Array(vec![Value::String("AllBlocksCleared".into())])
    }

    /// Wrap events in a 3-element batch [ts, events, attn_dp_rank].
    fn batch(events: Vec<Value>) -> Vec<u8> {
        encode(&Value::Array(vec![
            Value::from(1.0_f64),
            Value::Array(events),
            Value::from(0_i64),
        ]))
    }

    fn actions_of(events: Vec<Value>) -> Vec<Action> {
        decode_event_batch(&batch(events)).unwrap().actions
    }

    fn golden_bytes(hex: &str) -> Vec<u8> {
        assert_eq!(hex.len() % 2, 0);
        hex.as_bytes()
            .chunks_exact(2)
            .map(|pair| {
                let high = (pair[0] as char).to_digit(16).unwrap();
                let low = (pair[1] as char).to_digit(16).unwrap();
                ((high << 4) | low) as u8
            })
            .collect()
    }

    fn test_config(clear_tiers: Vec<i32>) -> BridgeConfig {
        BridgeConfig {
            worker_id: "worker-1".to_string(),
            worker_address: "127.0.0.1:9000".to_string(),
            event_endpoint: "tcp://127.0.0.1:5557".to_string(),
            event_topic: "kv-events".to_string(),
            indexer_endpoint: "http://[::1]:50051".to_string(),
            clear_tiers,
            cache_spec: None,
        }
    }

    /// Build the apply-batch request the bridge would send for a set of events.
    fn request_of(
        config: &BridgeConfig,
        seq: u64,
        events: Vec<Value>,
    ) -> ApplyExternalKvBatchRequest {
        build_apply_request(config, seq, decode_event_batch(&batch(events)).unwrap())
    }

    fn report(tier: i32, hashes: &[&str]) -> ExternalKvAction {
        ExternalKvAction {
            r#type: ExternalKvActionType::ActionReport as i32,
            tier,
            hashes: hashes.iter().map(|h| h.parse().unwrap()).collect(),
            component_masks: Vec::new(),
            block_sizes: Vec::new(),
        }
    }

    fn revoke(tier: i32, hashes: &[&str]) -> ExternalKvAction {
        ExternalKvAction {
            r#type: ExternalKvActionType::ActionRevoke as i32,
            tier,
            hashes: hashes.iter().map(|h| h.parse().unwrap()).collect(),
            component_masks: Vec::new(),
            block_sizes: Vec::new(),
        }
    }

    fn clear_at(tier: i32) -> ExternalKvAction {
        ExternalKvAction {
            r#type: ExternalKvActionType::ActionClearAllAtTier as i32,
            tier,
            hashes: Vec::new(),
            component_masks: Vec::new(),
            block_sizes: Vec::new(),
        }
    }

    #[test]
    fn request_carries_worker_id_and_seq() {
        let config = test_config(vec![hbm()]);
        let request = request_of(&config, 42, vec![stored(&[1], "GPU")]);
        assert_eq!(request.worker_id, "worker-1");
        assert_eq!(request.seq, 42);
    }

    #[test]
    fn request_carries_worker_address() {
        let config = test_config(vec![hbm()]);
        let request = request_of(&config, 0, vec![stored(&[1], "GPU")]);
        assert_eq!(request.worker_address, "127.0.0.1:9000");
    }

    #[test]
    fn oversized_report_is_split_with_aligned_metadata() {
        let count = MAX_HASHES_PER_REQUEST + 1;
        let request = ApplyExternalKvBatchRequest {
            worker_id: "worker-1".into(),
            seq: 42,
            actions: vec![ExternalKvAction {
                r#type: ExternalKvActionType::ActionReport as i32,
                tier: hbm(),
                hashes: (0..count).map(|index| index as i64).collect(),
                component_masks: (0..count as u32).collect(),
                block_sizes: (0..count as u32).map(|index| index + 1).collect(),
            }],
            worker_address: "http://worker-1".into(),
            cache_spec: None,
        };

        let batches = split_apply_request(request);

        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].actions[0].hashes.len(), MAX_HASHES_PER_REQUEST);
        assert_eq!(
            batches[1].actions[0].hashes,
            vec![MAX_HASHES_PER_REQUEST as i64]
        );
        assert_eq!(
            batches[1].actions[0].component_masks,
            vec![MAX_HASHES_PER_REQUEST as u32]
        );
        assert_eq!(
            batches[1].actions[0].block_sizes,
            vec![MAX_HASHES_PER_REQUEST as u32 + 1]
        );
        assert!(batches.iter().all(|batch| batch.seq == 42));
    }

    #[test]
    fn too_many_clear_actions_are_split_in_order() {
        let request = ApplyExternalKvBatchRequest {
            worker_id: "worker-1".into(),
            seq: 7,
            actions: (0..=MAX_ACTIONS_PER_BATCH)
                .map(|index| clear_at(if index % 2 == 0 { hbm() } else { dram() }))
                .collect(),
            worker_address: "http://worker-1".into(),
            cache_spec: None,
        };

        let batches = split_apply_request(request);

        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].actions.len(), MAX_ACTIONS_PER_BATCH);
        assert_eq!(batches[1].actions, vec![clear_at(hbm())]);
    }

    #[test]
    fn report_and_revoke_map_to_actions_in_order() {
        let config = test_config(vec![hbm()]);
        let request = request_of(
            &config,
            0,
            vec![removed(&[9], "GPU"), stored(&[9], "CPU_PINNED")],
        );
        assert_eq!(
            request.actions,
            vec![revoke(hbm(), &["9"]), report(dram(), &["9"])]
        );
    }

    #[test]
    fn clear_all_expands_to_one_action_per_clear_tier_in_place() {
        let config = test_config(vec![hbm(), dram(), ssd()]);
        let request = request_of(
            &config,
            7,
            vec![stored(&[1], "GPU"), cleared(), stored(&[2], "GPU")],
        );
        assert_eq!(
            request.actions,
            vec![
                report(hbm(), &["1"]),
                clear_at(hbm()),
                clear_at(dram()),
                clear_at(ssd()),
                report(hbm(), &["2"]),
            ]
        );
    }

    #[test]
    fn batch_with_only_ignored_events_has_no_actions() {
        let config = test_config(vec![hbm()]);
        let events = vec![Value::Array(vec![Value::String("BlockUpdated".into())])];
        assert!(request_of(&config, 0, events).actions.is_empty());
    }

    #[test]
    fn block_stored_maps_to_report_on_tier() {
        assert_eq!(
            actions_of(vec![stored(&[123], "GPU")]),
            vec![rep(hbm(), &["123"])]
        );
    }

    #[test]
    fn mediums_map_to_expected_tiers() {
        assert_eq!(
            actions_of(vec![stored(&[1], "CPU_PINNED")]),
            vec![rep(dram(), &["1"])]
        );
        assert_eq!(
            actions_of(vec![removed(&[2], "DISK")]),
            vec![rev(ssd(), &["2"])]
        );
    }

    #[test]
    fn bad_event_does_not_drop_valid_siblings() {
        assert_eq!(
            actions_of(vec![
                stored(&[1], "GPU"),
                stored(&[2], "EXTERNAL"),
                removed(&[3], "DISK"),
            ]),
            vec![rep(hbm(), &["1"]), rev(ssd(), &["3"])]
        );
    }

    #[test]
    fn permanent_rpc_codes_are_not_retried() {
        for code in [
            Code::InvalidArgument,
            Code::FailedPrecondition,
            Code::PermissionDenied,
        ] {
            assert!(classify_rpc(Status::new(code, "bad batch")).is_permanent());
        }
        assert!(!classify_rpc(Status::unavailable("retry")).is_permanent());
        assert!(!classify_rpc(Status::deadline_exceeded("retry")).is_permanent());
    }

    /// Indexer backpressure must not take the bridge down: the router treats the
    /// same code as recoverable, and a rejected batch is worth less than the
    /// entire event stream.
    #[test]
    fn shed_batches_keep_the_bridge_alive() {
        assert!(!classify_rpc(Status::resource_exhausted("batch too large")).is_permanent());
    }

    // --- ordering regressions ---

    #[test]
    fn remove_then_store_same_hash_keeps_order() {
        // Net state must be "stored"; reordering to report-then-revoke would drop it.
        assert_eq!(
            actions_of(vec![removed(&[9], "GPU"), stored(&[9], "GPU")]),
            vec![rev(hbm(), &["9"]), rep(hbm(), &["9"])]
        );
    }

    #[test]
    fn clear_then_store_keeps_order() {
        assert_eq!(
            actions_of(vec![cleared(), stored(&[7], "GPU")]),
            vec![Action::ClearAll, rep(hbm(), &["7"])]
        );
    }

    #[test]
    fn store_then_clear_keeps_order() {
        assert_eq!(
            actions_of(vec![stored(&[7], "GPU"), cleared()]),
            vec![rep(hbm(), &["7"]), Action::ClearAll]
        );
    }

    // --- coalescing rules ---

    #[test]
    fn adjacent_same_tier_stores_coalesce() {
        assert_eq!(
            actions_of(vec![stored(&[1], "GPU"), stored(&[2], "GPU")]),
            vec![rep(hbm(), &["1", "2"])]
        );
    }

    #[test]
    fn different_tier_stores_do_not_coalesce() {
        assert_eq!(
            actions_of(vec![stored(&[1], "GPU"), stored(&[2], "CPU_PINNED")]),
            vec![rep(hbm(), &["1"]), rep(dram(), &["2"])]
        );
    }

    #[test]
    fn store_then_remove_same_tier_do_not_merge() {
        assert_eq!(
            actions_of(vec![stored(&[1], "GPU"), removed(&[1], "GPU")]),
            vec![rep(hbm(), &["1"]), rev(hbm(), &["1"])]
        );
    }

    #[test]
    fn unknown_event_tag_is_ignored() {
        let events = vec![Value::Array(vec![Value::String("BlockUpdated".into())])];
        assert!(actions_of(events).is_empty());
    }

    #[test]
    fn two_element_batch_without_dp_rank_decodes() {
        let payload = encode(&Value::Array(vec![
            Value::from(1.0_f64),
            Value::Array(vec![stored(&[5], "GPU")]),
        ]));
        assert_eq!(
            decode_event_batch(&payload).unwrap().actions,
            vec![rep(hbm(), &["5"])]
        );
    }

    #[test]
    fn python_msgspec_mixed_batch_golden_decodes() {
        // Generated by msgspec.msgpack.Encoder from the authoritative Python
        // KVEventBatch schema in sglang.srt.disaggregation.kv_events.
        let payload = golden_bytes(concat!(
            "93cb405edd2f1a9fbe779397ab426c6f636b53746f72656492",
            "cf0000011f71fb04cbd2c521974f2a940a141e280407a3475055",
            "93ac426c6f636b52656d6f7665649264ccc8a44449534b",
            "91b0416c6c426c6f636b73436c656172656402"
        ));
        assert_eq!(
            decode_event_batch(&payload).unwrap().actions,
            vec![
                rep(hbm(), &["1234567890123", "-987654321"]),
                rev(ssd(), &["100", "200"]),
                Action::ClearAll,
            ]
        );
    }

    #[test]
    fn python_msgspec_bigram_tokens_golden_decodes() {
        // token_ids contains Python tuples as nested msgpack arrays; the
        // bridge ignores payload shape and indexes the published hashes.
        let payload = golden_bytes(concat!(
            "93cb3ff80000000000009197ab426c6f636b53746f726564916f",
            "c092920a1492141e02c0a347505503"
        ));
        assert_eq!(
            decode_event_batch(&payload).unwrap().actions,
            vec![rep(hbm(), &["111"])]
        );
    }

    #[test]
    fn python_msgspec_nil_medium_golden_is_safely_skipped() {
        // The Python schema permits medium=None; such events map to no
        // Indexer tier, so they are isolated rather than given a placement.
        let payload = golden_bytes(concat!(
            "93cb00000000000000009297ab426c6f636b53746f7265649101",
            "c092050602c0c093ac426c6f636b52656d6f7665649102c0c0"
        ));
        assert!(decode_event_batch(&payload).unwrap().actions.is_empty());
    }

    #[test]
    fn negative_hashes_remain_signed_integers() {
        assert_eq!(
            actions_of(vec![stored(&[-1905904552702706914], "GPU")]),
            vec![rep(hbm(), &["-1905904552702706914"])]
        );
    }

    // --- error / mapping units ---

    #[test]
    fn external_medium_event_is_skipped() {
        assert!(decode_event_batch(&batch(vec![stored(&[1], "EXTERNAL")]))
            .unwrap()
            .actions
            .is_empty());
    }

    #[test]
    fn unknown_medium_event_is_skipped() {
        assert!(decode_event_batch(&batch(vec![stored(&[1], "TAPE")]))
            .unwrap()
            .actions
            .is_empty());
    }

    #[test]
    fn medium_to_tier_mapping() {
        assert_eq!(medium_to_tier(Some("GPU")).unwrap(), hbm());
        assert_eq!(medium_to_tier(Some("CPU_PINNED")).unwrap(), dram());
        assert_eq!(medium_to_tier(Some("DISK")).unwrap(), ssd());
        assert!(medium_to_tier(Some("EXTERNAL")).is_err());
        assert!(medium_to_tier(None).is_err());
    }

    #[test]
    fn parse_clear_tiers_defaults_and_aliases() {
        assert_eq!(
            parse_clear_tiers("HBM,DRAM,SSD").unwrap(),
            vec![hbm(), dram(), ssd()]
        );
        assert_eq!(
            parse_clear_tiers(" GPU , CPU_PINNED , DISK ").unwrap(),
            vec![hbm(), dram(), ssd()]
        );
        assert!(parse_clear_tiers("HBM,NVME").is_err());
    }

    /// An event that serialises a hash as unsigned must decode to the value the
    /// router queries for. Anything else would file the block under a hash no
    /// query can reach, which reads as a silent cache miss rather than an error.
    #[test]
    fn decode_hashes_reinterprets_unsigned_as_the_same_bits() {
        let value = Value::Array(vec![
            Value::from(1_i64),
            Value::from(-2_i64),
            Value::from(i64::MAX as u64),
            Value::from(u64::MAX),
            Value::from(1_u64 << 63),
        ]);
        assert_eq!(
            decode_hashes(&value).unwrap(),
            vec![1, -2, i64::MAX, -1, i64::MIN]
        );
        assert!(decode_hashes(&Value::Array(vec![Value::String("x".into())])).is_err());
    }

    // --- frame / sequence parsing ---

    #[test]
    fn parse_zmq_frames_two_and_three() {
        let seq = 42_u64;
        let two = [
            Bytes::copy_from_slice(&seq.to_be_bytes()),
            Bytes::from_static(b"p"),
        ];
        assert_eq!(parse_zmq_frames(&two).unwrap().0, seq);
        let three = [
            Bytes::from_static(b"kv-events"),
            Bytes::copy_from_slice(&seq.to_be_bytes()),
            Bytes::from_static(b"p"),
        ];
        assert_eq!(parse_zmq_frames(&three).unwrap().0, seq);
        let one = [Bytes::from_static(b"p")];
        assert!(parse_zmq_frames(&one).is_err());
    }

    #[test]
    fn seq_decoders_are_big_endian() {
        assert_eq!(decode_seq(&5_u64.to_be_bytes()).unwrap(), 5);
        assert!(decode_seq(&[0_u8; 4]).is_err());
    }

    // --- component-aware decoding ---

    #[test]
    fn component_types_list_decodes_into_report() {
        assert_eq!(
            actions_of(vec![stored_c(&[1], "GPU", 64, strv(&["full", "swa"]))]),
            vec![Action::Report {
                tier: hbm(),
                hashes: vec![1],
                masks: vec![Some(
                    crate::service::COMPONENT_FULL | crate::service::COMPONENT_SWA
                )],
                block_sizes: vec![Some(64)],
            }]
        );
    }

    #[test]
    fn component_types_nil_decodes_as_legacy() {
        // An 8-element BlockStored whose trailing slot is nil is exactly the
        // legacy whole-block store: no components, no size.
        assert_eq!(
            actions_of(vec![stored_c(&[1], "GPU", 64, Value::Nil)]),
            vec![rep(hbm(), &["1"])]
        );
    }

    #[test]
    fn component_aware_report_carries_aligned_wire_arrays() {
        let config = test_config(vec![hbm()]);
        let request = request_of(
            &config,
            0,
            vec![
                stored_c(&[1], "GPU", 64, strv(&["full", "swa"])),
                stored_c(&[2], "GPU", 32, strv(&["full"])),
            ],
        );
        assert_eq!(request.actions.len(), 1);
        let action = &request.actions[0];
        assert_eq!(action.hashes, vec![1, 2]);
        assert_eq!(
            action.component_masks,
            vec![
                crate::service::COMPONENT_FULL | crate::service::COMPONENT_SWA,
                crate::service::COMPONENT_FULL,
            ]
        );
        assert_eq!(action.block_sizes, vec![64, 32]);
    }

    #[test]
    fn cache_spec_forwarded_on_request() {
        let mut config = test_config(vec![hbm()]);
        config.cache_spec = Some(WorkerCacheSpec {
            version: 1,
            components: crate::service::COMPONENT_FULL,
            swa_window_tokens: 0,
            full_tier_mask: 1 << hbm(),
            swa_tier_mask: 0,
            mamba_tier_mask: 0,
        });
        let request = request_of(&config, 0, vec![stored(&[1], "GPU")]);
        assert_eq!(request.cache_spec, config.cache_spec);
    }

    // --- cache spec config helpers ---

    #[test]
    fn config_helpers_map_tiers_and_components() {
        assert_eq!(tier_name_to_type("HBM").unwrap(), hbm());
        assert_eq!(tier_name_to_type("CPU_PINNED").unwrap(), dram());
        assert!(tier_name_to_type("NVME").is_err());
        assert_eq!(component_bit("full"), Some(crate::service::COMPONENT_FULL));
        assert_eq!(component_bit("swa"), Some(COMPONENT_SWA));
        assert_eq!(component_bit("bogus"), None);
    }
}
