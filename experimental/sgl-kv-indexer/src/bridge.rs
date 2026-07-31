// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! SGLang KV event bridge.
//!
//! Subscribes to a worker's ZMQ KV-event stream, decodes each batch, and
//! forwards it to the indexer over gRPC.
//!
//! This is the basic build. It keeps a reconnect supervisor so the bridge can be
//! started before SGLang, but it does not recover data: there is no sequence
//! tracking, no replay of missed batches, no incarnation token, and no liveness
//! heartbeat. A gap in the publisher's sequence is logged and otherwise ignored,
//! and events produced while the bridge is disconnected are lost.

use std::io::Cursor;
use std::time::Duration;

use rmpv::decode::value::read_value;
use rmpv::Value;
use tonic::transport::{Channel, Endpoint};
use tonic::{Code, Status};
use tracing::{debug, info, warn};
use zeromq::{Socket, SocketRecv, SubSocket};

use crate::pb::kv_indexer_client::KvIndexerClient;
use crate::pb::{ApplyExternalKvBatchRequest, ExternalKvAction, ExternalKvActionType, TierType};

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

        Ok(Self {
            worker_id,
            worker_address,
            event_endpoint,
            event_topic,
            indexer_endpoint,
            clear_tiers,
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
        | Code::ResourceExhausted
        | Code::Unauthenticated
        | Code::PermissionDenied
        | Code::Unimplemented
        | Code::DataLoss => BridgeError::PermanentRpc(status),
        _ => BridgeError::Rpc(status),
    }
}

/// A single indexer mutation, kept in the exact order it appeared in the event
/// batch so that store/remove/clear operations on the same hash are never
/// reordered relative to each other.
#[derive(Debug, PartialEq, Eq)]
enum Action {
    Report { tier: i32, hashes: Vec<String> },
    Revoke { tier: i32, hashes: Vec<String> },
    ClearAll,
}

#[derive(Debug, Default)]
struct EventActions {
    actions: Vec<Action>,
}

impl EventActions {
    /// Append a store, coalescing only with an immediately-preceding store to
    /// the same tier. Coalescing never crosses a revoke/clear, so ordering (and
    /// therefore the final per-hash state) is preserved.
    fn report(&mut self, tier: i32, hashes: Vec<String>) {
        if hashes.is_empty() {
            return;
        }
        if let Some(Action::Report {
            tier: last_tier,
            hashes: last,
        }) = self.actions.last_mut()
        {
            if *last_tier == tier {
                last.extend(hashes);
                return;
            }
        }
        self.actions.push(Action::Report { tier, hashes });
    }

    fn revoke(&mut self, tier: i32, hashes: Vec<String>) {
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

/// [`run_bridge`], but returns as soon as `shutdown` resolves.
///
/// An in-flight apply is simply dropped. This build does not track which batches
/// the indexer acknowledged, so a batch interrupted by shutdown is lost.
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
    //
    // Reconnecting exists so the bridge can be started before SGLang and survive
    // an indexer restart during joint debugging. It recovers the connection only:
    // events published while disconnected are lost.
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
    if request.actions.is_empty() {
        return Ok(());
    }
    client
        .apply_external_kv_batch(request)
        .await
        .map_err(classify_rpc)?;
    Ok(())
}

/// Maps a decoded `EventActions` into a single `ApplyExternalKvBatchRequest`,
/// preserving the exact per-action order. A `ClearAll` is expanded, in place,
/// into one `CLEAR_ALL_AT_TIER` action per configured clear tier so the batch
/// carries the same semantics as the legacy per-tier revoke-all RPCs.
fn build_apply_request(
    config: &BridgeConfig,
    seq: u64,
    events: EventActions,
) -> ApplyExternalKvBatchRequest {
    let mut actions = Vec::with_capacity(events.actions.len());
    for action in events.actions {
        match action {
            Action::Report { tier, hashes } => actions.push(ExternalKvAction {
                r#type: ExternalKvActionType::ActionReport as i32,
                tier,
                hashes,
            }),
            Action::Revoke { tier, hashes } => actions.push(ExternalKvAction {
                r#type: ExternalKvActionType::ActionRevoke as i32,
                tier,
                hashes,
            }),
            Action::ClearAll => {
                for tier in &config.clear_tiers {
                    actions.push(ExternalKvAction {
                        r#type: ExternalKvActionType::ActionClearAllAtTier as i32,
                        tier: *tier,
                        hashes: Vec::new(),
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
    }
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
            if event.len() < 7 {
                return Err(BridgeError::Decode(
                    "BlockStored must have 7 array fields".to_string(),
                ));
            }
            let tier = medium_to_tier(expect_optional_str(&event[6], "BlockStored.medium")?)?;
            actions.report(tier, decode_hashes(&event[1])?);
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

fn decode_hashes(value: &Value) -> Result<Vec<String>, BridgeError> {
    expect_array(value, "block_hashes")?
        .iter()
        .map(|value| {
            if let Some(value) = value.as_i64() {
                return Ok(value.to_string());
            }
            if let Some(value) = value.as_u64() {
                return Ok(value.to_string());
            }
            Err(BridgeError::Decode(
                "block hash must be an integer".to_string(),
            ))
        })
        .collect()
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
            hashes: hashes.iter().map(|h| h.to_string()).collect(),
        }
    }

    fn revoke(tier: i32, hashes: &[&str]) -> ExternalKvAction {
        ExternalKvAction {
            r#type: ExternalKvActionType::ActionRevoke as i32,
            tier,
            hashes: hashes.iter().map(|h| h.to_string()).collect(),
        }
    }

    fn clear_at(tier: i32) -> ExternalKvAction {
        ExternalKvAction {
            r#type: ExternalKvActionType::ActionClearAllAtTier as i32,
            tier,
            hashes: Vec::new(),
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
            vec![Action::Report {
                tier: hbm(),
                hashes: vec!["123".to_string()],
            }]
        );
    }

    #[test]
    fn mediums_map_to_expected_tiers() {
        assert_eq!(
            actions_of(vec![stored(&[1], "CPU_PINNED")]),
            vec![Action::Report {
                tier: dram(),
                hashes: vec!["1".to_string()],
            }]
        );
        assert_eq!(
            actions_of(vec![removed(&[2], "DISK")]),
            vec![Action::Revoke {
                tier: ssd(),
                hashes: vec!["2".to_string()],
            }]
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
            vec![
                Action::Report {
                    tier: hbm(),
                    hashes: vec!["1".to_string()],
                },
                Action::Revoke {
                    tier: ssd(),
                    hashes: vec!["3".to_string()],
                },
            ]
        );
    }

    #[test]
    fn permanent_rpc_codes_are_not_retried() {
        for code in [
            Code::InvalidArgument,
            Code::FailedPrecondition,
            Code::ResourceExhausted,
            Code::PermissionDenied,
        ] {
            assert!(classify_rpc(Status::new(code, "bad batch")).is_permanent());
        }
        assert!(!classify_rpc(Status::unavailable("retry")).is_permanent());
        assert!(!classify_rpc(Status::deadline_exceeded("retry")).is_permanent());
    }

    // --- ordering regressions (the whole point of the in-order rewrite) ---

    #[test]
    fn remove_then_store_same_hash_keeps_order() {
        // Net state must be "stored"; reordering to report-then-revoke would drop it.
        assert_eq!(
            actions_of(vec![removed(&[9], "GPU"), stored(&[9], "GPU")]),
            vec![
                Action::Revoke {
                    tier: hbm(),
                    hashes: vec!["9".to_string()],
                },
                Action::Report {
                    tier: hbm(),
                    hashes: vec!["9".to_string()],
                },
            ]
        );
    }

    #[test]
    fn clear_then_store_keeps_order() {
        assert_eq!(
            actions_of(vec![cleared(), stored(&[7], "GPU")]),
            vec![
                Action::ClearAll,
                Action::Report {
                    tier: hbm(),
                    hashes: vec!["7".to_string()],
                },
            ]
        );
    }

    #[test]
    fn store_then_clear_keeps_order() {
        assert_eq!(
            actions_of(vec![stored(&[7], "GPU"), cleared()]),
            vec![
                Action::Report {
                    tier: hbm(),
                    hashes: vec!["7".to_string()],
                },
                Action::ClearAll,
            ]
        );
    }

    // --- coalescing rules ---

    #[test]
    fn adjacent_same_tier_stores_coalesce() {
        assert_eq!(
            actions_of(vec![stored(&[1], "GPU"), stored(&[2], "GPU")]),
            vec![Action::Report {
                tier: hbm(),
                hashes: vec!["1".to_string(), "2".to_string()],
            }]
        );
    }

    #[test]
    fn different_tier_stores_do_not_coalesce() {
        assert_eq!(
            actions_of(vec![stored(&[1], "GPU"), stored(&[2], "CPU_PINNED")]),
            vec![
                Action::Report {
                    tier: hbm(),
                    hashes: vec!["1".to_string()],
                },
                Action::Report {
                    tier: dram(),
                    hashes: vec!["2".to_string()],
                },
            ]
        );
    }

    #[test]
    fn store_then_remove_same_tier_do_not_merge() {
        assert_eq!(
            actions_of(vec![stored(&[1], "GPU"), removed(&[1], "GPU")]),
            vec![
                Action::Report {
                    tier: hbm(),
                    hashes: vec!["1".to_string()],
                },
                Action::Revoke {
                    tier: hbm(),
                    hashes: vec!["1".to_string()],
                },
            ]
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
            vec![Action::Report {
                tier: hbm(),
                hashes: vec!["5".to_string()],
            }]
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
                Action::Report {
                    tier: hbm(),
                    hashes: vec!["1234567890123".to_string(), "-987654321".to_string()],
                },
                Action::Revoke {
                    tier: ssd(),
                    hashes: vec!["100".to_string(), "200".to_string()],
                },
                Action::ClearAll,
            ]
        );
    }

    #[test]
    fn python_msgspec_bigram_tokens_golden_decodes() {
        // token_ids contains Python tuples, encoded as nested msgpack arrays.
        // The bridge intentionally ignores token payload shape and indexes the
        // publisher-provided block hashes.
        let payload = golden_bytes(concat!(
            "93cb3ff80000000000009197ab426c6f636b53746f726564916f",
            "c092920a1492141e02c0a347505503"
        ));
        assert_eq!(
            decode_event_batch(&payload).unwrap().actions,
            vec![Action::Report {
                tier: hbm(),
                hashes: vec!["111".to_string()],
            }]
        );
    }

    #[test]
    fn python_msgspec_nil_medium_golden_is_safely_skipped() {
        // The Python schema permits medium=None. Such events cannot be mapped
        // to an Indexer tier, so they are isolated rather than poisoning the
        // batch or inventing a placement.
        let payload = golden_bytes(concat!(
            "93cb00000000000000009297ab426c6f636b53746f7265649101",
            "c092050602c0c093ac426c6f636b52656d6f7665649102c0c0"
        ));
        assert!(decode_event_batch(&payload).unwrap().actions.is_empty());
    }

    #[test]
    fn negative_hashes_are_stringified() {
        assert_eq!(
            actions_of(vec![stored(&[-1905904552702706914], "GPU")]),
            vec![Action::Report {
                tier: hbm(),
                hashes: vec!["-1905904552702706914".to_string()],
            }]
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

    #[test]
    fn decode_hashes_accepts_signed_and_unsigned() {
        let value = Value::Array(vec![
            Value::from(1_i64),
            Value::from(-2_i64),
            Value::from(u64::MAX),
        ]);
        assert_eq!(
            decode_hashes(&value).unwrap(),
            vec!["1".to_string(), "-2".to_string(), u64::MAX.to_string()]
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
}
