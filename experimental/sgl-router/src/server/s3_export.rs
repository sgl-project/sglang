//! S3 token-dataset export: a second tee independent of the cache-sim that
//! writes ingest/extend token sequences as NDJSON+gzip batches to S3 for
//! offline cache-hit recomputation.

use std::collections::HashMap;
use std::io::Write;
use std::time::{Duration, Instant};

use serde::Serialize;

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub(crate) enum ExportKind {
    Ingest,
    Extend,
}

/// One export record (serialized as one NDJSON line). Optional fields are
/// omitted when absent; semantics align with `cache_sim_tee::IngestIdsBody`.
#[derive(Clone, Debug, Serialize)]
pub(crate) struct ExportRecord {
    pub kind: ExportKind,
    pub request_id: String,
    pub slug: String,
    pub model: String,
    pub input_ids: Vec<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_len: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub choice_index: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub choice_count: Option<usize>,
    /// The upstream gateway's key identifier, read from `x-radixark-key-id`
    /// (the same attribution the cache-sim tee carries — see
    /// `cache_sim_tee::Attribution::key_id`). Lets downstream partition export
    /// records by client key without a separate lookup. Absent for requests
    /// with no gateway-resolved key (shared-bearer or direct-to-router).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub key_id: Option<String>,
    pub ts: String,
}

impl ExportRecord {
    /// Serialize to one NDJSON line (with a trailing newline). Serialization
    /// cannot fail for pure value types; if it does, return an empty string
    /// and let the caller count it rather than panicking.
    pub(crate) fn to_ndjson_line(&self) -> String {
        match serde_json::to_string(self) {
            Ok(mut s) => {
                s.push('\n');
                s
            }
            Err(_) => String::new(),
        }
    }
}

/// Parse `s3://bucket/prefix/...` into `(bucket, key_prefix)`.
/// `key_prefix` is stripped of leading and trailing `/` (re-added when
/// constructing the key). Returns `None` for non-s3:// URIs.
pub fn parse_s3_uri(uri: &str) -> Option<(String, String)> {
    let rest = uri.trim().strip_prefix("s3://")?;
    let mut parts = rest.splitn(2, '/');
    let bucket = parts.next().filter(|b| !b.is_empty())?.to_string();
    let prefix = parts.next().unwrap_or("").trim_matches('/').to_string();
    Some((bucket, prefix))
}

/// Hive-style partitioned object key. `unix_nanos + seq` prevents collisions
/// within one pod; `pod` prevents collisions across replicas.
pub(crate) fn object_key(
    prefix: &str,
    slug: &str,
    date: &str,
    pod: &str,
    unix_nanos: u128,
    seq: u64,
) -> String {
    let head = if prefix.is_empty() {
        String::new()
    } else {
        format!("{prefix}/")
    };
    format!("{head}slug={slug}/date={date}/{pod}-{unix_nanos}-{seq}.ndjson.gz")
}

/// The slug is a client-suppliable header (x-radixark-endpoint-slug). Only a
/// strict allowlist may reach S3 key paths or the batcher map: hostile values
/// (path separators, huge/high-cardinality strings) collapse to "unknown".
fn safe_slug(slug: Option<&str>) -> String {
    match slug {
        Some(s)
            if !s.is_empty()
                && s.len() <= 64
                && s.bytes()
                    .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-')) =>
        {
            s.to_string()
        }
        _ => "unknown".to_string(),
    }
}

pub(crate) const DEFAULT_MAX_BATCH_BYTES: usize = 8 << 20;
pub(crate) const DEFAULT_MAX_BATCH_AGE: Duration = Duration::from_secs(10);

pub(crate) struct ReadyObject {
    pub slug: String,
    pub raw_bytes: Vec<u8>,
    /// NDJSON record count in this batch; added to
    /// `sgl_router_s3_export_records_uploaded_total` on successful upload.
    pub records: u64,
}

struct Buf {
    raw: Vec<u8>,
    records: u64,
    opened_at: Instant,
}

/// Per-slug accumulator of uncompressed NDJSON bytes. Triggers on size or age
/// and yields raw (uncompressed) bytes. Gzip compression is done by the upload
/// worker asynchronously.
pub(crate) struct SlugBatcher {
    max_bytes: usize,
    max_age: Duration,
    bufs: HashMap<String, Buf>,
}

impl SlugBatcher {
    pub(crate) fn new(max_bytes: usize, max_age: Duration) -> Self {
        Self {
            max_bytes,
            max_age,
            bufs: HashMap::new(),
        }
    }

    pub(crate) fn push(&mut self, slug: &str, line: &str, now: Instant) {
        let buf = self.bufs.entry(slug.to_string()).or_insert_with(|| Buf {
            raw: Vec::new(),
            records: 0,
            opened_at: now,
        });
        buf.raw.extend_from_slice(line.as_bytes());
        buf.records += 1;
    }

    /// Remove all "ready" slug buffers (over size, over age, or `force=true`)
    /// and return their raw bytes. Unready buffers are retained. The size
    /// trigger compares uncompressed bytes against `max_bytes` (spec §5.3);
    /// gzip compression is done by the upload worker, not here.
    pub(crate) fn take_ready(&mut self, now: Instant, force: bool) -> Vec<ReadyObject> {
        let ready_slugs: Vec<String> = self
            .bufs
            .iter()
            .filter(|(_, b)| {
                force
                    || b.raw.len() >= self.max_bytes
                    || now.duration_since(b.opened_at) >= self.max_age
            })
            .map(|(s, _)| s.clone())
            .collect();

        let mut out = Vec::with_capacity(ready_slugs.len());
        for slug in ready_slugs {
            let buf = self.bufs.remove(&slug).expect("just listed");
            if buf.raw.is_empty() {
                continue;
            }
            out.push(ReadyObject {
                slug,
                raw_bytes: buf.raw,
                records: buf.records,
            });
        }
        out
    }
}

use crate::server::metrics::MetricsRegistry;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;
use tokio::sync::Mutex as AsyncMutex;

/// In-memory store for tests. The first `fail_first` `put` calls return an
/// error; subsequent calls succeed and record the key+body.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) struct FakeStore {
    pub puts: Mutex<Vec<(String, Vec<u8>)>>,
    fail_first: AtomicU64,
}

#[cfg_attr(not(test), allow(dead_code))]
impl FakeStore {
    pub(crate) fn new(fail_first: u64) -> Self {
        Self {
            puts: Mutex::new(Vec::new()),
            fail_first: AtomicU64::new(fail_first),
        }
    }
}

// ---- SigV4 helpers (pure functions, unit-testable) ----

pub(crate) fn sha256_hex(data: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(data);
    hex::encode(h.finalize())
}

pub(crate) fn hmac_sha256(key: &[u8], msg: &[u8]) -> [u8; 32] {
    use hmac::{Hmac, Mac};
    type H = Hmac<sha2::Sha256>;
    let mut m = <H as Mac>::new_from_slice(key).expect("hmac accepts any key length");
    m.update(msg);
    m.finalize().into_bytes().into()
}

/// SigV4 signing key: HMAC chain AWS4+secret -> date -> region -> service -> aws4_request.
pub(crate) fn signing_key(secret: &str, date_stamp: &str, region: &str, service: &str) -> [u8; 32] {
    let k_date = hmac_sha256(format!("AWS4{secret}").as_bytes(), date_stamp.as_bytes());
    let k_region = hmac_sha256(&k_date, region.as_bytes());
    let k_service = hmac_sha256(&k_region, service.as_bytes());
    hmac_sha256(&k_service, b"aws4_request")
}

/// RFC3986 percent-encode. When `encode_slash=false`, `/` is left literal
/// (used for S3 canonical URIs).
pub(crate) fn uri_encode(s: &str, encode_slash: bool) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => {
                out.push(b as char)
            }
            b'/' if !encode_slash => out.push('/'),
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// We sign exactly three headers: host, x-amz-content-sha256, x-amz-date.
/// The query string is empty.
pub(crate) fn canonical_request(
    method: &str,
    canonical_uri: &str,
    host: &str,
    amz_date: &str,
    payload_hash: &str,
) -> String {
    format!(
        "{method}\n{canonical_uri}\n\nhost:{host}\nx-amz-content-sha256:{payload_hash}\nx-amz-date:{amz_date}\n\nhost;x-amz-content-sha256;x-amz-date\n{payload_hash}"
    )
}

pub(crate) fn string_to_sign(
    amz_date: &str,
    date_stamp: &str,
    region: &str,
    service: &str,
    creq_hash: &str,
) -> String {
    format!(
        "AWS4-HMAC-SHA256\n{amz_date}\n{date_stamp}/{region}/{service}/aws4_request\n{creq_hash}"
    )
}

pub(crate) enum Uploader {
    S3 {
        http: reqwest::Client,
        access_key: String,
        secret_key: String,
        region: String,
        bucket: String,
    },
    // Test-only variant; suppressed outside #[cfg(test)] builds.
    #[cfg_attr(not(test), allow(dead_code))]
    Fake(Arc<FakeStore>),
}

impl Uploader {
    async fn put(&self, key: &str, body: Vec<u8>) -> anyhow::Result<()> {
        match self {
            Uploader::S3 {
                http,
                access_key,
                secret_key,
                region,
                bucket,
            } => {
                // Virtual-hosted-style URL (bucket names without dots have no
                // TLS SNI issue).
                let host = format!("{bucket}.s3.{region}.amazonaws.com");
                let canonical_uri = format!("/{}", uri_encode(key, false));
                let url = format!("https://{host}{canonical_uri}");
                let now = chrono::Utc::now();
                let amz_date = now.format("%Y%m%dT%H%M%SZ").to_string();
                let date_stamp = now.format("%Y%m%d").to_string();
                let payload_hash = sha256_hex(&body);

                let creq =
                    canonical_request("PUT", &canonical_uri, &host, &amz_date, &payload_hash);
                let sts = string_to_sign(
                    &amz_date,
                    &date_stamp,
                    region,
                    "s3",
                    &sha256_hex(creq.as_bytes()),
                );
                let sig = hex::encode(hmac_sha256(
                    &signing_key(secret_key, &date_stamp, region, "s3"),
                    sts.as_bytes(),
                ));
                let authz = format!(
                    "AWS4-HMAC-SHA256 Credential={access_key}/{date_stamp}/{region}/s3/aws4_request, \
                     SignedHeaders=host;x-amz-content-sha256;x-amz-date, Signature={sig}"
                );

                let resp = http
                    .put(&url)
                    .header("x-amz-date", &amz_date)
                    .header("x-amz-content-sha256", &payload_hash)
                    .header("authorization", &authz)
                    .header("content-encoding", "gzip")
                    .header("content-type", "application/x-ndjson")
                    .body(body)
                    .send()
                    .await?;
                let status = resp.status();
                if status.is_success() {
                    Ok(())
                } else {
                    // Include S3's error body (XML: <Code>…</Code>) so a failed
                    // upload is diagnosable (SignatureDoesNotMatch / AccessDenied /
                    // InvalidAccessKeyId …) instead of a bare status.
                    let body = resp.text().await.unwrap_or_default();
                    anyhow::bail!("s3 put returned {status}: {body}");
                }
            }
            Uploader::Fake(store) => {
                if store.fail_first.load(Ordering::SeqCst) > 0 {
                    store.fail_first.fetch_sub(1, Ordering::SeqCst);
                    anyhow::bail!("injected failure");
                }
                store.puts.lock().unwrap().push((key.to_string(), body));
                Ok(())
            }
        }
    }
}

/// Exponential-backoff retry. Returns true on success within `max_attempts`,
/// false otherwise (the caller is responsible for counting). Backoff starts at
/// 100 ms, doubles on each failure, and is capped at 5 s.
pub(crate) async fn put_with_retry(
    up: &Uploader,
    key: &str,
    body: Vec<u8>,
    max_attempts: u32,
) -> bool {
    let mut backoff = Duration::from_millis(100);
    for attempt in 1..=max_attempts {
        match up.put(key, body.clone()).await {
            Ok(()) => return true,
            Err(e) => {
                tracing::warn!(key, attempt, error = %e, "s3 export put failed; will retry");
                if attempt == max_attempts {
                    return false;
                }
                tokio::time::sleep(backoff).await;
                backoff = (backoff * 2).min(Duration::from_secs(5));
            }
        }
    }
    false
}

/// Count-only channel bound. At long-context record sizes this caps worst-case
/// queue memory while a fast non-blocking pump keeps normal buffering ample.
pub(crate) const CHANNEL_CAPACITY: usize = 2048;
const PUT_MAX_ATTEMPTS: u32 = 8;
/// Periodic flush heartbeat for the pump, paired with the batcher's age trigger.
const TICK: Duration = Duration::from_secs(1);
/// Bounded pool of concurrent gzip+upload worker tasks. Keeps CPU usage bounded
/// on the shared 8-core container while allowing parallelism.
const UPLOAD_CONCURRENCY: usize = 4;
/// Bounds pending raw-batch memory (~32×8 MiB) and keeps the pump loop live
/// during an S3 outage — a best-effort tee sheds, never stalls its own
/// heartbeat.
const MAX_PENDING_UPLOADS: usize = 32;

enum PumpMsg {
    Record(Box<ExportRecord>),
    Drain(tokio::sync::oneshot::Sender<()>),
}

pub struct S3ExportSink {
    tx: mpsc::Sender<PumpMsg>,
    metrics: Arc<MetricsRegistry>,
    join: AsyncMutex<Option<tokio::task::JoinHandle<()>>>,
    /// Semaphore bounding the number of concurrent captures in S3-only mode.
    /// Mirrors `CacheSimTee::try_acquire_capture_permit` so S3-only deployments
    /// have the same aggregate-capture memory bound as cache-sim deployments.
    capture_sem: Arc<tokio::sync::Semaphore>,
}

impl std::fmt::Debug for S3ExportSink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("S3ExportSink").finish_non_exhaustive()
    }
}

impl S3ExportSink {
    /// Production entry point: parse `uri`, read credentials/region from the
    /// standard AWS environment variables, build an S3 uploader, and start the
    /// pump. Returns `None` if the URI is invalid, credentials are missing, or
    /// temporary (session-token) credentials are detected.
    pub fn spawn(
        uri: &str,
        pod: String,
        metrics: Arc<MetricsRegistry>,
        max_captures: usize,
    ) -> Option<Arc<Self>> {
        let (bucket, prefix) = parse_s3_uri(uri)?;
        // `.trim()`: a secretKeyRef-injected value can carry a trailing newline
        // (if the stored secret was created with a trailing `\n`); an untrimmed
        // key silently breaks SigV4 with SignatureDoesNotMatch. Creds/region
        // never legitimately contain surrounding whitespace, so trimming is safe.
        let clean = |v: String| {
            let t = v.trim().to_string();
            if t.is_empty() {
                None
            } else {
                Some(t)
            }
        };
        let access_key = std::env::var("AWS_ACCESS_KEY_ID").ok().and_then(clean);
        let secret_key = std::env::var("AWS_SECRET_ACCESS_KEY").ok().and_then(clean);
        let region = std::env::var("AWS_REGION")
            .or_else(|_| std::env::var("AWS_DEFAULT_REGION"))
            .ok()
            .and_then(clean);
        let (access_key, secret_key, region) = match (access_key, secret_key, region) {
            (Some(a), Some(s), Some(r)) => (a, s, r),
            _ => {
                tracing::error!(
                    "token export uri set but AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY/AWS_REGION \
                     missing; disabling s3 export"
                );
                return None;
            }
        };
        // Reject temporary credentials: our static-key SigV4 signer only signs
        // host;x-amz-content-sha256;x-amz-date, so STS/SSO creds would produce
        // a 403 SignatureDoesNotMatch on every PUT because the unsigned
        // x-amz-security-token header is not in SignedHeaders.
        if std::env::var("AWS_SESSION_TOKEN")
            .map(|v| !v.trim().is_empty())
            .unwrap_or(false)
        {
            tracing::error!(
                "token export: AWS_SESSION_TOKEN is set (temporary credentials) but the signer \
                 only supports static keys; disabling s3 export"
            );
            return None;
        }
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .ok()?;
        let uploader = Uploader::S3 {
            http,
            access_key,
            secret_key,
            region,
            bucket,
        };

        let metrics2 = Arc::clone(&metrics);
        let (tx, rx) = mpsc::channel(CHANNEL_CAPACITY);
        let join = tokio::spawn(async move {
            run_pump(
                rx,
                Arc::new(uploader),
                prefix,
                pod,
                metrics2,
                DEFAULT_MAX_BATCH_BYTES,
            )
            .await;
        });
        let capture_sem = Arc::new(tokio::sync::Semaphore::new(max_captures.max(1)));
        tracing::info!("s3 token export enabled");
        Some(Arc::new(Self {
            tx,
            metrics,
            join: AsyncMutex::new(Some(join)),
            capture_sem,
        }))
    }

    /// Try to acquire one capture permit. Returns `None` when the pool is
    /// exhausted. Mirrors `CacheSimTee::try_acquire_capture_permit` so the
    /// same backpressure logic applies whether cache-sim is on or off.
    pub fn try_acquire_capture_permit(&self) -> Option<tokio::sync::OwnedSemaphorePermit> {
        Arc::clone(&self.capture_sem).try_acquire_owned().ok()
    }

    /// Test entry point: inject an uploader (does not touch AWS).
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn spawn_with_uploader(
        uploader: Uploader,
        prefix: String,
        pod: String,
        metrics: Arc<MetricsRegistry>,
    ) -> Arc<Self> {
        Self::spawn_with_uploader_batch(uploader, prefix, pod, metrics, DEFAULT_MAX_BATCH_BYTES)
    }

    /// Test entry point (configurable batch size): for tests that need to
    /// trigger multiple batches from a small number of records.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn spawn_with_uploader_batch(
        uploader: Uploader,
        prefix: String,
        pod: String,
        metrics: Arc<MetricsRegistry>,
        max_batch_bytes: usize,
    ) -> Arc<Self> {
        let (tx, rx) = mpsc::channel(CHANNEL_CAPACITY);
        let metrics2 = Arc::clone(&metrics);
        let uploader = Arc::new(uploader);
        let join = tokio::spawn(async move {
            run_pump(rx, uploader, prefix, pod, metrics2, max_batch_bytes).await;
        });
        let capture_sem = Arc::new(tokio::sync::Semaphore::new(64));
        Arc::new(Self {
            tx,
            metrics,
            join: AsyncMutex::new(Some(join)),
            capture_sem,
        })
    }

    fn enqueue(&self, rec: ExportRecord) {
        match self.tx.try_send(PumpMsg::Record(Box::new(rec))) {
            Ok(()) => self.metrics.record_s3_export("enqueued"),
            Err(mpsc::error::TrySendError::Full(_)) => {
                self.metrics.record_s3_export("dropped_queue_full")
            }
            Err(mpsc::error::TrySendError::Closed(_)) => {
                // Pump is gone (post-drain offers). Make these visible so a
                // closed sink doesn't look like zero traffic.
                self.metrics.record_s3_export("dropped_closed")
            }
        }
    }

    pub fn offer_ingest(
        &self,
        model: &str,
        input_ids: &[u32],
        request_id: &str,
        slug: Option<&str>,
        key_id: Option<&str>,
    ) {
        if input_ids.is_empty() {
            return;
        }
        self.enqueue(ExportRecord {
            kind: ExportKind::Ingest,
            request_id: request_id.to_string(),
            slug: safe_slug(slug),
            model: model.to_string(),
            input_ids: input_ids.to_vec(),
            // An ingest record's `input_ids` is the whole prompt, so the
            // boundary is its length. Emitted (rather than left for downstream
            // to compute) so ingest and extend records share one schema.
            prompt_len: Some(input_ids.len()),
            output_tokens: None,
            choice_index: None,
            choice_count: None,
            key_id: key_id.map(str::to_owned),
            ts: chrono::Utc::now().to_rfc3339(),
        });
    }

    #[allow(clippy::too_many_arguments)]
    pub fn offer_extend(
        &self,
        model: &str,
        input_ids: &[u32],
        request_id: &str,
        prompt_len: Option<usize>,
        output_tokens: Option<u64>,
        choice_index: Option<usize>,
        choice_count: Option<usize>,
        slug: Option<&str>,
        key_id: Option<&str>,
    ) {
        if input_ids.is_empty() {
            return;
        }
        self.enqueue(ExportRecord {
            kind: ExportKind::Extend,
            request_id: request_id.to_string(),
            slug: safe_slug(slug),
            model: model.to_string(),
            input_ids: input_ids.to_vec(),
            prompt_len,
            output_tokens,
            choice_index,
            choice_count,
            key_id: key_id.map(str::to_owned),
            ts: chrono::Utc::now().to_rfc3339(),
        });
    }

    /// Send the drain signal and wait for the pump to flush all remaining
    /// records and exit. Used in the SIGTERM shutdown sequence.
    pub async fn drain(&self) {
        let (ack_tx, ack_rx) = tokio::sync::oneshot::channel();
        // `send` awaits capacity; the pump keeps draining records and freeing
        // slots, so the Drain message enqueues behind any backlog and is
        // processed normally. The is_ok()==false branch is only reached when
        // the channel is already closed (pump already exited).
        if self.tx.send(PumpMsg::Drain(ack_tx)).await.is_ok() {
            let _ = ack_rx.await;
        }
        if let Some(handle) = self.join.lock().await.take() {
            let _ = handle.await;
        }
    }
}

/// Gzip-compress `raw` using fast compression. Returns None on error (never panics).
fn gzip_fast(raw: Vec<u8>) -> Option<Vec<u8>> {
    let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
    // Writing to an in-memory Vec cannot produce IO errors, but we match anyway
    // to preserve the never-panic invariant.
    if enc.write_all(&raw).is_err() {
        return None;
    }
    enc.finish().ok()
}

struct UploadJobArgs {
    raw_bytes: Vec<u8>,
    records: u64,
    slug: String,
    seq: u64,
    prefix: String,
    pod: String,
    uploader: Arc<Uploader>,
    metrics: Arc<MetricsRegistry>,
    is_drain: bool,
    /// The semaphore to acquire a permit from at job start. The permit is held
    /// for the job's lifetime to bound concurrency.
    sem: Arc<tokio::sync::Semaphore>,
}

/// Single upload job: gzip in a blocking thread, compute key, put with retry,
/// record metrics. Acquires a semaphore permit at start so the pump never
/// blocks on permit acquisition.
async fn upload_job(args: UploadJobArgs) {
    let UploadJobArgs {
        raw_bytes,
        records,
        slug,
        seq,
        prefix,
        pod,
        uploader,
        metrics,
        is_drain,
        sem,
    } = args;

    // Acquire the permit inside the job so the pump loop is never blocked.
    let _permit = match sem.acquire_owned().await {
        Ok(p) => p,
        Err(_) => return, // semaphore closed; should not happen in normal operation
    };

    // Gzip on a blocking thread so we don't hold the async executor during CPU work.
    let gz = match tokio::task::spawn_blocking(move || gzip_fast(raw_bytes)).await {
        Ok(Some(gz)) => gz,
        Ok(None) | Err(_) => {
            metrics.record_s3_export("put_failed");
            return;
        }
    };

    let date = chrono::Utc::now().format("%Y-%m-%d").to_string();
    let nanos = chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0) as u128;
    let key = object_key(&prefix, &slug, &date, &pod, nanos, seq);

    if put_with_retry(&uploader, &key, gz, PUT_MAX_ATTEMPTS).await {
        metrics.record_s3_export("object_put");
        metrics.add_s3_export_records_uploaded(records);
        if is_drain {
            metrics.record_s3_export("drain_flushed");
        }
    } else {
        metrics.record_s3_export("put_failed");
    }
    // _permit is dropped here, releasing the semaphore slot.
}

/// Background pump: consume records, accumulate batches, flush on time/size,
/// upload concurrently. Never panics.
async fn run_pump(
    mut rx: mpsc::Receiver<PumpMsg>,
    uploader: Arc<Uploader>,
    prefix: String,
    pod: String,
    metrics: Arc<MetricsRegistry>,
    max_batch_bytes: usize,
) {
    let mut batcher = SlugBatcher::new(max_batch_bytes, DEFAULT_MAX_BATCH_AGE);
    let mut seq: u64 = 0;
    let mut tick = tokio::time::interval(TICK);
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

    let sem = Arc::new(tokio::sync::Semaphore::new(UPLOAD_CONCURRENCY));
    let mut join_set: tokio::task::JoinSet<()> = tokio::task::JoinSet::new();

    loop {
        tokio::select! {
            maybe = rx.recv() => {
                match maybe {
                    Some(PumpMsg::Record(rec)) => {
                        let line = rec.to_ndjson_line();
                        if !line.is_empty() {
                            batcher.push(&rec.slug, &line, Instant::now());
                        }
                        dispatch_ready(
                            &mut batcher, &uploader, &prefix, &pod, &metrics,
                            &mut seq, false, &sem, &mut join_set,
                        );
                    }
                    Some(PumpMsg::Drain(ack)) => {
                        dispatch_ready(
                            &mut batcher, &uploader, &prefix, &pod, &metrics,
                            &mut seq, true, &sem, &mut join_set,
                        );
                        // Wait for all in-flight uploads before acking.
                        while join_set.join_next().await.is_some() {}
                        let _ = ack.send(());
                        return;
                    }
                    None => {
                        // All senders closed: force flush then exit.
                        dispatch_ready(
                            &mut batcher, &uploader, &prefix, &pod, &metrics,
                            &mut seq, true, &sem, &mut join_set,
                        );
                        while join_set.join_next().await.is_some() {}
                        return;
                    }
                }
            }
            _ = tick.tick() => {
                dispatch_ready(
                    &mut batcher, &uploader, &prefix, &pod, &metrics,
                    &mut seq, false, &sem, &mut join_set,
                );
            }
        }
        // Opportunistically reap finished jobs so the JoinSet doesn't grow unbounded.
        while join_set.try_join_next().is_some() {}
    }
}

/// Dispatch all ready batches from the batcher as concurrent upload jobs.
/// When `force` is true (drain/close), flushes every remaining batch and is
/// exempt from the pending-job cap, so shutdown loses nothing (upload
/// completion is separately bounded by the shutdown budget in `main.rs`).
///
/// This function is synchronous (no `.await`): the pump never blocks on
/// permit acquisition. Jobs self-acquire their semaphore permits, so a
/// stalled S3 upload does not freeze the tick or the drain path. In steady
/// state (`force=false`) pending job count is bounded by `MAX_PENDING_UPLOADS`;
/// excess batches are dropped and metered as `dropped_upload_backlog`.
#[allow(clippy::too_many_arguments)]
fn dispatch_ready(
    batcher: &mut SlugBatcher,
    uploader: &Arc<Uploader>,
    prefix: &str,
    pod: &str,
    metrics: &Arc<MetricsRegistry>,
    seq: &mut u64,
    force: bool,
    sem: &Arc<tokio::sync::Semaphore>,
    join_set: &mut tokio::task::JoinSet<()>,
) {
    for obj in batcher.take_ready(Instant::now(), force) {
        let current_seq = *seq;
        *seq += 1;

        // Steady-state backpressure only: cap pending jobs to bound memory.
        // The drain/close path (`force`) is exempt — shutdown must flush every
        // remaining batch.
        if !force && join_set.len() >= MAX_PENDING_UPLOADS {
            metrics.record_s3_export("dropped_upload_backlog");
            continue;
        }

        join_set.spawn(upload_job(UploadJobArgs {
            raw_bytes: obj.raw_bytes,
            records: obj.records,
            slug: obj.slug,
            seq: current_seq,
            prefix: prefix.to_string(),
            pod: pod.to_string(),
            uploader: Arc::clone(uploader),
            metrics: Arc::clone(metrics),
            is_drain: force,
            sem: Arc::clone(sem),
        }));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    fn gunzip(b: &[u8]) -> String {
        use std::io::Read;
        let mut d = flate2::read::GzDecoder::new(b);
        let mut s = String::new();
        d.read_to_string(&mut s).unwrap();
        s
    }

    #[test]
    fn batcher_flushes_on_size() {
        // line is ~53 bytes uncompressed, which is >= 32 (the uncompressed threshold)
        let mut b = SlugBatcher::new(32, Duration::from_secs(999));
        let t0 = Instant::now();
        // Single ~53-byte line >= max_bytes(32) -> immediately ready
        let line = "{\"kind\":\"ingest\",\"input_ids\":[1,2,3,4,5,6,7,8,9,10]}\n";
        b.push("slugA", line, t0);
        let ready = b.take_ready(t0, false);
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].slug, "slugA");
        assert_eq!(ready[0].records, 1);
        // Batcher now returns raw bytes; verify the content directly.
        assert_eq!(std::str::from_utf8(&ready[0].raw_bytes).unwrap(), line);
    }

    #[test]
    fn batcher_flushes_on_age_and_separates_slugs() {
        let mut b = SlugBatcher::new(1 << 20, Duration::from_millis(100));
        let t0 = Instant::now();
        b.push("slugA", "a\n", t0);
        b.push("slugB", "b\n", t0);
        // Not yet over age or size -> not ready
        assert!(b.take_ready(t0, false).is_empty());
        // Past the age threshold -> one object per slug
        let later = t0 + Duration::from_millis(150);
        let mut ready = b.take_ready(later, false);
        ready.sort_by(|x, y| x.slug.cmp(&y.slug));
        assert_eq!(ready.len(), 2);
        assert_eq!(ready[0].slug, "slugA");
        assert_eq!(std::str::from_utf8(&ready[0].raw_bytes).unwrap(), "a\n");
        assert_eq!(ready[1].slug, "slugB");
    }

    #[test]
    fn batcher_force_flushes_everything() {
        let mut b = SlugBatcher::new(1 << 20, Duration::from_secs(999));
        let t0 = Instant::now();
        b.push("slugA", "a\n", t0);
        let ready = b.take_ready(t0, true);
        assert_eq!(ready.len(), 1);
    }

    #[test]
    fn parse_s3_uri_splits_bucket_and_prefix() {
        assert_eq!(
            parse_s3_uri("s3://my-bucket/token-export/"),
            Some(("my-bucket".into(), "token-export".into()))
        );
        assert_eq!(
            parse_s3_uri("s3://b/a/b/c"),
            Some(("b".into(), "a/b/c".into()))
        );
        assert_eq!(
            parse_s3_uri("s3://only-bucket"),
            Some(("only-bucket".into(), "".into()))
        );
        assert_eq!(parse_s3_uri("https://x"), None);
        assert_eq!(parse_s3_uri("s3://"), None);
    }

    #[test]
    fn ingest_line_omits_extend_only_fields() {
        let r = ExportRecord {
            kind: ExportKind::Ingest,
            request_id: "rid".into(),
            slug: "slugA".into(),
            model: "m".into(),
            input_ids: vec![128, 9021],
            prompt_len: None,
            output_tokens: None,
            choice_index: None,
            choice_count: None,
            key_id: None,
            ts: "2026-08-10T00:00:00Z".into(),
        };
        let line = r.to_ndjson_line();
        assert!(line.ends_with('\n'));
        let v: serde_json::Value = serde_json::from_str(line.trim_end()).unwrap();
        assert_eq!(v["kind"], "ingest");
        assert_eq!(v["input_ids"], serde_json::json!([128, 9021]));
        assert_eq!(v["slug"], "slugA");
        assert!(v.get("prompt_len").is_none());
        assert!(v.get("output_tokens").is_none());
        assert!(v.get("choice_index").is_none());
        assert!(v.get("choice_count").is_none());
        assert!(v.get("key_id").is_none());
    }

    #[test]
    fn fanout_line_includes_choice_index_and_count() {
        let r = ExportRecord {
            kind: ExportKind::Extend,
            request_id: "rid".into(),
            slug: "s".into(),
            model: "m".into(),
            input_ids: vec![1, 2],
            prompt_len: None,
            output_tokens: None,
            choice_index: Some(0),
            choice_count: Some(3),
            key_id: None,
            ts: "2026-08-10T00:00:00Z".into(),
        };
        let v: serde_json::Value = serde_json::from_str(r.to_ndjson_line().trim_end()).unwrap();
        assert_eq!(v["choice_index"], 0);
        assert_eq!(v["choice_count"], 3);
    }

    #[test]
    fn extend_line_includes_prompt_len_and_output_tokens() {
        let r = ExportRecord {
            kind: ExportKind::Extend,
            request_id: "rid".into(),
            slug: "unknown".into(),
            model: "m".into(),
            input_ids: vec![1, 2, 3],
            prompt_len: Some(2),
            output_tokens: Some(9),
            choice_index: None,
            choice_count: None,
            key_id: Some("key-123".into()),
            ts: "2026-08-10T00:00:00Z".into(),
        };
        let v: serde_json::Value = serde_json::from_str(r.to_ndjson_line().trim_end()).unwrap();
        assert_eq!(v["kind"], "extend");
        assert_eq!(v["prompt_len"], 2);
        assert_eq!(v["output_tokens"], 9);
        assert_eq!(v["key_id"], "key-123");
    }

    #[test]
    fn object_key_is_hive_partitioned_and_unique() {
        let k1 = object_key("token-export", "slugA", "2026-08-10", "pod-1", 111, 0);
        assert_eq!(
            k1,
            "token-export/slug=slugA/date=2026-08-10/pod-1-111-0.ndjson.gz"
        );
        let k2 = object_key("token-export", "slugA", "2026-08-10", "pod-1", 111, 1);
        assert_ne!(k1, k2, "seq must disambiguate same-nanos objects");
        // Empty prefix must not produce a leading slash.
        let k3 = object_key("", "s", "2026-08-10", "pod-1", 1, 0);
        assert_eq!(k3, "slug=s/date=2026-08-10/pod-1-1-0.ndjson.gz");
    }

    #[tokio::test]
    async fn put_with_retry_succeeds_after_transient_failures() {
        let store = Arc::new(FakeStore::new(/*fail_first=*/ 2));
        let up = Uploader::Fake(Arc::clone(&store));
        let ok = put_with_retry(&up, "k", b"payload".to_vec(), 5).await;
        assert!(ok, "should succeed on the 3rd attempt");
        let puts = store.puts.lock().unwrap();
        assert_eq!(puts.len(), 1);
        assert_eq!(puts[0].0, "k");
        assert_eq!(puts[0].1, b"payload");
    }

    #[tokio::test]
    async fn put_with_retry_gives_up_after_max_attempts() {
        let store = Arc::new(FakeStore::new(/*fail_first=*/ 100));
        let up = Uploader::Fake(Arc::clone(&store));
        let ok = put_with_retry(&up, "k", b"x".to_vec(), 3).await;
        assert!(!ok);
        assert!(store.puts.lock().unwrap().is_empty());
    }

    /// Real S3 round-trip against the actual `Uploader::S3` SigV4 path.
    /// Ignored by default (no creds in CI). Run manually with STATIC keys:
    ///   AWS_ACCESS_KEY_ID=.. AWS_SECRET_ACCESS_KEY=.. AWS_REGION=us-west-2 \
    ///   TEST_S3_URI=s3://<bucket>/<prefix>/ \
    ///   cargo test --lib s3_export::tests::real_s3_roundtrip -- --ignored --nocapture
    /// (SSO/temporary creds won't work — we don't sign x-amz-security-token.)
    #[tokio::test]
    #[ignore]
    async fn real_s3_roundtrip() {
        let (ak, sk, region, uri) = match (
            std::env::var("AWS_ACCESS_KEY_ID"),
            std::env::var("AWS_SECRET_ACCESS_KEY"),
            std::env::var("AWS_REGION").or_else(|_| std::env::var("AWS_DEFAULT_REGION")),
            std::env::var("TEST_S3_URI"),
        ) {
            (Ok(a), Ok(s), Ok(r), Ok(u)) if !a.is_empty() && !s.is_empty() && !r.is_empty() => {
                (a, s, r, u)
            }
            _ => {
                eprintln!("real_s3_roundtrip skipped: set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY/AWS_REGION/TEST_S3_URI");
                return;
            }
        };
        let (bucket, prefix) = parse_s3_uri(&uri).expect("TEST_S3_URI must be s3://bucket/prefix/");
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("build reqwest client");
        if std::env::var("AWS_SESSION_TOKEN")
            .map(|v| !v.is_empty())
            .unwrap_or(false)
        {
            eprintln!(
                "WARNING: AWS_SESSION_TOKEN is set — these are TEMPORARY/SSO credentials. \
                 Our SigV4 signer does NOT send x-amz-security-token, so S3 will reject them. \
                 Use STATIC IAM user keys (not temporary/SSO credentials)."
            );
        }
        let up = Uploader::S3 {
            http,
            access_key: ak,
            secret_key: sk,
            region,
            bucket,
        };
        let key = if prefix.is_empty() {
            "roundtrip-check.txt".to_string()
        } else {
            format!("{prefix}/roundtrip-check.txt")
        };
        // Call put() directly (not put_with_retry) so the real S3 error surfaces.
        // Retry a few times to tolerate fresh-IAM propagation.
        let mut last_err = None;
        for attempt in 1..=6u32 {
            match up
                .put(&key, b"sgl-router s3 export roundtrip\n".to_vec())
                .await
            {
                Ok(()) => {
                    eprintln!("real_s3_roundtrip OK -> s3://.../{key}");
                    return;
                }
                Err(e) => {
                    eprintln!("attempt {attempt}/6 failed: {e}");
                    last_err = Some(e);
                    tokio::time::sleep(Duration::from_millis(500 * attempt as u64)).await;
                }
            }
        }
        panic!("real S3 PutObject to {key} failed: {}", last_err.unwrap());
    }

    // ---- SigV4 building blocks ----

    #[test]
    fn sha256_hex_of_empty_is_known_vector() {
        // Well-known SHA-256("") test vector.
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn hmac_sha256_matches_rfc4231_case2() {
        // RFC 4231 Test Case 2: key="Jefe", data="what do ya want for nothing?"
        let mac = hmac_sha256(b"Jefe", b"what do ya want for nothing?");
        assert_eq!(
            hex::encode(mac),
            "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843"
        );
    }

    #[test]
    fn signing_key_is_deterministic_and_32_bytes() {
        let k1 = signing_key("secret", "20260810", "us-west-2", "s3");
        let k2 = signing_key("secret", "20260810", "us-west-2", "s3");
        assert_eq!(k1, k2);
        assert_eq!(k1.len(), 32);
        // Different date -> different key (proves each segment contributes to the chain).
        assert_ne!(k1, signing_key("secret", "20260811", "us-west-2", "s3"));
    }

    #[test]
    fn uri_encode_preserves_slash_when_asked() {
        assert_eq!(uri_encode("a b/c", false), "a%20b/c");
        assert_eq!(uri_encode("a b/c", true), "a%20b%2Fc");
        assert_eq!(uri_encode("k-._~", false), "k-._~");
    }

    #[test]
    fn canonical_request_has_exact_shape() {
        let creq = canonical_request(
            "PUT",
            "/pfx/slug=s/date=2026-08-10/pod-1-1-0.ndjson.gz",
            "b.s3.us-west-2.amazonaws.com",
            "20260810T000000Z",
            "abc123",
        );
        let expected = "PUT\n\
            /pfx/slug=s/date=2026-08-10/pod-1-1-0.ndjson.gz\n\
            \n\
            host:b.s3.us-west-2.amazonaws.com\n\
            x-amz-content-sha256:abc123\n\
            x-amz-date:20260810T000000Z\n\
            \n\
            host;x-amz-content-sha256;x-amz-date\n\
            abc123";
        assert_eq!(creq, expected);
    }

    #[test]
    fn string_to_sign_has_exact_shape() {
        let sts = string_to_sign(
            "20260810T000000Z",
            "20260810",
            "us-west-2",
            "s3",
            "deadbeef",
        );
        assert_eq!(
            sts,
            "AWS4-HMAC-SHA256\n20260810T000000Z\n20260810/us-west-2/s3/aws4_request\ndeadbeef"
        );
    }

    fn test_metrics() -> Arc<MetricsRegistry> {
        MetricsRegistry::new()
    }

    #[tokio::test]
    async fn drain_flushes_offered_records_to_uploader() {
        let store = Arc::new(FakeStore::new(0));
        let up = Uploader::Fake(Arc::clone(&store));
        let sink =
            S3ExportSink::spawn_with_uploader(up, "pfx".into(), "pod-1".into(), test_metrics());
        sink.offer_ingest("m", &[1, 2, 3], "rid", Some("slugA"), Some("key-abc"));
        sink.offer_extend(
            "m",
            &[1, 2, 3, 4],
            "rid",
            Some(3),
            Some(1),
            None,
            None,
            Some("slugA"),
            Some("key-abc"),
        );
        sink.drain().await;

        let puts = store.puts.lock().unwrap();
        assert_eq!(puts.len(), 1, "one slug -> one object");
        assert!(puts[0].0.starts_with("pfx/slug=slugA/date="));
        assert!(puts[0].0.ends_with(".ndjson.gz"));
        let body = gunzip(&puts[0].1);
        assert_eq!(body.lines().count(), 2, "ingest + extend");
        assert!(body.contains("\"kind\":\"ingest\""));
        assert!(body.contains("\"kind\":\"extend\""));
        // Both records carry the gateway key_id; the ingest record now emits
        // its prompt_len (= input_ids.len()) so it shares the extend schema.
        assert_eq!(body.matches("\"key_id\":\"key-abc\"").count(), 2);
        assert!(body.contains("\"prompt_len\":3"));
    }

    #[tokio::test]
    async fn offer_never_blocks_and_missing_slug_becomes_unknown() {
        let store = Arc::new(FakeStore::new(0));
        let sink = S3ExportSink::spawn_with_uploader(
            Uploader::Fake(Arc::clone(&store)),
            "".into(),
            "pod-1".into(),
            test_metrics(),
        );
        sink.offer_ingest("m", &[7], "rid", None, None); // missing slug
        sink.drain().await;
        let puts = store.puts.lock().unwrap();
        assert_eq!(puts.len(), 1);
        assert!(puts[0].0.starts_with("slug=unknown/date="));
    }

    /// Verifies that a single slug can produce multiple concurrent upload objects, and
    /// that drain waits for ALL of them before returning. Uses a tiny batch threshold
    /// to force multiple batches from a small number of records.
    #[tokio::test]
    async fn drain_waits_for_all_concurrent_uploads() {
        // Use a 1-byte batch threshold to force one object per record.
        let store = Arc::new(FakeStore::new(0));
        let sink = S3ExportSink::spawn_with_uploader_batch(
            Uploader::Fake(Arc::clone(&store)),
            "pfx".into(),
            "pod-1".into(),
            test_metrics(),
            1, // 1-byte max: every push triggers a new batch immediately
        );

        // Offer several records for a single slug.
        const N: usize = 8;
        for i in 0..N {
            sink.offer_ingest("m", &[i as u32, i as u32 + 1], "rid", Some("slugA"), None);
        }
        sink.drain().await;

        let puts = store.puts.lock().unwrap();
        // Every record produces its own gzipped object (batch size = 1 byte).
        assert_eq!(puts.len(), N, "expected {N} objects, got {}", puts.len());

        // All objects are under the correct prefix.
        for (key, _) in puts.iter() {
            assert!(
                key.starts_with("pfx/slug=slugA/date="),
                "unexpected key: {key}"
            );
        }

        // Total decompressed line count must equal N.
        let total_lines: usize = puts.iter().map(|(_, gz)| gunzip(gz).lines().count()).sum();
        assert_eq!(total_lines, N, "decompressed line count mismatch");
    }

    #[test]
    fn safe_slug_allowlists() {
        // Well-formed slugs pass through unchanged.
        assert_eq!(safe_slug(Some("good-slug.1")), "good-slug.1");
        // Path separators collapse to "unknown".
        assert_eq!(safe_slug(Some("a/date=1")), "unknown");
        // Empty string collapses.
        assert_eq!(safe_slug(Some("")), "unknown");
        // A 65-character slug exceeds the 64-byte cap and collapses.
        assert_eq!(safe_slug(Some(&"a".repeat(65))), "unknown");
        // None collapses.
        assert_eq!(safe_slug(None), "unknown");
    }
}
