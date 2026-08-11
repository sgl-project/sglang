// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Deadline-based load shedding for the query path.
//!
//! A concurrency limit only measures overload for a backend that yields: one
//! doing IO holds its permit across the wait, whereas this build's process-local
//! backend returns its permit before the next request is even decoded, so the
//! limit stays far from its ceiling while requests pile up in the runtime
//! instead. What is observable in both cases is how long a request waited to
//! reach its handler.
//!
//! Once that wait has consumed the caller's whole deadline the answer can no
//! longer be read, so computing it only pushes the backlog further behind. The
//! budget is the caller's own `grpc-timeout`, which leaves no server-side
//! threshold to tune and never sheds a caller that declared no deadline.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use tonic::metadata::MetadataMap;
use tonic::{Extensions, Request, Status};

/// When the request's headers were read off the connection.
#[derive(Clone, Copy)]
struct Arrival(Instant);

/// Counts rejections of one kind and picks which ones to log.
///
/// Overload is a rate, not an event. Logging every rejection adds load exactly
/// when the service is shedding to get rid of some, while logging none leaves an
/// operator unable to tell shedding apart from ordinary latency. Reporting on
/// doubling counts keeps the first rejection immediate and the volume
/// logarithmic in a sustained overload.
pub(crate) struct RejectionLog(AtomicU64);

impl RejectionLog {
    pub(crate) const fn new() -> Self {
        Self(AtomicU64::new(0))
    }

    /// Records one rejection, returning the running total when it should be
    /// logged and `None` when it should be absorbed.
    pub(crate) fn record(&self) -> Option<u64> {
        let total = self.0.fetch_add(1, Ordering::Relaxed) + 1;
        total.is_power_of_two().then_some(total)
    }
}

static DEADLINE_SHED_LOG: RejectionLog = RejectionLog::new();

/// Server interceptor that timestamps a request's arrival, so the query path can
/// measure how long it then waited to be served.
///
/// Runs while the connection task is still dispatching the stream, before the
/// per-request task is spawned, so the stamp precedes any scheduling delay.
/// Without this interceptor installed nothing is ever shed.
pub fn stamp_arrival(mut request: Request<()>) -> Result<Request<()>, Status> {
    request.extensions_mut().insert(Arrival(Instant::now()));
    Ok(request)
}

/// Rejects a query that spent its caller's entire deadline waiting to be served.
///
/// Deliberately not applied to the apply path: dropping a KV event would leave
/// the index permanently diverged from the worker that reported it, while
/// dropping a query costs the caller one routing hint it had already given up on.
pub(crate) fn reject_if_deadline_passed(
    metadata: &MetadataMap,
    extensions: &Extensions,
) -> Result<(), Status> {
    let (Some(arrival), Some(budget)) = (extensions.get::<Arrival>(), caller_deadline(metadata))
    else {
        return Ok(());
    };
    let waited = arrival.0.elapsed();
    if waited < budget {
        return Ok(());
    }
    if let Some(shed_total) = DEADLINE_SHED_LOG.record() {
        tracing::info!(
            shed_total,
            waited_ms = waited.as_millis(),
            budget_ms = budget.as_millis(),
            "shedding prefix query whose caller deadline already passed"
        );
    }
    Err(Status::deadline_exceeded(
        "prefix query waited longer than its caller deadline",
    ))
}

/// The budget the caller declared in `grpc-timeout`, per the gRPC wire spec (up
/// to 8 digits followed by a unit). `None` for an absent or unparsable value,
/// which leaves the request unshed.
///
/// This is the budget as of the caller's send, so measuring it against the wait
/// since arrival ignores transit time and can only shed late, never early.
fn caller_deadline(metadata: &MetadataMap) -> Option<Duration> {
    let raw = metadata.get("grpc-timeout")?.to_str().ok()?;
    let unit = *raw.as_bytes().last()?;
    let value: u64 = raw.get(..raw.len() - 1)?.parse().ok()?;
    match unit {
        b'H' => value.checked_mul(60 * 60).map(Duration::from_secs),
        b'M' => value.checked_mul(60).map(Duration::from_secs),
        b'S' => Some(Duration::from_secs(value)),
        b'm' => Some(Duration::from_millis(value)),
        b'u' => Some(Duration::from_micros(value)),
        b'n' => Some(Duration::from_nanos(value)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metadata(timeout: Option<&str>) -> MetadataMap {
        let mut metadata = MetadataMap::new();
        if let Some(timeout) = timeout {
            metadata.insert("grpc-timeout", timeout.parse().unwrap());
        }
        metadata
    }

    fn extensions(arrival: Option<Instant>) -> Extensions {
        let mut extensions = Extensions::new();
        if let Some(arrival) = arrival {
            extensions.insert(Arrival(arrival));
        }
        extensions
    }

    #[test]
    fn parses_every_wire_unit() {
        for (raw, expected) in [
            ("1H", Duration::from_secs(3600)),
            ("2M", Duration::from_secs(120)),
            ("3S", Duration::from_secs(3)),
            ("100m", Duration::from_millis(100)),
            ("250u", Duration::from_micros(250)),
            ("400n", Duration::from_nanos(400)),
        ] {
            assert_eq!(caller_deadline(&metadata(Some(raw))), Some(expected));
        }
    }

    #[test]
    fn unparsable_deadline_is_ignored() {
        for raw in ["", "m", "100", "100x", "abcm", "99999999999999999999H"] {
            assert_eq!(caller_deadline(&metadata(Some(raw))), None, "{raw}");
        }
        assert_eq!(caller_deadline(&metadata(None)), None);
    }

    #[test]
    fn sheds_only_once_the_caller_deadline_has_passed() {
        let now = Instant::now();
        let waited_past = now - Duration::from_millis(150);
        let within = now - Duration::from_millis(10);

        assert_eq!(
            reject_if_deadline_passed(&metadata(Some("100m")), &extensions(Some(waited_past)))
                .unwrap_err()
                .code(),
            tonic::Code::DeadlineExceeded
        );
        assert!(
            reject_if_deadline_passed(&metadata(Some("100m")), &extensions(Some(within))).is_ok()
        );
    }

    #[test]
    fn rejections_are_reported_on_doubling_counts() {
        let log = RejectionLog::new();
        let reported: Vec<u64> = (0..16).filter_map(|_| log.record()).collect();
        // The first rejection is always reported, and the gaps then double, so a
        // sustained overload cannot flood the log.
        assert_eq!(reported, vec![1, 2, 4, 8, 16]);
    }

    #[test]
    fn missing_deadline_or_arrival_never_sheds() {
        let long_wait = Instant::now() - Duration::from_secs(60);
        // A caller that declared no deadline keeps the pre-existing behaviour.
        assert!(reject_if_deadline_passed(&metadata(None), &extensions(Some(long_wait))).is_ok());
        // No interceptor installed: nothing to measure the wait against.
        assert!(reject_if_deadline_passed(&metadata(Some("1m")), &extensions(None)).is_ok());
    }
}
