//! TTFT-breakdown stamps — throwaway profiling instrumentation for the
//! `kan/rust-tm-ttft-breakdown` branch; not intended to merge.
//!
//! Each stamp is one log line on the `ttft` target:
//! `TTFT_STAMP rid=<wire rid> st=<stage> t=<ns>`
//! where `t` is CLOCK_MONOTONIC nanoseconds — directly comparable with
//! Python's `time.perf_counter_ns()` (same clock on Linux) in the scheduler
//! process and in a same-host client.

/// CLOCK_MONOTONIC in nanoseconds.
#[inline]
pub fn mono_ns() -> u64 {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `ts` is a valid out-pointer; CLOCK_MONOTONIC cannot fail on Linux.
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts) };
    ts.tv_sec as u64 * 1_000_000_000 + ts.tv_nsec as u64
}

/// Stamp `stage` for `rid` at the current time.
#[inline]
pub fn stamp(stage: &'static str, rid: &str) {
    stamp_at(stage, rid, mono_ns());
}

/// Stamp `stage` for `rid` at a previously captured time (for stamps taken
/// before the rid was known, e.g. HTTP request-head arrival).
#[inline]
pub fn stamp_at(stage: &'static str, rid: &str, t_ns: u64) {
    tracing::info!(target: "ttft", "TTFT_STAMP rid={rid} st={stage} t={t_ns}");
}
