// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Run CPU-heavy request work (chat-template render, tokenize, body
//! rebuild) without stalling the async runtime.
//!
//! Why: a tokio worker that spends seconds inside one poll does not just delay
//! its own task. The idle workers stay parked and nobody polls the I/O driver,
//! so the whole router stops serving — no accepts, no h2 PING replies, no
//! stream bytes — until that poll returns. A pathological chat render is such a
//! poll. [`tokio::task::block_in_place`] hands this worker's core (and with it
//! the I/O driver) to another thread before the blocking work starts, and it
//! works on borrowed data, so callers keep their signatures.

use tokio::runtime::{Handle, RuntimeFlavor};

/// Run `work` on the current thread, first moving this worker's async duties
/// to another thread. On a current-thread runtime, outside any runtime, or on a
/// blocking-pool thread, `work` runs inline: `block_in_place` would panic in
/// the first case and has nothing to hand off in the others.
pub(crate) fn run<R>(work: impl FnOnce() -> R) -> R {
    match Handle::try_current() {
        Ok(handle) if handle.runtime_flavor() == RuntimeFlavor::MultiThread => {
            tokio::task::block_in_place(work)
        }
        _ => work(),
    }
}

/// Body size below which parsing and rebuilding JSON costs less than the
/// handoff in [`run`]: about 0.5 µs per KiB, against tens of µs per handoff.
pub(crate) const MIN_OFFLOAD_BODY_BYTES: usize = 256 * 1024;

/// [`run`] for work that grows with a `len`-byte body, such as JSON parse and
/// rebuild. Smaller bodies run inline and skip the handoff. Render and
/// tokenize go through [`run`] on their own, whatever the body size.
pub(crate) fn run_for_body<R>(len: usize, work: impl FnOnce() -> R) -> R {
    if len >= MIN_OFFLOAD_BODY_BYTES {
        run(work)
    } else {
        work()
    }
}

/// Fixtures for asserting that a code path keeps the runtime serving while it
/// does slow synchronous work.
#[cfg(test)]
pub(crate) mod test_support {
    use std::future::Future;
    use std::time::{Duration, Instant};

    /// A model directory whose chat template spins on every message whose
    /// content is `slow`, and renders anything else instantly, so loading and
    /// load-time probes stay quick.
    pub(crate) fn slow_template_dir() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        std::fs::copy(
            "tests/fixtures/tiny_tokenizer.json",
            dir.path().join("tokenizer.json"),
        )
        .unwrap();
        std::fs::write(dir.path().join("tokenizer_config.json"), "{}").unwrap();
        std::fs::write(
            dir.path().join("chat_template.jinja"),
            "{% for m in messages %}{% if m.content == 'slow' %}\
             {% for i in range(1000) %}{% for j in range(1000) %}{% endfor %}{% endfor %}\
             {% endif %}{{ m.content }}{% endfor %}",
        )
        .unwrap();
        dir
    }

    /// A chat body whose render takes at least `floor` on this machine and
    /// build profile: the number of `slow` messages doubles until a direct
    /// `render` call is that slow, so the fixture holds in release builds too.
    pub(crate) fn slow_chat_body(
        render: impl Fn(&serde_json::Value),
        floor: Duration,
    ) -> serde_json::Value {
        let mut slow = 1;
        loop {
            let messages: Vec<_> = (0..slow)
                .map(|_| serde_json::json!({"role": "user", "content": "slow"}))
                .collect();
            let body = serde_json::json!({"model": "stub-model", "messages": messages});
            let start = Instant::now();
            render(&body);
            if start.elapsed() >= floor || slow >= 1 << 12 {
                return body;
            }
            slow *= 2;
        }
    }

    /// Runs `work` in a spawned task on the current runtime (which should have
    /// ONE worker) while another task waits on a 20 ms timer. Returns how long
    /// `work` took and how long the timer task waited. If `work` blocks inside
    /// its poll, the timer task waits for all of it.
    pub(crate) async fn time_beside_a_timer<F, Fut>(work: F) -> (Duration, Duration)
    where
        F: FnOnce() -> Fut + Send + 'static,
        Fut: Future<Output = ()> + Send + 'static,
    {
        let (armed_tx, armed_rx) = tokio::sync::oneshot::channel();
        let ticker = tokio::spawn(async move {
            let start = Instant::now();
            let timer = tokio::time::sleep(Duration::from_millis(20));
            armed_tx.send(()).unwrap();
            timer.await;
            start.elapsed()
        });
        // The test body runs in `block_on` on the test thread, not on the
        // worker, so `work` must be spawned to occupy the worker.
        armed_rx.await.unwrap();
        let worker = tokio::spawn(async move {
            let start = Instant::now();
            work().await;
            start.elapsed()
        });
        let ticked_after = ticker.await.unwrap();
        (worker.await.unwrap(), ticked_after)
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::time_beside_a_timer;
    use super::*;
    use std::time::Duration;

    /// With ONE worker, a spawned task whose blocking work goes through [`run`]
    /// must not stop another task's timer from firing.
    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn blocking_work_leaves_the_runtime_serving() {
        let (took, ticked_after) = time_beside_a_timer(|| async {
            run(|| std::thread::sleep(Duration::from_millis(600)));
        })
        .await;
        assert!(
            ticked_after < took / 2,
            "timer task waited {ticked_after:?} while the work took {took:?}"
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn a_current_thread_runtime_runs_the_work_inline() {
        assert_eq!(run(|| 7), 7);
    }

    #[test]
    fn outside_a_runtime_the_work_runs_inline() {
        assert_eq!(run(|| "ok"), "ok");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_blocking_pool_thread_runs_the_work_inline() {
        let got = tokio::task::spawn_blocking(|| run(|| 7)).await.unwrap();
        assert_eq!(got, 7);
    }
}
