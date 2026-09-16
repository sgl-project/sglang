// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! In-flight HTTP accounting, for the termination drain to report on.
//!
//! [`ActiveLoadRegistry`](crate::policies::active_load::ActiveLoadRegistry)
//! counts *proxied* requests — what the workers are busy with. Axum's graceful
//! shutdown waits on something different and larger: every HTTP exchange still
//! open on an accepted connection, on any route, until its response body has
//! finished streaming. The two diverge exactly where the drain gets stuck — a
//! stalled SSE consumer, a held `/metrics` scrape, a request on a non-proxied
//! route — so a heartbeat reporting only the first says "0 in flight" about a
//! pod that is minutes from being SIGKILLed with work outstanding.
//!
//! The count is taken at the edge middleware and released when the response
//! **body** completes, not when the handler returns: for a streaming
//! completion the handler returns as soon as the headers are ready, which is
//! the beginning of the wait, not the end of it.

use axum::body::{Body, Bytes, HttpBody};
use http_body::{Frame, SizeHint};
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll};

/// Open HTTP exchanges, as axum's graceful shutdown counts them.
#[derive(Debug, Default)]
pub struct InflightHttp {
    open: AtomicUsize,
}

impl InflightHttp {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Exchanges whose response body has not finished. Relaxed throughout:
    /// this is a diagnostic gauge, and a reader racing an increment sees the
    /// count one tick later, never a torn value.
    pub fn count(&self) -> usize {
        self.open.load(Ordering::Relaxed)
    }

    /// Count one exchange until the returned guard drops.
    pub fn enter(self: &Arc<Self>) -> InflightGuard {
        self.open.fetch_add(1, Ordering::Relaxed);
        InflightGuard(Arc::clone(self))
    }
}

/// Releases its exchange on drop — including when the request future is
/// cancelled (client gone before the response was built), which is why this is
/// a guard and not a pair of explicit increment/decrement calls.
pub struct InflightGuard(Arc<InflightHttp>);

impl Drop for InflightGuard {
    fn drop(&mut self) {
        self.0.open.fetch_sub(1, Ordering::Relaxed);
    }
}

/// A response body that holds `guard` until the last frame is yielded or the
/// body is dropped.
struct TrackedBody {
    inner: Body,
    _guard: InflightGuard,
}

/// Wrap `body` so the exchange stays counted for as long as the client is
/// still being written to.
pub fn track_body(body: Body, guard: InflightGuard) -> Body {
    Body::new(TrackedBody {
        inner: body,
        _guard: guard,
    })
}

impl HttpBody for TrackedBody {
    type Data = Bytes;
    type Error = axum::Error;

    fn poll_frame(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        // `Body` and `InflightGuard` are both `Unpin`, so the projection is a
        // plain field borrow rather than anything `pin-project` is needed for.
        Pin::new(&mut self.get_mut().inner).poll_frame(cx)
    }

    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }

    fn size_hint(&self) -> SizeHint {
        self.inner.size_hint()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use http_body_util::BodyExt;

    #[test]
    fn the_guard_releases_on_drop() {
        let counter = InflightHttp::new();
        assert_eq!(counter.count(), 0);
        let guard = counter.enter();
        assert_eq!(counter.count(), 1);
        let second = counter.enter();
        assert_eq!(counter.count(), 2);
        drop(guard);
        assert_eq!(counter.count(), 1);
        drop(second);
        assert_eq!(counter.count(), 0);
    }

    /// The property the whole module exists for: a response whose headers are
    /// ready is NOT finished. A counter released when the handler returns
    /// reports 0 while a streaming completion is still being written — which is
    /// precisely the drain the heartbeat is supposed to explain.
    #[tokio::test]
    async fn a_streaming_body_stays_counted_until_its_last_frame() {
        let counter = InflightHttp::new();
        let (mut tx, rx) = futures::channel::mpsc::channel::<Result<Bytes, axum::Error>>(1);
        let body = track_body(Body::from_stream(rx), counter.enter());
        assert_eq!(counter.count(), 1, "entering must count the exchange");

        let mut stream = body.into_data_stream();
        futures::SinkExt::send(&mut tx, Ok(Bytes::from_static(b"chunk")))
            .await
            .expect("the stream must accept a chunk");
        let frame = futures::StreamExt::next(&mut stream).await;
        assert!(frame.is_some(), "the wrapper must pass frames through");
        assert_eq!(
            counter.count(),
            1,
            "a body with frames still to come must stay counted",
        );

        drop(tx);
        assert!(
            futures::StreamExt::next(&mut stream).await.is_none(),
            "the wrapper must end when the inner body does",
        );
        drop(stream);
        assert_eq!(counter.count(), 0, "a finished body must release the count");
    }

    /// A client that disconnects mid-stream drops the body rather than
    /// draining it. That must release the count too, or the gauge ratchets up
    /// over the life of the process and the drain heartbeat reads permanently
    /// busy.
    #[tokio::test]
    async fn an_abandoned_body_releases_the_count() {
        let counter = InflightHttp::new();
        let (_tx, rx) = futures::channel::mpsc::channel::<Result<Bytes, axum::Error>>(1);
        let body = track_body(Body::from_stream(rx), counter.enter());
        assert_eq!(counter.count(), 1);
        drop(body);
        assert_eq!(counter.count(), 0);
    }

    #[tokio::test]
    async fn a_unit_body_passes_through_unchanged() {
        let counter = InflightHttp::new();
        let body = track_body(Body::from("payload"), counter.enter());
        let bytes = body
            .collect()
            .await
            .expect("collecting a tracked body must not error")
            .to_bytes();
        assert_eq!(&bytes[..], b"payload");
        assert_eq!(counter.count(), 0);
    }
}
