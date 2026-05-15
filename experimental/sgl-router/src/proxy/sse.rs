// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! SSE passthrough — bridges a reqwest `bytes_stream()` into an axum Body.

use axum::body::Body;
use bytes::Bytes;
use futures::StreamExt;
use tokio_stream::wrappers::ReceiverStream;

/// Bridge a byte stream into an axum Body that streams chunks unchanged.
///
/// Spawns one tokio task per stream so the handler can return immediately.
/// Uses a **bounded** 64-slot channel so `tx.send().await` naturally
/// backpressures the upstream read when the client (axum Body consumer) falls
/// behind — an unbounded channel would buffer hundreds of MB for a slow client
/// receiving a long completion.
///
/// # Backpressure note
/// The channel bound of 64 absorbs short bursts while still limiting
/// worst-case outstanding bytes to 64 × chunk_size (typically a few MB).
///
/// # Client disconnect
/// When the axum Body is dropped the receiver is closed; `tx.send()` then
/// returns `Err`, which breaks the loop — no upstream bytes are read after the
/// client disconnects.
pub fn bytes_stream_to_body<S, E>(stream: S) -> Body
where
    S: futures::Stream<Item = Result<Bytes, E>> + Send + Unpin + 'static,
    E: std::fmt::Display + Send + Sync + 'static,
{
    let (tx, rx) = tokio::sync::mpsc::channel(64);
    tokio::spawn(async move {
        let mut s = stream;
        while let Some(chunk) = s.next().await {
            let item: Result<Bytes, std::io::Error> =
                chunk.map_err(|e| std::io::Error::other(e.to_string()));
            // tx.send err means the receiver was dropped (client disconnect); stop reading.
            if tx.send(item).await.is_err() {
                break;
            }
        }
    });
    Body::from_stream(ReceiverStream::new(rx))
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;
    use futures::stream;
    use http_body_util::BodyExt;

    #[tokio::test]
    async fn passes_through_a_simple_byte_stream() {
        let chunks = vec![
            Ok::<Bytes, std::io::Error>(Bytes::from_static(b"hello ")),
            Ok(Bytes::from_static(b"world")),
        ];
        let s = stream::iter(chunks);
        let body = bytes_stream_to_body(s);
        let bytes = body.collect().await.unwrap().to_bytes();
        assert_eq!(&bytes[..], b"hello world");
    }

    #[tokio::test]
    async fn upstream_error_surfaces_to_consumer() {
        let chunks: Vec<Result<Bytes, std::io::Error>> = vec![
            Ok(Bytes::from_static(b"ok-chunk")),
            Err(std::io::Error::other("upstream blew up mid-stream")),
        ];
        let s = stream::iter(chunks);
        let body = bytes_stream_to_body(s);
        // Collecting a body that terminates with an error must return Err.
        let result = body.collect().await;
        assert!(
            result.is_err(),
            "expected body collect to surface upstream error, got Ok"
        );
    }
}
