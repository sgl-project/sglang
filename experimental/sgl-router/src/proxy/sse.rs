// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! SSE passthrough — bridges a reqwest `bytes_stream()` into an axum Body.

use axum::body::Body;
use bytes::Bytes;
use futures::StreamExt;
use tokio_stream::wrappers::UnboundedReceiverStream;

/// Bridge a reqwest `bytes_stream()` into an axum Body that streams chunks
/// unchanged. Spawns one tokio task per stream so the handler can return
/// immediately. The channel is unbounded to avoid deadlock under backpressure.
pub fn bytes_stream_to_body<S>(stream: S) -> Body
where
    S: futures::Stream<Item = reqwest::Result<Bytes>> + Send + Unpin + 'static,
{
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    tokio::spawn(async move {
        let mut s = stream;
        while let Some(chunk) = s.next().await {
            match chunk {
                Ok(b) => {
                    if tx.send(Ok::<_, std::io::Error>(b)).is_err() {
                        break;
                    }
                }
                Err(e) => {
                    let _ = tx.send(Err(std::io::Error::other(e)));
                    break;
                }
            }
        }
    });
    Body::from_stream(UnboundedReceiverStream::new(rx))
}
