//! Replay client for SGLang's engine-side `ZmqEventPublisher`.
//!
//! Python binds a ROUTER socket for replay and expects DEALER clients to send
//! `[empty_delim, start_seq_bytes]`. The ROUTER observes
//! `[client_id, empty_delim, start_seq_bytes]` and replies with one message per
//! buffered batch: `[empty_delim, seq_bytes, payload]`, terminated by
//! `[empty_delim, END_SEQ, b""]`.

use std::time::Duration;

use bytes::Bytes;
use thiserror::Error;
use tokio::time::timeout;
use zeromq::{DealerSocket, Socket, SocketRecv, SocketSend, ZmqMessage};

use super::wire::{decode_event_batch, DecodeError, KvEventBatch};

/// Python `ZmqEventPublisher.END_SEQ = (-1).to_bytes(8, "big", signed=True)`.
const END_SEQ_SENTINEL: i64 = -1;

/// Default end-to-end ceiling for a single replay request.
pub(crate) const DEFAULT_REPLAY_TIMEOUT: Duration = Duration::from_secs(2);

/// Hard cap even if the worker advertises a very large replay buffer.
const HARD_REPLAY_BATCH_LIMIT: usize = 100_000;

#[derive(Debug, Error)]
pub(crate) enum ReplayError {
    #[error("replay request to {endpoint} timed out after {timeout:?}")]
    Timeout { endpoint: String, timeout: Duration },
    #[error("replay ZMQ error at {endpoint}: {source}")]
    Zmq {
        endpoint: String,
        source: zeromq::ZmqError,
    },
    #[error("replay reply has {got} frames, expected 3")]
    InvalidFrameCount { got: usize },
    #[error("replay reply delimiter frame is not empty")]
    InvalidDelimiter,
    #[error("replay reply seq frame has {len} bytes, expected 8")]
    InvalidSeqLength { len: usize },
    #[error("replay END_SEQ frame carried {len} payload bytes, expected 0")]
    InvalidEndPayload { len: usize },
    #[error("replay payload decode failed at seq {seq}: {source}")]
    Decode { seq: i64, source: DecodeError },
    #[error("replay response exceeded batch limit {limit} before END_SEQ")]
    BatchLimitExceeded { limit: usize },
}

/// Fetch replay batches starting at `start_seq` from one per-rank replay
/// endpoint. The returned vector excludes the terminating END_SEQ sentinel.
pub(crate) async fn fetch_replay_batches(
    endpoint: &str,
    start_seq: i64,
    buffer_steps: usize,
) -> Result<Vec<(i64, KvEventBatch)>, ReplayError> {
    fetch_replay_batches_with_timeout(endpoint, start_seq, buffer_steps, DEFAULT_REPLAY_TIMEOUT)
        .await
}

pub(crate) async fn fetch_replay_batches_with_timeout(
    endpoint: &str,
    start_seq: i64,
    buffer_steps: usize,
    timeout_duration: Duration,
) -> Result<Vec<(i64, KvEventBatch)>, ReplayError> {
    let endpoint_owned = endpoint.to_owned();
    timeout(
        timeout_duration,
        fetch_replay_batches_inner(endpoint, start_seq, buffer_steps),
    )
    .await
    .map_err(|_| ReplayError::Timeout {
        endpoint: endpoint_owned,
        timeout: timeout_duration,
    })?
}

async fn fetch_replay_batches_inner(
    endpoint: &str,
    start_seq: i64,
    buffer_steps: usize,
) -> Result<Vec<(i64, KvEventBatch)>, ReplayError> {
    let limit = buffer_steps.clamp(1, HARD_REPLAY_BATCH_LIMIT);
    let mut sock = DealerSocket::new();
    sock.connect(endpoint)
        .await
        .map_err(|source| ReplayError::Zmq {
            endpoint: endpoint.to_owned(),
            source,
        })?;

    let mut request = ZmqMessage::from(Bytes::new());
    request.push_back(Bytes::copy_from_slice(&start_seq.to_be_bytes()));
    sock.send(request)
        .await
        .map_err(|source| ReplayError::Zmq {
            endpoint: endpoint.to_owned(),
            source,
        })?;

    let mut batches = Vec::new();
    loop {
        if batches.len() >= limit {
            return Err(ReplayError::BatchLimitExceeded { limit });
        }
        let reply = sock.recv().await.map_err(|source| ReplayError::Zmq {
            endpoint: endpoint.to_owned(),
            source,
        })?;
        let (seq, payload) = parse_reply(reply)?;
        if seq == END_SEQ_SENTINEL {
            if !payload.is_empty() {
                return Err(ReplayError::InvalidEndPayload { len: payload.len() });
            }
            return Ok(batches);
        }
        let batch =
            decode_event_batch(&payload).map_err(|source| ReplayError::Decode { seq, source })?;
        batches.push((seq, batch));
    }
}

fn parse_reply(reply: ZmqMessage) -> Result<(i64, Bytes), ReplayError> {
    if reply.len() != 3 {
        return Err(ReplayError::InvalidFrameCount { got: reply.len() });
    }
    let mut frames = reply.into_vecdeque();
    let delimiter = frames.pop_front().expect("len checked");
    if !delimiter.is_empty() {
        return Err(ReplayError::InvalidDelimiter);
    }
    let seq_frame = frames.pop_front().expect("len checked");
    if seq_frame.len() != 8 {
        return Err(ReplayError::InvalidSeqLength {
            len: seq_frame.len(),
        });
    }
    let mut seq_bytes = [0_u8; 8];
    seq_bytes.copy_from_slice(&seq_frame);
    let seq = i64::from_be_bytes(seq_bytes);
    let payload = frames.pop_front().expect("len checked");
    Ok((seq, payload))
}

#[cfg(test)]
mod tests {
    use super::*;

    use rmp::encode as mp;
    use tokio::task::JoinHandle;
    use zeromq::{Endpoint, RouterSocket, SocketSend};

    fn encode_all_blocks_cleared_batch() -> Vec<u8> {
        let mut buf = Vec::new();
        mp::write_array_len(&mut buf, 3).unwrap();
        mp::write_f64(&mut buf, 0.0).unwrap();
        mp::write_array_len(&mut buf, 1).unwrap();
        mp::write_array_len(&mut buf, 1).unwrap();
        mp::write_str(&mut buf, "AllBlocksCleared").unwrap();
        mp::write_nil(&mut buf).unwrap();
        buf
    }

    async fn spawn_router<F>(serve: F) -> (String, JoinHandle<()>)
    where
        F: FnOnce(RouterSocket) -> JoinHandle<()> + Send + 'static,
    {
        let mut router = RouterSocket::new();
        let endpoint = router
            .bind("tcp://127.0.0.1:0")
            .await
            .expect("bind replay ROUTER");
        let port = match endpoint {
            Endpoint::Tcp(_, port) => port,
            other => panic!("unexpected endpoint: {other:?}"),
        };
        let handle = serve(router);
        (format!("tcp://127.0.0.1:{port}"), handle)
    }

    fn reply(identity: Bytes, seq: i64, payload: Vec<u8>) -> ZmqMessage {
        let mut msg = ZmqMessage::from(identity);
        msg.push_back(Bytes::new());
        msg.push_back(Bytes::copy_from_slice(&seq.to_be_bytes()));
        msg.push_back(Bytes::from(payload));
        msg
    }

    async fn recv_request(router: &mut RouterSocket, expected_start: i64) -> Bytes {
        let request = router.recv().await.expect("receive replay request");
        assert_eq!(request.len(), 3);
        let identity = request.get(0).expect("identity").clone();
        assert!(request.get(1).expect("delimiter").is_empty());
        assert_eq!(
            request.get(2).expect("start seq").as_ref(),
            &expected_start.to_be_bytes()
        );
        identity
    }

    #[tokio::test]
    async fn sends_dealer_request_and_reads_batches_until_end() {
        let payload1 = encode_all_blocks_cleared_batch();
        let payload2 = encode_all_blocks_cleared_batch();
        let (endpoint, server) = spawn_router(move |mut router| {
            tokio::spawn(async move {
                let identity = recv_request(&mut router, 7).await;
                router
                    .send(reply(identity.clone(), 7, payload1))
                    .await
                    .expect("send seq 7");
                router
                    .send(reply(identity.clone(), 8, payload2))
                    .await
                    .expect("send seq 8");
                router
                    .send(reply(identity, END_SEQ_SENTINEL, Vec::new()))
                    .await
                    .expect("send END");
            })
        })
        .await;

        let batches = fetch_replay_batches_with_timeout(&endpoint, 7, 16, Duration::from_secs(1))
            .await
            .expect("replay succeeds");
        server.await.expect("server task");
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].0, 7);
        assert_eq!(batches[1].0, 8);
    }

    #[tokio::test]
    async fn end_only_is_successful_empty_replay() {
        let (endpoint, server) = spawn_router(|mut router| {
            tokio::spawn(async move {
                let identity = recv_request(&mut router, 9).await;
                router
                    .send(reply(identity, END_SEQ_SENTINEL, Vec::new()))
                    .await
                    .expect("send END");
            })
        })
        .await;

        let batches = fetch_replay_batches_with_timeout(&endpoint, 9, 16, Duration::from_secs(1))
            .await
            .expect("END-only replay is valid");
        server.await.expect("server task");
        assert!(batches.is_empty());
    }

    #[tokio::test]
    async fn invalid_frame_count_is_error() {
        let (endpoint, server) = spawn_router(|mut router| {
            tokio::spawn(async move {
                let identity = recv_request(&mut router, 1).await;
                let mut msg = ZmqMessage::from(identity);
                msg.push_back(Bytes::new());
                router.send(msg).await.expect("send short frame");
            })
        })
        .await;

        let err = fetch_replay_batches_with_timeout(&endpoint, 1, 16, Duration::from_secs(1))
            .await
            .expect_err("invalid frame count");
        server.await.expect("server task");
        assert!(matches!(err, ReplayError::InvalidFrameCount { got: 1 }));
    }

    #[tokio::test]
    async fn bad_seq_length_is_error() {
        let (endpoint, server) = spawn_router(|mut router| {
            tokio::spawn(async move {
                let identity = recv_request(&mut router, 1).await;
                let mut msg = ZmqMessage::from(identity);
                msg.push_back(Bytes::new());
                msg.push_back(Bytes::from_static(b"bad"));
                msg.push_back(Bytes::new());
                router.send(msg).await.expect("send bad seq");
            })
        })
        .await;

        let err = fetch_replay_batches_with_timeout(&endpoint, 1, 16, Duration::from_secs(1))
            .await
            .expect_err("bad seq length");
        server.await.expect("server task");
        assert!(matches!(err, ReplayError::InvalidSeqLength { len: 3 }));
    }

    #[tokio::test]
    async fn payload_decode_failure_is_error() {
        let (endpoint, server) = spawn_router(|mut router| {
            tokio::spawn(async move {
                let identity = recv_request(&mut router, 1).await;
                router
                    .send(reply(identity, 1, b"not-msgpack".to_vec()))
                    .await
                    .expect("send bad payload");
            })
        })
        .await;

        let err = fetch_replay_batches_with_timeout(&endpoint, 1, 16, Duration::from_secs(1))
            .await
            .expect_err("decode failure");
        server.await.expect("server task");
        assert!(matches!(err, ReplayError::Decode { seq: 1, .. }));
    }

    #[tokio::test]
    async fn missing_end_hits_timeout() {
        let (endpoint, server) = spawn_router(|mut router| {
            tokio::spawn(async move {
                let _identity = recv_request(&mut router, 1).await;
                std::future::pending::<()>().await;
            })
        })
        .await;

        let err = fetch_replay_batches_with_timeout(&endpoint, 1, 16, Duration::from_millis(50))
            .await
            .expect_err("timeout");
        server.abort();
        assert!(matches!(err, ReplayError::Timeout { .. }));
    }
}
