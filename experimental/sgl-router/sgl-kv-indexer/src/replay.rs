// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Bounded epoch-fenced client for the Worker's replay-v2 ZMQ endpoint.

use std::time::Duration;

use bytes::Bytes;
use thiserror::Error;
use zeromq::{DealerSocket, Socket, SocketRecv, SocketSend, ZmqMessage};

const COMMAND: &[u8] = b"replay-v2";
const END_SEQ: i64 = -1;
const MAX_REPLAY_BATCHES: usize = 10_000;
const TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplayedBatch {
    pub seq: u64,
    pub payload: Vec<u8>,
}

#[derive(Debug, Error)]
pub enum ReplayError {
    #[error("replay request timed out")]
    Timeout,
    #[error("replay ZMQ error: {0}")]
    Zmq(#[from] zeromq::ZmqError),
    #[error("invalid replay response: {0}")]
    Invalid(String),
}

pub async fn fetch_replay(
    endpoint: &str,
    epoch: &str,
    start_seq: u64,
    end_seq: u64,
) -> Result<Vec<ReplayedBatch>, ReplayError> {
    tokio::time::timeout(
        TIMEOUT,
        fetch_replay_inner(endpoint, epoch, start_seq, end_seq),
    )
    .await
    .map_err(|_| ReplayError::Timeout)?
}

async fn fetch_replay_inner(
    endpoint: &str,
    epoch: &str,
    start_seq: u64,
    end_seq: u64,
) -> Result<Vec<ReplayedBatch>, ReplayError> {
    if end_seq < start_seq {
        return Err(ReplayError::Invalid("end sequence precedes start".into()));
    }
    if end_seq.saturating_sub(start_seq) as usize > MAX_REPLAY_BATCHES {
        return Err(ReplayError::Invalid(
            "requested replay range is too large".into(),
        ));
    }

    let mut socket = DealerSocket::new();
    socket.connect(endpoint).await?;
    let mut request = ZmqMessage::from(Bytes::new());
    request.push_back(Bytes::from_static(COMMAND));
    request.push_back(Bytes::copy_from_slice(epoch.as_bytes()));
    request.push_back(Bytes::copy_from_slice(&start_seq.to_be_bytes()));
    request.push_back(Bytes::copy_from_slice(&end_seq.to_be_bytes()));
    socket.send(request).await?;

    let mut expected = start_seq;
    let mut batches = Vec::with_capacity(end_seq.saturating_sub(start_seq) as usize);
    loop {
        let message = socket.recv().await?;
        let frames = message.into_vec();
        if frames.len() != 4 || !frames[0].is_empty() {
            return Err(ReplayError::Invalid(
                "expected [empty, epoch, sequence, payload]".into(),
            ));
        }
        if frames[1].as_ref() != epoch.as_bytes() {
            return Err(ReplayError::Invalid("worker epoch changed".into()));
        }
        let seq = i64::from_be_bytes(
            frames[2]
                .as_ref()
                .try_into()
                .map_err(|_| ReplayError::Invalid("sequence must be 8 bytes".into()))?,
        );
        if seq == END_SEQ {
            break;
        }
        let seq = u64::try_from(seq)
            .map_err(|_| ReplayError::Invalid("negative replay sequence".into()))?;
        if seq != expected || seq >= end_seq {
            return Err(ReplayError::Invalid(format!(
                "replay discontinuity: expected {expected}, got {seq}"
            )));
        }
        batches.push(ReplayedBatch {
            seq,
            payload: frames[3].to_vec(),
        });
        expected = expected.saturating_add(1);
    }
    if expected != end_seq {
        return Err(ReplayError::Invalid(format!(
            "replay window unavailable: stopped at {expected}, need {end_seq}"
        )));
    }
    Ok(batches)
}

#[cfg(test)]
mod tests {
    use super::*;
    use zeromq::{Endpoint, RouterSocket};

    async fn send_reply(
        router: &mut RouterSocket,
        identity: Bytes,
        epoch: &'static [u8],
        seq: i64,
        payload: &'static [u8],
    ) {
        let mut response = ZmqMessage::from(identity);
        response.push_back(Bytes::new());
        response.push_back(Bytes::from_static(epoch));
        response.push_back(Bytes::copy_from_slice(&seq.to_be_bytes()));
        response.push_back(Bytes::from_static(payload));
        router.send(response).await.unwrap();
    }

    #[tokio::test]
    async fn fetches_exact_bounded_replay_window() {
        let mut router = RouterSocket::new();
        let endpoint = router.bind("tcp://127.0.0.1:0").await.unwrap();
        let port = match endpoint {
            Endpoint::Tcp(_, port) => port,
            other => panic!("unexpected endpoint: {other:?}"),
        };
        let server = tokio::spawn(async move {
            let request = router.recv().await.unwrap().into_vec();
            assert_eq!(request.len(), 6);
            assert_eq!(request[2].as_ref(), COMMAND);
            assert_eq!(request[3].as_ref(), b"epoch");
            let identity = request[0].clone();
            send_reply(&mut router, identity.clone(), b"epoch", 3, b"a").await;
            send_reply(&mut router, identity.clone(), b"epoch", 4, b"b").await;
            send_reply(&mut router, identity, b"epoch", END_SEQ, b"").await;
        });

        let batches = fetch_replay(&format!("tcp://127.0.0.1:{port}"), "epoch", 3, 5)
            .await
            .unwrap();
        assert_eq!(
            batches,
            vec![
                ReplayedBatch {
                    seq: 3,
                    payload: b"a".to_vec(),
                },
                ReplayedBatch {
                    seq: 4,
                    payload: b"b".to_vec(),
                },
            ]
        );
        server.await.unwrap();
    }

    #[tokio::test]
    async fn rejects_replay_window_that_no_longer_contains_start() {
        let mut router = RouterSocket::new();
        let endpoint = router.bind("tcp://127.0.0.1:0").await.unwrap();
        let port = match endpoint {
            Endpoint::Tcp(_, port) => port,
            other => panic!("unexpected endpoint: {other:?}"),
        };
        tokio::spawn(async move {
            let request = router.recv().await.unwrap().into_vec();
            let identity = request[0].clone();
            send_reply(&mut router, identity.clone(), b"epoch", 4, b"late").await;
            send_reply(&mut router, identity, b"epoch", END_SEQ, b"").await;
        });

        let error = fetch_replay(&format!("tcp://127.0.0.1:{port}"), "epoch", 3, 5)
            .await
            .unwrap_err();
        assert!(matches!(error, ReplayError::Invalid(_)));
    }
}
