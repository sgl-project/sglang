// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Shared ZMQ wire-format helpers for the `policies::kv_events` component
//! tests. Encodes events in the same msgspec layout SGLang emits, builds
//! the two-frame `[seq, payload]` ZMQ message a real publisher sends, and
//! binds a loopback PUB socket on an OS-assigned port.

#![allow(dead_code)]

use bytes::Bytes;
use rmp::encode as mp;
use zeromq::{Endpoint, PubSocket, Socket, ZmqMessage};

/// Bind a PUB socket to an OS-assigned 127.0.0.1 port. Returns
/// `(socket, port)`.
pub async fn make_pub_bound() -> (PubSocket, u16) {
    let mut sock = PubSocket::new();
    let endpoint = sock
        .bind("tcp://127.0.0.1:0")
        .await
        .expect("bind PUB socket");
    let port = match endpoint {
        Endpoint::Tcp(_, p) => p,
        other => panic!("unexpected endpoint: {other:?}"),
    };
    (sock, port)
}

/// Encode a single `BlockStored` event in the wire format msgspec
/// emits. Layout: `["BlockStored", block_hashes, parent, token_ids,
/// block_size, lora_id, medium]`.
pub fn encode_block_stored_event(
    block_hashes: &[i64],
    parent: Option<i64>,
    token_ids: &[u32],
    block_size: u32,
) -> Vec<u8> {
    let mut buf = Vec::new();
    mp::write_array_len(&mut buf, 7).unwrap();
    mp::write_str(&mut buf, "BlockStored").unwrap();
    mp::write_array_len(&mut buf, block_hashes.len() as u32).unwrap();
    for v in block_hashes {
        mp::write_sint(&mut buf, *v).unwrap();
    }
    match parent {
        Some(v) => {
            mp::write_sint(&mut buf, v).unwrap();
        }
        None => mp::write_nil(&mut buf).unwrap(),
    }
    mp::write_array_len(&mut buf, token_ids.len() as u32).unwrap();
    for v in token_ids {
        mp::write_uint(&mut buf, *v as u64).unwrap();
    }
    mp::write_uint(&mut buf, block_size as u64).unwrap();
    mp::write_nil(&mut buf).unwrap(); // lora_id
    mp::write_str(&mut buf, "GPU").unwrap();
    buf
}

/// Wrap one or more pre-encoded events into a KVEventBatch with
/// timestamp + optional dp-rank.
pub fn encode_event_batch(ts: f64, events: Vec<Vec<u8>>, attn_dp_rank: Option<u32>) -> Vec<u8> {
    let mut buf = Vec::new();
    mp::write_array_len(&mut buf, 3).unwrap();
    mp::write_f64(&mut buf, ts).unwrap();
    mp::write_array_len(&mut buf, events.len() as u32).unwrap();
    for ev in events {
        buf.extend_from_slice(&ev);
    }
    match attn_dp_rank {
        Some(v) => {
            mp::write_uint(&mut buf, v as u64).unwrap();
        }
        None => mp::write_nil(&mut buf).unwrap(),
    }
    buf
}

/// Build the two-frame ZMQ message a real KV publisher sends:
/// `[seq (big-endian i64), payload]`.
pub fn build_multipart(seq: i64, payload: Vec<u8>) -> ZmqMessage {
    let mut msg = ZmqMessage::from(Bytes::new());
    msg.push_back(Bytes::copy_from_slice(&seq.to_be_bytes()));
    msg.push_back(Bytes::from(payload));
    msg
}
