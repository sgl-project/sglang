// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! A stand-in for SGLang's `ZmqEventPublisher`: a PUB socket for live batches
//! and a ROUTER that answers replay requests from a buffer, speaking the same
//! frames (`[topic][seq][payload]` live; `[identity][empty][start]` request;
//! `[identity][empty][seq][payload]` replies ending in `[identity][empty][-1][empty]`).

use std::sync::{Arc, Mutex};

use bytes::Bytes;
use rmpv::Value;
use zeromq::{PubSocket, RouterSocket, Socket, SocketRecv, SocketSend, ZmqMessage};

type ReplayBuffer = Arc<Mutex<Vec<(u64, Vec<u8>)>>>;

pub struct FakePublisher {
    pub pub_endpoint: String,
    pub replay_endpoint: String,
    /// Start sequences the bridge asked to replay from, in order.
    pub replay_requests: Arc<Mutex<Vec<u64>>>,
    topic: Vec<u8>,
    publisher: PubSocket,
    buffer: ReplayBuffer,
    replay_task: tokio::task::JoinHandle<()>,
}

impl FakePublisher {
    pub async fn bind(topic: &str) -> Self {
        let mut publisher = PubSocket::new();
        let pub_endpoint = publisher
            .bind("tcp://127.0.0.1:0")
            .await
            .unwrap()
            .to_string();
        let mut router = RouterSocket::new();
        let replay_endpoint = router.bind("tcp://127.0.0.1:0").await.unwrap().to_string();
        let buffer: ReplayBuffer = Arc::new(Mutex::new(Vec::new()));
        let served = Arc::clone(&buffer);
        let replay_requests = Arc::new(Mutex::new(Vec::new()));
        let requests = Arc::clone(&replay_requests);
        let replay_task = tokio::spawn(async move {
            loop {
                let Ok(request) = router.recv().await else {
                    return;
                };
                let frames = request.into_vec();
                // [identity][empty][start]
                if frames.len() != 3 || frames[2].len() != 8 {
                    continue;
                }
                let identity = frames[0].clone();
                let start = u64::from_be_bytes(frames[2].as_ref().try_into().unwrap());
                requests.lock().unwrap().push(start);
                let entries: Vec<(u64, Vec<u8>)> = served
                    .lock()
                    .unwrap()
                    .iter()
                    .filter(|(seq, _)| *seq >= start)
                    .cloned()
                    .collect();
                for (seq, payload) in entries {
                    let reply = ZmqMessage::try_from(vec![
                        identity.clone(),
                        Bytes::new(),
                        Bytes::copy_from_slice(&seq.to_be_bytes()),
                        Bytes::from(payload),
                    ])
                    .unwrap();
                    if router.send(reply).await.is_err() {
                        return;
                    }
                }
                let end = ZmqMessage::try_from(vec![
                    identity,
                    Bytes::new(),
                    Bytes::copy_from_slice(&(-1i64).to_be_bytes()),
                    Bytes::new(),
                ])
                .unwrap();
                if router.send(end).await.is_err() {
                    return;
                }
            }
        });
        Self {
            pub_endpoint,
            replay_endpoint,
            replay_requests,
            topic: topic.as_bytes().to_vec(),
            publisher,
            buffer,
            replay_task,
        }
    }

    /// Publishes live and remembers the batch for replay.
    pub async fn publish(&mut self, seq: u64, payload: Vec<u8>) {
        self.buffer.lock().unwrap().push((seq, payload.clone()));
        let message = ZmqMessage::try_from(vec![
            Bytes::from(self.topic.clone()),
            Bytes::copy_from_slice(&seq.to_be_bytes()),
            Bytes::from(payload),
        ])
        .unwrap();
        self.publisher.send(message).await.unwrap();
    }

    /// Remembers a batch as if it had been published while nobody listened.
    pub fn buffer_only(&self, seq: u64, payload: Vec<u8>) {
        self.buffer.lock().unwrap().push((seq, payload));
    }
}

impl Drop for FakePublisher {
    fn drop(&mut self) {
        self.replay_task.abort();
    }
}

// ---- msgpack fixtures in SGLang's KVEventBatch shape -------------------------

fn ints(values: &[i64]) -> Value {
    Value::Array(values.iter().map(|v| Value::from(*v)).collect())
}

fn map_event(entries: Vec<(&str, Value)>) -> Value {
    Value::Map(
        entries
            .into_iter()
            .map(|(key, value)| (Value::String(key.into()), value))
            .collect(),
    )
}

/// A `BlockStored` on GPU with an optional parent.
pub fn stored(hashes: &[i64], parent: Option<i64>) -> Value {
    map_event(vec![
        ("type", Value::String("BlockStored".into())),
        ("block_hashes", ints(hashes)),
        (
            "parent_block_hash",
            parent.map(Value::from).unwrap_or(Value::Nil),
        ),
        ("token_ids", Value::Array(Vec::new())),
        ("block_size", Value::from(16_i64)),
        ("lora_id", Value::Nil),
        ("medium", Value::String("GPU".into())),
    ])
}

pub fn removed(hashes: &[i64]) -> Value {
    map_event(vec![
        ("type", Value::String("BlockRemoved".into())),
        ("block_hashes", ints(hashes)),
        ("medium", Value::String("GPU".into())),
    ])
}

/// `[ts, events, attn_dp_rank]`, msgpack-encoded.
pub fn batch(events: Vec<Value>) -> Vec<u8> {
    let value = Value::Array(vec![
        Value::from(1.0_f64),
        Value::Array(events),
        Value::from(0_i64),
    ]);
    let mut buf = Vec::new();
    rmpv::encode::write_value(&mut buf, &value).unwrap();
    buf
}
