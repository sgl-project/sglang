// Copyright 2023-2024 SGLang Team
// Licensed under the Apache License, Version 2.0

//! Shared ZMQ IPC utilities: constants, AsyncFd wrapper, async recv/send tasks.

pub mod wire;

use log::error;
use std::os::unix::io::AsRawFd;
use tokio::io::unix::AsyncFd;
use tokio::sync::mpsc;

pub const MSGPACK_MAGIC: &[u8] = b"0xSG02";
pub const PICKLE_MAGIC: &[u8] = b"0xSG01";

// ────────────────────── AsyncFd wrapper for ZMQ ─────────────────────────────

/// Newtype that lets us register ZMQ's notification fd with tokio's epoll reactor.
pub(crate) struct ZmqNotifyFd(pub std::os::unix::io::RawFd);

impl AsRawFd for ZmqNotifyFd {
    fn as_raw_fd(&self) -> std::os::unix::io::RawFd {
        self.0
    }
}

// ─────────────────────── ZMQ multipart validation ───────────────────────────

pub fn validate_zmq_parts(parts: Vec<Vec<u8>>) -> Result<Vec<u8>, String> {
    if parts.len() == 1 {
        return Err("Received single-frame (pickle) message. Set SGLANG_IPC_USE_MSGPACK=1.".into());
    }
    let magic = &parts[0];
    if magic == PICKLE_MAGIC {
        return Err("Received pickle message. Set SGLANG_IPC_USE_MSGPACK=1.".into());
    }
    if magic != MSGPACK_MAGIC {
        return Err(format!("Unknown IPC magic: {magic:?}"));
    }
    Ok(parts[1].clone())
}

// ──────────────────────── ZMQ async recv task ───────────────────────────────

/// Receives multipart messages from a ZMQ PULL socket and forwards raw payloads
/// (magic-stripped) to `tx` without blocking a tokio worker thread.
pub async fn zmq_recv_task(sock: zmq::Socket, tx: mpsc::Sender<Vec<Vec<u8>>>) {
    let raw_fd = sock.get_fd().expect("zmq::Socket::get_fd failed");
    let async_fd = AsyncFd::new(ZmqNotifyFd(raw_fd)).expect("AsyncFd::new failed");

    'outer: loop {
        let mut guard = match async_fd.readable().await {
            Ok(g) => g,
            Err(e) => {
                error!("ZMQ recv poll error: {e}");
                break;
            }
        };
        guard.clear_ready();

        loop {
            match sock.recv_multipart(zmq::DONTWAIT) {
                Ok(parts) => {
                    if tx.send(parts).await.is_err() {
                        break 'outer;
                    }
                }
                Err(zmq::Error::EAGAIN) => break,
                Err(e) => {
                    error!("ZMQ recv error: {e}");
                    break 'outer;
                }
            }
        }
    }
}

// ──────────────────────── ZMQ async send task ───────────────────────────────

/// Sends pre-encoded msgpack payloads to a ZMQ PUSH socket.
/// Each payload is sent as a 2-frame message: `[MSGPACK_MAGIC, data]`.
pub async fn zmq_send_task(sock: zmq::Socket, mut rx: mpsc::Receiver<Vec<u8>>) {
    let raw_fd = sock.get_fd().expect("zmq::Socket::get_fd failed");
    let async_fd = AsyncFd::new(ZmqNotifyFd(raw_fd)).expect("AsyncFd::new failed");

    while let Some(data) = rx.recv().await {
        loop {
            match sock.send_multipart(&[MSGPACK_MAGIC, data.as_slice()], zmq::DONTWAIT) {
                Ok(()) => break,
                Err(zmq::Error::EAGAIN) => {
                    if let Ok(mut guard) = async_fd.writable().await {
                        guard.clear_ready();
                    }
                }
                Err(e) => {
                    error!("ZMQ send error: {e}");
                    return;
                }
            }
        }
    }
}
