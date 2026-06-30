//! TCP wire transport between a standalone api-server process and a headless
//! sglang-server (per DP rank). Used when `dp_size > 1`; the in-process path
//! (flume + `EgressSink::Local`) is unchanged.
//!
//! Framing is length-delimited and direction-agnostic so the same codec serves
//! both connections (api→rank ingress, rank→api egress):
//! ```text
//! [u32 BE total_len][u32 BE meta_len][meta: rmp_serde(Frame)][tail bytes]
//! ```
//! `meta` is the structured part (msgpack, field-named for schema evolution);
//! `tail` is an opaque raw segment the caller interprets per variant — for a
//! `Generate` frame it's the request's `input_ids` as raw i64-LE, kept *out* of
//! msgpack so the tensor never pays serialization cost (same columnar trick as
//! the in-process ring). Other variants carry an empty tail.
//!
//! Both ends are the same build (one pyo3 extension), so the wire types may
//! reuse internal structs directly; `WireEgress` exists only to carry `Error`
//! as `{message,status}` rather than the internal enum.

// Scaffolding: the codec + wire types land here first (Step 2); the api-server
// `NetTransport` client (Step 4) consumes the rest.
#![allow(dead_code)]

mod client;
mod server;
pub use client::NetClient;
pub use server::serve_headless;

use bytes::Bytes;

/// Handshake a client sends right after connect: `role` selects the direction
/// (so one rank listener serves both); `index` is the connection's pool slot —
/// ignored for ingress (frames carry their own `rid`), but for egress it's the
/// **shard** the rank routes `rid % egress_pool` frames to, so a request's
/// ordered egress always rides one connection.
///
/// Wire form is `[role: u8][index: u16-BE]` (3 bytes). `role` is a small enum
/// (`u8` leaves ample room for future directions); the `u16` index covers any
/// pool size (we cap well below that for FDs).
///
/// Pool sizes are not constant: requests round-robin onto `ingress_pool` write
/// connections and shard by `rid` onto `egress_pool` egress connections. The
/// rank (bind) and the api-server (connect) derive `egress_pool` from the same
/// `ServerArgs::egress_pool_size`, so the shard count agrees without negotiation
/// (`ingress_pool` is the api-server's choice — the rank accepts any number).
pub(crate) const ROLE_INGRESS: u8 = 0;
pub(crate) const ROLE_EGRESS: u8 = 1;

/// Encode the connection handshake `[role: u8][index: u16-BE]` (3 bytes).
pub(crate) fn handshake(role: u8, index: usize) -> [u8; 3] {
    let i = (index as u16).to_be_bytes();
    [role, i[0], i[1]]
}

/// Decode the 3-byte handshake header into `(role, index)`.
pub(crate) fn parse_handshake(buf: [u8; 3]) -> (u8, usize) {
    (buf[0], u16::from_be_bytes([buf[1], buf[2]]) as usize)
}

use serde::{Deserialize, Serialize};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

use crate::error::Error;
use crate::message::{EgressItem, GenerationOutput};

/// A single framed message. Variants are grouped by direction but share one
/// codec; each physical connection only ever carries one direction's variants.
/// `Clone` so a control frame can be broadcast to every rank.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Frame {
    // --- api-server → rank (ingress connection) ---
    /// A generate request. `input_ids` (when present) rides in the frame tail as
    /// raw i64-LE, not in this struct. `sampling_params` is the un-normalized
    /// rmpv map; the rank's tm-ingress normalizes it (the `Normalizing` step).
    Generate {
        rid: u64,
        text: Option<String>,
        sampling_params: Option<rmpv::Value>,
        stream: bool,
    },
    /// A control request (e.g. `/server_info`); routed without tokenization.
    Control { rid: u64, tag: String },

    // --- rank → api-server (egress connection) ---
    /// One generation/control/error frame for a request, keyed by `rid`.
    Egress { rid: u64, item: WireEgress },
    /// Periodic load telemetry (piggybacked on the egress connection) the
    /// api-server's router consumes for load-aware routing.
    Load(LoadStats),
}

/// Serde-friendly mirror of [`EgressItem`]: identical except `Error` is carried
/// as `{message,status}` so the internal error enum stays off the wire.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WireEgress {
    Frame(GenerationOutput),
    Done(GenerationOutput),
    Control(Vec<u8>),
    Error { message: String, status: u16 },
}

impl From<EgressItem> for WireEgress {
    fn from(it: EgressItem) -> Self {
        match it {
            EgressItem::Frame(o) => WireEgress::Frame(o),
            EgressItem::Done(o) => WireEgress::Done(o),
            EgressItem::Control(b) => WireEgress::Control(b.to_vec()),
            EgressItem::Error(e) => WireEgress::Error {
                message: e.to_string(),
                status: e.http_status(),
            },
        }
    }
}

impl From<WireEgress> for EgressItem {
    fn from(w: WireEgress) -> Self {
        match w {
            WireEgress::Frame(o) => EgressItem::Frame(o),
            WireEgress::Done(o) => EgressItem::Done(o),
            WireEgress::Control(b) => EgressItem::Control(Bytes::from(b)),
            WireEgress::Error { message, status } => {
                EgressItem::Error(Error::Remote { message, status })
            }
        }
    }
}

/// Per-rank load snapshot for routing. Intentionally small; extend as the
/// router (Level 1) starts consuming richer signals (KV/cache utilization).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LoadStats {
    /// Requests currently running on this rank's scheduler.
    pub num_running: u32,
    /// Requests waiting in this rank's queue.
    pub num_waiting: u32,
    /// KV-cache utilization in `[0, 1]`.
    pub kv_usage: f32,
}

/// Cap on a single frame's declared length (32 MiB) — guards against a corrupt
/// length header allocating unboundedly.
const MAX_FRAME_LEN: usize = 32 << 20;

fn invalid(msg: impl std::fmt::Display) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, msg.to_string())
}

/// Write one frame: `meta` (msgpack) followed by the opaque `tail` bytes.
pub async fn write_frame<W: AsyncWrite + Unpin>(
    w: &mut W,
    frame: &Frame,
    tail: &[u8],
) -> std::io::Result<()> {
    let meta = rmp_serde::to_vec_named(frame).map_err(invalid)?;
    let total = 4 + meta.len() + tail.len();
    if total > MAX_FRAME_LEN {
        return Err(invalid(format!("frame too large: {total}")));
    }
    w.write_u32(total as u32).await?;
    w.write_u32(meta.len() as u32).await?;
    w.write_all(&meta).await?;
    w.write_all(tail).await?;
    w.flush().await?;
    Ok(())
}

/// Read one frame: returns the decoded [`Frame`] and its raw tail bytes.
pub async fn read_frame<R: AsyncRead + Unpin>(r: &mut R) -> std::io::Result<(Frame, Bytes)> {
    let total = r.read_u32().await? as usize;
    if !(4..=MAX_FRAME_LEN).contains(&total) {
        return Err(invalid(format!("bad frame length: {total}")));
    }
    let mut buf = vec![0u8; total];
    r.read_exact(&mut buf).await?;
    let meta_len = u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;
    let meta_end = 4usize
        .checked_add(meta_len)
        .filter(|&e| e <= buf.len())
        .ok_or_else(|| invalid("meta_len exceeds frame"))?;
    let frame: Frame = rmp_serde::from_slice(&buf[4..meta_end]).map_err(invalid)?;
    let tail = Bytes::copy_from_slice(&buf[meta_end..]);
    Ok((frame, tail))
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn roundtrip(frame: Frame, tail: &[u8]) -> (Frame, Bytes) {
        // 64 KiB in-memory duplex pipe stands in for a TcpStream.
        let (mut a, mut b) = tokio::io::duplex(64 * 1024);
        write_frame(&mut a, &frame, tail).await.unwrap();
        read_frame(&mut b).await.unwrap()
    }

    #[tokio::test]
    async fn generate_frame_with_ids_tail() {
        let ids: Vec<u8> = [7i64, 8, 9].iter().flat_map(|v| v.to_le_bytes()).collect();
        let frame = Frame::Generate {
            rid: 42,
            text: Some("hello".into()),
            sampling_params: None,
            stream: true,
        };
        let (got, tail) = roundtrip(frame, &ids).await;
        match got {
            Frame::Generate {
                rid, text, stream, ..
            } => {
                assert_eq!(rid, 42);
                assert_eq!(text.as_deref(), Some("hello"));
                assert!(stream);
            }
            _ => panic!("wrong variant"),
        }
        assert_eq!(&tail[..], &ids[..]);
    }

    #[tokio::test]
    async fn egress_done_roundtrip_and_conversion() {
        let out = GenerationOutput {
            rid: "5".into(),
            text: "Paris".into(),
            prompt_tokens: 3,
            completion_tokens: 1,
            finish_reason: Some("stop".into()),
            ..Default::default()
        };
        let item: WireEgress = EgressItem::Done(out.clone()).into();
        let (got, tail) = roundtrip(Frame::Egress { rid: 5, item }, &[]).await;
        assert!(tail.is_empty());
        let Frame::Egress { rid, item } = got else {
            panic!("wrong variant")
        };
        assert_eq!(rid, 5);
        // Round-trips back into an EgressItem the handlers already understand.
        match EgressItem::from(item) {
            EgressItem::Done(o) => {
                assert_eq!(o.text, "Paris");
                assert_eq!(o.finish_reason.as_deref(), Some("stop"));
            }
            _ => panic!("wrong egress kind"),
        }
    }

    #[tokio::test]
    async fn error_carries_status_across_wire() {
        let item: WireEgress = EgressItem::Error(Error::Validation("bad top_p".into())).into();
        let (got, _) = roundtrip(Frame::Egress { rid: 1, item }, &[]).await;
        let Frame::Egress { item, .. } = got else {
            panic!()
        };
        match EgressItem::from(item) {
            EgressItem::Error(e) => {
                assert_eq!(e.http_status(), 400);
                // Display is preserved verbatim across the wire (incl. the
                // `Validation` prefix), now carried by `Error::Remote`.
                assert_eq!(e.to_string(), "validation failed: bad top_p");
            }
            _ => panic!("wrong egress kind"),
        }
    }

    #[tokio::test]
    async fn load_frame_roundtrip() {
        let stats = LoadStats {
            num_running: 12,
            num_waiting: 3,
            kv_usage: 0.5,
        };
        let (got, _) = roundtrip(Frame::Load(stats.clone()), &[]).await;
        match got {
            Frame::Load(s) => {
                assert_eq!(s.num_running, 12);
                assert_eq!(s.num_waiting, 3);
            }
            _ => panic!("wrong variant"),
        }
    }
}
