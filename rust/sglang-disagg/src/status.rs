//! Common control-plane frames (`common/conn.py`): KV_STATUS status
//! messages, ABORT / ABORT_ACK, BOOTSTRAP (deferred bootstrap), CHUNK_READY
//! (staging). All ASCII integer frames plus one utf-8 reason frame.
//!
//! KV_STATUS is flavor-parameterized (`CommonKVManager.kv_status_msg_tag` /
//! `kv_status_msg_carries_reason`):
//! - Mooncake / Ascend: untagged `[room, status, prefill_rank]`, no reason.
//! - NIXL: `[b"KV_STATUS", room, status, prefill_rank, reason]`.
//! - Mori: `[b"MoriMsgGuard", room, status, prefill_rank, reason]`.

use crate::pack::py_int;
use crate::{KVPoll, Room};

/// ABORT frame 0 tag (common/conn.py:75).
pub const ABORT_TAG: &[u8] = b"ABORT";
/// ABORT_ACK frame 0 tag (common/conn.py:76).
pub const ABORT_ACK_TAG: &[u8] = b"ABORT_ACK";
/// KV_STATUS frame 0 tag on the NIXL path (nixl/conn.py:515).
pub const KV_STATUS_TAG_NIXL: &[u8] = b"KV_STATUS";
/// KV_STATUS frame 0 tag on the Mori path (mori/conn.py:381).
pub const KV_STATUS_TAG_MORI: &[u8] = b"MoriMsgGuard";
/// BOOTSTRAP frame 0 tag (common/conn.py:695).
pub const BOOTSTRAP_TAG: &[u8] = b"BOOTSTRAP";
/// CHUNK_READY frame 0 tag (common/conn.py:820).
pub const CHUNK_READY_TAG: &[u8] = b"CHUNK_READY";

// --------------------------------------------------------------------------- //
// KV_STATUS (common/conn.py:735-776)
// --------------------------------------------------------------------------- //

/// `_encode_kv_status_message`: `[room, int(status), prefill_rank]` ascii,
/// optional reason frame (utf-8) when the flavor carries one, tag inserted
/// at 0.
pub fn encode_kv_status(
    room: Room,
    status: KVPoll,
    prefill_rank: i64,
    failure_reason: Option<&str>,
    tag: Option<&[u8]>,
    carries_reason: bool,
) -> Vec<Vec<u8>> {
    let mut parts = vec![
        room.to_string().into_bytes(),
        (status as i32).to_string().into_bytes(),
        prefill_rank.to_string().into_bytes(),
    ];
    if carries_reason {
        parts.push(failure_reason.unwrap_or("").as_bytes().to_vec());
    }
    if let Some(t) = tag {
        parts.insert(0, t.to_vec());
    }
    parts
}

/// `parse_kv_status_message`: None when the tag mismatches, fewer than 3
/// frames, or frames 0-2 don't int-decode. The reason frame decodes utf-8
/// with errors="replace" — Rust's from_utf8_lossy replaces per invalid
/// SEQUENCE (Python: per BYTE), differing only on truncated multibyte input,
/// which cannot occur from a conforming peer.
///
/// Returns `(room, status, prefill_rank, failure_reason)`.
pub fn parse_kv_status(
    msg: &[Vec<u8>],
    tag: Option<&[u8]>,
) -> Option<(Room, i32, i64, Option<String>)> {
    let body: &[Vec<u8>] = match tag {
        Some(t) => {
            if msg.first().map(Vec::as_slice) != Some(t) {
                return None;
            }
            &msg[1..]
        }
        None => msg,
    };
    if body.len() < 3 {
        return None;
    }
    let room = py_int(&body[0]).ok()?;
    let status = py_int(&body[1]).ok()? as i32;
    let prefill_rank = py_int(&body[2]).ok()?;
    let reason = body
        .get(3)
        .filter(|b| !b.is_empty())
        .map(|b| String::from_utf8_lossy(b).into_owned());
    Some((room, status, prefill_rank, reason))
}

// --------------------------------------------------------------------------- //
// ABORT (AbortNotification, common/conn.py:182-233)
// --------------------------------------------------------------------------- //

/// Parsed `AbortNotification`. Mirrors the dataclass: the return address is
/// only present when BOTH ip and port decoded; generation only then.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AbortNotification {
    pub room: Room,
    pub decode_ip: Option<String>,
    pub decode_port: Option<i64>,
    pub generation: Option<i64>,
}

/// `AbortNotification.to_zmq`: `[b"ABORT", room]`; the return address frames
/// are appended only when ip AND port are both set, generation only inside
/// that branch.
pub fn build_abort(n: &AbortNotification) -> Vec<Vec<u8>> {
    let mut frames = vec![ABORT_TAG.to_vec(), n.room.to_string().into_bytes()];
    if let (Some(ip), Some(port)) = (&n.decode_ip, n.decode_port) {
        frames.push(ip.as_bytes().to_vec());
        frames.push(port.to_string().into_bytes());
        if let Some(g) = n.generation {
            frames.push(g.to_string().into_bytes());
        }
    }
    frames
}

/// `AbortNotification.from_zmq`. `msg` INCLUDES the tag frame (the parser
/// never validates it — the dispatcher does). Malformed fields degrade
/// instead of failing: bad room -> None; missing/bad return address ->
/// room only; bad generation -> kept as None.
pub fn parse_abort(msg: &[Vec<u8>]) -> Option<AbortNotification> {
    if msg.len() < 2 {
        return None;
    }
    let room = py_int(&msg[1]).ok()?;
    if msg.len() < 4 {
        return Some(AbortNotification {
            room,
            decode_ip: None,
            decode_port: None,
            generation: None,
        });
    }
    let addr = std::str::from_utf8(&msg[2])
        .ok()
        .and_then(|ip| py_int(&msg[3]).ok().map(|port| (ip.to_string(), port)));
    let (decode_ip, decode_port) = match addr {
        Some((ip, port)) => (Some(ip), Some(port)),
        None => {
            return Some(AbortNotification {
                room,
                decode_ip: None,
                decode_port: None,
                generation: None,
            });
        }
    };
    let generation = if msg.len() >= 5 {
        py_int(&msg[4]).ok()
    } else {
        None
    };
    Some(AbortNotification {
        room,
        decode_ip,
        decode_port,
        generation,
    })
}

// --------------------------------------------------------------------------- //
// ABORT_ACK (AbortAck, common/conn.py:236-280)
// --------------------------------------------------------------------------- //

/// Parsed `AbortAck`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AbortAck {
    pub room: Room,
    pub prefill_rank: i64,
    pub generation: Option<i64>,
}

/// `AbortAck.to_zmq`: `[b"ABORT_ACK", room, prefill_rank, generation?]`.
pub fn build_abort_ack(a: &AbortAck) -> Vec<Vec<u8>> {
    let mut frames = vec![
        ABORT_ACK_TAG.to_vec(),
        a.room.to_string().into_bytes(),
        a.prefill_rank.to_string().into_bytes(),
    ];
    if let Some(g) = a.generation {
        frames.push(g.to_string().into_bytes());
    }
    frames
}

/// `AbortAck.from_zmq`. `msg` INCLUDES the tag frame. None when fewer than
/// 3 frames or the int frames don't decode.
pub fn parse_abort_ack(msg: &[Vec<u8>]) -> Option<AbortAck> {
    if msg.len() < 3 {
        return None;
    }
    let room = py_int(&msg[1]).ok()?;
    let prefill_rank = py_int(&msg[2]).ok()?;
    // Python's try wraps the whole construction: a bad generation frame
    // fails the entire parse, not just the generation field.
    let generation = if msg.len() >= 4 {
        Some(py_int(&msg[3]).ok()?)
    } else {
        None
    };
    Some(AbortAck {
        room,
        prefill_rank,
        generation,
    })
}

// --------------------------------------------------------------------------- //
// BOOTSTRAP (deferred bootstrap control, common/conn.py:692-712, :2573-2577)
// --------------------------------------------------------------------------- //

/// The decode's deferred-bootstrap announcement: `[b"BOOTSTRAP", room, ip, port]`
/// (common/conn.py:2573-2577).
pub fn build_bootstrap(room: Room, local_ip: &str, rank_port: i64) -> Vec<Vec<u8>> {
    vec![
        BOOTSTRAP_TAG.to_vec(),
        room.to_string().into_bytes(),
        local_ip.as_bytes().to_vec(),
        rank_port.to_string().into_bytes(),
    ]
}

/// The endpoint decode inside `_handle_bootstrap_message` for a BOOTSTRAP
/// message: `(room, (ip, port))`; None when the frames don't decode (the
/// Python handler drops such messages). `msg` INCLUDES the tag frame.
pub fn parse_bootstrap(msg: &[Vec<u8>]) -> Option<(Room, String, i64)> {
    if msg.first().map(Vec::as_slice) != Some(BOOTSTRAP_TAG) {
        return None;
    }
    let room = py_int(msg.get(1)?).ok()?;
    let ip = std::str::from_utf8(msg.get(2)?).ok()?.to_string();
    let port = py_int(msg.get(3)?).ok()?;
    Some((room, ip, port))
}

// --------------------------------------------------------------------------- //
// CHUNK_READY (staging, common/conn.py:813-849)
// --------------------------------------------------------------------------- //

/// `send_chunk_ready` frames: `[b"CHUNK_READY", room, chunk_idx, page_start,
/// num_pages, writer_id, prefill_rank]`.
pub fn build_chunk_ready(
    room: Room,
    chunk_idx: i64,
    page_start: i64,
    num_pages: i64,
    writer_id: &str,
    prefill_rank: i64,
) -> Vec<Vec<u8>> {
    vec![
        CHUNK_READY_TAG.to_vec(),
        room.to_string().into_bytes(),
        chunk_idx.to_string().into_bytes(),
        page_start.to_string().into_bytes(),
        num_pages.to_string().into_bytes(),
        writer_id.as_bytes().to_vec(),
        prefill_rank.to_string().into_bytes(),
    ]
}

/// `handle_chunk_ready` decode: `(room, chunk_idx, page_start, num_pages,
/// writer_id)` — the trailing prefill-rank frame is ignored by the Python
/// parser. None when tag, ints, or writer_id don't decode.
pub fn parse_chunk_ready(msg: &[Vec<u8>]) -> Option<(Room, i64, i64, i64, String)> {
    if msg.first().map(Vec::as_slice) != Some(CHUNK_READY_TAG) || msg.len() < 6 {
        return None;
    }
    let room = py_int(&msg[1]).ok()?;
    let chunk_idx = py_int(&msg[2]).ok()?;
    let page_start = py_int(&msg[3]).ok()?;
    let num_pages = py_int(&msg[4]).ok()?;
    let writer_id = std::str::from_utf8(&msg[5]).ok()?.to_string();
    Some((room, chunk_idx, page_start, num_pages, writer_id))
}
