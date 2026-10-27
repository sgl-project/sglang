//! NIXL zmq frame codecs (`nixl/conn.py`).
//!
//! Registration: `GUARD + 24` frames, sent once by each decode rank
//! (`build_receiver_registration_message` /
//! `KVArgsRegisterInfo.from_zmq`). Metadata: `GUARD + 10` frames, sent
//! per request (`build_receiver_metadata_message` / `TransferInfo.from_zmq`).
//!
//! Field names follow Python on each side: the BUILDER uses
//! `attn_tp_size`/`engine_rank`, the PARSER names the same two frames
//! `decode_tp_size`/`decode_tp_rank`.

use crate::error::Error;
use crate::pack::{
    pack_i32_lists, pack_ints_i32, pack_ints_u32, pack_ints_u64, pack_u32_lists, pack_u64_lists,
    py_int, unpack_i32_lists, unpack_u32_lists, unpack_u64_lists,
};
use crate::{Result, Room};

/// ZMQ multipart guard for the NIXL handshake frames (nixl/conn.py:78).
pub const NIXL_GUARD: &[u8] = b"NixlMsgGuard";

/// KV memory kind (`nixl/conn.py` `KV_MEM_KINDS = {"VRAM", "DRAM"}`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemKind {
    Vram,
    Dram,
}

impl MemKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            MemKind::Vram => "VRAM",
            MemKind::Dram => "DRAM",
        }
    }
}

/// `num_kv_slots` (nixl/conn.py:483-495): `min(d // i for d, i in
/// zip(kv_data_lens, kv_item_lens))`, default 0. A zero item_len raises
/// ZeroDivisionError in Python; the Rust divide panics the same way.
pub fn num_kv_slots(kv_data_lens: &[u64], kv_item_lens: &[u64]) -> u64 {
    kv_data_lens
        .iter()
        .zip(kv_item_lens.iter())
        .map(|(d, i)| d / i)
        .min()
        .unwrap_or(0)
}

/// `_pack_kv_mem_kinds` (nixl/conn.py:219-220).
fn pack_kv_mem_kinds(kinds: &[MemKind]) -> Vec<u8> {
    kinds
        .iter()
        .map(|k| k.as_str())
        .collect::<Vec<_>>()
        .join(",")
        .into_bytes()
}

/// `_unpack_kv_mem_kinds` + `_normalize_kv_mem_kinds` (nixl/conn.py:204-227):
/// empty buffer -> all VRAM; length mismatch and unknown kinds raise
/// ValueError in Python (Codec/Invalid errors here).
fn unpack_kv_mem_kinds(buf: &[u8], expected_len: usize) -> Result<Vec<MemKind>> {
    if buf.is_empty() {
        return Ok(vec![MemKind::Vram; expected_len]);
    }
    let s = std::str::from_utf8(buf)
        .map_err(|_| Error::Codec("registration: mem_kinds not ascii".into()))?;
    let kinds: Result<Vec<MemKind>> = s
        .split(',')
        .map(|k| match k {
            "VRAM" => Ok(MemKind::Vram),
            "DRAM" => Ok(MemKind::Dram),
            _ => Err(Error::Invalid(format!(
                "Unsupported NIXL KV memory kind(s): {:?}",
                k
            ))),
        })
        .collect();
    let kinds = kinds?;
    if kinds.len() != expected_len {
        return Err(Error::Invalid(format!(
            "kv_data_mem_kinds length mismatch: got {}, expected {}",
            kinds.len(),
            expected_len
        )));
    }
    Ok(kinds)
}

fn ascii_frame(s: &str) -> Vec<u8> {
    s.as_bytes().to_vec()
}

fn frame_u64s(buf: &[u8]) -> Result<Vec<u64>> {
    if !buf.len().is_multiple_of(8) {
        return Err(Error::Codec("registration: bad u64 frame length".into()));
    }
    Ok(buf
        .chunks_exact(8)
        .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

fn frame_u32s(buf: &[u8]) -> Result<Vec<u32>> {
    if !buf.len().is_multiple_of(4) {
        return Err(Error::Codec("registration: bad u32 frame length".into()));
    }
    Ok(buf
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

fn frame_str(buf: &[u8]) -> Result<String> {
    String::from_utf8(buf.to_vec()).map_err(|_| Error::Codec("frame: bad utf-8".into()))
}

// --------------------------------------------------------------------------- //
// Registration (nixl/conn.py:3428-3492 build, :342-467 parse)
// --------------------------------------------------------------------------- //

/// Everything the decode side must supply to build its registration message.
#[derive(Clone, Debug)]
pub struct RegistrationParams {
    pub local_ip: String,
    pub rank_port: i64,
    pub agent_name: String,
    pub agent_metadata: Vec<u8>,
    pub kv_data_ptrs: Vec<u64>,
    /// Per-slab total bytes; zipped with kv_item_lens for dst_num_slots.
    pub kv_data_lens: Vec<u64>,
    pub kv_data_mem_kinds: Vec<MemKind>,
    pub kv_item_lens: Vec<u64>,
    pub kv_layer_ids: Vec<u32>,
    pub aux_data_ptrs: Vec<u64>,
    pub state_data_ptrs: Vec<Vec<u64>>,
    pub state_item_lens: Vec<Vec<u32>>,
    pub state_data_lens: Vec<Vec<u64>>,
    pub state_dim_per_tensor: Vec<Vec<u32>>,
    pub state_layer_ids: Vec<Vec<u32>>,
    pub gpu_id: i64,
    pub attn_tp_size: i64,
    pub engine_rank: i64,
    pub dcp_size: i64,
    pub dcp_rank: i64,
    /// (base_ptr, total_size) when the staging allocator is live; None for
    /// the zero frames.
    pub staging: Option<(u64, u64)>,
}

/// Parsed registration (parser names, `KVArgsRegisterInfo`).
#[derive(Clone, Debug, PartialEq)]
pub struct RegistrationInfo {
    pub room: String,
    pub endpoint: String,
    pub dst_port: i64,
    pub agent_name: String,
    pub agent_metadata: Vec<u8>,
    pub dst_kv_ptrs: Vec<u64>,
    pub dst_kv_mem_kinds: Vec<MemKind>,
    pub dst_aux_ptrs: Vec<u64>,
    pub dst_state_data_ptrs: Vec<Vec<u64>>,
    pub gpu_id: i64,
    pub decode_tp_size: i64,
    pub decode_tp_rank: i64,
    pub dst_kv_item_len: u64,
    pub dst_kv_item_lens: Vec<u64>,
    pub dst_kv_layer_ids: Vec<u32>,
    pub dst_dcp_size: i64,
    pub dst_dcp_rank: i64,
    pub dst_num_slots: Option<u64>,
    pub dst_state_item_lens: Vec<Vec<u32>>,
    pub dst_state_dim_per_tensor: Vec<Vec<u32>>,
    pub dst_state_layer_ids: Vec<Vec<u32>>,
    pub staging_base_ptr: u64,
    pub staging_total_size: u64,
    pub dst_state_data_lens: Vec<Vec<u64>>,
}

/// `build_receiver_registration_message` — returns GUARD + 24 frames.
pub fn build_registration(p: &RegistrationParams) -> Vec<Vec<u8>> {
    let (dst_kv_item_len, dst_num_slots) = if p.kv_item_lens.is_empty() {
        (0, 0)
    } else {
        (
            p.kv_item_lens[0],
            num_kv_slots(&p.kv_data_lens, &p.kv_item_lens),
        )
    };
    let mem_kinds = if p.kv_data_mem_kinds.is_empty() {
        vec![MemKind::Vram; p.kv_data_ptrs.len()]
    } else {
        p.kv_data_mem_kinds.clone()
    };
    let (staging_base, staging_total) = match p.staging {
        Some((b, t)) => (b.to_le_bytes().to_vec(), t.to_string().into_bytes()),
        None => (Vec::new(), Vec::new()),
    };
    vec![
        NIXL_GUARD.to_vec(),
        ascii_frame("None"),
        ascii_frame(&p.local_ip),
        p.rank_port.to_string().into_bytes(),
        ascii_frame(&p.agent_name),
        p.agent_metadata.clone(),
        pack_ints_u64(&p.kv_data_ptrs),
        pack_ints_u64(&p.aux_data_ptrs),
        pack_u64_lists(&p.state_data_ptrs),
        p.gpu_id.to_string().into_bytes(),
        p.attn_tp_size.to_string().into_bytes(),
        p.engine_rank.to_string().into_bytes(),
        dst_kv_item_len.to_string().into_bytes(),
        pack_u32_lists(&p.state_item_lens),
        pack_u32_lists(&p.state_dim_per_tensor),
        staging_base,
        staging_total,
        dst_num_slots.to_string().into_bytes(),
        pack_kv_mem_kinds(&mem_kinds),
        pack_ints_u64(&p.kv_item_lens),
        pack_u32_lists(&p.state_layer_ids),
        pack_ints_u32(&p.kv_layer_ids),
        p.dcp_size.to_string().into_bytes(),
        p.dcp_rank.to_string().into_bytes(),
        pack_u64_lists(&p.state_data_lens),
    ]
}

/// `KVArgsRegisterInfo.from_zmq` (nixl/conn.py:376-467) — `msg` is the frame
/// list WITHOUT the GUARD (msg[0] is the room frame).
pub fn parse_registration(msg: &[Vec<u8>]) -> Result<RegistrationInfo> {
    let need = |i: usize| -> Result<&[u8]> {
        msg.get(i)
            .map(Vec::as_slice)
            .ok_or_else(|| Error::Codec(format!("registration: missing frame {}", i)))
    };
    // f0-11 unconditional (nixl/conn.py:377-393).
    let room = frame_str(need(0)?)?;
    let endpoint = frame_str(need(1)?)?;
    let dst_port = py_int(need(2)?)?;
    let agent_name = frame_str(need(3)?)?;
    let agent_metadata = need(4)?.to_vec();
    let dst_kv_ptrs = frame_u64s(need(5)?)?;
    let dst_aux_ptrs = frame_u64s(need(6)?)?;
    let dst_state_data_ptrs = unpack_u64_lists(need(7)?)?;
    let gpu_id = py_int(need(8)?)?;
    let decode_tp_size = py_int(need(9)?)?;
    let decode_tp_rank = py_int(need(10)?)?;
    let dst_kv_item_len = py_int(need(11)?)? as u64;
    // f12+ guarded by presence AND non-empty.
    let opt =
        |i: usize| -> Option<&[u8]> { msg.get(i).map(Vec::as_slice).filter(|b| !b.is_empty()) };
    let dst_state_item_lens = match opt(12) {
        Some(b) => unpack_u32_lists(b)?,
        None => Vec::new(),
    };
    let dst_state_dim_per_tensor = match opt(13) {
        Some(b) => unpack_u32_lists(b)?,
        None => Vec::new(),
    };
    let staging_base_ptr = match opt(14) {
        Some(b) if b.len() == 8 => u64::from_le_bytes(b.try_into().unwrap()),
        _ => 0,
    };
    let staging_total_size = match opt(15) {
        Some(b) => py_int(b)? as u64,
        None => 0,
    };
    let dst_num_slots = match opt(16) {
        Some(b) => Some(py_int(b)? as u64),
        None => None,
    };
    // Guarded by presence only; empty buffer unpacks to VRAM.
    let dst_kv_mem_kinds = match msg.get(17) {
        Some(b) => unpack_kv_mem_kinds(b, dst_kv_ptrs.len())?,
        None => vec![MemKind::Vram; dst_kv_ptrs.len()],
    };
    let dst_kv_item_lens = match opt(18) {
        Some(b) => frame_u64s(b)?,
        // Fall back to item_len * len(kv_ptrs).
        None => vec![dst_kv_item_len; dst_kv_ptrs.len()],
    };
    // Python raises ValueError on the length mismatch.
    if dst_kv_item_lens.len() != dst_kv_ptrs.len() {
        return Err(Error::Invalid(format!(
            "dst_kv_item_lens length mismatch: got {}, expected {}",
            dst_kv_item_lens.len(),
            dst_kv_ptrs.len()
        )));
    }
    let dst_state_layer_ids = match opt(19) {
        Some(b) => unpack_u32_lists(b)?,
        None => Vec::new(),
    };
    let dst_kv_layer_ids = match opt(20) {
        Some(b) => frame_u32s(b)?,
        None => Vec::new(),
    };
    let dst_dcp_size = match opt(21) {
        Some(b) => py_int(b)?,
        None => 1,
    };
    let dst_dcp_rank = match opt(22) {
        Some(b) => py_int(b)?,
        None => 0,
    };
    let dst_state_data_lens = match opt(23) {
        Some(b) => unpack_u64_lists(b)?,
        None => Vec::new(),
    };
    Ok(RegistrationInfo {
        room,
        endpoint,
        dst_port,
        agent_name,
        agent_metadata,
        dst_kv_ptrs,
        dst_kv_mem_kinds,
        dst_aux_ptrs,
        dst_state_data_ptrs,
        gpu_id,
        decode_tp_size,
        decode_tp_rank,
        dst_kv_item_len,
        dst_kv_item_lens,
        dst_kv_layer_ids,
        dst_dcp_size,
        dst_dcp_rank,
        dst_num_slots,
        dst_state_item_lens,
        dst_state_dim_per_tensor,
        dst_state_layer_ids,
        staging_base_ptr,
        staging_total_size,
        dst_state_data_lens,
    })
}

// --------------------------------------------------------------------------- //
// Metadata (nixl/conn.py:3495-3521 build, :284-339 parse)
// --------------------------------------------------------------------------- //

/// Parameters for `build_receiver_metadata_message` (decode side).
#[derive(Clone, Debug)]
pub struct MetadataParams {
    pub room: Room,
    pub local_ip: String,
    pub rank_port: i64,
    pub agent_name: String,
    pub kv_indices: Vec<i32>,
    /// None encodes as the literal frame `b"None"` (Python `str(None)`);
    /// the parser rejects it exactly like Python's `int("None")`.
    pub aux_index: Option<i64>,
    pub required_dst_info_num: i64,
    /// Per-request state indices; inner None -> empty list. Ignored when
    /// is_dummy (empty buffer).
    pub state_indices: Option<Vec<Option<Vec<i32>>>>,
    pub decode_prefix_len: Option<i64>,
    pub is_dummy: bool,
}

/// Parsed `TransferInfo`.
#[derive(Clone, Debug, PartialEq)]
pub struct MetadataInfo {
    pub room: i64,
    pub endpoint: String,
    pub dst_port: i64,
    pub agent_name: String,
    pub dst_kv_indices: Vec<i32>,
    pub dst_aux_index: i64,
    pub required_dst_info_num: i64,
    pub dst_state_indices: Vec<Vec<i32>>,
    pub decode_prefix_len: Option<i64>,
    pub is_dummy: bool,
}

/// `build_receiver_metadata_message` — GUARD + 10 frames.
pub fn build_metadata(p: &MetadataParams) -> Vec<Vec<u8>> {
    let kv_frame = if p.is_dummy {
        Vec::new()
    } else {
        pack_ints_i32(&p.kv_indices)
    };
    let state_frame = match (&p.state_indices, p.is_dummy) {
        (Some(indices), false) => {
            let lists: Vec<Vec<i32>> = indices
                .iter()
                .map(|opt| opt.clone().unwrap_or_default())
                .collect();
            pack_i32_lists(&lists)
        }
        _ => Vec::new(),
    };
    let aux_frame = match p.aux_index {
        Some(a) => a.to_string().into_bytes(),
        None => b"None".to_vec(),
    };
    vec![
        NIXL_GUARD.to_vec(),
        p.room.to_string().into_bytes(),
        ascii_frame(&p.local_ip),
        p.rank_port.to_string().into_bytes(),
        ascii_frame(&p.agent_name),
        kv_frame,
        aux_frame,
        p.required_dst_info_num.to_string().into_bytes(),
        state_frame,
        p.decode_prefix_len.unwrap_or(0).to_string().into_bytes(),
        (p.is_dummy as i32).to_string().into_bytes(),
    ]
}

/// `TransferInfo.from_zmq` (nixl/conn.py:303-339) — `msg` is the frame list
/// WITHOUT the GUARD.
pub fn parse_metadata(msg: &[Vec<u8>]) -> Result<MetadataInfo> {
    let need = |i: usize| -> Result<&[u8]> {
        msg.get(i)
            .map(Vec::as_slice)
            .ok_or_else(|| Error::Codec(format!("metadata: missing frame {}", i)))
    };
    let kv_buf = need(4)?;
    if kv_buf.len() % 4 != 0 {
        return Err(Error::Codec("metadata: kv_indices length not %4".into()));
    }
    let dst_kv_indices: Vec<i32> = kv_buf
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let decode_prefix_len = match msg.get(8).filter(|b| !b.is_empty()) {
        Some(b) => Some(py_int(b)?),
        None => None,
    };
    let dummy_rank = match msg.get(9).filter(|b| !b.is_empty()) {
        Some(b) => Some(py_int(b)? != 0),
        None => None,
    };
    // A transfer is "dummy" only for CP non-authoritative ranks. When
    // dst_kv_indices is empty due to a decode-side radix cache full hit
    // (decode_prefix_len > 0), the transfer is NOT dummy -- aux/state data
    // still needs to be sent.
    let is_dummy = match dummy_rank {
        Some(d) => d,
        None if dst_kv_indices.is_empty() && decode_prefix_len.map(|p| p != 0).unwrap_or(false) => {
            false
        }
        None => dst_kv_indices.is_empty(),
    };
    Ok(MetadataInfo {
        room: py_int(need(0)?)?,
        endpoint: frame_str(need(1)?)?,
        dst_port: py_int(need(2)?)?,
        agent_name: frame_str(need(3)?)?,
        dst_kv_indices,
        dst_aux_index: py_int(need(5)?)?,
        required_dst_info_num: py_int(need(6)?)?,
        dst_state_indices: match msg.get(7).filter(|b| !b.is_empty()) {
            Some(b) => unpack_i32_lists(b)?,
            None => Vec::new(),
        },
        decode_prefix_len,
        is_dummy,
    })
}
