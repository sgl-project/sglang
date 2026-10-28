//! PyO3 bindings for the wire codecs. The exposed functions take and
//! return the same primitive values the Python builders consume (ints,
//! strings, bytes, lists); Python-side wrappers in
//! sglang/srt/disaggregation/rust_wire adapt them onto the dataclass
//! surfaces (`KVArgsRegisterInfo`, `TransferInfo`). Wire bytes are produced
//! by the same vector-verified codecs the Rust tests replay.

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::nixl::{self, MemKind, MetadataParams, RegistrationParams};
use crate::status;

fn py_err(e: crate::Error) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(e.to_string())
}

fn mem_kinds(kinds: Vec<String>) -> PyResult<Vec<MemKind>> {
    kinds
        .iter()
        .map(|k| match k.as_str() {
            "VRAM" => Ok(MemKind::Vram),
            "DRAM" => Ok(MemKind::Dram),
            other => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unknown mem kind {other:?}"
            ))),
        })
        .collect()
}

/// `build_receiver_registration_message` (nixl/conn.py:3428) — returns the
/// GUARD + 24 frames, ready for `send_multipart`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn build_registration(
    local_ip: String,
    rank_port: i64,
    agent_name: String,
    agent_metadata: Vec<u8>,
    kv_data_ptrs: Vec<u64>,
    kv_data_lens: Vec<u64>,
    kv_data_mem_kinds: Vec<String>,
    kv_item_lens: Vec<u64>,
    kv_layer_ids: Vec<u32>,
    aux_data_ptrs: Vec<u64>,
    state_data_ptrs: Vec<Vec<u64>>,
    state_item_lens: Vec<Vec<u32>>,
    state_data_lens: Vec<Vec<u64>>,
    state_dim_per_tensor: Vec<Vec<u32>>,
    state_layer_ids: Vec<Vec<u32>>,
    gpu_id: i64,
    attn_tp_size: i64,
    engine_rank: i64,
    dcp_size: i64,
    dcp_rank: i64,
    staging: Option<(u64, u64)>,
) -> PyResult<Vec<Vec<u8>>> {
    let params = RegistrationParams {
        local_ip,
        rank_port,
        agent_name,
        agent_metadata,
        kv_data_ptrs,
        kv_data_lens,
        kv_data_mem_kinds: mem_kinds(kv_data_mem_kinds)?,
        kv_item_lens,
        kv_layer_ids,
        aux_data_ptrs,
        state_data_ptrs,
        state_item_lens,
        state_data_lens,
        state_dim_per_tensor,
        state_layer_ids,
        gpu_id,
        attn_tp_size,
        engine_rank,
        dcp_size,
        dcp_rank,
        staging,
    };
    Ok(nixl::build_registration(&params))
}

/// `KVArgsRegisterInfo.from_zmq` — `msg` excludes the GUARD frame. Returns a
/// dict keyed by the dataclass field names; the Python wrapper constructs
/// `KVArgsRegisterInfo(**result)`.
#[pyfunction]
fn parse_registration(py: Python<'_>, msg: Vec<Vec<u8>>) -> PyResult<Bound<'_, PyDict>> {
    let info = nixl::parse_registration(&msg).map_err(py_err)?;
    let d = PyDict::new(py);
    d.set_item("room", info.room)?;
    d.set_item("endpoint", info.endpoint)?;
    d.set_item("dst_port", info.dst_port)?;
    d.set_item("agent_name", info.agent_name)?;
    d.set_item("agent_metadata", info.agent_metadata)?;
    d.set_item("dst_kv_ptrs", info.dst_kv_ptrs)?;
    d.set_item(
        "dst_kv_mem_kinds",
        info.dst_kv_mem_kinds
            .iter()
            .map(MemKind::as_str)
            .collect::<Vec<_>>(),
    )?;
    d.set_item("dst_aux_ptrs", info.dst_aux_ptrs)?;
    d.set_item("dst_state_data_ptrs", info.dst_state_data_ptrs)?;
    d.set_item("gpu_id", info.gpu_id)?;
    d.set_item("decode_tp_size", info.decode_tp_size)?;
    d.set_item("decode_tp_rank", info.decode_tp_rank)?;
    d.set_item("dst_kv_item_len", info.dst_kv_item_len)?;
    d.set_item("dst_kv_item_lens", info.dst_kv_item_lens)?;
    d.set_item("dst_kv_layer_ids", info.dst_kv_layer_ids)?;
    d.set_item("dst_dcp_size", info.dst_dcp_size)?;
    d.set_item("dst_dcp_rank", info.dst_dcp_rank)?;
    d.set_item("dst_num_slots", info.dst_num_slots)?;
    d.set_item("dst_state_item_lens", info.dst_state_item_lens)?;
    d.set_item("dst_state_dim_per_tensor", info.dst_state_dim_per_tensor)?;
    d.set_item("dst_state_layer_ids", info.dst_state_layer_ids)?;
    d.set_item("staging_base_ptr", info.staging_base_ptr)?;
    d.set_item("staging_total_size", info.staging_total_size)?;
    d.set_item("dst_state_data_lens", info.dst_state_data_lens)?;
    Ok(d)
}

/// `build_receiver_metadata_message` (nixl/conn.py:3495) — GUARD + 10
/// frames.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn build_metadata(
    room: i64,
    local_ip: String,
    rank_port: i64,
    agent_name: String,
    kv_indices: Vec<i32>,
    aux_index: Option<i64>,
    required_dst_info_num: i64,
    state_indices: Option<Vec<Option<Vec<i32>>>>,
    decode_prefix_len: Option<i64>,
    is_dummy: bool,
) -> Vec<Vec<u8>> {
    nixl::build_metadata(&MetadataParams {
        room,
        local_ip,
        rank_port,
        agent_name,
        kv_indices,
        aux_index,
        required_dst_info_num,
        state_indices,
        decode_prefix_len,
        is_dummy,
    })
}

/// `TransferInfo.from_zmq` (nixl/conn.py:303) — `msg` excludes the GUARD.
/// Returns a dict keyed by the dataclass constructor kwargs; the Python
/// wrapper converts `dst_kv_indices` to numpy.
#[pyfunction]
fn parse_metadata(py: Python<'_>, msg: Vec<Vec<u8>>) -> PyResult<Bound<'_, PyDict>> {
    let info = nixl::parse_metadata(&msg).map_err(py_err)?;
    let d = PyDict::new(py);
    d.set_item("room", info.room)?;
    d.set_item("endpoint", info.endpoint)?;
    d.set_item("dst_port", info.dst_port)?;
    d.set_item("agent_name", info.agent_name)?;
    d.set_item("dst_kv_indices", info.dst_kv_indices)?;
    d.set_item("dst_aux_index", info.dst_aux_index)?;
    d.set_item("required_dst_info_num", info.required_dst_info_num)?;
    d.set_item("dst_state_indices", info.dst_state_indices)?;
    d.set_item("decode_prefix_len", info.decode_prefix_len)?;
    d.set_item("is_dummy", info.is_dummy)?;
    Ok(d)
}

/// The CHUNK_READY frames (common/conn.py:807): room, chunk_idx,
/// page_start, num_pages, writer_id, prefill rank — ready to send.
#[pyfunction]
fn build_chunk_ready(
    room: i64,
    chunk_idx: i64,
    page_start: i64,
    num_pages: i64,
    writer_id: String,
    prefill_rank: i64,
) -> Vec<Vec<u8>> {
    status::build_chunk_ready(
        room,
        chunk_idx,
        page_start,
        num_pages,
        &writer_id,
        prefill_rank,
    )
}

/// `handle_chunk_ready` parse half — None when the frames are not a
/// well-formed CHUNK_READY (caller falls through to other handlers).
#[pyfunction]
fn parse_chunk_ready(msg: Vec<Vec<u8>>) -> Option<(i64, i64, i64, i64, String)> {
    status::parse_chunk_ready(&msg)
}

/// The kv-status frames (room, status, prefill_rank, optional reason,
/// optional tag) exactly as common/conn.py's sender builds them.
#[pyfunction]
fn encode_kv_status(
    room: i64,
    status_value: i32,
    prefill_rank: i64,
    failure_reason: Option<String>,
    tag: Option<Vec<u8>>,
    carries_reason: bool,
) -> PyResult<Vec<Vec<u8>>> {
    let status = crate::KVPoll::from_i32(status_value).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!("bad KVPoll {status_value}"))
    })?;
    Ok(status::encode_kv_status(
        room,
        status,
        prefill_rank,
        failure_reason.as_deref(),
        tag.as_deref(),
        carries_reason,
    ))
}

/// `parse_kv_status_message` — None on tag mismatch or malformed frames;
/// otherwise (room, status, prefill_rank, failure_reason).
#[pyfunction]
fn parse_kv_status(
    msg: Vec<Vec<u8>>,
    tag: Option<Vec<u8>>,
) -> Option<(i64, i32, i64, Option<String>)> {
    status::parse_kv_status(&msg, tag.as_deref())
}

#[pymodule]
fn disagg_wire(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(build_registration, m)?)?;
    m.add_function(wrap_pyfunction!(parse_registration, m)?)?;
    m.add_function(wrap_pyfunction!(build_metadata, m)?)?;
    m.add_function(wrap_pyfunction!(parse_metadata, m)?)?;
    m.add_function(wrap_pyfunction!(build_chunk_ready, m)?)?;
    m.add_function(wrap_pyfunction!(parse_chunk_ready, m)?)?;
    m.add_function(wrap_pyfunction!(encode_kv_status, m)?)?;
    m.add_function(wrap_pyfunction!(parse_kv_status, m)?)?;
    Ok(())
}
