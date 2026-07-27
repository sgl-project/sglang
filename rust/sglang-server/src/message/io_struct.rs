//! The scheduler wire structs — the Rust mirror of the Python `io_struct`
//! messages this server sends (`python/sglang/srt/managers/io_struct.py`).
//! Each is a msgspec `array_like=True` struct, so **field order is wire order**
//! and `rmp_serde`'s default struct-as-array encoding reproduces it.

use bytes::Bytes;
use serde::Serialize;

use super::types::{StructTag, Tagged, control_messages, wire_struct};
use super::{GenerateRequest, SamplingParams, TokenIds};
use crate::error::Error;

wire_struct! {
    /// The scheduler's `TokenizedGenerateReqInput`. Keep in lockstep with the
    /// Python declaration: inserting a field anywhere but the end shifts every
    /// later field on the wire.
    pub(super) TokenizedGenerateReqInput<'a> {
        input_text: Option<&'a str>,
        /// Always nil: the ids ride the ring's columnar buffer, not msgpack.
        input_ids: (),
        input_embeds: (),
        mm_inputs: (),
        token_type_ids: (),
        #[serde(serialize_with = "sampling_params_as_map")]
        sampling_params: &'a SamplingParams,
        return_logprob: bool,
        logprob_start_len: i64,
        top_logprobs_num: i64,
        token_ids_logprob: Option<&'a TokenIds>,
        stream: bool,
        /// Not exposed by this server yet; the scheduler needs the slot filled.
        return_sampling_mask: bool,
        return_hidden_states: bool,
        /// Filler block (not exposed by this server yet): default/nil slots so
        /// the PD fields below land on their Python wire indices (24–30).
        return_routed_experts: bool,
        routed_experts_start_len: i64,
        return_indexer_topk: bool,
        session_id: (),
        session_params: (),
        lora_id: (),
        custom_logit_processor: (),
        positional_embed_overrides: (),
        /// PD-disaggregation block — the last fields emitted; everything after
        /// `disagg_prefill_dp_rank` in Python has a msgspec default and is
        /// omitted (short arrays decode with defaulted tails).
        bootstrap_host: Option<&'a str>,
        bootstrap_port: Option<i64>,
        bootstrap_room: Option<i64>,
        bootstrap_pair_key: Option<&'a str>,
        decode_tp_size: Option<i64>,
        routed_dp_rank: Option<i64>,
        disagg_prefill_dp_rank: Option<i64>,
    }
}

// Owned-rid messages: these are held by a [`ControlRequest`] inside an owned
// `Request`, so they cannot borrow the rid that request owns. `pub(crate)`
// because that enum is crate-visible — their fields stay private, so only the
// constructors below can build one.
control_messages! {
    /// The scheduler's `AbortReq`: stop generating for one rid.
    AbortReq {
        /// This server never aborts the whole queue — only the one rid.
        abort_all: bool,
        finished_reason: (),
        abort_message: (),
    }

    /// `/server_info`'s control request: a bare `BaseReq` with no extra fields.
    GetInternalStateReq {}
}

/// Python's `SamplingParams` is a plain msgspec Struct (not `array_like`), so it
/// must be a **map** inside the positional header — but `rmp_serde` writes every
/// struct positionally. Routing through `serde_json::Value`, whose `Serialize`
/// emits a map, keeps the field list in exactly one place (the derive on
/// [`SamplingParams`]) instead of a hand-written 30-entry `serialize_map`.
fn sampling_params_as_map<S: serde::Serializer>(
    params: &&SamplingParams,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    serde_json::to_value(params)
        .map_err(serde::ser::Error::custom)? // codespell:ignore ser
        .serialize(serializer)
}

/// Borrow a request as its wire struct, resolving `Option` scalars to the wire
/// defaults Python's own fields carry. Borrowed, not owned: every field is a
/// reference into `req`, so an owning `From` would return references to a
/// dropped local.
///
/// The rid comes from [`GenerateRequest::rid`] — the same value `submit` copied
/// into the owning `Request`, so the scheduler, the detok registration and
/// `meta_info.id` cannot disagree.
impl<'a> From<&'a GenerateRequest> for TokenizedGenerateReqInput<'a> {
    fn from(req: &'a GenerateRequest) -> Self {
        Self {
            tag: StructTag::default(),
            rid: &req.rid,
            http_worker_ipc: (),
            input_text: req.text.as_deref(),
            input_ids: (),
            input_embeds: (),
            mm_inputs: (),
            token_type_ids: (),
            sampling_params: &req.sampling_params,
            return_logprob: req.return_logprob.unwrap_or(false),
            logprob_start_len: req.logprob_start_len.unwrap_or(-1),
            top_logprobs_num: req.top_logprobs_num.unwrap_or(0),
            token_ids_logprob: req.token_ids_logprob.as_ref(),
            stream: req.stream,
            return_sampling_mask: false,
            return_hidden_states: req.return_hidden_states.unwrap_or(false),
            return_routed_experts: false,
            routed_experts_start_len: 0,
            return_indexer_topk: false,
            session_id: (),
            session_params: (),
            lora_id: (),
            custom_logit_processor: (),
            positional_embed_overrides: (),
            bootstrap_host: req.bootstrap_host.as_deref(),
            bootstrap_port: req.bootstrap_port,
            bootstrap_room: req.bootstrap_room,
            bootstrap_pair_key: req.bootstrap_pair_key.as_deref(),
            decode_tp_size: req.decode_tp_size,
            routed_dp_rank: req.routed_dp_rank,
            disagg_prefill_dp_rank: req.disagg_prefill_dp_rank,
        }
    }
}

impl GetInternalStateReq {
    pub fn new(rid: String) -> Self {
        Self {
            tag: StructTag::default(),
            rid,
            http_worker_ipc: (),
        }
    }
}

impl AbortReq {
    pub fn new(rid: String, abort_all: bool) -> Self {
        Self {
            tag: StructTag::default(),
            rid,
            http_worker_ipc: (),
            abort_all,
            finished_reason: (),
            abort_message: (),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abort_req_msgpack_shape() {
        let b = AbortReq::new("12345".into(), false).encode().unwrap();
        let val = rmpv::decode::read_value(&mut &b[..]).unwrap();
        let arr = val.as_array().expect("array");
        assert_eq!(
            arr.len(),
            6,
            "AbortReq = [tag, rid, http_ipc, abort_all, finished_reason, abort_message]"
        );
        assert_eq!(arr[0].as_str(), Some("AbortReq"));
        assert_eq!(arr[1].as_str(), Some("12345"));
        assert!(arr[2].is_nil());
        assert_eq!(arr[3].as_bool(), Some(false));
        assert!(arr[4].is_nil());
        assert!(arr[5].is_nil());
    }

    /// The header must be positionally aligned: `input_embeds` (idx 5) /
    /// `token_type_ids` (idx 7) present as nil so `sampling_params` lands at idx 8 and
    /// the array reaches msgspec's min length. Regression guard for that decode failure.
    #[test]
    fn to_header_msgpack_is_positionally_aligned() {
        let req = GenerateRequest {
            rid: "r1".into(),
            text: Some("hi".into()),
            input_ids: Some(vec![1, 2, 3]),
            sampling_params: SamplingParams {
                max_new_tokens: Some(5),
                ..Default::default()
            },
            return_logprob: Some(true),
            logprob_start_len: Some(-1),
            top_logprobs_num: Some(3),
            return_hidden_states: Some(true),
            stream: true,
            ..Default::default()
        };
        let bytes = TokenizedGenerateReqInput::from(&req).encode().unwrap();
        let val = rmpv::decode::read_value(&mut &bytes[..]).unwrap();
        let arr = val.as_array().expect("array");
        // msgspec requires >= 14 (through `stream`); we emit 31 (through
        // `disagg_prefill_dp_rank`). Trailing defaulted fields are omitted.
        assert_eq!(arr.len(), 31, "header ends at disagg_prefill_dp_rank");
        assert_eq!(arr[0].as_str(), Some("TokenizedGenerateReqInput"));
        assert_eq!(arr[1].as_str(), Some("r1"));
        assert!(arr[5].is_nil(), "idx 5 must be input_embeds (nil)");
        assert!(arr[7].is_nil(), "idx 7 must be token_type_ids (nil)");
        assert!(arr[8].is_map(), "sampling_params must land at idx 8");
        assert_eq!(arr[9].as_bool(), Some(true), "return_logprob at idx 9");
        assert_eq!(arr[11].as_u64(), Some(3), "top_logprobs_num at idx 11");
        assert_eq!(arr[13].as_bool(), Some(true), "stream at idx 13");
        // idx 14 is `return_sampling_mask` (never client-set); a shift here would
        // silently flip the wrong scheduler field.
        assert_eq!(
            arr[14].as_bool(),
            Some(false),
            "return_sampling_mask at idx 14"
        );
        assert_eq!(
            arr[15].as_bool(),
            Some(true),
            "return_hidden_states at idx 15"
        );
    }

    /// The PD block must land on Python's wire indices 24–30, with the filler
    /// block (16–23) holding its defaults — a shift here silently routes KV
    /// transfers to the wrong host/room.
    #[test]
    fn header_bootstrap_block_is_positionally_aligned() {
        let req = GenerateRequest {
            rid: "r1".into(),
            text: Some("hi".into()),
            bootstrap_host: Some("10.0.0.1".into()),
            bootstrap_port: Some(8998),
            bootstrap_room: Some(i64::MAX), // routers draw from [0, 2^63)
            bootstrap_pair_key: Some("pk".into()),
            decode_tp_size: Some(2),
            routed_dp_rank: Some(3),
            disagg_prefill_dp_rank: Some(4),
            ..Default::default()
        };
        let bytes = TokenizedGenerateReqInput::from(&req).encode().unwrap();
        let val = rmpv::decode::read_value(&mut &bytes[..]).unwrap();
        let arr = val.as_array().expect("array");
        assert_eq!(arr[16].as_bool(), Some(false), "return_routed_experts");
        assert_eq!(arr[17].as_u64(), Some(0), "routed_experts_start_len");
        assert_eq!(arr[18].as_bool(), Some(false), "return_indexer_topk");
        for i in 19..=23 {
            assert!(arr[i].is_nil(), "idx {i} must be a nil default");
        }
        assert_eq!(arr[24].as_str(), Some("10.0.0.1"), "bootstrap_host at 24");
        assert_eq!(arr[25].as_u64(), Some(8998), "bootstrap_port at 25");
        assert_eq!(arr[26].as_i64(), Some(i64::MAX), "bootstrap_room at 26");
        assert_eq!(arr[27].as_str(), Some("pk"), "bootstrap_pair_key at 27");
        assert_eq!(arr[28].as_i64(), Some(2), "decode_tp_size at 28");
        assert_eq!(arr[29].as_i64(), Some(3), "routed_dp_rank at 29");
        assert_eq!(arr[30].as_i64(), Some(4), "disagg_prefill_dp_rank at 30");
    }
}
