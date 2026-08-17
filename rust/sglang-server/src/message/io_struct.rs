//! The scheduler wire structs — the Rust mirror of the Python `io_struct`
//! messages this server sends (`python/sglang/srt/managers/io_struct.py`).
//! Each is a msgspec `array_like=True` struct, so **field order is wire order**
//! and `rmp_serde`'s default struct-as-array encoding reproduces it.

use bytes::Bytes;
use serde::Serialize;

use super::types::{Tagged, control_messages, wire_struct};
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
        sampling_params: &'a SamplingParams,
        return_logprob: bool,
        logprob_start_len: i64,
        top_logprobs_num: i64,
        token_ids_logprob: Option<&'a TokenIds>,
        stream: bool,
        /// Not exposed by this server yet; the scheduler needs the slot filled.
        return_sampling_mask: bool,
        return_flat_raw_top_logprobs: bool,
        return_hidden_states: bool,
        /// Filler block (not exposed by this server yet): default/nil slots so
        /// the PD fields below land on their Python wire indices (25–31).
        return_routed_experts: bool,
        routed_experts_start_len: i64,
        return_indexer_topk: bool,
        session_id: (),
        session_params: (),
        lora_id: (),
        custom_logit_processor: (),
        positional_embed_overrides: (),
        /// PD-disaggregation block. The selector below requires emitting the
        /// intervening defaulted fields as positional fillers.
        bootstrap_host: Option<&'a str>,
        bootstrap_port: Option<i64>,
        bootstrap_room: Option<i64>,
        bootstrap_pair_key: Option<&'a str>,
        decode_tp_size: Option<i64>,
        routed_dp_rank: Option<i64>,
        disagg_prefill_dp_rank: Option<i64>,
        /// Tail fields through Python's `ngram_corpus_id`. These fillers are
        /// required because msgspec's array-like wire format is positional.
        routing_key: (),
        require_reasoning: bool,
        priority: (),
        extra_key: (),
        no_logs: bool,
        return_bytes: bool,
        return_entropy: bool,
        need_wait_for_mm_inputs: (),
        num_items_assigned: (),
        mm_data_mooncake: (),
        encoder_urls: (),
        multi_item_delimiter_indices: (),
        time_stats: (),
        cache_salt: (),
        ngram_corpus_id: Option<&'a str>,
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

    /// Load already-tokenized external NGRAM corpus chunks.
    AddExternalCorpusReqInput {
        corpus_id: Option<String>,
        file_path: (),
        documents: (),
        token_chunks: Vec<Vec<i32>>,
    }

    RemoveExternalCorpusReqInput {
        corpus_id: String,
    }

    ListExternalCorporaReqInput {}
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
            rid: &req.rid,
            input_text: req.text.as_deref(),
            input_ids: (),
            input_embeds: (),
            mm_inputs: (),
            token_type_ids: (),
            sampling_params: &req.sampling_params,
            return_logprob: req.return_logprob,
            logprob_start_len: req.logprob_start_len,
            top_logprobs_num: req.top_logprobs_num,
            token_ids_logprob: req.token_ids_logprob.as_ref(),
            stream: req.stream,
            return_sampling_mask: req.return_sampling_mask,
            return_flat_raw_top_logprobs: false,
            return_hidden_states: req.return_hidden_states,
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
            routing_key: (),
            require_reasoning: false,
            priority: (),
            extra_key: (),
            no_logs: false,
            return_bytes: false,
            return_entropy: false,
            need_wait_for_mm_inputs: (),
            num_items_assigned: (),
            mm_data_mooncake: (),
            encoder_urls: (),
            multi_item_delimiter_indices: (),
            time_stats: (),
            cache_salt: (),
            ngram_corpus_id: req.ngram_corpus_id.as_deref(),
        }
    }
}

impl GetInternalStateReq {
    pub fn new(rid: String) -> Self {
        Self { rid }
    }
}

impl AddExternalCorpusReqInput {
    pub fn new(rid: String, corpus_id: String, token_chunks: Vec<Vec<i32>>) -> Self {
        Self {
            rid,
            corpus_id: Some(corpus_id),
            file_path: (),
            documents: (),
            token_chunks,
        }
    }
}

impl RemoveExternalCorpusReqInput {
    pub fn new(rid: String, corpus_id: String) -> Self {
        Self { rid, corpus_id }
    }
}

impl ListExternalCorporaReqInput {
    pub fn new(rid: String) -> Self {
        Self { rid }
    }
}

impl AbortReq {
    pub fn new(rid: String, abort_all: bool) -> Self {
        Self {
            rid,
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

    #[test]
    fn external_corpus_control_messages_match_python_layout() {
        let add = AddExternalCorpusReqInput::new(
            "load-rid".into(),
            "docs".into(),
            vec![vec![1, 2], vec![i32::MIN, 3]],
        )
        .encode()
        .unwrap();
        let value = rmpv::decode::read_value(&mut &add[..]).unwrap();
        let arr = value.as_array().expect("array");
        assert_eq!(arr.len(), 7);
        assert_eq!(arr[0].as_str(), Some("AddExternalCorpusReqInput"));
        assert_eq!(arr[3].as_str(), Some("docs"));
        assert!(arr[4].is_nil(), "file_path");
        assert!(arr[5].is_nil(), "documents");
        assert!(arr[6].is_array(), "token_chunks");

        let remove = RemoveExternalCorpusReqInput::new(
            "remove-rid".into(),
            "docs".into(),
        )
        .encode()
        .unwrap();
        let value = rmpv::decode::read_value(&mut &remove[..]).unwrap();
        let arr = value.as_array().expect("array");
        assert_eq!(arr.len(), 4);
        assert_eq!(arr[0].as_str(), Some("RemoveExternalCorpusReqInput"));
        assert_eq!(arr[3].as_str(), Some("docs"));

        let list = ListExternalCorporaReqInput::new("list-rid".into())
            .encode()
            .unwrap();
        let value = rmpv::decode::read_value(&mut &list[..]).unwrap();
        let arr = value.as_array().expect("array");
        assert_eq!(arr.len(), 3);
        assert_eq!(arr[0].as_str(), Some("ListExternalCorporaReqInput"));
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
            return_logprob: true,
            logprob_start_len: -1,
            top_logprobs_num: 3,
            return_hidden_states: true,
            ngram_corpus_id: Some("docs".into()),
            stream: true,
            ..Default::default()
        };
        let bytes = TokenizedGenerateReqInput::from(&req).encode().unwrap();
        let val = rmpv::decode::read_value(&mut &bytes[..]).unwrap();
        let arr = val.as_array().expect("array");
        // msgspec requires >= 14 (through `stream`); the selector is appended
        // after every pre-existing field, at the true positional tail.
        assert_eq!(arr.len(), 47, "header ends at ngram_corpus_id");
        assert_eq!(arr[0].as_str(), Some("TokenizedGenerateReqInput"));
        assert_eq!(arr[1].as_str(), Some("r1"));
        assert!(arr[5].is_nil(), "idx 5 must be input_embeds (nil)");
        assert!(arr[7].is_nil(), "idx 7 must be token_type_ids (nil)");
        // An ARRAY, not a map: Python's `SamplingParams` is
        // `msgspec.Struct(array_like=True)`, so it decodes positionally.
        assert!(arr[8].is_array(), "sampling_params must land at idx 8");
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
            Some(false),
            "return_flat_raw_top_logprobs at idx 15"
        );
        assert_eq!(
            arr[16].as_bool(),
            Some(true),
            "return_hidden_states at idx 16"
        );
        assert_eq!(arr[36].as_bool(), Some(false), "no_logs at idx 36");
        assert_eq!(arr[37].as_bool(), Some(false), "return_bytes at idx 37");
        assert_eq!(arr[38].as_bool(), Some(false), "return_entropy at idx 38");
        for (i, slot) in arr.iter().enumerate().take(46).skip(39) {
            assert!(slot.is_nil(), "idx {i} must be a nil default");
        }
        assert_eq!(arr[46].as_str(), Some("docs"), "ngram_corpus_id at idx 46");
    }

    /// The PD block must land on Python's wire indices 25–31, with the filler
    /// block (17–24) holding its defaults — a shift here silently routes KV
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
        assert_eq!(arr[17].as_bool(), Some(false), "return_routed_experts");
        assert_eq!(arr[18].as_u64(), Some(0), "routed_experts_start_len");
        assert_eq!(arr[19].as_bool(), Some(false), "return_indexer_topk");
        for (i, slot) in arr.iter().enumerate().take(25).skip(20) {
            assert!(slot.is_nil(), "idx {i} must be a nil default");
        }
        assert_eq!(arr[25].as_str(), Some("10.0.0.1"), "bootstrap_host at 25");
        assert_eq!(arr[26].as_u64(), Some(8998), "bootstrap_port at 26");
        assert_eq!(arr[27].as_i64(), Some(i64::MAX), "bootstrap_room at 27");
        assert_eq!(arr[28].as_str(), Some("pk"), "bootstrap_pair_key at 28");
        assert_eq!(arr[29].as_i64(), Some(2), "decode_tp_size at 29");
        assert_eq!(arr[30].as_i64(), Some(3), "routed_dp_rank at 30");
        assert_eq!(arr[31].as_i64(), Some(4), "disagg_prefill_dp_rank at 31");
    }
}
