//! The scheduler wire structs — the Rust mirror of the Python `io_struct`
//! messages this server sends (`python/sglang/srt/managers/io_struct.py`).
//! Each is a msgspec `array_like=True` struct, so **field order is wire order**
//! and `rmp_serde`'s default struct-as-array encoding reproduces it.

use bytes::Bytes;
use serde::Serialize;

use super::request::PdBootstrap;
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
        return_hidden_states: bool,
        /// Versioned Rust-PD identity/digest map. Appended at the Python-owned
        /// extension slot so upstream positional fields retain their indices.
        pd_sidecar: Option<&'a rmpv::Value>,
    }
}

/// PD requests must also populate Python's legacy bootstrap identity at indices
/// 25..27. Keep the ordinary header at its upstream length (17 elements) and
/// append the intervening Python defaults only when a bootstrap is present.
struct TokenizedGenerateReqInputWithPdBootstrap<'header, 'request> {
    header: &'header TokenizedGenerateReqInput<'request>,
    bootstrap: &'header PdBootstrap,
}

impl Serialize for TokenizedGenerateReqInputWithPdBootstrap<'_, '_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct; // codespell:ignore ser

        let header = self.header;
        let mut st = serializer.serialize_struct("TokenizedGenerateReqInput", 28)?;
        st.serialize_field("tag", <TokenizedGenerateReqInput<'_> as Tagged>::TAG)?;
        st.serialize_field("rid", &header.rid)?;
        st.serialize_field("http_worker_ipc", &())?;
        st.serialize_field("input_text", &header.input_text)?;
        st.serialize_field("input_ids", &header.input_ids)?;
        st.serialize_field("input_embeds", &header.input_embeds)?;
        st.serialize_field("mm_inputs", &header.mm_inputs)?;
        st.serialize_field("token_type_ids", &header.token_type_ids)?;
        st.serialize_field("sampling_params", &header.sampling_params)?;
        st.serialize_field("return_logprob", &header.return_logprob)?;
        st.serialize_field("logprob_start_len", &header.logprob_start_len)?;
        st.serialize_field("top_logprobs_num", &header.top_logprobs_num)?;
        st.serialize_field("token_ids_logprob", &header.token_ids_logprob)?;
        st.serialize_field("stream", &header.stream)?;
        st.serialize_field("return_sampling_mask", &header.return_sampling_mask)?;
        st.serialize_field("return_hidden_states", &header.return_hidden_states)?;
        st.serialize_field("pd_sidecar", &header.pd_sidecar)?;
        st.serialize_field("return_routed_experts", &false)?;
        st.serialize_field("routed_experts_start_len", &0_i64)?;
        st.serialize_field("return_indexer_topk", &false)?;
        st.serialize_field("session_id", &())?;
        st.serialize_field("session_params", &())?;
        st.serialize_field("lora_id", &())?;
        st.serialize_field("custom_logit_processor", &())?;
        st.serialize_field("positional_embed_overrides", &())?;
        st.serialize_field("bootstrap_host", &self.bootstrap.host)?;
        st.serialize_field("bootstrap_port", &self.bootstrap.port)?;
        st.serialize_field("bootstrap_room", &self.bootstrap.room)?;
        st.end()
    }
}

impl TokenizedGenerateReqInput<'_> {
    pub(super) fn encode_with_pd_bootstrap(&self, bootstrap: &PdBootstrap) -> Result<Bytes, Error> {
        rmp_serde::to_vec(&TokenizedGenerateReqInputWithPdBootstrap {
            header: self,
            bootstrap,
        })
        .map(Bytes::from)
        .map_err(|e| Error::Codec(e.to_string()))
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
            return_hidden_states: req.return_hidden_states,
            pd_sidecar: req.pd_sidecar.as_ref(),
        }
    }
}

impl GetInternalStateReq {
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
            stream: true,
            pd_sidecar: Some(rmpv::Value::Map(vec![(
                rmpv::Value::from("version"),
                rmpv::Value::from(1),
            )])),
            ..Default::default()
        };
        let bytes = TokenizedGenerateReqInput::from(&req).encode().unwrap();
        let val = rmpv::decode::read_value(&mut &bytes[..]).unwrap();
        let arr = val.as_array().expect("array");
        // No bootstrap means the upstream header remains exactly 17 elements.
        assert_eq!(arr.len(), 17);
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
            Some(true),
            "return_hidden_states at idx 15"
        );
        assert!(arr[16].is_map(), "pd_sidecar must land at idx 16");
    }

    #[test]
    fn pd_header_reaches_legacy_bootstrap_slots() {
        let req = GenerateRequest {
            rid: "r1".into(),
            sampling_params: SamplingParams::default(),
            pd_bootstrap: Some(PdBootstrap {
                host: "prefill.internal".into(),
                port: 8998,
                room: 42,
                attempt_id: "44444444-4444-4444-8444-444444444444".into(),
                batch_index: 0,
            }),
            ..Default::default()
        };

        let bytes = req.encode_header().unwrap();
        let val = rmpv::decode::read_value(&mut &bytes[..]).unwrap();
        let arr = val.as_array().expect("array");
        assert_eq!(arr.len(), 28);
        assert_eq!(arr[0].as_str(), Some("TokenizedGenerateReqInput"));
        assert_eq!(arr[1].as_str(), Some("r1"));
        assert!(arr[8].is_array(), "sampling_params must land at idx 8");
        assert_eq!(
            arr[25].as_str(),
            Some("prefill.internal"),
            "bootstrap_host at idx 25"
        );
        assert_eq!(arr[26].as_u64(), Some(8998), "bootstrap_port at idx 26");
        assert_eq!(arr[27].as_u64(), Some(42), "bootstrap_room at idx 27");
    }
}
