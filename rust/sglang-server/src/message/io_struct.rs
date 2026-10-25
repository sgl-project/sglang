//! The scheduler wire structs — the Rust mirror of the Python `io_struct`
//! messages this server sends (`python/sglang/srt/managers/io_struct.py`).
//! Each is a msgspec `array_like=True` struct, so **field order is wire order**
//! and `rmp_serde`'s default struct-as-array encoding reproduces it.

use bytes::Bytes;
use serde::Serialize;

use super::request::GenerateRequest;
use super::sampling::SamplingParams;
use super::types::{Tagged, control_messages, wire_struct};
use crate::utils::error::Error;

wire_struct! {
    /// The scheduler's `TokenizedGenerateReqInput`. Keep in lockstep with the
    /// Python declaration: inserting a field anywhere but the end shifts every
    /// later field on the wire.
    pub(super) TokenizedGenerateReqInput<'a> {
        input_text: Option<&'a str>,
        /// Always nil: the ids ride the ring as the `input_ids` buffer, not msgpack.
        input_ids: (),
        input_embeds: (),
        mm_inputs: (),
        token_type_ids: (),
        sampling_params: &'a SamplingParams,
        return_logprob: bool,
        logprob_start_len: i64,
        top_logprobs_num: i64,
        /// Always nil: rides the ring as the `token_ids_logprob` buffer.
        token_ids_logprob: (),
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
        lora_id: Option<&'a str>,
        custom_logit_processor: (),
        positional_embed_overrides: (),
        /// PD-disaggregation block. Everything after `kv_hints` (the last
        /// field emitted) has a msgspec default in Python and is omitted.
        bootstrap_host: Option<&'a str>,
        bootstrap_port: Option<i64>,
        bootstrap_room: Option<i64>,
        bootstrap_pair_key: Option<&'a str>,
        decode_tp_size: Option<i64>,
        routed_dp_rank: Option<i64>,
        disagg_prefill_dp_rank: Option<i64>,
        routing_key: Option<&'a str>,
        require_reasoning: bool,
        priority: Option<i64>,
        extra_key: Option<&'a str>,
        no_logs: bool,
        return_bytes: bool,
        return_entropy: bool,
        need_wait_for_mm_inputs: (),
        num_items_assigned: (),
        encoder_urls: (),
        multi_item_delimiter_indices: (),
        token_indices_to_pool: (),
        time_stats: (),
        cache_salt: Option<&'a str>,
        kv_hints: Option<&'a super::request::KvHintsEnvelope>,
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

    /// `/flush_cache`'s control request: the scheduler flushes its caches in
    /// its own loop, waiting up to `timeout_s` for running requests to drain.
    FlushCacheReqInput {
        timeout_s: Option<f64>,
    }

    /// `/weights_checker`'s control request: every scheduler rank runs the
    /// weight checker's `action` on the runners `selector` names.
    CheckWeightsReqInput {
        action: String,
        allow_quant_error: bool,
        skip_tensor_list: Option<Vec<String>>,
        selector: String,
    }

    /// `/set_internal_state`'s control request: the numeric knobs to
    /// override, as the JSON object the client sent (`Dict[str, int | float]`
    /// on the Python side; the scheduler validates the keys).
    SetInternalStateReq {
        server_args: serde_json::Map<String, serde_json::Value>,
    }

    /// `/start_profile` and `/stop_profile`'s control request, Python's
    /// `ProfileReq`. `req_type` is the `ProfileReqType` enum's value on the
    /// wire: 1 starts, 2 stops.
    ProfileReq {
        req_type: u8,
        output_dir: Option<String>,
        start_step: Option<i64>,
        num_steps: Option<i64>,
        activities: Option<Vec<String>>,
        profile_by_stage: bool,
        with_stack: Option<bool>,
        record_shapes: Option<bool>,
        profile_id: Option<String>,
        merge_profiles: bool,
        profile_prefix: Option<String>,
        profile_stages: Option<Vec<String>>,
        detailed_annotations: bool,
    }

    /// The expert-distribution recorder; `action` is `ExpertDistributionReqType`'s
    /// value: 1 start, 2 stop, 3 dump.
    ExpertDistributionReq {
        action: u8,
    }

    /// `/slow_down`: Python's `SlowDownReqInput`.
    SlowDownReqInput {
        forward_sleep_time: Option<f64>,
    }

    /// `/release_memory_occupation`: Python's `ReleaseMemoryOccupationReqInput`.
    ReleaseMemoryOccupationReqInput {
        tags: Option<Vec<String>>,
    }

    /// `/resume_memory_occupation`: Python's `ResumeMemoryOccupationReqInput`.
    ResumeMemoryOccupationReqInput {
        tags: Option<Vec<String>>,
    }

    /// `/get_weights_by_name`: Python's `GetWeightsByNameReqInput`.
    GetWeightsByNameReqInput {
        name: String,
        truncate_size: i64,
    }

    /// `/init_weights_update_group`: Python's `InitWeightsUpdateGroupReqInput`.
    InitWeightsUpdateGroupReqInput {
        master_address: String,
        master_port: i64,
        rank_offset: i64,
        world_size: i64,
        group_name: String,
        backend: String,
    }

    /// `/destroy_weights_update_group`.
    DestroyWeightsUpdateGroupReqInput {
        group_name: String,
    }

    /// `/init_weights_send_group_for_remote_instance`.
    InitWeightsSendGroupForRemoteInstanceReqInput {
        master_address: String,
        ports: String,
        group_rank: i64,
        world_size: i64,
        group_name: String,
        backend: String,
    }

    /// `/send_weights_to_remote_instance`.
    SendWeightsToRemoteInstanceReqInput {
        master_address: String,
        ports: String,
        group_name: String,
    }

    /// `/add_external_corpus`: Python's `AddExternalCorpusReqInput`, after
    /// the frontend tokenized `file_path` / `documents` into `token_chunks`.
    AddExternalCorpusReqInput {
        corpus_id: Option<String>,
        file_path: Option<String>,
        documents: Option<Vec<String>>,
        token_chunks: Option<Vec<Vec<i64>>>,
    }

    /// `/remove_external_corpus`.
    RemoveExternalCorpusReqInput {
        corpus_id: String,
    }

    /// `/list_external_corpora`.
    ListExternalCorporaReqInput {}

    /// `/freeze_gc`: Python's `FreezeGCReq`.
    FreezeGCReq {
        gc_threshold: Option<Vec<i64>>,
    }

    /// `/pause_generation`: Python's `PauseGenerationReqInput`. The scheduler
    /// accepts `in_place` and `retract`; `abort` never reaches it.
    PauseGenerationReqInput {
        mode: String,
    }

    /// `/continue_generation`: Python's `ContinueGenerationReqInput`.
    ContinueGenerationReqInput {
        torch_empty_cache: bool,
    }

    /// `/pd_role_switch`: Python's `PdRoleSwitchReqInput`.
    PdRoleSwitchReqInput {
        new_role: String,
        decode_cuda_graph_bs: Option<Vec<i64>>,
        decode_cuda_graph_memory_gb: Option<f64>,
        prepare_only: bool,
    }

    /// `/update_weight_version`: Python's `UpdateWeightVersionReqInput`.
    UpdateWeightVersionReqInput {
        new_version: String,
        abort_all_requests: bool,
    }

    /// Clear the HiCache storage backend.
    ClearHiCacheReqInput {}

    /// Detach the HiCache storage backend.
    DetachHiCacheStorageReqInput {}

    /// Attach a HiCache storage backend with optional configuration and policies.
    AttachHiCacheStorageReqInput {
        hicache_storage_backend: String,
        hicache_storage_backend_extra_config_json: Option<String>,
        hicache_storage_prefetch_policy: Option<String>,
        hicache_write_policy: Option<String>,
    }
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
            token_ids_logprob: (),
            stream: req.stream,
            return_sampling_mask: req.return_sampling_mask,
            return_flat_raw_top_logprobs: false,
            return_hidden_states: req.return_hidden_states,
            return_routed_experts: false,
            routed_experts_start_len: 0,
            return_indexer_topk: false,
            session_id: (),
            session_params: (),
            lora_id: req.lora_id.as_deref(),
            custom_logit_processor: (),
            positional_embed_overrides: (),
            bootstrap_host: req.bootstrap_host.as_deref(),
            bootstrap_port: req.bootstrap_port,
            bootstrap_room: req.bootstrap_room,
            bootstrap_pair_key: req.bootstrap_pair_key.as_deref(),
            decode_tp_size: req.decode_tp_size,
            routed_dp_rank: req.routed_dp_rank,
            disagg_prefill_dp_rank: req.disagg_prefill_dp_rank,
            routing_key: req.routing_key.as_deref(),
            require_reasoning: req.require_reasoning,
            priority: req.priority,
            extra_key: req.extra_key.as_deref(),
            no_logs: false,
            return_bytes: false,
            return_entropy: false,
            need_wait_for_mm_inputs: (),
            num_items_assigned: (),
            encoder_urls: (),
            multi_item_delimiter_indices: (),
            token_indices_to_pool: (),
            time_stats: (),
            cache_salt: req.cache_salt.as_deref(),
            kv_hints: req.kv_hints.as_ref(),
        }
    }
}

impl SetInternalStateReq {
    pub fn new(rid: String, server_args: serde_json::Map<String, serde_json::Value>) -> Self {
        Self { rid, server_args }
    }
}

impl GetWeightsByNameReqInput {
    pub fn new(rid: String, name: String, truncate_size: i64) -> Self {
        Self {
            rid,
            name,
            truncate_size,
        }
    }
}

/// The `/init_weights_update_group` fields; the owned form of
/// `api::InitWeightsUpdateGroupRequest` after its defaults are applied.
#[derive(Debug, Clone)]
pub struct WeightsUpdateGroup {
    pub master_address: String,
    pub master_port: i64,
    pub rank_offset: i64,
    pub world_size: i64,
    pub group_name: String,
    pub backend: String,
}

impl InitWeightsUpdateGroupReqInput {
    pub fn new(rid: String, group: WeightsUpdateGroup) -> Self {
        Self {
            rid,
            master_address: group.master_address,
            master_port: group.master_port,
            rank_offset: group.rank_offset,
            world_size: group.world_size,
            group_name: group.group_name,
            backend: group.backend,
        }
    }
}

impl DestroyWeightsUpdateGroupReqInput {
    pub fn new(rid: String, group_name: String) -> Self {
        Self { rid, group_name }
    }
}

/// The `/init_weights_send_group_for_remote_instance` fields.
#[derive(Debug, Clone)]
pub struct WeightsSendGroup {
    pub master_address: String,
    pub ports: String,
    pub group_rank: i64,
    pub world_size: i64,
    pub group_name: String,
    pub backend: String,
}

impl InitWeightsSendGroupForRemoteInstanceReqInput {
    pub fn new(rid: String, group: WeightsSendGroup) -> Self {
        Self {
            rid,
            master_address: group.master_address,
            ports: group.ports,
            group_rank: group.group_rank,
            world_size: group.world_size,
            group_name: group.group_name,
            backend: group.backend,
        }
    }
}

impl SendWeightsToRemoteInstanceReqInput {
    pub fn new(rid: String, master_address: String, ports: String, group_name: String) -> Self {
        Self {
            rid,
            master_address,
            ports,
            group_name,
        }
    }
}

impl AddExternalCorpusReqInput {
    pub fn new(rid: String, corpus_id: String, token_chunks: Vec<Vec<i64>>) -> Self {
        Self {
            rid,
            corpus_id: Some(corpus_id),
            file_path: None,
            documents: None,
            token_chunks: Some(token_chunks),
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

impl FreezeGCReq {
    pub fn new(rid: String, gc_threshold: Option<Vec<i64>>) -> Self {
        Self { rid, gc_threshold }
    }
}

impl PauseGenerationReqInput {
    pub fn new(rid: String, mode: String) -> Self {
        Self { rid, mode }
    }
}

impl ContinueGenerationReqInput {
    pub fn new(rid: String, torch_empty_cache: bool) -> Self {
        Self {
            rid,
            torch_empty_cache,
        }
    }
}

/// The `/pd_role_switch` fields; the owned form of `api::PdRoleSwitchRequest`.
#[derive(Debug, Clone)]
pub struct PdRoleSwitch {
    pub new_role: String,
    pub decode_cuda_graph_bs: Option<Vec<i64>>,
    pub decode_cuda_graph_memory_gb: Option<f64>,
    pub prepare_only: bool,
}

impl PdRoleSwitchReqInput {
    pub fn new(rid: String, switch: PdRoleSwitch) -> Self {
        Self {
            rid,
            new_role: switch.new_role,
            decode_cuda_graph_bs: switch.decode_cuda_graph_bs,
            decode_cuda_graph_memory_gb: switch.decode_cuda_graph_memory_gb,
            prepare_only: switch.prepare_only,
        }
    }
}

impl UpdateWeightVersionReqInput {
    pub fn new(rid: String, new_version: String, abort_all_requests: bool) -> Self {
        Self {
            rid,
            new_version,
            abort_all_requests,
        }
    }
}

/// `ExpertDistributionReqType` on the wire.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExpertDistributionAction {
    Start = 1,
    Stop = 2,
    Dump = 3,
}

impl ExpertDistributionReq {
    pub fn new(rid: String, action: ExpertDistributionAction) -> Self {
        Self {
            rid,
            action: action as u8,
        }
    }
}

impl SlowDownReqInput {
    pub fn new(rid: String, forward_sleep_time: Option<f64>) -> Self {
        Self {
            rid,
            forward_sleep_time,
        }
    }
}

impl ReleaseMemoryOccupationReqInput {
    pub fn new(rid: String, tags: Option<Vec<String>>) -> Self {
        Self { rid, tags }
    }
}

impl ResumeMemoryOccupationReqInput {
    pub fn new(rid: String, tags: Option<Vec<String>>) -> Self {
        Self { rid, tags }
    }
}

/// `ProfileReqType.START_PROFILE` / `STOP_PROFILE` on the wire.
pub const PROFILE_REQ_START: u8 = 1;
pub const PROFILE_REQ_STOP: u8 = 2;

/// The `/start_profile` fields a client may set, before the frontend fills
/// its defaults; the owned form of `api::StartProfileRequest`.
#[derive(Debug, Clone, Default)]
pub struct ProfileOptions {
    pub output_dir: Option<String>,
    pub start_step: Option<i64>,
    pub num_steps: Option<i64>,
    pub activities: Option<Vec<String>>,
    pub profile_by_stage: bool,
    pub with_stack: Option<bool>,
    pub record_shapes: Option<bool>,
    pub profile_id: Option<String>,
    pub merge_profiles: bool,
    pub profile_prefix: Option<String>,
    pub profile_stages: Option<Vec<String>>,
    pub detailed_annotations: bool,
}

impl ProfileReq {
    pub fn start(rid: String, options: ProfileOptions) -> Self {
        Self {
            rid,
            req_type: PROFILE_REQ_START,
            output_dir: options.output_dir,
            start_step: options.start_step,
            num_steps: options.num_steps,
            activities: options.activities,
            profile_by_stage: options.profile_by_stage,
            with_stack: options.with_stack,
            record_shapes: options.record_shapes,
            profile_id: options.profile_id,
            merge_profiles: options.merge_profiles,
            profile_prefix: options.profile_prefix,
            profile_stages: options.profile_stages,
            detailed_annotations: options.detailed_annotations,
        }
    }

    pub fn stop(rid: String) -> Self {
        Self {
            rid,
            req_type: PROFILE_REQ_STOP,
            output_dir: None,
            start_step: None,
            num_steps: None,
            activities: None,
            profile_by_stage: false,
            with_stack: None,
            record_shapes: None,
            profile_id: None,
            merge_profiles: false,
            profile_prefix: None,
            profile_stages: None,
            detailed_annotations: false,
        }
    }
}

impl ClearHiCacheReqInput {
    pub fn new(rid: String) -> Self {
        Self { rid }
    }
}

impl DetachHiCacheStorageReqInput {
    pub fn new(rid: String) -> Self {
        Self { rid }
    }
}

impl AttachHiCacheStorageReqInput {
    pub fn new(
        rid: String,
        hicache_storage_backend: String,
        hicache_storage_backend_extra_config_json: Option<String>,
        hicache_storage_prefetch_policy: Option<String>,
        hicache_write_policy: Option<String>,
    ) -> Self {
        Self {
            rid,
            hicache_storage_backend,
            hicache_storage_backend_extra_config_json,
            hicache_storage_prefetch_policy,
            hicache_write_policy,
        }
    }
}

impl GetInternalStateReq {
    pub fn new(rid: String) -> Self {
        Self { rid }
    }
}

impl FlushCacheReqInput {
    pub fn new(rid: String, timeout_s: Option<f64>) -> Self {
        Self { rid, timeout_s }
    }
}

impl CheckWeightsReqInput {
    pub fn new(
        rid: String,
        action: String,
        allow_quant_error: bool,
        skip_tensor_list: Option<Vec<String>>,
        selector: String,
    ) -> Self {
        Self {
            rid,
            action,
            allow_quant_error,
            skip_tensor_list,
            selector,
        }
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
    fn flush_cache_req_msgpack_shape() {
        let b = FlushCacheReqInput::new("7".into(), Some(30.0))
            .encode()
            .unwrap();
        let val = rmpv::decode::read_value(&mut &b[..]).unwrap();
        let arr = val.as_array().expect("array");
        assert_eq!(
            arr.len(),
            4,
            "FlushCacheReqInput = [tag, rid, http_ipc, timeout_s]"
        );
        assert_eq!(arr[0].as_str(), Some("FlushCacheReqInput"));
        assert_eq!(arr[1].as_str(), Some("7"));
        assert!(arr[2].is_nil());
        assert_eq!(arr[3].as_f64(), Some(30.0));
    }

    #[test]
    fn check_weights_req_msgpack_shape() {
        let b = CheckWeightsReqInput::new(
            "7".into(),
            "fingerprint".into(),
            true,
            Some(vec!["lm_head".into()]),
            "target".into(),
        )
        .encode()
        .unwrap();
        let val = rmpv::decode::read_value(&mut &b[..]).unwrap();
        let arr = val.as_array().expect("array");
        assert_eq!(
            arr.len(),
            7,
            "CheckWeightsReqInput = [tag, rid, http_ipc, action, \
             allow_quant_error, skip_tensor_list, selector]"
        );
        assert_eq!(arr[0].as_str(), Some("CheckWeightsReqInput"));
        assert_eq!(arr[1].as_str(), Some("7"));
        assert!(arr[2].is_nil());
        assert_eq!(arr[3].as_str(), Some("fingerprint"));
        assert_eq!(arr[4].as_bool(), Some(true));
        assert_eq!(arr[5][0].as_str(), Some("lm_head"));
        assert_eq!(arr[6].as_str(), Some("target"));
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
            ..Default::default()
        };
        let bytes = TokenizedGenerateReqInput::from(&req).encode().unwrap();
        let val = rmpv::decode::read_value(&mut &bytes[..]).unwrap();
        let arr = val.as_array().expect("array");
        // The orchestrator controls extend the array through `kv_hints`.
        assert_eq!(arr.len(), 47, "header ends at kv_hints");
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
    }

    /// Python's `KvHintsEnvelope` / `KvHintAction` are dict-shaped msgspec
    /// structs, so a positional array at the kv_hints slot fails to decode.
    #[test]
    fn header_kv_hints_are_map_encoded() {
        use sglang_api_types::api::v1 as api;
        let kv_hints = crate::message::request::KvHintsEnvelope::try_from(api::KvHintsEnvelope {
            protocol_version: "1".into(),
            message_id: "m".into(),
            actions: vec![api::KvHintAction {
                action_id: "a".into(),
                action_type: "pin".into(),
                action_version: "1".into(),
                payload: Some(r#"{"blocks":[1,2]}"#.into()),
            }],
        })
        .unwrap();
        let req = GenerateRequest {
            rid: "r1".into(),
            input_ids: Some(vec![1]),
            kv_hints: Some(kv_hints),
            ..Default::default()
        };
        let bytes = TokenizedGenerateReqInput::from(&req).encode().unwrap();
        let val = rmpv::decode::read_value(&mut &bytes[..]).unwrap();
        let envelope = val.as_array().expect("array")[46]
            .as_map()
            .expect("kv_hints must be a map");
        let get = |map: &[(rmpv::Value, rmpv::Value)], key: &str| {
            map.iter()
                .find(|(k, _)| k.as_str() == Some(key))
                .map(|(_, v)| v.clone())
                .unwrap_or_else(|| panic!("missing key {key}"))
        };
        assert_eq!(get(envelope, "protocol_version").as_str(), Some("1"));
        assert_eq!(get(envelope, "message_id").as_str(), Some("m"));
        let actions = get(envelope, "actions");
        let action = actions.as_array().unwrap()[0]
            .as_map()
            .expect("kv_hints action must be a map");
        assert_eq!(get(action, "action_type").as_str(), Some("pin"));
        assert!(get(action, "payload").is_map());
    }

    /// The PD block must land on Python's wire indices 25–31, with the filler
    /// block (17–24) holding its defaults — a shift here silently routes KV
    /// transfers to the wrong host/room. The cache keys and `require_reasoning`
    /// follow at 33–45.
    #[test]
    fn header_bootstrap_block_and_cache_keys_are_positionally_aligned() {
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
            extra_key: Some("salt".into()),
            cache_salt: Some("pepper".into()),
            priority: Some(-2),
            require_reasoning: true,
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
        assert!(arr[32].is_nil(), "routing_key at 32");
        assert_eq!(arr[33].as_bool(), Some(true), "require_reasoning at 33");
        assert_eq!(arr[34].as_i64(), Some(-2), "priority at 34");
        assert_eq!(arr[35].as_str(), Some("salt"), "extra_key at 35");
        for (i, slot) in arr.iter().enumerate().take(39).skip(36) {
            assert_eq!(slot.as_bool(), Some(false), "idx {i} must be false");
        }
        for (i, slot) in arr.iter().enumerate().take(45).skip(39) {
            assert!(slot.is_nil(), "idx {i} must be a nil default");
        }
        assert_eq!(arr[45].as_str(), Some("pepper"), "cache_salt at 45");
    }
}
