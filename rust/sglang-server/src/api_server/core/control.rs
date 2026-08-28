//! Transport-neutral control-plane logic: the control-request round-trip on
//! the shared `submit`, and the `/server_info` + model-info body shaping.

use crate::api_server::core::error::ApiError;
use crate::api_server::core::guard::AbortGuard;
use crate::api_server::core::state::CoreState;
use crate::api_server::core::submit::submit;
use crate::message::config::ServerArgs;
use crate::message::ids::Rid;
use crate::message::io_struct::{ControlRequest, GetInternalStateReq};
use crate::message::request::RequestKind;
use crate::message::response::ResponseItem;

/// Submit a control request through the request FSM (no tokenization) and await the
/// scheduler's single msgpack result (a `structs.asdict` named map). Returns the
/// raw bytes, or an [`ApiError`] for the caller to shape.
pub(crate) async fn await_control_result(
    state: &CoreState,
    control: ControlRequest,
) -> Result<bytes::Bytes, ApiError> {
    let (rid, mut rx) = submit(state, RequestKind::Control(Box::new(control))).await?;
    // Control requests register a detok entry like any other, and only
    // `handle_result` removes it — so a request that never produces one (a stalled
    // scheduler, a client that hangs up mid-await) leaves the entry behind. A
    // monitor polling `/server_info` then leaks one `DetokState` per poll, forever.
    // The guard deregisters on drop; it is disarmed below when the result lands.
    let mut guard = AbortGuard::new(state.senders.clone(), rid.clone());
    let received = rx.recv().await;
    if received.is_some() {
        guard.disarm(&rid); // completed normally — nothing to abort
    }
    match received {
        Some(ResponseItem::Control(bytes)) => Ok(bytes),
        Some(ResponseItem::Error(e)) => Err(ApiError::from_pipeline(&e)),
        // A control request never receives generation frames or service-call data.
        Some(ResponseItem::Frame(_))
        | Some(ResponseItem::Done(_))
        | Some(ResponseItem::Data(_)) => Err(ApiError::internal(
            "unexpected generation output for control request",
        )),
        None => Err(ApiError::new(499, "request aborted")),
    }
}

/// Static model metadata from `server_args` (no scheduler round-trip);
/// `is_generation` always true.
pub(crate) fn model_info_value(sa: &ServerArgs) -> serde_json::Value {
    serde_json::json!({
        "model_path": sa.model_path,
        "served_model_name": sa.served_model_name,
        "tokenizer_path": sa.tokenizer_path,
        "is_generation": true,
        // Python's `TokenizerManager` merges this into every request
        // (`{**preferred, **client}`); this server has no equivalent yet, so
        // `RustServer.launch` REFUSES to start when it is set. It can therefore
        // only be null here — echoing it keeps the field's shape.
        "preferred_sampling_params": sa.preferred_sampling_params,
        // Python answers this through `config_value`, so a control-plane write
        // moves it there; here it is the launch value.
        "weight_version": sa.weight_version,
        "load_format": sa.load_format,
        // `auto` never reaches the blob: `resolve_auto_parsers` writes the
        // selected parser into `server_args` before the scheduler forks.
        "reasoning_parser": sa.reasoning_parser,
        "tool_call_parser": sa.tool_call_parser,
    })
}

/// The `/server_info` body: one scheduler round-trip shaped through the
/// allowlist ([`INTERNAL_STATE_ALLOWLIST`] + curated [`ServerArgs`]
/// accessors), never the raw server-args dump (embeds
/// `api_key`/`admin_api_key`; see [`shape_server_info`]).
///
/// TODO(server_info): Python also includes `kv_events`; add once plumbed.
pub(crate) async fn server_info_json(state: &CoreState) -> Result<Vec<u8>, ApiError> {
    let bytes = await_control_result(
        state,
        ControlRequest::GetInternalStateReq(GetInternalStateReq::new(Rid::new().to_string())),
    )
    .await?;
    shape_server_info(&bytes, &state.server_args).map_err(|e| {
        tracing::error!(error = %e, "server_info: shaping failed");
        ApiError::internal("bad server_info response")
    })
}

/// Runtime-metric keys `get_internal_state` adds atop the server-args dump. We copy
/// ONLY these out of `internal_state` (an allowlist), so the co-mingled
/// `api_key`/`admin_api_key` can never reach the response.
const INTERNAL_STATE_ALLOWLIST: &[&str] = &[
    "last_gen_throughput",
    "memory_usage",
    "effective_max_running_requests_per_dp",
    "avg_spec_accept_length",
    "step_time_dict",
];

fn shape_server_info(msgpack: &[u8], server_args: &ServerArgs) -> Result<Vec<u8>, String> {
    // GetInternalStateReqOutput asdict → `{ "internal_state": { server-args dump +
    // metrics }, ... }`. Pull that inner map out (it is NOT safe to expose whole).
    let mut obj: serde_json::Map<String, serde_json::Value> =
        rmp_serde::from_slice(msgpack).map_err(|e| e.to_string())?;
    let internal = match obj.remove("internal_state") {
        Some(serde_json::Value::Object(m)) => m,
        _ => serde_json::Map::new(),
    };

    // Copy only the allowlisted runtime metrics — never the raw server-args dump.
    let mut state_out = serde_json::Map::new();
    for &k in INTERNAL_STATE_ALLOWLIST {
        match internal.get(k) {
            Some(v) if !v.is_null() => {
                state_out.insert(k.to_string(), v.clone());
            }
            _ => {}
        }
    }

    // Top-level non-secret config from typed accessors (structurally can't surface
    // a key field, unlike the raw dump).
    let response = serde_json::json!({
        "model_path": server_args.model_path,
        "served_model_name": server_args.served_model_name,
        "tokenizer_path": server_args.tokenizer_path,
        "max_context_length": server_args.model_config.context_len,
        "max_total_num_tokens": server_args.max_total_num_tokens,
        "version": server_args.version,
        "internal_states": [serde_json::Value::Object(state_out)],
    });
    serde_json::to_vec(&response).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The scheduler's `internal_state` embeds the full server-args dump (incl.
    /// `api_key`/`admin_api_key`). `/server_info` must surface only the allowlisted
    /// runtime metrics + curated config — never the secrets — and must not re-nest
    /// the dump under `internal_states[].internal_state`.
    #[test]
    fn shape_server_info_excludes_secrets_and_dump() {
        // GetInternalStateReqOutput.asdict → { "internal_state": { …dump+metrics… } }.
        let internal = rmpv::Value::Map(vec![
            (
                rmpv::Value::from("api_key"),
                rmpv::Value::from("secret-token"),
            ),
            (
                rmpv::Value::from("admin_api_key"),
                rmpv::Value::from("admin-token"),
            ),
            (rmpv::Value::from("model_path"), rmpv::Value::from("/m")),
            (
                rmpv::Value::from("last_gen_throughput"),
                rmpv::Value::from(1.5),
            ),
            (
                rmpv::Value::from("effective_max_running_requests_per_dp"),
                rmpv::Value::from(32),
            ),
        ]);
        let outer = rmpv::Value::Map(vec![(rmpv::Value::from("internal_state"), internal)]);
        let mut msgpack = Vec::new();
        rmpv::encode::write_value(&mut msgpack, &outer).unwrap();

        // `api_key` is deliberately NOT a `ServerArgs` field — the typed schema
        // cannot carry it — so the only place it could leak from is the raw
        // scheduler dump shaped above.
        let sa = ServerArgs {
            model_path: "/m".into(),
            ..Default::default()
        };
        let out = shape_server_info(&msgpack, &sa).unwrap();
        let text = String::from_utf8(out.clone()).unwrap();
        // No secret leaks anywhere in the serialized response.
        assert!(!text.contains("secret-token"), "api_key leaked: {text}");
        assert!(
            !text.contains("admin-token"),
            "admin_api_key leaked: {text}"
        );

        let v: serde_json::Value = serde_json::from_slice(&out).unwrap();
        // Allowlisted metric surfaced; the whole dump did not.
        let state0 = &v["internal_states"][0];
        assert_eq!(state0["last_gen_throughput"], 1.5);
        assert_eq!(state0["effective_max_running_requests_per_dp"], 32);
        assert!(
            state0.get("internal_state").is_none(),
            "must not re-nest the dump under internal_state"
        );
        assert!(state0.get("api_key").is_none());
        // Curated top-level config comes from typed accessors, not the dump.
        assert_eq!(v["model_path"], "/m");
    }
}
