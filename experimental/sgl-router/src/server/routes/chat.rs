// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::config::{
    ConflictPolicy, ParamSpec, PolicyKind, SamplingField, SamplingOverrides, SessionAffinityMode,
};
use crate::discovery::{ModelId, WorkerMode};
use crate::policies::admission::{
    resolve_cache_candidates, resolve_decode, resolve_prefill, resolve_prefill_admitted,
    CandidateDomain, CandidateRange, DecisionReason,
};
use crate::policies::buckets::BucketRequest;
use crate::policies::decode::{
    build_decode_policy, resolve_decode_with_capacity_fallback, DecodeSelectionContext,
};
use crate::policies::kv_events::{compute_block_hashes, compute_block_hashes_bigram};
use crate::policies::registry::{PdPoolResolver, PdResolveError};
use crate::policies::{
    request_tokens_for, ExternalPrefixSignal, PrefillProposal, ProposalKind, RequestTokens,
    SelectionContext,
};
use crate::server::app_context::AppContext;
use crate::server::error::ApiError;
use crate::server::metrics::{
    MetricsRegistry, PolicySelectionFailureReason, RequestOutcome, StaleRequestOutcome,
    WorkerModeLabel,
};
use crate::workers::{LoadGuard, Worker};
use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderMap, HeaderName, HeaderValue, Response};
use bytes::Bytes;
use serde::de::IgnoredAny;
use serde::Deserialize;
use std::cell::Cell;
use std::sync::Arc;

/// Observability header carrying the final decode-pool URL for a
/// PD-disaggregated request. The router fans the
/// bootstrap-injected request body to BOTH the prefill and the decode
/// worker concurrently; this header lets the prefill log the chosen
/// peer, and is mirrored onto the response so sidecars / tests can
/// observe affinity without sniffing the proxy hop. The `x-sgl-`
/// prefix matches `x-sgl-router-error-code` so router-emitted metadata
/// stays grouped.
const X_SGL_DECODE_URL: HeaderName = HeaderName::from_static("x-sgl-decode-url");
/// Optional caller requirement consumed only when a static P Bucket config is enabled.
const X_SGL_TTFT_SLO_MS: HeaderName = HeaderName::from_static("x-sgl-ttft-slo-ms");
/// Optional caller TPS requirement consumed only when a static D Bucket config is enabled.
const X_SGL_TPS_SLO: HeaderName = HeaderName::from_static("x-sgl-tps-slo");

/// Coarse char-count → token-count divisor used to estimate prefill load
/// from the request body when no real tokenizer count is available. Four
/// bytes per token is the standard SGLang upstream estimate; it
/// overcounts ASCII and undercounts CJK but stays within an order of
/// magnitude of the real token count, which is plenty for load
/// scoring. The active-load counters' role is relative ordering across
/// workers — not absolute accuracy — so the estimate is fit for
/// purpose.
const CHARS_PER_TOKEN_ESTIMATE: usize = 4;

/// Return the low-cardinality reason for the final Prefill decision.
fn prefill_policy_reason(
    policy: PolicyKind,
    proposal: ProposalKind,
    decision: DecisionReason,
    has_session_id: bool,
    affinity_lookup_enabled: bool,
) -> &'static str {
    match policy {
        PolicyKind::SessionAware => match (proposal, decision) {
            (ProposalKind::SessionAffinity, DecisionReason::Primary) => "session_primary",
            (ProposalKind::SessionAffinity, DecisionReason::BackupPrimaryAdmission) => {
                "session_admission_backup"
            }
            (ProposalKind::SessionAffinity, DecisionReason::BackupPressureGuard) => {
                "session_pressure_backup"
            }
            (ProposalKind::SessionAffinity, DecisionReason::RangeFallback) => {
                "session_range_fallback"
            }
            (_, DecisionReason::CapacityFallbackPowerOfTwo) => "capacity_fallback_power_of_two",
            (_, DecisionReason::RangeFallback) => "range_fallback",
            (_, _) if !affinity_lookup_enabled => "range_fallback",
            (_, _) if !has_session_id => "no_session",
            (ProposalKind::PowerOfTwo, _) => "assigned",
            _ => "primary",
        },
        PolicyKind::CacheAware => match (proposal, decision) {
            (_, DecisionReason::CacheCandidate)
            | (ProposalKind::CacheAffinity, DecisionReason::Primary) => "cache_candidate",
            (_, DecisionReason::Primary) => "no_cache_candidate",
            (_, DecisionReason::BackupPrimaryAdmission) => "no_cache_candidate_admission_backup",
            (_, DecisionReason::BackupPressureGuard) => "no_cache_candidate_pressure_backup",
            (_, DecisionReason::RangeFallback) => "no_cache_candidate_range_fallback",
            (_, DecisionReason::CapacityFallbackPowerOfTwo) => {
                "no_cache_candidate_capacity_fallback_power_of_two"
            }
        },
        _ => match decision {
            DecisionReason::Primary => "primary",
            DecisionReason::CacheCandidate => "cache_candidate",
            DecisionReason::BackupPrimaryAdmission => "admission_backup",
            DecisionReason::BackupPressureGuard => "pressure_backup",
            DecisionReason::RangeFallback => "range_fallback",
            DecisionReason::CapacityFallbackPowerOfTwo => "capacity_fallback_power_of_two",
        },
    }
}

/// Maximum buffered chat-completions body (32MiB). Sized for base64 multimodal inputs;
/// enforced by the `DefaultBodyLimit`, and returns 413 PAYLOAD_TOO_LARGE.
pub const MAX_CHAT_BODY_BYTES: usize = 32 << 20;

/// Minimal probe over the request body — the fields the ROUTER itself acts on,
/// plus the sampling parameters the fleet-wide contract governs.
/// Deserializing into this struct (vs `serde_json::Value`) does two things:
///
/// 1. Avoids building a `Value` tree over a multi-MiB body. NOTHING here
///    retains client-sized data: an unrecognized key is skipped through
///    `IgnoredAny`, and a sampling value that is not a number is drained the
///    same way (see [`ProbedValue`]).
/// 2. Pins the contract: the body MUST be a JSON object. Degenerate
///    shapes (`null`, `[]`, `"hi"`) fail at this step rather than being
///    silently forwarded with `stream=false`.
///
/// All other fields are ignored — the worker is authoritative for the
/// full request schema.
#[derive(Debug, Default)]
struct RequestProbe {
    stream: Option<bool>,
    model: Option<String>,
    /// Explicit output budget used by Decode Bucket routing.
    max_tokens: Option<u64>,
    max_completion_tokens: Option<u64>,
    /// What the request said about each governed sampling parameter, indexed
    /// by [`SamplingField::index`] so governing another needs no change here.
    /// Read by [`apply_sampling_overrides`]; see [`ProbedValue`].
    sampling: [ProbedValue; SamplingField::ALL.len()],
}

/// What the request said about one governed sampling parameter.
///
/// WHY not `Option<serde_json::Value>`: these keys carry client-controlled
/// JSON of arbitrary size and the probe runs on every request whether or not a
/// contract is configured, so holding a `Value` would let
/// `{"temperature": [1, 1, ...]}` allocate a tree proportional to a
/// [`MAX_CHAT_BODY_BYTES`] body for a field nothing then reads. Each variant is
/// resolved during deserialization; nothing client-sized is retained.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
enum ProbedValue {
    /// Omitted, or an explicit `null`. The OpenAI API types these parameters
    /// as nullable with a documented default, so `null` asks for the default —
    /// and on a governed fleet the configured value IS the default. Both mean
    /// the configured value is injected.
    #[default]
    Absent,
    /// A JSON number, or a string the engine's pydantic lax mode reads as one.
    Number(f64),
    /// Present but not numeric. Nobody's business but the engine's, which owns
    /// the request schema and gives the better message — so the contract
    /// neither compares it nor overwrites it.
    Unusable,
}

impl<'de> Deserialize<'de> for ProbedValue {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct ValueVisitor;
        impl<'de> serde::de::Visitor<'de> for ValueVisitor {
            type Value = ProbedValue;

            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a sampling parameter value")
            }

            fn visit_i64<E>(self, v: i64) -> Result<ProbedValue, E> {
                Ok(ProbedValue::Number(v as f64))
            }

            fn visit_u64<E>(self, v: u64) -> Result<ProbedValue, E> {
                Ok(ProbedValue::Number(v as f64))
            }

            fn visit_f64<E>(self, v: f64) -> Result<ProbedValue, E> {
                Ok(ProbedValue::Number(v))
            }

            /// Parsed, not retained: the engine reads a numeric string as a
            /// number, so a `reject` contract must too or `"1.5"` slips past
            /// a pin of 1.
            fn visit_str<E>(self, v: &str) -> Result<ProbedValue, E> {
                Ok(v.trim()
                    .parse::<f64>()
                    .map_or(ProbedValue::Unusable, ProbedValue::Number))
            }

            fn visit_unit<E>(self) -> Result<ProbedValue, E> {
                Ok(ProbedValue::Absent)
            }

            fn visit_bool<E>(self, _: bool) -> Result<ProbedValue, E> {
                Ok(ProbedValue::Unusable)
            }

            /// Drained, never collected — the allocation this type exists to
            /// avoid.
            fn visit_seq<A: serde::de::SeqAccess<'de>>(
                self,
                mut seq: A,
            ) -> Result<ProbedValue, A::Error> {
                while seq.next_element::<IgnoredAny>()?.is_some() {}
                Ok(ProbedValue::Unusable)
            }

            /// Drained, never collected — as [`Self::visit_seq`].
            fn visit_map<M: serde::de::MapAccess<'de>>(
                self,
                mut map: M,
            ) -> Result<ProbedValue, M::Error> {
                while map.next_entry::<IgnoredAny, IgnoredAny>()?.is_some() {}
                Ok(ProbedValue::Unusable)
            }
        }
        d.deserialize_any(ValueVisitor)
    }
}

/// A field the ROUTER acts on, as opposed to one it only forwards. A repeated
/// occurrence of one of these is a 400: a body that says two different things
/// about how to route itself is ambiguous at the edge, and the router must not
/// decide on one copy while the engine serves the other.
#[derive(Debug, Clone, Copy)]
enum RoutingKey {
    Stream,
    Model,
    MaxTokens,
    MaxCompletionTokens,
}

impl RoutingKey {
    /// Bit position in the visitor's occurrence mask. Unlike
    /// [`SamplingField::index`] this indexes no array, so it needs no
    /// agreement with a separate length constant.
    const fn bit(self) -> u8 {
        match self {
            Self::Stream => 1 << 0,
            Self::Model => 1 << 1,
            Self::MaxTokens => 1 << 2,
            Self::MaxCompletionTokens => 1 << 3,
        }
    }

    const fn wire_name(self) -> &'static str {
        match self {
            Self::Stream => "stream",
            Self::Model => "model",
            Self::MaxTokens => "max_tokens",
            Self::MaxCompletionTokens => "max_completion_tokens",
        }
    }
}

/// One key of the request object, resolved WITHOUT allocating: the visitor
/// matches a borrowed `&str` and keeps only a discriminant, so a body's
/// unrecognized majority costs no `String` per key.
enum ProbeKey {
    Routing(RoutingKey),
    /// A governed sampling parameter. A repeat LAST-WINS, matching the
    /// engine's own `json.loads` — so the value the contract compares against
    /// and the value the engine actually samples with are the same one.
    Sampling(SamplingField),
    Other,
}

impl<'de> Deserialize<'de> for ProbeKey {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct KeyVisitor;
        impl serde::de::Visitor<'_> for KeyVisitor {
            type Value = ProbeKey;

            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a request field name")
            }

            fn visit_str<E>(self, v: &str) -> Result<ProbeKey, E> {
                Ok(match v {
                    "stream" => ProbeKey::Routing(RoutingKey::Stream),
                    "model" => ProbeKey::Routing(RoutingKey::Model),
                    "max_tokens" => ProbeKey::Routing(RoutingKey::MaxTokens),
                    "max_completion_tokens" => ProbeKey::Routing(RoutingKey::MaxCompletionTokens),
                    other => match SamplingField::from_wire_name(other) {
                        Some(field) => ProbeKey::Sampling(field),
                        None => ProbeKey::Other,
                    },
                })
            }
        }
        d.deserialize_str(KeyVisitor)
    }
}

impl<'de> Deserialize<'de> for RequestProbe {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct ProbeVisitor;
        impl<'de> serde::de::Visitor<'de> for ProbeVisitor {
            type Value = RequestProbe;

            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a JSON object")
            }

            fn visit_map<M: serde::de::MapAccess<'de>>(
                self,
                mut map: M,
            ) -> Result<RequestProbe, M::Error> {
                let mut probe = RequestProbe::default();
                let mut seen = 0u8;
                while let Some(key) = map.next_key::<ProbeKey>()? {
                    match key {
                        ProbeKey::Routing(r) => {
                            if seen & r.bit() != 0 {
                                return Err(serde::de::Error::custom(format_args!(
                                    "duplicate field `{}`",
                                    r.wire_name()
                                )));
                            }
                            seen |= r.bit();
                            match r {
                                RoutingKey::Stream => probe.stream = map.next_value()?,
                                RoutingKey::Model => probe.model = map.next_value()?,
                                RoutingKey::MaxTokens => probe.max_tokens = map.next_value()?,
                                RoutingKey::MaxCompletionTokens => {
                                    probe.max_completion_tokens = map.next_value()?
                                }
                            }
                        }
                        ProbeKey::Sampling(field) => {
                            probe.sampling[field.index()] = map.next_value()?;
                        }
                        ProbeKey::Other => {
                            map.next_value::<IgnoredAny>()?;
                        }
                    }
                }
                Ok(probe)
            }
        }
        // `deserialize_map` is what pins the body to a JSON object: `null`,
        // `[]` and `"hi"` are all rejected here, so no separate shape-anchoring
        // pass is needed.
        d.deserialize_map(ProbeVisitor)
    }
}

impl RequestProbe {
    fn requested_max_output_tokens(&self) -> Option<u64> {
        self.max_completion_tokens.or(self.max_tokens)
    }

    /// Lets [`apply_sampling_overrides`] loop over whatever the operator
    /// configured instead of repeating a per-field ladder.
    fn sampling_field(&self, field: SamplingField) -> ProbedValue {
        self.sampling[field.index()]
    }
}

/// Project the peak sequence length without integer wraparound.
fn projected_decode_kv_tokens(input_tokens: u64, max_output_tokens: Option<u64>) -> u64 {
    max_output_tokens.map_or(input_tokens, |output_tokens| {
        input_tokens.saturating_add(output_tokens)
    })
}

/// RAII guard that records `sgl_router_request_duration_seconds` when
/// dropped. For streaming requests the handler returns at response-headers
/// time, so recording end-to-end latency at the dispatch site would capture
/// only time-to-headers (≈ TTFT). Instead this guard is packed into the SSE
/// pump's `stream_guards`, so it drops — and records — when the stream
/// completes (or the client disconnects), yielding true end-to-end latency.
/// Non-streaming requests record at the dispatch site directly (the body is
/// already buffered there) and do not use this guard.
struct RecordDurationOnDrop {
    metrics: Arc<MetricsRegistry>,
    model: String,
    start: std::time::Instant,
}

impl Drop for RecordDurationOnDrop {
    fn drop(&mut self) {
        self.metrics
            .observe_request_duration(&self.model, self.start.elapsed().as_secs_f64());
    }
}

fn policy_selection_failed(
    ctx: &AppContext,
    model: &str,
    reason: PolicySelectionFailureReason,
) -> ApiError {
    ctx.metrics
        .record_policy_selection_failure(ctx.config.model.policy, reason);
    tracing::warn!(
        policy = %ctx.config.model.policy,
        reason = reason.as_str(),
        model,
        "prefill policy selection failed"
    );
    ApiError::PolicySelectionFailed {
        model: model.to_owned(),
    }
}

/// POST /v1/chat/completions — parse model from body, select a healthy
/// worker via the per-model policy, then proxy the request. If the
/// request opts into streaming (`stream: true`), we pipe SSE bytes back;
/// otherwise buffer.
pub async fn chat_completions(
    State(ctx): State<Arc<AppContext>>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response<Body>, ApiError> {
    let start = std::time::Instant::now();
    let mut probe = parse_probe(&body)?;
    let streaming = probe.stream.unwrap_or(false);
    let requested_max_output_tokens = probe.requested_max_output_tokens();
    // `take`n rather than borrowed: the sampling contract further down reads
    // the rest of `probe`, but nothing reads `model` again, so the `String`
    // moves out instead of being cloned on every request.
    let model_str = probe
        .model
        .take()
        .ok_or_else(|| ApiError::BadRequest("missing `model` field".into()))?;
    let model_id = ModelId(model_str.clone());

    // PD pool isolation: for PD-mode deployments, prefill traffic
    // selects from the prefill pool only. Plain-mode deployments fall
    // through to the full candidate set. Partial-failure errors
    // (`no_prefill_workers_available`) are surfaced as 503 with a
    // distinct error code so operators can alert independently.
    let resolver = PdPoolResolver::new(Arc::clone(&ctx.registry));
    let workers = resolver
        .prefill_candidates(&model_id)
        .map_err(|e| match e {
            PdResolveError::NoHealthyWorkers => ApiError::NoHealthyWorkers {
                model: model_str.clone(),
            },
            PdResolveError::NoPrefillWorkersAvailable => ApiError::NoPrefillWorkersAvailable {
                model: model_str.clone(),
            },
            PdResolveError::NoDecodeWorkersAvailable => ApiError::NoDecodeWorkersAvailable {
                model: model_str.clone(),
            },
        })?;

    let policy = ctx
        .policies
        .get(&model_id)
        .ok_or_else(|| ApiError::ModelNotFound(model_str.clone()))?;

    // Fleet-wide sampling contract (`--override-sampling-params`), applied
    // once the model is known to be served (so a request naming an unknown
    // model still gets that answer) and before anything is admitted: under
    // `reject` a numeric value differing from the configured one is a 400
    // here, costing no queue slot and no engine round-trip. Either way the
    // configured values for fields the request omitted come back as the
    // inject-set for the forwarded body.
    let inject_sampling =
        apply_sampling_overrides(&ctx.config.model.sampling_overrides, &probe, &ctx.metrics)?;

    // Tokenize once at ingress whenever it can pay off — decoupled from the
    // routing policy, because forwarding `input_ids` is a property of the
    // MODEL (does it have a chat encoder so the router can produce
    // engine-equivalent tokens?), not of how we pick the worker. Two gates:
    //
    //   * `has_chat_encoder` → a chat request on this model yields
    //     engine-equivalent ids we can forward as `input_ids` so the engine
    //     skips re-tokenizing. This enables the offload for EVERY policy —
    //     sticky and round-robin included — not just cache-aware.
    //   * `needs_request_tokens()` → the cache-aware policy ALSO wants the
    //     raw-prompt path tokenized for tree matching even on a model with no
    //     chat encoder (`/v1/completions` / `text`), which the first gate
    //     alone wouldn't trigger.
    //
    //   * Bucket routing also needs the prompt token count.
    //
    // When none holds, `parse_probe`'s minimal probe is enough, so we keep
    // avoiding the full `serde_json::Value` allocation over a (up to 1 MiB)
    // body. When parsed, this single value is reused for the routing
    // tokenization and the outgoing-body injection below (and PD bootstrap
    // injection). `parse_probe` already validated the object shape.
    let want_tokens = should_tokenize_request(
        ctx.tokenizers.has_chat_encoder(&model_str),
        policy.needs_request_tokens(),
        ctx.bucket_selector.is_enabled(),
    );
    let request_value: Option<serde_json::Value> = if want_tokens {
        Some(serde_json::from_slice(&body).map_err(|_| {
            ApiError::BadRequest("invalid request: body must be a JSON object".into())
        })?)
    } else {
        None
    };

    // The ids feed both the routing decision (cache-aware consumes them; other
    // policies ignore them) and — when engine-equivalent — the engine itself,
    // forwarded as `input_ids` so it skips re-tokenizing the same prompt. The
    // ingress owns the tokenize via the shared registry, so the choice of
    // policy never changes whether we tokenize.
    let request_tokens = request_value
        .as_ref()
        .and_then(|v| request_tokens_for(&ctx.tokenizers, &model_id, v));
    let external_prefix = match (
        ctx.prefix_index.as_ref(),
        request_tokens.as_ref(),
        ctx.block_size_oracle.get(),
    ) {
        (Some(index), Some(tokens), Some(block_size)) => {
            let hashes = if ctx.block_size_oracle.is_bigram() {
                compute_block_hashes_bigram(&tokens.ids, block_size as usize)
            } else {
                compute_block_hashes(&tokens.ids, block_size as usize)
            };
            let query_blocks = hashes.len();
            let outcome = if hashes.is_empty() {
                sgl_kv_indexer::PrefixOutcome::Empty
            } else {
                resolve_prefix_query(index.match_prefix(hashes).await, &model_str)?
            };
            Some(ExternalPrefixSignal {
                outcome,
                query_blocks,
            })
        }
        _ => ctx
            .radix_tree_prefix_provider
            .as_ref()
            .zip(request_tokens.as_ref())
            .and_then(|(provider, tokens)| provider.match_request_tokens(&tokens.ids)),
    };

    // Prefer exact ingress tokens; otherwise use the conservative estimate.
    let prefill_load = request_tokens
        .as_ref()
        .map(|tokens| tokens.ids.len().max(1))
        .unwrap_or_else(|| estimate_prefill_tokens(&body));
    let request_input_tokens = prefill_load as u64;
    let needs_load_snapshot = policy.needs_load_snapshot()
        || workers
            .iter()
            .any(|worker| worker.mode() == WorkerMode::Prefill);
    let load_snapshot =
        needs_load_snapshot.then(|| ctx.engine_load.capture_snapshot(std::time::Instant::now()));
    let needs_dispatch_timestamps = policy.needs_dispatch_timestamps();
    let (ttft_slo_ms, tps_slo) = if ctx.bucket_selector.is_enabled() {
        (
            parse_optional_positive_u64_header(&headers, &X_SGL_TTFT_SLO_MS, "TTFT SLO")?,
            parse_optional_positive_f64_header(&headers, &X_SGL_TPS_SLO, "TPS SLO")?,
        )
    } else {
        (None, None)
    };

    // Sticky-session routing key. When the sticky policy is configured,
    // read the routing key from the operator-chosen header into the
    // selection context; the policy pins it to a worker. Other policies
    // leave `routing_key` `None` and ignore it.
    let routing_key = ctx
        .config
        .model
        .sticky
        .as_ref()
        .and_then(|s| headers.get(s.header_name.as_str()))
        .and_then(|v| v.to_str().ok())
        .filter(|s| !s.is_empty());
    let session_id = ctx
        .config
        .model
        .affinity
        .as_ref()
        .and_then(|config| headers.get(config.session_id_header.as_str()))
        .and_then(|value| value.to_str().ok())
        .filter(|value| !value.is_empty());
    // Each Bucket retry rebuilds the proposal and reruns Admission/Guard.
    let prefill_bucket_request = BucketRequest {
        input_tokens: request_input_tokens,
        expected_peak_sequence_tokens: None,
        ttft_slo_ms,
        tps_slo,
    };
    let configured_session_affinity_mode = ctx
        .config
        .model
        .affinity
        .as_ref()
        .map(|config| config.session_affinity_mode)
        .unwrap_or(SessionAffinityMode::Bucket);
    // Without Bucket partitioning all modes reduce to the single global domain.
    let session_affinity_mode = if ctx.bucket_selector.is_enabled() {
        configured_session_affinity_mode
    } else {
        SessionAffinityMode::Bucket
    };
    let use_global_affinity_probe = ctx.bucket_selector.is_enabled()
        && policy.is_bucket_affinity_policy()
        && session_affinity_mode != SessionAffinityMode::Bucket;
    let worker = {
        let selection_failure_reason = Cell::new(PolicySelectionFailureReason::ProposalEmpty);
        let select_prefill_in_domain = |domain: &CandidateDomain,
                                        affinity_lookup_enabled: bool,
                                        affinity_assignment_enabled: bool,
                                        allow_capacity_fallback: bool|
         -> Option<Arc<Worker>> {
            let candidate_range = domain.prefill_range()?;
            let mut selection_ctx =
                SelectionContext::with_routing_key(&model_id, Some(&body), routing_key)
                    .with_session_id(session_id)
                    .with_candidate_range_id(candidate_range.id)
                    .with_input_tokens(request_input_tokens)
                    .with_request_tokens(
                        request_tokens.as_ref().map(|tokens| tokens.ids.as_slice()),
                    )
                    .with_external_prefix(external_prefix.as_ref());
            if let Some(snapshot) = load_snapshot.as_ref() {
                selection_ctx = selection_ctx.with_load_snapshot(snapshot);
            }
            let selection_ctx = if !affinity_lookup_enabled {
                selection_ctx.without_affinity_lookup()
            } else if !affinity_assignment_enabled {
                selection_ctx.without_affinity_assignment()
            } else {
                selection_ctx
            };
            let Some(PrefillProposal::Pair(proposal)) =
                policy.propose_prefill(candidate_range.workers, &selection_ctx)
            else {
                // Domain retries are ordinary pair proposals.
                return None;
            };
            if policy.uses_shared_prefill_admission() {
                let snapshot = load_snapshot
                    .as_ref()
                    .expect("shared prefill admission requires a load snapshot");
                let decision = if allow_capacity_fallback {
                    resolve_prefill(&candidate_range, &proposal, request_input_tokens, snapshot)
                } else {
                    resolve_prefill_admitted(
                        &candidate_range,
                        &proposal,
                        request_input_tokens,
                        snapshot,
                    )
                };
                let Some(decision) = decision else {
                    selection_failure_reason
                        .set(PolicySelectionFailureReason::PrefillAdmissionExhausted);
                    return None;
                };
                let reason = prefill_policy_reason(
                    ctx.config.model.policy,
                    proposal.kind,
                    decision.reason,
                    session_id.is_some_and(|value| !value.is_empty()),
                    affinity_lookup_enabled,
                );
                policy.commit_prefill_selection(&selection_ctx, proposal.kind, &decision.selected);
                ctx.metrics
                    .record_policy_decision(&ctx.config.model.policy.to_string(), reason);
                tracing::debug!(
                    model = %model_str,
                    policy = ?proposal.kind,
                    range = %decision.candidate_range_id,
                    primary = %decision.primary.url,
                    backup = ?decision.backup.as_ref().map(|worker| worker.url.as_str()),
                    selected = %decision.selected.url,
                    reason = ?decision.reason,
                    load_snapshot_version = decision.load_snapshot_version,
                    "prefill policy decision",
                );
                Some(decision.selected)
            } else {
                tracing::debug!(
                    model = %model_str,
                    policy = ?proposal.kind,
                    range = %candidate_range.id,
                    selected = %proposal.primary.url,
                    "prefill policy decision without shared admission",
                );
                Some(proposal.primary)
            }
        };
        let select_prefill_domains =
            |domains: &[CandidateDomain],
             affinity_lookup_enabled: bool,
             affinity_assignment_enabled: bool| {
                domains
                    .iter()
                    .find_map(|domain| {
                        select_prefill_in_domain(
                            domain,
                            affinity_lookup_enabled,
                            affinity_assignment_enabled,
                            false,
                        )
                    })
                    .or_else(|| {
                        domains.iter().find_map(|domain| {
                            select_prefill_in_domain(
                                domain,
                                affinity_lookup_enabled,
                                affinity_assignment_enabled,
                                true,
                            )
                        })
                    })
            };

        // Cache-Aware resolves one bounded global candidate set and returns a final winner.
        let cache_winner = (ctx.config.model.policy == PolicyKind::CacheAware)
            .then(|| {
                let snapshot = load_snapshot.as_ref()?;
                let global_range = CandidateRange::global(&workers);
                let cache_ctx =
                    SelectionContext::with_routing_key(&model_id, Some(&body), routing_key)
                        .with_session_id(session_id)
                        .with_candidate_range_id(global_range.id)
                        .with_input_tokens(request_input_tokens)
                        .with_request_tokens(
                            request_tokens.as_ref().map(|tokens| tokens.ids.as_slice()),
                        )
                        .with_external_prefix(external_prefix.as_ref())
                        .with_load_snapshot(snapshot)
                        .with_prefill_cache_bucket(&ctx.bucket_selector, prefill_bucket_request);
                let PrefillProposal::CacheCandidates(proposal) =
                    policy.propose_prefill(global_range.workers, &cache_ctx)?
                else {
                    return None;
                };
                let bounded_candidate_count = proposal.candidates.len();
                let cache_decision =
                    resolve_cache_candidates(&proposal, request_input_tokens, snapshot);
                ctx.metrics.record_cache_admission_evaluations(
                    cache_decision.admission_evaluated_candidates,
                );
                ctx.metrics.record_cache_admission_rejections(
                    cache_decision.admission_rejected_candidates,
                );
                ctx.metrics.record_cache_pressure_guard(
                    cache_decision.pressure_guard_compared_pairs,
                    cache_decision.pressure_guard_overrides,
                );
                ctx.metrics
                    .record_cache_monitor_decision(cache_decision.prefill_pressure_source);
                let Some(decision) = cache_decision.decision else {
                    selection_failure_reason
                        .set(PolicySelectionFailureReason::CacheCandidatesExhausted);
                    return None;
                };
                let selected_candidate = proposal
                    .candidates
                    .iter()
                    .find(|candidate| candidate.worker.id == decision.selected.id)?;
                tracing::debug!(
                    model = %model_str,
                    policy = ?ProposalKind::CacheAffinity,
                    range = %decision.candidate_range_id,
                    selected = %decision.selected.url,
                    cache_candidates = bounded_candidate_count,
                    input_tokens = request_input_tokens,
                    matched_prefix_tokens = selected_candidate.matched_prefix_tokens,
                    uncached_tokens = selected_candidate.uncached_tokens,
                    reason = ?decision.reason,
                    load_snapshot_version = decision.load_snapshot_version,
                    prefill_pressure_source = cache_decision.prefill_pressure_source,
                    "cache candidate winner",
                );
                ctx.metrics
                    .record_policy_decision("cache_aware", "cache_candidate");
                Some(decision.selected)
            })
            .flatten();

        let global_affinity_probe = use_global_affinity_probe
            .then(|| {
                let snapshot = load_snapshot.as_ref()?;
                let global_range = CandidateRange::global(&workers);
                let probe_ctx =
                    SelectionContext::with_routing_key(&model_id, Some(&body), routing_key)
                        .with_session_id(session_id)
                        .with_candidate_range_id(global_range.id)
                        .with_input_tokens(request_input_tokens)
                        .with_request_tokens(
                            request_tokens.as_ref().map(|tokens| tokens.ids.as_slice()),
                        )
                        .with_external_prefix(external_prefix.as_ref())
                        .with_load_snapshot(snapshot)
                        .without_affinity_assignment();
                policy.propose(global_range.workers, &probe_ctx)
            })
            .flatten();
        // A new or stale session may create its first assignment in the target Bucket.
        let global_affinity_missed = global_affinity_probe
            .as_ref()
            .is_some_and(|proposal| !matches!(proposal.kind, ProposalKind::SessionAffinity));
        let global_affinity_worker = global_affinity_probe
            .and_then(|proposal| {
                matches!(proposal.kind, ProposalKind::SessionAffinity).then_some(proposal.primary)
            })
            .and_then(|primary| {
                ctx.bucket_selector.prefill_affinity_domain(
                    &workers,
                    &primary,
                    prefill_bucket_request,
                )
            })
            // Rebuild the backup inside the primary's own Bucket.
            .and_then(|domain| select_prefill_in_domain(&domain, true, false, false));
        cache_winner
            .or_else(|| {
                // Materialize normal domains only when Cache-Aware has no winner.
                let prefill_domains = ctx
                    .bucket_selector
                    .prefill_domains(&workers, prefill_bucket_request);
                if ctx.config.model.policy == PolicyKind::CacheAware {
                    // Cache miss or failure retries ordered domains with ordinary P2.
                    return select_prefill_domains(&prefill_domains, false, false);
                }
                global_affinity_worker.or_else(|| match session_affinity_mode {
                    SessionAffinityMode::GlobalPreserve if global_affinity_missed => {
                        select_prefill_domains(&prefill_domains, true, true)
                    }
                    SessionAffinityMode::GlobalPreserve => {
                        select_prefill_domains(&prefill_domains, false, false)
                    }
                    SessionAffinityMode::Bucket | SessionAffinityMode::GlobalRebind => {
                        select_prefill_domains(&prefill_domains, true, true)
                    }
                })
            })
            .ok_or_else(|| {
                policy_selection_failed(&ctx, &model_str, selection_failure_reason.get())
            })?
    };

    // Decode selection starts after Final P.
    //
    // Plain-mode workers skip the decode resolution entirely (no
    // decode peer to find). PD-mode requests that fail to resolve a
    // decode peer (`NoDecodeWorkersAvailable`) bubble up as 503 so
    // operators can alert on prefill-vs-decode pool imbalance.
    let decode_peer: Option<Arc<Worker>> = if worker.mode() == WorkerMode::Prefill {
        let decode_workers = resolver.decode_candidates(&model_id).map_err(|e| match e {
            PdResolveError::NoHealthyWorkers => ApiError::NoHealthyWorkers {
                model: model_str.clone(),
            },
            PdResolveError::NoDecodeWorkersAvailable => ApiError::NoDecodeWorkersAvailable {
                model: model_str.clone(),
            },
            PdResolveError::NoPrefillWorkersAvailable => ApiError::NoPrefillWorkersAvailable {
                model: model_str.clone(),
            },
        })?;
        let request_kv_tokens =
            projected_decode_kv_tokens(request_input_tokens, requested_max_output_tokens);
        let expected_peak_sequence_tokens = requested_max_output_tokens.map(|_| request_kv_tokens);
        let decode_domains = ctx.bucket_selector.decode_domains(
            &decode_workers,
            BucketRequest {
                input_tokens: request_input_tokens,
                expected_peak_sequence_tokens,
                ttft_slo_ms,
                tps_slo,
            },
        );
        let decode_policy = build_decode_policy(ctx.config.model.decode_policy);
        let select_decode_in_domain =
            |decode_domain: &CandidateDomain, allow_capacity_fallback: bool| {
                let snapshot = load_snapshot.as_ref()?;
                let decode_ctx = DecodeSelectionContext::new()
                    .with_load_snapshot(snapshot)
                    .with_prefill_url(&worker.url);
                let decode_proposal = decode_policy.propose(decode_domain, &decode_ctx)?;
                let decode_decision = if allow_capacity_fallback {
                    resolve_decode_with_capacity_fallback(
                        decode_domain,
                        &decode_proposal,
                        request_kv_tokens,
                        snapshot,
                    )
                } else {
                    resolve_decode(decode_domain, &decode_proposal, request_kv_tokens, snapshot)
                }?;
                tracing::debug!(
                    model = %model_str,
                    policy = ?ctx.config.model.decode_policy,
                    range = %decode_decision.candidate_range_id,
                    primary = %decode_decision.primary.url,
                    backup = ?decode_decision.backup.as_ref().map(|worker| worker.url.as_str()),
                    selected = %decode_decision.selected.url,
                    reason = ?decode_decision.reason,
                    load_snapshot_version = decode_decision.load_snapshot_version,
                    "decode policy decision",
                );
                Some(decode_decision.selected)
            };
        decode_domains
            .iter()
            .find_map(|domain| select_decode_in_domain(domain, false))
            .or_else(|| {
                decode_domains
                    .iter()
                    .find_map(|domain| select_decode_in_domain(domain, true))
            })
            .ok_or_else(|| ApiError::NoDecodeWorkersAvailable {
                model: model_str.clone(),
            })
            .map(Some)?
    } else {
        None
    };
    let decode_hint_url: Option<String> = decode_peer.as_ref().map(|d| d.url.clone());
    let mut request_headers = headers;
    if let Some(url) = &decode_hint_url {
        match HeaderValue::from_str(url) {
            Ok(v) => {
                request_headers.insert(X_SGL_DECODE_URL, v);
            }
            Err(e) => {
                // Discovery emits URLs the proxy has already used; a
                // header-value parse failure here means the URL
                // contains a control character (e.g. CR / LF) — drop
                // the header but keep the request: bootstrap injection
                // below carries the host/port the engine actually
                // needs; the header is purely observability.
                tracing::warn!(
                    decode_url = %url,
                    error = %e,
                    "decode worker URL rejected by header parser; sending request without decode hint",
                );
            }
        }
    }
    let headers = request_headers;

    // Per-worker `active_requests` guard. The `ActiveLoadGuard` below
    // sits beside this one: both track in-flight load, but the
    // ActiveLoadGuard entry is per-request (with timeout-based janitor)
    // while the worker-scoped counter is what the cache-aware policy
    // reads. Both must drop at the same time — when the response stream
    // ends, the client disconnects, or the handler returns an error. In
    // PD mode the pair moves into the spawned prefill task so prefill
    // load is tracked for the full duration of the KV transfer; in plain
    // mode the pair stays in this handler. Decode load is tracked on Final D.
    let guard = if needs_dispatch_timestamps {
        worker.timestamped_load_guard()
    } else {
        worker.load_guard()
    };
    let active_guard =
        ctx.active_load
            .register(worker.id.clone(), worker.url.clone(), prefill_load, 0);
    // Snapshot the stale-request cancel token BEFORE moving the guard
    // into the spawned prefill task / streaming pump / response future.
    // The token is cheap to clone (it's an `Arc<...>` internally) and
    // the chat handler races the client-facing fetch against
    // `token.cancelled()` to surface a 504 `stale_request_expired` if
    // the janitor expires the request mid-flight.
    let stale_token = active_guard.cancel_token().clone();

    // Snapshot the labels we need for metrics BEFORE moving the worker
    // / model_str values into the per-branch fetch futures.
    let metrics_worker_url = worker.url.clone();
    let metrics_mode = match worker.mode() {
        WorkerMode::Prefill => WorkerModeLabel::Prefill,
        WorkerMode::Decode => WorkerModeLabel::Decode,
        WorkerMode::Plain => WorkerModeLabel::Plain,
    };
    let metrics_model = model_str.clone();

    // Builds the time-to-first-token hook the SSE pump fires when the first
    // upstream chunk lands. Installed only on the streaming arms below —
    // non-streaming "first token" equals total latency, already captured by
    // `sgl_router_request_duration_seconds`. The proxy drops the hook for
    // non-2xx responses so error bodies don't pollute TTFT.
    let make_ttft_hook = || -> Box<dyn FnOnce() + Send + 'static> {
        let metrics = Arc::clone(&ctx.metrics);
        let model = metrics_model.clone();
        let started = start;
        Box::new(move || {
            metrics.observe_ttft(&model, started.elapsed().as_secs_f64());
        })
    };

    // Builds the end-to-end-latency guard for streaming requests. Packed into
    // `stream_guards` so it records when the SSE pump finishes (stream end or
    // client disconnect), not at response-headers time. Non-streaming records
    // at the dispatch site instead (see below).
    let make_duration_guard = || RecordDurationOnDrop {
        metrics: Arc::clone(&ctx.metrics),
        model: metrics_model.clone(),
        start,
    };

    // Forward the router-computed tokens to the engine as `input_ids` so it
    // skips re-tokenizing the same prompt — but only when they are
    // engine-equivalent (chat-encoder path) AND the request contains nothing
    // the router's encoder didn't replicate (see `input_ids_safe_to_forward`).
    // Otherwise omit them and the engine tokenizes from `messages` as usual —
    // a transparent, always-correct fallback (`messages` are always retained
    // in the forwarded body). `forward_input_ids` is `Some` only when
    // `request_value` is `Some` (a model the ingress tokenized for), so the
    // predicate always has a parsed body to inspect.
    let forward_input_ids: Option<&[u32]> = match (request_tokens.as_ref(), request_value.as_ref())
    {
        (Some(t), Some(v)) if t.engine_equivalent && input_ids_safe_to_forward(v) => {
            Some(t.ids.as_slice())
        }
        _ => None,
    };

    // Surface a broken offload: when the encoder SHOULD have produced
    // engine-equivalent ids but didn't, the chat request silently fell back to
    // engine-side tokenization. Count only that case (see
    // `ingress_tokenize_offload_failed`); successful forwards and expected
    // omissions are not problems.
    if ingress_tokenize_offload_failed(
        ctx.tokenizers.has_chat_encoder(&model_str),
        request_value.as_ref(),
        request_tokens.as_ref(),
    ) {
        ctx.metrics.record_ingress_tokenize_error(&metrics_model);
    }

    // PD-disagg bootstrap fields (prefill worker address + a per-request
    // room). Present only when a decode peer was resolved.
    let bootstrap = decode_peer.as_ref().map(|_| BootstrapFields {
        host: worker.bootstrap_host().to_string(),
        port: worker.bootstrap_port(),
        room: generate_room_id(),
    });
    let bootstrap_room = bootstrap.as_ref().map(|b| b.room);

    // Build the body forwarded to the engine(s) exactly once — injecting the
    // `input_ids`, bootstrap fields and sampling values, or forwarding the
    // original bytes untouched when none applies.
    let outgoing_body = build_outgoing_body(
        &body,
        request_value,
        forward_input_ids,
        bootstrap.as_ref(),
        &inject_sampling,
    )?;

    let result = if let Some(decode_worker) = decode_peer {
        // PD-disagg dispatch (Pattern B — spawn prefill, await decode).
        //
        // SGLang's HTTP-mode disagg-prefill requires three flat
        // top-level fields on the request body: `bootstrap_host`,
        // `bootstrap_port` (the prefill worker's bootstrap-server
        // address) and `bootstrap_room` (a per-request 63-bit u64 ID
        // used by both sides to pair up the KV transfer). We inject
        // these here and fan the same modified body to both the
        // prefill and decode workers concurrently.
        //
        // **Why spawn-and-forget for prefill instead of
        // `tokio::join!`?** All three peer SGLang-HTTP-PD routers
        // (Dynamo / llm-d / aibrix) converged on this shape: the
        // prefill request must outlive the client connection because
        // tying prefill to the client future opens a cancel-race
        // window where the engine's NIXL RPC teardown can leak KV
        // block refs (NVBugs 5969206 in Dynamo). The detached task
        // also keeps the LoadGuard + ActiveLoadGuard alive for the full
        // prefill duration — KV transfer can run for tens of seconds
        // even when the client gave up.
        //
        // No watchdog for fail-fast on prefill 5xx: llm-d / aibrix both
        // ship without one. On prefill failure the client experiences
        // the SGLang decode-side bootstrap_room timeout (~30–60 s by
        // default) instead of an immediate 502. A follow-up can wire a
        // `tokio::sync::watch` channel if telemetry shows it matters.
        //
        // **Scope of the "detached" guarantee.** The spawn protects
        // against client disconnect — the handler future being dropped
        // does NOT cancel the prefill HTTP request. It does NOT protect
        // against router shutdown: when `AppContext` tears down, the
        // tokio runtime cancels all unfinished tasks including this
        // one. A future follow-up could thread a `TaskTracker` /
        // `JoinSet` through `AppContext` for graceful shutdown drain;
        // the current implementation ships without one (matching SMG's
        // shutdown behaviour).
        let bootstrap_room = bootstrap_room.expect("PD dispatch implies a resolved bootstrap room");

        let prefill_url = worker.url.clone();
        let prefill_breaker = Arc::clone(&worker.breaker);
        let prefill_headers = headers.clone();
        let prefill_body = outgoing_body.clone();
        let prefill_proxy = Arc::clone(&ctx.proxy);
        let prefill_holds: (LoadGuard, _) = (guard, active_guard);
        tokio::spawn(async move {
            // The tuple binding extends both guards' lifetime to the
            // end of this async block, which lasts until the prefill
            // HTTP request returns (success / error / engine-side
            // bootstrap_room timeout). The result is logged and
            // swallowed — no channel back to the client. See the big
            // comment above for the rationale.
            let _hold = prefill_holds;
            match prefill_proxy
                .forward_json_to(
                    &prefill_url,
                    &prefill_breaker,
                    "/v1/chat/completions",
                    &prefill_headers,
                    prefill_body,
                )
                .await
            {
                Ok(_) => tracing::debug!(
                    prefill_url = %prefill_url,
                    bootstrap_room,
                    "prefill side completed",
                ),
                Err(e) => tracing::warn!(
                    prefill_url = %prefill_url,
                    bootstrap_room,
                    error = %e,
                    "prefill request failed; decode will time out on bootstrap_room",
                ),
            }
        });

        // Synchronously await the decode worker. Its response is what
        // the client sees. The decode side gets its own LoadGuard so
        // per-worker `active_requests` reflects load on Final D.
        let decode_guard = decode_worker.load_guard();
        let decode_active_guard =
            ctx.active_load
                .register(decode_worker.id.clone(), decode_worker.url.clone(), 0, 1);
        if streaming {
            let stream_guards: Box<dyn Send + 'static> =
                Box::new((decode_guard, decode_active_guard, make_duration_guard()));
            let fetch = ctx.proxy.forward_streaming_to(
                &decode_worker.url,
                &decode_worker.breaker,
                "/v1/chat/completions",
                &headers,
                outgoing_body,
                Some(stream_guards),
                Some(make_ttft_hook()),
            );
            tokio::select! {
                biased;
                r = fetch => r,
                _ = stale_token.cancelled() => Err(ApiError::StaleRequestExpired { model: model_str }),
            }
        } else {
            let _decode_hold = (decode_guard, decode_active_guard);
            let fetch = ctx.proxy.forward_json_to(
                &decode_worker.url,
                &decode_worker.breaker,
                "/v1/chat/completions",
                &headers,
                outgoing_body,
            );
            tokio::select! {
                biased;
                r = fetch => r,
                _ = stale_token.cancelled() => Err(ApiError::StaleRequestExpired { model: model_str }),
            }
        }
    } else if streaming {
        // Plain mode, streaming. Both guards ride the SSE pump until
        // the body completes — see the matching comment in the
        // non-streaming arm.
        let stream_guards: Box<dyn Send + 'static> =
            Box::new((guard, active_guard, make_duration_guard()));
        let fetch = ctx.proxy.forward_streaming_to(
            &worker.url,
            &worker.breaker,
            "/v1/chat/completions",
            &headers,
            outgoing_body,
            Some(stream_guards),
            Some(make_ttft_hook()),
        );
        // Bias `fetch` over the cancellation branch: a successful
        // response that completes in the same poll as the token firing
        // MUST win (returning 504 for a request that already has
        // headers is a correctness regression). The cancellation
        // branch only matters when fetch is still pending — at that
        // point biasing the order is a wash.
        tokio::select! {
            biased;
            r = fetch => r,
            _ = stale_token.cancelled() => Err(ApiError::StaleRequestExpired { model: model_str }),
        }
    } else {
        // Plain mode, non-streaming. The handler awaits the full
        // buffered response, so both guards live correctly in this
        // scope. The tuple binding exists only to extend the guards'
        // lifetime to the end of the function — the `forward_json_to`
        // future does not need them (it does not return until the
        // body is buffered).
        let _holds: (LoadGuard, _) = (guard, active_guard);
        let fetch = ctx.proxy.forward_json_to(
            &worker.url,
            &worker.breaker,
            "/v1/chat/completions",
            &headers,
            outgoing_body,
        );
        // Same `biased` order as the streaming arm.
        tokio::select! {
            biased;
            r = fetch => r,
            _ = stale_token.cancelled() => Err(ApiError::StaleRequestExpired { model: model_str }),
        }
    };

    // Record the dispatch outcome AFTER we know whether the upstream
    // accepted the request. A 504 from the stale-request branch counts as
    // `cancelled` — semantically distinct from upstream errors that bubble
    // through as `error`. The metric is per-worker so convergence tests
    // can scrape `/metrics` and assert that ≥N requests landed on a
    // single prefill worker.
    let outcome = match &result {
        Ok(_) => RequestOutcome::Success,
        Err(ApiError::StaleRequestExpired { .. }) => {
            // The janitor fired the stale-cancel and we observed it
            // user-side; record both the per-request `cancelled` outcome
            // AND the global `expired` count. The two views are useful for
            // different alerts: per-worker request_total{cancelled} flags a
            // worker that's hanging, while stale_requests_total{expired}
            // tracks the global health of the janitor.
            ctx.metrics
                .record_stale_request(StaleRequestOutcome::Expired);
            RequestOutcome::Cancelled
        }
        Err(_) => RequestOutcome::Error,
    };
    ctx.metrics
        .record_worker_request(&metrics_worker_url, &metrics_model, metrics_mode, outcome);

    // Per-request access log — always on at INFO so incoming traffic and its
    // status are visible without DEBUG. `request_id` is the client/gateway
    // X-Request-Id (echoed end-to-end); `worker` is the engine the policy
    // selected. The cache-aware routing rationale is logged separately at
    // DEBUG by the policy.
    let request_id = headers
        .get("x-request-id")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("-");
    let http_status = match &result {
        Ok(resp) => resp.status().as_u16(),
        Err(e) => e.status_code().as_u16(),
    };

    // Record end-to-end latency now that the outcome is known. Non-streaming:
    // body is buffered here, so `start.elapsed()` is true e2e — record directly.
    // Streaming: body still pending, so the `RecordDurationOnDrop` guard in
    // `stream_guards` records it at stream completion (not just time-to-headers).
    // `elapsed` still feeds the access-log `latency_ms` for both.
    //
    // HTTP status is counted into `responses_total` by the edge middleware
    // (app.rs), not here — the old per-handler site skipped early exits.
    let elapsed = start.elapsed();
    if !streaming {
        ctx.metrics
            .observe_request_duration(&metrics_model, elapsed.as_secs_f64());
    }
    let outcome_str = match outcome {
        RequestOutcome::Success => "success",
        RequestOutcome::Error => "error",
        RequestOutcome::Cancelled => "cancelled",
    };
    tracing::info!(
        request_id = %request_id,
        method = "POST",
        path = "/v1/chat/completions",
        model = %metrics_model,
        worker = %metrics_worker_url,
        outcome = outcome_str,
        http_status,
        stream = streaming,
        latency_ms = elapsed.as_millis() as u64,
        "chat_completions",
    );

    // Mirror the upstream `x-sgl-decode-url` hint onto the response so
    // external tests / sidecars can observe the final PD Decode selection without
    // sniffing the proxy hop. The request-side header was set above for
    // the prefill worker; copying it here makes the affinity observable
    // end-to-end. Plain-mode requests skip this (no decode peer was
    // resolved). A malformed URL was already rejected at the
    // request-side parse — we only reach this branch when the URL was
    // header-valid, so the second parse is safe.
    match (result, decode_hint_url) {
        (Ok(mut response), Some(url)) => {
            match HeaderValue::from_str(&url) {
                Ok(v) => {
                    response.headers_mut().insert(X_SGL_DECODE_URL, v);
                }
                Err(e) => {
                    // Already-validated upstream; defensive log only.
                    tracing::warn!(
                        decode_url = %url,
                        error = %e,
                        "decode worker URL rejected by header parser on response; omitting response-side hint",
                    );
                }
            }
            Ok(response)
        }
        (other, _) => other,
    }
}

fn resolve_prefix_query(
    result: Result<sgl_kv_indexer::PrefixOutcome, sgl_kv_indexer::PrefixIndexError>,
    model: &str,
) -> Result<sgl_kv_indexer::PrefixOutcome, ApiError> {
    use sgl_kv_indexer::PrefixIndexError;
    match result {
        Ok(outcome) => Ok(outcome),
        // The prefix hit only improves worker choice, so an indexer that is
        // shedding, slow, or down costs cache affinity — not availability.
        Err(
            error @ (PrefixIndexError::Overloaded
            | PrefixIndexError::Timeout
            | PrefixIndexError::Unreachable),
        ) => {
            tracing::warn!(%model, error = %error, "KV Indexer unavailable; falling back to min-load routing");
            Ok(sgl_kv_indexer::PrefixOutcome::Empty)
        }
        // A prompt too long to fit one gRPC message is still a prompt a worker
        // can serve, so it costs cache affinity like the cases above. Logged
        // separately because the remedy is operational — raise the indexer's
        // message limit — rather than waiting for the indexer to recover.
        Err(error @ PrefixIndexError::QueryTooLarge) => {
            tracing::warn!(%model, error = %error, "prompt exceeds the KV Indexer query size limit; falling back to min-load routing");
            Ok(sgl_kv_indexer::PrefixOutcome::Empty)
        }
        // A rejection means the router and the indexer disagree on the request
        // contract; degrading would hide that from every request.
        Err(error) => {
            tracing::warn!(%model, error = %error, "KV Indexer rejected the query");
            Err(ApiError::PolicySelectionFailed {
                model: model.to_string(),
            })
        }
    }
}

fn parse_optional_positive_u64_header(
    headers: &HeaderMap,
    name: &HeaderName,
    label: &str,
) -> Result<Option<u64>, ApiError> {
    let Some(value) = headers.get(name) else {
        return Ok(None);
    };
    let raw = value
        .to_str()
        .map_err(|_| ApiError::BadRequest(format!("{label} header must be ASCII")))?;
    let parsed = raw
        .parse::<u64>()
        .map_err(|_| ApiError::BadRequest(format!("{label} header must be a positive integer")))?;
    if parsed == 0 {
        return Err(ApiError::BadRequest(format!(
            "{label} header must be a positive integer"
        )));
    }
    Ok(Some(parsed))
}

fn parse_optional_positive_f64_header(
    headers: &HeaderMap,
    name: &HeaderName,
    label: &str,
) -> Result<Option<f64>, ApiError> {
    let Some(value) = headers.get(name) else {
        return Ok(None);
    };
    let raw = value
        .to_str()
        .map_err(|_| ApiError::BadRequest(format!("{label} header must be ASCII")))?;
    let parsed = raw
        .parse::<f64>()
        .map_err(|_| ApiError::BadRequest(format!("{label} header must be a positive number")))?;
    if !parsed.is_finite() || parsed <= 0.0 {
        return Err(ApiError::BadRequest(format!(
            "{label} header must be a finite positive number"
        )));
    }
    Ok(Some(parsed))
}

fn should_tokenize_request(
    has_chat_encoder: bool,
    policy_needs_request_tokens: bool,
    bucket_enabled: bool,
) -> bool {
    has_chat_encoder || policy_needs_request_tokens || bucket_enabled
}

/// Estimate prefill-token count from the raw request body for use as
/// the active-load `prefill_load` counter. Returns 1 at minimum so
/// a registered request always shows up as "load > 0" — under-counting
/// to zero would hide the request from the cache-aware policy's
/// load-imbalance fast-path.
///
/// Exact ingress tokens are preferred when available.
fn estimate_prefill_tokens(body: &Bytes) -> usize {
    (body.len() / CHARS_PER_TOKEN_ESTIMATE).max(1)
}

/// Mint a fresh `bootstrap_room` for a PD-disagg request.
///
/// SGLang's disagg-prefill stores the room as a signed `i64` internally
/// (see `python/sglang/srt/disaggregation/utils.py` — `bootstrap_room`
/// metadata buffer is allocated as `torch.int64`). Generating in
/// `[0, i64::MAX]` keeps the value safely positive when reinterpreted
/// signed. Mirrors SMG's `pd_types::generate_room_id`, Dynamo's
/// `rand::random_range(0..=i64::MAX.cast_unsigned())`, and SGLang's
/// own Python-side `random.randint(0, 2**63 - 1)`.
fn generate_room_id() -> u64 {
    rand::random::<u64>() & (i64::MAX as u64)
}

/// PD-disagg bootstrap fields injected into the body forwarded to both the
/// prefill and decode workers. SGLang's HTTP disagg-prefill validator
/// requires all three as flat top-level fields:
///
/// * `host` → `bootstrap_host` — the prefill worker's hostname; decode
///   connects here for the KV transfer.
/// * `port` → `bootstrap_port` — the prefill worker's bootstrap-server port
///   (`null` when the worker is misconfigured; the engine rejects with a
///   clear error). Emitted as JSON `null`, not omitted — SGLang's validator
///   distinguishes missing from null.
/// * `room` → `bootstrap_room` — a per-request 63-bit `u64` identifying this
///   request on both prefill and decode sides.
struct BootstrapFields {
    host: String,
    port: Option<u16>,
    room: u64,
}

/// Write `members` in as top-level keys of the JSON object in `body`, without
/// parsing it.
///
/// WHY: going through `serde_json` costs a full parse into a `Value` plus a
/// full re-serialize, over a body that runs to [`MAX_CHAT_BODY_BYTES`].
///
/// Members go in before the CLOSING brace so they win the last-wins reading
/// every JSON parser performs — the authority `obj.insert` has on the parse
/// path. Inserting after the opening brace would lose to a client's own later
/// copy of the key, which a request sending an explicit `null` for a governed
/// parameter has: the probe reads `null` as absent, so the value IS injected.
/// The last `}` is the object's closing brace, since `parse_probe` proved the
/// body is an object and only whitespace may follow it.
///
/// `None` (braces not located) falls back to the parse path rather than
/// panicking on a shape `parse_probe` should already have rejected.
fn splice_top_level(
    body: &Bytes,
    members: &[(SamplingField, serde_json::Number)],
) -> Option<Bytes> {
    use std::io::Write as _;

    let open = body.iter().position(|&b| b == b'{')?;
    let close = body.iter().rposition(|&b| b == b'}')?;
    if close <= open {
        return None;
    }
    // An empty object takes no separating comma: `{"temperature":1}`, not
    // `{,"temperature":1}`.
    let has_members = body[open + 1..close]
        .iter()
        .any(|b| !b.is_ascii_whitespace());
    // 24 bytes per member covers `"repetition_penalty":` plus a short number;
    // an over-run just costs one realloc, never correctness.
    let mut out = Vec::with_capacity(body.len() + 24 * members.len() + 1);
    out.extend_from_slice(&body[..close]);
    for (i, (field, value)) in members.iter().enumerate() {
        if has_members || i > 0 {
            out.push(b',');
        }
        // Wire names are a fixed set of JSON-safe identifiers and a
        // `serde_json::Number` renders as valid JSON, so neither needs
        // escaping. Written straight into `out` — no intermediate `String`.
        write!(out, "\"{}\":{}", field.wire_name(), value).ok()?;
    }
    out.extend_from_slice(&body[close..]);
    Some(Bytes::from(out))
}

/// Build the body forwarded to the engine, injecting (when present) the
/// precomputed `input_ids`, the PD `bootstrap_*` fields and the fleet-wide
/// sampling values into the already-parsed request object and serializing
/// once. When none is needed, returns the original bytes unchanged (no
/// re-serialize).
///
/// `input_ids`: the router-computed prompt tokens. When set the engine skips
/// its own chat-template tokenization, so `messages` are retained for the stop
/// tokens, tool-call constraint and response shape it still derives from them.
/// Set only when `input_ids_safe_to_forward` held.
///
/// `value` is the ingress parse when one is on hand, reused rather than
/// repeated — and dropped unused when splicing makes it unnecessary. Sampling
/// alone never reaches `serde_json`: only `input_ids` and bootstrap injection
/// do, because those may have to OVERWRITE a key the client sent, which
/// [`splice_top_level`] cannot. The non-object arm defends against a TOCTOU
/// regression rather than panicking.
fn build_outgoing_body(
    body: &Bytes,
    value: Option<serde_json::Value>,
    input_ids: Option<&[u32]>,
    bootstrap: Option<&BootstrapFields>,
    sampling: &[(SamplingField, serde_json::Number)],
) -> Result<Bytes, ApiError> {
    // `input_ids` and bootstrap injection may have to OVERWRITE a key the
    // client sent, which only the parse path can do; sampling injection never
    // does, because the inject-set holds only keys the request omitted.
    let only_sampling = input_ids.is_none() && bootstrap.is_none();
    if only_sampling && sampling.is_empty() {
        // Nothing to inject — forward the original bytes (cheap Arc clone).
        return Ok(body.clone());
    }
    // Splice regardless of whether a parse is already on hand: `value` is
    // read-only up to this point, so having one does not make splicing wrong
    // — it only means the parse was already paid for elsewhere. Gating on
    // `value.is_none()` would have skipped the splice on exactly the
    // configurations that parse at ingress (a chat encoder, the cache-aware
    // policy, bucket routing), i.e. most governed fleets.
    if only_sampling {
        if let Some(spliced) = splice_top_level(body, sampling) {
            return Ok(spliced);
        }
    }
    let parsed = match value {
        Some(v) => v,
        // The ingress skipped the parse, so re-parse. Reached for bootstrap
        // injection, and as the fallback if `splice_top_level` declined.
        None => serde_json::from_slice(body).map_err(|_| {
            ApiError::BadRequest("invalid request: body must be a JSON object".to_string())
        })?,
    };
    let mut obj = match parsed {
        serde_json::Value::Object(map) => map,
        _ => {
            return Err(ApiError::BadRequest(
                "invalid request: body must be a JSON object".to_string(),
            ));
        }
    };
    // The caller passes the inject-set from `apply_sampling_overrides` —
    // configured values for fields the request omitted — so writing them here
    // never masks a client value, in either conflict mode.
    for (field, value) in sampling {
        obj.insert(
            field.wire_name().to_string(),
            serde_json::Value::Number(value.clone()),
        );
    }
    if let Some(ids) = input_ids {
        obj.insert(
            "input_ids".to_string(),
            serde_json::Value::Array(
                ids.iter()
                    .map(|&i| serde_json::Value::Number(i.into()))
                    .collect(),
            ),
        );
    }
    if let Some(b) = bootstrap {
        obj.insert(
            "bootstrap_host".to_string(),
            serde_json::Value::String(b.host.clone()),
        );
        obj.insert(
            "bootstrap_port".to_string(),
            match b.port {
                Some(p) => serde_json::Value::Number(p.into()),
                None => serde_json::Value::Null,
            },
        );
        obj.insert(
            "bootstrap_room".to_string(),
            serde_json::Value::Number(b.room.into()),
        );
    }
    let bytes = serde_json::to_vec(&obj).map_err(|e| {
        ApiError::Internal(anyhow::Error::new(e).context("re-serialize injected request body"))
    })?;
    Ok(Bytes::from(bytes))
}

/// Whether the router's `input_ids` may be forwarded for this request.
///
/// We forward only when the engine, fed `input_ids`, would have produced the
/// SAME prompt the router tokenized. When `input_ids` is present the engine
/// uses it verbatim and ignores everything that would otherwise steer its
/// `messages`-side tokenization (only stop tokens / tool-call constraint are
/// still taken from `messages`). So any request field that changes that
/// tokenization but which the router's chat encoder does not replicate makes
/// the forwarded ids wrong. This predicate is conservative by construction —
/// any such signal returns `false` and the engine tokenizes from `messages`
/// (always correct).
///
/// Replicated-and-safe: plain text `messages` with a string `content`.
/// Not replicated → omit:
///   * `tools` / `functions` — the encoder doesn't render tool schemas.
///   * multimodal (array) `content` — a text tokenizer can't represent images.
///   * `chat_template` — an OpenAI-compatible per-request template override
///     (e.g. vLLM); the router renders with the model's default template, so a
///     custom one would diverge. (SGLang ignores it today, but block it so the
///     offload stays correct across engines / future versions.)
///   * `chat_template_kwargs` (carries `enable_thinking`/`thinking`),
///     `reasoning` / `reasoning_effort`, `task` — thinking/mode toggles the
///     encoder renders in the engine's default mode only.
///   * `continue_final_message: true`, or a trailing `assistant` message — the
///     engine rewrites/strips the final assistant turn; the encoder renders it
///     verbatim.
///
/// NOTE: the router's chat encoder renders in the engine's default
/// (non-thinking) mode. Current sglang derives thinking from the request
/// (`chat_template_kwargs`), which this guard already omits, so a plain request
/// the router rendered matches the engine. The only way to diverge is an engine
/// build that applies a non-default thinking mode the router can't observe from
/// the request — the same router↔engine tokenization-parity assumption that
/// cache-aware routing already depends on. The same assumption covers
/// `add_special_tokens`: the router renders specials via the chat template, which
/// matches the engine on tokenizers that auto-add them (the common case); a
/// tokenizer that does not would diverge by a leading special, again undetectable
/// from the request.
fn input_ids_safe_to_forward(value: &serde_json::Value) -> bool {
    if request_has_tools(value) || request_is_multimodal(value) {
        return false;
    }
    // Fields that steer the engine's template tokenization but which the
    // router's encoder does not thread through.
    for key in [
        "chat_template",
        "chat_template_kwargs",
        "reasoning",
        "reasoning_effort",
        "task",
    ] {
        if value.get(key).is_some_and(|v| !v.is_null()) {
            return false;
        }
    }
    if value
        .get("continue_final_message")
        .and_then(|v| v.as_bool())
        == Some(true)
    {
        return false;
    }
    !last_message_is_assistant(value)
}

/// Whether the ingress tokenization offload was expected to fire but failed —
/// the condition behind `sgl_router_ingress_tokenize_errors_total`.
///
/// True only when ALL of:
///   * the model has a chat encoder (`has_chat_encoder`), so a chat request
///     on it SHOULD have produced engine-equivalent ids;
///   * the request is a chat request (`messages` array present);
///   * the tokens are absent OR not engine-equivalent — i.e. `encode_chat`
///     render/encode failed and the request silently fell back to engine-side
///     tokenization.
///
/// Non-chat-encoder / non-`messages` requests never expected the offload, so
/// they are not failures. A tools / multimodal / thinking request on a
/// chat-encoder model still gets engine-equivalent ids (`encode_chat`
/// succeeded; the safe-predicate withholds forwarding for other reasons), so it
/// is an expected omission, not a failure.
fn ingress_tokenize_offload_failed(
    has_chat_encoder: bool,
    request_value: Option<&serde_json::Value>,
    request_tokens: Option<&RequestTokens>,
) -> bool {
    if !has_chat_encoder {
        return false;
    }
    let chat_request =
        request_value.is_some_and(|v| v.get("messages").is_some_and(|m| m.is_array()));
    if !chat_request {
        return false;
    }
    !request_tokens.is_some_and(|t| t.engine_equivalent)
}

/// Whether the final chat message has `role: "assistant"` (a prefix /
/// continuation turn the engine's template path special-cases).
fn last_message_is_assistant(value: &serde_json::Value) -> bool {
    value
        .get("messages")
        .and_then(|m| m.as_array())
        .and_then(|msgs| msgs.last())
        .and_then(|m| m.get("role"))
        .and_then(|r| r.as_str())
        == Some("assistant")
}

/// Whether the request carries tool / function definitions. The router's chat
/// encoder renders only `messages`, so its `input_ids` would omit the tool
/// schemas the engine's template injects into the prompt — the caller must let
/// the engine tokenize these itself.
fn request_has_tools(value: &serde_json::Value) -> bool {
    let nonempty = |key: &str| {
        value.get(key).is_some_and(|v| match v {
            serde_json::Value::Array(a) => !a.is_empty(),
            serde_json::Value::Null => false,
            _ => true,
        })
    };
    nonempty("tools") || nonempty("functions")
}

/// Whether any message carries non-string (array / multimodal) content. A text
/// tokenizer cannot represent image content, so the router's `input_ids` would
/// drop it — the caller must let the engine handle these requests.
fn request_is_multimodal(value: &serde_json::Value) -> bool {
    value
        .get("messages")
        .and_then(|m| m.as_array())
        .is_some_and(|msgs| {
            msgs.iter()
                .any(|m| matches!(m.get("content"), Some(serde_json::Value::Array(_))))
        })
}

/// Apply the fleet-wide sampling contract (`--override-sampling-params` /
/// `--sampling-param-conflict`) to one request, before admission.
///
/// Returns the inject-set: the configured value for every exact-valued
/// parameter the request OMITTED, which [`build_outgoing_body`] writes into
/// the forwarded body so the engine's own defaults can't drift from what the
/// operator declared. A band ([`ParamSpec::Range`]) names no single value, so
/// it never injects.
///
/// For a parameter the request DID send:
///   * [`ConflictPolicy::Allow`] forwards the client value untouched, which
///     is why the mode check comes before any comparison;
///   * [`ConflictPolicy::Reject`] 400s a numeric value that differs from the
///     configured one (or falls outside the band) — never a silent rewrite,
///     which is the one behavior no client can detect;
///   * a value that is not a number ([`ProbedValue::Unusable`]) is nobody's
///     business but the engine's, which is authoritative for the request
///     schema and produces the better message.
///
/// A rejection is counted per parameter before it is returned, because a
/// contract rollout turns served traffic into 400s and the operator needs to
/// see how much and where.
fn apply_sampling_overrides(
    overrides: &SamplingOverrides,
    probe: &RequestProbe,
    metrics: &MetricsRegistry,
) -> Result<Vec<(SamplingField, serde_json::Number)>, ApiError> {
    // Length is a startup constant, and `Vec::with_capacity(0)` does not
    // allocate — so an unconfigured contract still costs nothing.
    let mut inject = Vec::with_capacity(overrides.params.len());
    for (&field, spec) in &overrides.params {
        let name = field.wire_name();
        let reject = |detail: String| {
            metrics.record_sampling_contract_rejection(name);
            ApiError::SamplingContract {
                param: name,
                detail,
            }
        };
        let got = match probe.sampling_field(field) {
            // A band names no single value, so it never injects — a request
            // that omits the parameter gets the engine's own default.
            ProbedValue::Absent => {
                if let ParamSpec::Exact(v) = spec {
                    inject.push((field, v.clone()));
                }
                continue;
            }
            ProbedValue::Unusable => continue,
            // `allow` forwards a client value untouched, so the comparison
            // below is `reject`-only.
            ProbedValue::Number(_) if overrides.conflict == ConflictPolicy::Allow => continue,
            ProbedValue::Number(got) => got,
        };
        match spec {
            ParamSpec::Exact(want) => {
                if Some(got) != want.as_f64() {
                    return Err(reject(format!(
                        "got {got}, expected {want} (or omit the field)"
                    )));
                }
            }
            &ParamSpec::Range { lo, hi } => {
                if !(lo..=hi).contains(&got) {
                    return Err(reject(format!("must be between {lo} and {hi}, got {got}")));
                }
            }
        }
    }
    Ok(inject)
}

fn parse_probe(body: &Bytes) -> Result<RequestProbe, ApiError> {
    // We deliberately do NOT echo the serde error into the client-visible
    // message — that risks leaking field-level detail and is also of little
    // help to a real client (which already has its own JSON validator).
    // Server-side, the full error is logged with `tracing::debug!` for
    // operator triage.
    //
    // ONE deserialize. [`RequestProbe`]'s hand-written `visit_map` both
    // anchors the shape (its `deserialize_map` rejects `null` / `[]` / `"hi"`
    // — valid JSON, not request shape) and lifts out the probed fields,
    // skipping the unknown majority through `IgnoredAny`. It never builds a
    // `serde_json::Value` and allocates nothing per unrecognized key, so the
    // shape check costs no separate pass over a multi-MiB body.
    serde_json::from_slice(body).map_err(|e| {
        tracing::debug!(error = %e, "chat-completions request-probe deserialize failed");
        ApiError::BadRequest("invalid request: body must be a JSON object".to_string())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An unavailable indexer must never fail a request that min-load routing
    /// can still serve. `QueryTooLarge` belongs here too: a prompt that outgrows
    /// the query's message limit loses cache affinity, not availability.
    #[test]
    fn unavailable_indexer_degrades_to_empty_prefix_signal() {
        for error in [
            sgl_kv_indexer::PrefixIndexError::Overloaded,
            sgl_kv_indexer::PrefixIndexError::Timeout,
            sgl_kv_indexer::PrefixIndexError::Unreachable,
            sgl_kv_indexer::PrefixIndexError::QueryTooLarge,
        ] {
            assert_eq!(
                resolve_prefix_query(Err(error.clone()), "tiny").unwrap(),
                sgl_kv_indexer::PrefixOutcome::Empty,
                "{error} should degrade"
            );
        }
    }

    #[test]
    fn rejected_indexer_query_still_fails_selection() {
        assert!(matches!(
            resolve_prefix_query(
                Err(sgl_kv_indexer::PrefixIndexError::Rejected(
                    sgl_kv_indexer::RpcCode::InvalidArgument
                )),
                "tiny"
            ),
            Err(ApiError::PolicySelectionFailed { .. })
        ));
    }

    #[test]
    fn bucket_routing_requests_tokens_even_for_a_non_token_policy() {
        assert!(should_tokenize_request(false, false, true));
        assert!(!should_tokenize_request(false, false, false));
    }

    #[test]
    fn session_reason_distinguishes_hit_assignment_and_keyless_fallback() {
        assert_eq!(
            prefill_policy_reason(
                PolicyKind::SessionAware,
                ProposalKind::SessionAffinity,
                DecisionReason::Primary,
                true,
                true,
            ),
            "session_primary"
        );
        assert_eq!(
            prefill_policy_reason(
                PolicyKind::SessionAware,
                ProposalKind::PowerOfTwo,
                DecisionReason::Primary,
                true,
                true,
            ),
            "assigned"
        );
        assert_eq!(
            prefill_policy_reason(
                PolicyKind::SessionAware,
                ProposalKind::PowerOfTwo,
                DecisionReason::Primary,
                false,
                true,
            ),
            "no_session"
        );
    }

    #[test]
    fn session_reason_preserves_admission_and_pressure_escapes() {
        assert_eq!(
            prefill_policy_reason(
                PolicyKind::SessionAware,
                ProposalKind::SessionAffinity,
                DecisionReason::BackupPrimaryAdmission,
                true,
                true,
            ),
            "session_admission_backup"
        );
        assert_eq!(
            prefill_policy_reason(
                PolicyKind::SessionAware,
                ProposalKind::SessionAffinity,
                DecisionReason::BackupPressureGuard,
                true,
                true,
            ),
            "session_pressure_backup"
        );
    }

    #[test]
    fn cache_no_winner_p2_is_distinct_from_cache_candidate() {
        assert_eq!(
            prefill_policy_reason(
                PolicyKind::CacheAware,
                ProposalKind::PowerOfTwo,
                DecisionReason::Primary,
                false,
                false,
            ),
            "no_cache_candidate"
        );
        assert_eq!(
            prefill_policy_reason(
                PolicyKind::CacheAware,
                ProposalKind::CacheAffinity,
                DecisionReason::Primary,
                false,
                true,
            ),
            "cache_candidate"
        );
    }
    /// `generate_room_id` MUST return values in `[0, i64::MAX]`. The
    /// SGLang prefill stores `bootstrap_room` as `torch.int64`; a u64
    /// with the top bit set would wrap negative on the engine side.
    /// Sample many times to defend against future refactors of the
    /// mask (e.g. someone "simplifying" to plain `rand::random::<u64>()`).
    #[test]
    fn generate_room_id_stays_in_63_bit_range() {
        for _ in 0..10_000 {
            let r = generate_room_id();
            assert!(
                r <= i64::MAX as u64,
                "generate_room_id() returned {r} > i64::MAX; would wrap negative as torch.int64",
            );
        }
    }

    /// When the prefill worker has no `bootstrap_port` configured
    /// (a misconfiguration the engine will reject loudly), the
    /// injected field MUST be JSON `null` — not omitted, not 0.
    /// SGLang's validator distinguishes "missing field" from
    /// "null field" in some code paths.
    #[test]
    fn build_outgoing_body_emits_null_for_missing_port() {
        let body = Bytes::from_static(br#"{"model":"x","messages":[]}"#);
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let bootstrap = BootstrapFields {
            host: "host".into(),
            port: None,
            room: 42,
        };
        let injected =
            build_outgoing_body(&body, Some(value), None, Some(&bootstrap), &[]).unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&injected).unwrap();
        assert_eq!(parsed.get("bootstrap_port"), Some(&serde_json::Value::Null));
        assert_eq!(
            parsed.get("bootstrap_host"),
            Some(&serde_json::Value::String("host".into()))
        );
        assert_eq!(
            parsed.get("bootstrap_room"),
            Some(&serde_json::Value::Number(42.into()))
        );
    }

    /// `input_ids` are injected and `messages` retained (the engine still
    /// needs them for stop tokens / tool-call constraint / response shape).
    #[test]
    fn build_outgoing_body_injects_input_ids_and_keeps_messages() {
        let body =
            Bytes::from_static(br#"{"model":"x","messages":[{"role":"user","content":"hi"}]}"#);
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let ids = [1u32, 2, 3];
        let out = build_outgoing_body(&body, Some(value), Some(&ids), None, &[]).unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(parsed.get("input_ids"), Some(&serde_json::json!([1, 2, 3])));
        assert!(
            parsed.get("messages").is_some(),
            "messages must be retained alongside input_ids"
        );
    }

    /// With nothing to inject, the original bytes are forwarded unchanged
    /// (no re-serialize) — the transparent no-op fallback.
    #[test]
    fn build_outgoing_body_no_injection_returns_original_bytes() {
        let body = Bytes::from_static(br#"{"model":"x","messages":[]}"#);
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let out = build_outgoing_body(&body, Some(value), None, None, &[]).unwrap();
        assert_eq!(
            out, body,
            "no injection must forward the original bytes unchanged"
        );
    }

    /// PD + forwarding: both `input_ids` and the bootstrap fields land in one
    /// serialized body.
    #[test]
    fn build_outgoing_body_injects_both_input_ids_and_bootstrap() {
        let body =
            Bytes::from_static(br#"{"model":"x","messages":[{"role":"user","content":"hi"}]}"#);
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let ids = [7u32, 8];
        let bootstrap = BootstrapFields {
            host: "h".into(),
            port: Some(9),
            room: 5,
        };
        let out =
            build_outgoing_body(&body, Some(value), Some(&ids), Some(&bootstrap), &[]).unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(parsed.get("input_ids"), Some(&serde_json::json!([7, 8])));
        assert_eq!(
            parsed.get("bootstrap_room"),
            Some(&serde_json::Value::Number(5.into()))
        );
        assert_eq!(
            parsed.get("bootstrap_port"),
            Some(&serde_json::Value::Number(9.into()))
        );
    }

    /// Tool / function requests are detected so the caller omits `input_ids`
    /// (the router's encoder doesn't render tools).
    #[test]
    fn request_has_tools_detects_tools_and_functions() {
        assert!(request_has_tools(
            &serde_json::json!({"tools":[{"type":"function"}]})
        ));
        assert!(request_has_tools(
            &serde_json::json!({"functions":[{"name":"f"}]})
        ));
        assert!(!request_has_tools(&serde_json::json!({"tools":[]})));
        assert!(!request_has_tools(&serde_json::json!({"messages":[]})));
    }

    /// Array (multimodal) message content is detected so the caller omits
    /// `input_ids` (a text tokenizer can't represent image content).
    #[test]
    fn request_is_multimodal_detects_array_content() {
        assert!(request_is_multimodal(&serde_json::json!({
            "messages":[{"role":"user","content":[{"type":"image_url","image_url":"x"}]}]
        })));
        assert!(!request_is_multimodal(&serde_json::json!({
            "messages":[{"role":"user","content":"hello"}]
        })));
    }

    /// Plain text chat with nothing unreplicated → input_ids may be forwarded.
    #[test]
    fn input_ids_safe_to_forward_allows_plain_text_chat() {
        assert!(input_ids_safe_to_forward(&serde_json::json!({
            "messages": [{"role": "user", "content": "hello"}]
        })));
    }

    /// Every field the engine honors on the `messages` path but which the
    /// router's encoder does not replicate must block forwarding — otherwise
    /// the engine uses the router's ids verbatim and silently runs a different
    /// prompt than the request asked for.
    #[test]
    fn input_ids_safe_to_forward_blocks_unreplicated_signals() {
        let blockers = [
            serde_json::json!({"messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function"}]}),
            serde_json::json!({"messages":[{"role":"user","content":[{"type":"image_url","image_url":"x"}]}]}),
            serde_json::json!({"messages":[{"role":"user","content":"hi"}],"chat_template":"{{ custom }}"}),
            serde_json::json!({"messages":[{"role":"user","content":"hi"}],"chat_template_kwargs":{"enable_thinking":true}}),
            serde_json::json!({"messages":[{"role":"user","content":"hi"}],"reasoning_effort":"high"}),
            serde_json::json!({"messages":[{"role":"user","content":"hi"}],"reasoning":{"enabled":true}}),
            serde_json::json!({"messages":[{"role":"user","content":"hi"}],"task":"generate"}),
            serde_json::json!({"messages":[{"role":"user","content":"hi"}],"continue_final_message":true}),
            serde_json::json!({"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"partial"}]}),
        ];
        for b in blockers {
            assert!(
                !input_ids_safe_to_forward(&b),
                "must NOT forward input_ids for: {b}"
            );
        }
    }

    /// Null / false-valued fields do not block (absent ≡ null ≡ default).
    #[test]
    fn input_ids_safe_to_forward_ignores_null_and_false_fields() {
        assert!(input_ids_safe_to_forward(&serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "chat_template": null,
            "reasoning_effort": null,
            "chat_template_kwargs": null,
            "continue_final_message": false
        })));
    }

    /// Load-only + PD: `build_outgoing_body` is handed `None` for the value
    /// (the ingress skipped the parse for a load-only policy) and re-parses the
    /// bytes to inject the bootstrap fields. `input_ids` is never set here.
    #[test]
    fn build_outgoing_body_reparses_when_value_absent() {
        let body = Bytes::from_static(br#"{"model":"x","messages":[]}"#);
        let bootstrap = BootstrapFields {
            host: "h".into(),
            port: Some(1),
            room: 2,
        };
        let out = build_outgoing_body(&body, None, None, Some(&bootstrap), &[]).unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(
            parsed.get("bootstrap_room"),
            Some(&serde_json::Value::Number(2.into()))
        );
        assert!(parsed.get("input_ids").is_none());
    }

    /// A chat request on a chat-encoder model that yields engine-equivalent
    /// ids (encode succeeded) is NOT a failure — the offload worked.
    #[test]
    fn offload_failed_false_when_tokens_engine_equivalent() {
        let value = serde_json::json!({"messages":[{"role":"user","content":"hi"}]});
        let tokens = RequestTokens {
            ids: vec![1, 2, 3],
            engine_equivalent: true,
        };
        assert!(!ingress_tokenize_offload_failed(
            true,
            Some(&value),
            Some(&tokens)
        ));
    }

    /// A chat request on a chat-encoder model whose tokenization yielded NO
    /// tokens (encode_chat returned None → request_tokens None) IS a failure:
    /// the encoder should have fired but didn't.
    #[test]
    fn offload_failed_true_when_chat_encoder_request_has_no_tokens() {
        let value = serde_json::json!({"messages":[{"role":"user","content":"hi"}]});
        assert!(ingress_tokenize_offload_failed(true, Some(&value), None));
    }

    /// Encode produced ids but NOT via the chat encoder (raw fallback,
    /// `engine_equivalent = false`) on a chat-encoder model + chat request →
    /// the chat-encode render/encode failed and fell through to the raw path.
    #[test]
    fn offload_failed_true_when_tokens_not_engine_equivalent() {
        let value = serde_json::json!({"messages":[{"role":"user","content":"hi"}]});
        let tokens = RequestTokens {
            ids: vec![1, 2, 3],
            engine_equivalent: false,
        };
        assert!(ingress_tokenize_offload_failed(
            true,
            Some(&value),
            Some(&tokens)
        ));
    }

    /// Non-chat-encoder models never expected the offload → not a failure even
    /// with no tokens.
    #[test]
    fn offload_failed_false_without_chat_encoder() {
        let value = serde_json::json!({"messages":[{"role":"user","content":"hi"}]});
        assert!(!ingress_tokenize_offload_failed(false, Some(&value), None));
    }

    /// A non-chat (no `messages`) request on a chat-encoder model — e.g.
    /// `/v1/completions` `prompt` — never expected the chat-encode offload, so
    /// the absence of engine-equivalent ids is not a failure.
    #[test]
    fn offload_failed_false_for_non_messages_request() {
        let value = serde_json::json!({"prompt":"hi"});
        assert!(!ingress_tokenize_offload_failed(true, Some(&value), None));
    }

    #[test]
    fn parse_probe_reads_stream_bool_from_object() {
        let b = Bytes::from_static(br#"{"stream": true, "model": "tiny"}"#);
        assert_eq!(parse_probe(&b).unwrap().stream, Some(true));
        let b = Bytes::from_static(br#"{"stream": false, "model": "tiny"}"#);
        assert_eq!(parse_probe(&b).unwrap().stream, Some(false));
    }

    #[test]
    fn parse_probe_defaults_when_stream_absent() {
        // Existing happy-path contract: well-formed object missing `stream`
        // must default to None (caller picks false). The minimal `RequestProbe`
        // (Option<bool> + #[serde(default)]) must NOT break this.
        let b = Bytes::from_static(br#"{"model": "tiny", "messages": []}"#);
        let p = parse_probe(&b).unwrap();
        assert_eq!(p.stream, None);
        assert_eq!(p.model.as_deref(), Some("tiny"));
    }

    #[test]
    fn parse_probe_accepts_modern_openai_completion_budget() {
        let body =
            Bytes::from_static(br#"{"model":"tiny","messages":[],"max_completion_tokens":256}"#);
        assert_eq!(
            parse_probe(&body).unwrap().requested_max_output_tokens(),
            Some(256)
        );
    }

    #[test]
    fn modern_completion_budget_takes_precedence_when_both_fields_are_present() {
        let body =
            Bytes::from_static(br#"{"model":"tiny","max_tokens":128,"max_completion_tokens":256}"#);
        assert_eq!(
            parse_probe(&body).unwrap().requested_max_output_tokens(),
            Some(256)
        );
    }

    #[test]
    fn decode_kv_projection_includes_the_explicit_output_budget() {
        assert_eq!(projected_decode_kv_tokens(1_024, Some(512)), 1_536);
        assert_eq!(projected_decode_kv_tokens(1_024, None), 1_024);
        assert_eq!(projected_decode_kv_tokens(u64::MAX - 1, Some(8)), u64::MAX);
    }

    #[test]
    fn parse_probe_rejects_non_object_shapes() {
        // Pin the contract: degenerate JSON (valid JSON but wrong shape)
        // must be rejected, not silently forwarded with `stream=false`.
        for bad in [&b"null"[..], &b"[]"[..], &b"\"hi\""[..], &b"42"[..]] {
            let b = Bytes::copy_from_slice(bad);
            let err = parse_probe(&b).unwrap_err();
            match err {
                ApiError::BadRequest(_) => {}
                other => panic!("expected BadRequest for {bad:?}, got {other:?}"),
            }
        }
    }

    #[test]
    fn parse_probe_rejects_malformed_json() {
        let b = Bytes::from_static(b"{not json}");
        let err = parse_probe(&b).unwrap_err();
        assert!(matches!(err, ApiError::BadRequest(_)));
    }

    #[test]
    fn parse_probe_handles_nested_messages_with_stream_true() {
        // Well-formed object with nested arrays/objects (real chat-completions
        // payloads carry `messages: [{role, content: [{type, text}]}]`). The
        // two-step deserialize must not balk on this — only the top-level
        // object shape and the `stream`/`model` fields matter.
        let b = Bytes::from_static(
            br#"{
              "model": "x",
              "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
              "stream": true
            }"#,
        );
        assert_eq!(parse_probe(&b).unwrap().stream, Some(true));
    }

    #[test]
    fn parse_probe_handles_nested_messages_with_stream_false() {
        let b = Bytes::from_static(
            br#"{
              "model": "x",
              "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
              "stream": false
            }"#,
        );
        assert_eq!(parse_probe(&b).unwrap().stream, Some(false));
    }

    #[test]
    fn parse_probe_handles_duplicate_stream_keys() {
        // RFC 8259 says "names within an object SHOULD be unique" but a
        // parser MAY accept duplicates. Step 1 (HashMap) silently
        // last-wins, but step 2 deserializes into the typed `RequestProbe`
        // struct, and `serde_json`'s `#[derive(Deserialize)]` REJECTS
        // duplicate fields with a `duplicate field` error.
        //
        // We map that to `BadRequest` (same path as other malformed input).
        // Pinning "reject" rather than "last-wins" is intentional —
        // ambiguous bodies should fail loudly at the edge, not silently
        // route based on which copy serde happened to see last.
        let b = Bytes::from_static(br#"{"stream": true, "stream": false}"#);
        let err = parse_probe(&b).unwrap_err();
        match err {
            ApiError::BadRequest(_) => {}
            other => panic!("expected BadRequest on duplicate `stream` key, got {other:?}"),
        }
    }

    #[test]
    fn parse_probe_bad_request_message_does_not_leak_serde_detail() {
        // Info-leak guard: the client-visible message must be a fixed
        // string, not the serde error (which can contain line/column
        // detail or hint at field shape).
        let b = Bytes::from_static(br#"{"stream": "not-a-bool"}"#);
        let err = parse_probe(&b).unwrap_err();
        match err {
            ApiError::BadRequest(msg) => assert_eq!(
                msg, "invalid request: body must be a JSON object",
                "client-visible message must be fixed; got: {msg}"
            ),
            other => panic!("expected BadRequest, got {other:?}"),
        }
    }

    fn probe_of(body: &str) -> RequestProbe {
        parse_probe(&Bytes::copy_from_slice(body.as_bytes())).unwrap()
    }

    /// Build a [`SamplingOverrides`] the only way production does — through
    /// the flag parser.
    ///
    /// Hand-building the struct here would re-implement `canonical_number`'s
    /// integral normalization in the test, so a regression in the parser would
    /// leave these assertions green while the forwarded body changed.
    fn overrides_of(conflict: ConflictPolicy, json: &str) -> SamplingOverrides {
        crate::config::parse_sampling_overrides(json, conflict).expect("test config must parse")
    }

    /// A throwaway registry, so the contract's rejection counter has somewhere
    /// to land.
    fn metrics() -> Arc<MetricsRegistry> {
        MetricsRegistry::new()
    }

    /// With nothing configured the sampling contract is inert: no request is
    /// ever inspected, and nothing is injected.
    #[test]
    fn unconfigured_sampling_overrides_inject_nothing() {
        let overrides = SamplingOverrides::default();
        let p = probe_of(r#"{"model":"x","temperature":0.7,"n":4}"#);
        assert_eq!(
            apply_sampling_overrides(&overrides, &p, &metrics()).unwrap(),
            vec![]
        );
    }

    /// The `reject` contract at the decision level: omitted -> inject the
    /// configured value; equal to it -> pass untouched; any other numeric
    /// value -> 400; non-numeric garbage -> forwarded for the engine's own
    /// schema error. A band admits its range, 400s outside it, and injects
    /// nothing.
    #[test]
    fn reject_mode_pins_configured_values_and_admits_a_band() {
        let overrides = overrides_of(
            ConflictPolicy::Reject,
            r#"{"top_p": 0.95, "frequency_penalty": 0.0, "presence_penalty": 0.0,
                "n": 1, "temperature": {"min": 0, "max": 1}}"#,
        );

        // Omitted params: accepted, exact values injected, band injects nothing.
        let p = probe_of(r#"{"model":"x","messages":[]}"#);
        let inject = apply_sampling_overrides(&overrides, &p, &metrics()).unwrap();
        assert_eq!(
            inject
                .iter()
                .map(|(f, v)| (f.wire_name(), v.to_string()))
                .collect::<Vec<_>>(),
            vec![
                ("top_p", "0.95".to_string()),
                ("frequency_penalty", "0.0".to_string()),
                ("presence_penalty", "0.0".to_string()),
                ("n", "1".to_string()),
            ]
        );

        for accepted in [
            r#"{"model":"x","temperature":0.0}"#,
            r#"{"model":"x","temperature":0.6}"#,
            r#"{"model":"x","temperature":1.0}"#,
            r#"{"model":"x","top_p":0.95}"#,
            r#"{"model":"x","presence_penalty":0}"#,
            r#"{"model":"x","frequency_penalty":0}"#,
            r#"{"model":"x","n":1}"#,
        ] {
            let p = probe_of(accepted);
            apply_sampling_overrides(&overrides, &p, &metrics())
                .unwrap_or_else(|e| panic!("{accepted} must be accepted: {e:?}"));
        }

        // Nothing is injected over a field the client already sent.
        let p = probe_of(r#"{"model":"x","top_p":0.95}"#);
        let inject = apply_sampling_overrides(&overrides, &p, &metrics()).unwrap();
        assert!(!inject.iter().any(|(f, _)| *f == SamplingField::TopP));

        for rejected in [
            r#"{"model":"x","temperature":1.1}"#,
            r#"{"model":"x","temperature":2.0}"#,
            r#"{"model":"x","temperature":-0.1}"#,
            r#"{"model":"x","top_p":0.8}"#,
            r#"{"model":"x","presence_penalty":0.5}"#,
            r#"{"model":"x","frequency_penalty":0.5}"#,
            r#"{"model":"x","n":2}"#,
        ] {
            let p = probe_of(rejected);
            let err = apply_sampling_overrides(&overrides, &p, &metrics())
                .expect_err(&format!("{rejected} must be rejected"));
            assert!(
                matches!(err, ApiError::SamplingContract { .. }),
                "{rejected}: got {err:?}"
            );
        }

        // Non-numeric garbage is not ours to judge: forwarded untouched (and
        // not injected over), the engine's schema validation owns the 400.
        let p = probe_of(r#"{"model":"x","top_p":"hot","n":true}"#);
        let inject = apply_sampling_overrides(&overrides, &p, &metrics()).unwrap();
        assert!(inject
            .iter()
            .all(|(f, _)| !matches!(f, SamplingField::TopP | SamplingField::N)));

        // Numeric strings coerce the way the engine's pydantic lax mode does:
        // "0.95" equals the configured value, "0.8" differs and 400s here.
        let p = probe_of(r#"{"model":"x","top_p":"0.95"}"#);
        assert!(apply_sampling_overrides(&overrides, &p, &metrics()).is_ok());
        let p = probe_of(r#"{"model":"x","top_p":"0.8"}"#);
        assert!(apply_sampling_overrides(&overrides, &p, &metrics()).is_err());

        // An exact temperature (no band) rejects differing values and injects
        // when absent, like every other parameter.
        let exact = overrides_of(ConflictPolicy::Reject, r#"{"temperature": 1.0}"#);
        let p = probe_of(r#"{"model":"x","temperature":0.6}"#);
        assert!(apply_sampling_overrides(&exact, &p, &metrics()).is_err());
        let p = probe_of(r#"{"model":"x"}"#);
        assert_eq!(
            apply_sampling_overrides(&exact, &p, &metrics()).unwrap(),
            vec![(
                SamplingField::Temperature,
                serde_json::Number::from_f64(1.0).unwrap()
            )]
        );
    }

    /// `allow` keeps the fill-when-absent half of the contract and drops the
    /// rejection half: a client value — right, wrong or garbage — is forwarded
    /// untouched, so the configured values are fleet-wide defaults.
    #[test]
    fn allow_mode_never_rejects_and_never_masks_a_client_value() {
        let overrides = overrides_of(
            ConflictPolicy::Allow,
            r#"{"temperature": 1, "top_p": 0.95, "n": 1}"#,
        );

        // Omitted -> injected, exactly as under `reject`.
        let p = probe_of(r#"{"model":"x"}"#);
        assert_eq!(
            apply_sampling_overrides(&overrides, &p, &metrics())
                .unwrap()
                .iter()
                .map(|(f, _)| f.wire_name())
                .collect::<Vec<_>>(),
            vec!["temperature", "top_p", "n"]
        );

        // Every value `reject` would 400 is accepted here, and nothing is
        // injected over it — the client's value reaches the engine.
        let p = probe_of(r#"{"model":"x","temperature":0.6,"top_p":0.8,"n":4}"#);
        assert_eq!(
            apply_sampling_overrides(&overrides, &p, &metrics()).unwrap(),
            vec![]
        );

        // Partial overlap: the client set temperature, so only the untouched
        // parameters are filled in.
        let p = probe_of(r#"{"model":"x","temperature":0.6}"#);
        assert_eq!(
            apply_sampling_overrides(&overrides, &p, &metrics())
                .unwrap()
                .iter()
                .map(|(f, _)| f.wire_name())
                .collect::<Vec<_>>(),
            vec!["top_p", "n"]
        );
    }

    /// An explicit `null` is absent for the engine, so it is absent here too:
    /// the configured value is injected rather than the field being read as a
    /// client-supplied conflict.
    #[test]
    fn explicit_null_sampling_value_counts_as_omitted() {
        let overrides = overrides_of(ConflictPolicy::Reject, r#"{"temperature": 1}"#);
        let p = probe_of(r#"{"model":"x","temperature":null}"#);
        assert_eq!(
            apply_sampling_overrides(&overrides, &p, &metrics())
                .unwrap()
                .iter()
                .map(|(f, _)| f.wire_name())
                .collect::<Vec<_>>(),
            vec!["temperature"]
        );
    }

    /// The inject-set lands in the forwarded body, and rides the same
    /// `build_outgoing_body` serialize as the forwarded `input_ids` — so
    /// chat-encoder traffic gets the contract too, not just the raw-prompt
    /// path.
    #[test]
    fn build_outgoing_body_injects_sampling_overrides_alongside_input_ids() {
        let body =
            Bytes::from_static(br#"{"model":"x","messages":[{"role":"user","content":"hi"}]}"#);
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let ids = [1u32, 2, 3];
        let overrides = overrides_of(
            ConflictPolicy::Reject,
            r#"{"top_p": 0.95, "top_k": 1000, "frequency_penalty": 0.0,
                "presence_penalty": 0.0, "n": 1}"#,
        );
        let inject = apply_sampling_overrides(
            &overrides,
            &probe_of(r#"{"model":"x","messages":[{"role":"user","content":"hi"}]}"#),
            &metrics(),
        )
        .unwrap();
        let out = build_outgoing_body(&body, Some(value), Some(&ids), None, &inject).unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(parsed.get("top_p"), Some(&serde_json::json!(0.95)));
        // `top_k` and `n` are engine-typed `int`: the injected literals must
        // not be `1000.0` / `1.0`.
        assert_eq!(parsed.get("top_k"), Some(&serde_json::json!(1000)));
        assert_eq!(parsed.get("n"), Some(&serde_json::json!(1)));
        assert_eq!(
            parsed.get("frequency_penalty"),
            Some(&serde_json::json!(0.0))
        );
        assert_eq!(
            parsed.get("presence_penalty"),
            Some(&serde_json::json!(0.0))
        );
        assert_eq!(parsed.get("input_ids"), Some(&serde_json::json!([1, 2, 3])));
        assert!(parsed.get("messages").is_some());
    }
    /// A repeated sampling key LAST-WINS instead of 400ing, matching the
    /// engine's own `json.loads`: the value the contract compares against must
    /// be the value the engine will actually sample with. Probing these fields
    /// must not turn a body the router used to forward into a rejection.
    #[test]
    fn duplicate_sampling_key_takes_the_last_value_like_the_engine() {
        let overrides = overrides_of(ConflictPolicy::Reject, r#"{"temperature": 1}"#);

        // Last value matches the contract -> accepted.
        let p = probe_of(r#"{"model":"x","temperature":0.5,"temperature":1}"#);
        assert!(apply_sampling_overrides(&overrides, &p, &metrics()).is_ok());

        // Last value differs -> rejected on THAT value, not the first one.
        let p = probe_of(r#"{"model":"x","temperature":1,"temperature":0.5}"#);
        let err = apply_sampling_overrides(&overrides, &p, &metrics()).unwrap_err();
        assert!(
            format!("{err}").contains("got 0.5"),
            "must judge the last value; got {err}"
        );
    }

    /// The router-actionable fields keep the stricter pre-existing contract:
    /// a body that says two different things about how to route itself is
    /// ambiguous at the edge. See
    /// `parse_probe_handles_duplicate_stream_keys`.
    #[test]
    fn duplicate_routing_key_still_rejects() {
        for body in [
            r#"{"model":"a","model":"b"}"#,
            r#"{"stream":true,"stream":false}"#,
            r#"{"max_tokens":1,"max_tokens":2}"#,
            r#"{"max_completion_tokens":1,"max_completion_tokens":2}"#,
            // An explicit null still counts as an occurrence, so this is a
            // duplicate even though the first value reads as `None`.
            r#"{"stream":null,"stream":true}"#,
        ] {
            let b = Bytes::copy_from_slice(body.as_bytes());
            assert!(
                parse_probe(&b).is_err(),
                "{body} must be rejected as ambiguous"
            );
        }
    }

    /// A sampling key carrying a huge non-numeric value must cost nothing: it
    /// is drained, not materialized, and it does not fail the probe. Before
    /// these fields were probed the value was skipped by `IgnoredAny`, and
    /// probing them must not put a client-sized allocation back on the path
    /// nor start rejecting bodies that used to be forwarded.
    #[test]
    fn oversized_non_numeric_sampling_value_is_drained_not_materialized() {
        let big_array = format!("[{}]", "1,".repeat(50_000) + "1");
        let big_string = format!("\"{}\"", "x".repeat(200_000));
        let big_object = format!("{{{}\"k\":1}}", "\"j\":[[[1]]],".repeat(10_000));
        for value in [&big_array, &big_string, &big_object] {
            let body = format!(r#"{{"model":"x","temperature":{value}}}"#);
            let probe = probe_of(&body);
            assert_eq!(
                probe.sampling_field(SamplingField::Temperature),
                ProbedValue::Unusable,
                "a non-numeric value must collapse to Unusable"
            );
            // ...and it stays the engine's business: neither compared nor
            // injected over.
            let overrides = overrides_of(ConflictPolicy::Reject, r#"{"temperature": 1}"#);
            let inject = apply_sampling_overrides(&overrides, &probe, &metrics()).unwrap();
            assert!(inject.is_empty(), "must not inject over a client value");
        }
    }

    /// A numeric string is read the way the engine's pydantic lax mode reads
    /// it, but is not retained as a string.
    #[test]
    fn probed_sampling_values_normalize_to_numbers() {
        let p = probe_of(r#"{"model":"x","temperature":" 0.7 ","top_k":40,"min_p":0.05}"#);
        assert_eq!(
            p.sampling_field(SamplingField::Temperature),
            ProbedValue::Number(0.7)
        );
        assert_eq!(
            p.sampling_field(SamplingField::TopK),
            ProbedValue::Number(40.0)
        );
        assert_eq!(
            p.sampling_field(SamplingField::MinP),
            ProbedValue::Number(0.05)
        );
        assert_eq!(p.sampling_field(SamplingField::N), ProbedValue::Absent);
    }

    /// A contract rejection must be distinguishable from every other 400 —
    /// malformed JSON, a missing `model`, a bad SLO header — and must name the
    /// parameter, since that is what an operator rolling the flag out needs.
    #[test]
    fn contract_rejection_has_its_own_error_code_and_counter() {
        let overrides = overrides_of(ConflictPolicy::Reject, r#"{"top_p": 0.95}"#);
        let metrics = metrics();
        let p = probe_of(r#"{"model":"x","top_p":0.5}"#);

        let err = apply_sampling_overrides(&overrides, &p, &metrics).unwrap_err();
        let ApiError::SamplingContract { param, .. } = &err else {
            panic!("expected SamplingContract, got {err:?}");
        };
        assert_eq!(*param, "top_p");
        // The parameter and the offending value both reach the client, so a
        // 400 is self-explanatory without an operator in the loop. The
        // distinct `x-router-error-code` is pinned in `server::error`.
        let msg = format!("{err}");
        assert!(msg.contains("top_p") && msg.contains("0.5"), "got {msg}");

        apply_sampling_overrides(&overrides, &p, &metrics).unwrap_err();
        assert!(
            metrics
                .render()
                .contains(r#"sgl_router_sampling_contract_rejections_total{param="top_p"} 2"#),
            "rejections must be counted per parameter:\n{}",
            metrics.render()
        );
    }

    /// The steady state of a governed fleet: a load-only policy on a model
    /// with no chat encoder, so the ingress never parsed, and only sampling
    /// scalars to add. This must NOT re-parse and re-serialize the body —
    /// proven by the original bytes surviving verbatim, which a
    /// `serde_json::Value` round-trip would have normalized away.
    #[test]
    fn build_outgoing_body_splices_sampling_without_reparsing() {
        let body = Bytes::from_static(br#"{ "model" : "x" ,  "messages" : [ ] }"#);
        let inject = apply_sampling_overrides(
            &overrides_of(ConflictPolicy::Reject, r#"{"temperature": 1.0, "n": 1}"#),
            &probe_of(r#"{"model":"x"}"#),
            &metrics(),
        )
        .unwrap();

        let out = build_outgoing_body(&body, None, None, None, &inject).unwrap();
        assert_eq!(
            std::str::from_utf8(&out).unwrap(),
            r#"{ "model" : "x" ,  "messages" : [ ] ,"temperature":1.0,"n":1}"#
        );
        let parsed: serde_json::Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(parsed.get("temperature"), Some(&serde_json::json!(1.0)));
        assert_eq!(parsed.get("n"), Some(&serde_json::json!(1)));
        assert_eq!(parsed.get("model"), Some(&serde_json::json!("x")));
    }

    /// Splice edge cases: an empty object must not gain a trailing comma, and
    /// leading whitespace before the root brace must not shift the insert.
    #[test]
    fn splice_top_level_handles_empty_objects_and_leading_whitespace() {
        let inject = apply_sampling_overrides(
            &overrides_of(ConflictPolicy::Reject, r#"{"temperature": 1.0}"#),
            &probe_of(r#"{"model":"x"}"#),
            &metrics(),
        )
        .unwrap();

        for (raw, want) in [
            (r#"{}"#, r#"{"temperature":1.0}"#),
            (r#"{ }"#, r#"{ "temperature":1.0}"#),
            ("\n\t {\"a\":1}", "\n\t {\"a\":1,\"temperature\":1.0}"),
            // A `}` inside a string literal is not the closing brace.
            (r#"{"a":"}"}"#, r#"{"a":"}","temperature":1.0}"#),
            // Trailing whitespace stays outside the object.
            ("{\"a\":1} \n", "{\"a\":1,\"temperature\":1.0} \n"),
        ] {
            let out = splice_top_level(&Bytes::copy_from_slice(raw.as_bytes()), &inject).unwrap();
            assert_eq!(std::str::from_utf8(&out).unwrap(), want, "input {raw:?}");
            serde_json::from_slice::<serde_json::Value>(&out)
                .unwrap_or_else(|e| panic!("{raw:?} spliced to invalid JSON: {e}"));
        }
    }

    /// Nothing configured -> the body is forwarded as the same `Bytes`, with
    /// neither a parse nor a copy.
    #[test]
    fn build_outgoing_body_without_injection_forwards_the_same_allocation() {
        let body = Bytes::from_static(br#"{"model":"x"}"#);
        let out = build_outgoing_body(&body, None, None, None, &[]).unwrap();
        assert_eq!(
            out.as_ptr(),
            body.as_ptr(),
            "must be an Arc clone, not a copy"
        );
    }
    /// A request sending an explicit `null` for a governed parameter is the
    /// one case where the inject-set and a key PRESENT in the body overlap:
    /// the probe reads `null` as absent (the OpenAI contract), so the value is
    /// injected even though the key is there. The injected value therefore has
    /// to win the engine's last-wins parse — which is why members are spliced
    /// in before the CLOSING brace. Inserting after the opening brace would
    /// leave the client's trailing `null` authoritative and silently defeat
    /// the contract.
    #[test]
    fn spliced_value_outranks_an_explicit_null_the_client_sent() {
        let overrides = overrides_of(ConflictPolicy::Reject, r#"{"temperature": 1.0}"#);
        let raw = r#"{"model":"x","temperature":null}"#;
        let body = Bytes::copy_from_slice(raw.as_bytes());
        let inject = apply_sampling_overrides(&overrides, &probe_of(raw), &metrics()).unwrap();
        assert_eq!(inject.len(), 1, "null must be treated as omitted");

        let out = build_outgoing_body(&body, None, None, None, &inject).unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(
            parsed.get("temperature"),
            Some(&serde_json::json!(1.0)),
            "the engine must read the configured value, not the client's null: {}",
            std::str::from_utf8(&out).unwrap()
        );
    }

    /// The splice must also fire when the ingress ALREADY parsed the body — a
    /// chat-encoder model, the cache-aware policy or bucket routing — as long
    /// as nothing needs overwriting. Gating on `value.is_none()` skipped
    /// exactly those configurations, i.e. most governed fleets.
    #[test]
    fn splice_fires_even_when_a_parse_is_already_on_hand() {
        let body = Bytes::from_static(br#"{ "model" : "x" }"#);
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let inject = apply_sampling_overrides(
            &overrides_of(ConflictPolicy::Reject, r#"{"temperature": 1.0}"#),
            &probe_of(r#"{"model":"x"}"#),
            &metrics(),
        )
        .unwrap();

        let out = build_outgoing_body(&body, Some(value), None, None, &inject).unwrap();
        // Byte-identical to the no-parse case: the parse was dropped unused.
        assert_eq!(
            std::str::from_utf8(&out).unwrap(),
            r#"{ "model" : "x" ,"temperature":1.0}"#
        );
    }
}
