//! `POST /v1/score`: Python's `OpenAIServingScore` on a generation model.
//! Each item is scored by the model's log-probabilities of the candidate
//! label tokens at the position after the prompt: one generate per item with
//! `max_new_tokens: 0` and the labels as `token_ids_logprob`, as Python's
//! `TokenizerManager.score_request` submits it. Pointwise only: the
//! multi-item (`--enable-mis`), setwise (`score_extraction_token`) and
//! embed-override variants are refused, and the pooled path of
//! non-generation models never arises (Rust-server mode refuses those at
//! launch).

use std::sync::Arc;

use axum::{
    Json, Router,
    extract::{State, rejection::JsonRejection},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::post,
};
use sglang_api_types::api::v1 as api;

use super::{collect_output, openai_error};
use crate::api_server::core::{CoreError, ReceivedAs};
use crate::api_server::handlers::CoreHandle;
use crate::api_server::http::app::AppState;
use crate::message::config::ServerArgs;
use crate::message::ids::Rid;
use crate::message::request::GenerateRequest;
use crate::message::sampling::SamplingParams;

pub(super) fn routes() -> Router<Arc<AppState>> {
    Router::new().route("/v1/score", post(score))
}

/// The scored items: one row of label scores per item, Python's
/// `ScoreResult`.
pub(crate) struct Scored {
    pub(crate) scores: Vec<Vec<f64>>,
    pub(crate) token_logprobs: Option<Vec<Vec<f64>>>,
    pub(crate) prompt_tokens: u64,
}

impl Scored {
    /// Python's `ScoringResponse`.
    pub(crate) fn into_response(self, model: String) -> api::OpenAiScoreResponse {
        let rows = |rows: Vec<Vec<f64>>| {
            rows.into_iter()
                .map(|items| api::DoubleList { items })
                .collect::<Vec<_>>()
        };
        api::OpenAiScoreResponse {
            scores: Some(api::OpenAiScores {
                value: Some(api::open_ai_scores::Value::Pointwise(
                    api::OpenAiScoreMatrix {
                        rows: rows(self.scores),
                    },
                )),
            }),
            pooled_hidden_states: None,
            token_logprobs: self.token_logprobs.map(rows).unwrap_or_default(),
            model,
            usage: Some(api::OpenAiUsage {
                prompt_tokens: self.prompt_tokens as i64,
                total_tokens: self.prompt_tokens as i64,
                completion_tokens: Some(0),
                prompt_tokens_details: None,
                reasoning_tokens: Some(0),
                completion_tokens_details: None,
            }),
            object: "scoring".into(),
        }
    }
}

/// One item's prompt: the query and the item, concatenated as Python
/// concatenates them (no separator), item first when asked.
enum Prompt {
    Text(String),
    Ids(Vec<i64>),
}

fn bad(message: impl Into<String>) -> (StatusCode, String) {
    (StatusCode::BAD_REQUEST, message.into())
}

/// Python's `expand_base_rid`: `<rid>_<i>`, zero-padded so the scheduler's
/// abort-by-prefix cannot match a sibling.
fn expand_base_rid(rid: &str, num: usize) -> Result<Vec<String>, (StatusCode, String)> {
    if rid.is_empty() {
        return Err(bad("The rid should be a non-empty string."));
    }
    if rid.starts_with(crate::message::ids::HEALTH_CHECK_RID_PREFIX) {
        return Err(bad(format!(
            "The rid must not start with the reserved prefix {:?}.",
            crate::message::ids::HEALTH_CHECK_RID_PREFIX
        )));
    }
    let width = if num > 1 {
        (num - 1).to_string().len()
    } else {
        1
    };
    Ok((0..num).map(|i| format!("{rid}_{i:0width$}")).collect())
}

/// Python's `score_request` validation and prompt building for the
/// pointwise generation path.
/// Each item's prompt and its candidate labels.
type PromptsAndLabels = (Vec<Prompt>, Vec<Vec<i64>>);

fn prompts_and_labels(
    request: &api::OpenAiScoreRequest,
    vocab_size: u64,
) -> Result<Option<PromptsAndLabels>, (StatusCode, String)> {
    use api::open_ai_score_items::Value as Items;
    use api::open_ai_score_query::Value as Query;
    if request.return_pooled_hidden_states {
        return Err(bad(
            "return_pooled_hidden_states is not supported for CausalLM models. It requires a \
             model with a task-specific head (e.g. SequenceClassification or RewardModel).",
        ));
    }
    if request.embed_override_token_id.is_some()
        || !request.query_embed_overrides.is_empty()
        || !request.item_embed_overrides.is_empty()
    {
        return Err(bad("embed overrides are not served by the Rust frontend"));
    }
    if request.score_extraction_token.is_some() {
        return Err(bad(
            "score_extraction_token is not served by the Rust frontend",
        ));
    }
    let Some(label_token_ids) = request
        .label_token_ids
        .as_ref()
        .and_then(|v| v.value.as_ref())
    else {
        return Err(bad(
            "label_token_ids is required for generation (CausalLM) models.",
        ));
    };
    let Some(items) = request.items.as_ref().and_then(|v| v.value.as_ref()) else {
        return Err(bad("items must be provided"));
    };
    let query = request.query.as_ref().and_then(|v| v.value.as_ref());
    let prompts: Vec<Prompt> = match (query, items) {
        (Some(Query::Text(query)), Items::Text(item)) => {
            vec![concat_text(query, item, request.item_first)]
        }
        (Some(Query::Text(query)), Items::Texts(items)) => items
            .items
            .iter()
            .map(|item| concat_text(query, item, request.item_first))
            .collect(),
        (Some(Query::Ids(query)), Items::IdsList(items)) => items
            .items
            .iter()
            .map(|item| {
                let (first, second) = if request.item_first {
                    (&item.ids, &query.ids)
                } else {
                    (&query.ids, &item.ids)
                };
                Prompt::Ids(first.iter().chain(second.iter()).copied().collect())
            })
            .collect(),
        (_, Items::Texts(items)) if items.items.is_empty() => Vec::new(),
        (_, Items::IdsList(items)) if items.items.is_empty() => Vec::new(),
        _ => {
            return Err(bad(
                "Invalid combination of query/items types for score_request.",
            ));
        }
    };
    if prompts.is_empty() {
        return Ok(None);
    }
    let labels: Vec<Vec<i64>> = match label_token_ids {
        api::token_ids_or_list::Value::One(ids) => vec![ids.ids.clone(); prompts.len()],
        api::token_ids_or_list::Value::Many(lists) => {
            if lists.items.len() != prompts.len() {
                return Err(bad("label_token_ids must have one list per item"));
            }
            lists.items.iter().map(|ids| ids.ids.clone()).collect()
        }
    };
    for labels in &labels {
        if labels.is_empty() {
            return Err(bad("each item must have a nonempty candidate list"));
        }
        if labels.iter().any(|token| *token < 0) {
            return Err(bad("label_token_ids must contain nonnegative integers"));
        }
        let distinct: std::collections::HashSet<i64> = labels.iter().copied().collect();
        if distinct.len() != labels.len() {
            return Err(bad("label_token_ids must not contain duplicate tokens"));
        }
        if let Some(token) = labels.iter().find(|token| **token as u64 >= vocab_size) {
            return Err(bad(format!(
                "Token ID {token} is out of vocabulary (vocab size: {vocab_size})"
            )));
        }
    }
    Ok(Some((prompts, labels)))
}

fn concat_text(query: &str, item: &str, item_first: bool) -> Prompt {
    Prompt::Text(if item_first {
        format!("{item}{query}")
    } else {
        format!("{query}{item}")
    })
}

/// Python's `_convert_logprobs_to_scores`: probabilities, or a
/// max-centered softmax over the labels at `temperature`.
fn scores_from_logprobs(logprobs: &[f64], apply_softmax: bool, temperature: f64) -> Vec<f64> {
    if apply_softmax {
        let maximum = logprobs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let weights: Vec<f64> = logprobs
            .iter()
            .map(|score| ((score - maximum) / temperature).exp())
            .collect();
        let denominator: f64 = weights.iter().sum();
        weights.into_iter().map(|w| w / denominator).collect()
    } else {
        logprobs
            .iter()
            .map(|x| if x.is_finite() { x.exp() } else { 0.0 })
            .collect()
    }
}

/// Score `request`'s items on `core`, as Python's `score_request` does on a
/// generation model. `Ok(None)` is an empty `items`.
pub(crate) async fn score_request(
    core: &CoreHandle,
    server_args: &ServerArgs,
    received: impl Fn() -> ReceivedAs,
    request: &api::OpenAiScoreRequest,
) -> Result<Scored, (StatusCode, String)> {
    let Some((prompts, labels)) = prompts_and_labels(request, server_args.model_config.vocab_size)?
    else {
        return Ok(Scored {
            scores: Vec::new(),
            token_logprobs: request.return_token_logprobs.then(Vec::new),
            prompt_tokens: 0,
        });
    };
    let temperature = request.temperature.unwrap_or(1.0);
    if !temperature.is_finite() || temperature <= 0.0 {
        return Err(bad("temperature must be a positive finite number"));
    }
    let rids = match request.rid.as_deref() {
        Some(rid) => expand_base_rid(rid, prompts.len())?
            .into_iter()
            .map(|rid| Rid::from_client(&rid))
            .collect(),
        None => (0..prompts.len()).map(|_| Rid::new()).collect::<Vec<_>>(),
    };
    let mut sampling_params = SamplingParams {
        max_new_tokens: Some(0),
        ..Default::default()
    };
    sampling_params
        .normalize(
            server_args.skip_tokenizer_init,
            server_args.model_config.vocab_size,
        )
        .map_err(|error| bad(error.to_string()))?;
    let mut calls = Vec::with_capacity(prompts.len());
    for ((prompt, labels), rid) in prompts.into_iter().zip(&labels).zip(rids) {
        let (text, input_ids) = match prompt {
            Prompt::Text(text) => (Some(text), None),
            Prompt::Ids(ids) => (None, Some(ids)),
        };
        let native = GenerateRequest {
            rid,
            text,
            input_ids,
            sampling_params: sampling_params.clone(),
            stream: false,
            return_logprob: true,
            logprob_start_len: -1,
            token_ids_logprob: Some(labels.clone()),
            ..Default::default()
        };
        let call = core
            .generate(native, received())
            .await
            .map_err(|error| match error {
                CoreError::Unavailable => (
                    StatusCode::SERVICE_UNAVAILABLE,
                    "service unavailable".into(),
                ),
                CoreError::InvalidArgument(message) => bad(message),
                error => (error.http_status(), error.to_string()),
            })?;
        calls.push(call);
    }
    let outputs = futures::future::join_all(calls.into_iter().map(collect_output)).await;
    let mut scores = Vec::with_capacity(labels.len());
    let mut token_logprobs = request.return_token_logprobs.then(Vec::new);
    let mut prompt_tokens = 0u64;
    for (output, labels) in outputs.into_iter().zip(&labels) {
        let output = output?;
        prompt_tokens += u64::from(output.prompt_tokens);
        let meta = output.meta_info("", None);
        let Some(row) = meta
            .output_token_ids_logprobs
            .first()
            .and_then(|row| row.row.as_ref())
        else {
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                "Internal server error: output_logprobs is empty for request.".into(),
            ));
        };
        let logprobs: Vec<f64> = labels
            .iter()
            .map(|label| {
                row.entries
                    .iter()
                    .find(|entry| entry.token_id == *label)
                    .and_then(|entry| entry.logprob)
                    .unwrap_or(f64::NEG_INFINITY)
            })
            .collect();
        scores.push(scores_from_logprobs(
            &logprobs,
            request.apply_softmax,
            temperature,
        ));
        if let Some(token_logprobs) = token_logprobs.as_mut() {
            token_logprobs.push(logprobs);
        }
    }
    Ok(Scored {
        scores,
        token_logprobs,
        prompt_tokens,
    })
}

/// `POST /v1/score` — Python's `ScoringResponse`, or its 400 for a request
/// the scorer refuses.
async fn score(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Result<Json<api::OpenAiScoreRequest>, JsonRejection>,
) -> Response {
    let request = match body {
        Ok(Json(body)) => body,
        Err(rejection) => {
            return openai_error(StatusCode::BAD_REQUEST, rejection.body_text(), false);
        }
    };
    let received = || ReceivedAs::Typed(state.core.logged_headers(&headers));
    match score_request(&state.core, &state.server_args, received, &request).await {
        Ok(scored) => {
            Json(scored.into_response(request.model.clone().unwrap_or_else(|| "default".into())))
                .into_response()
        }
        Err((status, message)) => openai_error(status, message, false),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::request::RequestKind;
    use crate::message::response::{ChunkEvent, ChunkExtras, ResponseItem};
    use crate::tokenizer_manager::wiring::TmEvent;
    use axum::body::{Body, to_bytes};
    use axum::http::{Request as HttpRequest, header::CONTENT_TYPE};
    use tower::ServiceExt;

    /// Python's zero-padded sub-rids.
    #[test]
    fn sub_rids_are_zero_padded() {
        assert_eq!(expand_base_rid("r", 1).unwrap(), ["r_0"]);
        assert_eq!(expand_base_rid("r", 11).unwrap()[10], "r_10");
        assert_eq!(expand_base_rid("r", 11).unwrap()[1], "r_01");
        assert!(expand_base_rid("", 2).is_err());
    }

    /// Scores are probabilities, or a temperature softmax over the labels;
    /// a label the model did not report scores 0 (probability) or takes no
    /// mass (softmax).
    #[test]
    fn scores_follow_python() {
        let probs = scores_from_logprobs(
            &[0.0_f64.ln(), (0.25_f64).ln(), f64::NEG_INFINITY],
            false,
            1.0,
        );
        assert_eq!(probs, [0.0, 0.25, 0.0]);
        let soft = scores_from_logprobs(&[0.0, 0.0, f64::NEG_INFINITY], true, 1.0);
        assert!((soft[0] - 0.5).abs() < 1e-9 && soft[2] == 0.0);
        let sharp = scores_from_logprobs(&[0.0, -1.0], true, 0.1);
        assert!(sharp[0] > 0.9999);
    }

    /// `/v1/score` concatenates query and item (item first when asked),
    /// submits one zero-token generate per item carrying the labels, reads
    /// the label logprobs back, and answers Python's shape; refusals and an
    /// empty `items` never reach the scheduler.
    #[tokio::test]
    async fn score_route_scores_each_item_by_label_logprobs() {
        use crate::api_server::handlers::{CoreConfig, CoreHandle, CoreMetadata};
        let server_args = super::super::test_utils::server_args();
        let (intake_tx, intake_rx) = flume::unbounded();
        let (abort_tx, _abort_rx) = flume::unbounded();
        let core = CoreHandle::new(
            intake_tx,
            abort_tx,
            CoreConfig {
                response_capacity: 8,
                response_activity: Default::default(),
                startup_ready: true,
                is_disaggregation: false,
                mm_limits: Default::default(),
                multimodal: false,
                metrics: None,
                request_log: None,
                generate_validator: None,
                peer_endpoints: Vec::new(),
                metadata: CoreMetadata::from(server_args.as_ref()),
            },
        );
        let state = super::super::test_utils::app_state(core);
        let app = routes().with_state(state.clone());
        let response = tokio::spawn(
            app.clone().oneshot(
                HttpRequest::builder()
                    .method("POST")
                    .uri("/v1/score")
                    .header(CONTENT_TYPE, "application/json")
                    .body(Body::from(
                        r#"{"query": "Q", "items": ["a", "b"], "label_token_ids": [5, 7],
                            "apply_softmax": true, "return_token_logprobs": true,
                            "item_first": true, "rid": "r"}"#,
                    ))
                    .unwrap(),
            ),
        );
        for expected_text in ["aQ", "bQ"] {
            let TmEvent::Intake { request, admission } = intake_rx.recv_async().await.unwrap()
            else {
                panic!("expected a generation");
            };
            assert!(admission.try_accept());
            let RequestKind::Generate(generate) = &request.kind else {
                panic!("expected a generate request");
            };
            assert_eq!(generate.text.as_deref(), Some(expected_text));
            assert_eq!(generate.sampling_params.max_new_tokens, Some(0));
            assert!(generate.return_logprob);
            assert_eq!(
                generate.token_ids_logprob.as_deref(),
                Some([5, 7].as_slice())
            );
            assert!(generate.rid.client_facing().starts_with("r_"));
            let rid = request.rid.clone();
            request
                .sink
                .try_send(ResponseItem::Done(ChunkEvent {
                    rid,
                    prompt_tokens: 3,
                    extras: Some(Box::new(ChunkExtras {
                        out_tid_val: vec![-1.0, -2.0],
                        out_tid_idx: vec![7, 5],
                        out_tid_lens: vec![2],
                        ..Default::default()
                    })),
                    ..Default::default()
                }))
                .unwrap();
        }
        let response = response.await.unwrap().unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = to_bytes(response.into_body(), 1 << 16).await.unwrap();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["object"], "scoring");
        assert_eq!(body["model"], "default");
        assert_eq!(body["usage"]["prompt_tokens"], 6);
        let scores = body["scores"].as_array().unwrap();
        assert_eq!(scores.len(), 2);
        let row = scores[0].as_array().unwrap();
        // label 5 -> -2.0, label 7 -> -1.0: softmax favours 7.
        assert!(row[1].as_f64().unwrap() > row[0].as_f64().unwrap());
        assert!((row[0].as_f64().unwrap() + row[1].as_f64().unwrap() - 1.0).abs() < 1e-9);
        assert_eq!(body["token_logprobs"][0], serde_json::json!([-2.0, -1.0]));

        for (body, message) in [
            (
                r#"{"query": "Q", "items": ["a"]}"#,
                "label_token_ids is required",
            ),
            (
                r#"{"query": "Q", "items": ["a"], "label_token_ids": [[1], [2]]}"#,
                "one list per item",
            ),
            (
                r#"{"query": "Q", "items": ["a"], "label_token_ids": [1, 1]}"#,
                "duplicate",
            ),
            (
                r#"{"query": [1], "items": ["a"], "label_token_ids": [1]}"#,
                "Invalid combination",
            ),
            (
                r#"{"query": "Q", "items": ["a"], "label_token_ids": [1], "return_pooled_hidden_states": true}"#,
                "CausalLM",
            ),
        ] {
            let response = app
                .clone()
                .oneshot(
                    HttpRequest::builder()
                        .method("POST")
                        .uri("/v1/score")
                        .header(CONTENT_TYPE, "application/json")
                        .body(Body::from(body))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{body}");
            let bytes = to_bytes(response.into_body(), 1 << 16).await.unwrap();
            let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            assert!(
                json["error"]["message"].as_str().unwrap().contains(message),
                "{body}: {json}"
            );
            assert!(
                intake_rx.try_recv().is_err(),
                "{body}: refused before the scheduler"
            );
        }
        let response = app
            .oneshot(
                HttpRequest::builder()
                    .method("POST")
                    .uri("/v1/score")
                    .header(CONTENT_TYPE, "application/json")
                    .body(Body::from(
                        r#"{"query": "Q", "items": [], "label_token_ids": [1]}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = to_bytes(response.into_body(), 1 << 16).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["scores"], serde_json::json!([]));
    }
}
