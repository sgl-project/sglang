//! `/v1/score` on a generation model, as Python's `OpenAIServingScore` and
//! `TokenizerManager.score_request` serve it: one zero-token generate per item
//! with the candidate labels as `token_ids_logprob`, scored by the labels'
//! log-probabilities after the prompt. Pointwise only.

use sglang_api_types::api::v1::{
    self as api, GenerateRequest, GenerateResponse, SamplingParams, SamplingParamsOrList, TokenIds,
    TokenIdsList, TokenIdsOrList, sampling_params_or_list, token_ids_or_list,
};

use super::request::OneOrList;
use super::wire::Reply;

/// `HEALTH_CHECK_RID_PREFIX` in python/sglang/srt/constants.py.
const HEALTH_CHECK_RID_PREFIX: &str = "HEALTH_CHECK";

/// One item's prompt: the query and the item with nothing between, item
/// first when asked.
enum Prompt {
    Text(String),
    Ids(Vec<i64>),
}

/// Python's `expand_base_rid`: `<rid>_<i>`, zero-padded so the scheduler's
/// abort-by-prefix cannot match a sibling.
fn expand_base_rid(rid: &str, num: usize) -> Result<Vec<String>, String> {
    if rid.is_empty() {
        return Err("The rid should be a non-empty string.".into());
    }
    if rid.starts_with(HEALTH_CHECK_RID_PREFIX) {
        return Err(format!(
            "The rid must not start with the reserved prefix {HEALTH_CHECK_RID_PREFIX:?}."
        ));
    }
    let width = match num {
        0 | 1 => 1,
        num => (num - 1).to_string().len(),
    };
    Ok((0..num).map(|i| format!("{rid}_{i:0width$}")).collect())
}

fn concat_text(query: &str, item: &str, item_first: bool) -> Prompt {
    Prompt::Text(match item_first {
        true => format!("{item}{query}"),
        false => format!("{query}{item}"),
    })
}

/// Each item's prompt and its candidate labels.
type PromptsAndLabels = (Vec<Prompt>, Vec<Vec<i64>>);

/// `score_request`'s validation and prompt building; `None` for no items.
fn prompts_and_labels(
    request: &api::OpenAiScoreRequest,
    vocab_size: u64,
) -> Result<Option<PromptsAndLabels>, String> {
    use api::open_ai_score_items::Value as Items;
    use api::open_ai_score_query::Value as Query;
    if request.return_pooled_hidden_states {
        return Err(
            "return_pooled_hidden_states is not supported for CausalLM models. It requires a \
             model with a task-specific head (e.g. SequenceClassification or RewardModel)."
                .into(),
        );
    }
    if request.embed_override_token_id.is_some()
        || !request.query_embed_overrides.is_empty()
        || !request.item_embed_overrides.is_empty()
    {
        return Err("embed overrides are not served by the Rust frontend".into());
    }
    if request.score_extraction_token.is_some() {
        return Err("score_extraction_token is not served by the Rust frontend".into());
    }
    let label_token_ids = request.label_token_ids.as_ref();
    let Some(label_token_ids) = label_token_ids.and_then(|v| v.value.as_ref()) else {
        return Err("label_token_ids is required for generation (CausalLM) models.".into());
    };
    let Some(items) = request.items.as_ref().and_then(|v| v.value.as_ref()) else {
        return Err("items must be provided".into());
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
                let (first, second) = match request.item_first {
                    true => (&item.ids, &query.ids),
                    false => (&query.ids, &item.ids),
                };
                Prompt::Ids(first.iter().chain(second.iter()).copied().collect())
            })
            .collect(),
        (_, Items::Texts(items)) if items.items.is_empty() => Vec::new(),
        (_, Items::IdsList(items)) if items.items.is_empty() => Vec::new(),
        _ => return Err("Invalid combination of query/items types for score_request.".into()),
    };
    if prompts.is_empty() {
        return Ok(None);
    }
    let labels: Vec<Vec<i64>> = match label_token_ids {
        token_ids_or_list::Value::One(ids) => vec![ids.ids.clone(); prompts.len()],
        token_ids_or_list::Value::Many(lists) => {
            if lists.items.len() != prompts.len() {
                return Err("label_token_ids must have one list per item".into());
            }
            lists.items.iter().map(|ids| ids.ids.clone()).collect()
        }
    };
    for labels in &labels {
        if labels.is_empty() {
            return Err("each item must have a nonempty candidate list".into());
        }
        if labels.iter().any(|token| *token < 0) {
            return Err("label_token_ids must contain nonnegative integers".into());
        }
        let distinct: std::collections::HashSet<i64> = labels.iter().copied().collect();
        if distinct.len() != labels.len() {
            return Err("label_token_ids must not contain duplicate tokens".into());
        }
        if let Some(token) = labels.iter().find(|token| **token as u64 >= vocab_size) {
            return Err(format!(
                "Token ID {token} is out of vocabulary (vocab size: {vocab_size})"
            ));
        }
    }
    Ok(Some((prompts, labels)))
}

fn token_id_lists(lists: impl IntoIterator<Item = Vec<i64>>) -> TokenIdsOrList {
    let items = lists.into_iter().map(|ids| TokenIds { ids }).collect();
    TokenIdsOrList {
        value: Some(token_ids_or_list::Value::Many(TokenIdsList { items })),
    }
}

/// Lower a score request into a batch `/generate` body, one item per prompt;
/// no body when there are no items. `Err` is the message of Python's 400.
pub fn lower_score(
    request: &api::OpenAiScoreRequest,
    vocab_size: u64,
) -> Result<(Option<GenerateRequest>, ScoreResponder), String> {
    let prompts = prompts_and_labels(request, vocab_size)?;
    let temperature = request.temperature.unwrap_or(1.0);
    if prompts.is_some() && (!temperature.is_finite() || temperature <= 0.0) {
        return Err("temperature must be a positive finite number".into());
    }
    let (prompts, labels) = prompts.unwrap_or_default();
    let responder = ScoreResponder {
        model: request.model.clone().unwrap_or_else(|| "default".into()),
        labels: labels.clone(),
        apply_softmax: request.apply_softmax,
        temperature,
        return_token_logprobs: request.return_token_logprobs,
    };
    if prompts.is_empty() {
        return Ok((None, responder));
    }
    let rid = match request.rid.as_deref() {
        Some(rid) => Some(OneOrList::List(expand_base_rid(rid, prompts.len())?).into()),
        None => None,
    };
    let (mut texts, mut ids) = (Vec::new(), Vec::new());
    for prompt in prompts {
        match prompt {
            Prompt::Text(text) => texts.push(text),
            Prompt::Ids(prompt) => ids.push(prompt),
        }
    }
    let sampling = SamplingParams {
        max_new_tokens: Some(0),
        ..Default::default()
    };
    let generate = GenerateRequest {
        rid,
        text: (!texts.is_empty()).then(|| OneOrList::List(texts).into()),
        input_ids: (!ids.is_empty()).then(|| token_id_lists(ids)),
        sampling_params: Some(SamplingParamsOrList {
            value: Some(sampling_params_or_list::Value::One(sampling)),
        }),
        return_logprob: Some(OneOrList::One(true).into()),
        logprob_start_len: Some(OneOrList::One(-1).into()),
        token_ids_logprob: Some(token_id_lists(labels)),
        stream: Some(false),
        ..Default::default()
    };
    Ok((Some(generate), responder))
}

/// Python's `_convert_logprobs_to_scores`: probabilities, or a max-centered
/// softmax over the labels at `temperature`.
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

/// Builds the scoring response from the engine's `/generate` output.
pub struct ScoreResponder {
    model: String,
    /// Each item's candidate labels, in request order.
    labels: Vec<Vec<i64>>,
    apply_softmax: bool,
    temperature: f64,
    return_token_logprobs: bool,
}

impl ScoreResponder {
    /// Python's `ScoringResponse` over the batch's outputs, one per item.
    pub fn unary(self, output: Vec<GenerateResponse>) -> Reply {
        let mut scores = Vec::with_capacity(self.labels.len());
        let mut token_logprobs = Vec::new();
        let mut prompt_tokens = 0i64;
        for (item, labels) in output.iter().zip(&self.labels) {
            let meta = item.meta_info.as_ref();
            prompt_tokens += meta.map_or(0, |meta| i64::from(meta.prompt_tokens));
            let row = meta
                .and_then(|meta| meta.output_token_ids_logprobs.first())
                .and_then(|row| row.row.as_ref());
            let Some(row) = row else {
                let message = "Internal server error: output_logprobs is empty for request.";
                return Reply::error(500, message);
            };
            let logprob = |label: &i64| {
                let entry = row.entries.iter().find(|entry| entry.token_id == *label);
                entry
                    .and_then(|entry| entry.logprob)
                    .unwrap_or(f64::NEG_INFINITY)
            };
            let logprobs: Vec<f64> = labels.iter().map(logprob).collect();
            scores.push(api::DoubleList {
                items: scores_from_logprobs(&logprobs, self.apply_softmax, self.temperature),
            });
            if self.return_token_logprobs {
                token_logprobs.push(api::DoubleList { items: logprobs });
            }
        }
        let response = api::OpenAiScoreResponse {
            scores: Some(api::OpenAiScores {
                value: Some(api::open_ai_scores::Value::Pointwise(
                    api::OpenAiScoreMatrix { rows: scores },
                )),
            }),
            pooled_hidden_states: None,
            token_logprobs,
            model: self.model,
            usage: Some(api::OpenAiUsage {
                prompt_tokens,
                total_tokens: prompt_tokens,
                completion_tokens: Some(0),
                prompt_tokens_details: None,
                reasoning_tokens: Some(0),
                completion_tokens_details: None,
            }),
            object: "scoring".into(),
        };
        Reply {
            status: 200,
            body: serde_json::to_string(&response).expect("api.v1 types serialize to JSON"),
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::*;

    fn request(body: Value) -> api::OpenAiScoreRequest {
        serde_json::from_value(body).unwrap()
    }

    /// Python's zero-padded sub-rids.
    #[test]
    fn sub_rids_are_zero_padded() {
        assert_eq!(expand_base_rid("r", 1).unwrap(), ["r_0"]);
        assert_eq!(expand_base_rid("r", 11).unwrap()[10], "r_10");
        assert_eq!(expand_base_rid("r", 11).unwrap()[1], "r_01");
        assert!(expand_base_rid("", 2).is_err());
    }

    /// Scores are probabilities, or a temperature softmax over the labels;
    /// a label the model did not report scores 0 or takes no mass.
    #[test]
    fn scores_follow_python() {
        let logprobs = [0.0_f64.ln(), (0.25_f64).ln(), f64::NEG_INFINITY];
        assert_eq!(
            scores_from_logprobs(&logprobs, false, 1.0),
            [0.0, 0.25, 0.0]
        );
        let soft = scores_from_logprobs(&[0.0, 0.0, f64::NEG_INFINITY], true, 1.0);
        assert!((soft[0] - 0.5).abs() < 1e-9 && soft[2] == 0.0);
        let sharp = scores_from_logprobs(&[0.0, -1.0], true, 0.1);
        assert!(sharp[0] > 0.9999);
    }

    /// One zero-token generate per item, query and item joined (item first
    /// when asked), carrying the labels and numbered rids.
    #[test]
    fn lowers_each_item_to_a_zero_token_generate() {
        let (generate, _) = lower_score(
            &request(
                json!({"query": "Q", "items": ["a", "b"], "label_token_ids": [5, 7],
                "item_first": true, "rid": "r"}),
            ),
            100,
        )
        .unwrap();
        let generate = serde_json::to_value(generate.unwrap()).unwrap();
        assert_eq!(generate["text"], json!(["aQ", "bQ"]));
        assert_eq!(generate["rid"], json!(["r_0", "r_1"]));
        assert_eq!(generate["token_ids_logprob"], json!([[5, 7], [5, 7]]));
        assert_eq!(generate["sampling_params"]["max_new_tokens"], 0);
        assert_eq!(generate["return_logprob"], true);

        let (generate, responder) = lower_score(
            &request(json!({"query": "Q", "items": [], "label_token_ids": [1]})),
            100,
        )
        .unwrap();
        assert!(generate.is_none());
        let empty: Value = serde_json::from_str(&responder.unary(Vec::new()).body).unwrap();
        assert_eq!(empty["scores"], json!([]));
    }

    #[test]
    fn refuses_what_python_refuses() {
        for (body, message) in [
            (
                json!({"query": "Q", "items": ["a"]}),
                "label_token_ids is required",
            ),
            (
                json!({"query": "Q", "items": ["a"], "label_token_ids": [[1], [2]]}),
                "one list per item",
            ),
            (
                json!({"query": "Q", "items": ["a"], "label_token_ids": [1, 1]}),
                "duplicate",
            ),
            (
                json!({"query": [1], "items": ["a"], "label_token_ids": [1]}),
                "Invalid combination",
            ),
            (
                json!({"query": "Q", "items": ["a"], "label_token_ids": [1],
                    "return_pooled_hidden_states": true}),
                "CausalLM",
            ),
            (
                json!({"query": "Q", "items": ["a"], "label_token_ids": [100]}),
                "out of vocabulary",
            ),
        ] {
            let error = lower_score(&request(body.clone()), 100)
                .err()
                .expect("refused");
            assert!(error.contains(message), "{body}: {error}");
        }
    }

    /// The label logprobs are read back by token id, in label order.
    #[test]
    fn scores_each_item_by_its_label_logprobs() {
        let (_, responder) = lower_score(
            &request(
                json!({"query": "Q", "items": ["a"], "label_token_ids": [5, 7],
                "apply_softmax": true, "return_token_logprobs": true}),
            ),
            100,
        )
        .unwrap();
        let output: GenerateResponse = serde_json::from_value(json!({
            "text": "",
            "meta_info": {
                "prompt_tokens": 3,
                "output_token_ids_logprobs": [[[-1.0, 7, null], [-2.0, 5, null]]],
            },
        }))
        .unwrap();
        let reply = responder.unary(vec![output]);
        assert_eq!(reply.status, 200);
        let body: Value = serde_json::from_str(&reply.body).unwrap();
        assert_eq!(body["object"], "scoring");
        assert_eq!(body["model"], "default");
        assert_eq!(body["usage"]["prompt_tokens"], 3);
        let row = body["scores"][0].as_array().unwrap();
        assert!(row[1].as_f64().unwrap() > row[0].as_f64().unwrap());
        assert_eq!(body["token_logprobs"][0], json!([-2.0, -1.0]));
    }
}
