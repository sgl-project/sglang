//! Projection of native output onto Python's OpenAI extension schema.

use serde_json::{Map, Value, json};

use super::extensions::OutputOptions;
use crate::message::response::ChunkEvent;
use crate::message::types::HiddenStatesMode;

pub(super) struct StreamEvent {
    pub data: String,
    pub event: Option<&'static str>,
}

impl From<String> for StreamEvent {
    fn from(data: String) -> Self {
        Self { data, event: None }
    }
}

/// Retain only response annotations and requested IDs. Stream text and logprobs
/// have already been emitted and must not accumulate a second time here.
#[derive(Default)]
pub(super) struct StreamChoice {
    metadata: Map<String, Value>,
    prompt_ids: Option<std::sync::Arc<[i32]>>,
    output_ids: Vec<i32>,
    completion_tokens: u64,
    visible_tokens: usize,
}

impl StreamChoice {
    pub(super) fn observe(&mut self, output: &ChunkEvent, retain_ids: bool) {
        self.completion_tokens = self
            .completion_tokens
            .saturating_add(output.completion_tokens);
        self.visible_tokens += output.token_ids.len();
        if retain_ids {
            self.output_ids.extend_from_slice(&output.token_ids);
        }
        let counts = serde_json::to_value(output.counts.metadata(self.visible_tokens))
            .expect("token counts must serialize");
        self.metadata.extend(
            counts
                .as_object()
                .expect("response field must be an object")
                .clone(),
        );
        self.metadata
            .insert("prompt_tokens".into(), output.prompt_tokens.into());
        self.metadata
            .insert("completion_tokens".into(), self.completion_tokens.into());
        self.metadata
            .insert("id".into(), output.rid.client_facing().into());
        self.metadata
            .insert("finish_reason".into(), json!(output.finish_reason));
        if let Some(extras) = output.extras.as_deref() {
            self.metadata.extend(extras.metadata.fields.clone());
            // Streaming reports the accumulated count rather than this delta.
            self.metadata
                .insert("completion_tokens".into(), self.completion_tokens.into());
            if let Some(ids) = &extras.prompt_token_ids {
                self.prompt_ids = Some(ids.clone());
            }
            if extras.hidden_shape.is_some() || !extras.hidden_lens.is_empty() {
                self.metadata.insert(
                    "hidden_states".into(),
                    super::super::frame::hidden_states_value(extras),
                );
            }
        }
    }

    pub(super) fn metadata_item(&self) -> Value {
        json!({"meta_info":self.metadata})
    }

    pub(super) fn into_item(self) -> Value {
        let mut item = json!({"meta_info":self.metadata,"output_ids":self.output_ids});
        if let Some(ids) = self.prompt_ids {
            item["prompt_token_ids"] = json!(ids.as_ref());
        }
        item
    }
}

pub(super) struct StreamState {
    pub choices: Vec<StreamChoice>,
    pub last_id: String,
}

impl StreamState {
    pub(super) fn new(count: usize, id: String) -> Self {
        Self {
            choices: (0..count).map(|_| StreamChoice::default()).collect(),
            last_id: id,
        }
    }

    pub(super) fn observe(&mut self, index: usize, output: &ChunkEvent, retain_ids: bool) {
        self.last_id = output.rid.client_facing().into();
        self.choices[index].observe(output, retain_ids);
    }
}

pub(super) fn stream_frame(
    id: &str,
    model: &str,
    created: u32,
    chat: bool,
    choices: Value,
) -> Value {
    json!({"id":id,"model":model,"created":created,"choices":choices,
           "object":if chat {"chat.completion.chunk"} else {"text_completion"},"usage":null})
}

impl OutputOptions {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn stream_tail(
        &self,
        items: &[Value],
        n: usize,
        chat: bool,
        id: &str,
        model: &str,
        created: u32,
        include_usage: bool,
    ) -> Vec<StreamEvent> {
        let mut events = Vec::new();
        for (index, item) in items.iter().enumerate() {
            if item["meta_info"]["hidden_states"]
                .as_array()
                .is_some_and(|v| !v.is_empty())
                && let Some(states) = self.hidden_states(item)
            {
                let mut choice =
                    json!({"index":index,"finish_reason":null,"logprobs":null,"matched_stop":null});
                if chat {
                    choice["delta"] = json!({"role":null,"content":null,"reasoning_content":null,
                        "tool_calls":null,"hidden_states":states});
                } else {
                    choice["text"] = "".into();
                    choice["hidden_states"] = states;
                }
                events.push(
                    stream_frame(id, model, created, chat, json!([choice]))
                        .to_string()
                        .into(),
                );
            }
        }
        if let Some(mut sglext) = self.sglext(items, n, chat) {
            let mut ids = Map::new();
            if chat && self.ids_framed {
                for key in ["input_ids", "output_ids"] {
                    if let Some(value) = sglext
                        .as_object_mut()
                        .expect("response field must be an object")
                        .remove(key)
                    {
                        ids.insert(key.into(), value);
                    }
                }
            }
            if !sglext
                .as_object()
                .expect("response field must be an object")
                .is_empty()
            {
                let mut frame = stream_frame(id, model, created, chat, json!([]));
                frame["sglext"] = sglext;
                events.push(frame.to_string().into());
            }
            if !ids.is_empty() {
                let mut frame = stream_frame(id, model, created, chat, json!([]));
                frame["sglext"] = Value::Object(ids);
                events.push(StreamEvent {
                    data: frame.to_string(),
                    event: Some("sglext_ids"),
                });
            }
        }
        if include_usage {
            let mut frame = stream_frame(id, model, created, chat, json!([]));
            frame["usage"] = self.usage(items, n, chat);
            if !chat && frame["usage"]["prompt_tokens_details"].is_null() {
                frame["usage"]
                    .as_object_mut()
                    .unwrap()
                    .remove("prompt_tokens_details");
            }
            events.push(frame.to_string().into());
        }
        events
    }

    pub(super) fn hidden_states(&self, item: &Value) -> Option<Value> {
        let states = item["meta_info"]
            .get("hidden_states")
            .filter(|v| !v.is_null())?;
        match self.return_hidden_states {
            HiddenStatesMode::Off => None,
            HiddenStatesMode::Last => Some(states.clone()),
            HiddenStatesMode::Full => Some(
                states
                    .as_array()
                    .filter(|rows| rows.len() > 1)
                    .and_then(|rows| rows.last())
                    .cloned()
                    .unwrap_or_else(|| json!([])),
            ),
        }
    }

    pub(super) fn choice_fields(&self, choice: &mut Value, item: &Value, chat: bool) {
        if chat {
            let message = choice["message"].as_object_mut().unwrap();
            if message.get("refusal").is_some_and(Value::is_null) {
                message.remove("refusal");
            }
        }
        choice
            .as_object_mut()
            .unwrap()
            .entry("logprobs")
            .or_insert(Value::Null);
        if let Some(states) = self.hidden_states(item) {
            choice["hidden_states"] = states;
        }
        if self.return_token_ids {
            choice[if chat {
                "response_token_ids"
            } else {
                "token_ids"
            }] = item.get("output_ids").cloned().unwrap_or_else(|| json!([]));
        }
        if (self.return_token_ids || (chat && self.return_prompt_token_ids))
            && let Some(ids) = item.get("prompt_token_ids")
        {
            choice["prompt_token_ids"] = ids.clone();
        }
        if chat {
            if let Some(message) = choice.get_mut("message").and_then(Value::as_object_mut) {
                message.entry("tool_calls").or_insert(Value::Null);
                message.entry("reasoning_content").or_insert(Value::Null);
            }
            choice["matched_stop"] = item["meta_info"]["finish_reason"]["matched"].clone();
            if item["meta_info"]["finish_reason"]["type"] == "abort" {
                choice["finish_reason"] = "abort".into();
            }
            if self.return_meta_info {
                choice["meta_info"] = item["meta_info"].clone();
            }
        }
    }

    pub(super) fn sglext(&self, items: &[Value], n: usize, chat: bool) -> Option<Value> {
        let first = items.first()?;
        let meta = &first["meta_info"];
        let mut fields = Map::new();
        if self.return_routed_experts
            && !(chat && self.return_meta_info)
            && let Some(routed) = meta.get("routed_experts").filter(|v| !v.is_null())
        {
            fields.insert("routed_experts".into(), routed.clone());
        }
        if self.return_cached_tokens_details
            && let Some(details) = meta.get("cached_tokens_details").and_then(Value::as_object)
        {
            let mut cached = json!({"device": details.get("device").unwrap_or(&json!(0)),
                                    "host": details.get("host").unwrap_or(&json!(0))});
            if let Some(storage) = details.get("storage").filter(|v| !v.is_null()) {
                cached["storage"] = storage.clone();
                if let Some(backend) = details.get("storage_backend").filter(|v| !v.is_null()) {
                    cached["storage_backend"] = backend.clone();
                }
            }
            fields.insert("cached_tokens_details".into(), cached);
        }
        if self.return_spec_tokens_details {
            let details: Vec<_> = items.iter().filter_map(spec_details).collect();
            if !details.is_empty() {
                fields.insert(
                    "spec_tokens_details".into(),
                    if n > 1 {
                        json!(details)
                    } else {
                        details[0].clone()
                    },
                );
            }
        }
        if chat {
            if self.return_input_ids_in_sglext
                && let Some(ids) = first.get("prompt_token_ids")
            {
                fields.insert("input_ids".into(), ids.clone());
            }
            if self.return_output_ids_in_sglext {
                fields.insert(
                    "output_ids".into(),
                    Value::Array(
                        items
                            .iter()
                            .map(|item| {
                                item.get("output_ids").cloned().unwrap_or_else(|| json!([]))
                            })
                            .collect(),
                    ),
                );
            }
        }
        (!fields.is_empty()).then_some(Value::Object(fields))
    }

    pub(super) fn usage(&self, items: &[Value], n: usize, chat: bool) -> Value {
        let sum = |key: &str, stride: usize| -> u64 {
            items
                .iter()
                .step_by(stride.max(1))
                .map(|item| item["meta_info"][key].as_u64().unwrap_or(0))
                .fold(0, u64::saturating_add)
        };
        let prompt = sum("prompt_tokens", n);
        let completion = sum("completion_tokens", 1);
        let mut details = Map::new();
        let cached = if self.enable_cache_report {
            sum("cached_tokens", n)
        } else {
            0
        };
        if cached > 0 {
            details.insert("cached_tokens".into(), cached.into());
        }
        if chat {
            for name in ["image_tokens", "audio_tokens", "video_tokens"] {
                let count = sum(name, n);
                if count > 0 {
                    details.insert(name.into(), count.into());
                }
            }
        }
        let details = if details.is_empty() {
            Value::Null
        } else {
            details.entry("cached_tokens").or_insert(json!(0));
            Value::Object(details)
        };
        json!({"prompt_tokens":prompt, "completion_tokens":completion,
               "total_tokens":prompt.saturating_add(completion),
               "reasoning_tokens":sum("reasoning_tokens",1), "prompt_tokens_details":details})
    }

    pub(super) fn unary_fields(&self, response: &mut Value, items: &[Value], n: usize, chat: bool) {
        for (choice, item) in response["choices"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .zip(items)
        {
            self.choice_fields(choice, item, chat);
        }
        response["usage"] = self.usage(items, n, chat);
        response["metadata"] = items
            .first()
            .and_then(|first| {
                first["meta_info"].get("weight_version").map(|version| {
                    let mut metadata = json!({"weight_version": version});
                    if let Some(versions) = first["meta_info"].get("weight_versions") {
                        metadata["weight_versions"] = versions.clone();
                    }
                    metadata
                })
            })
            .unwrap_or(Value::Null);
        if let Some(sglext) = self.sglext(items, n, chat) {
            response["sglext"] = sglext;
        }
        if chat {
            response
                .as_object_mut()
                .unwrap()
                .remove("system_fingerprint");
            response
                .as_object_mut()
                .expect("response field must be an object")
                .remove("service_tier");
        }
    }
}

fn spec_details(item: &Value) -> Option<Value> {
    let meta = &item["meta_info"];
    let defaults = json!({
        "spec_accept_rate":0.0,"spec_accept_length":0.0,"spec_cap_length":0.0,
        "spec_block_accept_length":0.0,"spec_num_correct_drafts":0,
        "spec_num_proposed_drafts":0,"spec_verify_ct":0,
        "spec_correct_drafts_histogram":[],"spec_cap_lens_histogram":[]
    });
    let mut fields = defaults
        .as_object()
        .expect("response field must be an object")
        .clone();
    let mut found = false;
    for (key, value) in &mut fields {
        if let Some(actual) = meta.get(key) {
            found = true;
            if !actual.is_null() {
                *value = actual.clone();
            }
        }
    }
    found.then_some(Value::Object(fields))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api_server::guard::AbortGuard;
    use crate::api_server::openai::{
        chat::unary_chat,
        completions::{SubmittedChoice, unary_completion},
        test_utils::senders,
    };
    use crate::message::{
        config::ServerArgs,
        ids::Rid,
        response::{ChunkEvent, ChunkExtras, ResponseItem},
    };
    use axum::http::HeaderMap;

    fn fixture_output(item: &Value) -> ChunkEvent {
        ChunkEvent {
            rid: Rid::from_client(item["meta_info"]["id"].as_str().unwrap()),
            text: item["text"].as_str().unwrap().into(),
            token_ids: serde_json::from_value(item["output_ids"].clone()).unwrap(),
            prompt_tokens: item["meta_info"]["prompt_tokens"].as_u64().unwrap() as u32,
            completion_tokens: item["meta_info"]["completion_tokens"].as_u64().unwrap(),
            finish_reason: serde_json::from_value(item["meta_info"]["finish_reason"].clone())
                .unwrap(),
            extras: Some(Box::new(ChunkExtras {
                prompt_token_ids: Some(
                    serde_json::from_value::<Vec<i32>>(item["prompt_token_ids"].clone())
                        .unwrap()
                        .into(),
                ),
                metadata: crate::message::response::OutputMetadata {
                    fields: item["meta_info"].as_object().unwrap().clone(),
                    ..Default::default()
                },
                ..Default::default()
            })),
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn unary_extensions_match_python_response_formatters() {
        let cases: Value = serde_json::from_str(include_str!(
            "../../../testdata/openai_responses_python.json"
        ))
        .unwrap();
        for case in cases.as_array().unwrap() {
            let chat = case["endpoint"] == "chat";
            let args = ServerArgs {
                enable_cache_report: case["args"]["enable_cache_report"].as_bool().unwrap(),
                return_input_ids: case["args"]["return_input_ids"].as_bool().unwrap(),
                return_output_ids: case["args"]["return_output_ids"].as_bool().unwrap(),
                ..Default::default()
            };
            let (_, options) = super::super::extensions::request_options(
                &case["body"],
                &HeaderMap::new(),
                &args,
                chat,
            )
            .unwrap();
            let n = case["body"]["n"].as_u64().unwrap() as usize;
            let mut receivers = Vec::new();
            for (index, item) in case["items"].as_array().unwrap().iter().enumerate() {
                let rid = Rid::from_client(item["meta_info"]["id"].as_str().unwrap());
                let (tx, rx) = tokio::sync::mpsc::channel(4);
                tx.send(ResponseItem::Done(fixture_output(item)))
                    .await
                    .unwrap();
                receivers.push((index, rid, rx.into()));
            }
            let response = if chat {
                unary_chat(
                    receivers,
                    AbortGuard::new_empty(senders()),
                    "contract-0".into(),
                    "org/model".into(),
                    123,
                    false,
                    None,
                    None,
                    None,
                    true,
                    None,
                    options,
                )
                .await
            } else {
                let submitted = receivers
                    .into_iter()
                    .map(|(index, rid, rx)| SubmittedChoice {
                        index,
                        prompt_index: index / n,
                        rid,
                        echo: String::new(),
                        rx,
                    })
                    .collect();
                unary_completion(
                    submitted,
                    AbortGuard::new_empty(senders()),
                    "contract-0".into(),
                    "org/model".into(),
                    123,
                    false,
                    false,
                    options,
                )
                .await
            };
            let body = axum::body::to_bytes(response.into_body(), 1024 * 1024)
                .await
                .unwrap();
            let actual: Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(
                actual, case["expected"],
                "{} {}",
                case["endpoint"], case["body"]
            );
        }
    }

    #[tokio::test]
    async fn streaming_extensions_match_python_generators() {
        use super::super::{chat::chat_event_stream, completions::completion_event_stream};
        use futures::StreamExt;

        let cases: Value =
            serde_json::from_str(include_str!("../../../testdata/openai_streams_python.json"))
                .unwrap();
        for case in cases.as_array().unwrap() {
            let chat = case["endpoint"] == "chat";
            let args = ServerArgs {
                enable_cache_report: case["args"]["enable_cache_report"].as_bool().unwrap(),
                return_input_ids: case["args"]["return_input_ids"].as_bool().unwrap(),
                return_output_ids: case["args"]["return_output_ids"].as_bool().unwrap(),
                ..Default::default()
            };
            let mut headers = HeaderMap::new();
            if case["headers"]["x-sglext-ids-framed"] == "1" {
                headers.insert("x-sglext-ids-framed", "1".parse().unwrap());
            }
            let (_, options) =
                super::super::extensions::request_options(&case["body"], &headers, &args, chat)
                    .unwrap();
            let n = case["body"]["n"].as_u64().unwrap() as usize;
            let count = case["chunks"].as_array().unwrap().len() / 2;
            let mut receivers = Vec::new();
            let mut transmitters = Vec::new();
            for index in 0..count {
                let (tx, rx) = tokio::sync::mpsc::channel(4);
                transmitters.push(tx);
                receivers.push((
                    index,
                    Rid::from_client(&format!("contract-{index}")),
                    rx.into(),
                ));
            }
            for chunk in case["chunks"].as_array().unwrap() {
                let index = chunk["index"].as_u64().unwrap() as usize;
                let mut output = fixture_output(chunk);
                let step = output.completion_tokens as usize;
                output.token_ids = output.token_ids[step - 1..].to_vec();
                output.text = output.text.chars().skip(step - 1).collect();
                output.completion_tokens = 1;
                let item = if output.finish_reason.is_some() {
                    ResponseItem::Done(output)
                } else {
                    ResponseItem::Frame(output)
                };
                transmitters[index].send(item).await.unwrap();
            }
            let events: Vec<StreamEvent> = if chat {
                chat_event_stream(
                    receivers,
                    AbortGuard::new_empty(senders()),
                    "contract-0".into(),
                    "org/model".into(),
                    123,
                    false,
                    true,
                    None,
                    None,
                    false,
                    None,
                    None,
                    false,
                    true,
                    None,
                    options,
                )
                .collect()
                .await
            } else {
                let submitted = receivers
                    .into_iter()
                    .map(|(index, rid, rx)| SubmittedChoice {
                        index,
                        prompt_index: index / n,
                        rid,
                        echo: String::new(),
                        rx,
                    })
                    .collect();
                completion_event_stream(
                    submitted,
                    AbortGuard::new_empty(senders()),
                    "contract-0".into(),
                    "org/model".into(),
                    123,
                    false,
                    false,
                    true,
                    true,
                    options,
                )
                .map(StreamEvent::from)
                .collect()
                .await
            };
            assert_eq!(events.last().unwrap().data, "[DONE]");
            let frames: Vec<Value> = events[..events.len()-1].iter().map(|event| json!({
                "event": event.event, "data": serde_json::from_str::<Value>(&event.data).unwrap()
            })).collect();
            let tail: Vec<_> = frames
                .iter()
                .filter(|frame| {
                    let choices = frame["data"]["choices"].as_array().unwrap();
                    choices.is_empty()
                        || choices.iter().any(|choice| {
                            choice.get("hidden_states").is_some()
                                || choice["delta"].get("hidden_states").is_some()
                        })
                })
                .cloned()
                .collect();
            assert_eq!(
                json!(tail),
                case["tail"],
                "{} {}",
                case["endpoint"],
                case["body"]
            );
            for index in 0..count {
                let choices: Vec<_> = frames
                    .iter()
                    .flat_map(|frame| frame["data"]["choices"].as_array().unwrap())
                    .filter(|choice| choice["index"] == index)
                    .collect();
                let terminal = choices
                    .iter()
                    .find(|choice| !choice["finish_reason"].is_null())
                    .unwrap();
                assert_eq!(terminal["matched_stop"], "END");
                if !chat && case["body"]["return_token_ids"] == true {
                    let ids: Vec<_> = choices
                        .iter()
                        .filter_map(|choice| choice["token_ids"].as_array())
                        .flatten()
                        .cloned()
                        .collect();
                    assert_eq!(json!(ids), json!([10 + index, 20 + index]));
                    assert_eq!(
                        choices
                            .iter()
                            .filter(|choice| choice.get("prompt_token_ids").is_some())
                            .count(),
                        1
                    );
                }
            }
        }
    }
}
