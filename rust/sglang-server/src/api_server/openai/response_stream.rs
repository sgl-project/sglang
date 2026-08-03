//! Responses lifecycle and streaming event serialization.

use dynamo_parsers::ToolDefinition;
use dynamo_parsers::tool_calling::jail::{Annotated, apply_tool_calling_jail};
use dynamo_protocols::types::responses::{
    AssistantRole, CreateResponse, IncompleteDetails, InputTokenDetails, Instructions, LogProb,
    OutputContent, OutputItem, OutputMessage, OutputMessageContent, OutputStatus,
    OutputTextContent, OutputTokenDetails, Response as OpenAIResponse, ResponseCompletedEvent,
    ResponseContentPartAddedEvent, ResponseContentPartDoneEvent, ResponseCreatedEvent,
    ResponseFunctionCallArgumentsDeltaEvent, ResponseFunctionCallArgumentsDoneEvent,
    ResponseInProgressEvent, ResponseIncompleteEvent, ResponseLogProb,
    ResponseOutputItemAddedEvent, ResponseOutputItemDoneEvent, ResponseReasoningTextDeltaEvent,
    ResponseReasoningTextDoneEvent, ResponseStreamEvent, ResponseTextDeltaEvent,
    ResponseTextDoneEvent, ResponseTopLobProb, ResponseUsage, Status, TopLogProb,
};
use dynamo_protocols::types::{
    ChatChoiceLogprobs, ChatChoiceStream, ChatCompletionMessageContent,
    ChatCompletionMessageToolCall, ChatCompletionRequestMessage, ChatCompletionToolChoiceOption,
    CreateChatCompletionStreamResponse, FunctionCall, FunctionType,
};
use futures::StreamExt;
use tokio::sync::mpsc;

use super::super::frame::OutputAccumulator;
use super::super::guard::AbortGuard;
use super::chat::chat_logprobs;
use super::reasoning::ReasoningStreamSplitter;
use super::responses::{
    ResponseStore, StoredResponse, append_response_output, pending_reasoning_item,
    response_function_call, response_reasoning_item,
};
use super::tools::{chat_delta, chat_finish_reason, dynamo_parser_name};
use super::{streaming_error, unix_seconds};
use crate::ids::Rid;
use crate::message::{ChunkEvent, ChunkExtras, EgressItem};

#[allow(clippy::too_many_arguments)]
pub(super) fn responses_event_stream(
    mut rx: mpsc::Receiver<EgressItem>,
    mut guard: AbortGuard,
    rid: Rid,
    response_id: String,
    created_at: u64,
    model: String,
    request: CreateResponse,
    parser: Option<String>,
    reasoning_parser: Option<String>,
    tools: Option<Vec<ToolDefinition>>,
    tool_choice: Option<ChatCompletionToolChoiceOption>,
    uses_tool_call_structural_tag: bool,
    store: ResponseStore,
    mut messages: Vec<ChatCompletionRequestMessage>,
) -> impl futures::Stream<Item = String> {
    let stream_id = response_id.clone();
    let stream_model = model.clone();
    let want_logprobs = request.top_logprobs.is_some();
    let (output_tx, output_rx) = tokio::sync::oneshot::channel();
    let raw = async_stream::stream! {
        let mut output_tx = Some(output_tx);
        let mut accumulator = OutputAccumulator::default();
        // One stateful reasoning splitter for the stream (mirrors the chat
        // path's per-choice state).
        let mut splitter = ReasoningStreamSplitter::new(reasoning_parser.as_deref());
        loop {
            let (output, done) = match rx.recv().await {
                Some(EgressItem::Frame(output)) => (output, false),
                Some(EgressItem::Done(output)) => {
                    guard.disarm(&rid);
                    (output, true)
                }
                Some(EgressItem::Error(error)) => {
                    guard.disarm(&rid);
                    yield Annotated {
                        data: None,
                        id: None,
                        event: None,
                        comment: None,
                        error: Some(streaming_error(error.http_status(), error.to_string())),
                    };
                    return;
                }
                Some(EgressItem::Control(_)) | Some(EgressItem::Data(_)) => continue,
                None => {
                    yield Annotated {
                        data: None,
                        id: None,
                        event: None,
                        comment: None,
                        error: Some(streaming_error(500, "response truncated before completion")),
                    };
                    return;
                }
            };
            if let Some((code, message)) = output
                .finish_reason
                .as_ref()
                .and_then(|reason| reason.abort_status())
            {
                yield Annotated {
                    data: None,
                    id: None,
                    event: None,
                    comment: None,
                    error: Some(streaming_error(code, message)),
                };
                return;
            }

            accumulator.fold(&output);
            let finish_reason = chat_finish_reason(&output);
            // Split the frame's text into (reasoning, normal) chat deltas when
            // `--reasoning-parser` is set, mirroring chat_event_stream: the
            // reasoning delta comes first, then the normal delta; the
            // parser-buffered tail flushes before the terminal frame — both
            // columns, since parsers like MiniMax M3 hold the answer text
            // until EOF — and the finish reason rides the last emitted chunk.
            let mut choices = Vec::with_capacity(2);
            if splitter.enabled() {
                let (reasoning_text, normal_text) =
                    splitter.split(&output.text, &output.token_ids);
                let mut remaining_logprobs =
                    want_logprobs.then(|| chat_logprobs(output.extras.as_deref()));
                if !reasoning_text.is_empty() {
                    choices.push(ChatChoiceStream {
                        index: 0,
                        delta: chat_delta(None, None, None, Some(reasoning_text)),
                        finish_reason: None,
                        logprobs: remaining_logprobs.clone(),
                    });
                    remaining_logprobs = None;
                }
                if !normal_text.is_empty() {
                    choices.push(ChatChoiceStream {
                        index: 0,
                        delta: chat_delta(Some(normal_text), None, None, None),
                        finish_reason: None,
                        logprobs: remaining_logprobs,
                    });
                }
                if finish_reason.is_some() {
                    let (reasoning_tail, normal_tail) = splitter.finish();
                    if !reasoning_tail.is_empty() {
                        choices.push(ChatChoiceStream {
                            index: 0,
                            delta: chat_delta(None, None, None, Some(reasoning_tail)),
                            finish_reason: None,
                            logprobs: None,
                        });
                    }
                    if !normal_tail.is_empty() {
                        choices.push(ChatChoiceStream {
                            index: 0,
                            delta: chat_delta(Some(normal_tail), None, None, None),
                            finish_reason: None,
                            logprobs: None,
                        });
                    }
                }
                match choices.last_mut() {
                    Some(last) => last.finish_reason = finish_reason,
                    None => choices.push(ChatChoiceStream {
                        index: 0,
                        delta: chat_delta(None, None, None, None),
                        finish_reason,
                        logprobs: None,
                    }),
                }
            } else {
                choices.push(ChatChoiceStream {
                    index: 0,
                    delta: chat_delta(
                        (!output.text.is_empty()).then_some(output.text),
                        None,
                        None,
                        None,
                    ),
                    finish_reason,
                    logprobs: want_logprobs.then(|| chat_logprobs(output.extras.as_deref())),
                });
            }
            for choice in choices {
                yield Annotated {
                    data: Some(CreateChatCompletionStreamResponse {
                        id: stream_id.clone(),
                        choices: vec![choice],
                        created: u32::try_from(created_at).unwrap_or(u32::MAX),
                        model: stream_model.clone(),
                        service_tier: None,
                        system_fingerprint: None,
                        object: "chat.completion.chunk".into(),
                        usage: None,
                    }),
                    id: None,
                    event: None,
                    comment: None,
                    error: None,
                };
            }
            if done {
                if let Some(tx) = output_tx.take() {
                    let _ = tx.send(accumulator.into_output());
                }
                break;
            }
        }
    };
    let parsed: std::pin::Pin<
        Box<dyn futures::Stream<Item = Annotated<CreateChatCompletionStreamResponse>> + Send>,
    > = if let Some(parser) = parser {
        Box::pin(apply_tool_calling_jail(
            Some(dynamo_parser_name(&parser).to_owned()),
            tool_choice,
            tools,
            uses_tool_call_structural_tag,
            raw,
        ))
    } else {
        Box::pin(raw)
    };

    async_stream::stream! {
        let mut sequence = 0u64;
        let created = response_object(
            &response_id,
            &model,
            &request,
            created_at,
            Status::InProgress,
            vec![],
            None,
        );
        yield serialize_response_event(ResponseStreamEvent::ResponseCreated(
            ResponseCreatedEvent {
                sequence_number: sequence,
                response: created.clone(),
            },
        ));
        sequence += 1;
        yield serialize_response_event(ResponseStreamEvent::ResponseInProgress(
            ResponseInProgressEvent {
                sequence_number: sequence,
                response: created,
            },
        ));
        sequence += 1;

        let mut completed_items = Vec::new();
        let mut open_message: Option<(String, u32, OutputTextContent)> = None;
        // Open reasoning item (item_id, output_index, text). Python keeps the
        // same single-open-item invariant: reasoning closes the message, and
        // normal text or tool calls close the reasoning item.
        let mut reasoning_state: Option<(String, u32, String)> = None;
        let mut tool_call_emitted = false;
        futures::pin_mut!(parsed);
        while let Some(item) = parsed.next().await {
            if let Some(error) = item.error {
                yield error;
                yield "[DONE]".to_string();
                return;
            }
            let Some(response) = item.data else {
                continue;
            };
            for choice in response.choices {
                // Reasoning deltas first (Python `parse_stream_chunk` emits
                // the reasoning chunk before the normal one; a reasoning chunk
                // closes any open message item).
                if let Some(reasoning) = choice
                    .delta
                    .reasoning_content
                    .as_deref()
                    .filter(|text| !text.is_empty())
                {
                    if let Some((item_id, output_index, content)) = open_message.take() {
                        let (events, item) = finish_response_text_item(
                            &mut sequence, item_id, output_index, content,
                        );
                        for event in events {
                            yield event;
                        }
                        completed_items.push(item);
                    }
                    if reasoning_state.is_none() {
                        let item_id = format!("rs_{}", uuid::Uuid::new_v4().simple());
                        let output_index =
                            u32::try_from(completed_items.len()).unwrap_or(u32::MAX);
                        yield serialize_response_event(
                            ResponseStreamEvent::ResponseOutputItemAdded(
                                ResponseOutputItemAddedEvent {
                                    sequence_number: sequence,
                                    output_index,
                                    item: pending_reasoning_item(item_id.clone()),
                                },
                            ),
                        );
                        sequence += 1;
                        reasoning_state = Some((item_id, output_index, String::new()));
                    }
                    if let Some((item_id, output_index, text)) = reasoning_state.as_mut() {
                        text.push_str(reasoning);
                        yield serialize_response_event(
                            ResponseStreamEvent::ResponseReasoningTextDelta(
                                ResponseReasoningTextDeltaEvent {
                                    sequence_number: sequence,
                                    item_id: item_id.clone(),
                                    output_index: *output_index,
                                    content_index: 0,
                                    delta: reasoning.to_owned(),
                                },
                            ),
                        );
                        sequence += 1;
                    }
                }
                let mut calls = choice.delta.tool_calls.unwrap_or_default();
                if !request.parallel_tool_calls.unwrap_or(true) {
                    if tool_call_emitted {
                        calls.clear();
                    } else {
                        calls.truncate(1);
                    }
                }
                if !calls.is_empty() {
                    // Python closes the reasoning item before opening tool-call
                    // items.
                    if let Some((item_id, output_index, text)) = reasoning_state.take() {
                        let (events, item) =
                            finish_reasoning_item(&mut sequence, item_id, output_index, text);
                        for event in events {
                            yield event;
                        }
                        completed_items.push(item);
                    }
                    if let Some((item_id, output_index, content)) = open_message.take() {
                        let (events, item) = finish_response_text_item(
                            &mut sequence, item_id, output_index, content,
                        );
                        for event in events {
                            yield event;
                        }
                        completed_items.push(item);
                    }
                    for call in calls {
                        tool_call_emitted = true;
                        let call = ChatCompletionMessageToolCall {
                            id: call
                                .id
                                .unwrap_or_else(|| format!("call_{}", uuid::Uuid::new_v4().simple())),
                            r#type: FunctionType::Function,
                            function: FunctionCall {
                                name: call
                                    .function
                                    .as_ref()
                                    .and_then(|function| function.name.clone())
                                    .unwrap_or_default(),
                                arguments: call
                                    .function
                                    .and_then(|function| function.arguments)
                                    .unwrap_or_default(),
                            },
                        };
                        let item = response_function_call(call);
                        let (item_id, name, arguments) = match &item {
                            OutputItem::FunctionCall(call) => (
                                call.id.clone().unwrap_or_else(|| {
                                    format!("fc_{}", uuid::Uuid::new_v4().simple())
                                }),
                                call.name.clone(),
                                call.arguments.clone(),
                            ),
                            _ => unreachable!(),
                        };
                        let mut pending = item.clone();
                        if let OutputItem::FunctionCall(call) = &mut pending {
                            call.status = Some(OutputStatus::InProgress);
                            call.arguments.clear();
                        }
                        let output_index =
                            u32::try_from(completed_items.len()).unwrap_or(u32::MAX);
                        yield serialize_response_event(
                            ResponseStreamEvent::ResponseOutputItemAdded(
                                ResponseOutputItemAddedEvent {
                                    sequence_number: sequence,
                                    output_index,
                                    item: pending,
                                },
                            ),
                        );
                        sequence += 1;
                        yield serialize_response_event(
                            ResponseStreamEvent::ResponseFunctionCallArgumentsDelta(
                                ResponseFunctionCallArgumentsDeltaEvent {
                                    sequence_number: sequence,
                                    item_id: item_id.clone(),
                                    output_index,
                                    delta: arguments.clone(),
                                },
                            ),
                        );
                        sequence += 1;
                        yield serialize_response_event(
                            ResponseStreamEvent::ResponseFunctionCallArgumentsDone(
                                ResponseFunctionCallArgumentsDoneEvent {
                                    name: Some(name),
                                    sequence_number: sequence,
                                    item_id,
                                    output_index,
                                    arguments,
                                },
                            ),
                        );
                        sequence += 1;
                        yield serialize_response_event(
                            ResponseStreamEvent::ResponseOutputItemDone(
                                ResponseOutputItemDoneEvent {
                                    sequence_number: sequence,
                                    output_index,
                                    item: item.clone(),
                                },
                            ),
                        );
                        sequence += 1;
                        completed_items.push(item);
                    }
                }
                if let Some(ChatCompletionMessageContent::Text(delta)) = choice.delta.content
                    && !delta.is_empty()
                {
                    // Python closes the reasoning item when normal text
                    // resumes, before opening or continuing the message.
                    if let Some((item_id, output_index, text)) = reasoning_state.take() {
                        let (events, item) =
                            finish_reasoning_item(&mut sequence, item_id, output_index, text);
                        for event in events {
                            yield event;
                        }
                        completed_items.push(item);
                    }
                    if open_message.is_none() {
                        let item_id = format!("msg_{}", uuid::Uuid::new_v4().simple());
                        let output_index =
                            u32::try_from(completed_items.len()).unwrap_or(u32::MAX);
                        let content = text_output_content(
                            String::new(),
                            want_logprobs.then(Vec::new),
                        );
                        let pending = text_response_message(
                            item_id.clone(),
                            OutputStatus::InProgress,
                            content.clone(),
                        );
                        yield serialize_response_event(
                            ResponseStreamEvent::ResponseOutputItemAdded(
                                ResponseOutputItemAddedEvent {
                                    sequence_number: sequence,
                                    output_index,
                                    item: pending,
                                },
                            ),
                        );
                        sequence += 1;
                        yield serialize_response_event(
                            ResponseStreamEvent::ResponseContentPartAdded(
                                ResponseContentPartAddedEvent {
                                    sequence_number: sequence,
                                    item_id: item_id.clone(),
                                    output_index,
                                    content_index: 0,
                                    part: OutputContent::OutputText(content.clone()),
                                },
                            ),
                        );
                        sequence += 1;
                        open_message = Some((item_id, output_index, content));
                    }
                    if let Some((item_id, output_index, content)) = open_message.as_mut() {
                        let delta_logprobs = output_text_logprobs(choice.logprobs.as_ref());
                        let stream_logprobs = delta_logprobs
                            .as_deref()
                            .map(response_stream_logprobs);
                        content.text.push_str(&delta);
                        if let Some(delta_logprobs) = delta_logprobs {
                            content
                                .logprobs
                                .get_or_insert_default()
                                .extend(delta_logprobs);
                        }
                        yield serialize_response_event(
                            ResponseStreamEvent::ResponseOutputTextDelta(ResponseTextDeltaEvent {
                                sequence_number: sequence,
                                item_id: item_id.clone(),
                                output_index: *output_index,
                                content_index: 0,
                                delta,
                                logprobs: stream_logprobs,
                            }),
                        );
                        sequence += 1;
                    }
                }
            }
        }

        let output = match output_rx.await {
            Ok(output) => output,
            Err(_) => {
                yield streaming_error(500, "response truncated before completion");
                yield "[DONE]".to_string();
                return;
            }
        };
        if let Some((item_id, output_index, text)) = reasoning_state.take() {
            let (events, item) =
                finish_reasoning_item(&mut sequence, item_id, output_index, text);
            for event in events {
                yield event;
            }
            completed_items.push(item);
        }
        if open_message.is_none() && completed_items.is_empty() {
            let item_id = format!("msg_{}", uuid::Uuid::new_v4().simple());
            let output_index = 0;
            let content =
                text_output_content(String::new(), want_logprobs.then(Vec::new));
            let pending = text_response_message(
                item_id.clone(),
                OutputStatus::InProgress,
                content.clone(),
            );
            yield serialize_response_event(ResponseStreamEvent::ResponseOutputItemAdded(
                ResponseOutputItemAddedEvent {
                    sequence_number: sequence,
                    output_index,
                    item: pending,
                },
            ));
            sequence += 1;
            yield serialize_response_event(ResponseStreamEvent::ResponseContentPartAdded(
                ResponseContentPartAddedEvent {
                    sequence_number: sequence,
                    item_id: item_id.clone(),
                    output_index,
                    content_index: 0,
                    part: OutputContent::OutputText(content.clone()),
                },
            ));
            sequence += 1;
            open_message = Some((item_id, output_index, content));
        }
        if let Some((item_id, output_index, content)) = open_message {
            let (events, item) =
                finish_response_text_item(&mut sequence, item_id, output_index, content);
            for event in events {
                yield event;
            }
            completed_items.push(item);
        }

        let final_status = response_status(&output);
        let completed = response_object(
            &response_id,
            &model,
            &request,
            created_at,
            final_status.clone(),
            completed_items,
            Some(responses_usage(output.prompt_tokens, output.completion_tokens)),
        );
        append_response_output(&mut messages, &completed.output);
        if request.store.unwrap_or(true) {
            store.write().await.insert(
                response_id.clone(),
                StoredResponse {
                    response: completed.clone(),
                    messages,
                    rid: None,
                },
            );
        }
        let event = if final_status == Status::Incomplete {
            ResponseStreamEvent::ResponseIncomplete(ResponseIncompleteEvent {
                sequence_number: sequence,
                response: completed,
            })
        } else {
            ResponseStreamEvent::ResponseCompleted(ResponseCompletedEvent {
                sequence_number: sequence,
                response: completed,
            })
        };
        yield serialize_response_event(event);
        yield "[DONE]".to_string();
    }
}

fn finish_response_text_item(
    sequence: &mut u64,
    item_id: String,
    output_index: u32,
    content: OutputTextContent,
) -> ([String; 3], OutputItem) {
    let text_done = serialize_response_event(ResponseStreamEvent::ResponseOutputTextDone(
        ResponseTextDoneEvent {
            sequence_number: *sequence,
            item_id: item_id.clone(),
            output_index,
            content_index: 0,
            text: content.text.clone(),
            logprobs: content.logprobs.as_deref().map(response_stream_logprobs),
        },
    ));
    *sequence += 1;
    let part_done = serialize_response_event(ResponseStreamEvent::ResponseContentPartDone(
        ResponseContentPartDoneEvent {
            sequence_number: *sequence,
            item_id: item_id.clone(),
            output_index,
            content_index: 0,
            part: OutputContent::OutputText(content.clone()),
        },
    ));
    *sequence += 1;
    let item = text_response_message(item_id, OutputStatus::Completed, content);
    let item_done = serialize_response_event(ResponseStreamEvent::ResponseOutputItemDone(
        ResponseOutputItemDoneEvent {
            sequence_number: *sequence,
            output_index,
            item: item.clone(),
        },
    ));
    *sequence += 1;
    ([text_done, part_done, item_done], item)
}

/// Close an open reasoning item: the `reasoning_text.done` event followed by
/// the item's `output_item.done` (Python `_close_reasoning_item` without the
/// summary events — summary requests are rejected up front).
fn finish_reasoning_item(
    sequence: &mut u64,
    item_id: String,
    output_index: u32,
    text: String,
) -> ([String; 2], OutputItem) {
    let text_done = serialize_response_event(ResponseStreamEvent::ResponseReasoningTextDone(
        ResponseReasoningTextDoneEvent {
            sequence_number: *sequence,
            item_id: item_id.clone(),
            output_index,
            content_index: 0,
            text: text.clone(),
        },
    ));
    *sequence += 1;
    let item = response_reasoning_item(item_id, text, Some(OutputStatus::Completed));
    let item_done = serialize_response_event(ResponseStreamEvent::ResponseOutputItemDone(
        ResponseOutputItemDoneEvent {
            sequence_number: *sequence,
            output_index,
            item: item.clone(),
        },
    ));
    *sequence += 1;
    ([text_done, item_done], item)
}

fn serialize_response_event(event: ResponseStreamEvent) -> String {
    serde_json::to_string(&event).expect("OpenAI response event must serialize")
}

pub(super) fn text_output_content(
    text: String,
    logprobs: Option<Vec<LogProb>>,
) -> OutputTextContent {
    OutputTextContent {
        annotations: vec![],
        logprobs,
        text,
    }
}

pub(super) fn text_response_message(
    id: String,
    status: OutputStatus,
    content: OutputTextContent,
) -> OutputItem {
    OutputItem::Message(OutputMessage {
        content: vec![OutputMessageContent::OutputText(content)],
        id,
        role: AssistantRole::Assistant,
        phase: None,
        status,
    })
}

fn output_text_logprobs(logprobs: Option<&ChatChoiceLogprobs>) -> Option<Vec<LogProb>> {
    logprobs.map(|logprobs| {
        logprobs
            .content
            .as_deref()
            .unwrap_or_default()
            .iter()
            .map(|token| LogProb {
                bytes: token
                    .bytes
                    .clone()
                    .unwrap_or_else(|| token.token.as_bytes().to_vec()),
                logprob: f64::from(token.logprob),
                token: token.token.clone(),
                top_logprobs: token
                    .top_logprobs
                    .iter()
                    .map(|top| TopLogProb {
                        bytes: top
                            .bytes
                            .clone()
                            .unwrap_or_else(|| top.token.as_bytes().to_vec()),
                        logprob: f64::from(top.logprob),
                        token: top.token.clone(),
                    })
                    .collect(),
            })
            .collect()
    })
}

fn response_stream_logprobs(logprobs: &[LogProb]) -> Vec<ResponseLogProb> {
    logprobs
        .iter()
        .map(|token| ResponseLogProb {
            logprob: token.logprob,
            token: token.token.clone(),
            top_logprobs: token
                .top_logprobs
                .iter()
                .map(|top| ResponseTopLobProb {
                    logprob: top.logprob,
                    token: top.token.clone(),
                })
                .collect(),
        })
        .collect()
}

pub(super) fn chunk_response_logprobs(extras: Option<&ChunkExtras>) -> Vec<LogProb> {
    let logprobs = chat_logprobs(extras);
    output_text_logprobs(Some(&logprobs)).unwrap_or_default()
}

pub(super) fn responses_usage(prompt_tokens: u32, completion_tokens: u64) -> ResponseUsage {
    let output_tokens = u32::try_from(completion_tokens).unwrap_or(u32::MAX);
    ResponseUsage {
        input_tokens: prompt_tokens,
        input_tokens_details: InputTokenDetails { cached_tokens: 0 },
        output_tokens,
        output_tokens_details: OutputTokenDetails {
            reasoning_tokens: 0,
        },
        total_tokens: prompt_tokens.saturating_add(output_tokens),
    }
}

pub(super) fn response_status(output: &ChunkEvent) -> Status {
    let reached_limit = output
        .finish_reason
        .as_ref()
        .and_then(|reason| reason.kind_name())
        == Some("length");
    if reached_limit {
        Status::Incomplete
    } else {
        Status::Completed
    }
}

pub(super) fn response_object(
    id: &str,
    model: &str,
    request: &CreateResponse,
    created_at: u64,
    status: Status,
    output: Vec<OutputItem>,
    usage: Option<ResponseUsage>,
) -> OpenAIResponse {
    OpenAIResponse {
        background: request.background,
        billing: None,
        conversation: None,
        created_at,
        completed_at: (status == Status::Completed).then(unix_seconds),
        error: None,
        id: id.to_owned(),
        incomplete_details: (status == Status::Incomplete).then(|| IncompleteDetails {
            reason: "max_output_tokens".into(),
        }),
        instructions: request.instructions.clone().map(Instructions::Text),
        max_output_tokens: request.max_output_tokens,
        metadata: request.metadata.clone(),
        model: model.to_owned(),
        object: "response".into(),
        output,
        parallel_tool_calls: request.parallel_tool_calls,
        previous_response_id: request.previous_response_id.clone(),
        prompt: request.prompt.clone(),
        prompt_cache_key: request.prompt_cache_key.clone(),
        prompt_cache_retention: request.prompt_cache_retention,
        reasoning: request.reasoning.clone(),
        safety_identifier: request.safety_identifier.clone(),
        service_tier: request.service_tier,
        status,
        temperature: request.temperature,
        text: request.text.clone(),
        tool_choice: request.tool_choice.clone(),
        tools: request.tools.clone(),
        top_logprobs: request.top_logprobs,
        top_p: request.top_p,
        truncation: request.truncation,
        usage,
    }
}
