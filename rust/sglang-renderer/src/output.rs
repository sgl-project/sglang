//! Request-scoped OpenAI chat output interpretation.
//!
//! The processor owns parser selection and mutable reasoning/tool state. Its
//! input is decoded engine output; its output is typed chat semantics.
//! Submission, cancellation, scheduler transport, JSON/SSE serialization, and
//! gRPC framing remain host responsibilities.

use std::pin::Pin;

use dynamo_parsers::ToolDefinition;
use dynamo_parsers::reasoning::{
    ReasoningParser as _, ReasoningParserType, ReasoningParserWrapper,
};
use dynamo_parsers::tool_calling::jail::{Annotated, apply_tool_calling_jail};
use dynamo_parsers::try_tool_call_parse_aggregate_finalize;
use dynamo_protocols::types::{
    ChatChoiceLogprobs, ChatChoiceStream, ChatCompletionMessageContent,
    ChatCompletionMessageToolCallChunk, ChatCompletionStreamResponseDelta,
    ChatCompletionToolChoiceOption, CreateChatCompletionStreamResponse, FinishReason, Role,
};
use futures::{Stream, StreamExt};
use serde::{Deserialize, Serialize};

use crate::openai::dynamo_parser_name;

/// Engine-neutral terminal reason understood by chat response processing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatFinishReason {
    Stop,
    Length,
    ContentFilter,
    ToolCalls,
}

/// One decoded engine update after host-specific egress conversion.
pub struct DecodedChatEvent {
    pub choice: usize,
    pub text: String,
    pub token_ids: Vec<i32>,
    pub finish_reason: Option<ChatFinishReason>,
    pub logprobs: Option<ChatChoiceLogprobs>,
    pub prompt_tokens: u32,
    pub completion_tokens: u64,
}

/// A host error carried through semantic processing without interpreting it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatOutputError {
    pub status_code: u16,
    pub message: String,
}

/// Input to the output processor. Errors are passed through unchanged.
pub enum ChatOutputInput {
    Decoded(DecodedChatEvent),
    Error(ChatOutputError),
}

/// One semantic tool-call delta, independent of HTTP or gRPC framing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatToolCallDelta {
    pub index: u32,
    pub id: Option<String>,
    pub name: Option<String>,
    pub arguments: Option<String>,
}

/// One completed semantic tool call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

/// Semantic chat output. Protocol adapters add response metadata and wire
/// framing without knowing how reasoning or tool syntax was parsed.
#[derive(Debug, Clone)]
pub enum ChatEvent {
    Role {
        choice: usize,
    },
    Delta {
        choice: usize,
        content: Option<String>,
        reasoning_content: Option<String>,
        tool_calls: Option<Vec<ChatToolCallDelta>>,
        finish_reason: Option<ChatFinishReason>,
        logprobs: Option<ChatChoiceLogprobs>,
    },
    Usage {
        prompt_tokens: u32,
        completion_tokens: u64,
    },
}

/// Output from semantic processing. Request lifecycle errors remain owned by
/// the host and are merely preserved while parser streams are running.
pub enum ChatOutputItem {
    Event(ChatEvent),
    Error(ChatOutputError),
}

/// Parsed result for one completed non-streaming choice.
pub struct ParsedChatOutput {
    pub content: String,
    pub reasoning_content: String,
    pub tool_calls: Option<Vec<ChatToolCall>>,
}

/// Mutable parser state for one generated choice.
struct ChoiceOutputProcessor {
    reasoning: ReasoningStreamSplitter,
}

/// Request-scoped chat response processor.
///
/// Parser names, tool definitions, structural-tag decisions, and mutable
/// per-choice state are private so protocol adapters cannot accidentally
/// reimplement the semantic contract.
pub struct ChatOutputProcessor {
    tool_parser: Option<String>,
    tools: Option<Vec<ToolDefinition>>,
    tool_choice: Option<ChatCompletionToolChoiceOption>,
    uses_tool_call_structural_tag: bool,
    parallel_tool_calls: bool,
    choices: Vec<ChoiceOutputProcessor>,
}

impl ChatOutputProcessor {
    pub(crate) fn new(
        tool_parser: Option<String>,
        reasoning_parser: Option<String>,
        tools: Option<Vec<ToolDefinition>>,
        tool_choice: Option<ChatCompletionToolChoiceOption>,
        uses_tool_call_structural_tag: bool,
        parallel_tool_calls: bool,
        choice_count: usize,
    ) -> Self {
        Self {
            tool_parser,
            tools,
            tool_choice,
            uses_tool_call_structural_tag,
            parallel_tool_calls,
            choices: (0..choice_count)
                .map(|_| ChoiceOutputProcessor {
                    reasoning: ReasoningStreamSplitter::new(reasoning_parser.as_deref()),
                })
                .collect(),
        }
    }

    pub fn choice_count(&self) -> usize {
        self.choices.len()
    }

    /// Interpret one completed generation using the same parser selection that
    /// affected request lowering.
    pub async fn process_unary(&mut self, text: String, token_ids: &[i32]) -> ParsedChatOutput {
        let reasoning_parser = self
            .choices
            .first()
            .and_then(|choice| choice.reasoning.name.as_deref());
        let (reasoning_content, content) =
            split_reasoning_unary(reasoning_parser, &text, token_ids);
        let (content, tool_calls) = parse_chat_tool_calls(
            content,
            self.tool_parser.as_deref(),
            self.tools.as_deref(),
            self.parallel_tool_calls,
        )
        .await;
        ParsedChatOutput {
            content,
            reasoning_content,
            tool_calls,
        }
    }

    /// Interpret decoded output and emit semantic chat events.
    ///
    /// OpenAI-shaped values are used only as a private adapter to Dynamo's
    /// stateful tool-call jail. They are removed before events leave this
    /// crate, so response identity, model metadata, usage policy, and wire
    /// framing remain outside the renderer.
    pub fn process_stream<S>(
        mut self,
        input: S,
    ) -> Pin<Box<dyn Stream<Item = ChatOutputItem> + Send>>
    where
        S: Stream<Item = ChatOutputInput> + Send + 'static,
    {
        let count = self.choices.len();
        let raw = async_stream::stream! {
            let mut prompt_tokens = 0u32;
            let mut completion_tokens = 0u64;

            for index in 0..count {
                yield annotated_choices(vec![ChatChoiceStream {
                    index: index as u32,
                    delta: chat_delta(None, Some(Role::Assistant), None, None),
                    finish_reason: None,
                    logprobs: None,
                }]);
            }

            futures::pin_mut!(input);
            while let Some(item) = input.next().await {
                let decoded = match item {
                    ChatOutputInput::Decoded(decoded) => decoded,
                    ChatOutputInput::Error(error) => {
                        yield Annotated {
                            data: None,
                            id: None,
                            event: None,
                            comment: None,
                            error: serde_json::to_string(&error).ok(),
                        };
                        continue;
                    }
                };

                if prompt_tokens == 0 {
                    prompt_tokens = decoded.prompt_tokens;
                }
                completion_tokens = completion_tokens.saturating_add(decoded.completion_tokens);

                let Some(choice) = self.choices.get_mut(decoded.choice) else {
                    yield Annotated {
                        data: None,
                        id: None,
                        event: None,
                        comment: None,
                        error: serde_json::to_string(&ChatOutputError {
                            status_code: 500,
                            message: format!("output choice {} is out of range", decoded.choice),
                        }).ok(),
                    };
                    continue;
                };
                let index = decoded.choice as u32;
                let (reasoning_text, normal_text) =
                    choice.reasoning.split(&decoded.text, &decoded.token_ids);
                let mut remaining_logprobs = decoded.logprobs;
                let mut emitted = Vec::with_capacity(3);
                if !reasoning_text.is_empty() {
                    emitted.push(ChatChoiceStream {
                        index,
                        delta: chat_delta(None, None, None, Some(reasoning_text)),
                        finish_reason: None,
                        logprobs: remaining_logprobs.take(),
                    });
                }
                if !normal_text.is_empty() {
                    emitted.push(ChatChoiceStream {
                        index,
                        delta: chat_delta(Some(normal_text), None, None, None),
                        finish_reason: None,
                        logprobs: remaining_logprobs.take(),
                    });
                }

                if decoded.finish_reason.is_some() {
                    let (reasoning_tail, normal_tail) = choice.reasoning.finish();
                    if !reasoning_tail.is_empty() {
                        emitted.push(ChatChoiceStream {
                            index,
                            delta: chat_delta(None, None, None, Some(reasoning_tail)),
                            finish_reason: None,
                            logprobs: None,
                        });
                    }
                    if !normal_tail.is_empty() {
                        emitted.push(ChatChoiceStream {
                            index,
                            delta: chat_delta(Some(normal_tail), None, None, None),
                            finish_reason: None,
                            logprobs: None,
                        });
                    }
                }

                let finish_reason = decoded.finish_reason.map(to_dynamo_finish_reason);
                match emitted.last_mut() {
                    Some(last) => last.finish_reason = finish_reason,
                    None => emitted.push(ChatChoiceStream {
                        index,
                        delta: chat_delta(None, None, None, None),
                        finish_reason,
                        logprobs: remaining_logprobs,
                    }),
                }
                yield annotated_choices(emitted);
            }

            yield annotated_usage(prompt_tokens, completion_tokens);
        };

        let parsed: Pin<
            Box<dyn Stream<Item = Annotated<CreateChatCompletionStreamResponse>> + Send>,
        > = if let Some(parser) = self.tool_parser {
            Box::pin(apply_tool_calling_jail(
                Some(dynamo_parser_name(&parser).to_owned()),
                self.tool_choice,
                self.tools,
                self.uses_tool_call_structural_tag,
                raw,
            ))
        } else {
            Box::pin(raw)
        };
        let parallel_tool_calls = self.parallel_tool_calls;

        Box::pin(async_stream::stream! {
            let mut tool_calls_seen = vec![false; count];
            futures::pin_mut!(parsed);
            while let Some(mut item) = parsed.next().await {
                if let Some(response) = item.data.take() {
                    if response.choices.is_empty()
                        && let Some(usage) = response.usage
                    {
                        yield ChatOutputItem::Event(ChatEvent::Usage {
                            prompt_tokens: usage.prompt_tokens,
                            completion_tokens: u64::from(usage.completion_tokens),
                        });
                        continue;
                    }
                    for choice in response.choices {
                        let index = choice.index as usize;
                        let mut tool_calls = choice.delta.tool_calls.map(|calls| {
                            calls.into_iter().map(tool_call_delta).collect::<Vec<_>>()
                        });
                        if !parallel_tool_calls
                            && let Some(calls) = tool_calls.as_mut()
                            && let Some(seen) = tool_calls_seen.get_mut(index)
                        {
                            if *seen {
                                calls.clear();
                            } else {
                                calls.truncate(1);
                                *seen = !calls.is_empty();
                            }
                            if calls.is_empty() {
                                tool_calls = None;
                            }
                        }
                        let content = match choice.delta.content {
                            Some(ChatCompletionMessageContent::Text(text)) => Some(text),
                            _ => None,
                        };
                        if choice.delta.role.is_some()
                            && content.is_none()
                            && choice.delta.reasoning_content.is_none()
                            && tool_calls.is_none()
                            && choice.finish_reason.is_none()
                        {
                            yield ChatOutputItem::Event(ChatEvent::Role { choice: index });
                            continue;
                        }
                        yield ChatOutputItem::Event(ChatEvent::Delta {
                            choice: index,
                            content,
                            reasoning_content: choice.delta.reasoning_content,
                            tool_calls,
                            finish_reason: choice.finish_reason.map(from_dynamo_finish_reason),
                            logprobs: choice.logprobs,
                        });
                    }
                } else if let Some(error) = item.error {
                    let error = serde_json::from_str(&error).unwrap_or(ChatOutputError {
                        status_code: 500,
                        message: error,
                    });
                    yield ChatOutputItem::Error(error);
                }
            }
        })
    }
}

#[allow(deprecated)]
fn chat_delta(
    content: Option<String>,
    role: Option<Role>,
    tool_calls: Option<Vec<ChatCompletionMessageToolCallChunk>>,
    reasoning_content: Option<String>,
) -> ChatCompletionStreamResponseDelta {
    ChatCompletionStreamResponseDelta {
        content: content.map(ChatCompletionMessageContent::Text),
        function_call: None,
        tool_calls,
        role,
        refusal: None,
        reasoning_content,
    }
}

fn annotated_choices(
    choices: Vec<ChatChoiceStream>,
) -> Annotated<CreateChatCompletionStreamResponse> {
    Annotated {
        data: Some(CreateChatCompletionStreamResponse {
            id: String::new(),
            choices,
            created: 0,
            model: String::new(),
            service_tier: None,
            system_fingerprint: None,
            object: String::new(),
            usage: None,
        }),
        id: None,
        event: None,
        comment: None,
        error: None,
    }
}

fn annotated_usage(
    prompt_tokens: u32,
    completion_tokens: u64,
) -> Annotated<CreateChatCompletionStreamResponse> {
    Annotated {
        data: Some(CreateChatCompletionStreamResponse {
            id: String::new(),
            choices: Vec::new(),
            created: 0,
            model: String::new(),
            service_tier: None,
            system_fingerprint: None,
            object: String::new(),
            usage: Some(dynamo_protocols::types::CompletionUsage {
                prompt_tokens,
                completion_tokens: u32::try_from(completion_tokens).unwrap_or(u32::MAX),
                total_tokens: prompt_tokens
                    .saturating_add(u32::try_from(completion_tokens).unwrap_or(u32::MAX)),
                prompt_tokens_details: None,
                completion_tokens_details: None,
            }),
        }),
        id: None,
        event: None,
        comment: None,
        error: None,
    }
}

fn tool_call_delta(call: ChatCompletionMessageToolCallChunk) -> ChatToolCallDelta {
    ChatToolCallDelta {
        index: call.index,
        id: call.id,
        name: call
            .function
            .as_ref()
            .and_then(|function| function.name.clone()),
        arguments: call.function.and_then(|function| function.arguments),
    }
}

fn to_dynamo_finish_reason(reason: ChatFinishReason) -> FinishReason {
    match reason {
        ChatFinishReason::Stop => FinishReason::Stop,
        ChatFinishReason::Length => FinishReason::Length,
        ChatFinishReason::ContentFilter => FinishReason::ContentFilter,
        ChatFinishReason::ToolCalls => FinishReason::ToolCalls,
    }
}

fn from_dynamo_finish_reason(reason: FinishReason) -> ChatFinishReason {
    match reason {
        FinishReason::Stop => ChatFinishReason::Stop,
        FinishReason::Length => ChatFinishReason::Length,
        FinishReason::ContentFilter => ChatFinishReason::ContentFilter,
        FinishReason::ToolCalls | FinishReason::FunctionCall => ChatFinishReason::ToolCalls,
    }
}

async fn parse_chat_tool_calls(
    content: String,
    parser: Option<&str>,
    tools: Option<&[ToolDefinition]>,
    parallel_tool_calls: bool,
) -> (String, Option<Vec<ChatToolCall>>) {
    let Some(parser) = parser else {
        return (content, None);
    };
    let parser = dynamo_parser_name(parser);
    match try_tool_call_parse_aggregate_finalize(&content, Some(parser), tools).await {
        Ok((mut calls, normal)) if !calls.is_empty() => {
            if !parallel_tool_calls {
                calls.truncate(1);
            }
            (
                normal.unwrap_or_default(),
                Some(
                    calls
                        .into_iter()
                        .map(|call| ChatToolCall {
                            id: call.id,
                            name: call.function.name,
                            arguments: call.function.arguments,
                        })
                        .collect(),
                ),
            )
        }
        _ => (content, None),
    }
}

fn build_reasoning_parser(server_name: &str) -> ReasoningParserWrapper {
    let name = match server_name {
        "deepseek-r1" | "step3p5" => "deepseek_r1",
        "kimi_k2" => "kimi_k25",
        "gpt-oss" => "gpt_oss",
        "nemotron_3" => "nemotron3",
        "interns1" => "qwen3",
        "qwen3-thinking" | "minimax" => "deepseek_r1",
        _ => server_name,
    };
    ReasoningParserType::get_reasoning_parser_from_name(name)
}

fn split_reasoning_unary(name: Option<&str>, text: &str, token_ids: &[i32]) -> (String, String) {
    let Some(name) = name else {
        return (String::new(), text.to_owned());
    };
    let mut parser = build_reasoning_parser(name);
    let token_ids = token_ids
        .iter()
        .filter_map(|&id| u32::try_from(id).ok())
        .collect::<Vec<_>>();
    let split = parser.detect_and_parse_reasoning(text, &token_ids);
    (split.reasoning_text, split.normal_text)
}

struct ReasoningStreamSplitter {
    name: Option<String>,
    parser: Option<ReasoningParserWrapper>,
}

impl ReasoningStreamSplitter {
    fn new(name: Option<&str>) -> Self {
        Self {
            name: name.map(str::to_owned),
            parser: None,
        }
    }

    fn split(&mut self, text: &str, token_ids: &[i32]) -> (String, String) {
        let Some(name) = self.name.as_deref() else {
            return (String::new(), text.to_owned());
        };
        let parser = self
            .parser
            .get_or_insert_with(|| build_reasoning_parser(name));
        let token_ids = token_ids
            .iter()
            .filter_map(|&id| u32::try_from(id).ok())
            .collect::<Vec<_>>();
        let split = parser.parse_reasoning_streaming_incremental(text, &token_ids);
        (split.reasoning_text, split.normal_text)
    }

    fn finish(&mut self) -> (String, String) {
        let Some(parser) = self.parser.as_mut() else {
            return (String::new(), String::new());
        };
        let tail = parser.finish_reasoning_stream();
        (tail.reasoning_text, tail.normal_text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::stream;

    fn processor(
        tool_parser: Option<&str>,
        reasoning_parser: Option<&str>,
        choices: usize,
    ) -> ChatOutputProcessor {
        ChatOutputProcessor::new(
            tool_parser.map(str::to_owned),
            reasoning_parser.map(str::to_owned),
            None,
            Some(ChatCompletionToolChoiceOption::Auto),
            false,
            true,
            choices,
        )
    }

    fn chunk(choice: usize, text: &str, done: bool) -> ChatOutputInput {
        ChatOutputInput::Decoded(DecodedChatEvent {
            choice,
            text: text.into(),
            token_ids: vec![],
            finish_reason: done.then_some(ChatFinishReason::Stop),
            logprobs: None,
            prompt_tokens: 5,
            completion_tokens: 1,
        })
    }

    #[test]
    fn unary_processor_owns_reasoning_and_tool_interpretation() {
        let parsed = futures::executor::block_on(
            processor(Some("llama3"), Some("deepseek-r1"), 1).process_unary(
                r#"<think>check</think><|python_tag|>{"name":"weather","parameters":{}}"#.into(),
                &[],
            ),
        );
        assert_eq!(parsed.reasoning_content, "check");
        assert!(parsed.content.is_empty());
        assert_eq!(parsed.tool_calls.unwrap()[0].name, "weather");
    }

    #[test]
    fn streaming_processor_emits_semantics_without_wire_metadata() {
        let events = futures::executor::block_on(
            processor(None, Some("deepseek-r1"), 1)
                .process_stream(stream::iter(vec![
                    chunk(0, "<think>be", false),
                    chunk(0, "cause</think>Paris", true),
                ]))
                .collect::<Vec<_>>(),
        );
        let reasoning = events
            .iter()
            .filter_map(|event| match event {
                ChatOutputItem::Event(ChatEvent::Delta {
                    reasoning_content: Some(text),
                    ..
                }) => Some(text.as_str()),
                _ => None,
            })
            .collect::<String>();
        assert_eq!(reasoning, "because");
        assert!(events.iter().any(|event| matches!(
            event,
            ChatOutputItem::Event(ChatEvent::Delta {
                content: Some(text), ..
            }) if text == "Paris"
        )));
        assert!(matches!(
            events.last(),
            Some(ChatOutputItem::Event(ChatEvent::Usage {
                prompt_tokens: 5,
                completion_tokens: 2
            }))
        ));
    }

    #[test]
    fn each_choice_has_isolated_reasoning_state() {
        let events = futures::executor::block_on(
            processor(None, Some("deepseek-r1"), 2)
                .process_stream(stream::iter(vec![
                    chunk(0, "<think>zero", false),
                    chunk(1, "<think>one", false),
                    chunk(0, "</think>A", true),
                    chunk(1, "</think>B", true),
                ]))
                .collect::<Vec<_>>(),
        );
        let deltas = events.iter().filter_map(|event| match event {
            ChatOutputItem::Event(ChatEvent::Delta {
                choice,
                content: Some(content),
                ..
            }) => Some((*choice, content.as_str())),
            _ => None,
        });
        assert_eq!(deltas.collect::<Vec<_>>(), vec![(0, "A"), (1, "B")]);
    }
}
