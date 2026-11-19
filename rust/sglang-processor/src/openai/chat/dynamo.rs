//! Dynamo's reasoning and tool parsers, for the `--reasoning-parser` and
//! `--tool-call-parser` names `think/` and `tool_call/` do not port.

use std::sync::{Mutex, PoisonError};
use std::task::{Context, Poll};

use dynamo_parsers::parsers::get_tool_parser_map;
use dynamo_parsers::tool_calling::jail::{Annotated, apply_tool_calling_jail};
use dynamo_parsers::{ToolChoice, ToolDefinition, try_tool_call_parse_aggregate_finalize};
use dynamo_protocols::types::{
    ChatChoiceStream, ChatCompletionMessageContent, ChatCompletionStreamResponseDelta,
    ChatCompletionToolChoiceOption, CreateChatCompletionStreamResponse, FinishReason,
};
use futures::StreamExt;
use futures::channel::mpsc::{UnboundedSender, unbounded};
use futures::stream::BoxStream;
use serde_json::Value;

use super::{ToolParsing, Unsupported};
use crate::parser::{
    ReasoningStreamSplitter, ToolConstraint, dynamo_tool_parser_name, split_reasoning,
};
use crate::think::ReasoningOptions;
use crate::tool_call::{Parsed, ToolCallItem, ToolDetector};

/// Any reasoning parser name is served: one Dynamo does not know is its basic parser.
pub(super) const PARSES_REASONING: bool = true;

type Chunk = Annotated<CreateChatCompletionStreamResponse>;

/// A whole output as `(reasoning, normal)`.
pub(super) fn parse_reasoning(
    name: &str,
    options: &ReasoningOptions,
    text: &str,
    ids: &[i64],
) -> (String, String) {
    split_reasoning(Some(name), options, text, ids)
}

/// One streamed choice's reasoning split.
pub(super) struct Reasoning(ReasoningStreamSplitter);

impl Reasoning {
    pub(super) fn new(name: &str, options: &ReasoningOptions) -> Self {
        Self(ReasoningStreamSplitter::new(Some(name), options.clone()))
    }

    pub(super) fn push(&mut self, text: &str) -> (String, String) {
        self.0.split::<u32>(text, &[])
    }

    pub(super) fn finish(&mut self) -> (String, String) {
        self.0.finish()
    }
}

/// A forced `tool_choice` and strict tools are served.
pub(super) const CONSTRAINS_TOOLS: bool = true;

fn definitions(tools: &[Value]) -> Vec<ToolDefinition> {
    let definition = |tool: &Value| {
        let function = &tool["function"];
        Some(ToolDefinition {
            name: function["name"].as_str()?.to_owned(),
            parameters: function.get("parameters").cloned(),
            strict: function["strict"].as_bool(),
        })
    };
    tools.iter().filter_map(definition).collect()
}

/// The grammar `choice` holds the output to under `parser`, as
/// `(is a structural tag, grammar)`: a structural tag, else a JSON schema.
pub(super) fn tool_constraint(
    parser: &str,
    choice: Option<&Value>,
    tools: &[Value],
    parallel_tool_calls: Option<bool>,
) -> Result<Option<(bool, String)>, Unsupported> {
    let choice = match choice {
        None => ToolChoice::Auto,
        Some(Value::String(choice)) if choice == "required" => ToolChoice::Required,
        Some(choice) => match choice["function"]["name"].as_str() {
            Some(name) => ToolChoice::Named(name.to_owned()),
            None => return Err(Unsupported("tool_choice")),
        },
    };
    let tools = definitions(tools);
    let constraint =
        crate::parser::tool_constraint(Some(parser), &choice, &tools, parallel_tool_calls)
            .map_err(|_| Unsupported("invalid_request"))?;
    Ok(constraint.map(|constraint| match constraint {
        ToolConstraint::StructuralTag(tag) => (true, tag),
        ToolConstraint::JsonSchema(schema) => (false, schema),
    }))
}

/// The detector for a tool parser Dynamo knows.
pub(super) fn tool_detector(parsing: &ToolParsing) -> Option<Box<dyn ToolDetector>> {
    let parser = dynamo_tool_parser_name(&parsing.parser);
    if !get_tool_parser_map().contains_key(parser) {
        return None;
    }
    let (choice, structural_tag) = match &parsing.forced {
        Some((choice, structural_tag)) => {
            (serde_json::from_value(choice.clone()).ok(), *structural_tag)
        }
        None => (None, false),
    };
    Some(Box::new(DynamoTools {
        parser: parser.to_owned(),
        tools: definitions(&parsing.tools),
        json_calls: parsing.forced.is_some() && !structural_tag,
        choice,
        structural_tag,
        jail: None,
    }))
}

struct DynamoTools {
    parser: String,
    tools: Vec<ToolDefinition>,
    /// A forced choice under a JSON schema: the output is the calls as JSON.
    json_calls: bool,
    choice: Option<ChatCompletionToolChoiceOption>,
    structural_tag: bool,
    /// The streaming jail, fed one chunk per delta.
    jail: Option<Jail>,
}

/// `[{"name": ..., "parameters": ...}]`, the output of a JSON-schema constraint.
fn json_calls(text: &str) -> Option<Vec<ToolCallItem>> {
    let calls: Vec<Value> = serde_json::from_str(text).ok()?;
    let call = |(call, tool_index): (Value, i64)| {
        Some(ToolCallItem {
            tool_index,
            name: call["name"].as_str()?.to_owned(),
            parameters: call.get("parameters")?.to_string(),
        })
    };
    calls.into_iter().zip(0..).map(call).collect()
}

struct Jail {
    chunks: UnboundedSender<Chunk>,
    // Only reached through `get_mut`, so it never locks.
    parsed: Mutex<BoxStream<'static, Chunk>>,
}

fn chunk(text: &str, finish_reason: Option<FinishReason>) -> Chunk {
    #[allow(deprecated)]
    let delta = ChatCompletionStreamResponseDelta {
        content: Some(ChatCompletionMessageContent::Text(text.to_owned())),
        function_call: None,
        tool_calls: None,
        role: None,
        refusal: None,
        reasoning_content: None,
    };
    Annotated {
        data: Some(CreateChatCompletionStreamResponse {
            id: String::new(),
            choices: vec![ChatChoiceStream {
                index: 0,
                delta,
                finish_reason,
                logprobs: None,
            }],
            created: 0,
            model: String::new(),
            service_tier: None,
            system_fingerprint: None,
            object: "chat.completion.chunk".into(),
            usage: None,
        }),
        id: None,
        event: None,
        comment: None,
        error: None,
    }
}

impl DynamoTools {
    fn jail(&mut self) -> &mut Jail {
        self.jail.get_or_insert_with(|| {
            let (chunks, received) = unbounded();
            let parsed = apply_tool_calling_jail(
                Some(self.parser.clone()),
                self.choice.clone(),
                Some(self.tools.clone()),
                self.structural_tag,
                received,
            );
            Jail {
                chunks,
                parsed: Mutex::new(parsed.boxed()),
            }
        })
    }
}

impl ToolDetector for DynamoTools {
    /// Dynamo has no cheap probe, so every output is parsed.
    fn has_tool_call(&self, _text: &str) -> bool {
        true
    }

    fn parse_non_stream(&self, text: &str) -> Parsed {
        if self.json_calls
            && let Some(calls) = json_calls(text)
        {
            return (String::new(), calls);
        }
        // The parse only computes, so it finishes without a runtime.
        let parsed = futures::executor::block_on(try_tool_call_parse_aggregate_finalize(
            text,
            Some(&self.parser),
            Some(&self.tools),
        ));
        let Ok((calls, normal)) = parsed else {
            return (text.to_owned(), Vec::new());
        };
        let calls = calls.into_iter().zip(0..).map(|(call, tool_index)| {
            let function = call.function;
            ToolCallItem {
                tool_index,
                name: function.name,
                parameters: function.arguments,
            }
        });
        (normal.unwrap_or_default(), calls.collect())
    }

    fn parse_stream(&mut self, text: &str, flush: bool) -> Parsed {
        let jail = self.jail();
        let finish_reason = flush.then_some(FinishReason::Stop);
        // The receiver lives as long as the jail.
        let _ = jail.chunks.unbounded_send(chunk(text, finish_reason));
        if flush {
            jail.chunks.close_channel();
        }
        let parsed = jail
            .parsed
            .get_mut()
            .unwrap_or_else(PoisonError::into_inner);
        let mut context = Context::from_waker(futures::task::noop_waker_ref());
        let (mut normal, mut calls) = (String::new(), Vec::new());
        // The jail waits on nothing but its input, so pending means it is drained.
        while let Poll::Ready(Some(item)) = parsed.as_mut().poll_next(&mut context) {
            for choice in item.data.into_iter().flat_map(|chunk| chunk.choices) {
                if let Some(ChatCompletionMessageContent::Text(text)) = choice.delta.content {
                    normal.push_str(&text);
                }
                for call in choice.delta.tool_calls.into_iter().flatten() {
                    let function = call.function.unwrap_or_default();
                    calls.push(ToolCallItem {
                        tool_index: i64::from(call.index),
                        name: function.name.unwrap_or_default(),
                        parameters: function.arguments.unwrap_or_default(),
                    });
                }
            }
        }
        (normal, calls)
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn weather_tools() -> Vec<Value> {
        vec![json!({"type": "function", "function": {
            "name": "get_weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        }})]
    }

    const CALL: &str = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>";

    #[test]
    fn unported_reasoning_parser_splits_whole_and_streamed() {
        let options = ReasoningOptions::default();
        assert_eq!(
            parse_reasoning("deepseek-r1", &options, "<think>why</think>Paris", &[]),
            ("why".into(), "Paris".into())
        );
        let mut reasoning = Reasoning::new("deepseek-r1", &options);
        let mut columns = (String::new(), String::new());
        for delta in ["<think>wh", "y</think>Par", "is"] {
            let (thought, normal) = reasoning.push(delta);
            columns.0 += &thought;
            columns.1 += &normal;
        }
        let (thought, normal) = reasoning.finish();
        assert_eq!(
            (columns.0 + &thought, columns.1 + &normal),
            ("why".into(), "Paris".into())
        );
    }

    fn detector(parser: &str, forced: Option<(Value, bool)>) -> Option<Box<dyn ToolDetector>> {
        tool_detector(&ToolParsing {
            parser: parser.into(),
            tools: weather_tools(),
            forced,
        })
    }

    #[test]
    fn unknown_tool_parser_has_no_detector() {
        assert!(detector("no-such-parser", None).is_none());
    }

    #[test]
    fn forced_choice_is_constrained_and_read_as_json() {
        let tools = weather_tools();
        let required = json!("required");
        // Hermes has a structural tag, which keeps the parser's own format.
        let (structural, tag) = tool_constraint("hermes", Some(&required), &tools, None)
            .unwrap()
            .unwrap();
        assert!(structural && tag.contains("get_weather"), "{tag}");
        let missing = json!({"type": "function", "function": {"name": "nope"}});
        assert_eq!(
            tool_constraint("hermes", Some(&missing), &tools, None),
            Err(Unsupported("invalid_request"))
        );

        let output = r#"[{"name": "get_weather", "parameters": {"city": "Paris"}}]"#;
        let (normal, calls) = detector("hermes", Some((required, false)))
            .unwrap()
            .parse_non_stream(output);
        assert_eq!(normal, "");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].parameters, r#"{"city":"Paris"}"#);
    }

    #[test]
    fn unported_tool_parser_parses_whole_output() {
        let detector = detector("hermes", None).unwrap();
        let (normal, calls) = detector.parse_non_stream(CALL);
        assert_eq!(normal.trim(), "");
        assert_eq!(calls.len(), 1, "{calls:?}");
        assert_eq!(calls[0].name, "get_weather");
        let arguments: Value = serde_json::from_str(&calls[0].parameters).unwrap();
        assert_eq!(arguments, json!({"city": "Paris"}));
        let (normal, calls) = detector.parse_non_stream("no call here");
        assert_eq!((normal.as_str(), calls.len()), ("no call here", 0));
    }

    #[test]
    fn unported_tool_parser_parses_a_stream() {
        let mut detector = detector("hermes", None).unwrap();
        let (mut normal, mut calls) = (String::new(), Vec::new());
        let (head, tail) = CALL.split_at(30);
        for (delta, flush) in [("Sure. ", false), (head, false), (tail, true)] {
            let (text, parsed) = detector.parse_stream(delta, flush);
            normal += &text;
            calls.extend(parsed);
        }
        assert_eq!(normal.trim(), "Sure.");
        let name: String = calls.iter().map(|call| call.name.as_str()).collect();
        let arguments: String = calls.iter().map(|call| call.parameters.as_str()).collect();
        assert_eq!(name, "get_weather");
        let arguments: Value = serde_json::from_str(&arguments).unwrap();
        assert_eq!(arguments, json!({"city": "Paris"}));
    }
}
