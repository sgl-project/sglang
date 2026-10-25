//! Frontend-only parsing over a finished text, with the parsers the chat
//! completions route uses. No scheduler round-trip.

use dynamo_parsers::ToolDefinition;
use sglang_processor::{ReasoningOptions, parse_tool_calls, split_reasoning};

use crate::api_server::core::CoreError;

/// A tool the text may call, as the client described it.
#[derive(Debug, Clone, Default)]
pub(crate) struct ToolSpec {
    pub(crate) name: Option<String>,
    /// The arguments' JSON Schema, as JSON text.
    pub(crate) parameters: Option<String>,
}

/// Python's `ToolCallItem`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ParsedCall {
    /// The index of the called tool in the request's list, -1 when unknown.
    pub(crate) tool_index: i64,
    pub(crate) name: Option<String>,
    /// The arguments as a JSON string.
    pub(crate) parameters: String,
}

/// `(normal_text, calls)`: the text outside any tool call and the calls, in
/// order. A parser the frontend does not know is the client's error.
pub(crate) async fn parse_function_call(
    text: &str,
    tools: &[ToolSpec],
    tool_call_parser: Option<&str>,
) -> Result<(String, Vec<ParsedCall>), CoreError> {
    let Some(parser) = tool_call_parser.filter(|parser| !parser.is_empty()) else {
        return Err(CoreError::InvalidArgument(
            "missing field `tool_call_parser`".into(),
        ));
    };
    let definitions: Vec<ToolDefinition> = tools
        .iter()
        .map(|tool| ToolDefinition {
            name: tool.name.clone().unwrap_or_default(),
            parameters: tool
                .parameters
                .as_deref()
                .and_then(|schema| serde_json::from_str(schema).ok()),
            strict: None,
        })
        .collect();
    let (calls, normal) = parse_tool_calls(parser, text, Some(&definitions))
        .await
        .map_err(CoreError::InvalidArgument)?;
    let calls = calls
        .into_iter()
        .map(|call| ParsedCall {
            tool_index: tools
                .iter()
                .position(|tool| tool.name.as_deref() == Some(call.function.name.as_str()))
                .map_or(-1, |index| index as i64),
            name: Some(call.function.name),
            parameters: call.function.arguments,
        })
        .collect();
    Ok((normal.unwrap_or_default(), calls))
}

/// `(reasoning_text, normal_text)` for a finished text under `reasoning_parser`.
pub(crate) fn separate_reasoning(
    text: &str,
    reasoning_parser: &str,
) -> Result<(String, String), CoreError> {
    if reasoning_parser.is_empty() {
        return Err(CoreError::InvalidArgument(
            "missing field `reasoning_parser`".into(),
        ));
    }
    Ok(split_reasoning::<u32>(
        Some(reasoning_parser),
        &ReasoningOptions::default(),
        text,
        &[],
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api_server::core::CoreErrorKind;

    /// The call's `tool_index` is its position in the request's tools and its
    /// arguments stay a JSON string; an unknown parser is the client's 400.
    #[tokio::test]
    async fn parse_function_call_indexes_tools_and_rejects_unknown_parsers() {
        let tools = [
            ToolSpec {
                name: Some("get_time".into()),
                parameters: None,
            },
            ToolSpec {
                name: Some("get_weather".into()),
                parameters: Some(
                    r#"{"type":"object","properties":{"city":{"type":"string"}}}"#.into(),
                ),
            },
        ];
        let (normal, calls) = parse_function_call(
            r#"Sure. <tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>"#,
            &tools,
            Some("qwen25"),
        )
        .await
        .unwrap();
        assert_eq!(normal.trim(), "Sure.");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].tool_index, 1);
        assert_eq!(calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&calls[0].parameters).unwrap(),
            serde_json::json!({"city": "Paris"})
        );

        let (normal, calls) = parse_function_call("plain text", &tools, Some("qwen25"))
            .await
            .unwrap();
        assert_eq!((normal.as_str(), calls.len()), ("plain text", 0));

        for parser in [None, Some(""), Some("not-a-parser")] {
            let error = parse_function_call("x", &tools, parser).await.unwrap_err();
            assert_eq!(error.kind(), CoreErrorKind::InvalidArgument, "{parser:?}");
        }
    }

    #[test]
    fn separate_reasoning_splits_a_think_block() {
        let (reasoning, text) =
            separate_reasoning("<think>let me see</think>The answer is 4.", "deepseek-r1").unwrap();
        assert_eq!(reasoning.trim(), "let me see");
        assert_eq!(text.trim(), "The answer is 4.");
        assert_eq!(
            separate_reasoning("x", "").unwrap_err().kind(),
            CoreErrorKind::InvalidArgument
        );
    }
}
