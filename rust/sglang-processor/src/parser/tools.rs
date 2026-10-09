//! The tool schemas shared by output parsing and constrained generation.

use dynamo_parsers::ToolDefinition;
use dynamo_protocols::types::{ChatCompletionRequestMessage, ChatCompletionTool, FunctionObject};

use crate::ProcessorError;

/// Collect top-level and dynamic system-message tools without moving their
/// declarations in the prompt. Dynamic tools accept wrapped OpenAI declarations
/// or bare function schemas, as Dynamo's Kimi formatter does.
pub fn chat_tool_definitions(
    tools: Option<&[ChatCompletionTool]>,
    messages: &[ChatCompletionRequestMessage],
) -> Result<Vec<ToolDefinition>, ProcessorError> {
    let mut definitions: Vec<_> = tools
        .into_iter()
        .flatten()
        .map(|tool| definition(tool.function.clone()))
        .collect();
    for message in messages {
        let ChatCompletionRequestMessage::System(system) = message else {
            continue;
        };
        for tool in system.tools.iter().flatten() {
            let function = if tool.get("type").is_some() || tool.get("function").is_some() {
                serde_json::from_value::<ChatCompletionTool>(tool.clone()).map(|tool| tool.function)
            } else {
                serde_json::from_value::<FunctionObject>(tool.clone())
            }
            .map_err(|error| {
                ProcessorError::InvalidRequest(format!("invalid system-message tool: {error}"))
            })?;
            definitions.push(definition(function));
        }
    }
    Ok(definitions)
}

fn definition(function: FunctionObject) -> ToolDefinition {
    ToolDefinition {
        name: function.name,
        parameters: function.parameters,
        strict: function.strict,
    }
}
