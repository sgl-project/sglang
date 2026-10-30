//! DeepSeek-V4 / V4.1 output, which recipe parses from its token ids in place
//! of the reasoning and tool-call parsers (`DeepSeekRecipeChatEncodingAdapter`).

use std::collections::HashMap;

use serde_json::Value;
use sglang_api_types::api::v1::{FinishReason, GenerateMetaInfo, GenerateResponse, matched};

use super::{ChatModel, OpenAiSettings, Unsupported};
use crate::deepseek_output::select_new_output_ids;
use crate::tool_call::ToolCallItem;
use crate::{DeepSeekFinish, DeepSeekOutputOptions, DeepSeekOutputParser, DeepSeekStopMatch};

/// One request's parse: its options, and a parser per streamed choice.
pub(super) struct DeepSeekParsing {
    options: DeepSeekOutputOptions,
    parsers: HashMap<u64, DeepSeekOutputParser>,
}

/// How recipe parses `request`'s output, `None` when the server's parsers
/// are not the model's DeepSeek ones.
pub(super) fn parsing(
    model: &ChatModel,
    settings: &OpenAiSettings,
    request: &Value,
    has_tools: bool,
) -> Result<Option<DeepSeekParsing>, Unsupported> {
    let reasoning_parser = settings.reasoning_parser.as_deref();
    let tool_call_parser = settings.tool_call_parser.as_deref();
    let Some(output) = model.deepseek_output(reasoning_parser, tool_call_parser)? else {
        return Ok(None);
    };
    let defaults = settings
        .default_chat_template_kwargs
        .clone()
        .unwrap_or_default()
        .into_iter()
        .collect();
    let options = output
        .for_request(
            request,
            &defaults,
            reasoning_parser.is_some(),
            tool_call_parser.is_some() && has_tools,
        )
        .map_err(|_| Unsupported("invalid_request"))?;
    Ok(Some(DeepSeekParsing {
        options,
        parsers: HashMap::new(),
    }))
}

impl DeepSeekParsing {
    /// One choice's whole output as `(reasoning, content, tool calls)`.
    pub(super) fn unary(&self, item: &GenerateResponse) -> (String, String, Vec<ToolCallItem>) {
        let finish = item
            .meta_info
            .as_ref()
            .and_then(|meta| meta.finish_reason.as_ref());
        let delta = self
            .options
            .parser()
            .finish(output_ids(item), deepseek_finish(finish));
        let calls = delta
            .whole_tool_calls()
            .into_iter()
            .map(|call| ToolCallItem {
                tool_index: call.index as i64,
                name: call.name,
                parameters: call.arguments,
            });
        (delta.reasoning, delta.content, calls.collect())
    }

    /// `stream_chat_chunks`: what one frame's new ids add to choice `index`, as
    /// `(reasoning, content, tool-call pieces)`, a piece's name empty after its
    /// call's first; and whether the choice has called a tool.
    pub(super) fn stream(
        &mut self,
        index: u64,
        item: &GenerateResponse,
        meta: &GenerateMetaInfo,
        finish_type: Option<&str>,
        incremental: bool,
    ) -> ((String, String, Vec<ToolCallItem>), bool) {
        let parser = self
            .parsers
            .entry(index)
            .or_insert_with(|| self.options.parser());
        let ids = select_new_output_ids(
            output_ids(item),
            parser.num_consumed_tokens(),
            meta.completion_tokens as usize,
            finish_type == Some("abort"),
            incremental,
        );
        let finish = finish_type.map(|_| deepseek_finish(meta.finish_reason.as_ref()));
        let delta = parser.step(ids, finish);
        let pieces = delta.tool_calls.into_iter().map(|piece| ToolCallItem {
            tool_index: piece.index as i64,
            name: piece.name.unwrap_or_default(),
            parameters: piece.arguments,
        });
        let parsed = (delta.reasoning, delta.content, pieces.collect());
        (parsed, parser.has_tool_calls())
    }
}

fn output_ids(item: &GenerateResponse) -> &[i64] {
    item.output_ids.as_ref().map_or(&[], |ids| &ids.ids)
}

/// The engine's finish reason as recipe's parser takes it; a type other than
/// `stop` and `length` ends the parse without a backend finish.
fn deepseek_finish(reason: Option<&FinishReason>) -> DeepSeekFinish<'_> {
    let Some(reason) = reason else {
        return DeepSeekFinish::Abort;
    };
    match reason.kind_name().as_deref() {
        Some("stop") => DeepSeekFinish::Stop(match reason.matched() {
            Some(matched::Value::Token(id)) => Some(DeepSeekStopMatch::Token(*id)),
            Some(matched::Value::Str(stop)) => Some(DeepSeekStopMatch::Text(stop)),
            _ => None,
        }),
        Some("length") => DeepSeekFinish::Length,
        _ => DeepSeekFinish::Abort,
    }
}
