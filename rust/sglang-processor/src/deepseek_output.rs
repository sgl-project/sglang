//! DeepSeek-V4 / V4.1 output through deepseek-recipe's `StreamProcessor`, fed
//! the engine's token ids, as `serving_deepseek_recipe.py` and
//! `parser/deepseek_recipe_output.py` parse it: `</think>` and the DSML tags are
//! recognized by token id, and tool-call arguments arrive as recipe streams them.

use std::collections::{BTreeMap, HashMap};
use std::pin::Pin;
use std::sync::{Arc, Mutex, PoisonError};
use std::task::{Context, Poll, Waker};

use deepseek_recipe::stream::{
    ChunkGenerator, InferenceChunk, InferenceFinishReason, OutputChunk, ParsingOptions,
    StreamError, StreamProcessor, ThinkingInitialStage,
};
use deepseek_recipe_core::special_tokens::SpecialTokensMap;
use futures::Stream;
use futures::channel::mpsc::{UnboundedSender, unbounded};
use serde_json::Value;
use tokenizers::Tokenizer;

use crate::render::{
    DeepSeekTokenizer, deepseek_v4_continued_message, deepseek_v4_parse_start,
    deepseek_v4_special_tokens, deepseek_v4_task, deepseek_v4_thinking, pydantic_bool,
};

/// A served DeepSeek-V4 or V4.1 checkpoint whose output recipe parses.
#[derive(Debug, Clone)]
pub struct DeepSeekOutput {
    v41: bool,
    tokenizer: Arc<DeepSeekTokenizer>,
    special_tokens: Arc<SpecialTokensMap>,
}

/// How one request's output is parsed (`_new_output_parser`).
#[derive(Debug, Clone)]
pub struct DeepSeekOutputOptions {
    tokenizer: Arc<Tokenizer>,
    special_tokens: Arc<SpecialTokensMap>,
    thinking_stage: Option<ThinkingInitialStage>,
    parse_tool_calls: bool,
    parse_json_output: bool,
    separate_reasoning: bool,
    stream_reasoning: bool,
    keep_matched_stop: bool,
    // Ids a stop string may still start in wait until the scheduler rules it
    // out, so a matched stop is cut before recipe has emitted any of it.
    stop_hold_chars: usize,
}

impl DeepSeekOutput {
    pub(crate) fn new(v41: bool, tokenizer: Arc<DeepSeekTokenizer>) -> Self {
        Self {
            v41,
            tokenizer,
            special_tokens: Arc::new(deepseek_v4_special_tokens()),
        }
    }

    /// The parsing of `request`, a chat body, under the server's default
    /// template kwargs. `separate_reasoning` is whether the server has a
    /// reasoning parser, and `parse_tool_calls` whether the request's output
    /// runs through the tool-call parser (`_tool_call_parsing_active`).
    pub fn for_request(
        &self,
        request: &Value,
        defaults: &HashMap<String, Value>,
        separate_reasoning: bool,
        parse_tool_calls: bool,
    ) -> Result<DeepSeekOutputOptions, String> {
        let flag = |name: &str| request.get(name).map_or(Ok(true), pydantic_bool);
        let task = deepseek_v4_task(request)?;
        let continued = deepseek_v4_continued_message(request)?;
        let thinking_stage = deepseek_v4_parse_start(
            self.v41,
            deepseek_v4_thinking(request, defaults),
            task,
            continued.as_ref(),
        )?;
        let stops = match &request["stop"] {
            stop @ Value::String(_) => std::slice::from_ref(stop),
            Value::Array(stops) => stops.as_slice(),
            _ => &[],
        };
        let longest_stop = stops
            .iter()
            .filter_map(|stop| stop.as_str().map(|stop| stop.chars().count()))
            .max();
        Ok(DeepSeekOutputOptions {
            tokenizer: self.tokenizer.full().clone(),
            special_tokens: self.special_tokens.clone(),
            thinking_stage,
            parse_tool_calls: task.is_none() && parse_tool_calls,
            parse_json_output: matches!(
                request["response_format"]["type"].as_str(),
                Some("json_object" | "json_schema")
            ),
            separate_reasoning: separate_reasoning && flag("separate_reasoning")?,
            stream_reasoning: flag("stream_reasoning")?,
            keep_matched_stop: request
                .get("no_stop_trim")
                .map_or(Ok(false), pydantic_bool)?,
            stop_hold_chars: longest_stop.unwrap_or(1).saturating_sub(1),
        })
    }
}

impl DeepSeekOutputOptions {
    /// The parser for one choice.
    pub fn parser(&self) -> DeepSeekOutputParser {
        DeepSeekOutputParser::new(self)
    }
}

/// A piece of one tool call's JSON arguments; the first piece also names the
/// call (`RecipeToolCallDelta`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepSeekToolCallDelta {
    pub index: usize,
    pub name: Option<String>,
    pub arguments: String,
}

/// The reasoning, content and tool-call pieces one step of output adds
/// (`RecipeOutputDelta`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DeepSeekDelta {
    pub reasoning: String,
    pub content: String,
    pub tool_calls: Vec<DeepSeekToolCallDelta>,
}

/// A whole tool call, its pieces joined (`RecipeToolCall`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepSeekToolCall {
    pub index: usize,
    pub name: String,
    pub arguments: String,
}

impl DeepSeekDelta {
    /// `whole_tool_calls`: the calls the pieces add up to, by index.
    pub fn whole_tool_calls(&self) -> Vec<DeepSeekToolCall> {
        let mut calls: BTreeMap<usize, (String, String)> = BTreeMap::new();
        for piece in &self.tool_calls {
            let call = calls.entry(piece.index).or_default();
            if let Some(name) = &piece.name {
                *call = (name.clone(), String::new());
            }
            call.1.push_str(&piece.arguments);
        }
        calls
            .into_iter()
            .map(|(index, (name, arguments))| DeepSeekToolCall {
                index,
                name,
                arguments,
            })
            .collect()
    }
}

/// How the engine ended the output.
#[derive(Debug, Clone, Copy)]
pub enum DeepSeekFinish<'a> {
    Stop(Option<DeepSeekStopMatch<'a>>),
    Length,
    Abort,
}

/// The stop the scheduler matched.
#[derive(Debug, Clone, Copy)]
pub enum DeepSeekStopMatch<'a> {
    Token(i64),
    Text(&'a str),
}

type OutputStream = Pin<Box<dyn Stream<Item = Result<OutputChunk, StreamError>> + Send>>;

/// One choice's output through recipe's `StreamProcessor`
/// (`DeepSeekRecipeOutputParser`). Stops stay SGLang's: the scheduler matches
/// them and [`Self::finish`] cuts the match.
pub struct DeepSeekOutputParser {
    input: Option<UnboundedSender<InferenceChunk>>,
    // sgl-router keeps parsers in `Sync` state; reached only via `get_mut`, so it
    // never locks.
    output: Mutex<OutputStream>,
    finished: bool,
    tokenizer: Arc<Tokenizer>,
    reasoning_as_content: bool,
    hold_reasoning: bool,
    held_reasoning: String,
    keep_matched_stop: bool,
    stop_hold_chars: usize,
    pending_ids: Vec<u32>,
    next_tool_call_index: usize,
    num_consumed_tokens: usize,
}

/// recipe's parsed chunks as they are.
struct OutputChunks;

impl ChunkGenerator for OutputChunks {
    type Chunk = OutputChunk;

    async fn generate(&mut self, chunk: OutputChunk) -> Vec<OutputChunk> {
        vec![chunk]
    }
}

impl DeepSeekOutputParser {
    fn new(options: &DeepSeekOutputOptions) -> Self {
        let parsing = ParsingOptions {
            parse_tool_calls: options.parse_tool_calls,
            parse_json_output: options.parse_json_output,
            thinking_initial_stage: options.thinking_stage,
            special_tokens: options.special_tokens.clone(),
            ..ParsingOptions::default()
        };
        let (input, inference) = unbounded();
        let output = StreamProcessor::new(OutputChunks, parsing)
            .with_tokenizer(options.tokenizer.clone())
            .into_protocol_stream(inference);
        Self {
            input: Some(input),
            output: Mutex::new(Box::pin(output)),
            finished: false,
            tokenizer: options.tokenizer.clone(),
            reasoning_as_content: !options.separate_reasoning,
            hold_reasoning: options.separate_reasoning && !options.stream_reasoning,
            held_reasoning: String::new(),
            keep_matched_stop: options.keep_matched_stop,
            stop_hold_chars: options.stop_hold_chars,
            pending_ids: Vec::new(),
            next_tool_call_index: 0,
            num_consumed_tokens: 0,
        }
    }

    /// The output token ids fed so far, held ones included.
    pub(crate) fn num_consumed_tokens(&self) -> usize {
        self.num_consumed_tokens
    }

    /// Whether the parse has named a tool call.
    pub fn has_tool_calls(&self) -> bool {
        self.next_tool_call_index > 0
    }

    /// One engine chunk's new ids, with its finish when it is the last.
    pub fn step(&mut self, token_ids: &[i64], finish: Option<DeepSeekFinish<'_>>) -> DeepSeekDelta {
        match finish {
            None => self.feed(token_ids),
            Some(finish) => self.finish(token_ids, finish),
        }
    }

    /// One intermediate engine chunk's new ids.
    fn feed(&mut self, token_ids: &[i64]) -> DeepSeekDelta {
        self.num_consumed_tokens += token_ids.len();
        if self.finished {
            return DeepSeekDelta::default();
        }
        self.pending_ids.extend(vocab_ids(token_ids));
        let ready = self.release_pending_ids();
        if ready.is_empty() {
            return DeepSeekDelta::default();
        }
        let chunks = self.push(InferenceChunk::Tokens {
            token_ids: ready,
            logprobs: None,
        });
        self.deltas(chunks)
    }

    fn release_pending_ids(&mut self) -> Vec<u32> {
        if self.stop_hold_chars == 0 {
            return std::mem::take(&mut self.pending_ids);
        }
        let mut held = self.pending_ids.len();
        while held > 0
            && self.stop_hold_chars > self.decode(&self.pending_ids[held..]).chars().count()
        {
            held -= 1;
        }
        let pending = self.pending_ids.split_off(held);
        std::mem::replace(&mut self.pending_ids, pending)
    }

    /// The last engine chunk: its ids, less the matched stop, and its reason.
    pub fn finish(&mut self, token_ids: &[i64], finish: DeepSeekFinish<'_>) -> DeepSeekDelta {
        self.num_consumed_tokens += token_ids.len();
        if self.finished {
            return DeepSeekDelta::default();
        }
        let mut ids = std::mem::take(&mut self.pending_ids);
        ids.extend(vocab_ids(token_ids));
        match finish {
            DeepSeekFinish::Stop(Some(DeepSeekStopMatch::Text(stop))) => {
                ids = self.cut_at_stop_string(ids, stop);
            }
            DeepSeekFinish::Stop(Some(DeepSeekStopMatch::Token(id)))
                if !self.keep_matched_stop
                    && ids.last().is_some_and(|&last| i64::from(last) == id) =>
            {
                ids.pop();
            }
            _ => {}
        }
        let mut chunks = Vec::new();
        if !ids.is_empty() {
            chunks = self.push(InferenceChunk::Tokens {
                token_ids: ids,
                logprobs: None,
            });
        }
        if !self.finished {
            let reason = match finish {
                DeepSeekFinish::Stop(_) => Some(InferenceFinishReason::Stop),
                DeepSeekFinish::Length => Some(InferenceFinishReason::Length),
                DeepSeekFinish::Abort => None,
            };
            chunks.extend(match reason {
                Some(finish_reason) => self.push(InferenceChunk::Finish { finish_reason }),
                // An abort flushes the parse without a backend finish.
                None => {
                    self.input = None;
                    self.pull()
                }
            });
        }
        debug_assert!(self.finished, "recipe ends its stream at the finish");
        self.deltas(chunks)
    }

    /// The ids up to the stop string the scheduler matched: it ended on the
    /// token completing `stop`, and the earliest match ending there starts at
    /// most `stop.chars().count() - 1` characters before that token. A stop not
    /// found there, as a `stop_regex` pattern is not, is not cut.
    fn cut_at_stop_string(&self, ids: Vec<u32>, stop: &str) -> Vec<u32> {
        let Some(mut start) = ids.len().checked_sub(1) else {
            return ids;
        };
        let mut text = self.decode(&ids[start..]);
        let reach = text.chars().count() + stop.chars().count().saturating_sub(1);
        let mut tokens_past_reach = 0;
        while start > 0 && !(text.contains(stop) && text.chars().count() >= reach) {
            if text.chars().count() >= reach {
                // A character split across tokens decodes whole only with all of them.
                if tokens_past_reach == MAX_UTF8_BYTES {
                    break;
                }
                tokens_past_reach += 1;
            }
            start -= 1;
            text = self.decode(&ids[start..]);
        }
        let Some(mut cut) = text.find(stop) else {
            return ids;
        };
        if self.keep_matched_stop {
            cut += stop.len();
        }
        let mut out = ids[..start].to_vec();
        match self.tokenizer.encode_fast(&text[..cut], false) {
            Ok(encoding) => out.extend_from_slice(encoding.get_ids()),
            Err(error) => {
                tracing::warn!(%error, "deepseek-recipe output: stop cut failed to encode")
            }
        }
        out
    }

    fn decode(&self, ids: &[u32]) -> String {
        self.tokenizer.decode(ids, false).unwrap_or_else(|error| {
            tracing::warn!(%error, "deepseek-recipe output: decode failed");
            String::new()
        })
    }

    fn push(&mut self, chunk: InferenceChunk) -> Vec<OutputChunk> {
        if let Some(input) = &self.input {
            // The receiver lives as long as the output stream, which is unfinished.
            _ = input.unbounded_send(chunk);
        }
        self.pull()
    }

    /// The chunks recipe emits for what it was sent, up to where it waits for
    /// more. recipe's stream waits only on its input, which `push` has filled,
    /// so a noop waker drives it.
    fn pull(&mut self) -> Vec<OutputChunk> {
        let output = self
            .output
            .get_mut()
            .unwrap_or_else(PoisonError::into_inner);
        let mut context = Context::from_waker(Waker::noop());
        let mut chunks = Vec::new();
        while !self.finished {
            match output.as_mut().poll_next(&mut context) {
                Poll::Ready(Some(Ok(chunk))) => chunks.push(chunk),
                Poll::Ready(Some(Err(error))) => {
                    tracing::warn!(%error, "deepseek-recipe output parsing failed");
                    self.finished = true;
                }
                Poll::Ready(None) => self.finished = true,
                Poll::Pending => break,
            }
        }
        chunks
    }

    /// `_deltas`: what recipe's chunks add, as its Chat Completions chunks
    /// carry them.
    fn deltas(&mut self, chunks: Vec<OutputChunk>) -> DeepSeekDelta {
        let mut delta = DeepSeekDelta::default();
        for chunk in chunks {
            let (mut reasoning, mut content, mut tool_call, mut finish) =
                (String::new(), String::new(), None, false);
            match chunk {
                // The handler emits its own role chunk.
                OutputChunk::Start { .. } | OutputChunk::ToolCallBegin => continue,
                OutputChunk::Content { content: text, .. } => content = text,
                OutputChunk::Reasoning { content: text, .. } if self.reasoning_as_content => {
                    content = text
                }
                OutputChunk::Reasoning { content: text, .. } => reasoning = text,
                OutputChunk::ToolCallNamed { tool_name } => {
                    tool_call = Some(DeepSeekToolCallDelta {
                        index: self.next_tool_call_index,
                        name: Some(tool_name),
                        arguments: String::new(),
                    });
                    self.next_tool_call_index += 1;
                }
                OutputChunk::ToolCallArgumentsDelta { arguments } => {
                    // recipe names a call before its arguments.
                    let Some(index) = self.next_tool_call_index.checked_sub(1) else {
                        continue;
                    };
                    tool_call = Some(DeepSeekToolCallDelta {
                        index,
                        name: None,
                        arguments,
                    });
                }
                OutputChunk::Finish { .. } => finish = true,
            }
            if self.hold_reasoning {
                self.held_reasoning
                    .push_str(&std::mem::take(&mut reasoning));
                if !content.is_empty() || tool_call.is_some() || finish {
                    reasoning = std::mem::take(&mut self.held_reasoning);
                }
            }
            delta.reasoning.push_str(&reasoning);
            delta.content.push_str(&content);
            delta.tool_calls.extend(tool_call);
        }
        delta
    }
}

/// The most tokens one UTF-8 character can span, a byte each.
const MAX_UTF8_BYTES: usize = 4;

/// The engine's output ids as the tokenizer's vocabulary ids.
fn vocab_ids(ids: &[i64]) -> impl Iterator<Item = u32> + '_ {
    ids.iter().filter_map(|&id| u32::try_from(id).ok())
}

/// `select_new_output_ids`: the ids a `/generate` chunk adds. Intermediate
/// cumulative chunks share the live output list, so `completion_tokens` bounds
/// this chunk's snapshot; an incremental abort chunk re-sends streamed ids.
pub(crate) fn select_new_output_ids(
    output_ids: &[i64],
    num_consumed_tokens: usize,
    completion_tokens: usize,
    abort: bool,
    incremental: bool,
) -> &[i64] {
    if !incremental {
        let end = completion_tokens.min(output_ids.len());
        return output_ids.get(num_consumed_tokens..end).unwrap_or_default();
    }
    if abort {
        let count = completion_tokens.saturating_sub(num_consumed_tokens);
        return &output_ids[..count.min(output_ids.len())];
    }
    output_ids
}
