//! Transport-neutral pieces of the OpenAI legacy completions endpoint.

use std::collections::BTreeMap;

use futures::StreamExt;

use crate::api_server::core::error::ApiError;
use crate::api_server::core::event::CoreEvent;
use crate::api_server::core::frame::OutputAccumulator;
use crate::api_server::core::generate::{
    FrameShaper, GeneratePlan, RequestTiming, UnaryDrainPolicy, drain_plan_unary,
    generation_event_stream_with, unary_output,
};
use crate::message::response::{ChunkEvent, ChunkExtras};
use crate::message::sampling::SamplingParams;
use crate::message::types::{OneOrMany, TokenIds};
use dynamo_protocols::types::{
    Choice, CompletionFinishReason, CompletionUsage, CreateCompletionRequest,
    CreateCompletionResponse, Logprobs, Prompt, Stop,
};

pub(crate) fn completion_usage(prompt_tokens: u32, completion_tokens: u32) -> CompletionUsage {
    CompletionUsage {
        prompt_tokens,
        completion_tokens,
        total_tokens: prompt_tokens.saturating_add(completion_tokens),
        ..Default::default()
    }
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum PromptSpec {
    Text(String),
    TokenIds(TokenIds),
}

/// Record one output's prompt count for `prompt_index`, keeping the first
/// nonzero value: a scheduler's first frame may report 0 before the real count
/// is known, and sibling choices sharing a prompt must neither double-count nor
/// overwrite a known count with 0. Returns the recorded count.
fn record_prompt_tokens(counts: &mut BTreeMap<usize, u32>, prompt_index: usize, count: u32) -> u32 {
    let entry = counts.entry(prompt_index).or_insert(0);
    if *entry == 0 {
        *entry = count;
    }
    *entry
}

#[derive(Debug, Default)]
pub(crate) struct ChoiceExtensions {
    matched_stop: Option<serde_json::Value>,
    /// Dynamo's enum covers the standard values. Python additionally exposes
    /// `abort`, and native unknown finish types are preserved rather than lost.
    finish_reason_override: Option<String>,
}

/// Normalize the endpoint's single-shape prompt into one owned payload per
/// prompt. Consumes the prompt so text moves instead of cloning.
pub(crate) fn completion_prompt_specs(prompt: Prompt) -> Result<Vec<PromptSpec>, String> {
    match prompt {
        Prompt::String(text) => {
            if text.is_empty() {
                return Err("Prompt cannot be empty".into());
            }
            Ok(vec![PromptSpec::Text(text)])
        }
        Prompt::StringArray(texts) => {
            if texts.is_empty() || texts.iter().any(String::is_empty) {
                return Err("Prompt cannot be empty".into());
            }
            Ok(texts.into_iter().map(PromptSpec::Text).collect())
        }
        Prompt::IntegerArray(ids) => Ok(vec![token_prompt_spec(ids)?]),
        Prompt::ArrayOfIntegerArray(prompts) => {
            if prompts.is_empty() {
                return Err("Prompt cannot be empty".into());
            }
            prompts.into_iter().map(token_prompt_spec).collect()
        }
    }
}

/// Validate and convert one token-id prompt. The wire type is `Vec<u32>` while
/// [`TokenIds`] is `Vec<i32>`, so this is one reallocation per prompt — not a
/// clone that ownership rules could avoid.
fn token_prompt_spec(ids: Vec<u32>) -> Result<PromptSpec, String> {
    if ids.is_empty() {
        return Err("Prompt cannot be empty".into());
    }
    let input_ids = ids
        .into_iter()
        .map(|id| i32::try_from(id).map_err(|_| format!("Token ID {id} is out of range")))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(PromptSpec::TokenIds(input_ids))
}

pub(crate) fn completion_sampling_params(
    request: &CreateCompletionRequest,
) -> Result<SamplingParams, String> {
    let mut stop = None;
    let mut stop_token_ids = None;
    match request.stop.as_ref() {
        Some(Stop::String(value)) => stop = Some(OneOrMany::One(value.clone())),
        Some(Stop::StringArray(values)) => stop = Some(OneOrMany::Many(values.clone())),
        Some(Stop::TokenIdArray(values)) => {
            stop_token_ids
                .get_or_insert_with(Vec::new)
                .extend(values.iter().map(|&id| id as i64));
        }
        None => {}
    }

    let mut logit_bias = BTreeMap::new();
    if let Some(values) = request.logit_bias.as_ref() {
        for (token, bias) in values {
            let bias = bias
                .as_f64()
                .ok_or_else(|| format!("logit_bias[{token:?}] must be a number"))?;
            logit_bias.insert(token.clone(), bias);
        }
    }

    Ok(SamplingParams {
        max_new_tokens: Some(request.max_tokens.unwrap_or(16) as i64),
        stop,
        stop_token_ids,
        temperature: request.temperature.unwrap_or(1.0) as f64,
        top_p: request.top_p.unwrap_or(1.0) as f64,
        frequency_penalty: request.frequency_penalty.unwrap_or(0.0) as f64,
        presence_penalty: request.presence_penalty.unwrap_or(0.0) as f64,
        // OpenAI `n` is implemented by fan-out: every native request has one
        // output, avoiding the native path's intentional `n > 1` rejection.
        n: 1,
        logit_bias: (!logit_bias.is_empty()).then_some(logit_bias),
        sampling_seed: request.seed,
        ..Default::default()
    })
}

pub(crate) struct CompletionRenderingOptions {
    pub(crate) response_id: String,
    pub(crate) model: String,
    pub(crate) created: u32,
    pub(crate) echo: bool,
    pub(crate) want_logprobs: bool,
    pub(crate) n: usize,
}

pub(crate) async fn unary_completion(
    plan: GeneratePlan,
    options: CompletionRenderingOptions,
) -> Result<serde_json::Value, ApiError> {
    // Drain concurrently through the common path; results retain request order.
    let drained = drain_plan_unary(plan, UnaryDrainPolicy::AggregateFailFast).await?;
    let mut choices = Vec::with_capacity(drained.len());
    let mut extensions = Vec::with_capacity(drained.len());
    let mut prompt_tokens = BTreeMap::<usize, u32>::new();
    let mut completion_tokens = 0u64;

    for (choice_index, (_, outcome)) in drained.into_iter().enumerate() {
        let output = unary_output(outcome)?;
        let prompt_index = choice_index / options.n;

        record_prompt_tokens(&mut prompt_tokens, prompt_index, output.prompt_tokens);
        completion_tokens = completion_tokens.saturating_add(output.completion_tokens);
        let (response_choice, extension) =
            completion_choice(choice_index, output, options.echo, options.want_logprobs)?;
        choices.push(response_choice);
        extensions.push(extension);
    }

    let prompt_tokens = prompt_tokens
        .values()
        .copied()
        .fold(0u32, u32::saturating_add);
    let usage = completion_usage(
        prompt_tokens,
        u32::try_from(completion_tokens).unwrap_or(u32::MAX),
    );

    Ok(completion_response_value(
        CreateCompletionResponse {
            id: options.response_id,
            choices,
            created: options.created,
            model: options.model,
            system_fingerprint: None,
            object: "text_completion".into(),
            usage: Some(usage),
        },
        &extensions,
    ))
}

fn completion_choice(
    index: usize,
    mut output: ChunkEvent,
    echoed: bool,
    want_logprobs: bool,
) -> Result<(Choice, ChoiceExtensions), ApiError> {
    // Prepend the prompt metadata the generation lifecycle attached for
    // `echo=true`; its absence is a lifecycle fault, not a client error.
    if echoed {
        let prefix = output
            .extras
            .as_deref()
            .and_then(|extras| extras.prompt_text.as_deref())
            .ok_or_else(|| ApiError::internal("generation output is missing prompt metadata"))?;
        output.text.insert_str(0, prefix);
    }
    let reason = output.finish_reason.as_ref();
    let (finish_reason, finish_reason_override) = {
        match reason.and_then(|reason| reason.kind_name()).as_deref() {
            Some("stop") => (Some(CompletionFinishReason::Stop), None),
            Some("length") => (Some(CompletionFinishReason::Length), None),
            Some("content_filter") => (Some(CompletionFinishReason::ContentFilter), None),
            Some(other) => (None, Some(other.into())),
            None => (None, None),
        }
    };
    let matched_stop = reason.and_then(super::matched_stop_value);
    Ok((
        Choice {
            text: output.text,
            index: u32::try_from(index).unwrap_or(u32::MAX),
            logprobs: want_logprobs.then(|| completion_logprobs(output.extras.as_deref(), echoed)),
            finish_reason,
        },
        ChoiceExtensions {
            matched_stop,
            finish_reason_override,
        },
    ))
}

/// Serialize Dynamo's standard response and add only SGLang/Python fields that
/// its schema cannot represent. `text_offset` is corrected here because Dynamo
/// types it as `u32`, while Python deliberately emits `-1`.
pub(crate) fn completion_response_value(
    response: CreateCompletionResponse,
    extensions: &[ChoiceExtensions],
) -> serde_json::Value {
    let mut value = serde_json::to_value(response).expect("OpenAI response must serialize");
    let Some(root) = value.as_object_mut() else {
        return value;
    };
    // Python's Completion response does not expose this OpenAI field.
    root.remove("system_fingerprint");
    let Some(choices) = root
        .get_mut("choices")
        .and_then(serde_json::Value::as_array_mut)
    else {
        return value;
    };
    for (choice, extension) in choices.iter_mut().zip(extensions) {
        let Some(choice) = choice.as_object_mut() else {
            continue;
        };
        if let Some(reason) = &extension.finish_reason_override {
            choice.insert("finish_reason".into(), serde_json::json!(reason));
        }
        choice.insert(
            "matched_stop".into(),
            extension
                .matched_stop
                .clone()
                .unwrap_or(serde_json::Value::Null),
        );
        if let Some(logprobs) = choice
            .get_mut("logprobs")
            .and_then(serde_json::Value::as_object_mut)
        {
            let count = logprobs
                .get("tokens")
                .and_then(serde_json::Value::as_array)
                .map_or(0, Vec::len);
            logprobs.insert("text_offset".into(), serde_json::json!(vec![-1; count]));
        }
    }
    value
}

pub(crate) struct CompletionFrameShaper {
    response_id: String,
    model: String,
    created: u32,
    echo: bool,
    want_logprobs: bool,
    include_usage: bool,
    continuous_usage: bool,
    n: usize,
    first_chunks: Vec<bool>,
    prompt_tokens_by_prompt: BTreeMap<usize, u32>,
    completion_tokens_by_choice: Vec<u64>,
}

impl CompletionFrameShaper {
    pub(crate) fn new(
        count: usize,
        options: CompletionRenderingOptions,
        include_usage: bool,
        continuous_usage: bool,
    ) -> Self {
        let CompletionRenderingOptions {
            response_id,
            model,
            created,
            echo,
            want_logprobs,
            n,
        } = options;
        Self {
            response_id,
            model,
            created,
            echo,
            want_logprobs,
            include_usage,
            continuous_usage,
            n,
            first_chunks: vec![true; count],
            prompt_tokens_by_prompt: BTreeMap::new(),
            completion_tokens_by_choice: vec![0; count],
        }
    }

    fn frame(
        &mut self,
        choice_index: usize,
        out: ChunkEvent,
        completion_tokens: u64,
        first: bool,
    ) -> Result<CoreEvent<serde_json::Value>, ApiError> {
        let prompt_index = choice_index / self.n;
        let prompt_tokens = record_prompt_tokens(
            &mut self.prompt_tokens_by_prompt,
            prompt_index,
            out.prompt_tokens,
        );
        self.completion_tokens_by_choice[choice_index] = completion_tokens;
        let chunk_usage = self.continuous_usage.then(|| {
            completion_usage(
                prompt_tokens,
                u32::try_from(completion_tokens).unwrap_or(u32::MAX),
            )
        });
        let (choice, extension) =
            completion_choice(choice_index, out, self.echo && first, self.want_logprobs)?;
        Ok(CoreEvent::Item(completion_response_value(
            CreateCompletionResponse {
                id: self.response_id.clone(),
                choices: vec![choice],
                created: self.created,
                model: self.model.clone(),
                system_fingerprint: None,
                object: "text_completion".into(),
                usage: chunk_usage,
            },
            &[extension],
        )))
    }
}

impl FrameShaper for CompletionFrameShaper {
    type Frame = Result<CoreEvent<serde_json::Value>, ApiError>;

    fn delta(
        &mut self,
        out: ChunkEvent,
        acc: &OutputAccumulator,
        _rid: &str,
        index: Option<usize>,
    ) -> Self::Frame {
        let item_index = index.unwrap_or(0);
        let completion_tokens = acc.snapshot().completion_tokens;
        let first = std::mem::replace(&mut self.first_chunks[item_index], false);
        self.frame(item_index, out, completion_tokens, first)
    }

    fn coalesced(
        &mut self,
        acc: &OutputAccumulator,
        _rid: &str,
        index: Option<usize>,
    ) -> Self::Frame {
        self.delta(acc.snapshot().clone(), acc, "", index)
    }

    fn terminal(
        &mut self,
        out: ChunkEvent,
        acc: &OutputAccumulator,
        _incremental: bool,
        _rid: &str,
        index: Option<usize>,
        _timing: &RequestTiming,
    ) -> Self::Frame {
        self.delta(out, acc, _rid, index)
    }

    fn item_error(&mut self, code: u16, message: &str, _index: Option<usize>) -> Self::Frame {
        let error = ApiError::new(code, message);
        Ok(CoreEvent::ItemError(error))
    }

    fn finish(&mut self) -> Option<Self::Frame> {
        if !self.include_usage {
            return None;
        }
        let prompt_tokens = self
            .prompt_tokens_by_prompt
            .values()
            .copied()
            .fold(0u32, u32::saturating_add);
        let completion_tokens = self
            .completion_tokens_by_choice
            .iter()
            .copied()
            .fold(0u64, u64::saturating_add);
        Some(Ok(CoreEvent::Item(completion_response_value(
            CreateCompletionResponse {
                id: self.response_id.clone(),
                choices: vec![],
                created: self.created,
                model: self.model.clone(),
                system_fingerprint: None,
                object: "text_completion".into(),
                usage: Some(completion_usage(
                    prompt_tokens,
                    u32::try_from(completion_tokens).unwrap_or(u32::MAX),
                )),
            },
            &[],
        ))))
    }
}

/// Rendering failures end the logical Completion response. Scheduler item
/// errors are successful frame values and retain the shared per-item behavior.
pub(crate) fn completion_event_stream(
    plan: GeneratePlan,
    shaper: CompletionFrameShaper,
) -> impl futures::Stream<Item = CoreEvent<serde_json::Value>> {
    async_stream::stream! {
        let error = {
            let raw = generation_event_stream_with(
                GeneratePlan { incremental: true, ..plan },
                shaper,
            );
            futures::pin_mut!(raw);
            loop {
                match raw.next().await {
                    Some(Ok(frame)) => yield frame,
                    Some(Err(error)) => break error,
                    None => return,
                }
            }
        }; // Drop the raw stream and abort unfinished work before yielding the error.
        yield CoreEvent::ItemError(error);
    }
}

/// One completions stream event as its SSE `data` payload: the shaped chunk
/// JSON or the OpenAI error body.
pub(crate) fn completion_sse_payload(event: CoreEvent<serde_json::Value>) -> String {
    match event {
        CoreEvent::Item(value) => value.to_string(),
        CoreEvent::ItemError(e) => {
            crate::api_server::core::openai::error_payload_value(e.http_code, &e.message)
                .to_string()
        }
    }
}

pub(crate) fn completion_logprobs(extras: Option<&ChunkExtras>, include_input: bool) -> Logprobs {
    let mut result = Logprobs {
        tokens: Vec::new(),
        token_logprobs: Vec::new(),
        top_logprobs: Vec::new(),
        text_offset: Vec::new(),
    };
    let Some(extras) = extras else {
        return result;
    };
    if include_input {
        append_selected_logprobs(
            &mut result,
            &extras.in_lp_val,
            &extras.in_lp_idx,
            &extras.in_lp_txt,
        );
        append_top_logprobs(
            &mut result,
            &extras.in_top_val,
            &extras.in_top_idx,
            &extras.in_top_lens,
            &extras.in_top_txt,
        );
    }
    append_selected_logprobs(
        &mut result,
        &extras.out_lp_val,
        &extras.out_lp_idx,
        &extras.out_lp_txt,
    );
    append_top_logprobs(
        &mut result,
        &extras.out_top_val,
        &extras.out_top_idx,
        &extras.out_top_lens,
        &extras.out_top_txt,
    );
    result
}

fn append_selected_logprobs(result: &mut Logprobs, values: &[f32], ids: &[i32], texts: &[String]) {
    for (index, (&value, &id)) in values.iter().zip(ids).enumerate() {
        result.tokens.push(
            texts
                .get(index)
                .cloned()
                .unwrap_or_else(|| format!("token_id:{id}")),
        );
        result
            .token_logprobs
            .push((!value.is_nan()).then_some(value));
        // Dynamo's field is `u32`; Python's `-1` sentinel is applied once at
        // final wire shaping in `completion_response_value`.
        result.text_offset.push(0);
    }
}

fn append_top_logprobs(
    result: &mut Logprobs,
    values: &[f32],
    ids: &[i32],
    lens: &[u32],
    texts: &[String],
) {
    let mut offset = 0usize;
    for &len in lens {
        let len = len as usize;
        if len == 0 {
            result.top_logprobs.push(serde_json::Value::Null);
            continue;
        }
        let mut top = BTreeMap::new();
        for index in offset..offset.saturating_add(len) {
            let (Some(&value), Some(&id)) = (values.get(index), ids.get(index)) else {
                continue;
            };
            top.insert(
                texts
                    .get(index)
                    .cloned()
                    .unwrap_or_else(|| format!("token_id:{id}")),
                value,
            );
        }
        result.top_logprobs.push(serde_json::json!(top));
        offset = offset.saturating_add(len);
    }
}

#[cfg(test)]
mod tests {
    use crate::api_server::core::test_utils::{
        abort_senders, aborted_guard_rids, chunk, plan, planned, senders,
    };

    use super::{
        ChoiceExtensions, CompletionFrameShaper, CompletionRenderingOptions, PromptSpec,
        completion_event_stream, completion_logprobs, completion_prompt_specs,
        completion_response_value, unary_completion,
    };
    use crate::api_server::core::event::CoreEvent;
    use crate::message::response::{ChunkExtras, ResponseItem};
    use dynamo_protocols::types::{Choice, CreateCompletionResponse, Prompt};
    use futures::{FutureExt, StreamExt};

    /// The common rendering options; tests override fields with struct-update
    /// syntax.
    fn completion_options() -> CompletionRenderingOptions {
        CompletionRenderingOptions {
            response_id: "cmpl-test".into(),
            model: "model".into(),
            created: 1,
            echo: false,
            want_logprobs: false,
            n: 1,
        }
    }

    fn chunk_with_prompt(rid: &str, text: &str, done: bool, prompt_tokens: u32) -> ResponseItem {
        let mut item = chunk(rid, text, done);
        match &mut item {
            ResponseItem::Frame(event) | ResponseItem::Done(event) => {
                event.prompt_tokens = prompt_tokens;
            }
            _ => unreachable!("chunk builds a frame"),
        }
        item
    }

    #[test]
    fn prompt_specs_consume_text_and_convert_token_ids() {
        assert_eq!(
            completion_prompt_specs(Prompt::String("hi".into())).unwrap(),
            [PromptSpec::Text("hi".into())]
        );
        assert_eq!(
            completion_prompt_specs(Prompt::StringArray(vec!["a".into(), "b".into()])).unwrap(),
            [PromptSpec::Text("a".into()), PromptSpec::Text("b".into())]
        );
        assert_eq!(
            completion_prompt_specs(Prompt::IntegerArray(vec![1, 2])).unwrap(),
            [PromptSpec::TokenIds(vec![1, 2])]
        );
        assert_eq!(
            completion_prompt_specs(Prompt::ArrayOfIntegerArray(vec![vec![1], vec![2, 3]]))
                .unwrap(),
            [
                PromptSpec::TokenIds(vec![1]),
                PromptSpec::TokenIds(vec![2, 3])
            ]
        );
        assert_eq!(
            completion_prompt_specs(Prompt::String(String::new())).unwrap_err(),
            "Prompt cannot be empty"
        );
        assert!(completion_prompt_specs(Prompt::IntegerArray(vec![])).is_err());
        assert!(completion_prompt_specs(Prompt::IntegerArray(vec![u32::MAX])).is_err());
    }

    #[test]
    fn zero_top_logprobs_keeps_selected_token_and_empty_top_map() {
        let extras = ChunkExtras {
            out_lp_val: vec![-0.25],
            out_lp_idx: vec![7],
            out_lp_txt: vec!["x".into()],
            out_top_lens: vec![0],
            ..Default::default()
        };
        let logprobs = completion_logprobs(Some(&extras), false);
        assert_eq!(logprobs.tokens, ["x"]);
        assert_eq!(logprobs.token_logprobs, [Some(-0.25)]);
        assert_eq!(logprobs.top_logprobs, [serde_json::Value::Null]);

        let value = completion_response_value(
            CreateCompletionResponse {
                id: "cmpl-test".into(),
                choices: vec![Choice {
                    text: "x".into(),
                    index: 0,
                    logprobs: Some(logprobs),
                    finish_reason: None,
                }],
                created: 1,
                model: "model".into(),
                system_fingerprint: None,
                object: "text_completion".into(),
                usage: None,
            },
            &[ChoiceExtensions::default()],
        );
        assert_eq!(
            value["choices"][0]["logprobs"]["text_offset"],
            serde_json::json!([-1])
        );
    }

    #[tokio::test]
    async fn unary_fold_orders_choices_and_counts_each_prompt_once() {
        let (choice0, tx0) = planned("r0");
        let (choice1, tx1) = planned("r1");
        tx0.send(chunk("r0", "a", false)).await.unwrap();
        tx0.send(chunk("r0", "b", true)).await.unwrap();
        tx1.send(chunk("r1", "x", false)).await.unwrap();
        tx1.send(chunk("r1", "y", true)).await.unwrap();

        let response = unary_completion(
            plan(vec![choice0, choice1], senders()),
            CompletionRenderingOptions {
                n: 2,
                ..completion_options()
            },
        )
        .await
        .expect("unary completion succeeds");
        let value = response;
        assert_eq!(value["choices"][0]["text"], "ab");
        assert_eq!(value["choices"][1]["text"], "xy");
        assert_eq!(value["choices"][0]["matched_stop"], "</s>");
        assert_eq!(value["usage"]["prompt_tokens"], 5);
        assert_eq!(value["usage"]["completion_tokens"], 4);
    }

    /// A zero prompt count on the first sibling must not latch: a later
    /// sibling's real count recovers it, and the shared prompt is counted once.
    #[tokio::test]
    async fn unary_usage_recovers_a_zero_prompt_count_from_a_sibling_choice() {
        let (choice0, tx0) = planned("r0");
        let (choice1, tx1) = planned("r1");
        tx0.send(chunk_with_prompt("r0", "a", true, 0))
            .await
            .unwrap();
        tx1.send(chunk_with_prompt("r1", "b", true, 5))
            .await
            .unwrap();

        let value = unary_completion(
            plan(vec![choice0, choice1], senders()),
            CompletionRenderingOptions {
                n: 2,
                ..completion_options()
            },
        )
        .await
        .expect("unary completion succeeds");
        assert_eq!(value["usage"]["prompt_tokens"], 5);
        assert_eq!(value["usage"]["completion_tokens"], 2);
    }

    #[tokio::test]
    async fn echo_uses_prompt_text_from_the_generation_frame() {
        let (choice, tx) = planned("r0");
        let mut output = chunk("r0", "completion", true);
        if let ResponseItem::Done(event) = &mut output {
            event
                .extras
                .get_or_insert_with(Default::default)
                .prompt_text = Some("decoded token prompt".into());
        }
        tx.send(output).await.unwrap();

        let value = unary_completion(
            plan(vec![choice], senders()),
            CompletionRenderingOptions {
                echo: true,
                ..completion_options()
            },
        )
        .await
        .expect("echo completion succeeds");
        assert_eq!(
            value["choices"][0]["text"],
            "decoded token promptcompletion"
        );
    }

    #[tokio::test]
    async fn missing_echo_prompt_metadata_is_an_error() {
        let (choice, tx) = planned("r0");
        tx.send(chunk("r0", "completion", true)).await.unwrap();

        let error = unary_completion(
            plan(vec![choice], senders()),
            CompletionRenderingOptions {
                echo: true,
                ..completion_options()
            },
        )
        .await
        .expect_err("echo without prompt metadata must fail");
        assert_eq!(error.http_code, 500);
        assert!(error.message.contains("missing prompt metadata"));
    }

    #[tokio::test]
    async fn rendering_error_aborts_unfinished_choices_before_emitting_error() {
        for terminal in [false, true] {
            for failing_index in 0..2 {
                let (choice0, tx0) = planned("r0");
                let (choice1, tx1) = planned("r1");
                let rids = ["r0", "r1"];
                let txs = [tx0, tx1];
                txs[failing_index]
                    .send(chunk(rids[failing_index], "no echo metadata", terminal))
                    .await
                    .unwrap();
                let (senders, abort_rx) = abort_senders();
                let plan = plan(vec![choice0, choice1], senders);
                let stream = completion_event_stream(
                    plan,
                    CompletionFrameShaper::new(
                        2,
                        CompletionRenderingOptions {
                            echo: true,
                            n: 2,
                            ..completion_options()
                        },
                        true,
                        false,
                    ),
                );
                futures::pin_mut!(stream);

                let CoreEvent::ItemError(error) = stream.next().await.unwrap() else {
                    panic!("missing requested echo metadata must fail rendering");
                };
                assert_eq!(error.http_code, 500);
                assert!(error.message.contains("missing prompt metadata"));

                // Cancellation must already have happened while the adapter is
                // suspended at its error yield, not only after the client drops it.
                let expected = if terminal {
                    vec![rids[1 - failing_index]]
                } else {
                    rids.to_vec()
                };
                assert_eq!(
                    aborted_guard_rids(&abort_rx),
                    expected,
                    "terminal={terminal}, index={failing_index}"
                );
                assert!(
                    stream
                        .next()
                        .now_or_never()
                        .expect("rendering failure must end immediately")
                        .is_none(),
                    "no sibling content or final usage after rendering failure"
                );
            }
        }
    }

    #[tokio::test]
    async fn scheduler_item_error_keeps_sibling_output_and_final_usage() {
        let (choice0, tx0) = planned("r0");
        let (choice1, tx1) = planned("r1");
        tx0.send(ResponseItem::Error(crate::utils::error::Error::Validation(
            "bad choice".into(),
        )))
        .await
        .unwrap();
        tx1.send(chunk("r1", "a", false)).await.unwrap();
        tx1.send(chunk("r1", "b", true)).await.unwrap();

        let frames: Vec<_> = completion_event_stream(
            plan(vec![choice0, choice1], senders()),
            CompletionFrameShaper::new(
                2,
                CompletionRenderingOptions {
                    n: 2,
                    ..completion_options()
                },
                true,
                false,
            ),
        )
        .collect()
        .await;
        assert_eq!(frames.len(), 4);
        assert_eq!(
            frames
                .iter()
                .filter(|frame| matches!(frame, CoreEvent::ItemError(_)))
                .count(),
            1
        );
        let outputs: Vec<_> = frames
            .iter()
            .filter_map(|frame| match frame {
                CoreEvent::Item(value) => Some(value),
                CoreEvent::ItemError(error) => {
                    assert_eq!(error.http_code, 400);
                    None
                }
            })
            .collect();
        assert_eq!(outputs[0]["choices"][0]["text"], "a");
        assert_eq!(outputs[1]["choices"][0]["text"], "b");
        assert!(outputs[2]["choices"].as_array().unwrap().is_empty());
        assert_eq!(outputs[2]["usage"]["completion_tokens"], 2);
    }

    #[tokio::test]
    async fn stream_uses_deltas_then_usage_and_done() {
        let (choice, tx) = planned("r0");
        tx.send(chunk("r0", "a", false)).await.unwrap();
        tx.send(chunk("r0", "b", true)).await.unwrap();

        let plan = plan(vec![choice], senders());
        let stream = completion_event_stream(
            plan,
            CompletionFrameShaper::new(1, completion_options(), true, false),
        );
        futures::pin_mut!(stream);
        let frames: Vec<String> = stream.map(super::completion_sse_payload).collect().await;
        assert_eq!(frames.len(), 3);
        let first: serde_json::Value = serde_json::from_str(&frames[0]).unwrap();
        let terminal: serde_json::Value = serde_json::from_str(&frames[1]).unwrap();
        let usage: serde_json::Value = serde_json::from_str(&frames[2]).unwrap();
        assert_eq!(first["choices"][0]["text"], "a");
        assert_eq!(terminal["choices"][0]["text"], "b");
        assert_eq!(terminal["choices"][0]["finish_reason"], "stop");
        assert!(usage["choices"].as_array().unwrap().is_empty());
        assert_eq!(usage["usage"]["prompt_tokens"], 5);
        assert_eq!(usage["usage"]["completion_tokens"], 2);
    }

    /// `continuous_usage_stats` puts a running usage object on every chunk
    /// instead of only on the final usage trailer.
    #[tokio::test]
    async fn stream_continuous_usage_reports_monotonic_totals_per_chunk() {
        let (choice, tx) = planned("r0");
        tx.send(chunk("r0", "a", false)).await.unwrap();
        tx.send(chunk("r0", "b", true)).await.unwrap();

        let stream = completion_event_stream(
            plan(vec![choice], senders()),
            CompletionFrameShaper::new(
                1,
                completion_options(),
                false, // no separate final usage trailer
                true,  // continuous per-chunk usage
            ),
        );
        futures::pin_mut!(stream);
        let frames: Vec<serde_json::Value> = stream
            .map(super::completion_sse_payload)
            .map(|payload| serde_json::from_str::<serde_json::Value>(&payload).unwrap())
            .collect()
            .await;
        assert_eq!(frames.len(), 2, "usage rides the chunks, not a trailer");
        assert_eq!(frames[0]["usage"]["prompt_tokens"], 5);
        assert_eq!(frames[0]["usage"]["completion_tokens"], 1);
        assert_eq!(frames[1]["usage"]["completion_tokens"], 2);
    }

    /// The first nonzero prompt count wins per original prompt: an early zero
    /// is recovered, siblings sharing a prompt count once, and the final
    /// trailer sums distinct prompts. Continuous chunks report recovered counts.
    #[tokio::test]
    async fn stream_usage_recovers_zero_prompt_counts_and_deduplicates_prompts() {
        let (receivers, txs): (Vec<_>, Vec<_>) =
            (0..4).map(|index| planned(&format!("r{index}"))).unzip();
        // Two prompts x two choices. Prompt 0 reports 0 from both choices
        // before the real count arrives; prompt 1 is known from the start.
        txs[0]
            .send(chunk_with_prompt("r0", "a", false, 0))
            .await
            .unwrap();
        txs[0]
            .send(chunk_with_prompt("r0", "b", true, 5))
            .await
            .unwrap();
        txs[1]
            .send(chunk_with_prompt("r1", "c", false, 0))
            .await
            .unwrap();
        txs[1]
            .send(chunk_with_prompt("r1", "d", true, 5))
            .await
            .unwrap();
        txs[2]
            .send(chunk_with_prompt("r2", "e", false, 7))
            .await
            .unwrap();
        txs[2]
            .send(chunk_with_prompt("r2", "f", true, 7))
            .await
            .unwrap();
        txs[3]
            .send(chunk_with_prompt("r3", "g", true, 7))
            .await
            .unwrap();

        let stream = completion_event_stream(
            plan(receivers, senders()),
            CompletionFrameShaper::new(
                4,
                CompletionRenderingOptions {
                    n: 2,
                    ..completion_options()
                },
                true, // final usage trailer
                true, // continuous per-chunk usage
            ),
        );
        futures::pin_mut!(stream);
        let frames: Vec<serde_json::Value> = stream
            .map(super::completion_sse_payload)
            .map(|payload| serde_json::from_str::<serde_json::Value>(&payload).unwrap())
            .collect()
            .await;
        assert_eq!(frames.len(), 8, "7 chunks + the final usage trailer");
        let trailer = frames.last().unwrap();
        assert!(trailer["choices"].as_array().unwrap().is_empty());
        assert_eq!(trailer["usage"]["prompt_tokens"], 12);
        assert_eq!(trailer["usage"]["completion_tokens"], 7);
        assert_eq!(trailer["usage"]["total_tokens"], 19);

        let mut recovered_by_choice = std::collections::BTreeMap::new();
        for frame in &frames {
            let prompt = frame["usage"]["prompt_tokens"].as_u64().unwrap();
            for choice in frame["choices"].as_array().unwrap() {
                let index = choice["index"].as_u64().unwrap();
                recovered_by_choice
                    .entry(index)
                    .and_modify(|value: &mut u64| *value = (*value).max(prompt))
                    .or_insert(prompt);
            }
        }
        assert_eq!(
            recovered_by_choice,
            std::collections::BTreeMap::from([(0, 5), (1, 5), (2, 7), (3, 7)])
        );
    }

    #[tokio::test]
    async fn stream_echo_uses_prompt_text_on_first_delta() {
        let (choice, tx) = planned("r0");
        let mut first = chunk("r0", "a", false);
        if let ResponseItem::Frame(event) = &mut first {
            event
                .extras
                .get_or_insert_with(Default::default)
                .prompt_text = Some("decoded prompt".into());
        }
        tx.send(first).await.unwrap();
        tx.send(chunk("r0", "b", true)).await.unwrap();

        let stream = completion_event_stream(
            plan(vec![choice], senders()),
            CompletionFrameShaper::new(
                1,
                CompletionRenderingOptions {
                    echo: true,
                    ..completion_options()
                },
                false,
                false,
            ),
        );
        futures::pin_mut!(stream);
        let frames: Vec<String> = stream
            .filter_map(|event| async move {
                match event {
                    CoreEvent::Item(value) => Some(value.to_string()),
                    CoreEvent::ItemError(_) => None,
                }
            })
            .collect()
            .await;
        assert_eq!(frames.len(), 2);
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&frames[0]).unwrap()["choices"][0]["text"],
            "decoded prompta"
        );
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&frames[1]).unwrap()["choices"][0]["text"],
            "b"
        );
    }
}
