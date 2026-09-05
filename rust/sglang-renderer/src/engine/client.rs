//! HTTP client from renderer-owned generation requests to SGLang `/generate`.

use std::time::Duration;

use async_stream::stream;
use futures::StreamExt;
use serde::{Deserialize, Serialize};

use crate::{
    GenerateRequest, GenerationFinishReason, GenerationOutput, GenerationOutputExtras,
    GenerationStream, MatchedStop, PositionLogprobs, ResponseError, TokenIds, TokenLogprob,
};

// SGLang's deep health probe defaults to 20 seconds. Leave it time to return
// its own status while still bounding a peer that never sends response headers.
const ENGINE_HEALTH_REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Clone)]
pub struct HttpGenerateClient {
    client: reqwest::Client,
    generate_url: reqwest::Url,
    health_url: reqwest::Url,
    health_timeout: Duration,
    tokenizer: dynamo_tokenizers::Tokenizer,
}

/// Renderer-to-engine framing options stay private to this HTTP transport and
/// do not become part of the model server's `/generate` contract.
#[derive(Serialize)]
struct EngineGenerateBody<'a> {
    #[serde(flatten)]
    request: &'a GenerateRequest,
    incremental_streaming_output: bool,
}

impl HttpGenerateClient {
    pub fn new(
        engine_url: impl AsRef<str>,
        tokenizer: dynamo_tokenizers::Tokenizer,
    ) -> Result<Self, String> {
        let engine_url = engine_url.as_ref();
        let base_url = reqwest::Url::parse(engine_url)
            .map_err(|error| format!("invalid engine URL {engine_url:?}: {error}"))?;
        let is_http_origin = matches!(base_url.scheme(), "http" | "https")
            && base_url.host_str().is_some()
            && base_url.username().is_empty()
            && base_url.password().is_none()
            && base_url.path() == "/"
            && base_url.query().is_none()
            && base_url.fragment().is_none();
        if !is_http_origin {
            return Err(format!(
                "invalid engine URL {engine_url:?}: expected an HTTP(S) origin without credentials, a path, query, or fragment"
            ));
        }
        let generate_url = base_url
            .join("/generate")
            .map_err(|error| format!("joining /generate to engine URL failed: {error}"))?;
        let health_url = base_url
            .join("/health")
            .map_err(|error| format!("joining /health to engine URL failed: {error}"))?;
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .build()
            .map_err(|error| format!("building engine HTTP client failed: {error}"))?;
        Ok(Self {
            client,
            generate_url,
            health_url,
            health_timeout: ENGINE_HEALTH_REQUEST_TIMEOUT,
            tokenizer,
        })
    }

    #[cfg(test)]
    pub(crate) fn with_health_timeout(mut self, timeout: Duration) -> Self {
        self.health_timeout = timeout;
        self
    }

    pub(crate) async fn health_status(&self) -> Result<reqwest::StatusCode, ResponseError> {
        let request = self
            .client
            .get(self.health_url.clone())
            .timeout(self.health_timeout);
        let response = request
            .send()
            .await
            .map_err(|error| unavailable(format!("engine health check failed: {error}")))?;
        Ok(response.status())
    }

    pub async fn generate(
        &self,
        mut request: GenerateRequest,
    ) -> Result<GenerationStream, ResponseError> {
        let mut stop_matcher = take_text_stops(&mut request)?;

        let prompt_ids = request
            .input_ids
            .iter()
            .map(|&id| u32::try_from(id))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| invalid("input_ids must be non-negative"))?;
        let skip_special_tokens = request.sampling_params.skip_special_tokens;
        let decode_logprob_text = request.return_text_in_logprobs.unwrap_or(false);
        // The renderer needs token deltas even for a unary OpenAI request. It
        // collects them locally and dropping this response stream propagates
        // client cancellation to `/generate`.
        request.stream = true;
        // The token-only engine cannot fill text columns. Decode them below.
        request.return_text_in_logprobs = Some(false);

        let response = self
            .client
            .post(self.generate_url.clone())
            .json(&EngineGenerateBody {
                request: &request,
                incremental_streaming_output: true,
            })
            .send()
            .await
            .map_err(|error| unavailable(format!("engine request failed: {error}")))?;
        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(ResponseError {
                status_code: status.as_u16(),
                message: engine_error_message(&body)
                    .unwrap_or_else(|| format!("engine returned HTTP {status}")),
            });
        }

        let tokenizer = self.tokenizer.clone();
        let logprob_tokenizer = tokenizer.clone();
        let mut decoder = tokenizer.decode_stream(&prompt_ids, skip_special_tokens);
        let mut chunks = response.bytes_stream();
        let events = stream! {
            let mut parser = SseParser::default();
            let mut terminal = false;
            let mut emitted_tokens = 0;
            while let Some(chunk) = chunks.next().await {
                let chunk = match chunk {
                    Ok(chunk) => chunk,
                    Err(error) => {
                        yield Err(unavailable(format!("engine stream failed: {error}")));
                        return;
                    }
                };
                for payload in parser.push(&chunk) {
                    if payload == "[DONE]" {
                        if !terminal {
                            yield Err(internal("engine stream ended before a terminal frame"));
                        }
                        return;
                    }
                    let mut output = match parse_engine_frame(&payload) {
                        Ok(output) => output,
                        Err(error) => {
                            yield Err(error);
                            return;
                        }
                    };
                    if let Err(error) = normalize_engine_output(&mut output, &mut emitted_tokens) {
                        yield Err(error);
                        return;
                    }
                    let matched_stop = match decode_output(
                        &mut decoder,
                        &mut output,
                        stop_matcher.as_mut(),
                    ) {
                        Ok(matched_stop) => matched_stop,
                        Err(error) => {
                            yield Err(error);
                            return;
                        }
                    };
                    if decode_logprob_text {
                        fill_logprob_text(&logprob_tokenizer, output.extras.as_deref_mut());
                    }
                    if let Some(matched_stop) = matched_stop {
                        output.finish_reason = Some(GenerationFinishReason::Stop(Some(
                            MatchedStop::Text(matched_stop),
                        )));
                        yield Ok(output);
                        return;
                    }
                    terminal = output.finish_reason.is_some();
                    yield Ok(output);
                }
            }
            if !terminal {
                yield Err(internal("engine response closed before [DONE]"));
            }
        }
        .boxed();

        Ok(events)
    }

    pub fn detokenize(&self, token_ids: TokenIds) -> Result<String, ResponseError> {
        let ids = token_ids
            .into_iter()
            .map(u32::try_from)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| invalid("token IDs must be non-negative"))?;
        self.tokenizer
            .decode(&ids, true)
            .map(String::from)
            .map_err(|error| invalid(format!("detokenizing prompt failed: {error}")))
    }
}

fn take_text_stops(
    request: &mut GenerateRequest,
) -> Result<Option<StopStringMatcher>, ResponseError> {
    let params = &mut request.sampling_params;
    let matcher = StopStringMatcher::new(std::mem::take(&mut params.stop), params.no_stop_trim);
    Ok(matcher)
}

struct StopStringMatcher {
    stops: Vec<String>,
    pending: String,
    include_stop: bool,
}

struct StopMatch {
    text: String,
    matched: Option<String>,
}

impl StopStringMatcher {
    fn new(stops: Vec<String>, include_stop: bool) -> Option<Self> {
        (!stops.is_empty()).then_some(Self {
            stops,
            pending: String::new(),
            include_stop,
        })
    }

    fn push(&mut self, text: &str) -> StopMatch {
        self.pending.push_str(text);
        if let Some((position, stop)) = self
            .stops
            .iter()
            .filter_map(|stop| {
                self.pending
                    .find(stop)
                    .map(|position| (position, stop.clone()))
            })
            .min_by_key(|(position, _)| *position)
        {
            if stop.is_empty() {
                return StopMatch {
                    text: std::mem::take(&mut self.pending),
                    matched: Some(stop),
                };
            }
            let end = if self.include_stop {
                position + stop.len()
            } else {
                position
            };
            let text = self.pending[..end].to_owned();
            self.pending.clear();
            return StopMatch {
                text,
                matched: Some(stop),
            };
        }

        let held_start = self
            .pending
            .char_indices()
            .map(|(start, _)| start)
            .chain(std::iter::once(self.pending.len()))
            .find(|&start| {
                self.stops
                    .iter()
                    .any(|stop| stop.starts_with(&self.pending[start..]))
            })
            .unwrap_or(self.pending.len());
        let held = self.pending.split_off(held_start);
        let text = std::mem::replace(&mut self.pending, held);
        StopMatch {
            text,
            matched: None,
        }
    }

    fn flush(&mut self) -> String {
        std::mem::take(&mut self.pending)
    }
}

fn decode_output(
    decoder: &mut dynamo_tokenizers::DecodeStream,
    output: &mut GenerationOutput,
    mut stop_matcher: Option<&mut StopStringMatcher>,
) -> Result<Option<String>, ResponseError> {
    let mut text = String::new();
    for index in 0..output.token_ids.len() {
        let id = output.token_ids[index];
        let id = u32::try_from(id).map_err(|_| internal("engine returned a negative token ID"))?;
        let delta = decoder
            .step(id)
            .map_err(|error| internal(format!("detokenizing engine output failed: {error}")))?;
        if let Some(matcher) = stop_matcher.as_deref_mut() {
            let matched = matcher.push(delta.as_deref().unwrap_or_default());
            text.push_str(&matched.text);
            if let Some(stop) = matched.matched {
                truncate_output(output, index + 1)?;
                output.text = text;
                return Ok(Some(stop));
            }
        } else if let Some(delta) = delta {
            text.push_str(&delta);
        }
    }
    if output.finish_reason.is_some()
        && let Some(matcher) = stop_matcher
    {
        text.push_str(&matcher.flush());
    }
    output.text = text;
    Ok(None)
}

fn truncate_output(output: &mut GenerationOutput, kept_tokens: usize) -> Result<(), ResponseError> {
    output.token_ids.truncate(kept_tokens);
    output.completion_tokens = u64::try_from(kept_tokens).unwrap_or(u64::MAX);
    let Some(extras) = output.extras.as_deref_mut() else {
        return Ok(());
    };
    truncate_optional(
        &mut extras.output_logprobs,
        kept_tokens,
        "output logprob positions",
    )
}

fn truncate_optional<T>(
    values: &mut Vec<T>,
    length: usize,
    description: &str,
) -> Result<(), ResponseError> {
    if values.is_empty() {
        return Ok(());
    }
    if values.len() < length {
        return Err(internal(format!(
            "engine returned {} {description} values for {length} retained tokens",
            values.len()
        )));
    }
    values.truncate(length);
    Ok(())
}

fn fill_logprob_text(
    tokenizer: &dynamo_tokenizers::Tokenizer,
    extras: Option<&mut GenerationOutputExtras>,
) {
    let Some(extras) = extras else { return };
    for position in extras
        .output_logprobs
        .iter_mut()
        .chain(&mut extras.input_logprobs)
    {
        fill_text(tokenizer, &mut position.token);
        for token in &mut position.top {
            fill_text(tokenizer, token);
        }
    }
}

fn fill_text(tokenizer: &dynamo_tokenizers::Tokenizer, token: &mut TokenLogprob) {
    if token.text.is_some() {
        return;
    }
    token.text = Some(
        u32::try_from(token.token_id)
            .ok()
            .and_then(|id| tokenizer.decode(&[id], false).ok())
            .map(String::from)
            .unwrap_or_default(),
    );
}

#[derive(Default)]
struct SseParser {
    bytes: Vec<u8>,
}

impl SseParser {
    fn push(&mut self, chunk: &[u8]) -> Vec<String> {
        self.bytes.extend_from_slice(chunk);
        let mut payloads = Vec::new();
        while let Some((end, separator_len)) = event_end(&self.bytes) {
            let event = self.bytes.drain(..end).collect::<Vec<_>>();
            self.bytes.drain(..separator_len);
            let event = String::from_utf8_lossy(&event);
            let data = event
                .lines()
                .filter_map(|line| line.strip_prefix("data:").map(str::trim_start))
                .collect::<Vec<_>>()
                .join("\n");
            if !data.is_empty() {
                payloads.push(data);
            }
        }
        payloads
    }
}

fn event_end(bytes: &[u8]) -> Option<(usize, usize)> {
    let crlf = bytes
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .map(|position| (position, 4));
    let lf = bytes
        .windows(2)
        .position(|window| window == b"\n\n")
        .map(|position| (position, 2));
    match (crlf, lf) {
        (Some(crlf), Some(lf)) => Some(crlf.min(lf)),
        (Some(crlf), None) => Some(crlf),
        (None, Some(lf)) => Some(lf),
        (None, None) => None,
    }
}

type WireLogprob = (Option<f32>, i32, Option<String>);
type WireTopLogprobs = Vec<Option<Vec<WireLogprob>>>;

#[derive(Deserialize)]
struct EngineFrame {
    #[serde(default)]
    output_ids: TokenIds,
    meta_info: EngineMeta,
}

#[derive(Deserialize)]
struct EngineMeta {
    #[serde(default)]
    prompt_tokens: u32,
    #[serde(default)]
    completion_tokens: u64,
    #[serde(default)]
    finish_reason: Option<EngineFinishReason>,
    #[serde(default)]
    output_token_logprobs: Vec<WireLogprob>,
    #[serde(default)]
    input_token_logprobs: Vec<WireLogprob>,
    #[serde(default)]
    output_top_logprobs: WireTopLogprobs,
    #[serde(default)]
    input_top_logprobs: WireTopLogprobs,
}

#[derive(Deserialize)]
struct EngineFinishReason {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    matched: Option<EngineMatchedStop>,
    #[serde(default)]
    status_code: Option<u16>,
    #[serde(default)]
    message: Option<String>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum EngineMatchedStop {
    Token(i64),
    Text(String),
    Tokens(Vec<i64>),
}

#[derive(Deserialize)]
struct EngineErrorEnvelope {
    error: EngineError,
}

#[derive(Deserialize)]
struct EngineError {
    #[serde(default = "default_error_code")]
    code: u16,
    message: String,
}

fn default_error_code() -> u16 {
    500
}

fn parse_engine_frame(payload: &str) -> Result<GenerationOutput, ResponseError> {
    if let Ok(error) = serde_json::from_str::<EngineErrorEnvelope>(payload) {
        return Err(ResponseError {
            status_code: error.error.code,
            message: error.error.message,
        });
    }
    let frame: EngineFrame = serde_json::from_str(payload)
        .map_err(|error| internal(format!("invalid engine frame: {error}")))?;
    if let Some(reason) = frame.meta_info.finish_reason.as_ref()
        && reason.kind == "abort"
        && let Some(status_code) = reason.status_code
    {
        return Err(ResponseError {
            status_code,
            message: reason
                .message
                .clone()
                .unwrap_or_else(|| "request aborted".to_owned()),
        });
    }
    let finish_reason = frame
        .meta_info
        .finish_reason
        .map(|reason| match reason.kind.as_str() {
            "stop" => GenerationFinishReason::Stop(reason.matched.map(|matched| match matched {
                EngineMatchedStop::Token(id) => MatchedStop::Token(id),
                EngineMatchedStop::Text(text) => MatchedStop::Text(text),
                EngineMatchedStop::Tokens(ids) => MatchedStop::Tokens(ids),
            })),
            "length" => GenerationFinishReason::Length,
            "abort" => GenerationFinishReason::Abort,
            "content_filter" => GenerationFinishReason::ContentFilter,
            other => GenerationFinishReason::Other(other.to_owned()),
        });
    let has_extras = !frame.meta_info.output_token_logprobs.is_empty()
        || !frame.meta_info.input_token_logprobs.is_empty()
        || !frame.meta_info.output_top_logprobs.is_empty()
        || !frame.meta_info.input_top_logprobs.is_empty();
    let output_logprobs = group_logprobs(
        frame.meta_info.output_token_logprobs,
        frame.meta_info.output_top_logprobs,
        "output",
    )?;
    let input_logprobs = group_logprobs(
        frame.meta_info.input_token_logprobs,
        frame.meta_info.input_top_logprobs,
        "input",
    )?;
    let extras = has_extras.then_some(Box::new(GenerationOutputExtras {
        output_logprobs,
        input_logprobs,
    }));
    Ok(GenerationOutput {
        text: String::new(),
        token_ids: frame.output_ids,
        finish_reason,
        prompt_tokens: frame.meta_info.prompt_tokens,
        completion_tokens: frame.meta_info.completion_tokens,
        extras,
    })
}

fn normalize_engine_output(
    output: &mut GenerationOutput,
    emitted_tokens: &mut u64,
) -> Result<(), ResponseError> {
    let total = output.completion_tokens;
    let delta = total.checked_sub(*emitted_tokens).ok_or_else(|| {
        internal(format!(
            "engine completion token count decreased from {} to {total}",
            *emitted_tokens
        ))
    })?;
    let output_len = u64::try_from(output.token_ids.len()).unwrap_or(u64::MAX);
    let trimmed_stop_tokens = match output.finish_reason.as_ref() {
        Some(GenerationFinishReason::Stop(Some(MatchedStop::Token(_)))) => 1,
        Some(GenerationFinishReason::Stop(Some(MatchedStop::Tokens(ids)))) => {
            u64::try_from(ids.len()).unwrap_or(u64::MAX)
        }
        _ => 0,
    };
    let cumulative =
        output_len == total || output_len.checked_add(trimmed_stop_tokens) == Some(total);
    let incremental =
        output_len == delta || output_len.checked_add(trimmed_stop_tokens) == Some(delta);

    if cumulative {
        let prefix = usize::try_from(*emitted_tokens)
            .map_err(|_| internal("engine completion token count exceeds addressable memory"))?;
        if prefix > output.token_ids.len() {
            return Err(internal(format!(
                "engine returned {output_len} cumulative output token IDs after {prefix} were already emitted"
            )));
        }
        output.token_ids.drain(..prefix);
        if let Some(extras) = output.extras.as_deref_mut() {
            trim_cumulative_output_extras(extras, prefix)?;
        }
    } else if !incremental {
        return Err(internal(format!(
            "engine returned {output_len} output token IDs after reporting {delta} new completion tokens"
        )));
    }

    output.completion_tokens = delta;
    *emitted_tokens = total;
    Ok(())
}

fn trim_cumulative_output_extras(
    extras: &mut GenerationOutputExtras,
    prefix: usize,
) -> Result<(), ResponseError> {
    drain_optional_prefix(
        &mut extras.output_logprobs,
        prefix,
        "output logprob positions",
    )
}

fn drain_prefix<T>(
    values: &mut Vec<T>,
    prefix: usize,
    description: &str,
) -> Result<(), ResponseError> {
    if values.len() < prefix {
        return Err(internal(format!(
            "engine returned {} {description} values for a {prefix}-token cumulative prefix",
            values.len()
        )));
    }
    values.drain(..prefix);
    Ok(())
}

fn drain_optional_prefix<T>(
    values: &mut Vec<T>,
    prefix: usize,
    description: &str,
) -> Result<(), ResponseError> {
    if values.is_empty() {
        return Ok(());
    }
    drain_prefix(values, prefix, description)
}

fn wire_logprob((logprob, token_id, text): WireLogprob) -> TokenLogprob {
    TokenLogprob {
        logprob,
        token_id,
        text,
    }
}

fn group_logprobs(
    values: Vec<WireLogprob>,
    top_values: WireTopLogprobs,
    kind: &str,
) -> Result<Vec<PositionLogprobs>, ResponseError> {
    if !top_values.is_empty() && top_values.len() != values.len() {
        return Err(internal(format!(
            "engine returned {} {kind} top-logprob positions for {} selected-token positions",
            top_values.len(),
            values.len()
        )));
    }

    if top_values.is_empty() {
        return Ok(values
            .into_iter()
            .map(|token| PositionLogprobs {
                token: wire_logprob(token),
                top: Vec::new(),
            })
            .collect());
    }

    Ok(values
        .into_iter()
        .zip(top_values)
        .map(|(token, top)| PositionLogprobs {
            token: wire_logprob(token),
            top: top
                .unwrap_or_default()
                .into_iter()
                .map(wire_logprob)
                .collect(),
        })
        .collect())
}

fn engine_error_message(body: &str) -> Option<String> {
    serde_json::from_str::<EngineErrorEnvelope>(body)
        .ok()
        .map(|error| error.error.message)
}

fn invalid(message: impl Into<String>) -> ResponseError {
    ResponseError {
        status_code: 400,
        message: message.into(),
    }
}

fn unavailable(message: impl Into<String>) -> ResponseError {
    ResponseError {
        status_code: 503,
        message: message.into(),
    }
}

fn internal(message: impl Into<String>) -> ResponseError {
    ResponseError {
        status_code: 500,
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GenerationOptions, SamplingParams, TokenIdsRequest};
    use axum::{
        Json, Router,
        extract::State,
        response::sse::{Event, Sse},
        routing::post,
    };
    use std::convert::Infallible;
    use std::sync::{Arc, Mutex};

    fn logprob(token_id: i32, logprob: f32) -> TokenLogprob {
        TokenLogprob {
            logprob: Some(logprob),
            token_id,
            text: None,
        }
    }

    fn position(token_id: i32, value: f32, top: &[(i32, f32)]) -> PositionLogprobs {
        PositionLogprobs {
            token: logprob(token_id, value),
            top: top
                .iter()
                .map(|&(token_id, logprob)| self::logprob(token_id, logprob))
                .collect(),
        }
    }

    fn request(stop: Vec<&str>) -> GenerateRequest {
        TokenIdsRequest {
            rid: "r".into(),
            input_ids: vec![1],
            options: GenerationOptions {
                sampling_params: SamplingParams {
                    stop_strs: stop.into_iter().map(str::to_owned).collect(),
                    ..Default::default()
                },
                ..Default::default()
            },
            metadata: Default::default(),
        }
        .into()
    }

    #[test]
    fn text_stops_stay_in_the_frontend_and_token_stops_reach_the_engine() {
        let mut request = request(vec!["<eos>"]);
        request.sampling_params.stop_token_ids = Some(vec![9]);
        let matcher = take_text_stops(&mut request).unwrap();

        assert!(matcher.is_some());
        assert_eq!(request.sampling_params.stop_token_ids, Some(vec![9]));
        assert!(request.sampling_params.stop.is_empty());
    }

    #[test]
    fn regex_stops_and_min_tokens_reach_the_engine() {
        let mut request = request(vec!["END"]);
        request.sampling_params.stop_regex = vec!["[0-9]{3}".into()];
        request.sampling_params.min_new_tokens = 4;

        take_text_stops(&mut request).unwrap();

        assert_eq!(request.sampling_params.stop_regex, ["[0-9]{3}"]);
        assert_eq!(request.sampling_params.min_new_tokens, 4);
    }

    #[test]
    fn decoded_stop_matcher_handles_cross_frame_matches_and_order() {
        let mut matcher = StopStringMatcher::new(vec!["END".into(), "ND".into()], false).unwrap();

        let first = matcher.push("value E");
        assert_eq!(first.text, "value ");
        assert!(first.matched.is_none());

        let second = matcher.push("ND trailing");
        assert_eq!(second.text, "");
        assert_eq!(second.matched.as_deref(), Some("END"));
    }

    #[test]
    fn decoded_stop_matcher_uses_the_earliest_match() {
        let mut matcher =
            StopStringMatcher::new(vec!["later".into(), "first".into()], false).unwrap();

        let matched = matcher.push("first then later");

        assert_eq!(matched.text, "");
        assert_eq!(matched.matched.as_deref(), Some("first"));
    }

    #[test]
    fn no_stop_trim_includes_the_matched_text() {
        let mut matcher = StopStringMatcher::new(vec!["END".into()], true).unwrap();
        let matched = matcher.push("value END trailing");

        assert_eq!(matched.text, "value END");
        assert_eq!(matched.matched.as_deref(), Some("END"));
    }

    #[test]
    fn local_stop_truncates_token_aligned_logprobs() {
        let mut output = GenerationOutput {
            token_ids: vec![7, 8, 9],
            completion_tokens: 3,
            extras: Some(Box::new(GenerationOutputExtras {
                output_logprobs: vec![
                    position(7, -0.1, &[(7, -0.1), (6, -1.0)]),
                    position(8, -0.2, &[(8, -0.2)]),
                    position(9, -0.3, &[(9, -0.3)]),
                ],
                ..Default::default()
            })),
            ..Default::default()
        };

        truncate_output(&mut output, 2).unwrap();

        assert_eq!(output.token_ids, [7, 8]);
        assert_eq!(output.completion_tokens, 2);
        let extras = output.extras.unwrap();
        assert_eq!(extras.output_logprobs.len(), 2);
        assert_eq!(extras.output_logprobs[0].top.len(), 2);
        assert_eq!(extras.output_logprobs[1].top.len(), 1);
        assert_eq!(extras.output_logprobs[1].token.token_id, 8);
    }

    #[test]
    fn text_stops_are_matched_on_contextual_decoder_output() {
        let tokenizer = tiny_tokenizer();
        let token_ids = tokenizer
            .encode("hello")
            .unwrap()
            .token_ids()
            .iter()
            .map(|&id| id as i32)
            .collect::<Vec<_>>();
        let mut expected_decoder = tokenizer.decode_stream(&[65], true);
        let mut decoded = String::new();
        for &id in &token_ids {
            if let Some(delta) = expected_decoder.step(id as u32).unwrap() {
                decoded.push_str(&delta);
            }
        }
        assert!(!decoded.is_empty());

        let mut decoder = tokenizer.decode_stream(&[65], true);
        let mut output = GenerationOutput {
            token_ids,
            completion_tokens: 1,
            ..Default::default()
        };
        let mut matcher = StopStringMatcher::new(vec![decoded.clone()], false).unwrap();

        let matched = decode_output(&mut decoder, &mut output, Some(&mut matcher)).unwrap();

        assert_eq!(matched.as_deref(), Some(decoded.as_str()));
        assert!(output.text.is_empty());
    }

    #[test]
    fn empty_stop_matches_after_the_first_generated_token() {
        let tokenizer = tiny_tokenizer();
        let mut decoder = tokenizer.decode_stream(&[65], true);
        let mut output = GenerationOutput {
            token_ids: vec![104, 101],
            completion_tokens: 2,
            ..Default::default()
        };
        let mut matcher = StopStringMatcher::new(vec!["never".into(), String::new()], false)
            .expect("the empty stop must remain active");

        let matched = decode_output(&mut decoder, &mut output, Some(&mut matcher)).unwrap();

        assert_eq!(matched.as_deref(), Some(""));
        assert_eq!(output.token_ids, [104]);
        assert_eq!(output.completion_tokens, 1);
        assert_eq!(output.text, "h");
    }

    #[test]
    fn sse_parser_handles_split_crlf_and_lf_frames() {
        let mut parser = SseParser::default();
        assert!(parser.push(b"data: {\"a\":1}\r\n").is_empty());
        assert_eq!(
            parser.push(b"\r\ndata: [DONE]\n\n"),
            ["{\"a\":1}", "[DONE]"]
        );
    }

    #[test]
    fn sse_parser_uses_the_earliest_mixed_delimiter() {
        let mut parser = SseParser::default();

        let payloads = parser.push(b"data: {\"a\":1}\n\ndata: {\"b\":2}\r\n\r\n");

        assert_eq!(payloads, ["{\"a\":1}", "{\"b\":2}"]);
    }

    #[test]
    fn engine_frame_maps_tokens_usage_finish_and_logprobs() {
        let output = parse_engine_frame(
            r#"{
                "output_ids":[7],
                "meta_info":{
                    "prompt_tokens":3,
                    "completion_tokens":1,
                    "finish_reason":{"type":"stop","matched":9},
                    "output_token_logprobs":[[-0.25,7,null]],
                    "output_top_logprobs":[[[-0.25,7,null],[-1.0,8,null]]]
                }
            }"#,
        )
        .unwrap();
        assert_eq!(output.token_ids, [7]);
        assert_eq!(output.prompt_tokens, 3);
        assert_eq!(output.completion_tokens, 1);
        assert_eq!(
            output.finish_reason,
            Some(GenerationFinishReason::Stop(Some(MatchedStop::Token(9))))
        );
        let extras = output.extras.unwrap();
        assert_eq!(extras.output_logprobs.len(), 1);
        assert_eq!(extras.output_logprobs[0].token.token_id, 7);
        assert_eq!(extras.output_logprobs[0].top.len(), 2);
    }

    #[test]
    fn engine_frame_rejects_misaligned_logprob_positions() {
        let error = parse_engine_frame(
            r#"{
                "output_ids":[7,8],
                "meta_info":{
                    "completion_tokens":2,
                    "output_token_logprobs":[[-0.25,7,null],[-0.5,8,null]],
                    "output_top_logprobs":[[[-0.25,7,null]]]
                }
            }"#,
        )
        .unwrap_err();

        assert_eq!(error.status_code, 500);
        assert_eq!(
            error.message,
            "engine returned 1 output top-logprob positions for 2 selected-token positions"
        );
    }

    #[test]
    fn engine_error_frame_preserves_status_and_message() {
        let error = parse_engine_frame(
            r#"{"error":{"message":"too long","type":"BadRequestError","code":400}}"#,
        )
        .unwrap_err();
        assert_eq!(error.status_code, 400);
        assert_eq!(error.message, "too long");
    }

    #[test]
    fn coded_abort_frame_preserves_status_and_message() {
        let error = parse_engine_frame(
            r#"{"output_ids":[],"meta_info":{"finish_reason":{"type":"abort","status_code":503,"message":"out of memory"}}}"#,
        )
        .unwrap_err();

        assert_eq!(error.status_code, 503);
        assert_eq!(error.message, "out of memory");
    }

    #[test]
    fn uncoded_abort_frame_remains_a_finish_reason() {
        let output = parse_engine_frame(
            r#"{"output_ids":[],"meta_info":{"finish_reason":{"type":"abort","status_code":null,"message":"cancelled"}}}"#,
        )
        .unwrap();

        assert_eq!(output.finish_reason, Some(GenerationFinishReason::Abort));
    }

    #[test]
    fn cumulative_engine_frames_become_deltas() {
        let mut emitted_tokens = 1;
        let mut output = GenerationOutput {
            token_ids: vec![7, 8],
            completion_tokens: 2,
            extras: Some(Box::new(GenerationOutputExtras {
                output_logprobs: vec![
                    position(7, -0.5, &[(7, -0.5), (9, -1.0)]),
                    position(8, -0.25, &[(8, -0.25)]),
                ],
                ..Default::default()
            })),
            ..Default::default()
        };

        normalize_engine_output(&mut output, &mut emitted_tokens).unwrap();

        assert_eq!(output.token_ids, [8]);
        assert_eq!(output.completion_tokens, 1);
        assert_eq!(emitted_tokens, 2);
        let extras = output.extras.unwrap();
        assert_eq!(extras.output_logprobs.len(), 1);
        assert_eq!(extras.output_logprobs[0].token.token_id, 8);
        assert_eq!(extras.output_logprobs[0].top.len(), 1);
    }

    #[test]
    fn token_stops_may_be_trimmed_from_incremental_or_cumulative_frames() {
        for token_ids in [vec![], vec![7]] {
            let mut emitted_tokens = 1;
            let mut output = GenerationOutput {
                token_ids,
                completion_tokens: 2,
                finish_reason: Some(GenerationFinishReason::Stop(Some(MatchedStop::Token(9)))),
                ..Default::default()
            };

            normalize_engine_output(&mut output, &mut emitted_tokens).unwrap();

            assert!(output.token_ids.is_empty());
            assert_eq!(output.completion_tokens, 1);
            assert_eq!(emitted_tokens, 2);
        }
    }

    #[test]
    fn inconsistent_engine_token_counts_are_rejected() {
        let mut emitted_tokens = 2;
        let mut output = GenerationOutput {
            token_ids: vec![7, 8],
            completion_tokens: 3,
            ..Default::default()
        };

        let error = normalize_engine_output(&mut output, &mut emitted_tokens).unwrap_err();

        assert_eq!(error.status_code, 500);
        assert!(error.message.contains("2 output token IDs"));
        assert_eq!(emitted_tokens, 2);
    }

    #[derive(Clone)]
    struct EngineState {
        requests: Arc<Mutex<Vec<serde_json::Value>>>,
        output_ids: TokenIds,
    }

    async fn generate(
        State(state): State<EngineState>,
        Json(body): Json<serde_json::Value>,
    ) -> Sse<impl futures::Stream<Item = Result<Event, Infallible>>> {
        state.requests.lock().unwrap().push(body);
        let frame = serde_json::json!({
            "output_ids": state.output_ids,
            "meta_info": {
                "prompt_tokens": 1,
                "completion_tokens": state.output_ids.len(),
                "finish_reason": {"type": "stop", "matched": null}
            }
        })
        .to_string();
        Sse::new(futures::stream::iter([
            Ok(Event::default().data(frame)),
            Ok(Event::default().data("[DONE]")),
        ]))
    }

    async fn incremental_generate() -> Sse<impl futures::Stream<Item = Result<Event, Infallible>>> {
        let frame = |completion_tokens, finish_reason: serde_json::Value| {
            Event::default().data(
                serde_json::json!({
                    "output_ids": [104],
                    "meta_info": {
                        "prompt_tokens": 1,
                        "completion_tokens": completion_tokens,
                        "finish_reason": finish_reason,
                    }
                })
                .to_string(),
            )
        };
        Sse::new(futures::stream::iter([
            Ok(frame(1, serde_json::Value::Null)),
            Ok(frame(2, serde_json::json!({"type": "length", "length": 2}))),
            Ok(Event::default().data("[DONE]")),
        ]))
    }

    fn tiny_tokenizer() -> dynamo_tokenizers::Tokenizer {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../experimental/sgl-router/tests/fixtures/tiny_tokenizer.json");
        dynamo_tokenizers::Tokenizer::from_file_with_options(
            path.to_str().unwrap(),
            dynamo_tokenizers::TokenizerOptions {
                add_special_tokens: false,
            },
        )
        .unwrap()
    }

    #[test]
    fn engine_origins_are_validated_and_joined_during_client_construction() {
        let tokenizer = tiny_tokenizer();
        for invalid_url in [
            "127.0.0.1:30001",
            "ftp://engine.example",
            "http://user@engine.example",
            "http://engine.example/base",
            "http://engine.example?query",
            "http://engine.example#fragment",
        ] {
            let error = match HttpGenerateClient::new(invalid_url, tokenizer.clone()) {
                Ok(_) => panic!("{invalid_url:?} must be rejected"),
                Err(error) => error,
            };
            assert!(error.contains("invalid engine URL"));
        }

        let client = HttpGenerateClient::new("http://engine.example:30001/", tokenizer).unwrap();
        assert_eq!(
            client.generate_url.as_str(),
            "http://engine.example:30001/generate"
        );
        assert_eq!(
            client.health_url.as_str(),
            "http://engine.example:30001/health"
        );
    }

    #[tokio::test]
    async fn backend_posts_token_ids_and_decodes_the_engine_stream() {
        let tokenizer = tiny_tokenizer();
        let output_ids = tokenizer
            .encode("hello")
            .unwrap()
            .token_ids()
            .iter()
            .map(|&id| id as i32)
            .collect::<Vec<_>>();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(
            axum::serve(
                listener,
                Router::new()
                    .route("/generate", post(generate))
                    .with_state(EngineState {
                        requests: requests.clone(),
                        output_ids: output_ids.clone(),
                    }),
            )
            .into_future(),
        );

        let client =
            HttpGenerateClient::new(format!("http://{address}"), tokenizer.clone()).unwrap();
        let request = TokenIdsRequest {
            rid: "client-request".into(),
            input_ids: vec![65],
            options: GenerationOptions {
                return_text_in_logprobs: Some(true),
                ..Default::default()
            },
            metadata: Default::default(),
        };
        let mut events = client.generate(request.into()).await.unwrap();
        let output = events.next().await.unwrap().unwrap();
        assert!(output.finish_reason.is_some());

        let mut expected_decoder = tokenizer.decode_stream(&[65], true);
        let mut expected = String::new();
        for id in output_ids {
            if let Some(delta) = expected_decoder.step(id as u32).unwrap() {
                expected.push_str(&delta);
            }
        }
        assert_eq!(output.text, expected);
        assert_eq!(output.prompt_tokens, 1);
        let request = requests.lock().unwrap().pop().unwrap();
        assert_eq!(request["rid"], "client-request");
        assert_eq!(request["input_ids"], serde_json::json!([65]));
        assert_eq!(request["stream"], true);
        assert_eq!(request["incremental_streaming_output"], true);
        assert_eq!(request["return_text_in_logprobs"], false);
        server.abort();
    }

    #[tokio::test]
    async fn incremental_engine_frames_are_forwarded_once() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(
            axum::serve(
                listener,
                Router::new().route("/generate", post(incremental_generate)),
            )
            .into_future(),
        );

        let client =
            HttpGenerateClient::new(format!("http://{address}"), tiny_tokenizer()).unwrap();
        let mut events = client
            .generate(
                TokenIdsRequest {
                    rid: "incremental".into(),
                    input_ids: vec![65],
                    options: GenerationOptions::default(),
                    metadata: Default::default(),
                }
                .into(),
            )
            .await
            .unwrap();

        let first = events.next().await.unwrap().unwrap();
        assert!(first.finish_reason.is_none());
        assert_eq!(first.token_ids, [104]);
        assert_eq!(first.completion_tokens, 1);

        let second = events.next().await.unwrap().unwrap();
        assert!(second.finish_reason.is_some());
        assert_eq!(second.token_ids, [104]);
        assert_eq!(second.completion_tokens, 1);
        assert!(events.next().await.is_none());
        server.abort();
    }

    struct DropNotice(Option<tokio::sync::oneshot::Sender<()>>);

    impl Drop for DropNotice {
        fn drop(&mut self) {
            if let Some(sender) = self.0.take() {
                let _ = sender.send(());
            }
        }
    }

    async fn slow_generate(
        State(notice): State<Arc<Mutex<Option<tokio::sync::oneshot::Sender<()>>>>>,
    ) -> Sse<impl futures::Stream<Item = Result<Event, Infallible>>> {
        let guard = DropNotice(notice.lock().unwrap().take());
        Sse::new(stream! {
            let _guard = guard;
            yield Ok(Event::default().data(serde_json::json!({
                "output_ids": [104],
                "meta_info": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "finish_reason": null
                }
            }).to_string()));
            futures::future::pending::<()>().await;
        })
    }

    #[tokio::test]
    async fn dropping_renderer_events_closes_the_engine_stream() {
        let (notice_tx, notice_rx) = tokio::sync::oneshot::channel();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(
            axum::serve(
                listener,
                Router::new()
                    .route("/generate", post(slow_generate))
                    .with_state(Arc::new(Mutex::new(Some(notice_tx)))),
            )
            .into_future(),
        );
        let client =
            HttpGenerateClient::new(format!("http://{address}"), tiny_tokenizer()).unwrap();
        let request = TokenIdsRequest {
            rid: "cancel-me".into(),
            input_ids: vec![65],
            options: GenerationOptions::default(),
            metadata: Default::default(),
        };
        let mut events = client.generate(request.into()).await.unwrap();
        assert!(events.next().await.is_some());
        drop(events);
        tokio::time::timeout(Duration::from_secs(2), notice_rx)
            .await
            .expect("engine response stream was not dropped")
            .unwrap();
        server.abort();
    }
}
