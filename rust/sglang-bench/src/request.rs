//! One request's lifetime: build the body, post it, and read the streamed
//! response into per-token timings.
//!
//! Each request is an independent task, so the response streams are parsed on
//! whichever Tokio worker thread polls them. That is the one structural
//! difference from the Python script, where a single asyncio thread parses
//! every stream and becomes the bottleneck well before the server does.
//!
//! The three wire protocols are read exactly as their Python counterparts in
//! `serving.py` do, including where each one takes its timestamps and how it
//! attributes inter-token latency.

use std::time::Instant;

use bytes::BytesMut;
use futures::StreamExt;
use reqwest::header::HeaderMap;
use serde_json::{Map, Value};

use crate::args::Protocol;
use crate::dataset::{DatasetRow, Prompt};

/// Read buffer growth hint. The Python client raises aiohttp's read buffer to
/// 10 MB because cumulative `/generate` frames outgrow the 64 KB default under
/// load; reqwest hands us whatever the socket yields, so the only sizing
/// decision left is this crate's line buffer.
const LINE_BUFFER_HINT: usize = 64 * 1024;

#[derive(Clone, Debug, Default)]
pub struct RequestOutput {
    pub generated_text: String,
    pub success: bool,
    /// Seconds from the request's first byte out to its last chunk in.
    pub latency: f64,
    pub ttft: f64,
    /// One entry per generated token after the first.
    pub itl: Vec<f64>,
    /// Prompt length as the server reported it, else the dataset row's.
    pub prompt_len: usize,
    pub error: String,
    pub output_len: usize,
    /// Request start, in seconds from this process's time base. The peak
    /// windows in `metrics` place every request on this one timeline.
    pub start_time: f64,
    /// The dataset row's lengths. The input totals sum these, so one row
    /// counts once even where the server reports its own prompt length.
    pub dataset_prompt_len: usize,
    pub dataset_text_prompt_len: usize,
    pub dataset_vision_prompt_len: usize,
}

/// Everything about a request body that does not vary per prompt.
#[derive(Clone, Debug)]
pub struct RequestTemplate {
    pub protocol: Protocol,
    pub api_url: String,
    pub model: String,
    pub stream: bool,
    pub temperature: f64,
    pub top_p: f64,
    pub ignore_eos: bool,
    pub return_logprob: bool,
    pub top_logprobs_num: u32,
    pub logprob_start_len: i64,
    pub extra_body: Map<String, Value>,
    pub headers: HeaderMap,
}

impl RequestTemplate {
    /// The JSON body for one row, with `--extra-request-body` merged last so a
    /// caller can override any default this builds.
    fn body(&self, row: &DatasetRow, output_len: usize) -> Value {
        let mut body = match self.protocol {
            Protocol::SglangGenerate => {
                let mut sampling = serde_json::json!({
                    "temperature": self.temperature,
                    "max_new_tokens": output_len,
                    "ignore_eos": self.ignore_eos,
                });
                if self.top_p < 1.0 {
                    sampling["top_p"] = serde_json::json!(self.top_p);
                }
                let mut body = serde_json::json!({
                    "sampling_params": sampling,
                    "stream": self.stream,
                    "return_logprob": self.return_logprob,
                    "logprob_start_len": self.logprob_start_len,
                });
                match &row.prompt {
                    Prompt::Text(text) => body["text"] = serde_json::json!(text),
                    Prompt::TokenIds(ids) => body["input_ids"] = serde_json::json!(ids),
                }
                if self.top_logprobs_num > 0 {
                    body["top_logprobs_num"] = serde_json::json!(self.top_logprobs_num);
                }
                body
            }
            Protocol::OpenaiCompletions => {
                let prompt = match &row.prompt {
                    Prompt::Text(text) => serde_json::json!(text),
                    Prompt::TokenIds(ids) => serde_json::json!(ids),
                };
                let mut body = serde_json::json!({
                    "model": self.model,
                    "prompt": prompt,
                    "best_of": 1,
                    "max_tokens": output_len,
                    "stream": self.stream,
                });
                if !self.extra_body.contains_key("temperature") {
                    body["temperature"] = serde_json::json!(0.0);
                }
                if !self.extra_body.contains_key("ignore_eos") {
                    body["ignore_eos"] = serde_json::json!(self.ignore_eos);
                }
                if self.return_logprob && self.top_logprobs_num > 0 {
                    body["logprobs"] = serde_json::json!(self.top_logprobs_num);
                }
                body
            }
            Protocol::OpenaiChat => {
                let Prompt::Text(text) = &row.prompt else {
                    unreachable!("--tokenize-prompt is rejected for chat backends");
                };
                let mut body = serde_json::json!({
                    "model": self.model,
                    "messages": [{"role": "user", "content": text}],
                    "max_completion_tokens": output_len,
                    "stream": self.stream,
                });
                if !self.extra_body.contains_key("temperature") {
                    body["temperature"] = serde_json::json!(0.0);
                }
                if !self.extra_body.contains_key("ignore_eos") {
                    body["ignore_eos"] = serde_json::json!(self.ignore_eos);
                }
                body
            }
        };

        let Some(object) = body.as_object_mut() else {
            unreachable!("every branch builds a JSON object");
        };
        for (key, value) in &self.extra_body {
            object.insert(key.clone(), value.clone());
        }
        body
    }
}

/// Send one request and read its response to completion.
///
/// Never returns `Err`: a transport failure, a non-200 status, or a malformed
/// stream is recorded on the output as the Python script records it, so one
/// bad request does not end the run.
pub async fn send(
    http: &reqwest::Client,
    template: &RequestTemplate,
    row: &DatasetRow,
    base: Instant,
) -> RequestOutput {
    let mut output = RequestOutput {
        prompt_len: row.prompt_len,
        output_len: row.output_len,
        dataset_prompt_len: row.prompt_len,
        dataset_text_prompt_len: row.text_prompt_len,
        dataset_vision_prompt_len: row.vision_prompt_len,
        ..Default::default()
    };

    let body = template.body(row, row.output_len);
    let start = Instant::now();
    output.start_time = start.duration_since(base).as_secs_f64();

    let response = match http
        .post(&template.api_url)
        .headers(template.headers.clone())
        .json(&body)
        .send()
        .await
    {
        Ok(response) => response,
        Err(error) => {
            output.error = error.to_string();
            return output;
        }
    };

    let status = response.status();
    if !status.is_success() {
        let reason = status.canonical_reason().unwrap_or("");
        let text = response.text().await.unwrap_or_default();
        output.error = format!("{reason}: {text}");
        return output;
    }

    // A non-streaming chat reply is one JSON object, not a stream of deltas,
    // so it is read whole; the other protocols' non-streaming replies are a
    // single frame that the streaming parser already handles.
    if !template.stream && template.protocol == Protocol::OpenaiChat {
        match response.json::<Value>().await {
            Ok(data) => {
                finish_non_streaming_chat(&data, start.elapsed().as_secs_f64(), &mut output)
            }
            Err(error) => output.error = format!("response was not JSON: {error}"),
        }
        return output;
    }

    match read_stream(response, template.protocol, start, &mut output).await {
        Ok(()) => output.success = true,
        Err(error) => {
            output.success = false;
            output.error = error;
        }
    }
    output
}

/// Consume the response body line by line, feeding each line to the
/// protocol's parser. aiohttp's `async for chunk in response.content` yields
/// newline-delimited pieces, which is what the Python parsers assume, so the
/// framing is reproduced here rather than parsing raw socket chunks.
async fn read_stream(
    response: reqwest::Response,
    protocol: Protocol,
    start: Instant,
    output: &mut RequestOutput,
) -> Result<(), String> {
    let mut state = StreamState::new(protocol, output.output_len);
    let mut body = response.bytes_stream();
    let mut buffer = BytesMut::with_capacity(LINE_BUFFER_HINT);
    // Where the newline search left off. Without it a long cumulative frame
    // reassembled from many socket reads would be rescanned from the start on
    // every read, which is quadratic in the frame size.
    let mut scanned = 0usize;

    while let Some(chunk) = body.next().await {
        let chunk = chunk.map_err(|error| format!("stream read failed: {error}"))?;
        buffer.extend_from_slice(&chunk);
        while let Some(offset) = buffer[scanned..].iter().position(|&byte| byte == b'\n') {
            let end = scanned + offset;
            let line = buffer.split_to(end + 1);
            scanned = 0;
            state.feed(&line[..end], start, output);
        }
        scanned = buffer.len();
    }
    if !buffer.is_empty() {
        state.feed(&buffer, start, output);
    }

    state.finish(output)
}

/// The per-request parse state, one arm per wire protocol.
struct StreamState {
    protocol: Protocol,
    /// Latency of the most recent chunk, which becomes the request's latency.
    /// `None` means nothing arrived, which the Python script surfaces as a
    /// failure (its `latency` local stays unbound).
    latency: Option<f64>,
    /// Instant of the last counted token, for the next inter-token latency.
    most_recent: Instant,
    /// Tokens counted so far, for the `/generate` cumulative frames.
    last_output_len: usize,
    /// Falls back to the requested length until the server reports one.
    output_len: usize,
    generated_text: String,
    ttft: Option<f64>,
}

impl StreamState {
    fn new(protocol: Protocol, requested_output_len: usize) -> Self {
        Self {
            protocol,
            latency: None,
            most_recent: Instant::now(),
            last_output_len: 0,
            output_len: requested_output_len,
            generated_text: String::new(),
            ttft: None,
        }
    }

    fn feed(&mut self, line: &[u8], start: Instant, output: &mut RequestOutput) {
        let line = trim(line);
        if line.is_empty() {
            return;
        }
        let payload = line.strip_prefix(b"data: ").unwrap_or(line);
        self.latency = Some(start.elapsed().as_secs_f64());
        if payload == b"[DONE]" {
            return;
        }
        let Ok(data) = serde_json::from_slice::<Value>(payload) else {
            // A frame this client cannot read is not the server's failure to
            // report; skip it as the Python parsers' `json.loads` would raise
            // and end the request.
            return;
        };
        record_server_prompt_len(&data, output);

        match self.protocol {
            Protocol::SglangGenerate => self.feed_generate(&data, start, output),
            Protocol::OpenaiCompletions => self.feed_completions(&data, start, output),
            Protocol::OpenaiChat => self.feed_chat(&data, start, output),
        }
    }

    /// `/generate`: every frame carries the cumulative text and the running
    /// token count, so a frame's inter-token latency is spread evenly over the
    /// tokens it added.
    fn feed_generate(&mut self, data: &Value, start: Instant, output: &mut RequestOutput) {
        let Some(text) = data.get("text").and_then(Value::as_str) else {
            return;
        };
        if text.is_empty() {
            return;
        }
        let timestamp = Instant::now();
        self.generated_text.clear();
        self.generated_text.push_str(text);
        let completion_tokens = data
            .get("meta_info")
            .and_then(|meta| meta.get("completion_tokens"))
            .and_then(Value::as_u64);
        if let Some(tokens) = completion_tokens {
            self.output_len = tokens as usize;
        }

        if self.ttft.is_none() {
            self.ttft = Some(timestamp.duration_since(start).as_secs_f64());
        } else {
            let new_tokens = self.output_len.saturating_sub(self.last_output_len);
            if new_tokens == 0 {
                return;
            }
            let gap = timestamp.duration_since(self.most_recent).as_secs_f64();
            let per_token = gap / new_tokens as f64;
            output
                .itl
                .extend(std::iter::repeat_n(per_token, new_tokens));
        }
        self.most_recent = timestamp;
        self.last_output_len = self.output_len;
    }

    /// `/v1/completions`: each frame carries one delta, so each frame after
    /// the first is one inter-token latency.
    fn feed_completions(&mut self, data: &Value, start: Instant, output: &mut RequestOutput) {
        let text = data
            .get("choices")
            .and_then(|choices| choices.get(0))
            .and_then(|choice| choice.get("text"))
            .and_then(Value::as_str)
            .unwrap_or_default();
        if text.is_empty() {
            return;
        }
        self.count_delta(text, start, output);
        if let Some(tokens) = completion_tokens(data) {
            self.output_len = tokens;
        }
    }

    /// `/v1/chat/completions`: as completions, except the usage block may
    /// arrive on a frame with no choices, and reasoning models stream their
    /// thoughts in a sibling field that counts as content.
    fn feed_chat(&mut self, data: &Value, start: Instant, output: &mut RequestOutput) {
        if let Some(tokens) = completion_tokens(data) {
            self.output_len = tokens;
        }
        let Some(delta) = data
            .get("choices")
            .and_then(|choices| choices.get(0))
            .and_then(|choice| choice.get("delta"))
        else {
            return;
        };
        let content = chat_content(delta);
        if content.is_empty() {
            return;
        }
        self.count_delta(&content, start, output);
    }

    /// The shared delta accounting for the two OpenAI protocols.
    fn count_delta(&mut self, text: &str, start: Instant, output: &mut RequestOutput) {
        let timestamp = Instant::now();
        if self.ttft.is_none() {
            self.ttft = Some(timestamp.duration_since(start).as_secs_f64());
        } else {
            output
                .itl
                .push(timestamp.duration_since(self.most_recent).as_secs_f64());
        }
        self.most_recent = timestamp;
        self.generated_text.push_str(text);
    }

    /// Commit the parse to the output, or report why the response was unusable.
    fn finish(self, output: &mut RequestOutput) -> Result<(), String> {
        let Some(latency) = self.latency else {
            return Err("the response body was empty".to_owned());
        };
        output.generated_text = self.generated_text;
        output.latency = latency;
        output.output_len = self.output_len;
        // A 200 that produced no token is still a failed request: with no TTFT
        // there is nothing to report, and counting it would dilute every
        // latency percentile with a zero.
        match self.ttft {
            Some(ttft) => {
                output.ttft = ttft;
                Ok(())
            }
            None => Err("the response contained no generated text".to_owned()),
        }
    }
}

/// A non-streaming chat reply, whose content sits under `message` rather than
/// arriving as `delta` frames. TTFT is the whole latency, as in Python.
fn finish_non_streaming_chat(data: &Value, latency: f64, output: &mut RequestOutput) {
    record_server_prompt_len(data, output);
    let message = data
        .get("choices")
        .and_then(|choices| choices.get(0))
        .and_then(|choice| choice.get("message"));
    output.generated_text = message.map(chat_content).unwrap_or_default();
    output.latency = latency;
    output.ttft = latency;
    if let Some(tokens) = completion_tokens(data) {
        output.output_len = tokens;
    }
    output.success = true;
}

/// Reasoning models stream their thoughts in `reasoning_content`, and vLLM's
/// Kimi parser uses `reasoning`. Prefer the standard spelling so a server
/// exposing both aliases does not count the same tokens twice.
fn chat_content(message: &Value) -> String {
    let reasoning = message
        .get("reasoning_content")
        .and_then(Value::as_str)
        .or_else(|| message.get("reasoning").and_then(Value::as_str))
        .unwrap_or_default();
    let content = message
        .get("content")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let mut combined = String::with_capacity(reasoning.len() + content.len());
    combined.push_str(reasoning);
    combined.push_str(content);
    combined
}

fn completion_tokens(data: &Value) -> Option<usize> {
    data.get("usage")?
        .get("completion_tokens")?
        .as_u64()
        .map(|tokens| tokens as usize)
}

/// Take the prompt length from the server, the only side that knows it:
/// `usage.prompt_tokens` on the OpenAI routes, `meta_info.prompt_tokens` on
/// the native one. A server that reports neither leaves the row's value.
fn record_server_prompt_len(data: &Value, output: &mut RequestOutput) {
    let reported = data.get("usage").or_else(|| data.get("meta_info"));
    let Some(tokens) = reported
        .and_then(|reported| reported.get("prompt_tokens"))
        .and_then(Value::as_u64)
    else {
        return;
    };
    if tokens > 0 {
        output.prompt_len = tokens as usize;
    }
}

fn trim(line: &[u8]) -> &[u8] {
    let start = line
        .iter()
        .position(|byte| !byte.is_ascii_whitespace())
        .unwrap_or(line.len());
    let end = line
        .iter()
        .rposition(|byte| !byte.is_ascii_whitespace())
        .map_or(start, |index| index + 1);
    &line[start..end]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::DatasetRow;

    fn template(protocol: Protocol) -> RequestTemplate {
        RequestTemplate {
            protocol,
            api_url: "http://localhost/x".into(),
            model: "m".into(),
            stream: true,
            temperature: 0.0,
            top_p: 1.0,
            ignore_eos: true,
            return_logprob: false,
            top_logprobs_num: 0,
            logprob_start_len: -1,
            extra_body: Map::new(),
            headers: HeaderMap::new(),
        }
    }

    fn row(prompt: Prompt) -> DatasetRow {
        DatasetRow {
            prompt,
            prompt_len: 7,
            output_len: 16,
            text_prompt_len: 7,
            vision_prompt_len: 0,
        }
    }

    fn feed_all(protocol: Protocol, lines: &[&str]) -> RequestOutput {
        let mut output = RequestOutput {
            output_len: 16,
            ..Default::default()
        };
        let start = Instant::now();
        let mut state = StreamState::new(protocol, 16);
        for line in lines {
            state.feed(line.as_bytes(), start, &mut output);
        }
        let result = state.finish(&mut output);
        output.success = result.is_ok();
        if let Err(error) = result {
            output.error = error;
        }
        output
    }

    /// `/generate` sends `text` and `sampling_params`, and the token-id form
    /// switches the key rather than the value's type.
    #[test]
    fn generate_body_matches_the_native_route() {
        let template = template(Protocol::SglangGenerate);
        let body = template.body(&row(Prompt::Text("hi".into())), 8);
        assert_eq!(body["text"], "hi");
        assert_eq!(body["sampling_params"]["max_new_tokens"], 8);
        assert_eq!(body["sampling_params"]["ignore_eos"], true);
        assert_eq!(body["stream"], true);
        // top_p is only sent when it narrows the distribution.
        assert!(body["sampling_params"].get("top_p").is_none());

        let ids = template.body(&row(Prompt::TokenIds(vec![1, 2])), 8);
        assert_eq!(ids["input_ids"], serde_json::json!([1, 2]));
        assert!(ids.get("text").is_none());
    }

    /// `--extra-request-body` is merged last, so it overrides a default this
    /// crate chose, and suppresses the OpenAI temperature/ignore_eos defaults.
    #[test]
    fn extra_body_overrides_every_default() {
        let mut template = template(Protocol::OpenaiCompletions);
        template.extra_body =
            serde_json::from_str(r#"{"temperature": 0.9, "max_tokens": 3, "ignore_eos": false}"#)
                .unwrap();
        let body = template.body(&row(Prompt::Text("hi".into())), 8);
        assert_eq!(body["temperature"], 0.9);
        assert_eq!(body["max_tokens"], 3);
        assert_eq!(body["ignore_eos"], false);
    }

    #[test]
    fn chat_body_wraps_the_prompt_in_one_user_message() {
        let body = template(Protocol::OpenaiChat).body(&row(Prompt::Text("hi".into())), 8);
        assert_eq!(body["messages"][0]["role"], "user");
        assert_eq!(body["messages"][0]["content"], "hi");
        assert_eq!(body["max_completion_tokens"], 8);
    }

    /// The native route's frames are cumulative, so a frame that advances the
    /// token count by k contributes k inter-token latencies, each the frame's
    /// gap divided by k. Reporting one latency per frame instead would
    /// understate ITL on a server that batches tokens into a frame.
    #[test]
    fn generate_spreads_a_frame_gap_over_the_tokens_it_added() {
        let output = feed_all(
            Protocol::SglangGenerate,
            &[
                r#"data: {"text":"a","meta_info":{"completion_tokens":1,"prompt_tokens":5}}"#,
                r#"data: {"text":"abcd","meta_info":{"completion_tokens":4}}"#,
                "data: [DONE]",
            ],
        );
        assert!(output.success, "{}", output.error);
        assert_eq!(output.generated_text, "abcd");
        assert_eq!(output.output_len, 4);
        // Three tokens arrived on the second frame, so three equal ITLs.
        assert_eq!(output.itl.len(), 3);
        assert!((output.itl[0] - output.itl[2]).abs() < 1e-12);
        // The server's prompt length wins over the row's.
        assert_eq!(output.prompt_len, 5);
        assert!(output.ttft > 0.0);
    }

    /// A frame that repeats the token count adds no latency sample.
    #[test]
    fn generate_ignores_a_frame_that_added_no_token() {
        let output = feed_all(
            Protocol::SglangGenerate,
            &[
                r#"data: {"text":"a","meta_info":{"completion_tokens":1}}"#,
                r#"data: {"text":"a","meta_info":{"completion_tokens":1}}"#,
                r#"data: {"text":"ab","meta_info":{"completion_tokens":2}}"#,
            ],
        );
        assert_eq!(output.itl.len(), 1);
        assert_eq!(output.output_len, 2);
    }

    /// The OpenAI routes send deltas, so the text concatenates and each frame
    /// after the first is exactly one inter-token latency.
    #[test]
    fn completions_concatenates_deltas() {
        let output = feed_all(
            Protocol::OpenaiCompletions,
            &[
                r#"data: {"choices":[{"text":"He"}],"usage":{"prompt_tokens":4}}"#,
                r#"data: {"choices":[{"text":"llo"}]}"#,
                r#"data: {"choices":[{"text":""}],"usage":{"completion_tokens":9}}"#,
                "data: [DONE]",
            ],
        );
        assert!(output.success, "{}", output.error);
        assert_eq!(output.generated_text, "Hello");
        assert_eq!(output.itl.len(), 1);
        assert_eq!(output.prompt_len, 4);
        // An empty-text usage frame does not update the length, matching the
        // Python parser, which reads usage only beside a non-empty delta.
        assert_eq!(output.output_len, 16);
    }

    /// A chat stream may close with a usage-only frame that carries no
    /// choices; its token count must still be taken.
    #[test]
    fn chat_reads_a_usage_only_final_frame() {
        let output = feed_all(
            Protocol::OpenaiChat,
            &[
                r#"data: {"choices":[{"delta":{"content":"a"}}]}"#,
                r#"data: {"choices":[{"delta":{"content":"b"}}]}"#,
                r#"data: {"choices":[],"usage":{"completion_tokens":2,"prompt_tokens":3}}"#,
                "data: [DONE]",
            ],
        );
        assert!(output.success, "{}", output.error);
        assert_eq!(output.generated_text, "ab");
        assert_eq!(output.output_len, 2);
        assert_eq!(output.prompt_len, 3);
        assert_eq!(output.itl.len(), 1);
    }

    /// Reasoning tokens count as generated content, and the standard spelling
    /// wins when a server exposes both aliases.
    #[test]
    fn chat_counts_reasoning_content_once() {
        let output = feed_all(
            Protocol::OpenaiChat,
            &[
                r#"data: {"choices":[{"delta":{"reasoning_content":"think","reasoning":"dup","content":"!"}}]}"#,
            ],
        );
        assert_eq!(output.generated_text, "think!");
    }

    /// An empty body and a 200 that generated nothing are both failures: the
    /// first has no latency to report, the second no TTFT, and counting either
    /// as a success would pull every percentile toward zero.
    #[test]
    fn a_response_without_tokens_is_a_failure() {
        let empty = feed_all(Protocol::SglangGenerate, &[]);
        assert!(!empty.success);
        assert!(empty.error.contains("empty"));

        let no_tokens = feed_all(Protocol::OpenaiChat, &["data: [DONE]"]);
        assert!(!no_tokens.success);
        assert!(no_tokens.error.contains("no generated text"));
    }

    /// SSE framing details: blank keep-alive lines are skipped, a frame with
    /// no `data: ` prefix is still parsed, and a frame this client cannot read
    /// does not abort the request.
    #[test]
    fn framing_tolerates_blank_and_bare_lines() {
        let output = feed_all(
            Protocol::OpenaiCompletions,
            &[
                "",
                "   ",
                r#"{"choices":[{"text":"x"}]}"#,
                "data: not json",
                r#"data: {"choices":[{"text":"y"}]}"#,
            ],
        );
        assert!(output.success, "{}", output.error);
        assert_eq!(output.generated_text, "xy");
    }

    #[test]
    fn non_streaming_chat_reports_latency_as_ttft() {
        let mut output = RequestOutput {
            output_len: 16,
            ..Default::default()
        };
        let data = serde_json::json!({
            "choices": [{"message": {"content": "hello"}}],
            "usage": {"completion_tokens": 2, "prompt_tokens": 6},
        });
        finish_non_streaming_chat(&data, 1.5, &mut output);
        assert!(output.success);
        assert_eq!(output.generated_text, "hello");
        assert_eq!(output.ttft, 1.5);
        assert_eq!(output.latency, 1.5);
        assert_eq!(output.output_len, 2);
        assert_eq!(output.prompt_len, 6);
        assert!(output.itl.is_empty());
    }
}
