//! SSE stream parsing for the sglang `/generate` endpoint.
//!
//! [`LineBuffer`] re-frames an arbitrary byte stream into lines (aiohttp's
//! `response.content` iterates lines, including a trailing partial line at
//! EOF). [`FrameAccumulator`] replays the exact per-line semantics of
//! `async_request_sglang_generate` in `python/sglang/benchmark/serving.py`,
//! with timestamps injected as parameters so the state machine is testable
//! without a clock.

use std::borrow::Cow;

use serde::Deserialize;

#[derive(Debug)]
pub enum FrameError {
    Json(serde_json::Error),
    /// A frame carried non-empty `text` but no `meta_info.completion_tokens`
    /// (the Python client raises `KeyError` and fails the request).
    MissingCompletionTokens,
}

impl std::fmt::Display for FrameError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FrameError::Json(e) => write!(f, "invalid SSE JSON frame: {e}"),
            FrameError::MissingCompletionTokens => {
                write!(f, "SSE frame has text but no meta_info.completion_tokens")
            }
        }
    }
}

impl std::error::Error for FrameError {}

/// Incremental byte→line splitter. Emitted lines include the trailing `\n`
/// (callers strip whitespace, matching Python's `chunk_bytes.strip()`).
#[derive(Default)]
pub struct LineBuffer {
    buf: Vec<u8>,
}

impl LineBuffer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Append `chunk` and invoke `on_line` for each complete line, short-
    /// circuiting on the first error. Consumed lines are dropped from the
    /// buffer either way.
    pub fn feed<E>(
        &mut self,
        chunk: &[u8],
        mut on_line: impl FnMut(&[u8]) -> Result<(), E>,
    ) -> Result<(), E> {
        self.buf.extend_from_slice(chunk);
        let mut start = 0;
        let mut result = Ok(());
        while let Some(pos) = memchr::memchr(b'\n', &self.buf[start..]) {
            let end = start + pos;
            let line_result = on_line(&self.buf[start..=end]);
            start = end + 1;
            if line_result.is_err() {
                result = line_result;
                break;
            }
        }
        self.buf.drain(..start);
        result
    }

    /// Trailing bytes with no final `\n` — aiohttp yields these as the last
    /// "line" at EOF (this is also how a non-stream single-JSON body arrives).
    pub fn take_remainder(&mut self) -> Option<Vec<u8>> {
        if self.buf.is_empty() {
            None
        } else {
            Some(std::mem::take(&mut self.buf))
        }
    }
}

/// The subset of a `/generate` stream frame the client reads. Unknown fields
/// (logprobs, routed experts, ...) are ignored.
#[derive(Deserialize)]
struct GenerateFrame<'a> {
    #[serde(default, borrow)]
    text: Option<Cow<'a, str>>,
    #[serde(default, borrow)]
    meta_info: Option<MetaInfo<'a>>,
}

#[derive(Deserialize)]
struct MetaInfo<'a> {
    #[serde(default)]
    completion_tokens: Option<i64>,
    #[serde(default)]
    spec_accept_length: Option<f64>,
    #[serde(default)]
    cached_tokens: Option<u64>,
    #[serde(default, borrow)]
    cached_tokens_details: Option<&'a serde_json::value::RawValue>,
}

/// Per-request stream state. All times are seconds relative to the request
/// send instant (`st` in the Python client), so `most_recent_timestamp`
/// starts at 0.0 exactly like Python's `most_recent_timestamp = st`.
pub struct FrameAccumulator {
    cache_report: bool,
    pub generated_text: String,
    /// Initialized to the requested `output_len`, overwritten by each frame's
    /// `completion_tokens` (same as the Python local `output_len`).
    pub output_len: i64,
    pub ttft: f64,
    pub itl: Vec<f64>,
    /// Restamped on every non-empty line, including the final `[DONE]`.
    pub latency: f64,
    pub spec_accept_length: f64,
    pub cached_tokens: u64,
    pub cached_tokens_details_json: Option<String>,
    most_recent_timestamp: f64,
    last_output_len: i64,
}

impl FrameAccumulator {
    pub fn new(cache_report: bool, request_output_len: i64) -> Self {
        Self {
            cache_report,
            generated_text: String::new(),
            output_len: request_output_len,
            ttft: 0.0,
            itl: Vec::new(),
            latency: 0.0,
            spec_accept_length: 0.0,
            cached_tokens: 0,
            cached_tokens_details_json: None,
            most_recent_timestamp: 0.0,
            last_output_len: 0,
        }
    }

    pub fn on_line(&mut self, line: &[u8], now_s: f64) -> Result<(), FrameError> {
        let line = line.trim_ascii();
        if line.is_empty() {
            return Ok(());
        }
        let sse_data = line.strip_prefix(b"data: ").unwrap_or(line);
        self.latency = now_s;
        if sse_data == b"[DONE]" {
            return Ok(());
        }

        let frame: GenerateFrame = serde_json::from_slice(sse_data).map_err(FrameError::Json)?;
        let meta = frame.meta_info.as_ref();
        if let Some(v) = meta.and_then(|m| m.spec_accept_length) {
            self.spec_accept_length = v;
        }
        // Python overwrites both cache fields on every parsed frame, falling
        // back to 0 / None when absent — so only the last frame's value sticks.
        if self.cache_report {
            self.cached_tokens = meta.and_then(|m| m.cached_tokens).unwrap_or(0);
            self.cached_tokens_details_json = meta
                .and_then(|m| m.cached_tokens_details)
                .map(|raw| raw.get().to_string());
        }

        let Some(text) = frame.text.as_deref().filter(|t| !t.is_empty()) else {
            return Ok(());
        };
        let completion_tokens = meta
            .and_then(|m| m.completion_tokens)
            .ok_or(FrameError::MissingCompletionTokens)?;

        self.generated_text.clear();
        self.generated_text.push_str(text);
        self.output_len = completion_tokens;

        if self.ttft == 0.0 {
            self.ttft = now_s;
        } else {
            let num_new_tokens = completion_tokens - self.last_output_len;
            if num_new_tokens == 0 {
                // Python `continue`s here without updating the timestamp or
                // last_output_len, so the gap rolls into the next chunk.
                return Ok(());
            }
            if num_new_tokens > 0 {
                let adjust_itl = (now_s - self.most_recent_timestamp) / num_new_tokens as f64;
                self.itl
                    .extend(std::iter::repeat_n(adjust_itl, num_new_tokens as usize));
            }
            // A negative delta matches Python's `[x] * negative == []`: no itl
            // entries, but the timestamp / last_output_len still advance.
        }
        self.most_recent_timestamp = now_s;
        self.last_output_len = completion_tokens;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn collect_lines(chunks: &[&[u8]]) -> Vec<Vec<u8>> {
        let mut buf = LineBuffer::new();
        let mut lines: Vec<Vec<u8>> = Vec::new();
        for chunk in chunks {
            buf.feed::<()>(chunk, |line| {
                lines.push(line.to_vec());
                Ok(())
            })
            .unwrap();
        }
        if let Some(rest) = buf.take_remainder() {
            lines.push(rest);
        }
        lines
    }

    #[test]
    fn splits_lines_across_chunk_boundaries() {
        let lines = collect_lines(&[b"da", b"ta: {\"a\":1}\n\nda", b"ta: [DONE]\n"]);
        assert_eq!(
            lines,
            vec![
                b"data: {\"a\":1}\n".to_vec(),
                b"\n".to_vec(),
                b"data: [DONE]\n".to_vec(),
            ]
        );
    }

    #[test]
    fn yields_trailing_partial_line_as_remainder() {
        let lines = collect_lines(&[b"{\"text\": \"hi\", ", b"\"meta_info\": {}}"]);
        assert_eq!(
            lines,
            vec![b"{\"text\": \"hi\", \"meta_info\": {}}".to_vec()]
        );
    }

    #[test]
    fn feed_short_circuits_on_error_but_consumes_the_line() {
        let mut buf = LineBuffer::new();
        let mut seen = 0;
        let result = buf.feed(b"one\ntwo\nthree\n", |_line| {
            seen += 1;
            if seen == 2 { Err("boom") } else { Ok(()) }
        });
        assert_eq!(result, Err("boom"));
        assert_eq!(seen, 2);
    }

    fn frame(text: &str, completion_tokens: i64) -> Vec<u8> {
        format!(
            "data: {{\"text\": {}, \"meta_info\": {{\"completion_tokens\": {}}}}}\n",
            serde_json::to_string(text).unwrap(),
            completion_tokens
        )
        .into_bytes()
    }

    #[test]
    fn ttft_itl_and_latency_match_python_semantics() {
        let mut acc = FrameAccumulator::new(false, 128);
        // CRLF + keep-alive blank lines are stripped/skipped.
        acc.on_line(b"\r\n", 0.05).unwrap();
        // First token at t=0.5.
        acc.on_line(&frame("a", 1), 0.5).unwrap();
        assert_eq!(acc.ttft, 0.5);
        assert!(acc.itl.is_empty());
        // Zero new tokens: no itl, timestamp NOT advanced.
        acc.on_line(&frame("a", 1), 0.6).unwrap();
        // 3 new tokens over the gap since t=0.5 (not 0.6): (0.9-0.5)/3 each.
        acc.on_line(&frame("abcd", 4), 0.9).unwrap();
        let expected = (0.9 - 0.5) / 3.0;
        assert_eq!(acc.itl, vec![expected; 3]);
        // Single token chunk.
        acc.on_line(&frame("abcde", 5), 1.0).unwrap();
        assert_eq!(acc.itl.len(), 4);
        assert!((acc.itl[3] - (1.0 - 0.9)).abs() < 1e-12);
        // [DONE] still restamps latency.
        acc.on_line(b"data: [DONE]\n", 1.2).unwrap();
        assert_eq!(acc.latency, 1.2);
        assert_eq!(acc.output_len, 5);
        assert_eq!(acc.generated_text, "abcde");
    }

    #[test]
    fn output_len_defaults_to_requested_len_without_text_frames() {
        let mut acc = FrameAccumulator::new(false, 77);
        acc.on_line(b"data: [DONE]\n", 0.3).unwrap();
        assert_eq!(acc.output_len, 77);
        assert_eq!(acc.ttft, 0.0);
    }

    #[test]
    fn missing_completion_tokens_is_an_error() {
        let mut acc = FrameAccumulator::new(false, 8);
        let err = acc
            .on_line(b"data: {\"text\": \"hi\", \"meta_info\": {}}\n", 0.1)
            .unwrap_err();
        assert!(matches!(err, FrameError::MissingCompletionTokens));
        let err = acc.on_line(b"data: {\"text\": \"hi\"}\n", 0.1).unwrap_err();
        assert!(matches!(err, FrameError::MissingCompletionTokens));
    }

    #[test]
    fn invalid_json_is_an_error() {
        let mut acc = FrameAccumulator::new(false, 8);
        let err = acc.on_line(b"data: {not json}\n", 0.1).unwrap_err();
        assert!(matches!(err, FrameError::Json(_)));
    }

    #[test]
    fn spec_and_cache_fields_follow_last_frame() {
        let mut acc = FrameAccumulator::new(true, 8);
        acc.on_line(
            b"data: {\"text\": \"a\", \"meta_info\": {\"completion_tokens\": 1, \
              \"spec_accept_length\": 2.5, \"cached_tokens\": 10, \
              \"cached_tokens_details\": {\"device\": 10}}}\n",
            0.1,
        )
        .unwrap();
        assert_eq!(acc.spec_accept_length, 2.5);
        assert_eq!(acc.cached_tokens, 10);
        assert_eq!(
            acc.cached_tokens_details_json.as_deref(),
            Some("{\"device\": 10}")
        );
        // A later frame without cache fields resets them (Python `.get(..., 0)`
        // overwrite), while spec_accept_length is only overwritten when present.
        acc.on_line(&frame("ab", 2), 0.2).unwrap();
        assert_eq!(acc.spec_accept_length, 2.5);
        assert_eq!(acc.cached_tokens, 0);
        assert_eq!(acc.cached_tokens_details_json, None);
    }

    #[test]
    fn unknown_fields_and_prefixless_json_are_accepted() {
        let mut acc = FrameAccumulator::new(false, 8);
        // Non-stream responses arrive as one bare JSON body (no "data: ").
        acc.on_line(
            b"{\"text\": \"hi\", \"output_ids\": [1, 2], \"meta_info\": \
              {\"completion_tokens\": 2, \"finish_reason\": {\"type\": \"length\"}, \
              \"input_token_logprobs\": [[-0.1, 5, null]]}}",
            0.4,
        )
        .unwrap();
        assert_eq!(acc.generated_text, "hi");
        assert_eq!(acc.output_len, 2);
        assert_eq!(acc.ttft, 0.4);
    }
}
