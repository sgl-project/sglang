//! Kimi-K3 (`KimiK3Detector` in `parser/reasoning_parser.py`): the XTML
//! think channel, with post-reasoning content unwrapped from the `response`
//! and `message` markers and a `tools` channel passed through raw.

use crate::kimi_k3_format::{
    MESSAGE_CLOSE, RESPONSE_CLOSE, RESPONSE_OPEN, THINK_CLOSE, THINK_OPEN, TOOLS_OPEN,
    partial_suffix_len, strip_partial_marker_suffix, strip_response_wrappers,
};
use crate::think::ReasoningOptions;

#[derive(Debug, Clone)]
pub(crate) struct KimiK3Detector {
    in_reasoning: bool,
    stream_reasoning: bool,
    force_nonempty_content: bool,
    buffer: String,
    stripped_think_start: bool,
    reasoning_done: bool,
    tools_passthrough: bool,
    stream_text: String,
    streamed_reasoning: Vec<String>,
    discard_delayed_think_close: bool,
}

impl KimiK3Detector {
    pub(crate) fn new(options: &ReasoningOptions) -> Self {
        // K3 forces reasoning unless the request says otherwise.
        let mut in_reasoning = options.force_reasoning.unwrap_or(true);
        let previous = options.previous_content.as_deref().unwrap_or_default();
        if previous.contains(THINK_OPEN) {
            in_reasoning = true;
        }
        if previous.contains(THINK_CLOSE) {
            in_reasoning = false;
        }
        Self {
            in_reasoning,
            stream_reasoning: options.stream_reasoning,
            force_nonempty_content: options.force_nonempty_content,
            buffer: String::new(),
            stripped_think_start: false,
            reasoning_done: false,
            tools_passthrough: false,
            stream_text: String::new(),
            streamed_reasoning: Vec::new(),
            discard_delayed_think_close: false,
        }
    }

    /// `detect_and_parse`.
    pub(crate) fn parse(&mut self, text: &str) -> (String, String) {
        let in_reasoning = self.in_reasoning || text.contains(THINK_OPEN);
        if !in_reasoning && !text.contains(THINK_CLOSE) {
            return (String::new(), clean_content(text));
        }
        if self.force_nonempty_content && is_skipped_think_answer(text) {
            return (String::new(), clean_content(text));
        }
        let start = text.find(THINK_OPEN).map_or(0, |i| i + THINK_OPEN.len());
        let close_idx = find_from(text, THINK_CLOSE, start);
        let tools_idx = find_from(text, TOOLS_OPEN, start);
        let channel_idx = next_channel_idx(text, start);
        if skipped_think_channel(text, start, close_idx, channel_idx) {
            let skipped = text[start..].replace(THINK_CLOSE, "");
            return (String::new(), clean_content(&skipped));
        }
        if let (Some(close), Some(tools)) = (close_idx, tools_idx)
            && tools < close
        {
            return (
                strip_partial_marker_suffix(&text[start..tools]).to_owned(),
                clean_content(&text[tools..]),
            );
        }
        let Some(close) = close_idx else {
            if let Some(channel) = channel_idx {
                return (
                    strip_partial_marker_suffix(&text[start..channel]).to_owned(),
                    clean_content(&text[channel..]),
                );
            }
            return (
                strip_partial_marker_suffix(&text[start..]).to_owned(),
                String::new(),
            );
        };
        let rest = &text[close + THINK_CLOSE.len()..];
        (text[start..close].to_owned(), clean_content(rest))
    }

    /// `parse_streaming_increment`.
    pub(crate) fn push(&mut self, text: &str) -> (String, String) {
        self.buffer.push_str(text);
        if self.force_nonempty_content {
            self.stream_text.push_str(text);
        }

        if !self.in_reasoning && !self.reasoning_done {
            if let Some(open) = self.buffer.find(THINK_OPEN) {
                self.buffer.drain(..open + THINK_OPEN.len());
                self.in_reasoning = true;
                self.stripped_think_start = true;
            } else if THINK_OPEN.starts_with(self.buffer.as_str()) {
                return Default::default();
            } else {
                self.reasoning_done = true;
            }
        }

        if !self.in_reasoning {
            return (String::new(), self.drain_content());
        }
        if !self.stripped_think_start
            && let Some(open) = self.buffer.find(THINK_OPEN)
        {
            self.buffer.drain(..open + THINK_OPEN.len());
            self.stripped_think_start = true;
        }
        let buf = self.buffer.clone();
        let close_idx = buf.find(THINK_CLOSE);
        let tools_idx = buf.find(TOOLS_OPEN);
        let channel_idx = next_channel_idx(&buf, 0);
        if skipped_think_channel(&buf, 0, close_idx, channel_idx) {
            let replay = std::mem::take(&mut self.streamed_reasoning).concat();
            self.in_reasoning = false;
            self.reasoning_done = true;
            self.discard_delayed_think_close = true;
            return (String::new(), replay + &self.drain_content());
        }
        if let Some(close) = close_idx
            && tools_idx.is_none_or(|tools| tools >= close)
        {
            self.buffer = buf[close + THINK_CLOSE.len()..].to_owned();
            self.in_reasoning = false;
            self.reasoning_done = true;
            self.streamed_reasoning.clear();
            return (buf[..close].to_owned(), self.drain_content());
        }
        if let Some(channel) = channel_idx {
            let reasoning = strip_partial_marker_suffix(&buf[..channel]).to_owned();
            self.buffer = buf[channel..].to_owned();
            self.in_reasoning = false;
            self.reasoning_done = true;
            self.streamed_reasoning.clear();
            self.tools_passthrough = buf[channel..].starts_with(TOOLS_OPEN);
            return (reasoning, self.drain_content());
        }
        if !self.stream_reasoning {
            return Default::default();
        }
        let mut markers = vec![
            THINK_CLOSE,
            TOOLS_OPEN,
            RESPONSE_OPEN,
            RESPONSE_CLOSE,
            MESSAGE_CLOSE,
        ];
        if !self.stripped_think_start {
            markers.push(THINK_OPEN);
        }
        let holdback = partial_suffix_len(&buf, &markers);
        let emit = strip_partial_marker_suffix(&buf[..buf.len() - holdback]).to_owned();
        self.buffer = buf[emit.len()..].to_owned();
        self.streamed_reasoning.push(emit.clone());
        (emit, String::new())
    }

    /// `finish`.
    pub(crate) fn finish(&mut self) -> (String, String) {
        self.streamed_reasoning.clear();
        if !self.force_nonempty_content {
            return self.base_finish();
        }
        let text = std::mem::take(&mut self.stream_text);
        if self.in_reasoning && is_skipped_think_answer(&text) {
            self.buffer.clear();
            return (String::new(), clean_content(&text));
        }
        if self.in_reasoning && !self.stream_reasoning && !self.buffer.is_empty() {
            return (std::mem::take(&mut self.buffer), String::new());
        }
        Default::default()
    }

    /// `BaseReasoningFormatDetector.finish` without `force_nonempty_content`.
    fn base_finish(&mut self) -> (String, String) {
        let buffer = std::mem::take(&mut self.buffer);
        if !self.in_reasoning {
            return (String::new(), buffer);
        }
        let buffer = buffer.strip_prefix(THINK_OPEN).unwrap_or(&buffer);
        (buffer.to_owned(), String::new())
    }

    /// `_drain_content`: the buffered content, unwrapped, holding back partial markers.
    fn drain_content(&mut self) -> String {
        if self.buffer.is_empty() {
            return String::new();
        }
        let buf = std::mem::take(&mut self.buffer);
        let delayed_close = |buf: &str| match self.discard_delayed_think_close {
            true => partial_suffix_len(buf, &[THINK_CLOSE]),
            false => 0,
        };
        let discard = |text: &str| match self.discard_delayed_think_close {
            true => text.replace(THINK_CLOSE, ""),
            false => text.to_owned(),
        };
        if self.tools_passthrough {
            let (emit, held) = buf.split_at(buf.len() - delayed_close(&buf));
            self.buffer = held.to_owned();
            return discard(emit);
        }
        if let Some(tools) = buf.find(TOOLS_OPEN) {
            let (emit, held) = buf.split_at(buf.len() - delayed_close(&buf));
            self.buffer = held.to_owned();
            let (head, tail) = emit.split_at(tools.min(emit.len()));
            let head = discard(&strip_markers(head));
            self.tools_passthrough = true;
            return head + &discard(tail);
        }
        let mut markers = vec![RESPONSE_OPEN, RESPONSE_CLOSE, MESSAGE_CLOSE, TOOLS_OPEN];
        if self.discard_delayed_think_close {
            markers.push(THINK_CLOSE);
        }
        let (emit, held) = buf.split_at(buf.len() - partial_suffix_len(&buf, &markers));
        self.buffer = held.to_owned();
        discard(&strip_markers(emit))
    }
}

/// `str.find(sub, start)`.
fn find_from(text: &str, sub: &str, start: usize) -> Option<usize> {
    text[start..].find(sub).map(|i| start + i)
}

fn strip_markers(text: &str) -> String {
    [RESPONSE_OPEN, RESPONSE_CLOSE, MESSAGE_CLOSE]
        .iter()
        .fold(text.to_owned(), |text, marker| text.replace(marker, ""))
}

/// `_next_channel_idx`: the first `response` or `tools` channel at or after `start`.
fn next_channel_idx(text: &str, start: usize) -> Option<usize> {
    [RESPONSE_OPEN, TOOLS_OPEN]
        .iter()
        .filter_map(|token| find_from(text, token, start))
        .min()
}

/// `_skipped_think_channel`: a `response` close before any think close or
/// channel proves the text was the response channel.
fn skipped_think_channel(
    text: &str,
    start: usize,
    think_close: Option<usize>,
    next_channel: Option<usize>,
) -> bool {
    find_from(text, RESPONSE_CLOSE, start).is_some_and(|close| {
        [think_close, next_channel]
            .iter()
            .all(|boundary| boundary.is_none_or(|b| close < b))
    })
}

/// `_is_skipped_think_answer`.
fn is_skipped_think_answer(text: &str) -> bool {
    [THINK_OPEN, THINK_CLOSE, "<|open|>think", "<|close|>think"]
        .iter()
        .all(|token| !text.contains(token))
        && (text.contains(RESPONSE_CLOSE) || text.contains(MESSAGE_CLOSE))
}

/// `_clean_content`: unwrap the response channel; a `tools` channel stays raw.
fn clean_content(text: &str) -> String {
    match text.find(TOOLS_OPEN) {
        Some(tools) => strip_response_wrappers(&text[..tools]) + &text[tools..],
        None => strip_response_wrappers(text),
    }
}
