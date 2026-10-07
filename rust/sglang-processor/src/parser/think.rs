//! SGLang's `BaseReasoningFormatDetector` (`parser/reasoning_parser.py`).

use super::reasoning::ReasoningOptions;

/// The tokens one `--reasoning-parser` configures the base detector with.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ThinkConfig {
    pub start: &'static str,
    pub end: &'static str,
    pub tool_start: Option<&'static str>,
    pub tool_start_at_line_start: bool,
}

/// Streaming and one-shot `<think>` splitting, as SGLang's base detector does it.
#[derive(Debug, Clone)]
pub(super) struct ThinkDetector {
    start: &'static str,
    end: &'static str,
    tool_start: Option<&'static str>,
    tool_start_at_line_start: bool,
    in_reasoning: bool,
    stream_reasoning: bool,
    force_nonempty_content: bool,
    previous_content: String,
    buffer: String,
    streamed_reasoning_tail: String,
    stripped_think_start: bool,
    accumulated_reasoning: String,
}

impl ThinkDetector {
    pub(super) fn new(config: ThinkConfig, options: &ReasoningOptions) -> Self {
        let previous_content = options.previous_content.clone().unwrap_or_default();
        let mut in_reasoning = options.force_reasoning.unwrap_or(false);
        if previous_content.contains(config.start) {
            in_reasoning = true;
        }
        if previous_content.contains(config.end) {
            in_reasoning = false;
        }
        Self {
            start: config.start,
            end: config.end,
            tool_start: config.tool_start,
            tool_start_at_line_start: config.tool_start_at_line_start,
            in_reasoning,
            stream_reasoning: options.stream_reasoning,
            force_nonempty_content: options.force_nonempty_content,
            previous_content,
            buffer: String::new(),
            streamed_reasoning_tail: String::new(),
            stripped_think_start: false,
            accumulated_reasoning: String::new(),
        }
    }

    /// `detect_and_parse`: `(reasoning_text, normal_text)` of a whole output.
    pub(super) fn parse(&mut self, text: &str) -> (String, String) {
        let split = self.parse_impl(text);
        self.nonempty_content(split)
    }

    fn parse_impl(&mut self, text: &str) -> (String, String) {
        let in_reasoning = self.in_reasoning || text.contains(self.start);
        if !in_reasoning {
            return (String::new(), text.to_owned());
        }
        let mut processed = text;
        while let Some(rest) = processed.strip_prefix(self.start) {
            processed = rest;
        }
        if !processed.contains(self.end) && !self.previous_content.contains(self.end) {
            if let Some(idx) = self.find_tool_start(processed, "") {
                return (processed[..idx].to_owned(), processed[idx..].to_owned());
            }
            return (
                self.strip_partial_tool_start(processed, "").to_owned(),
                String::new(),
            );
        }
        match processed.split_once(self.end) {
            Some((reasoning, normal)) => (reasoning.to_owned(), normal.to_owned()),
            None => (String::new(), processed.to_owned()),
        }
    }

    /// `parse_streaming_increment`: the `(reasoning, normal)` delta of one chunk.
    pub(super) fn push(&mut self, text: &str) -> (String, String) {
        let split = self.push_impl(text);
        if self.force_nonempty_content {
            if self.in_reasoning {
                self.accumulated_reasoning.push_str(&split.0);
            } else {
                self.accumulated_reasoning.clear();
            }
        }
        split
    }

    fn push_impl(&mut self, text: &str) -> (String, String) {
        self.buffer.push_str(text);
        let mut current = self.buffer.clone();
        let partial = [Some(self.start), Some(self.end), self.tool_start]
            .into_iter()
            .flatten()
            .any(|token| token.starts_with(current.as_str()) && token != current);
        if partial {
            return Default::default();
        }
        if !self.stripped_think_start && current.contains(self.start) {
            current = current.replacen(self.start, "", 1);
            self.buffer = current.clone();
            self.stripped_think_start = true;
            self.in_reasoning = true;
        }
        if self.in_reasoning
            && let Some(end_idx) = current.find(self.end)
        {
            self.buffer.clear();
            self.in_reasoning = false;
            let normal = current[end_idx + self.end.len()..].to_owned();
            return (current[..end_idx].to_owned(), normal);
        }
        if self.in_reasoning {
            if let Some(idx) = self.find_tool_start(&current, &self.streamed_reasoning_tail) {
                self.buffer.clear();
                self.in_reasoning = false;
                return (current[..idx].to_owned(), current[idx..].to_owned());
            }
            if !self.stream_reasoning {
                return Default::default();
            }
            let mut holdback_tokens = vec![self.end];
            holdback_tokens.extend(self.tool_start);
            if !self.stripped_think_start {
                holdback_tokens.push(self.start);
            }
            let holdback = holdback_tokens
                .into_iter()
                .map(|token| ends_with_partial_token(&current, token))
                .max()
                .unwrap_or(0);
            let (reasoning, held) = current.split_at(current.len() - holdback);
            self.buffer = held.to_owned();
            if let Some(last) = reasoning.chars().last() {
                self.streamed_reasoning_tail = last.to_string();
            }
            return (reasoning.to_owned(), String::new());
        }
        self.buffer.clear();
        (String::new(), current)
    }

    /// `finish`: flush what the stream still holds when it ends.
    pub(super) fn finish(&mut self) -> (String, String) {
        if !self.in_reasoning {
            return (String::new(), std::mem::take(&mut self.buffer));
        }
        let buffer = std::mem::take(&mut self.buffer);
        let buffer = buffer
            .strip_prefix(self.start)
            .unwrap_or(&buffer)
            .to_owned();
        let tail = self.streamed_reasoning_tail.clone();
        let buffer = self.strip_partial_tool_start(&buffer, &tail).to_owned();
        if self.force_nonempty_content {
            let normal = std::mem::take(&mut self.accumulated_reasoning) + &buffer;
            return (String::new(), normal);
        }
        (buffer, String::new())
    }

    fn nonempty_content(&self, (reasoning, normal): (String, String)) -> (String, String) {
        match self.force_nonempty_content && normal.is_empty() {
            true => (normal, reasoning),
            false => (reasoning, normal),
        }
    }

    /// The first tool-start token that interrupts reasoning; `preceded_by` is
    /// the character before `text`.
    fn find_tool_start(&self, text: &str, preceded_by: &str) -> Option<usize> {
        let token = self.tool_start?;
        let mut from = 0;
        while let Some(found) = text[from..].find(token) {
            let idx = from + found;
            let prev = text[..idx].chars().last().map(String::from);
            let prev = prev.as_deref().unwrap_or(preceded_by);
            if !self.tool_start_at_line_start || prev.is_empty() || prev == "\n" {
                return Some(idx);
            }
            from = idx + token.chars().next().map_or(1, char::len_utf8);
        }
        None
    }

    /// Drop a trailing partial tool-start token that begins a line.
    fn strip_partial_tool_start<'a>(&self, text: &'a str, preceded_by: &str) -> &'a str {
        let Some(token) = self.tool_start.filter(|_| self.tool_start_at_line_start) else {
            return text;
        };
        let n = ends_with_partial_token(text, token);
        if n == 0 {
            return text;
        }
        let start = text.len() - n;
        let prev = text[..start].chars().last().map(String::from);
        match prev.as_deref().unwrap_or(preceded_by) {
            "" | "\n" => &text[..start],
            _ => text,
        }
    }
}

/// Byte length of the longest suffix of `buffer` that is a strict prefix of `token`.
fn ends_with_partial_token(buffer: &str, token: &str) -> usize {
    let max = buffer
        .chars()
        .count()
        .min(token.chars().count().saturating_sub(1));
    let starts: Vec<usize> = buffer.char_indices().map(|(i, _)| i).collect();
    (1..=max)
        .rev()
        .map(|n| starts[starts.len() - n])
        .find(|&start| token.starts_with(&buffer[start..]))
        .map_or(0, |start| buffer.len() - start)
}
