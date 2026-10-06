// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Chat SSE → Messages events: message_start, content blocks, message_delta,
//! message_stop. `message_start` is sent on the first usage or content chunk.

use serde_json::{json, Value};

use super::{error_type, new_id, stop_reason, usage_from_chat, EchoContext};
use crate::protocol::sse::{data_payload, write_event, LineBuffer, SseTransducer};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Thinking,
    Text,
    Tool,
}

pub struct MessagesStream {
    echo: EchoContext,
    id: String,
    lines: LineBuffer,
    started: bool,
    terminal: bool,
    open: Option<(Kind, usize)>,
    /// Text or thinking that arrived while a tool call was open; emitted
    /// after the call so it cannot split it.
    held: Vec<(Kind, String)>,
    next_index: usize,
    usage: Option<Value>,
    finish_reason: Option<String>,
    matched_stop: Option<Value>,
}

impl MessagesStream {
    pub fn new(echo: EchoContext) -> Self {
        Self {
            echo,
            id: new_id("msg"),
            lines: LineBuffer::default(),
            started: false,
            terminal: false,
            open: None,
            held: Vec::new(),
            next_index: 0,
            usage: None,
            finish_reason: None,
            matched_stop: None,
        }
    }

    fn handle_line(&mut self, line: &[u8], out: &mut Vec<u8>) {
        if self.terminal {
            return;
        }
        let Some(chunk) = data_payload(line) else {
            return;
        };
        if let Some(err) = chunk.get("error").filter(|e| !e.is_null()) {
            let message = err
                .get("message")
                .and_then(Value::as_str)
                .or_else(|| err.as_str())
                .unwrap_or("upstream error");
            let status = err.get("code").and_then(Value::as_u64);
            let status = status.and_then(|c| u16::try_from(c).ok()).unwrap_or(500);
            self.emit_error(error_type(status), message, out);
            return;
        }
        if let Some(u) = chunk.get("usage").filter(|u| u.is_object()) {
            self.usage = Some(u.clone());
            self.ensure_started(out);
        }
        let Some(choice) = chunk.pointer("/choices/0") else {
            return;
        };
        if let Some(delta) = choice.get("delta") {
            let text = |k: &str| {
                delta
                    .get(k)
                    .and_then(Value::as_str)
                    .filter(|s| !s.is_empty())
            };
            if let Some(r) = text("reasoning_content") {
                self.text(Kind::Thinking, r, out);
            }
            if let Some(t) = text("content") {
                self.text(Kind::Text, t, out);
            }
            for call in delta
                .get("tool_calls")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                self.tool_call(call, out);
            }
        }
        if let Some(fr) = choice.get("finish_reason").and_then(Value::as_str) {
            self.finish_reason = Some(fr.to_owned());
            self.matched_stop = choice.get("matched_stop").cloned();
        }
    }

    fn ensure_started(&mut self, out: &mut Vec<u8>) {
        if self.started {
            return;
        }
        self.started = true;
        let mut usage = usage_from_chat(self.usage.as_ref());
        usage["output_tokens"] = json!(0);
        let data = json!({
            "type": "message_start",
            "message": {
                "id": self.id,
                "type": "message",
                "role": "assistant",
                "model": self.echo.model,
                "content": [],
                "stop_reason": null,
                "stop_sequence": null,
                "usage": usage,
            },
        });
        write_event(out, "message_start", &data);
    }

    fn text(&mut self, kind: Kind, s: &str, out: &mut Vec<u8>) {
        if matches!(self.open, Some((Kind::Tool, _))) {
            match self.held.last_mut() {
                Some((k, held)) if *k == kind => held.push_str(s),
                _ => self.held.push((kind, s.to_owned())),
            }
            return;
        }
        if !matches!(self.open, Some((k, _)) if k == kind) {
            self.open_block(kind, None, out);
        }
        let delta = match kind {
            Kind::Thinking => json!({"type": "thinking_delta", "thinking": s}),
            _ => json!({"type": "text_delta", "text": s}),
        };
        self.delta(delta, out);
    }

    /// SGLang names a call only on its first chunk, so a name starts a new
    /// block: the index alone can repeat across calls.
    fn tool_call(&mut self, call: &Value, out: &mut Vec<u8>) {
        let named = call
            .pointer("/function/name")
            .and_then(Value::as_str)
            .is_some_and(|n| !n.is_empty());
        if named || !matches!(self.open, Some((Kind::Tool, _))) {
            self.close_block(out);
            self.flush_held(out);
            self.open_block(Kind::Tool, Some(call), out);
        }
        if let Some(args) = call
            .pointer("/function/arguments")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
        {
            self.delta(
                json!({"type": "input_json_delta", "partial_json": args}),
                out,
            );
        }
    }

    /// Emits held text after a tool call; whitespace alone is dropped.
    fn flush_held(&mut self, out: &mut Vec<u8>) {
        for (kind, s) in std::mem::take(&mut self.held) {
            if !s.trim().is_empty() {
                self.text(kind, &s, out);
            }
        }
    }

    fn open_block(&mut self, kind: Kind, call: Option<&Value>, out: &mut Vec<u8>) {
        self.ensure_started(out);
        self.close_block(out);
        let index = self.next_index;
        self.next_index += 1;
        let block = match kind {
            Kind::Thinking => json!({"type": "thinking", "thinking": "", "signature": ""}),
            Kind::Text => json!({"type": "text", "text": ""}),
            Kind::Tool => json!({
                "type": "tool_use",
                "id": call.and_then(|c| c.get("id")).and_then(Value::as_str)
                    .map(str::to_owned).unwrap_or_else(|| new_id("toolu")),
                "name": call.and_then(|c| c.pointer("/function/name"))
                    .and_then(Value::as_str).unwrap_or(""),
                "input": {},
            }),
        };
        write_event(
            out,
            "content_block_start",
            &json!({"type": "content_block_start", "index": index, "content_block": block}),
        );
        self.open = Some((kind, index));
    }

    fn delta(&mut self, delta: Value, out: &mut Vec<u8>) {
        let Some((_, index)) = self.open else {
            return;
        };
        write_event(
            out,
            "content_block_delta",
            &json!({"type": "content_block_delta", "index": index, "delta": delta}),
        );
    }

    fn close_block(&mut self, out: &mut Vec<u8>) {
        if let Some((_, index)) = self.open.take() {
            write_event(
                out,
                "content_block_stop",
                &json!({"type": "content_block_stop", "index": index}),
            );
        }
    }

    fn emit_error(&mut self, typ: &str, message: &str, out: &mut Vec<u8>) {
        write_event(
            out,
            "error",
            &json!({"type": "error", "error": {"type": typ, "message": message}}),
        );
        self.terminal = true;
    }
}

impl SseTransducer for MessagesStream {
    fn is_terminal(&self) -> bool {
        self.terminal
    }

    fn feed(&mut self, chunk: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        let mut lines = std::mem::take(&mut self.lines);
        lines.push(chunk, |line| self.handle_line(line, &mut out));
        self.lines = lines;
        out
    }

    fn finish(&mut self) -> Vec<u8> {
        let mut out = Vec::new();
        let mut lines = std::mem::take(&mut self.lines);
        lines.flush(|line| self.handle_line(line, &mut out));
        if self.terminal {
            return out;
        }
        if self.finish_reason.is_none() {
            self.emit_error(
                "api_error",
                "upstream stream ended before the message completed",
                &mut out,
            );
            return out;
        }
        self.ensure_started(&mut out);
        self.close_block(&mut out);
        self.flush_held(&mut out);
        self.close_block(&mut out);
        let (reason, sequence) = stop_reason(
            self.finish_reason.as_deref(),
            self.matched_stop.as_ref(),
            &self.echo,
        );
        write_event(
            &mut out,
            "message_delta",
            &json!({
                "type": "message_delta",
                "delta": {"stop_reason": reason, "stop_sequence": sequence},
                "usage": usage_from_chat(self.usage.as_ref()),
            }),
        );
        write_event(&mut out, "message_stop", &json!({"type": "message_stop"}));
        self.terminal = true;
        out
    }

    fn fail(&mut self, message: &str) -> Vec<u8> {
        let mut out = Vec::new();
        if !self.terminal {
            self.emit_error("api_error", message, &mut out);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn echo() -> EchoContext {
        EchoContext {
            model: "m".into(),
            stop_sequences: vec!["<END>".into()],
        }
    }

    fn events(raw: &[u8]) -> Vec<(String, Value)> {
        std::str::from_utf8(raw)
            .unwrap()
            .split("\n\n")
            .filter(|b| !b.is_empty())
            .map(|b| {
                let mut l = b.lines();
                let ev = l
                    .next()
                    .unwrap()
                    .strip_prefix("event: ")
                    .unwrap()
                    .to_owned();
                let data = serde_json::from_str(l.next().unwrap().strip_prefix("data: ").unwrap())
                    .unwrap();
                (ev, data)
            })
            .collect()
    }

    const USAGE: &str =
        r#"{"prompt_tokens":10,"completion_tokens":1,"prompt_tokens_details":{"cached_tokens":6}}"#;

    fn chunk(delta: Value, finish: Option<&str>, usage: bool) -> String {
        let mut c = json!({"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]});
        if usage {
            c["usage"] = serde_json::from_str(USAGE).unwrap();
        }
        format!("data: {c}\n\n")
    }

    fn run(chunks: &[String]) -> Vec<(String, Value)> {
        let mut s = MessagesStream::new(echo());
        let mut raw = Vec::new();
        for c in chunks {
            let (a, b) = c.as_bytes().split_at(c.len() / 2);
            raw.extend(s.feed(a));
            raw.extend(s.feed(b));
        }
        raw.extend(s.finish());
        let evs = events(&raw);
        for (ev, data) in &evs {
            assert_eq!(data["type"], ev.as_str(), "event name must equal data.type");
        }
        evs
    }

    #[test]
    fn thinking_then_text() {
        let evs = run(&[
            chunk(json!({"role": "assistant", "content": ""}), None, false),
            chunk(json!({"reasoning_content": "hm"}), None, true),
            chunk(json!({"content": "O"}), None, true),
            chunk(json!({"content": "K"}), None, true),
            chunk(json!({}), Some("stop"), true),
            "data: [DONE]\n\n".into(),
        ]);
        let names: Vec<&str> = evs.iter().map(|(e, _)| e.as_str()).collect();
        assert_eq!(
            names,
            [
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "content_block_start",
                "content_block_delta",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop",
            ]
        );
        let start = &evs[0].1["message"];
        assert_eq!(start["usage"]["input_tokens"], 4);
        assert_eq!(start["usage"]["cache_read_input_tokens"], 6);
        assert_eq!(start["usage"]["output_tokens"], 0);
        assert_eq!(evs[1].1["content_block"]["type"], "thinking");
        assert_eq!(
            evs[2].1["delta"],
            json!({"type": "thinking_delta", "thinking": "hm"})
        );
        assert_eq!(evs[4].1["index"], 1);
        let text: String = evs[5..7]
            .iter()
            .map(|(_, d)| d["delta"]["text"].as_str().unwrap())
            .collect();
        assert_eq!(text, "OK");
        assert_eq!(evs[6].1["index"], 1);
        assert_eq!(evs[8].1["delta"]["stop_reason"], "end_turn");
        assert_eq!(evs[8].1["usage"]["output_tokens"], 1);
    }

    #[test]
    fn tool_use_stream_and_stop_sequence() {
        let evs = run(&[
            chunk(
                json!({"tool_calls": [{"index": 0, "id": "call_1",
                        "function": {"name": "get_weather", "arguments": ""}}]}),
                None,
                true,
            ),
            chunk(
                json!({"tool_calls": [{"index": 0, "function": {"arguments": "{\"city\""}}]}),
                None,
                true,
            ),
            chunk(
                json!({"tool_calls": [{"index": 0, "function": {"arguments": ":\"bj\"}"}}]}),
                None,
                true,
            ),
            chunk(json!({}), Some("tool_calls"), true),
        ]);
        assert_eq!(
            evs[1].1["content_block"],
            json!({"type": "tool_use", "id": "call_1", "name": "get_weather", "input": {}})
        );
        let json: String = evs
            .iter()
            .filter(|(e, _)| e == "content_block_delta")
            .map(|(_, d)| d["delta"]["partial_json"].as_str().unwrap())
            .collect();
        assert_eq!(json, "{\"city\":\"bj\"}");
        let md = &evs.iter().find(|(e, _)| e == "message_delta").unwrap().1;
        assert_eq!(md["delta"]["stop_reason"], "tool_use");

        let mut s = MessagesStream::new(echo());
        let mut raw = s.feed(chunk(json!({"content": "a"}), None, true).as_bytes());
        raw.extend(s.feed(
            b"data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\",\"matched_stop\":\"<END>\"}]}\n\n",
        ));
        raw.extend(s.finish());
        let md = events(&raw)
            .into_iter()
            .find(|(e, _)| e == "message_delta")
            .unwrap()
            .1;
        assert_eq!(
            md["delta"],
            json!({"stop_reason": "stop_sequence", "stop_sequence": "<END>"})
        );
    }

    #[test]
    fn inband_error_and_cut_stream() {
        let mut s = MessagesStream::new(echo());
        let mut raw = s.feed(chunk(json!({"content": "a"}), None, true).as_bytes());
        raw.extend(s.feed(b"data: {\"error\": {\"message\": \"boom\"}}\n\n"));
        assert!(s.is_terminal());
        raw.extend(s.finish());
        let (last, data) = events(&raw).pop().unwrap();
        assert_eq!(last, "error");
        assert_eq!(
            data["error"],
            json!({"type": "api_error", "message": "boom"})
        );

        // An upstream status maps to its Anthropic error type.
        let mut s = MessagesStream::new(echo());
        let raw = s.feed(b"data: {\"error\": {\"message\": \"too long\", \"code\": 400}}\n\n");
        let (_, data) = events(&raw).pop().unwrap();
        assert_eq!(data["error"]["type"], "invalid_request_error");

        let evs = run(&[chunk(json!({"content": "cut"}), None, true)]);
        assert_eq!(evs.last().unwrap().0, "error");
    }

    #[test]
    fn usage_only_at_the_end_still_starts_the_message() {
        let evs = run(&[
            chunk(json!({"content": "OK"}), Some("stop"), false),
            format!(
                "data: {}\n\n",
                json!({"choices": [], "usage": serde_json::from_str::<Value>(USAGE).unwrap()})
            ),
        ]);
        assert_eq!(evs[0].0, "message_start");
        let md = &evs.iter().find(|(e, _)| e == "message_delta").unwrap().1;
        assert_eq!(md["usage"]["input_tokens"], 4);
    }

    /// The partial JSON each `tool_use` block received, in block order.
    fn tool_inputs(evs: &[(String, Value)]) -> Vec<(String, String)> {
        let mut blocks: Vec<(u64, String, String)> = Vec::new();
        for (e, d) in evs {
            if e == "content_block_start" && d["content_block"]["type"] == "tool_use" {
                let name = d["content_block"]["name"].as_str().unwrap().to_owned();
                blocks.push((d["index"].as_u64().unwrap(), name, String::new()));
            }
            if e == "content_block_delta" && d["delta"]["type"] == "input_json_delta" {
                let i = d["index"].as_u64().unwrap();
                let b = blocks.iter_mut().find(|b| b.0 == i).unwrap();
                b.2.push_str(d["delta"]["partial_json"].as_str().unwrap());
            }
        }
        blocks.into_iter().map(|(_, n, a)| (n, a)).collect()
    }

    fn call(name: Option<&str>, args: &str) -> Value {
        let mut f = json!({"arguments": args});
        if let Some(n) = name {
            f["name"] = json!(n);
        }
        json!({"tool_calls": [{"index": 0, "function": f}]})
    }

    #[test]
    fn a_repeated_index_with_a_new_name_is_a_new_call() {
        let evs = run(&[
            chunk(call(Some("weather"), r#"{"city":"Paris"}"#), None, true),
            chunk(
                call(Some("weather"), r#"{"city":"Rome"}"#),
                Some("tool_calls"),
                true,
            ),
        ]);
        assert_eq!(
            tool_inputs(&evs),
            [
                ("weather".into(), r#"{"city":"Paris"}"#.into()),
                ("weather".into(), r#"{"city":"Rome"}"#.into())
            ]
        );
    }

    fn text_deltas(evs: &[(String, Value)]) -> Vec<&str> {
        evs.iter()
            .filter(|(e, d)| e == "content_block_delta" && d["delta"]["type"] == "text_delta")
            .map(|(_, d)| d["delta"]["text"].as_str().unwrap())
            .collect()
    }

    #[test]
    fn text_inside_a_call_does_not_split_it() {
        let split = |after: &str| {
            run(&[
                chunk(call(Some("f"), r#"{"x":"#), None, true),
                chunk(json!({"content": "\n"}), None, true),
                chunk(call(None, "1}"), None, true),
                chunk(json!({"content": after}), Some("tool_calls"), true),
            ])
        };
        // Held whitespace alone is dropped.
        let evs = split(" ");
        assert_eq!(tool_inputs(&evs), [("f".into(), r#"{"x":1}"#.into())]);
        assert!(text_deltas(&evs).is_empty());
        // Real text is kept, after the call.
        let evs = split("done");
        assert_eq!(tool_inputs(&evs), [("f".into(), r#"{"x":1}"#.into())]);
        assert_eq!(text_deltas(&evs), ["\ndone"]);
    }

    #[test]
    fn a_string_error_keeps_its_message() {
        let mut s = MessagesStream::new(echo());
        let raw = s.feed(b"data: {\"error\": \"queue full\"}\n\n");
        let (_, data) = events(&raw).pop().unwrap();
        assert_eq!(data["error"]["message"], "queue full");
    }
}
