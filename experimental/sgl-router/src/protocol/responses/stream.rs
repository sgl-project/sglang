// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Chat SSE → Responses events: created, in_progress, one output item at a
//! time (added … done), then completed | incomplete | failed.

use serde_json::{json, Value};

use super::{
    call_item, custom_input, failed_error, message_item, new_id, now_secs, output_text_part,
    reasoning_item, response_object, usage_from_chat, EchoContext, Finish,
};
use crate::protocol::sse::{data_payload, write_event, LineBuffer, SseTransducer};

enum Open {
    Reasoning {
        id: String,
        index: usize,
        text: String,
    },
    Message {
        id: String,
        index: usize,
        text: String,
    },
    Function {
        id: String,
        index: usize,
        call_id: String,
        name: String,
        args: String,
    },
}

/// Reasoning or message text held while a function call is open.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Held {
    Reasoning,
    Text,
}

pub struct ResponsesStream {
    echo: EchoContext,
    id: String,
    created_at: u64,
    seq: u64,
    lines: LineBuffer,
    started: bool,
    terminal: bool,
    open: Option<Open>,
    /// Emitted after the open call so it cannot split it.
    held: Vec<(Held, String)>,
    output: Vec<Value>,
    finish_reason: Option<String>,
    usage: Option<Value>,
}

impl ResponsesStream {
    pub fn new(echo: EchoContext) -> Self {
        Self {
            echo,
            id: new_id("resp"),
            created_at: now_secs(),
            seq: 0,
            lines: LineBuffer::default(),
            started: false,
            terminal: false,
            open: None,
            held: Vec::new(),
            output: Vec::new(),
            finish_reason: None,
            usage: None,
        }
    }

    fn handle_line(&mut self, line: &[u8], out: &mut Vec<u8>) {
        if self.terminal {
            return;
        }
        let Some(chunk) = data_payload(line) else {
            return;
        };
        self.ensure_started(out);
        if let Some(err) = chunk.get("error").filter(|e| !e.is_null()) {
            let message = err
                .get("message")
                .and_then(Value::as_str)
                .or_else(|| err.as_str())
                .unwrap_or("upstream error");
            self.emit_failed(message, out);
            return;
        }
        if let Some(u) = chunk.get("usage").filter(|u| u.is_object()) {
            self.usage = Some(u.clone());
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
                self.reasoning_delta(r, out);
            }
            if let Some(c) = text("content") {
                self.text_delta(c, out);
            }
            if let Some(calls) = delta.get("tool_calls").and_then(Value::as_array) {
                for call in calls {
                    self.tool_delta(call, out);
                }
            }
        }
        if let Some(fr) = choice.get("finish_reason").and_then(Value::as_str) {
            self.finish_reason = Some(fr.to_owned());
        }
    }

    fn ensure_started(&mut self, out: &mut Vec<u8>) {
        if self.started {
            return;
        }
        self.started = true;
        let resp = self.response("in_progress");
        self.emit("response.created", json!({ "response": resp }), out);
        let resp = self.response("in_progress");
        self.emit("response.in_progress", json!({ "response": resp }), out);
    }

    /// `true` when `delta` was held because a function call is open.
    fn hold(&mut self, kind: Held, delta: &str) -> bool {
        if !matches!(self.open, Some(Open::Function { .. })) {
            return false;
        }
        match self.held.last_mut() {
            Some((k, text)) if *k == kind => text.push_str(delta),
            _ => self.held.push((kind, delta.to_owned())),
        }
        true
    }

    /// Emits held text after a call; whitespace alone is dropped.
    fn flush_held(&mut self, out: &mut Vec<u8>) {
        for (kind, text) in std::mem::take(&mut self.held) {
            match kind {
                _ if text.trim().is_empty() => {}
                Held::Reasoning => self.reasoning_delta(&text, out),
                Held::Text => self.text_delta(&text, out),
            }
        }
    }

    fn reasoning_delta(&mut self, delta: &str, out: &mut Vec<u8>) {
        if self.hold(Held::Reasoning, delta) {
            return;
        }
        if !matches!(self.open, Some(Open::Reasoning { .. })) {
            self.close_open(Finish::Completed, out);
            let id = new_id("rs");
            let index = self.output.len();
            let mut item = reasoning_item(&id, "", "in_progress");
            item["content"] = json!([]);
            self.emit(
                "response.output_item.added",
                json!({"output_index": index, "item": item}),
                out,
            );
            self.emit(
                "response.content_part.added",
                json!({"item_id": id, "output_index": index, "content_index": 0,
                       "part": {"type": "reasoning_text", "text": ""}}),
                out,
            );
            self.open = Some(Open::Reasoning {
                id,
                index,
                text: String::new(),
            });
        }
        let Some(Open::Reasoning { id, index, text }) = &mut self.open else {
            unreachable!()
        };
        text.push_str(delta);
        let data = json!({"item_id": id, "output_index": *index, "content_index": 0,
                          "delta": delta});
        self.emit("response.reasoning_text.delta", data, out);
    }

    fn text_delta(&mut self, delta: &str, out: &mut Vec<u8>) {
        if self.hold(Held::Text, delta) {
            return;
        }
        if !matches!(self.open, Some(Open::Message { .. })) {
            self.close_open(Finish::Completed, out);
            let id = new_id("msg");
            let index = self.output.len();
            let mut item = message_item(&id, "", "in_progress");
            item["content"] = json!([]);
            self.emit(
                "response.output_item.added",
                json!({"output_index": index, "item": item}),
                out,
            );
            self.emit(
                "response.content_part.added",
                json!({"item_id": id, "output_index": index, "content_index": 0,
                       "part": output_text_part("")}),
                out,
            );
            self.open = Some(Open::Message {
                id,
                index,
                text: String::new(),
            });
        }
        let Some(Open::Message { id, index, text }) = &mut self.open else {
            unreachable!()
        };
        text.push_str(delta);
        let data = json!({"item_id": id, "output_index": *index, "content_index": 0,
                          "delta": delta, "logprobs": []});
        self.emit("response.output_text.delta", data, out);
    }

    /// SGLang names a call only on its first chunk, so a name starts a new
    /// item: the index alone can repeat across calls.
    fn tool_delta(&mut self, call: &Value, out: &mut Vec<u8>) {
        let named = call
            .pointer("/function/name")
            .and_then(Value::as_str)
            .is_some_and(|n| !n.is_empty());
        if named || !matches!(self.open, Some(Open::Function { .. })) {
            self.close_open(Finish::Completed, out);
            self.flush_held(out);
            self.close_open(Finish::Completed, out);
            let id = new_id("fc");
            let index = self.output.len();
            let call_id = call
                .get("id")
                .and_then(Value::as_str)
                .map(str::to_owned)
                .unwrap_or_else(|| new_id("call"));
            let name = call
                .pointer("/function/name")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_owned();
            self.emit(
                "response.output_item.added",
                json!({"output_index": index,
                       "item": call_item(&id, &call_id, &name, "", "in_progress", &self.echo.tools)}),
                out,
            );
            self.open = Some(Open::Function {
                id,
                index,
                call_id,
                name,
                args: String::new(),
            });
        }
        let Some(Open::Function {
            id,
            index,
            name,
            args,
            ..
        }) = &mut self.open
        else {
            unreachable!()
        };
        if name.is_empty() {
            if let Some(n) = call.pointer("/function/name").and_then(Value::as_str) {
                *name = n.to_owned();
            }
        }
        if let Some(delta) = call
            .pointer("/function/arguments")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
        {
            args.push_str(delta);
            // A custom tool's input is JSON-wrapped; it is sent whole on close.
            if !self.echo.tools.get(name.as_str()).is_some_and(|t| t.custom) {
                let data = json!({"item_id": id, "output_index": *index, "delta": delta});
                self.emit("response.function_call_arguments.delta", data, out);
            }
        }
    }

    fn close_open(&mut self, finish: Finish, out: &mut Vec<u8>) {
        let Some(open) = self.open.take() else {
            return;
        };
        let item = match open {
            Open::Reasoning { id, index, text } => {
                self.emit(
                    "response.reasoning_text.done",
                    json!({"item_id": id, "output_index": index, "content_index": 0,
                           "text": text}),
                    out,
                );
                self.emit(
                    "response.content_part.done",
                    json!({"item_id": id, "output_index": index, "content_index": 0,
                           "part": {"type": "reasoning_text", "text": text}}),
                    out,
                );
                (index, reasoning_item(&id, &text, "completed"))
            }
            Open::Message { id, index, text } => {
                self.emit(
                    "response.output_text.done",
                    json!({"item_id": id, "output_index": index, "content_index": 0,
                           "text": text, "logprobs": []}),
                    out,
                );
                self.emit(
                    "response.content_part.done",
                    json!({"item_id": id, "output_index": index, "content_index": 0,
                           "part": output_text_part(&text)}),
                    out,
                );
                (index, message_item(&id, &text, finish.item_status()))
            }
            Open::Function {
                id,
                index,
                call_id,
                name,
                args,
                ..
            } => {
                let item = call_item(&id, &call_id, &name, &args, "completed", &self.echo.tools);
                if item["type"] == "custom_tool_call" {
                    self.emit(
                        "response.custom_tool_call_input.done",
                        json!({"item_id": id, "output_index": index,
                               "input": custom_input(&args)}),
                        out,
                    );
                } else {
                    self.emit(
                        "response.function_call_arguments.done",
                        json!({"item_id": id, "output_index": index, "name": item["name"],
                               "arguments": args}),
                        out,
                    );
                }
                (index, item)
            }
        };
        let (index, item) = item;
        self.emit(
            "response.output_item.done",
            json!({"output_index": index, "item": item}),
            out,
        );
        self.output.push(item);
    }

    fn emit_failed(&mut self, message: &str, out: &mut Vec<u8>) {
        self.ensure_started(out);
        let mut resp = self.response("failed");
        resp["error"] = failed_error(message);
        self.emit("response.failed", json!({ "response": resp }), out);
        self.terminal = true;
    }

    fn response(&self, status: &str) -> Value {
        let usage = match status {
            "in_progress" => Value::Null,
            _ => usage_from_chat(self.usage.as_ref()),
        };
        response_object(
            &self.echo,
            &self.id,
            self.created_at,
            status,
            self.output.clone(),
            usage,
        )
    }

    fn emit(&mut self, event: &str, mut data: Value, out: &mut Vec<u8>) {
        if let Value::Object(m) = &mut data {
            let mut ordered = serde_json::Map::new();
            ordered.insert("type".into(), event.into());
            ordered.insert("sequence_number".into(), self.seq.into());
            ordered.append(m);
            data = Value::Object(ordered);
        }
        self.seq += 1;
        write_event(out, event, &data);
    }
}

impl SseTransducer for ResponsesStream {
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

    /// No `finish_reason` means generation did not finish: report failed.
    fn finish(&mut self) -> Vec<u8> {
        let mut out = Vec::new();
        let mut lines = std::mem::take(&mut self.lines);
        lines.flush(|line| self.handle_line(line, &mut out));
        if self.terminal {
            return out;
        }
        if self.finish_reason.is_none() {
            self.emit_failed(
                "upstream stream ended before the response completed",
                &mut out,
            );
            return out;
        }
        let finish = Finish::from_chat(self.finish_reason.as_deref());
        if !self.held.is_empty() {
            self.close_open(Finish::Completed, &mut out);
            self.flush_held(&mut out);
        }
        self.close_open(finish, &mut out);
        let mut resp = self.response(finish.status());
        finish.annotate(&mut resp);
        self.emit(finish.event(), json!({ "response": resp }), &mut out);
        self.terminal = true;
        out
    }

    fn fail(&mut self, message: &str) -> Vec<u8> {
        let mut out = Vec::new();
        if !self.terminal {
            self.emit_failed(message, &mut out);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::responses::to_chat;

    fn echo() -> EchoContext {
        to_chat(json!({"model": "m", "input": "x", "stream": true}))
            .unwrap()
            .echo
    }

    pub(crate) fn events(raw: &[u8]) -> Vec<(String, Value)> {
        let text = std::str::from_utf8(raw).unwrap();
        text.split("\n\n")
            .filter(|b| !b.is_empty())
            .map(|block| {
                let mut lines = block.lines();
                let ev = lines.next().unwrap().strip_prefix("event: ").unwrap();
                let data = lines.next().unwrap().strip_prefix("data: ").unwrap();
                (ev.to_owned(), serde_json::from_str(data).unwrap())
            })
            .collect()
    }

    fn chunk(delta: Value, finish: Option<&str>) -> String {
        format!(
            "data: {}\n\n",
            json!({"id": "c", "object": "chat.completion.chunk",
                   "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})
        )
    }

    fn run(chunks: &[String]) -> Vec<(String, Value)> {
        let mut s = ResponsesStream::new(echo());
        let mut raw = Vec::new();
        for c in chunks {
            let (a, b) = c.as_bytes().split_at(c.len() / 2);
            raw.extend(s.feed(a));
            raw.extend(s.feed(b));
        }
        raw.extend(s.finish());
        events(&raw)
    }

    fn assert_framing(evs: &[(String, Value)]) {
        for (i, (ev, data)) in evs.iter().enumerate() {
            assert_eq!(data["type"], ev.as_str(), "event name must equal data.type");
            assert_eq!(
                data["sequence_number"], i as u64,
                "sequence_number must be gapless"
            );
        }
    }

    #[test]
    fn reasoning_then_text_sequence() {
        let evs = run(&[
            chunk(json!({"role": "assistant", "content": ""}), None),
            chunk(json!({"reasoning_content": "th"}), None),
            chunk(json!({"reasoning_content": "ink"}), None),
            chunk(json!({"content": "O"}), None),
            chunk(json!({"content": "K"}), Some("stop")),
            format!(
                "data: {}\n\n",
                json!({"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 4,
                       "prompt_tokens_details": {"cached_tokens": 2}}})
            ),
            "data: [DONE]\n\n".into(),
        ]);
        assert_framing(&evs);
        let names: Vec<&str> = evs.iter().map(|(e, _)| e.as_str()).collect();
        assert_eq!(
            names,
            [
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.content_part.added",
                "response.reasoning_text.delta",
                "response.reasoning_text.delta",
                "response.reasoning_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.output_item.added",
                "response.content_part.added",
                "response.output_text.delta",
                "response.output_text.delta",
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.completed",
            ]
        );
        assert_eq!(evs[0].1["response"]["status"], "in_progress");
        assert_eq!(evs[6].1["text"], "think");
        assert_eq!(evs[9].1["output_index"], 1);
        assert_eq!(evs[13].1["text"], "OK");
        let done = &evs[16].1["response"];
        assert_eq!(done["status"], "completed");
        assert_eq!(done["output"][1]["content"][0]["text"], "OK");
        assert_eq!(done["output"][0]["status"], "completed");
        assert_eq!(done["usage"]["input_tokens_details"]["cached_tokens"], 2);
        assert_eq!(done["usage"]["total_tokens"], 7);
    }

    #[test]
    fn length_ends_with_response_incomplete() {
        let evs = run(&[chunk(json!({"reasoning_content": "abc"}), Some("length"))]);
        assert_framing(&evs);
        let (last, data) = evs.last().unwrap();
        assert_eq!(last, "response.incomplete");
        assert_eq!(data["response"]["status"], "incomplete");
        assert_eq!(
            data["response"]["incomplete_details"],
            json!({"reason": "max_output_tokens"})
        );
        assert_eq!(data["response"]["output"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn streamed_tool_calls() {
        let evs = run(&[
            chunk(
                json!({"tool_calls": [{"index": 0, "id": "call_a", "type": "function",
                                       "function": {"name": "get_weather", "arguments": ""}}]}),
                None,
            ),
            chunk(
                json!({"tool_calls": [{"index": 0, "function": {"arguments": "{\"city\""}}]}),
                None,
            ),
            chunk(
                json!({"tool_calls": [{"index": 0, "function": {"arguments": ":\"bj\"}"}}]}),
                None,
            ),
            chunk(
                json!({"tool_calls": [{"index": 1, "id": "call_b",
                                       "function": {"name": "get_time", "arguments": "{}"}}]}),
                Some("tool_calls"),
            ),
        ]);
        assert_framing(&evs);
        let dones: Vec<&Value> = evs
            .iter()
            .filter(|(e, _)| e == "response.function_call_arguments.done")
            .map(|(_, d)| d)
            .collect();
        assert_eq!(dones.len(), 2);
        assert_eq!(dones[0]["arguments"], "{\"city\":\"bj\"}");
        assert_eq!(dones[0]["name"], "get_weather");
        assert_eq!(dones[1]["arguments"], "{}");
        let done = &evs.last().unwrap().1["response"];
        assert_eq!(done["status"], "completed");
        assert_eq!(done["output"][0]["call_id"], "call_a");
        assert_eq!(done["output"][1]["call_id"], "call_b");
    }

    #[test]
    fn inband_error_becomes_response_failed_and_stops() {
        let mut s = ResponsesStream::new(echo());
        let mut raw = s.feed(chunk(json!({"content": "par"}), None).as_bytes());
        raw.extend(s.feed(b"data: {\"error\": {\"message\": \"boom\", \"code\": 500}}\n\n"));
        assert!(s.is_terminal());
        raw.extend(s.feed(chunk(json!({"content": "more"}), None).as_bytes()));
        raw.extend(s.finish());
        let evs = events(&raw);
        assert_framing(&evs);
        let (last, data) = evs.last().unwrap();
        assert_eq!(last, "response.failed");
        assert_eq!(data["response"]["status"], "failed");
        assert_eq!(data["response"]["error"]["message"], "boom");
    }

    #[test]
    fn stream_without_finish_reason_fails() {
        let evs = run(&[chunk(json!({"content": "cut"}), None)]);
        assert_eq!(evs.last().unwrap().0, "response.failed");
    }

    #[test]
    fn streamed_custom_tool_call() {
        let echo = to_chat(
            json!({"model": "m", "input": "x", "stream": true, "tools": [
            {"type": "custom", "name": "apply_patch"}]}),
        )
        .unwrap()
        .echo;
        let mut s = ResponsesStream::new(echo);
        let mut raw = Vec::new();
        for c in [
            chunk(
                json!({"tool_calls": [{"index": 0, "id": "c1", "type": "function",
                         "function": {"name": "apply_patch", "arguments": "{\"input\":"}}]}),
                None,
            ),
            chunk(
                json!({"tool_calls": [{"index": 0, "function": {"arguments": "\"*** P\"}"}}]}),
                Some("tool_calls"),
            ),
        ] {
            raw.extend(s.feed(c.as_bytes()));
        }
        raw.extend(s.finish());
        let evs = events(&raw);
        assert_framing(&evs);
        assert!(!evs
            .iter()
            .any(|(e, _)| e == "response.function_call_arguments.delta"));
        let done = evs
            .iter()
            .find(|(e, _)| e == "response.custom_tool_call_input.done")
            .unwrap();
        assert_eq!(done.1["input"], "*** P");
        let item = &evs.last().unwrap().1["response"]["output"][0];
        assert_eq!(item["type"], "custom_tool_call");
        assert_eq!(item["input"], "*** P");
    }

    #[test]
    fn engine_abort_ends_with_response_failed() {
        let evs = run(&[
            chunk(json!({"content": "part"}), None),
            chunk(json!({}), Some("abort")),
        ]);
        assert_framing(&evs);
        let (event, data) = evs.last().unwrap();
        assert_eq!(event, "response.failed");
        assert_eq!(data["response"]["status"], "failed");
        assert_eq!(data["response"]["output"][0]["status"], "incomplete");
    }

    fn call(name: Option<&str>, args: &str) -> Value {
        let mut f = json!({"arguments": args});
        if let Some(n) = name {
            f["name"] = json!(n);
        }
        json!({"tool_calls": [{"index": 0, "function": f}]})
    }

    /// `(type, name or text, arguments)` of each finished output item.
    fn items(evs: &[(String, Value)]) -> Vec<(String, String, String)> {
        let done = &evs.last().unwrap().1["response"]["output"];
        done.as_array()
            .unwrap()
            .iter()
            .map(|i| {
                let t = i["type"].as_str().unwrap().to_owned();
                let label = i["name"]
                    .as_str()
                    .or_else(|| i.pointer("/content/0/text").and_then(Value::as_str))
                    .unwrap_or("")
                    .to_owned();
                (t, label, i["arguments"].as_str().unwrap_or("").to_owned())
            })
            .collect()
    }

    #[test]
    fn a_repeated_index_with_a_new_name_is_a_new_call() {
        let evs = run(&[
            chunk(call(Some("weather"), r#"{"city":"Paris"}"#), None),
            chunk(
                call(Some("weather"), r#"{"city":"Rome"}"#),
                Some("tool_calls"),
            ),
        ]);
        assert_framing(&evs);
        let fc = |a: &str| ("function_call".into(), "weather".into(), a.into());
        assert_eq!(
            items(&evs),
            [fc(r#"{"city":"Paris"}"#), fc(r#"{"city":"Rome"}"#)]
        );
    }

    #[test]
    fn text_inside_a_call_does_not_split_it() {
        let evs = run(&[
            chunk(call(Some("f"), r#"{"x":"#), None),
            chunk(json!({"content": "\n"}), None),
            chunk(call(None, "1}"), None),
            chunk(json!({"content": "done"}), Some("tool_calls")),
        ]);
        assert_framing(&evs);
        assert_eq!(
            items(&evs),
            [
                ("function_call".into(), "f".into(), r#"{"x":1}"#.into()),
                ("message".into(), "\ndone".into(), String::new())
            ]
        );
    }

    #[test]
    fn a_string_error_keeps_its_message() {
        let mut s = ResponsesStream::new(echo());
        let raw = s.feed(b"data: {\"error\": \"queue full\"}\n\n");
        let (event, data) = events(&raw).pop().unwrap();
        assert_eq!(event, "response.failed");
        assert_eq!(data["response"]["error"]["message"], "queue full");
    }
}
