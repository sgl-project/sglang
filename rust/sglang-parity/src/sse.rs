//! Incremental SSE framing, independent of any API's event payloads.
//!
//! Bytes may split UTF-8 characters or CRLF delimiters anywhere. Only an empty
//! line dispatches an event; incomplete data events at EOF are reported rather
//! than turned into successful responses. Business terminators are plain data.

use serde::{Deserialize, Serialize};

/// One dispatched SSE event. IDs persist until replaced or explicitly reset.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct SseEvent {
    pub event: String,
    pub data: String,
    pub id: Option<String>,
}

/// Stateful decoder for a single HTTP response body.
#[derive(Debug)]
pub struct SseDecoder {
    line: Vec<u8>,
    skip_lf: bool,
    first_line: bool,
    event: String,
    data: String,
    id: Option<String>,
    finished: bool,
    completed: Vec<SseEvent>,
}

impl Default for SseDecoder {
    fn default() -> Self {
        Self {
            line: Vec::new(),
            skip_lf: false,
            first_line: true,
            event: String::new(),
            data: String::new(),
            id: None,
            finished: false,
            completed: Vec::new(),
        }
    }
}

impl SseDecoder {
    /// Consume a network chunk and return all complete events it contains.
    ///
    /// Invalid UTF-8 is an error, not replacement text that could mask a broken
    /// response. Stop using the decoder after any error.
    pub fn push(&mut self, bytes: &[u8]) -> Result<Vec<SseEvent>, String> {
        if self.finished {
            return Err("SSE decoder is already finished".into());
        }
        for &byte in bytes {
            if self.skip_lf {
                self.skip_lf = false;
                if byte == b'\n' {
                    continue;
                }
            }
            match byte {
                b'\r' | b'\n' => {
                    self.process_line()?;
                    self.skip_lf = byte == b'\r';
                }
                _ => self.line.push(byte),
            }
        }
        Ok(self.take_completed())
    }

    /// Recover events completed before a framing error in the same chunk.
    /// Successful `push`/`finish` calls already return and drain these events.
    pub fn take_completed(&mut self) -> Vec<SseEvent> {
        std::mem::take(&mut self.completed)
    }

    /// Check EOF without dispatching an unterminated data event.
    ///
    /// A final comment or ignored field need not end in a newline. Repeated
    /// calls succeed with no events; pushing bytes after EOF is an error.
    pub fn finish(&mut self) -> Result<Vec<SseEvent>, String> {
        if self.finished {
            return Ok(Vec::new());
        }
        self.finished = true;
        if !self.line.is_empty() {
            self.process_line()?;
        }
        if !self.data.is_empty() {
            return Err("SSE response ended before the event's empty-line delimiter".into());
        }
        Ok(self.take_completed())
    }

    fn process_line(&mut self) -> Result<(), String> {
        let line = std::str::from_utf8(&self.line)
            .map_err(|error| format!("invalid UTF-8 in SSE response: {error}"))?;
        let line = if std::mem::replace(&mut self.first_line, false) {
            line.strip_prefix('\u{feff}').unwrap_or(line)
        } else {
            line
        };
        if line.is_empty() {
            if !self.data.is_empty() {
                self.data.pop(); // Each data field appended exactly one newline.
                self.completed.push(SseEvent {
                    event: if self.event.is_empty() {
                        "message".into()
                    } else {
                        std::mem::take(&mut self.event)
                    },
                    data: std::mem::take(&mut self.data),
                    id: self.id.clone(),
                });
            }
            self.event.clear();
        } else {
            let (field, value) = line.split_once(':').unwrap_or((line, ""));
            let value = value.strip_prefix(' ').unwrap_or(value);
            match field {
                "data" => {
                    self.data.push_str(value);
                    self.data.push('\n');
                }
                "event" => self.event = value.into(),
                "id" if !value.contains('\0') => self.id = Some(value.into()),
                // Comments, retry hints and unknown fields do not carry event data.
                // The client never reconnects, so retry hints have no side effects.
                _ => {}
            }
        }
        self.line.clear();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decode_chunks(bytes: &[u8], chunk_size: usize) -> Result<Vec<SseEvent>, String> {
        let mut decoder = SseDecoder::default();
        let mut events = Vec::new();
        for chunk in bytes.chunks(chunk_size) {
            events.extend(decoder.push(chunk)?);
        }
        events.extend(decoder.finish()?);
        Ok(events)
    }

    #[test]
    fn arbitrary_chunk_sizes_preserve_utf8_bom_crlf_and_multiline_data() {
        let bytes = "\u{feff}: keepalive\r\nid: 7\r\nevent: output\r\ndata: 你好\r\ndata: café\r\n\r\ndata: tail\n\n".as_bytes();
        let expected = vec![
            SseEvent {
                event: "output".into(),
                data: "你好\ncafé".into(),
                id: Some("7".into()),
            },
            SseEvent {
                event: "message".into(),
                data: "tail".into(),
                id: Some("7".into()),
            },
        ];
        for chunk_size in 1..=bytes.len() {
            assert_eq!(
                decode_chunks(bytes, chunk_size).unwrap(),
                expected,
                "chunk size {chunk_size}"
            );
        }
    }

    #[test]
    fn all_line_endings_dispatch_empty_data_without_extra_events() {
        for ending in ["\n", "\r", "\r\n"] {
            let text = format!(
                "event: ignored{ending}{ending}data{ending}{ending}{ending}data: x{ending}{ending}"
            );
            let events = decode_chunks(text.as_bytes(), 1).unwrap();
            assert_eq!(events.len(), 2);
            assert_eq!(events[0].data, "");
            assert_eq!(events[0].event, "message");
            assert_eq!(events[1].data, "x");
        }
    }

    #[test]
    fn ids_persist_reset_and_ignore_nul_fields() {
        let events = decode_chunks(
            b"id: old\n\nid: ignored\0id\ndata: first\n\nid:\ndata: second\n\ndata: third\n\n",
            2,
        )
        .unwrap();
        assert_eq!(events[0].id.as_deref(), Some("old"));
        assert_eq!(events[1].id.as_deref(), Some(""));
        assert_eq!(events[2].id.as_deref(), Some(""));
    }

    #[test]
    fn framing_does_not_interpret_payloads_or_trim_data() {
        let events = decode_chunks(
            b": comment\nretry: 100\nunknown: ignored\ndata: [DONE]\n\ndata:  after:done \ndata:\n\n",
            3,
        )
        .unwrap();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].data, "[DONE]");
        assert_eq!(events[1].data, " after:done \n");
    }

    #[test]
    fn incomplete_event_and_invalid_utf8_are_errors() {
        for bytes in [b"data: tail".as_slice(), b"data: tail\n", b"data:\r"] {
            assert!(decode_chunks(bytes, 1).unwrap_err().contains("delimiter"));
        }
        for bytes in [b"data: \xff\n\n".as_slice(), b"data: \xe4\xbd"] {
            assert!(decode_chunks(bytes, 1).unwrap_err().contains("UTF-8"));
        }
    }

    #[test]
    fn valid_prefix_is_recoverable_regardless_of_the_failing_chunk_boundary() {
        let bytes = b"data: valid\n\ndata: \xff\n\n";
        for size in 1..=bytes.len() {
            let mut decoder = SseDecoder::default();
            let mut events = Vec::new();
            let mut failed = false;
            for chunk in bytes.chunks(size) {
                match decoder.push(chunk) {
                    Ok(complete) => events.extend(complete),
                    Err(_) => {
                        events.extend(decoder.take_completed());
                        failed = true;
                        break;
                    }
                }
            }
            assert!(failed);
            assert_eq!(events.len(), 1);
            assert_eq!(events[0].data, "valid");
        }
    }

    #[test]
    fn only_initial_bom_is_removed_and_eof_does_not_fabricate_events() {
        let events = decode_chunks(
            "\u{feff}data: ok\n\n\u{feff}data: ignored\n\n: trailing comment".as_bytes(),
            1,
        )
        .unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "ok");
        let mut decoder = SseDecoder::default();
        assert!(decoder.finish().unwrap().is_empty());
        assert!(decoder.finish().unwrap().is_empty());
        assert!(decoder.push(b"data: late\n\n").is_err());
    }
}
