//! Common SSE parsing and processing utilities for OpenAI responses
//!
//! This module contains shared helpers used by both streaming and accumulator modules.

use std::borrow::Cow;

use serde_json::Value;

// ============================================================================
// Helper Functions
// ============================================================================

/// Extract output_index from a JSON value
#[inline]
pub(super) fn extract_output_index(value: &Value) -> Option<usize> {
    value.get("output_index")?.as_u64().map(|v| v as usize)
}

/// Get event type from event name or parsed JSON, returning a reference to avoid allocation
#[inline]
pub(super) fn get_event_type<'a>(event_name: Option<&'a str>, parsed: &'a Value) -> &'a str {
    event_name
        .or_else(|| parsed.get("type").and_then(|v| v.as_str()))
        .unwrap_or("")
}

// ============================================================================
// Chunk Processor
// ============================================================================

/// Processes incoming byte chunks into complete SSE blocks.
/// Handles buffering of partial chunks and CRLF normalization.
pub(super) struct ChunkProcessor {
    pending: String,
}

impl ChunkProcessor {
    pub fn new() -> Self {
        Self {
            pending: String::new(),
        }
    }

    /// Append a chunk to the buffer, normalizing line endings
    pub fn push_chunk(&mut self, chunk: &[u8]) {
        let chunk_str = match std::str::from_utf8(chunk) {
            Ok(s) => Cow::Borrowed(s),
            Err(_) => Cow::Owned(String::from_utf8_lossy(chunk).into_owned()),
        };
        // Normalize CRLF to LF without extra allocation
        let mut chars = chunk_str.chars().peekable();
        while let Some(c) = chars.next() {
            if c == '\r' && chars.peek() == Some(&'\n') {
                // Skip \r when followed by \n
                continue;
            }
            self.pending.push(c);
        }
    }

    /// Extract the next complete SSE block from the buffer, if available
    pub fn next_block(&mut self) -> Option<String> {
        loop {
            let pos = self.pending.find("\n\n")?;
            let block = self.pending[..pos].to_string();
            self.pending.drain(..pos + 2);

            if !block.trim().is_empty() {
                return Some(block);
            }
            // If block is empty, loop again to find the next one
        }
    }

    /// Check if there's remaining content in the buffer
    pub fn has_remaining(&self) -> bool {
        !self.pending.trim().is_empty()
    }

    /// Take any remaining content from the buffer
    pub fn take_remaining(&mut self) -> String {
        std::mem::take(&mut self.pending)
    }
}

// ============================================================================
// SSE Parsing
// ============================================================================

/// Parse an SSE block into event name and data
///
/// Returns borrowed strings when possible to avoid allocations in hot paths.
/// Only allocates when multiple data lines need to be joined.
pub(super) fn parse_sse_block(block: &str) -> (Option<&str>, Cow<'_, str>) {
    let mut event_name: Option<&str> = None;
    let mut data_lines: Vec<&str> = Vec::new();

    for line in block.lines() {
        if let Some(rest) = line.strip_prefix("event:") {
            event_name = Some(rest.trim());
        } else if let Some(rest) = line.strip_prefix("data:") {
            data_lines.push(rest.trim_start());
        }
    }

    let data = if data_lines.len() == 1 {
        Cow::Borrowed(data_lines[0])
    } else {
        Cow::Owned(data_lines.join("\n"))
    };

    (event_name, data)
}
