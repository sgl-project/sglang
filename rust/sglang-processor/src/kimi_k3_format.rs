//! Kimi-K3's XTML markers and the helpers SGLang's K3 reasoning parser, tool
//! parser and structural tag share (`function_call/kimik3_format.py`).

pub(crate) const THINK_OPEN: &str = "<|open|>think<|sep|>";
pub(crate) const THINK_CLOSE: &str = "<|close|>think<|sep|>";
pub(crate) const RESPONSE_OPEN: &str = "<|open|>response<|sep|>";
pub(crate) const RESPONSE_CLOSE: &str = "<|close|>response<|sep|>";
pub(crate) const TOOLS_OPEN: &str = "<|open|>tools<|sep|>";
pub(crate) const TOOLS_CLOSE: &str = "<|close|>tools<|sep|>";
pub(crate) const MESSAGE_CLOSE: &str = "<|close|>message<|sep|>";
pub(crate) const CALL_OPEN: &str = "<|open|>call";
pub(crate) const CALL_CLOSE: &str = "<|close|>call<|sep|>";

/// `_PARTIAL_MARKER_SUFFIXES`: where `max_tokens` can cut a marker before `<|sep|>`.
const PARTIAL_MARKER_SUFFIXES: [&str; 8] = [
    "<|open|>",
    "<|close|>",
    "<|close|>think",
    "<|open|>response",
    "<|close|>response",
    "<|open|>tools",
    "<|close|>tools",
    "<|close|>message",
];

/// `partial_suffix_len`: the longest proper marker prefix `text` ends with.
/// Markers are ASCII, so the byte length is a char boundary.
pub(crate) fn partial_suffix_len(text: &str, markers: &[&str]) -> usize {
    let mut best = 0;
    for marker in markers {
        let longest = (marker.len() - 1).min(text.len());
        if let Some(len) = (best + 1..=longest)
            .rev()
            .find(|&len| text.ends_with(&marker[..len]))
        {
            best = len;
        }
    }
    best
}

/// `strip_partial_marker_suffix`: drop the first matching suffix.
pub(crate) fn strip_partial_marker_suffix(text: &str) -> &str {
    PARTIAL_MARKER_SUFFIXES
        .iter()
        .find_map(|suffix| text.strip_suffix(suffix))
        .unwrap_or(text)
}

/// `strip_response_wrappers`: the first response channel's text if there is
/// one, else the text without response closes; message closes dropped.
pub(crate) fn strip_response_wrappers(text: &str) -> String {
    let text = match text.find(RESPONSE_OPEN) {
        Some(open) => {
            let inner = &text[open + RESPONSE_OPEN.len()..];
            inner
                .find(RESPONSE_CLOSE)
                .map_or(inner, |c| &inner[..c])
                .to_owned()
        }
        None => text.replace(RESPONSE_CLOSE, ""),
    };
    strip_partial_marker_suffix(&text.replace(MESSAGE_CLOSE, "")).to_owned()
}
