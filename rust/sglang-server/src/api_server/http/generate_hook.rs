//! The seam a model package uses to add its own keys to native `/generate`
//! frames — deployment metadata the shared schema does not model. Without a
//! hook the frames are exactly the `api.v1` schema.

use std::time::Duration;

use axum::http::{HeaderMap, StatusCode};
use serde_json::{Map, Value};
pub use sglang_api_types::api::v1::GenerateMetaInfo;

use crate::message::request::GenerateRequest;

/// Adds keys to native `/generate` frames; installed at launch with
/// [`crate::Server::start_with_extensions`].
pub trait GenerateHook: Send + Sync + std::fmt::Debug {
    /// The annotator for one submitted request (one item of a batch):
    /// `request` as submitted (the launch-time preferred sampling params
    /// merged in, not yet normalized), the HTTP request's `headers`, and the
    /// `sampling_params` object the client sent for it, before the preferred
    /// params were merged, if it sent one.
    fn start(
        &self,
        request: &GenerateRequest,
        headers: &HeaderMap,
        client_sampling_params: Option<&Map<String, Value>>,
    ) -> Box<dyn GenerateFrameHook>;

    /// Whether this hook consumes a scheduler `customized_info` key: annotators
    /// still see its values ([`GenerateFrameInfo::new_customized_info`]), but
    /// no frame reports it, cumulative re-reports included.
    fn consumes_customized_info_key(&self, _key: &str) -> bool {
        false
    }
}

/// One request's annotator, called for each frame the request emits, in
/// order, the terminal one last.
pub trait GenerateFrameHook: Send {
    fn annotate(&mut self, frame: &GenerateFrameInfo<'_>, keys: &mut GenerateFrameKeys);

    /// The request's terminal frame, after [`Self::annotate`] added its keys.
    fn finished(&mut self, _terminal: &GenerateTerminal<'_>) {}

    /// The request failed after it was submitted, with this HTTP status and
    /// message; it has no terminal frame.
    fn failed(&mut self, _status: StatusCode, _message: &str) {}
}

/// What an annotator sees of one frame.
#[derive(Clone, Copy, Debug, Default)]
pub struct GenerateFrameInfo<'a> {
    /// The request's terminal frame.
    pub finished: bool,
    /// On the terminal frame, the time from the request's first token to its
    /// last output, as Python's TokenizerManager measures it
    /// (`get_decode_latency`): for a non-streaming request, from the
    /// scheduler's prefill stamp when it has one.
    pub decode_latency: Option<Duration>,
    /// The scheduler `customized_info` keys this frame reports, consumed ones
    /// included, each with the values that arrived since the request's
    /// previous frame that showed it. A cumulative stream's frame shows every
    /// key the request has had so far (with no new values when its latest
    /// output lacked one); an incremental stream's frame and a unary response
    /// show the keys their latest output carried. A stream shows each value on
    /// exactly one frame, the same ones in both modes; a unary response shows
    /// all the values of the keys its terminal output carried.
    pub customized_info: &'a [(&'a str, &'a [Value])],
}

impl<'a> GenerateFrameInfo<'a> {
    /// The new values of one `customized_info` key, if this frame reports it.
    pub fn new_customized_info(&self, key: &str) -> Option<&'a [Value]> {
        self.customized_info
            .iter()
            .find_map(|&(known, values)| (known == key).then_some(values))
    }
}

/// A request's terminal frame as the client receives it, apart from its
/// `customized_info` keys (annotators see them in
/// [`GenerateFrameInfo::customized_info`]) and the hook's own keys.
#[derive(Clone, Copy, Debug)]
pub struct GenerateTerminal<'a> {
    /// The schema's `meta_info`.
    pub meta_info: &'a GenerateMetaInfo,
    /// The scheduler's stats for the request, reported in `meta_info` after
    /// the schema's keys.
    pub request_stats: Option<&'a Map<String, Value>>,
    /// As [`GenerateFrameInfo::decode_latency`].
    pub decode_latency: Option<Duration>,
    /// The output token logprobs the request returned over all its frames
    /// (an incremental stream's terminal frame carries only its last step's).
    pub output_token_logprobs: usize,
}

/// Keys an annotator adds to one frame. They must not repeat a key the schema
/// already writes there; a `customized_info` key they repeat gives way.
#[derive(Debug, Default)]
pub struct GenerateFrameKeys {
    /// Added to the frame's `meta_info`.
    pub meta_info: Map<String, Value>,
    /// Added to the frame object, after the schema's keys.
    pub frame: Map<String, Value>,
}
