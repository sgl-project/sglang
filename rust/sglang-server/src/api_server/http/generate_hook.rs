//! The seam a model package uses to add its own keys to native `/generate`
//! frames — deployment metadata the shared schema does not model. Without a
//! hook the frames are exactly the `api.v1` schema.

use std::time::Duration;

use serde_json::{Map, Value};

use crate::message::sampling::SamplingParams;

/// Adds keys to native `/generate` frames; installed at launch with
/// [`crate::Server::start_with_extensions`].
pub trait GenerateHook: Send + Sync + std::fmt::Debug {
    /// The annotator for one submitted request, from its sampling parameters
    /// as submitted (the launch-time preferred params merged in, not yet
    /// normalized).
    fn start(&self, sampling_params: &SamplingParams) -> Box<dyn GenerateFrameHook>;

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

/// Keys an annotator adds to one frame. They must not repeat a key the schema
/// already writes there; a `customized_info` key they repeat gives way.
#[derive(Debug, Default)]
pub struct GenerateFrameKeys {
    /// Added to the frame's `meta_info`.
    pub meta_info: Map<String, Value>,
    /// Added to the frame object, after the schema's keys.
    pub frame: Map<String, Value>,
}
