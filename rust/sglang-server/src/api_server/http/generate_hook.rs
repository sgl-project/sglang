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
}

/// One request's annotator, called for each frame the request emits, in
/// order, the terminal one last.
pub trait GenerateFrameHook: Send {
    fn annotate(&mut self, frame: &GenerateFrameInfo, keys: &mut GenerateFrameKeys);
}

/// What an annotator sees of one frame.
#[derive(Clone, Copy, Debug, Default)]
pub struct GenerateFrameInfo {
    /// The request's terminal frame.
    pub finished: bool,
    /// On the terminal frame, the time from the request's first token to its
    /// last output, as Python's TokenizerManager measures it
    /// (`get_decode_latency`): for a non-streaming request, from the
    /// scheduler's prefill stamp when it has one.
    pub decode_latency: Option<Duration>,
}

/// Keys an annotator adds to one frame. They must not repeat a key the schema
/// already writes there.
#[derive(Debug, Default)]
pub struct GenerateFrameKeys {
    /// Added to the frame's `meta_info`.
    pub meta_info: Map<String, Value>,
    /// Added to the frame object, after the schema's keys.
    pub frame: Map<String, Value>,
}
