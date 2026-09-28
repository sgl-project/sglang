//! Shared OpenAI request handling and generation response processing.
//! Hosts mount the routes and supply a token generation transport. This library
//! does not create a runtime, bind a listener, or select an engine connection.

mod engine;
pub mod http;
mod openai;

pub use engine::{GenerateTransport, GenerationService, TokenDecoder, TokenDelta, TokenStream};
pub use engine::{
    GenerationFinishReason, GenerationOutput, GenerationOutputExtras, GenerationStream,
    MatchedStop, PositionLogprobs, TokenLogprob,
};
pub use openai::OpenAIService;
pub use sglang_renderer::*;
