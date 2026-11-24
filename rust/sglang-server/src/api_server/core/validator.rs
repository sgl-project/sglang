//! The seam a model package uses to refuse generations its deployment cannot
//! serve, whichever ingress they come from.

use crate::message::request::GenerateRequest;

/// Refuses a generation at the shared core's admission; installed at launch
/// with [`crate::Server::start_with_extensions`]. Every client ingress (native
/// `/generate` over HTTP and gRPC, the OpenAI routes) runs it before media
/// prefetch: a refusal is the request's `InvalidArgument` (HTTP 400, gRPC
/// INVALID_ARGUMENT), and a batch with any refused item submits none.
pub trait GenerateValidator: Send + Sync + std::fmt::Debug {
    fn validate(&self, request: &GenerateRequest) -> Result<(), String>;
}

/// Refuses requests asking for more than one top logprob.
#[cfg(test)]
#[derive(Debug)]
pub(crate) struct TopLogprobsCap;

#[cfg(test)]
impl GenerateValidator for TopLogprobsCap {
    fn validate(&self, request: &GenerateRequest) -> Result<(), String> {
        if request.top_logprobs_num > 1 {
            return Err(format!("top_logprobs_num {}", request.top_logprobs_num));
        }
        Ok(())
    }
}
