//! Token-only generation transport and decoded engine output.

use futures::{StreamExt, TryStreamExt, future::BoxFuture};

use crate::{GenerateRequest, ResponseError};

mod decode;
#[cfg(feature = "http")]
mod http;
pub(crate) mod response;
mod types;

pub(crate) use decode::TokenDecoder;
#[cfg(feature = "http")]
pub(crate) use http::HttpGenerateClient;
pub(crate) use types::{
    GenerationFinishReason, GenerationOutput, GenerationOutputExtras, GenerationStream,
    MatchedStop, PositionLogprobs, TokenDelta, TokenLogprob,
};

pub(crate) type TokenStream =
    futures::stream::BoxStream<'static, Result<TokenDelta, ResponseError>>;

/// Backend generation from prepared token requests to normalized token deltas.
///
/// Successful streams carry a finish reason on their terminal output. The caller
/// owns the submission future and response stream; dropping either must release the
/// corresponding transport work. HTTP health checks and proxying are separate.
pub(crate) trait GenerateTransport: Send + Sync {
    fn generate(
        &self,
        request: GenerateRequest,
    ) -> BoxFuture<'_, Result<TokenStream, ResponseError>>;
}

// Bound pending submissions per request without duplicating scheduler admission.
const CONCURRENT_ENGINE_SUBMISSIONS: usize = 32;

/// Shared generation policy and decoding, independent of the engine transport.
pub(crate) struct GenerationService {
    transport: std::sync::Arc<dyn GenerateTransport>,
    pub(crate) decoder: TokenDecoder,
}

impl GenerationService {
    pub(crate) fn new(
        transport: std::sync::Arc<dyn GenerateTransport>,
        decoder: TokenDecoder,
    ) -> Self {
        Self { transport, decoder }
    }

    pub(crate) async fn generate(
        &self,
        mut request: GenerateRequest,
    ) -> Result<GenerationStream, ResponseError> {
        let decode = self.decoder.prepare(&mut request)?;
        let tokens = self.transport.generate(request).await?;
        Ok(self.decoder.decode(tokens, decode))
    }

    /// Establish all choice streams before consumption, retaining input order.
    pub(crate) async fn generate_many(
        &self,
        inputs: Vec<GenerateRequest>,
    ) -> Result<Vec<GenerationStream>, ResponseError> {
        futures::stream::iter(inputs.into_iter().map(|input| self.generate(input)))
            .buffered(CONCURRENT_ENGINE_SUBMISSIONS)
            .try_collect()
            .await
    }
}

fn invalid(message: impl Into<String>) -> ResponseError {
    ResponseError {
        kind: crate::ResponseErrorKind::InvalidRequest,
        message: message.into(),
    }
}

fn internal(message: impl Into<String>) -> ResponseError {
    ResponseError {
        kind: crate::ResponseErrorKind::Internal,
        message: message.into(),
    }
}

#[cfg(test)]
pub(crate) mod test_utils;
#[cfg(test)]
mod tests;
