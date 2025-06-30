use thiserror::Error;

#[derive(Debug, Error)]
pub enum RoutingError {
    #[error("No healthy workers available")]
    NoHealthyWorkers,
    #[error("Could not extract text from request for cache-aware routing")]
    TextExtractionFailed,
}
