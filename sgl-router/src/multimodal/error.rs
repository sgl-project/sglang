use std::time::Duration;

use thiserror::Error;

use super::types::Modality;

pub type MultiModalResult<T> = Result<T, MultiModalError>;

#[derive(Debug, Error)]
pub enum MediaConnectorError {
    #[error("unsupported media scheme: {0}")]
    UnsupportedScheme(String),
    #[error("invalid media URL: {0}")]
    InvalidUrl(String),
    #[error("media domain '{0}' is not in the allow list")]
    DisallowedDomain(String),
    #[error("local media path is not allowed: {0}")]
    DisallowedLocalPath(String),
    #[error("HTTP error while fetching media: {0}")]
    Http(#[from] reqwest::Error),
    #[error("I/O error while reading media: {0}")]
    Io(#[from] std::io::Error),
    #[error("base64 decode error: {0}")]
    Base64Decode(#[from] base64::DecodeError),
    #[error("data URL parse error: {0}")]
    DataUrl(String),
    #[error("media decode task failed: {0}")]
    Blocking(#[from] tokio::task::JoinError),
    #[error("image decode error: {0}")]
    Image(#[from] image::ImageError),
    #[error("media fetch timed out after {0:?}")]
    Timeout(Duration),
}

#[derive(Debug, Error)]
pub enum MultiModalError {
    #[error(transparent)]
    Media(#[from] MediaConnectorError),
    #[error("unsupported content part: {0}")]
    UnsupportedContent(&'static str),
    #[error("too many {modality:?} items provided. limit={limit}")]
    ModalityLimit { modality: Modality, limit: usize },
    #[error("tracker task join error: {0}")]
    Join(#[from] tokio::task::JoinError),
    #[error("tracker validation error: {0}")]
    Validation(String),
}
