use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MmError {
    InvalidRequest(String),
    Internal(String),
}

pub type Result<T> = std::result::Result<T, MmError>;

impl MmError {
    pub fn invalid(message: impl Into<String>) -> Self {
        Self::InvalidRequest(message.into())
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self::Internal(message.into())
    }

    pub fn message(&self) -> &str {
        match self {
            Self::InvalidRequest(message) | Self::Internal(message) => message,
        }
    }
}

impl fmt::Display for MmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.message())
    }
}

impl std::error::Error for MmError {}
