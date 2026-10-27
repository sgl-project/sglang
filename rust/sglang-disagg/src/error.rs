//! Error type for the wire codecs.
//!
//! Parsing mirrors Python: malformed frames map to the same "drop/raise"
//! behavior as the `from_zmq` parsers. A `Codec` error corresponds to the
//! Python path raising (`ValueError`/`struct.error`) or dropping the message.

use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// Byte-level decode failure (bad utf-8, truncated struct, bad int frame).
    Codec(String),
    /// Semantically invalid input (e.g. length mismatch Python raises
    /// `ValueError` for).
    Invalid(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Codec(m) => write!(f, "codec: {}", m),
            Error::Invalid(m) => write!(f, "invalid: {}", m),
        }
    }
}

impl std::error::Error for Error {}
