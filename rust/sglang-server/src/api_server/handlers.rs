//! The transport-neutral handlers behind both wire adapters: one
//! [`CoreHandle`] whose operations are split by function (generation,
//! control, info, tokenization, health, HiCache). `core` keeps what they are
//! built from: errors, events, the fan-out, peers, and media prefetch.

mod control;
mod generate;
mod handle;
mod health;
mod hicache;
mod info;
#[cfg(test)]
mod test_utils;
mod tokenize;

pub(crate) use generate::{CoreCall, recv_indexed};
pub(crate) use handle::{CoreConfig, CoreHandle, CoreMetadata};
pub(crate) use hicache::HiCacheManager;
