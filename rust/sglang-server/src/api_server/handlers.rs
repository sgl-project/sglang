//! The transport-neutral handlers behind both wire adapters: one
//! [`CoreHandle`] whose operations are split by function (generation,
//! control, info, tokenization, health, HiCache). `core` keeps what they are
//! built from: errors, events, the fan-out, peers, and media prefetch.

mod control;
mod corpus;
mod generate;
mod handle;
mod health;
mod hicache;
mod info;
mod parsers;
#[cfg(test)]
mod test_utils;
mod tokenize;
mod weights;

pub(crate) use control::ProfileOutcome;
pub(crate) use corpus::CorpusSource;
pub(crate) use generate::{CoreCall, recv_indexed};
pub(crate) use handle::{CoreConfig, CoreHandle, CoreMetadata};
pub(crate) use hicache::HiCacheManager;
pub(crate) use parsers::{ToolSpec, parse_function_call, separate_reasoning};
pub(crate) use weights::WeightGroupOutcome;
