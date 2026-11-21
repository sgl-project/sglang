//! The transport-neutral handlers behind both wire adapters: one
//! [`CoreHandle`] whose operations are split by function (generation,
//! control, info, tokenization, health, HiCache). `core` keeps what they are
//! built from: errors, events, the fan-out, peers, and media prefetch.

mod control;
mod corpus;
mod frontend_state;
mod generate;
mod handle;
mod health;
mod hicache;
mod info;
mod parsers;
#[cfg(test)]
mod test_utils;
mod tokenize;
mod weight_coordinator;
#[cfg(test)]
mod weight_update_relay_tests;
mod weights;

pub(crate) use control::{DumperOutcome, ProfileOutcome};
pub(crate) use corpus::CorpusSource;
pub use generate::{AbortEnding, Aborted};
pub(crate) use generate::{CoreCall, PendingGeneration, recv_indexed};
pub(crate) use handle::{CoreConfig, CoreHandle, CoreMetadata, GRACEFUL_CLOSE_TIMEOUT};
pub(crate) use hicache::{ADMIN_KEY_MISSING, HiCacheError, HiCacheManager};
pub(crate) use info::ElasticEpState;
pub(crate) use parsers::{ToolSpec, parse_function_call, separate_reasoning};
pub(crate) use weight_coordinator::coordinated_route;
pub(crate) use weights::{
    DiskUpdateOutcome, LORA_DISABLED, PluginUpdateOutcome, WeightGroupOutcome, WeightUpdateOutcome,
};
