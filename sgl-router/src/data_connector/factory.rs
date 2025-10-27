// factory.rs
//
// Factory function to create storage backends based on configuration.
// This centralizes storage initialization logic and fixes the bug where
// conversation_item_storage was missing/incorrect in server.rs.

use std::sync::Arc;

use tracing::info;

use super::{
    core::{ConversationItemStorage, ConversationStorage, ResponseStorage},
    memory::{MemoryConversationItemStorage, MemoryConversationStorage, MemoryResponseStorage},
    noop::{NoOpConversationItemStorage, NoOpConversationStorage, NoOpResponseStorage},
    oracle::{OracleConversationItemStorage, OracleConversationStorage, OracleResponseStorage},
};
use crate::config::{HistoryBackend, OracleConfig, RouterConfig};

/// Type alias for the storage tuple returned by factory functions.
/// This avoids clippy::type_complexity warnings while keeping Arc explicit.
pub type StorageTuple = (
    Arc<dyn ResponseStorage>,
    Arc<dyn ConversationStorage>,
    Arc<dyn ConversationItemStorage>,
);

/// Create all three storage backends based on router configuration.
///
/// # Arguments
/// * `config` - Router configuration containing history_backend and oracle settings
///
/// # Returns
/// Tuple of (response_storage, conversation_storage, conversation_item_storage)
///
/// # Errors
/// Returns error string if Oracle configuration is missing or initialization fails
pub fn create_storage(config: &RouterConfig) -> Result<StorageTuple, String> {
    match config.history_backend {
        HistoryBackend::Memory => {
            info!("Initializing data connector: Memory");
            Ok((
                Arc::new(MemoryResponseStorage::new()),
                Arc::new(MemoryConversationStorage::new()),
                Arc::new(MemoryConversationItemStorage::new()),
            ))
        }
        HistoryBackend::None => {
            info!("Initializing data connector: None (no persistence)");
            Ok((
                Arc::new(NoOpResponseStorage::new()),
                Arc::new(NoOpConversationStorage::new()),
                Arc::new(NoOpConversationItemStorage::new()),
            ))
        }
        HistoryBackend::Oracle => {
            let oracle_cfg = config
                .oracle
                .clone()
                .ok_or("oracle configuration is required when history_backend=oracle")?;

            info!(
                "Initializing data connector: Oracle ATP (pool: {}-{})",
                oracle_cfg.pool_min, oracle_cfg.pool_max
            );

            let storages = create_oracle_storage(&oracle_cfg)?;

            info!("Data connector initialized successfully: Oracle ATP");
            Ok(storages)
        }
    }
}

/// Create Oracle storage backends
fn create_oracle_storage(oracle_cfg: &OracleConfig) -> Result<StorageTuple, String> {
    let response_storage = OracleResponseStorage::new(oracle_cfg.clone())
        .map_err(|err| format!("failed to initialize Oracle response storage: {err}"))?;

    let conversation_storage = OracleConversationStorage::new(oracle_cfg.clone())
        .map_err(|err| format!("failed to initialize Oracle conversation storage: {err}"))?;

    let conversation_item_storage = OracleConversationItemStorage::new(oracle_cfg.clone())
        .map_err(|err| format!("failed to initialize Oracle conversation item storage: {err}"))?;

    Ok((
        Arc::new(response_storage),
        Arc::new(conversation_storage),
        Arc::new(conversation_item_storage),
    ))
}
