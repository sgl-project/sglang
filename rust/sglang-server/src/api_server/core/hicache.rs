//! HiCache management shared by the HTTP and gRPC adapters.

use sglang_api_types::api::v1 as api;
use tokio::sync::RwLock;

use super::{CoreError, CoreHandle};
use crate::message::config::ServerArgs;
use crate::message::ids::Rid;
use crate::message::io_struct::{
    AttachHiCacheStorageReqInput, ClearHiCacheReqInput, ControlRequest,
    DetachHiCacheStorageReqInput,
};

pub(crate) struct HiCacheManager {
    core: CoreHandle,
    config: RwLock<api::HiCacheStatus>,
}

impl HiCacheManager {
    pub(crate) fn new(core: CoreHandle, args: &ServerArgs) -> Self {
        Self {
            core,
            config: RwLock::new(api::HiCacheStatus {
                hicache_storage_backend: args.hicache_storage_backend.clone(),
                hicache_storage_backend_extra_config: args
                    .hicache_storage_backend_extra_config
                    .clone(),
                hicache_storage_prefetch_policy: args.hicache_storage_prefetch_policy.clone(),
                hicache_write_policy: args.hicache_write_policy.clone(),
            }),
        }
    }

    pub(crate) async fn status(&self) -> api::HiCacheStatus {
        self.config.read().await.clone()
    }

    pub(crate) async fn clear(&self) -> Result<api::HiCacheStorageResponse, CoreError> {
        let bytes = self
            .core
            .control(ControlRequest::ClearHiCacheReqInput(
                ClearHiCacheReqInput::new(Rid::new().to_string()),
            ))
            .await?;
        rmp_serde::from_slice::<ClearHiCacheOutput>(&bytes)
            .map(|output| api::HiCacheStorageResponse {
                success: output.success,
                message: String::new(),
            })
            .map_err(|error| {
                CoreError::InvalidResponse(format!("invalid HiCache clear response: {error}"))
            })
    }

    pub(crate) async fn detach(&self) -> Result<api::HiCacheStorageResponse, CoreError> {
        let mut config = self.config.write().await;
        let bytes = self
            .core
            .control(ControlRequest::DetachHiCacheStorageReqInput(
                DetachHiCacheStorageReqInput::new(Rid::new().to_string()),
            ))
            .await?;
        let output: DetachHiCacheOutput = rmp_serde::from_slice(&bytes).map_err(|error| {
            CoreError::InvalidResponse(format!("invalid HiCache detach response: {error}"))
        })?;
        if output.success {
            config.hicache_storage_backend = None;
            config.hicache_storage_backend_extra_config = None;
        }
        Ok(api::HiCacheStorageResponse {
            success: output.success,
            message: output.message,
        })
    }

    pub(crate) async fn attach(
        &self,
        body: api::AttachHiCacheStorageRequest,
    ) -> Result<api::HiCacheStorageResponse, CoreError> {
        let backend = body.hicache_storage_backend.as_ref().ok_or_else(|| {
            CoreError::InvalidArgument("missing field `hicache_storage_backend`".into())
        })?;
        let mut config = self.config.write().await;
        let bytes = self
            .core
            .control(ControlRequest::AttachHiCacheStorageReqInput(
                AttachHiCacheStorageReqInput::new(
                    Rid::new().to_string(),
                    backend.clone(),
                    body.hicache_storage_backend_extra_config_json.clone(),
                    body.hicache_storage_prefetch_policy.clone(),
                    body.hicache_write_policy.clone(),
                ),
            ))
            .await?;
        let output: AttachHiCacheOutput = rmp_serde::from_slice(&bytes).map_err(|error| {
            CoreError::InvalidResponse(format!("invalid HiCache attach response: {error}"))
        })?;
        if output.success {
            config.hicache_storage_backend = body.hicache_storage_backend;
            if let Some(value) = body.hicache_storage_backend_extra_config_json {
                config.hicache_storage_backend_extra_config = Some(value);
            }
            if let Some(value) = body.hicache_storage_prefetch_policy {
                config.hicache_storage_prefetch_policy = Some(value);
            }
            if let Some(value) = body.hicache_write_policy {
                config.hicache_write_policy = Some(value);
            }
        }
        Ok(api::HiCacheStorageResponse {
            success: output.success,
            message: output.message,
        })
    }
}

/// Scheduler response for clearing HiCache storage.
#[derive(serde::Deserialize)]
struct ClearHiCacheOutput {
    success: bool,
}

/// Scheduler response for detaching HiCache storage.
#[derive(serde::Deserialize)]
struct DetachHiCacheOutput {
    success: bool,
    #[serde(default)]
    message: String,
}

/// Scheduler response for attaching HiCache storage.
#[derive(serde::Deserialize)]
struct AttachHiCacheOutput {
    success: bool,
    #[serde(default)]
    message: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api_server::core::{CoreConfig, CoreMetadata};
    use crate::message::response::ResponseItem;
    use crate::tokenizer_manager::wiring::TmEvent;
    use bytes::Bytes;
    use std::sync::Arc;
    use std::time::Duration;

    /// GET snapshots share the read lock, while an in-flight attach excludes readers.
    #[tokio::test]
    async fn hicache_readers_wait_for_successful_configuration_update() {
        let (intake_tx, intake_rx) = flume::unbounded();
        let (abort_tx, _) = flume::unbounded();
        let core = CoreHandle::new(
            intake_tx,
            abort_tx,
            CoreConfig {
                response_capacity: 8,
                response_activity: Default::default(),
                startup_ready: true,
                is_disaggregation: false,
                mm_limits: Default::default(),
                metadata: CoreMetadata::default(),
            },
        );
        let manager = Arc::new(HiCacheManager::new(core, &ServerArgs::default()));
        // Readers can share the lock, but a writer must wait for them to finish.
        let reader = manager.config.read().await;
        assert!(manager.config.try_read().is_ok());
        assert!(manager.config.try_write().is_err());
        drop(reader);

        // Run attach separately so this test can act as the scheduler.
        let handle = manager.clone();
        let attach = tokio::spawn(async move {
            handle
                .attach(api::AttachHiCacheStorageRequest {
                    hicache_storage_backend: Some("file".into()),
                    ..Default::default()
                })
                .await
        });
        let TmEvent::Intake { request, admission } =
            tokio::time::timeout(Duration::from_secs(5), intake_rx.recv_async())
                .await
                .unwrap()
                .unwrap()
        else {
            panic!("expected control intake");
        };
        assert!(admission.try_accept());
        // Attach holds the write lock while waiting for the scheduler response.
        assert!(manager.config.try_read().is_err());
        // Reply successfully so attach can update the configuration and release the lock.
        request
            .sink
            .try_send(ResponseItem::Control(Bytes::from(
                rmp_serde::to_vec_named(&serde_json::json!({"success": true})).unwrap(),
            )))
            .unwrap();
        tokio::time::timeout(Duration::from_secs(5), attach)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        // Once attach completes, GET can read the updated configuration.
        assert_eq!(
            manager.status().await.hicache_storage_backend.as_deref(),
            Some("file")
        );
    }
}
