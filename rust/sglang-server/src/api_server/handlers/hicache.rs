//! HiCache management shared by the HTTP and gRPC adapters.

use sglang_api_types::api::v1 as api;
use tokio::sync::RwLock;

use super::handle::CoreHandle;
use crate::api_server::core::CoreError;
use crate::api_server::core::fanout;
use crate::message::config::ServerArgs;
use crate::message::ids::Rid;
use crate::message::io_struct::{
    AttachHiCacheStorageReqInput, ClearHiCacheReqInput, ControlRequest,
    DetachHiCacheStorageReqInput,
};

/// Python refuses storage-backend changes when the server has no admin key,
/// even to a request the API key authenticates.
pub(crate) const ADMIN_KEY_MISSING: &str = "This endpoint requires admin API key, but this \
    server was started without one (admin-api-key). Restart with --admin-api-key to enable.";

#[derive(Debug, thiserror::Error)]
pub(crate) enum HiCacheError {
    /// Refused before any rank saw the request.
    #[error("{}", ADMIN_KEY_MISSING)]
    AdminKeyMissing,
    #[error(transparent)]
    Core(#[from] CoreError),
}

pub(crate) struct HiCacheManager {
    core: CoreHandle,
    config: RwLock<api::HiCacheStatus>,
    admin_key_configured: bool,
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
            admin_key_configured: args.api_keys.admin_api_key.is_some(),
        }
    }

    pub(crate) fn admin_key_configured(&self) -> bool {
        self.admin_key_configured
    }

    /// Checked before the fan-out: every rank shares the server's keys, so a
    /// peer would refuse the relayed change while the local rank applied it.
    fn require_admin_key(&self) -> Result<(), HiCacheError> {
        if self.admin_key_configured {
            Ok(())
        } else {
            Err(HiCacheError::AdminKeyMissing)
        }
    }

    pub(crate) async fn status(&self) -> api::HiCacheStatus {
        self.config.read().await.clone()
    }

    /// Clear the storage backend on every DP rank; cleared only where every
    /// rank cleared.
    pub(crate) async fn clear(&self) -> Result<api::HiCacheStorageResponse, CoreError> {
        let outputs = self
            .core
            .control(&|| {
                ControlRequest::ClearHiCacheReqInput(ClearHiCacheReqInput::new(
                    Rid::new().to_string(),
                ))
            })
            .await
            .decode::<ClearHiCacheOutput>("HiCache clear")?;
        let (success, message) =
            fanout::merge_status(outputs.iter().map(|output| (output.success, "")));
        Ok(api::HiCacheStorageResponse { success, message })
    }

    /// Detach the storage backend on every DP rank. The recorded configuration
    /// changes only when every rank detached, as Python's does.
    pub(crate) async fn detach(&self) -> Result<api::HiCacheStorageResponse, HiCacheError> {
        self.require_admin_key()?;
        let mut config = self.config.write().await;
        let outputs = self
            .core
            .control(&|| {
                ControlRequest::DetachHiCacheStorageReqInput(DetachHiCacheStorageReqInput::new(
                    Rid::new().to_string(),
                ))
            })
            .await
            .decode::<DetachHiCacheOutput>("HiCache detach")?;
        let (success, message) = fanout::merge_status(
            outputs
                .iter()
                .map(|output| (output.success, output.message.as_str())),
        );
        if success {
            config.hicache_storage_backend = None;
            config.hicache_storage_backend_extra_config = None;
        }
        Ok(api::HiCacheStorageResponse { success, message })
    }

    pub(crate) async fn attach(
        &self,
        body: api::AttachHiCacheStorageRequest,
    ) -> Result<api::HiCacheStorageResponse, HiCacheError> {
        self.require_admin_key()?;
        let backend = body.hicache_storage_backend.as_ref().ok_or_else(|| {
            CoreError::InvalidArgument("missing field `hicache_storage_backend`".into())
        })?;
        let mut config = self.config.write().await;
        let outputs = self
            .core
            .control(&|| {
                ControlRequest::AttachHiCacheStorageReqInput(AttachHiCacheStorageReqInput::new(
                    Rid::new().to_string(),
                    backend.clone(),
                    body.hicache_storage_backend_extra_config_json.clone(),
                    body.hicache_storage_prefetch_policy.clone(),
                    body.hicache_write_policy.clone(),
                ))
            })
            .await
            .decode::<AttachHiCacheOutput>("HiCache attach")?;
        // Attached only where every rank attached; a partial attach is
        // reported as a failure (no rollback, as in Python).
        let (success, message) = fanout::merge_status(
            outputs
                .iter()
                .map(|output| (output.success, output.message.as_str())),
        );
        if success {
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
        Ok(api::HiCacheStorageResponse { success, message })
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
    use crate::api_server::core::fanout::ControlTarget;
    use crate::api_server::core::fanout::testing::DummyTarget;
    use crate::api_server::handlers::{CoreConfig, CoreMetadata};
    use crate::message::response::ResponseItem;
    use crate::tokenizer_manager::wiring::TmEvent;
    use bytes::Bytes;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    fn manager_with(
        admin_api_key: Option<&str>,
        peer_endpoints: Vec<Arc<dyn ControlTarget>>,
    ) -> (Arc<HiCacheManager>, flume::Receiver<TmEvent>) {
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
                multimodal: false,
                peer_endpoints,
                metadata: CoreMetadata::default(),
                metrics: None,
                request_log: None,
                generate_validator: None,
            },
        );
        let mut args = ServerArgs::default();
        args.api_keys.admin_api_key = admin_api_key.map(str::to_owned);
        (Arc::new(HiCacheManager::new(core, &args)), intake_rx)
    }

    fn success_reply() -> Bytes {
        rmp_serde::to_vec_named(&serde_json::json!({"success": true}))
            .unwrap()
            .into()
    }

    /// Without an admin key, attach and detach are refused before the
    /// fan-out, so neither the local scheduler nor a peer rank receives them;
    /// with one, both do.
    #[tokio::test]
    async fn storage_changes_without_an_admin_key_reach_no_rank() {
        let peer_calls = Arc::new(AtomicUsize::new(0));
        let peer = |calls: Arc<AtomicUsize>| {
            DummyTarget::replying(&[1], move |_| {
                calls.fetch_add(1, Ordering::Relaxed);
                Ok(success_reply())
            })
        };
        let attach_file = || api::AttachHiCacheStorageRequest {
            hicache_storage_backend: Some("file".into()),
            ..Default::default()
        };

        let (manager, intake_rx) = manager_with(None, vec![peer(peer_calls.clone())]);
        assert!(matches!(
            manager.attach(attach_file()).await,
            Err(HiCacheError::AdminKeyMissing)
        ));
        assert!(matches!(
            manager.detach().await,
            Err(HiCacheError::AdminKeyMissing)
        ));
        assert!(intake_rx.is_empty(), "the local scheduler received it");
        assert_eq!(peer_calls.load(Ordering::Relaxed), 0, "a peer received it");
        assert_eq!(manager.status().await, api::HiCacheStatus::default());

        let (manager, intake_rx) = manager_with(Some("sk-admin"), vec![peer(peer_calls.clone())]);
        let attach = tokio::spawn({
            let manager = manager.clone();
            async move { manager.attach(attach_file()).await }
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
        request
            .sink
            .try_send(ResponseItem::Control(success_reply()))
            .unwrap();
        assert!(attach.await.unwrap().unwrap().success);
        assert_eq!(peer_calls.load(Ordering::Relaxed), 1);
    }

    /// GET snapshots share the read lock, while an in-flight attach excludes readers.
    #[tokio::test]
    async fn hicache_readers_wait_for_successful_configuration_update() {
        let (manager, intake_rx) = manager_with(Some("sk-admin"), Vec::new());
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
            .try_send(ResponseItem::Control(success_reply()))
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
