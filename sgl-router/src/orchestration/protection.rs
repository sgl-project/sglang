//! The Router owns one explicitly configured backend protection domain. Engine
//! storage_targets supplies actual K/V keys; no event hash or prefix digest is a key.
use super::{client::EngineClient, OrchestrationConfig};
use crate::{discovery::ModelId, workers::WorkerRegistry};
use serde::Deserialize;
use serde_json::json;
use sgl_kv_orchestrator::{backend_protection::*, models::PrefixTarget};
use std::{sync::Arc, time::Duration};

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BackendOwnerConfig {
    pub admin_url: String,
    pub domain: ProtectionDomain,
    pub owner_epoch: String,
    /// FreshDomain requires an explicit external single-writer/recovery assertion.
    /// Omit on restart until old writes/claims have been reconciled externally.
    pub fresh_domain_assertion: Option<String>,
    /// Must belong to this deployment and be known absent. API probe cannot
    /// mutate an existing application's object.
    pub private_missing_probe_key: String,
    pub max_ttl_ms: u64,
    pub max_objects: usize,
    pub max_objects_per_claim: usize,
    pub max_claims_per_object: usize,
    pub max_inflight_batches: usize,
    pub batch_size: usize,
}
struct Resolver {
    client: EngineClient,
    registry: Arc<WorkerRegistry>,
    model: ModelId,
    config: OrchestrationConfig,
    domain: ProtectionDomain,
}
impl StorageTargetResolver for Resolver {
    fn resolve<'a>(
        &'a self,
        target: &'a PrefixTarget,
    ) -> BoxFuture<'a, Result<ResolvedStorageTarget, ProtectionError>> {
        Box::pin(async move {
            for worker in self
                .registry
                .healthy_workers_for(&self.model)
                .into_iter()
                .take(self.config.max_workers)
            {
                let Ok(caps) = self.client.capabilities(&worker.url).await else {
                    continue;
                };
                if !caps.compatible(target)
                    || caps.public["storage_backend"] != "mooncake"
                    || caps.public["backend_namespace"] != self.domain.storage_namespace
                    || caps.public["storage_targets"] != true
                {
                    continue;
                }
                let command = json!({"kind":"storage_targets","context":caps.context(&self.config.caller_id),"target":target,
                    "location":{"tier":"storage","backend":"mooncake","backend_namespace":self.domain.storage_namespace}});
                let Ok(reply) = self.client.command(&worker.url, &command).await else {
                    continue;
                };
                let wire: EngineStorageTargets = serde_json::from_value(reply)
                    .map_err(|_| ProtectionError::ResolverScopeMismatch)?;
                return wire.into_resolved(target, &self.domain, &caps.salt_encoding);
            }
            Err(ProtectionError::ResolverScopeMismatch)
        })
    }
}
pub async fn start(
    config: &OrchestrationConfig,
    registry: Arc<WorkerRegistry>,
    model: ModelId,
    client: EngineClient,
) -> Result<Option<BackendProtectionOwner>, String> {
    let Some(owner) = &config.backend_owner else {
        return Ok(None);
    };
    let mut backend = MooncakeAdminBackend::new(
        &owner.admin_url,
        Duration::from_millis(config.rpc_timeout_ms),
        owner.max_ttl_ms,
        config.max_engine_bytes,
    )
    .map_err(|e| e.to_string())?;
    let recovery = if let Some(assertion) = &owner.fresh_domain_assertion {
        if assertion.trim().is_empty() {
            return Err("empty FreshDomain assertion".into());
        }
        backend
            .verify_api_with_missing_key(&owner.domain.tenant_id, &owner.private_missing_probe_key)
            .await
            .map_err(|e| e.to_string())?;
        RecoveryMode::FreshDomain {
            deployment_assertion: assertion.clone(),
        }
    } else {
        RecoveryMode::UnknownDomain {
            reason: "Router restarted without externally established fresh domain".into(),
        }
    };
    let resolver = Arc::new(Resolver {
        client,
        registry,
        model,
        config: config.clone(),
        domain: owner.domain.clone(),
    });
    BackendProtectionOwner::spawn(
        ProtectionConfig {
            domain: owner.domain.clone(),
            owner_epoch: owner.owner_epoch.clone(),
            recovery,
            max_claims: config.limits.max_claims,
            max_tombstones: config.limits.max_tombstones,
            max_objects: owner.max_objects,
            max_objects_per_claim: owner.max_objects_per_claim,
            max_claims_per_object: owner.max_claims_per_object,
            max_identity_bytes: config.limits.max_identifier_bytes,
            command_capacity: config.queue_capacity,
            max_inflight_batches: owner.max_inflight_batches,
            max_batches_per_tick: 1,
            batch_size: owner.batch_size,
            max_ttl: Duration::from_millis(owner.max_ttl_ms),
            rpc_observation_timeout: Duration::from_millis(config.rpc_timeout_ms),
            retry_delay: Duration::from_millis(config.poll_ms),
        },
        Arc::new(backend),
        resolver,
    )
    .map(Some)
    .map_err(|e| e.to_string())
}
