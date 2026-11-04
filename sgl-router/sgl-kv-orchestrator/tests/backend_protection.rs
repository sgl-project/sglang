use sgl_kv_orchestrator::{
    backend_protection::*,
    models::{Compatibility, PrefixTarget},
};
use std::{sync::Arc, time::Duration};
use tokio::sync::{mpsc, oneshot};

fn domain() -> ProtectionDomain {
    ProtectionDomain {
        cluster: "cluster-a".into(),
        tenant_id: "default".into(),
        model_namespace: "weights".into(),
        storage_namespace: "weights".into(),
        layout_id: "full-kv".into(),
    }
}
fn config() -> ProtectionConfig {
    ProtectionConfig {
        domain: domain(),
        owner_epoch: "epoch-a".into(),
        recovery: RecoveryMode::FreshDomain {
            deployment_assertion: "test domain is exclusively owned, no prior RPCs".into(),
        },
        max_claims: 16,
        max_tombstones: 32,
        max_objects: 64,
        max_objects_per_claim: 16,
        max_claims_per_object: 16,
        max_identity_bytes: 256,
        command_capacity: 16,
        max_inflight_batches: 4,
        max_batches_per_tick: 4,
        batch_size: 4096,
        max_ttl: Duration::from_secs(60),
        rpc_observation_timeout: Duration::from_secs(2),
        retry_delay: Duration::from_secs(1),
    }
}
fn prefix(pages: u64) -> PrefixTarget {
    PrefixTarget {
        kind: "prefix".into(),
        compatibility: Compatibility {
            namespace: "weights".into(),
            cache_salt: None,
            layout_id: "full-kv".into(),
            page_size: 4,
            components: vec!["full".into()],
            hash_scheme: "sglang.sha256.u32le.i64.v1".into(),
        },
        start_token: 0,
        end_token: pages * 4,
        token_ids: Some(vec![1; (pages * 4) as usize]),
        blocks: vec![],
    }
}
fn request(id: &str, pages: u64, ttl: f64) -> ProtectionRequest {
    ProtectionRequest {
        identity: ClaimIdentity {
            caller_id: "caller-a".into(),
            task_incarnation: "task-a".into(),
            claim_id: id.into(),
        },
        revision: 1,
        mode: ClaimMode::Acquire,
        target: prefix(pages),
        ttl_seconds: Some(ttl),
        required_protection: "backend_soft_pin".into(),
    }
}
fn update(
    r: &ProtectionRequest,
    mode: ClaimMode,
    revision: u64,
    ttl: Option<f64>,
) -> ProtectionRequest {
    ProtectionRequest {
        mode,
        revision,
        ttl_seconds: ttl,
        ..r.clone()
    }
}
struct Resolver;
impl StorageTargetResolver for Resolver {
    fn resolve<'a>(
        &'a self,
        target: &'a PrefixTarget,
    ) -> BoxFuture<'a, Result<ResolvedStorageTarget, ProtectionError>> {
        Box::pin(async move {
            Ok(ResolvedStorageTarget {
                domain: domain(),
                scopes: (0..target.end_token / 4)
                    .map(|page| ResolvedStorageScope {
                        component: "full".into(),
                        start_token: page * 4,
                        end_token: page * 4 + 4,
                        dependency_start: 0,
                        keys: vec![format!("digest-{page}-k"), format!("digest-{page}-v")],
                    })
                    .collect(),
            })
        })
    }
}
struct Call {
    batch: BackendBatch,
    reply: oneshot::Sender<BackendReply>,
}
struct ControlledBackend {
    calls: mpsc::Sender<Call>,
}
impl SoftPinBackend for ControlledBackend {
    fn api_verified(&self) -> bool {
        true
    }
    fn max_ttl_ms(&self) -> u64 {
        60000
    }
    fn mutate<'a>(&'a self, batch: BackendBatch) -> BoxFuture<'a, BackendReply> {
        Box::pin(async move {
            let (reply, rx) = oneshot::channel();
            self.calls.send(Call { batch, reply }).await.unwrap();
            rx.await
                .unwrap_or_else(|_| BackendReply::Unknown("test abandoned".into()))
        })
    }
}
fn harness(c: ProtectionConfig) -> (BackendProtectionOwner, mpsc::Receiver<Call>) {
    let (tx, rx) = mpsc::channel(64);
    let owner = BackendProtectionOwner::spawn(
        c,
        Arc::new(ControlledBackend { calls: tx }),
        Arc::new(Resolver),
    )
    .unwrap();
    (owner, rx)
}
fn success(call: Call) {
    let rows = call
        .batch
        .keys
        .iter()
        .map(|key| KeyResult {
            key: key.clone(),
            error_code: 0,
        })
        .collect();
    call.reply.send(BackendReply::Quiescent(rows)).unwrap();
}
async fn settle() {
    for _ in 0..8 {
        tokio::task::yield_now().await;
    }
}
async fn view(owner: &BackendProtectionOwner, r: &ProtectionRequest) -> ProtectionClaimView {
    settle().await;
    owner
        .query(Some(r.identity.clone()))
        .await
        .unwrap()
        .claim
        .unwrap()
}

#[tokio::test(start_paused = true)]
async fn shared_release_and_short_renew_preserve_other_callers_confirmed_deadline() {
    let (owner, mut calls) = harness(config());
    let a = request("a", 1, 20.0);
    owner.submit(a.clone()).await.unwrap();
    success(calls.recv().await.unwrap());
    let mut b = request("b", 1, 40.0);
    b.identity.caller_id = "caller-b".into();
    b.identity.claim_id = a.identity.claim_id.clone();
    owner.submit(b.clone()).await.unwrap();
    success(calls.recv().await.unwrap());
    assert!(view(&owner, &b).await.converged);
    owner
        .submit(update(&a, ClaimMode::Release, 2, None))
        .await
        .unwrap();
    settle().await;
    assert!(calls.try_recv().is_err());
    assert_eq!(view(&owner, &b).await.phase, ClaimPhase::Active);
    owner
        .submit(update(&b, ClaimMode::Renew, 2, Some(10.0)))
        .await
        .unwrap();
    let shortened = calls.recv().await.unwrap();
    assert!(matches!(
        shortened.batch.mutation,
        BackendMutation::SoftPin { ttl_ms: 10000 }
    ));
    success(shortened);
    owner
        .submit(update(&b, ClaimMode::Release, 3, None))
        .await
        .unwrap();
    let unpin = calls.recv().await.unwrap();
    assert_eq!(unpin.batch.mutation, BackendMutation::SoftUnpin);
    success(unpin);
    assert_eq!(view(&owner, &a).await.phase, ClaimPhase::Released);
    assert_eq!(owner.query(None).await.unwrap().objects, 0);
}

#[tokio::test(start_paused = true)]
async fn overlapping_prefixes_release_only_unshared_objects() {
    let (owner, mut calls) = harness(config());
    let a = request("short", 1, 20.0);
    let b = request("long", 2, 20.0);
    owner.submit(a.clone()).await.unwrap();
    success(calls.recv().await.unwrap());
    owner.submit(b.clone()).await.unwrap();
    success(calls.recv().await.unwrap());
    owner
        .submit(update(&b, ClaimMode::Release, 2, None))
        .await
        .unwrap();
    let unpin = calls.recv().await.unwrap();
    assert_eq!(unpin.batch.keys.len(), 2);
    assert!(unpin
        .batch
        .keys
        .iter()
        .all(|key| key.starts_with("digest-1")));
    success(unpin);
    assert_eq!(view(&owner, &a).await.phase, ClaimPhase::Active);
}

#[tokio::test(start_paused = true)]
async fn shortening_one_claim_honors_longer_shared_claim() {
    let (owner, mut calls) = harness(config());
    let a = request("a", 1, 40.0);
    let b = request("b", 1, 30.0);
    owner.submit(a.clone()).await.unwrap();
    success(calls.recv().await.unwrap());
    owner.submit(b.clone()).await.unwrap();
    settle().await;
    assert!(calls.try_recv().is_err());
    owner
        .submit(update(&a, ClaimMode::Renew, 2, Some(5.0)))
        .await
        .unwrap();
    let call = calls.recv().await.unwrap();
    assert_eq!(
        call.batch.mutation,
        BackendMutation::SoftPin { ttl_ms: 30000 }
    );
    success(call);
    assert_eq!(view(&owner, &b).await.confirmed_remaining_ms, Some(30000));
}

#[tokio::test(start_paused = true)]
async fn identical_revision_is_idempotent_stale_and_terminal_replays_cannot_revive() {
    let (owner, mut calls) = harness(config());
    let a = request("a", 1, 10.0);
    owner.submit(a.clone()).await.unwrap();
    success(calls.recv().await.unwrap());
    assert!(owner.submit(a.clone()).await.is_ok());
    settle().await;
    assert!(calls.try_recv().is_err());
    assert_eq!(
        owner
            .submit(update(&a, ClaimMode::Renew, 1, Some(9.0)))
            .await
            .unwrap_err(),
        ProtectionError::Conflict
    );
    let release = update(&a, ClaimMode::Release, 2, None);
    owner.submit(release.clone()).await.unwrap();
    success(calls.recv().await.unwrap());
    assert_eq!(
        owner.submit(a.clone()).await.unwrap_err(),
        ProtectionError::StaleRevision
    );
    assert_eq!(
        owner
            .submit(update(&a, ClaimMode::Renew, 3, Some(10.0)))
            .await
            .unwrap_err(),
        ProtectionError::TerminalClaim
    );
    assert_eq!(
        owner.submit(release).await.unwrap().phase,
        ClaimPhase::Released
    );
}

#[tokio::test(start_paused = true)]
async fn partial_object_success_does_not_claim_missing_kv_page() {
    let (owner, mut calls) = harness(config());
    let a = request("a", 2, 20.0);
    owner.submit(a.clone()).await.unwrap();
    let call = calls.recv().await.unwrap();
    let rows = call
        .batch
        .keys
        .iter()
        .map(|key| KeyResult {
            key: key.clone(),
            error_code: if key == "digest-1-v" { -704 } else { 0 },
        })
        .collect();
    call.reply.send(BackendReply::Quiescent(rows)).unwrap();
    let v = view(&owner, &a).await;
    assert_eq!(v.phase, ClaimPhase::Partial);
    assert!(v.scopes[0].confirmed_remaining_ms.is_some());
    assert!(v.scopes[1].missing);
    assert!(v.scopes[1].confirmed_remaining_ms.is_none());
    assert!(v.confirmed_remaining_ms.is_none());
}

#[tokio::test(start_paused = true)]
async fn missing_receipt_freezes_only_affected_object() {
    let (owner, mut calls) = harness(config());
    let a = request("a", 1, 20.0);
    owner.submit(a.clone()).await.unwrap();
    let call = calls.recv().await.unwrap();
    call.reply
        .send(BackendReply::Quiescent(vec![KeyResult {
            key: "digest-0-k".into(),
            error_code: 0,
        }]))
        .unwrap();
    assert_eq!(view(&owner, &a).await.phase, ClaimPhase::Unknown);
    owner
        .submit(update(&a, ClaimMode::Release, 2, None))
        .await
        .unwrap();
    let call = calls.recv().await.unwrap();
    assert_eq!(call.batch.keys, vec!["digest-0-k"]);
    success(call);
    settle().await;
    assert!(calls.try_recv().is_err());
    assert_eq!(owner.query(None).await.unwrap().uncertain_objects, 1);
}

#[tokio::test(start_paused = true)]
async fn unknown_effect_blocks_release_even_after_many_ttls_until_reconciled() {
    let (owner, mut calls) = harness(config());
    let a = request("a", 1, 5.0);
    owner.submit(a.clone()).await.unwrap();
    let call = calls.recv().await.unwrap();
    call.reply
        .send(BackendReply::Unknown(
            "timeout; server may still apply".into(),
        ))
        .unwrap();
    assert_eq!(view(&owner, &a).await.phase, ClaimPhase::Unknown);
    owner
        .submit(update(&a, ClaimMode::Release, 2, None))
        .await
        .unwrap();
    tokio::time::advance(Duration::from_secs(120)).await;
    settle().await;
    assert!(calls.try_recv().is_err());
    assert_eq!(owner.query(None).await.unwrap().uncertain_objects, 2);
    owner
        .reconcile_object_after_quiescence(
            "digest-0-k".into(),
            Some(Duration::from_secs(3)),
            "execution isolation and authoritative object query".into(),
        )
        .await
        .unwrap();
    // Reconciliation may leave the finite observed TTL to expire; it never
    // touches the still-unknown V object or claims it released.
    assert_eq!(owner.query(None).await.unwrap().uncertain_objects, 1);
}

#[tokio::test(start_paused = true)]
async fn observation_timeout_keeps_future_and_late_pin_cannot_revive_released_claim() {
    let (owner, mut calls) = harness(config());
    let a = request("a", 1, 20.0);
    owner.submit(a.clone()).await.unwrap();
    let old_pin = calls.recv().await.unwrap();
    tokio::time::advance(Duration::from_secs(3)).await;
    assert_eq!(view(&owner, &a).await.phase, ClaimPhase::Unknown);
    owner
        .submit(update(&a, ClaimMode::Release, 2, None))
        .await
        .unwrap();
    settle().await;
    assert!(calls.try_recv().is_err());
    success(old_pin);
    let unpin = calls.recv().await.unwrap();
    assert_eq!(unpin.batch.mutation, BackendMutation::SoftUnpin);
    success(unpin);
    let v = view(&owner, &a).await;
    assert_eq!(v.phase, ClaimPhase::Released);
    assert!(v.effective_protection.is_none());
}

#[tokio::test(start_paused = true)]
async fn late_short_pin_is_serialized_before_new_renew_and_never_confirms_new_desired_ttl() {
    let (owner, mut calls) = harness(config());
    let a = request("a", 1, 10.0);
    owner.submit(a.clone()).await.unwrap();
    let old_pin = calls.recv().await.unwrap();
    owner
        .submit(update(&a, ClaimMode::Renew, 2, Some(40.0)))
        .await
        .unwrap();
    settle().await;
    assert!(calls.try_recv().is_err());
    success(old_pin);
    let renewal = calls.recv().await.unwrap();
    let v = view(&owner, &a).await;
    assert_eq!(v.revision, 2);
    assert!(!v.converged);
    assert!(v.confirmed_remaining_ms.unwrap() <= 10000);
    assert_eq!(
        renewal.batch.mutation,
        BackendMutation::SoftPin { ttl_ms: 40000 }
    );
    success(renewal);
    assert!(view(&owner, &a).await.converged);
}

#[tokio::test(start_paused = true)]
async fn slow_ack_uses_send_time_lower_bound_and_expiry_runs_without_submissions() {
    let (owner, mut calls) = harness(config());
    let a = request("a", 1, 10.0);
    let mut changes = owner.changes();
    owner.submit(a.clone()).await.unwrap();
    let call = calls.recv().await.unwrap();
    assert!(owner.next_deadline().await.unwrap().is_some());
    tokio::time::advance(Duration::from_secs(1)).await;
    success(call);
    assert_eq!(view(&owner, &a).await.confirmed_remaining_ms, Some(9000));
    tokio::time::advance(Duration::from_secs(10)).await;
    let unpin = calls.recv().await.unwrap();
    assert_eq!(unpin.batch.mutation, BackendMutation::SoftUnpin);
    success(unpin);
    changes.changed().await.unwrap();
    let v = view(&owner, &a).await;
    assert_eq!(v.phase, ClaimPhase::Expired);
    assert!(v.tombstone);
    assert_eq!(v.desired_remaining_ms, 0);
}

#[tokio::test(start_paused = true)]
async fn released_claim_new_acquire_during_unpin_waits_for_final_receipt() {
    let (owner, mut calls) = harness(config());
    let a = request("a", 1, 20.0);
    owner.submit(a.clone()).await.unwrap();
    success(calls.recv().await.unwrap());
    owner
        .submit(update(&a, ClaimMode::Release, 2, None))
        .await
        .unwrap();
    let unpin = calls.recv().await.unwrap();
    let b = request("b", 1, 10.0);
    owner.submit(b.clone()).await.unwrap();
    assert!(view(&owner, &b).await.effective_protection.is_none());
    settle().await;
    assert!(calls.try_recv().is_err());
    success(unpin);
    let pin = calls.recv().await.unwrap();
    assert!(matches!(
        pin.batch.mutation,
        BackendMutation::SoftPin { .. }
    ));
    success(pin);
    assert_eq!(view(&owner, &b).await.phase, ClaimPhase::Active);
}

#[tokio::test(start_paused = true)]
async fn bounds_reject_admission_and_retain_tombstones() {
    let mut c = config();
    c.max_claims = 1;
    c.max_tombstones = 1;
    let (owner, mut calls) = harness(c);
    let a = request("a", 1, 10.0);
    owner.submit(a.clone()).await.unwrap();
    success(calls.recv().await.unwrap());
    assert_eq!(
        owner.submit(request("b", 1, 10.0)).await.unwrap_err(),
        ProtectionError::Capacity
    );
    owner
        .submit(update(&a, ClaimMode::Release, 2, None))
        .await
        .unwrap();
    success(calls.recv().await.unwrap());
    assert_eq!(
        owner.submit(request("b", 1, 10.0)).await.unwrap_err(),
        ProtectionError::Capacity
    );
    assert!(view(&owner, &a).await.tombstone);
}

#[tokio::test(start_paused = true)]
async fn unsupported_hard_pin_invalid_ttls_and_namespace_rejected_without_backend_io() {
    let (owner, mut calls) = harness(config());
    let a = request("a", 1, 10.0);
    let mut hard = a.clone();
    hard.required_protection = "eviction_exclusion".into();
    assert_eq!(
        owner.submit(hard).await.unwrap_err(),
        ProtectionError::UnsupportedProtection
    );
    for ttl in [0.0, -1.0, f64::NAN, f64::INFINITY, 61.0, 0.00001] {
        let mut bad = a.clone();
        bad.ttl_seconds = Some(ttl);
        assert_eq!(
            owner.submit(bad).await.unwrap_err(),
            ProtectionError::InvalidRequest
        );
    }
    let mut bad = a;
    bad.target.compatibility.namespace = "other-weights".into();
    assert_eq!(
        owner.submit(bad).await.unwrap_err(),
        ProtectionError::DomainMismatch
    );
    assert!(calls.try_recv().is_err());
}

#[tokio::test(start_paused = true)]
async fn restart_unknown_mode_never_automatically_takes_over_or_unpins() {
    let mut c = config();
    c.owner_epoch = "epoch-restarted".into();
    c.recovery = RecoveryMode::UnknownDomain {
        reason: "old claims/RPCs cannot be reconstructed".into(),
    };
    let (owner, mut calls) = harness(c);
    let a = request("old", 1, 5.0);
    assert_eq!(
        owner.submit(a.clone()).await.unwrap_err(),
        ProtectionError::RecoveryRequired
    );
    assert_eq!(
        owner
            .submit(update(&a, ClaimMode::Release, 2, None))
            .await
            .unwrap_err(),
        ProtectionError::RecoveryRequired
    );
    tokio::time::advance(Duration::from_secs(600)).await;
    let state = owner.query(Some(a.identity)).await.unwrap();
    assert!(state.recovery_unknown);
    assert!(state.claim.is_none());
    assert!(state.supported_protection.is_empty());
    assert!(calls.try_recv().is_err());
}

async fn one_http_reply(
    response: &'static str,
    delay: Duration,
) -> (String, tokio::task::JoinHandle<String>) {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let task = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        let mut buffer = Vec::new();
        let mut chunk = [0; 1024];
        loop {
            let n = stream.read(&mut chunk).await.unwrap();
            if n == 0 {
                break;
            }
            buffer.extend_from_slice(&chunk[..n]);
            if let Some(end) = buffer.windows(4).position(|w| w == b"\r\n\r\n") {
                let headers = String::from_utf8_lossy(&buffer[..end]);
                let length: usize = headers
                    .lines()
                    .find_map(|line| {
                        line.to_ascii_lowercase()
                            .strip_prefix("content-length:")
                            .map(|v| v.trim().parse().unwrap())
                    })
                    .unwrap_or(0);
                if buffer.len() >= end + 4 + length {
                    break;
                }
            }
        }
        tokio::time::sleep(delay).await;
        let wire = format!(
            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            response.len(),
            response
        );
        let _ = stream.write_all(wire.as_bytes()).await;
        String::from_utf8(buffer).unwrap()
    });
    (format!("http://{addr}"), task)
}
#[tokio::test]
async fn http_adapter_sends_real_admin_schema_and_retains_per_key_errors() {
    let (url, server) = one_http_reply(
        r#"{"results":[{"key":"k","error_code":0},{"key":"v","error_code":-704}]}"#,
        Duration::ZERO,
    )
    .await;
    let backend = MooncakeAdminBackend::new(&url, Duration::from_secs(2), 60000, 4096).unwrap();
    let result = backend
        .mutate(BackendBatch {
            tenant_id: "tenant-a".into(),
            keys: vec!["k".into(), "v".into()],
            mutation: BackendMutation::SoftPin { ttl_ms: 1234 },
        })
        .await;
    assert!(
        matches!(result, BackendReply::Quiescent(ref rows) if rows.len()==2 && rows[1].error_code == -704)
    );
    let request = server.await.unwrap();
    assert!(request.starts_with("POST /api/v1/soft_pin HTTP/1.1"));
    let body: serde_json::Value =
        serde_json::from_str(request.split("\r\n\r\n").nth(1).unwrap()).unwrap();
    assert_eq!(body["tenant_id"], "tenant-a");
    assert_eq!(body["ttl_ms"], 1234);
    let (url, server) = one_http_reply(
        r#"{"results":[{"key":"k","error_code":0}]}"#,
        Duration::ZERO,
    )
    .await;
    let backend = MooncakeAdminBackend::new(&url, Duration::from_secs(2), 60000, 4096).unwrap();
    backend
        .mutate(BackendBatch {
            tenant_id: "tenant-a".into(),
            keys: vec!["k".into()],
            mutation: BackendMutation::SoftUnpin,
        })
        .await;
    let request = server.await.unwrap();
    assert!(request.starts_with("POST /api/v1/soft_unpin HTTP/1.1"));
    assert!(!request.contains("ttl_ms"));
}
#[tokio::test]
async fn http_timeout_and_oversized_invalid_receipts_are_unknown_effects() {
    let (url, server) = one_http_reply(
        r#"{"results":[{"key":"k","error_code":0}]}"#,
        Duration::from_millis(80),
    )
    .await;
    let backend = MooncakeAdminBackend::new(&url, Duration::from_millis(10), 60000, 4096).unwrap();
    let batch = BackendBatch {
        tenant_id: "default".into(),
        keys: vec!["k".into()],
        mutation: BackendMutation::SoftPin { ttl_ms: 1000 },
    };
    assert!(matches!(
        backend.mutate(batch.clone()).await,
        BackendReply::Unknown(_)
    ));
    server.await.unwrap();
    let (url, server) = one_http_reply(
        r#"{"results":[{"key":"k","error_code":0}]}"#,
        Duration::ZERO,
    )
    .await;
    let backend = MooncakeAdminBackend::new(&url, Duration::from_secs(1), 60000, 8).unwrap();
    assert!(matches!(
        backend.mutate(batch).await,
        BackendReply::Unknown(_)
    ));
    server.await.unwrap();
}

/// Seed MOONCAKE_SMOKE_KEY in a live store and keep its owning segment/client
/// alive. This opt-in test modifies only that exact key and a unique missing key.
#[tokio::test]
#[ignore = "requires a real Mooncake 0.3.14 master and externally seeded private key"]
async fn live_mooncake_admin_soft_pin_unpin() {
    let url =
        std::env::var("MOONCAKE_ADMIN_URL").unwrap_or_else(|_| "http://127.0.0.1:52092".into());
    let tenant_id = std::env::var("MOONCAKE_TENANT_ID").unwrap_or_else(|_| "default".into());
    let key =
        std::env::var("MOONCAKE_SMOKE_KEY").expect("seed private key and set MOONCAKE_SMOKE_KEY");
    let mut backend =
        MooncakeAdminBackend::new(&url, Duration::from_secs(3), 60000, 1024 * 1024).unwrap();
    backend
        .verify_api_with_missing_key(&tenant_id, &format!("{key}-api-probe-missing"))
        .await
        .unwrap();
    assert!(backend.api_verified());
    for mutation in [
        BackendMutation::SoftPin { ttl_ms: 2000 },
        BackendMutation::SoftUnpin,
    ] {
        let reply = backend
            .mutate(BackendBatch {
                tenant_id: tenant_id.clone(),
                keys: vec![key.clone()],
                mutation,
            })
            .await;
        assert!(
            matches!(reply, BackendReply::Quiescent(ref rows) if rows.len() == 1 && rows[0].key == key && rows[0].error_code == 0),
            "{reply:?}"
        );
    }
    let reply = backend
        .mutate(BackendBatch {
            tenant_id: tenant_id.clone(),
            keys: vec![format!("{key}-missing")],
            mutation: BackendMutation::SoftPin { ttl_ms: 1000 },
        })
        .await;
    assert!(
        matches!(reply, BackendReply::Quiescent(ref rows) if rows.len() == 1 && rows[0].error_code == -704),
        "{reply:?}"
    );
    struct LiveResolver {
        key: String,
        domain: ProtectionDomain,
    }
    impl StorageTargetResolver for LiveResolver {
        fn resolve<'a>(
            &'a self,
            _: &'a PrefixTarget,
        ) -> BoxFuture<'a, Result<ResolvedStorageTarget, ProtectionError>> {
            Box::pin(async move {
                Ok(ResolvedStorageTarget {
                    domain: self.domain.clone(),
                    scopes: vec![ResolvedStorageScope {
                        component: "full".into(),
                        start_token: 0,
                        end_token: 4,
                        dependency_start: 0,
                        keys: vec![self.key.clone()],
                    }],
                })
            })
        }
    }
    let mut c = config();
    c.domain.tenant_id = tenant_id;
    // The private test object is deliberately isolated from live Engine keys.
    let owner = BackendProtectionOwner::spawn(
        c.clone(),
        Arc::new(backend),
        Arc::new(LiveResolver {
            key,
            domain: c.domain,
        }),
    )
    .unwrap();
    let a = request("live-a", 1, 10.0);
    let b = request("live-b", 1, 20.0);
    owner.submit(a.clone()).await.unwrap();
    owner.submit(b.clone()).await.unwrap();
    for _ in 0..100 {
        if view(&owner, &b).await.converged {
            break;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    assert!(view(&owner, &b).await.converged);
    owner
        .submit(update(&a, ClaimMode::Release, 2, None))
        .await
        .unwrap();
    assert_eq!(view(&owner, &b).await.phase, ClaimPhase::Active);
    owner
        .submit(update(&b, ClaimMode::Release, 2, None))
        .await
        .unwrap();
    for _ in 0..100 {
        if view(&owner, &b).await.backend_quiescent {
            break;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    assert!(view(&owner, &b).await.backend_quiescent);
}

#[tokio::test(start_paused = true)]
async fn reordered_release_before_acquire_keeps_revision_fence_without_io() {
    let (owner, mut calls) = harness(config());
    let a = request("a", 1, 20.0);
    let released = owner
        .submit(update(&a, ClaimMode::Release, 2, None))
        .await
        .unwrap();
    assert_eq!(released.phase, ClaimPhase::Released);
    assert!(released.backend_quiescent);
    assert_eq!(
        owner.submit(a.clone()).await.unwrap_err(),
        ProtectionError::StaleRevision
    );
    assert_eq!(
        owner
            .submit(update(&a, ClaimMode::Renew, 3, Some(20.0)))
            .await
            .unwrap_err(),
        ProtectionError::TerminalClaim
    );
    assert!(calls.try_recv().is_err());
}
#[tokio::test(start_paused = true)]
async fn missing_ancestor_prevents_successful_later_page_from_being_ready() {
    let (owner, mut calls) = harness(config());
    let a = request("a", 2, 20.0);
    owner.submit(a.clone()).await.unwrap();
    let call = calls.recv().await.unwrap();
    let rows = call
        .batch
        .keys
        .iter()
        .map(|key| KeyResult {
            key: key.clone(),
            error_code: if key == "digest-0-v" { -704 } else { 0 },
        })
        .collect();
    call.reply.send(BackendReply::Quiescent(rows)).unwrap();
    let v = view(&owner, &a).await;
    assert_eq!(v.phase, ClaimPhase::Desired);
    assert!(v
        .scopes
        .iter()
        .all(|s| s.missing && s.confirmed_remaining_ms.is_none()));
    assert_eq!(v.scopes[1].backend_error_codes, vec![-704]);
}
#[tokio::test]
async fn unverified_admin_api_cannot_advertise_soft_pin() {
    let backend =
        MooncakeAdminBackend::new("http://127.0.0.1:1", Duration::from_secs(1), 60000, 4096)
            .unwrap();
    assert!(!backend.api_verified());
    let result = BackendProtectionOwner::spawn(config(), Arc::new(backend), Arc::new(Resolver));
    assert!(matches!(
        result,
        Err(ProtectionError::UnsupportedProtection)
    ));
}
#[test]
fn trusted_engine_description_copies_exact_keys_and_checks_both_namespaces() {
    let value = serde_json::json!({"description_version":"1.0", "namespace":"weights", "compatibility":{"cache_salt":null,"extra_key":null,"layout_id":"full-kv","is_multimodal":false}, "key_salt_encoding":"sglang.kv.weight.v1", "page_size":4, "location":{"tier":"storage","engine_id":null,"backend":"mooncake","backend_namespace":"sglang.storage.v1:configured-physical-layout"}, "existence":"unknown", "pages":[{"start_token":0,"end_token":4,"component":"full","storage_digest":"a".repeat(64),"keys":["tenant-config-digest_k_7","tenant-config-digest_v_7"],"object_bytes":[4096,4096]}]});
    let description: EngineStorageTargets = serde_json::from_value(value).unwrap();
    let mut d = domain();
    d.storage_namespace = "sglang.storage.v1:configured-physical-layout".into();
    let resolved = description
        .clone()
        .into_resolved(&prefix(1), &d, "sglang.kv.weight.v1")
        .unwrap();
    assert_eq!(
        resolved.scopes[0].keys,
        vec!["tenant-config-digest_k_7", "tenant-config-digest_v_7"]
    );
    assert!(description
        .clone()
        .into_resolved(&prefix(1), &domain(), "sglang.kv.weight.v1")
        .is_err());
    assert!(description
        .into_resolved(&prefix(1), &d, "owner_assertion")
        .is_err());
}
