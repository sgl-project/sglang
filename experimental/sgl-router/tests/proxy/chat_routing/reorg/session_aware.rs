// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[tokio::test]
async fn session_aware_reuses_custom_header_binding_after_load_changes() {
    use sgl_router::config::AffinityConfig;
    use sgl_router::policies_reorg::session_aware::SessionAwarePolicy;
    use sgl_router::state::load_monitor::engine_reported_load::EngineReportedLoadTable;
    use sgl_router::state::AffinityStore;
    use std::sync::atomic::Ordering;
    use std::time::Duration;

    let primary = MockWorker::start(vec![]).await;
    let other = MockWorker::start(vec![]).await;
    let store = AffinityStore::new(Duration::from_secs(60));
    let policy = Arc::new(SessionAwarePolicy::new(
        store.clone(),
        EngineReportedLoadTable::new(),
    ));
    let mut ctx = context(
        &[
            ("primary", Stage::Plain, &primary),
            ("other", Stage::Plain, &other),
        ],
        vec![Bucket::new(
            "session",
            BucketGroups::Plain(EngineGroup::new(policy)),
        )],
    );
    Arc::get_mut(&mut ctx).unwrap().config.model.affinity = Some(AffinityConfig {
        session_id_header: "x-test-session".into(),
        ..Default::default()
    });
    let primary_worker = ctx.registry.get(&WorkerId("primary".into())).unwrap();
    let other_worker = ctx.registry.get(&WorkerId("other".into())).unwrap();
    other_worker.active_requests.store(10, Ordering::Relaxed);
    let app = build_router(ctx);
    for _ in 0..2 {
        let mut req = request(body("hello"));
        req.headers_mut()
            .insert("x-test-session", "same-session".parse().unwrap());
        let response = app.clone().oneshot(req).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let _ = response.into_body().collect().await.unwrap();
        assert!(primary.captured.lock().unwrap().last_body.is_some());
        assert!(other.captured.lock().unwrap().last_body.is_none());
        primary_worker.active_requests.store(100, Ordering::Relaxed);
        other_worker.active_requests.store(0, Ordering::Relaxed);
    }
    assert_eq!(store.len(), 1);
}

#[tokio::test]
async fn rejected_session_binding_advances_buckets_without_reassignment_or_dispatch() {
    use sgl_router::config::AffinityConfig;
    use sgl_router::policies_reorg::session_aware::SessionAwarePolicy;
    use sgl_router::state::load_monitor::engine_reported_load::EngineReportedLoadTable;
    use sgl_router::state::AffinityStore;
    use std::time::Duration;

    let primary = MockWorker::start(vec![]).await;
    let backup = MockWorker::start(vec![]).await;
    let store = AffinityStore::new(Duration::from_secs(60));
    let table = EngineReportedLoadTable::new();
    let mut rejected = SessionAwarePolicy::new(store.clone(), table.clone());
    rejected.admission = Arc::new(RejectAll);
    let accepted = SessionAwarePolicy::new(store.clone(), table.clone());
    let mut ctx = context(
        &[
            ("primary", Stage::Plain, &primary),
            ("backup", Stage::Plain, &backup),
        ],
        vec![
            Bucket::new(
                "a-primary",
                BucketGroups::Plain(EngineGroup {
                    worker_ids: Some([WorkerId("primary".into())].into_iter().collect()),
                    policy: Arc::new(rejected),
                }),
            ),
            Bucket::new(
                "b-backup",
                BucketGroups::Plain(EngineGroup {
                    worker_ids: Some([WorkerId("backup".into())].into_iter().collect()),
                    policy: Arc::new(accepted),
                }),
            ),
        ],
    );
    Arc::get_mut(&mut ctx).unwrap().config.model.affinity = Some(AffinityConfig::default());
    let seed = SessionAwarePolicy::new(store.clone(), table);
    let model = ModelId("tiny".into());
    let pick_request = PickRequest {
        bucket: "a-primary",
        session_key: Some("same-session"),
        ..PickRequest::new(&model, Stage::Plain, 1)
    };
    let engine = ctx.registry.get(&WorkerId("primary".into())).unwrap();
    seed.pick(std::slice::from_ref(&engine), &pick_request)
        .await
        .unwrap();
    let mut req = request(body("hello"));
    req.headers_mut()
        .insert("x-session-id", "same-session".parse().unwrap());
    let response = build_router(ctx).oneshot(req).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let _ = response.into_body().collect().await.unwrap();
    assert!(primary.captured.lock().unwrap().last_body.is_none());
    assert!(backup.captured.lock().unwrap().last_body.is_some());
    assert_eq!(store.len(), 2);
    let retained = seed
        .pick(std::slice::from_ref(&engine), &pick_request)
        .await
        .unwrap();
    assert_eq!(retained.reason, "session_primary");
    assert_eq!(retained.engine.id.0, "primary");
}
