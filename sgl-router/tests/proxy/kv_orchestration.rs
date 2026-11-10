//! Real Router + real HTTP Engine mocks; no live GPU or external services.
use crate::common::cache_aware_fixture::{config, MODEL};
use axum::{
    extract::State,
    routing::{get, post},
    Json, Router,
};
use serde_json::{json, Value};
use sgl_router::{
    config::{AffinityConfig, CachePrefixProvider},
    discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec},
    orchestration::OrchestrationConfig,
    policies::factory::build_registry,
    policies_reorg::factory::build_resolver,
    proxy::Proxy,
    server::{
        app::build_router,
        app_context::{AppContext, ChatRouting},
    },
    state::kv_events::{
        compute_block_hashes, HashTree, KvEventIndex, KvWorkerId, RadixTreePrefixProvider,
    },
    tokenizer::TokenizerRegistry,
    workers::WorkerRegistry,
};
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
    time::Duration,
};
use tokio::task::JoinHandle;

#[derive(Default)]
struct EngineState {
    operations: BTreeMap<String, Value>,
    commands: Vec<Value>,
    inference: Vec<Value>,
    resident: u64,
    partial: Option<u64>,
    running: bool,
    cancelled: bool,
    claims: BTreeMap<String, Value>,
    epoch: String,
    fail_residency: bool,
    lose_submit_once: bool,
    feedback: bool,
    hide_feedback: bool,
    fail_feedback: bool,
    feedback_epoch: String,
    page_epoch_override: Option<String>,
    cache_epoch: String,
    cache_version: u64,
    native_generation: u64,
    feedback_sequence: u64,
    feedback_gap: bool,
    feedback_events: Vec<Value>,
    residency_override: Option<Value>,
    delayed_query: Option<Arc<DelayedQuery>>,
}
#[derive(Default)]
struct DelayedQuery {
    started: tokio::sync::Notify,
    release: tokio::sync::Notify,
}
fn cache_watermark(state: &EngineState) -> Value {
    json!({"source_id":"cache:mock","source_epoch":state.cache_epoch,"generation":state.native_generation,"version":state.cache_version})
}
fn feedback_event(state: &mut EngineState, kind: &str, payload: Value) {
    state.feedback_sequence += 1;
    let mark = cache_watermark(state);
    state.feedback_events.push(json!({"sequence":state.feedback_sequence,"kind":kind,"caller_id":null,"cache_watermark":mark,"payload":payload}));
}
struct Engine {
    url: String,
    state: Arc<Mutex<EngineState>>,
    server: JoinHandle<()>,
}
impl Drop for Engine {
    fn drop(&mut self) {
        self.server.abort();
    }
}
async fn capabilities(State(state): State<Arc<Mutex<EngineState>>>) -> Json<Value> {
    let s = state.lock().unwrap();
    let mut reply = json!({"success":true,"data":{"engine_id":"mock-engine","engine_epoch":s.epoch,"namespace":"weights-v1",
        "cache_salt_encoding":"owner_assertion","executor":{"cache_generation":1},"public_protocol":{"actions":{"sglang.prefetch":["1.0"],"sglang.demote":["1.0"],"sglang.pin":["1.0"]},
        "paths":{"sglang.prefetch":[["storage","host"],["host","gpu"]],"sglang.demote":[["host","storage"]]},
        "storage_backend":"mooncake","backend_namespace":"pool-a","storage_targets":true,"prefix":{"namespace":"weights-v1","layout_id":"full:p1","page_size":1,"components":["full"],"hash_scheme":"sglang.sha256.u32le.i64.v1"}}}});
    if s.feedback && !s.hide_feedback {
        reply["data"]["feedback"] = json!({"query":"events","version":"1.0","source_id":"control:mock","source_epoch":s.feedback_epoch,
            "recovery":"query_known_scopes","cache_watermark":cache_watermark(&s)});
    }
    Json(reply)
}
async fn command(
    State(state): State<Arc<Mutex<EngineState>>>,
    Json(body): Json<Value>,
) -> (axum::http::StatusCode, Json<Value>) {
    let delayed = if body["kind"] == "residency" {
        state.lock().unwrap().delayed_query.take()
    } else {
        None
    };
    let response = command_now(state, body);
    if let Some(delayed) = delayed {
        delayed.started.notify_one();
        delayed.release.notified().await;
    }
    response
}
fn command_now(
    state: Arc<Mutex<EngineState>>,
    body: Value,
) -> (axum::http::StatusCode, Json<Value>) {
    use axum::http::StatusCode;
    let mut s = state.lock().unwrap();
    s.commands.push(body.clone());
    if body["kind"] == "events" {
        if s.fail_feedback {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({"success":false,"error":{"code":"unavailable"}})),
            );
        }
        let after = body["cursor"]["sequence"].as_u64().unwrap_or(0);
        let limit = body["limit"].as_u64().unwrap_or(256) as usize;
        let items: Vec<_> = s
            .feedback_events
            .iter()
            .filter(|event| event["sequence"].as_u64().unwrap() > after)
            .take(limit)
            .cloned()
            .collect();
        let seq = if items.len() == limit {
            items.last().unwrap()["sequence"].as_u64().unwrap()
        } else {
            s.feedback_sequence
        };
        let gap = std::mem::take(&mut s.feedback_gap);
        return (
            StatusCode::OK,
            Json(
                json!({"success":true,"data":{"source_id":"control:mock","source_epoch":s.page_epoch_override.as_ref().unwrap_or(&s.feedback_epoch),
            "engine_id":"mock-engine","engine_epoch":s.epoch,"namespace":"weights-v1","items":items,
            "next_cursor":{"source_id":"control:mock","source_epoch":s.page_epoch_override.as_ref().unwrap_or(&s.feedback_epoch),"sequence":seq},"gap":gap,
            "earliest_sequence":s.feedback_events.first().map_or(s.feedback_sequence+1,|event|event["sequence"].as_u64().unwrap()),
            "latest_sequence":s.feedback_sequence,"cache_watermark":cache_watermark(&s),"recovery":"query_known_scopes"}}),
            ),
        );
    }
    if body["kind"] == "claim" {
        let claim = s
            .claims
            .get(body["claim_id"].as_str().unwrap())
            .cloned()
            .unwrap_or(Value::Null);
        return (StatusCode::OK, Json(json!({"success":true,"data":claim})));
    }
    if body["kind"] == "residency" {
        if s.fail_residency {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({"success":false,"error":{"code":"unavailable"}})),
            );
        }
        let n = body["target"]["token_ids"].as_array().unwrap().len() as u64;
        let end = s.resident.min(n);
        let items = if end > 0 {
            vec![
                json!({"coverage_kind":"tokens","descriptor":{"component":"full","start_token":0,"end_token":end,"dependency_start":0},"location":body["location"]}),
            ]
        } else {
            vec![]
        };
        let mut reply = json!({"success":true,"data":{"namespace":"weights-v1","engine_epoch":s.epoch,"cache_generation":1,"observed_at":1,"items":items,"missing":[]}});
        if s.feedback {
            reply["data"]["cache_watermark"] = s
                .residency_override
                .clone()
                .unwrap_or_else(|| cache_watermark(&s));
            reply["data"]["coverage_semantics"] = json!("requested_scope_only");
        }
        return (StatusCode::OK, Json(reply));
    }
    if body["kind"] == "submit" {
        let id = body["envelope"]["message_id"].as_str().unwrap().to_string();
        let action = &body["envelope"]["actions"][0];
        let count = action["payload"]["target"]["token_ids"]
            .as_array()
            .unwrap()
            .len() as u64;
        let result = json!({"ref":{"context":body["context"],"message_id":id,"action_id":"transfer","operation_id":format!("op-{id}")},
            "sequence":1,"update":{"status":if s.running{"running"}else{"completed"},"quiescent":!s.running},"public_result":{"state_version":99_999}});
        if action["action_type"] == "sglang.pin" {
            let payload = &action["payload"];
            let epoch = s.epoch.clone();
            s.claims.insert(payload["claim_id"].as_str().unwrap().into(),json!({"owner_epoch":epoch,"revision":payload["revision"],
                "state":if payload["mode"]=="release"{"released"}else{"active"},"effective_protection":"eviction_exclusion","actual_scope":[{"start_token":0,"end_token":count}]}));
        }
        s.operations.entry(id).or_insert(result);
        if !s.running && action["action_type"] == "sglang.prefetch" {
            s.resident = s.partial.unwrap_or(count);
        }
        if s.lose_submit_once {
            s.lose_submit_once = false;
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({"success":false,"error":{"code":"lost-response"}})),
            );
        }
    }
    if body["kind"] == "cancel" {
        s.cancelled = true;
        let id = body["ref"]["message_id"].as_str().unwrap();
        let result = s.operations.get_mut(id).unwrap();
        result["sequence"] = json!(2);
        result["update"] = json!({"status":"cancelled","quiescent":true});
    }
    let id = body["message_id"]
        .as_str()
        .or_else(|| body["envelope"]["message_id"].as_str())
        .or_else(|| body["ref"]["message_id"].as_str())
        .unwrap_or("");
    match s.operations.get(id) {
        Some(r) => (
            StatusCode::OK,
            Json(json!({"success":true,"data":{"results":[r]}})),
        ),
        None => (
            StatusCode::NOT_FOUND,
            Json(json!({"success":false,"error":{"code":"missing"}})),
        ),
    }
}
async fn inference(
    State(state): State<Arc<Mutex<EngineState>>>,
    Json(body): Json<Value>,
) -> Json<Value> {
    state.lock().unwrap().inference.push(body);
    Json(
        json!({"id":"mock","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":4,"completion_tokens":1,"total_tokens":5}}),
    )
}
impl Engine {
    async fn start() -> Self {
        let state = Arc::new(Mutex::new(EngineState {
            epoch: "epoch-1".into(),
            feedback_epoch: "journal-1".into(),
            cache_epoch: "cache-1".into(),
            native_generation: 1,
            ..Default::default()
        }));
        let app = Router::new()
            .route("/v1/kv/capabilities", get(capabilities))
            .route("/v1/kv/control", post(command))
            .route("/generate", post(inference))
            .route("/v1/chat/completions", post(inference))
            .with_state(state.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        Self { url, state, server }
    }
}
struct TestRouter {
    tree: Arc<HashTree>,
    url: String,
    server: JoinHandle<()>,
    registry: Arc<WorkerRegistry>,
}
impl Drop for TestRouter {
    fn drop(&mut self) {
        self.server.abort();
    }
}
fn orchestration() -> OrchestrationConfig {
    serde_json::from_value(json!({"caller_id":"trusted-router","control_bearer":"test-control",
    "queue_capacity":32,"max_operations":64,"max_workers":4,"max_engine_bytes":1048576,"rpc_timeout_ms":1000,"poll_ms":50,"operation_ttl_ms":10000,
    "limits":{"max_sessions":16,"max_branches":32,"max_history_revisions":64,"max_requests":64,"max_content_nodes":1024,"max_token_material_bytes":1048576,
        "max_tombstones":32,"max_claims":32,"max_intents":64,"max_identifier_bytes":128,"max_components":8}})).unwrap()
}
async fn router(engines: &[&Engine], reorg: bool, enabled: bool) -> TestRouter {
    router_with(engines, reorg, enabled.then(orchestration)).await
}
async fn router_with(
    engines: &[&Engine],
    reorg: bool,
    orchestration_config: Option<OrchestrationConfig>,
) -> TestRouter {
    let mut cfg = config();
    cfg.server.orchestration = orchestration_config;
    cfg.model.cache_aware.as_mut().unwrap().prefix_provider = CachePrefixProvider::RadixTree;
    cfg.model.affinity = Some(AffinityConfig {
        cache_affinity_min_matched_tokens: Some(0),
        cache_candidate_min_workers: 1,
        cache_candidate_ratio: 1.0,
        cache_candidate_max_workers: 1,
        ..Default::default()
    });
    let registry = Arc::new(WorkerRegistry::default());
    for (i, e) in engines.iter().enumerate() {
        registry
            .add(WorkerSpec {
                id: WorkerId(format!("engine-{i}")),
                url: e.url.clone(),
                model_ids: vec![ModelId(MODEL.into())],
                mode: WorkerMode::Plain,
                ..Default::default()
            })
            .unwrap();
    }
    let state = KvEventIndex::new();
    let tree = state.tree();
    let oracle = state.block_size_oracle();
    oracle.try_set(1).unwrap();
    let policies = build_registry(&cfg, tree.clone(), oracle.clone()).unwrap();
    let mut ctx = AppContext::new(
        cfg.clone(),
        Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap()),
        Arc::new(Proxy::new(Duration::from_secs(3)).unwrap()),
        registry.clone(),
        Arc::new(policies),
    );
    if reorg {
        let (resolver, _) = build_resolver(&cfg.model, &state, None).unwrap();
        ctx.chat_routing = ChatRouting::Reorg([(ModelId(MODEL.into()), resolver)].into());
    }
    ctx.radix_tree_prefix_provider = Some(RadixTreePrefixProvider::new(tree.clone(), oracle));
    let app = build_router(Arc::new(ctx));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    TestRouter {
        tree,
        url,
        server,
        registry,
    }
}
fn task(id: &str) -> Value {
    json!({"session_id":id,"incarnation":"task-1"})
}
fn prefix(tokens: Vec<u32>) -> Value {
    json!({"kind":"prefix","compatibility":{"namespace":"weights-v1","cache_salt":null,"layout_id":"full:p1","page_size":1,"components":["full"],"hash_scheme":"sglang.sha256.u32le.i64.v1"},"start_token":0,"end_token":tokens.len(),"token_ids":tokens})
}
async fn api(router: &TestRouter, body: Value) -> Value {
    let response = reqwest::Client::new()
        .post(format!("{}/v1/kv/control", router.url))
        .bearer_auth("test-control")
        .json(&body)
        .send()
        .await
        .unwrap();
    let result: Value = response.json().await.unwrap();
    assert_eq!(result["success"], true, "{result}");
    result["data"].clone()
}
async fn event(router: &TestRouter, id: &str, event: &str, revision: u64, event_id: &str) -> Value {
    api(router,json!({"kind":"session_event","event":event,"task":task(id),"revision":revision,"event_id":event_id})).await
}
async fn start(router: &TestRouter, id: &str) {
    event(router, id, "task_started", 0, &format!("start-{id}")).await;
}
async fn submit(
    router: &TestRouter,
    id: &str,
    message: &str,
    target: Value,
    worker: &Engine,
) -> Value {
    api(router,json!({"kind":"submit","task":task(id),"message_id":message,"action":"sglang.prefetch","target":target,"worker_url":worker.url,
        "source":{"tier":"storage"},"destination":{"tier":"host"}})).await
}
async fn query(router: &TestRouter, id: &str, message: &str) -> Value {
    api(
        router,
        json!({"kind":"query","task":task(id),"message_id":message}),
    )
    .await
}
async fn ready(router: &TestRouter, id: &str, message: &str) -> Value {
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            let view = query(router, id, message).await;
            if view["operations"][0]["ready_scope"].is_object() {
                break view;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap()
}

#[tokio::test]
async fn lifecycle_prefetch_reports_readiness_without_overriding_worker_selection() {
    for reorg in [false, true] {
        let a = Engine::start().await;
        let b = Engine::start().await;
        let router = router(&[&a, &b], reorg, true).await;
        start(&router, "chat").await;
        let body = json!({"model":MODEL,"messages":[{"role":"user","content":"prepared route"}]});
        let tokenizers = TokenizerRegistry::load_from_config(&config()).unwrap();
        let tokens =
            sgl_router::policies::request_tokens_for(&tokenizers, &ModelId(MODEL.into()), &body)
                .unwrap()
                .ids;
        submit(&router, "chat", "prepare", prefix(tokens.clone()), &b).await;
        let view = ready(&router, "chat", "prepare").await;
        assert_eq!(
            view["operations"][0]["ready_scope"]["end_token"],
            tokens.len()
        );
        assert_eq!(view["operations"][0]["owner_epoch"], "epoch-1");
        assert!(router
            .registry
            .all()
            .iter()
            .all(|w| w.router_inflight_load() == 0));
        let hashes = compute_block_hashes(&tokens[..tokens.len() - 1], 1);
        router
            .tree
            .insert(&KvWorkerId::new(a.url.clone(), 0), None, &hashes);
        let response = reqwest::Client::new()
            .post(format!("{}/v1/chat/completions", router.url))
            .json(&body)
            .send()
            .await
            .unwrap();
        assert!(response.status().is_success());
        assert_eq!(response.headers()["x-sgl-worker-url"], a.url);
        assert_eq!(a.state.lock().unwrap().inference.len(), 1);
        assert!(b.state.lock().unwrap().inference.is_empty());
        let commands = &b.state.lock().unwrap().commands;
        assert!(commands
            .iter()
            .any(|c| c["kind"] == "residency" && c["target"]["kind"] == "tokens"));
    }
}

#[tokio::test]
async fn shared_interests_cleanup_partial_and_unavailable_queries_preserve_uncertainty() {
    let engine = Engine::start().await;
    let router = router(&[&engine], false, true).await;
    start(&router, "a").await;
    start(&router, "b").await;
    engine.state.lock().unwrap().partial = Some(2);
    let a = submit(&router, "a", "prepare", prefix(vec![1, 2, 3, 4]), &engine).await;
    let b = submit(&router, "b", "prepare", prefix(vec![1, 2, 3, 4]), &engine).await;
    assert_eq!(
        a["operations"][0]["child_id"],
        b["operations"][0]["child_id"]
    );
    let view = ready(&router, "b", "prepare").await;
    assert_eq!(view["operations"][0]["ready_scope"]["end_token"], 2);
    event(&router, "a", "task_finished", 1, "finish-a").await;
    assert_eq!(
        query(&router, "b", "prepare").await["operations"][0]["cancel_requested"],
        false
    );
    engine.state.lock().unwrap().fail_residency = true;
    let view = query(&router, "b", "prepare").await;
    assert!(view["operations"][0]["ready_scope"].is_null());
    assert!(view["operations"][0]["error"].is_string());
    assert_eq!(
        engine
            .state
            .lock()
            .unwrap()
            .commands
            .iter()
            .filter(|c| c["kind"] == "submit")
            .count(),
        1
    );
}

#[tokio::test]
async fn rendered_generate_association_strips_metadata_freezes_branches_and_task_lifetime_is_explicit(
) {
    let engine = Engine::start().await;
    let router = router(&[&engine], false, true).await;
    let meta = json!({"session_id":"conversation","incarnation":"task-1","branch_id":"one","revision":1,"branch_revision":1,"request_id":"r","request_incarnation":"binding-1"});
    let response = reqwest::Client::new()
        .post(format!("{}/generate", router.url))
        .json(&json!({"input_ids":[1,2,3,4],"kv_context":meta}))
        .send()
        .await
        .unwrap();
    assert!(response.status().is_success());
    assert!(engine.state.lock().unwrap().inference[0]
        .get("kv_context")
        .is_none());
    let view=api(&router,json!({"kind":"submit","task":task("conversation"),"message_id":"freeze","action":"sglang.prefetch","target":{"kind":"session"},"source":{"tier":"storage"},"destination":{"tier":"host"}})).await;
    assert_eq!(view["known_scope"].as_array().unwrap().len(), 1);
    assert_eq!(view["snapshot_revision"], 1);
    let mut meta = meta;
    meta["branch_id"] = json!("two");
    meta["revision"] = json!(2);
    meta["request_incarnation"] = json!("binding-2");
    reqwest::Client::new()
        .post(format!("{}/generate", router.url))
        .json(&json!({"input_ids":[1,2,3,5],"kv_context":meta}))
        .send()
        .await
        .unwrap();
    let view = query(&router, "conversation", "freeze").await;
    assert_eq!(view["known_scope"].as_array().unwrap().len(), 1);
    assert_eq!(view["snapshot_revision"], 1);
    let live=api(&router,json!({"kind":"submit","task":task("conversation"),"message_id":"both","action":"sglang.prefetch","target":{"kind":"session"},"source":{"tier":"storage"},"destination":{"tier":"host"}})).await;
    assert_eq!(live["known_scope"].as_array().unwrap().len(), 2);
}

#[tokio::test]
async fn disabled_feature_and_control_auth_keep_ordinary_requests_working() {
    let engine = Engine::start().await;
    let disabled = router(&[&engine], false, false).await;
    let client = reqwest::Client::new();
    assert_eq!(
        client
            .post(format!("{}/v1/kv/control", disabled.url))
            .json(&json!({"kind":"capabilities"}))
            .send()
            .await
            .unwrap()
            .status(),
        404
    );
    assert!(client
        .post(format!("{}/generate", disabled.url))
        .json(&json!({"input_ids":[1,2,3,4]}))
        .send()
        .await
        .unwrap()
        .status()
        .is_success());
    assert!(engine.state.lock().unwrap().commands.is_empty());
    let enabled = router(&[&engine], false, true).await;
    assert_eq!(
        client
            .post(format!("{}/v1/kv/control", enabled.url))
            .json(&json!({"kind":"capabilities"}))
            .send()
            .await
            .unwrap()
            .status(),
        401
    );
    let response=client.post(format!("{}/generate",enabled.url)).json(&json!({"input_ids":[1,2],"session_id":"mismatch","kv_context":{"session_id":"x","incarnation":"1","branch_id":"b","revision":1,"branch_revision":1,"request_id":"r","request_incarnation":"1"}})).send().await.unwrap();
    assert_eq!(response.status(), 400);
}

#[tokio::test]
async fn lost_submit_response_reuses_child_identity_and_cancel_waits_for_drain() {
    let engine = Engine::start().await;
    let router = router(&[&engine], false, true).await;
    start(&router, "task").await;
    {
        let mut state = engine.state.lock().unwrap();
        state.running = true;
        state.lose_submit_once = true;
    }
    submit(
        &router,
        "task",
        "prepare",
        prefix(vec![1, 2, 3, 4]),
        &engine,
    )
    .await;
    // Query drives one lost-response attempt, cancellation must replay the same
    // child to discover its ref before issuing cancel; no invented quiescence.
    let _ = query(&router, "task", "prepare").await;
    let cancelled = api(
        &router,
        json!({"kind":"cancel","task":task("task"),"message_id":"prepare"}),
    )
    .await;
    assert_eq!(cancelled["quiescent"], false);
    let view = query(&router, "task", "prepare").await;
    assert_eq!(view["quiescent"], true);
    let state = engine.state.lock().unwrap();
    assert!(state.cancelled);
    assert_eq!(state.operations.len(), 1);
    let ids: BTreeMap<String, usize> = state
        .commands
        .iter()
        .filter(|c| c["kind"] == "submit")
        .map(|c| (c["envelope"]["message_id"].as_str().unwrap().into(), 1))
        .collect();
    assert_eq!(ids.len(), 1);
    assert!(view["operations"][0]["ready_scope"].is_null());
}

#[tokio::test]
async fn changed_formal_route_releases_only_its_preparation() {
    let a = Engine::start().await;
    let b = Engine::start().await;
    let router = router(&[&a, &b], false, true).await;
    start(&router, "task").await;
    start(&router, "other").await;
    b.state.lock().unwrap().running = true;
    submit(&router, "task", "prepare", prefix(vec![1, 2, 3, 4]), &b).await;
    submit(&router, "other", "prepare", prefix(vec![1, 2, 3, 4]), &b).await;
    // Keep a as the only formal candidate without changing the preparation job.
    router.registry.remove(&WorkerId("engine-1".into()));
    let meta = json!({"session_id":"task","incarnation":"task-1","branch_id":"main","revision":1,"branch_revision":1,"request_id":"formal","request_incarnation":"1"});
    let response = reqwest::Client::new()
        .post(format!("{}/generate", router.url))
        .json(&json!({"input_ids":[1,2,3,4],"kv_context":meta}))
        .send()
        .await
        .unwrap();
    assert_eq!(response.headers()["x-sgl-worker-url"], a.url);
    assert_eq!(query(&router, "task", "prepare").await["released"], true);
    assert_eq!(query(&router, "other", "prepare").await["released"], false);
    assert!(a
        .state
        .lock()
        .unwrap()
        .commands
        .iter()
        .all(|c| c["kind"] != "submit"));
}

#[tokio::test]
async fn l2_pin_queries_claim_state_and_task_finish_releases_without_engine_task_binding() {
    let engine = Engine::start().await;
    let router = router(&[&engine], false, true).await;
    start(&router, "task").await;
    let view=api(&router,json!({"kind":"submit","task":task("task"),"message_id":"keep","action":"sglang.pin","target":prefix(vec![1,2,3,4]),
        "location":{"tier":"host"},"mode":"acquire","claim_id":"keep","revision":1,"ttl_seconds":10,"required_protection":"eviction_exclusion"})).await;
    assert_eq!(view["operations"].as_array().unwrap().len(), 1);
    let view = query(&router, "task", "keep").await;
    assert_eq!(view["operations"][0]["claim"]["state"], "active");
    assert!(view["operations"][0]["ready_scope"].is_null());
    event(&router, "task", "task_finished", 1, "finish").await;
    tokio::time::timeout(Duration::from_secs(3), async {
        loop {
            let view = query(&router, "task", "keep").await;
            if view["operations"][0]["claim"]["state"] == "released" {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    let state = engine.state.lock().unwrap();
    assert_eq!(state.claims.len(), 1);
    assert!(state
        .commands
        .iter()
        .filter(|c| c["kind"] == "submit")
        .all(|c| c["envelope"]["actions"][0]["payload"]
            .get("owner_task")
            .is_none()));
}

#[tokio::test]
async fn explicit_prepare_freezes_scope_and_progresses_without_another_event() {
    let engine = Engine::start().await;
    let router = router(&[&engine], false, true).await;
    start(&router, "task").await;
    api(&router,json!({"kind":"session_event","task":task("task"),"event":"request_observed","event_id":"observed","revision":1,
        "branch_id":"main","branch_revision":1,"request_id":"r","request_incarnation":"one","input_ids":[1,2,3,4],"worker_url":engine.url})).await;
    let demote=api(&router,json!({"kind":"session_event","task":task("task"),"event":"tool_wait","event_id":"wait","revision":2,
        "target":{"kind":"session"},"pressure":true,"worker_url":engine.url})).await;
    assert_eq!(demote["operations"][0]["action"], "sglang.demote");
    query(&router, "task", "wait").await;
    assert!(engine.state.lock().unwrap().commands.iter().any(|c| {
        let a = &c["envelope"]["actions"][0];
        a["action_type"] == "sglang.demote"
            && a["payload"]["source"]["tier"] == "host"
            && a["payload"]["destination"]["tier"] == "storage"
    }));
    let resume=api(&router,json!({"kind":"session_event","task":task("task"),"event":"prepare","event_id":"resume","revision":3,
        "target":{"kind":"session"},"worker_url":engine.url})).await;
    assert_eq!(resume["known_scope"].as_array().unwrap().len(), 1);
    api(&router,json!({"kind":"session_event","task":task("task"),"event":"request_observed","event_id":"other","revision":4,
        "branch_id":"other","branch_revision":1,"request_id":"r","request_incarnation":"two","input_ids":[1,2,3,5],"worker_url":engine.url})).await;
    let frozen = query(&router, "task", "resume").await;
    assert_eq!(frozen["snapshot_revision"], resume["snapshot_revision"]);
    assert_eq!(frozen["known_scope"].as_array().unwrap().len(), 1);
    let view = ready(&router, "task", "resume").await;
    assert_eq!(view["operations"][0]["ready_scope"]["end_token"], 4);
    let stale = reqwest::Client::new()
        .post(format!("{}/v1/kv/session/events", router.url))
        .bearer_auth("test-control")
        .json(&json!({"task":task("task"),"event":"resume","event_id":"late","revision":2}))
        .send()
        .await
        .unwrap();
    assert_eq!(stale.status(), 400);
}

#[tokio::test]
async fn hostile_time_revision_and_token_bounds_do_not_terminate_the_control_actor() {
    let engine = Engine::start().await;
    let router = router(&[&engine], false, true).await;
    let client = reqwest::Client::new();
    for body in [
        json!({"kind":"session_event","task":task("task"),"event":"prepare","event_id":"huge-time","revision":1,"ttl_ms":u64::MAX}),
        json!({"kind":"session_event","task":task("task"),"event":"resume","event_id":"float-time","revision":1,"ttl_ms":1e200}),
        json!({"kind":"session_event","task":task("task"),"event":"task_finished","event_id":"huge-revision","revision":u64::MAX}),
        json!({"kind":"session_event","task":task("task"),"event":"request_observed","event_id":"huge-input","revision":1,"input_ids":vec![1;65_537]}),
    ] {
        let response = client
            .post(format!("{}/v1/kv/control", router.url))
            .bearer_auth("test-control")
            .json(&body)
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 400);
        let value: Value = response.json().await.unwrap();
        assert_eq!(value["error"]["code"], "invalid_payload");
    }
    assert!(engine.state.lock().unwrap().commands.is_empty());
    start(&router, "task").await;
    submit(&router, "task", "valid", prefix(vec![1, 2, 3, 4]), &engine).await;
    assert_eq!(
        ready(&router, "task", "valid").await["operations"][0]["ready_scope"]["end_token"],
        4
    );
}

#[tokio::test]
async fn enabled_without_preparation_preserves_pending_stakes_and_follow_accounting() {
    for reorg in [false, true] {
        let engine = Engine::start().await;
        let router = router(&[&engine], reorg, true).await;
        router.tree.pending().enable(Duration::from_secs(5));
        let ids = vec![1, 2, 3, 4];
        let client = reqwest::Client::new();
        for _ in 0..2 {
            assert!(client
                .post(format!("{}/generate", router.url))
                .json(&json!({"input_ids":ids}))
                .send()
                .await
                .unwrap()
                .status()
                .is_success());
        }
        let hashes = compute_block_hashes(&ids, 1);
        let depths = router.tree.pending().depths(&hashes);
        assert_eq!(depths.len(), 1, "routing path reorg={reorg}");
        assert_eq!(depths[0].0.as_ref(), engine.url);
        assert_eq!(depths[0].1, 4);
        assert_eq!(
            router.tree.pending().followed(),
            1,
            "routing path reorg={reorg}"
        );
        assert!(engine.state.lock().unwrap().commands.is_empty());
    }
}

async fn feedback_view(router: &TestRouter) -> Value {
    api(router, json!({"kind":"feedback"})).await
}
async fn feedback_engine() -> Engine {
    let engine = Engine::start().await;
    {
        let mut s = engine.state.lock().unwrap();
        s.feedback = true;
        s.cache_version = 1;
    }
    engine
}

#[tokio::test]
async fn newer_native_removal_fences_a_delayed_query_and_late_completed_operation() {
    let engine = feedback_engine().await;
    let router = router(&[&engine], false, true).await;
    start(&router, "task").await;
    submit(
        &router,
        "task",
        "prepare",
        prefix(vec![1, 2, 3, 4]),
        &engine,
    )
    .await;
    ready(&router, "task", "prepare").await;
    let delayed = Arc::new(DelayedQuery::default());
    engine.state.lock().unwrap().delayed_query = Some(delayed.clone());
    let url = router.url.clone();
    let query = tokio::spawn(async move {
        reqwest::Client::new()
            .post(format!("{url}/v1/kv/control"))
            .bearer_auth("test-control")
            .json(&json!({"kind":"query","task":task("task"),"message_id":"prepare"}))
            .send()
            .await
            .unwrap()
            .json::<Value>()
            .await
            .unwrap()
    });
    tokio::time::timeout(Duration::from_secs(2), delayed.started.notified())
        .await
        .unwrap();
    {
        let mut s = engine.state.lock().unwrap();
        let old = cache_watermark(&s);
        s.cache_version = 2;
        s.resident = 0;
        s.fail_residency = true;
        let mut completed = s.operations.values().next().unwrap().clone();
        completed["cache_watermark"] = old;
        completed["sequence"] = json!(9_999_999);
        feedback_event(&mut s, "operation", completed);
    }
    delayed.release.notify_one();
    let view = query.await.unwrap();
    assert!(view["data"]["operations"][0]["ready_scope"].is_null());
    assert!(
        feedback_view(&router).await["owners"][0]["rejected_scope_queries"]
            .as_u64()
            .unwrap()
            > 0
    );
    assert_eq!(
        feedback_view(&router).await["owners"][0]["cache_watermark"]["version"],
        2
    );
    let view = query_api(&router).await;
    assert!(view["operations"][0]["ready_scope"].is_null());
    // A genuinely current scope query can later recover that exact scope.
    {
        let mut state = engine.state.lock().unwrap();
        state.resident = 4;
        state.fail_residency = false;
    }
    let view = ready(&router, "task", "prepare").await;
    assert_eq!(view["operations"][0]["cache_watermark"]["version"], 2);
}
async fn query_api(router: &TestRouter) -> Value {
    query(router, "task", "prepare").await
}

#[tokio::test]
async fn feedback_gap_and_poll_failure_require_new_scope_queries_even_at_the_same_watermark() {
    let engine = feedback_engine().await;
    let router = router(&[&engine], false, true).await;
    start(&router, "task").await;
    submit(
        &router,
        "task",
        "prepare",
        prefix(vec![1, 2, 3, 4]),
        &engine,
    )
    .await;
    ready(&router, "task", "prepare").await;
    let delayed = Arc::new(DelayedQuery::default());
    engine.state.lock().unwrap().delayed_query = Some(delayed.clone());
    let url = router.url.clone();
    let pending = tokio::spawn(async move {
        reqwest::Client::new()
            .post(format!("{url}/v1/kv/control"))
            .bearer_auth("test-control")
            .json(&json!({"kind":"query","task":task("task"),"message_id":"prepare"}))
            .send()
            .await
            .unwrap()
            .json::<Value>()
            .await
            .unwrap()
    });
    tokio::time::timeout(Duration::from_secs(2), delayed.started.notified())
        .await
        .unwrap();
    {
        let mut state = engine.state.lock().unwrap();
        state.feedback_gap = true;
        state.fail_residency = true;
    }
    delayed.release.notify_one();
    let view = pending.await.unwrap();
    assert!(view["data"]["operations"][0]["ready_scope"].is_null());
    let state = feedback_view(&router).await;
    assert_eq!(state["global_snapshot_complete"], false);
    assert!(state["owners"][0]["last_recovery"]
        .as_str()
        .unwrap()
        .contains("journal_gap"));
    engine.state.lock().unwrap().fail_residency = false;
    ready(&router, "task", "prepare").await;
    engine.state.lock().unwrap().fail_feedback = true;
    let view = query_api(&router).await;
    assert!(view["operations"][0]["ready_scope"].is_null());
    assert!(feedback_view(&router).await["owners"][0]["error"].is_string());
    engine.state.lock().unwrap().fail_feedback = false;
    ready(&router, "task", "prepare").await;
}

#[tokio::test]
async fn feedback_rebind_requires_discovery_and_disappearance_cannot_downgrade_to_legacy() {
    let engine = feedback_engine().await;
    let router = router(&[&engine], false, true).await;
    start(&router, "task").await;
    submit(
        &router,
        "task",
        "prepare",
        prefix(vec![1, 2, 3, 4]),
        &engine,
    )
    .await;
    ready(&router, "task", "prepare").await;
    engine.state.lock().unwrap().hide_feedback = true;
    assert!(query_api(&router).await["operations"][0]["ready_scope"].is_null());
    assert!(feedback_view(&router).await["owners"][0]["error"].is_string());
    {
        let mut s = engine.state.lock().unwrap();
        s.hide_feedback = false;
        feedback_event(&mut s, "operation", json!({"historical":"old recorder"}));
        s.feedback_epoch = "journal-2".into();
        s.cache_epoch = "cache-2".into();
        s.page_epoch_override = Some("journal-1".into());
    }
    let view = query_api(&router).await;
    assert!(view["operations"][0]["ready_scope"].is_null());
    assert!(view["operations"][0]["error"]
        .as_str()
        .unwrap()
        .contains("source mismatch"));
    engine.state.lock().unwrap().page_epoch_override = None;
    let view = ready(&router, "task", "prepare").await;
    assert_eq!(
        view["operations"][0]["cache_watermark"]["source_epoch"],
        "cache-2"
    );
    assert_eq!(
        feedback_view(&router).await["owners"][0]["cursor"]["source_epoch"],
        "journal-2"
    );
    engine.state.lock().unwrap().epoch = "engine-2".into();
    assert!(query_api(&router).await["operations"][0]["ready_scope"].is_null());
}

#[tokio::test]
async fn feedback_diagnostics_are_bounded_and_cannot_resurrect_retired_tasks() {
    let engine = feedback_engine().await;
    let router = router(&[&engine], false, true).await;
    start(&router, "ended").await;
    event(&router, "ended", "task_finished", 1, "finish").await;
    {
        let mut s = engine.state.lock().unwrap();
        for i in 0..80 {
            feedback_event(
                &mut s,
                "request_use",
                json!({"request_id":format!("r{i}"),"association_state":"provided",
            "association":{"session_id":"ended","session_incarnation":"task-1","branch_id":"late","branch_revision":10},
            "cached_tokens":4,"host_tokens":null,"tier_accounting":"unknown","committed_prefix":{"token_ids":vec![1;1024]}}),
            );
        }
    }
    let view = tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            let view = feedback_view(&router).await;
            if view["owners"][0]["cursor"]["sequence"] == 80 {
                break view;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .unwrap();
    assert_eq!(view["facts"].as_array().unwrap().len(), 64);
    assert_eq!(view["dropped_diagnostics"], 16);
    assert!(!view.to_string().contains("committed_prefix"));
    assert!(view["facts"]
        .as_array()
        .unwrap()
        .iter()
        .all(|fact| fact["provenance"] == "unknown" && fact["task_authority"] == "none"));
    let response=reqwest::Client::new().post(format!("{}/v1/kv/control",router.url)).bearer_auth("test-control")
        .json(&json!({"kind":"session_event","task":task("ended"),"event":"task_started","event_id":"late-start","revision":11})).send().await.unwrap();
    assert_eq!(response.status(), 400);
    assert!(response
        .text()
        .await
        .unwrap()
        .contains("RetiredIncarnation"));
}

#[tokio::test]
async fn feedback_cache_generation_and_current_delayed_query_use_native_order_only() {
    let engine = feedback_engine().await;
    engine.state.lock().unwrap().cache_version = 10;
    let router = router(&[&engine], false, true).await;
    start(&router, "task").await;
    submit(
        &router,
        "task",
        "prepare",
        prefix(vec![1, 2, 3, 4]),
        &engine,
    )
    .await;
    ready(&router, "task", "prepare").await;
    {
        let mut state = engine.state.lock().unwrap();
        let mut old = cache_watermark(&state);
        old["version"] = json!((1u64 << 53) - 1);
        state.native_generation = 2;
        state.cache_version = 0;
        state.residency_override = Some(old);
    }
    // A huge old-generation version still cannot overtake generation 2.
    assert!(query_api(&router).await["operations"][0]["ready_scope"].is_null());
    engine.state.lock().unwrap().residency_override = None;
    let current = ready(&router, "task", "prepare").await;
    assert_eq!(current["operations"][0]["cache_watermark"]["generation"], 2);
    assert_eq!(current["operations"][0]["cache_watermark"]["version"], 0);
    let delayed = Arc::new(DelayedQuery::default());
    engine.state.lock().unwrap().delayed_query = Some(delayed.clone());
    let url = router.url.clone();
    let pending = tokio::spawn(async move {
        reqwest::Client::new()
            .post(format!("{url}/v1/kv/control"))
            .bearer_auth("test-control")
            .json(&json!({"kind":"query","task":task("task"),"message_id":"prepare"}))
            .send()
            .await
            .unwrap()
            .json::<Value>()
            .await
            .unwrap()
    });
    tokio::time::timeout(Duration::from_secs(2), delayed.started.notified())
        .await
        .unwrap();
    delayed.release.notify_one();
    let reply = pending.await.unwrap();
    assert_eq!(
        reply["data"]["operations"][0]["ready_scope"]["end_token"],
        4
    );
    let state = feedback_view(&router).await;
    assert_eq!(state["global_snapshot_complete"], false);
}
