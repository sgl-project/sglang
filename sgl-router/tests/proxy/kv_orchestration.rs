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
    state::kv_events::{HashTree, KvEventIndex, RadixTreePrefixProvider},
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
    Json(
        json!({"success":true,"data":{"engine_id":"mock-engine","engine_epoch":s.epoch,"namespace":"weights-v1",
        "cache_salt_encoding":"owner_assertion","executor":{"cache_generation":1},"public_protocol":{"actions":{"sglang.prefetch":["1.0"],"sglang.demote":["1.0"],"sglang.pin":["1.0"]},
        "paths":{"sglang.prefetch":[["storage","host"],["host","gpu"]],"sglang.demote":[["gpu","storage"],["gpu","host"]]},
        "storage_backend":"mooncake","backend_namespace":"pool-a","storage_targets":true,"prefix":{"namespace":"weights-v1","layout_id":"full:p1","page_size":1,"components":["full"],"hash_scheme":"sglang.sha256.u32le.i64.v1"}}}}),
    )
}
async fn command(
    State(state): State<Arc<Mutex<EngineState>>>,
    Json(body): Json<Value>,
) -> (axum::http::StatusCode, Json<Value>) {
    use axum::http::StatusCode;
    let mut s = state.lock().unwrap();
    s.commands.push(body.clone());
    if body["kind"] == "claim" {
        let claim = s
            .claims
            .get(body["claim_id"].as_str().unwrap())
            .cloned()
            .unwrap_or(Value::Null);
        return (StatusCode::OK, Json(json!({"success":true,"data":claim})));
    }
    if body["kind"] == "storage_targets" {
        let tokens = body["target"]["token_ids"].as_array().unwrap();
        let pages:Vec<_>=tokens.iter().enumerate().map(|(i,token)|json!({"start_token":i,"end_token":i+1,"component":"full","storage_digest":"a".repeat(64),
            "keys":[format!("object-{token}-k"),format!("object-{token}-v")],"object_bytes":[16,16]})).collect();
        return (
            StatusCode::OK,
            Json(
                json!({"success":true,"data":{"description_version":"1.0","namespace":"weights-v1",
            "compatibility":{"cache_salt":null,"extra_key":null,"layout_id":"full:p1","is_multimodal":false},"key_salt_encoding":"owner_assertion",
            "page_size":1,"location":body["location"],"existence":"unknown","pages":pages}}),
            ),
        );
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
        return (
            StatusCode::OK,
            Json(
                json!({"success":true,"data":{"namespace":"weights-v1","engine_epoch":s.epoch,"cache_generation":1,"observed_at":1,"items":items,"missing":[]}}),
            ),
        );
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
    "initial_prefetch_ms_per_token":0.5,"lead_margin_ms":20,"max_resume_prefetch_ms":100,
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
async fn lifecycle_prefetch_scoped_query_influences_both_chat_paths_without_control_load_commits() {
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
        let response = reqwest::Client::new()
            .post(format!("{}/v1/chat/completions", router.url))
            .json(&body)
            .send()
            .await
            .unwrap();
        assert!(response.status().is_success());
        assert_eq!(response.headers()["x-sgl-worker-url"], b.url);
        assert!(a.state.lock().unwrap().inference.is_empty());
        assert_eq!(b.state.lock().unwrap().inference.len(), 1);
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
async fn changed_formal_route_releases_only_its_preparation_and_policy_refuses_late_work() {
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
    let policy=api(&router,json!({"kind":"session_event","task":task("task"),"event_id":"resume","event":"resume_expected","revision":2,
        "expected_wait_ms":1,"target":{"kind":"session"}})).await;
    assert_eq!(policy["state"], "on_demand_fallback");
    assert!(a
        .state
        .lock()
        .unwrap()
        .commands
        .iter()
        .all(|c| c["kind"] != "submit"));
}

#[tokio::test]
async fn real_backend_owner_resolves_actual_keys_and_preserves_other_tasks_soft_pin() {
    use sgl_router::orchestration::BackendOwnerConfig;
    type BackendState = Arc<Mutex<(BTreeMap<String, u64>, Vec<Value>)>>;
    async fn pin(State(state): State<BackendState>, Json(body): Json<Value>) -> Json<Value> {
        let mut state = state.lock().unwrap();
        state.1.push(body.clone());
        let rows: Vec<_> = body["keys"]
            .as_array()
            .unwrap()
            .iter()
            .map(|key| {
                let key = key.as_str().unwrap();
                let missing = key == "private-never-stored-probe";
                if !missing {
                    state.0.insert(key.into(), body["ttl_ms"].as_u64().unwrap());
                }
                json!({"key":key,"error_code":if missing{-704}else{0}})
            })
            .collect();
        Json(json!({"results":rows}))
    }
    async fn unpin(State(state): State<BackendState>, Json(body): Json<Value>) -> Json<Value> {
        let mut state = state.lock().unwrap();
        state.1.push(body.clone());
        let rows: Vec<_> = body["keys"]
            .as_array()
            .unwrap()
            .iter()
            .map(|key| {
                let key = key.as_str().unwrap();
                let missing = key == "private-never-stored-probe";
                state.0.remove(key);
                json!({"key":key,"error_code":if missing{-704}else{0}})
            })
            .collect();
        Json(json!({"results":rows}))
    }
    let backend: BackendState = Arc::default();
    let app = Router::new()
        .route("/api/v1/soft_pin", post(pin))
        .route("/api/v1/soft_unpin", post(unpin))
        .with_state(backend.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let admin_url = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    let mut config = orchestration();
    config.backend_owner=Some(serde_json::from_value::<BackendOwnerConfig>(json!({"admin_url":admin_url,
        "domain":{"cluster":"test-cluster","tenant_id":"tenant-a","model_namespace":"weights-v1","storage_namespace":"pool-a","layout_id":"full:p1"},
        "owner_epoch":"backend-owner-1","fresh_domain_assertion":"test owns fresh domain; no old writer",
        "private_missing_probe_key":"private-never-stored-probe","max_ttl_ms":60000,"max_objects":64,"max_objects_per_claim":16,
        "max_claims_per_object":16,"max_inflight_batches":2,"batch_size":16})).unwrap());
    let engine = Engine::start().await;
    let router = router_with(&[&engine], false, Some(config)).await;
    start(&router, "a").await;
    start(&router, "b").await;
    for id in ["a", "b"] {
        api(&router,json!({"kind":"submit","task":task(id),"message_id":"pin","action":"sglang.pin","target":prefix(vec![1,2]),
            "location":{"tier":"storage"},"mode":"acquire","claim_id":"keep","revision":1,"ttl_seconds":10,"required_protection":"backend_soft_pin"})).await;
        tokio::time::timeout(Duration::from_secs(3), async {
            loop {
                let view = query(&router, id, "pin").await;
                if view["claims"][0]["claim"]["phase"] == "active" {
                    assert_eq!(
                        view["claims"][0]["claim"]["effective_protection"],
                        "backend_soft_pin"
                    );
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
    }
    assert_eq!(backend.lock().unwrap().0.len(), 4);
    event(&router, "a", "task_finished", 1, "end-a").await;
    tokio::time::sleep(Duration::from_millis(100)).await;
    let b = query(&router, "b", "pin").await;
    assert_eq!(b["claims"][0]["claim"]["phase"], "active");
    assert_eq!(backend.lock().unwrap().0.len(), 4);
    event(&router, "b", "task_finished", 1, "end-b").await;
    tokio::time::timeout(Duration::from_secs(3), async {
        loop {
            query(&router, "b", "pin").await;
            if backend.lock().unwrap().0.is_empty() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    assert!(engine
        .state
        .lock()
        .unwrap()
        .commands
        .iter()
        .any(|c| c["kind"] == "storage_targets"));
    server.abort();
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
async fn lifecycle_schedule_freezes_scope_and_tool_done_advances_only_preparation() {
    let engine = Engine::start().await;
    let router = router(&[&engine], false, true).await;
    start(&router, "task").await;
    api(&router,json!({"kind":"session_event","task":task("task"),"event":"request_observed","event_id":"observed","revision":1,
        "branch_id":"main","branch_revision":1,"request_id":"r","request_incarnation":"one","input_ids":[1,2,3,4],"worker_url":engine.url})).await;
    let resume=api(&router,json!({"kind":"session_event","task":task("task"),"event":"resume_expected","event_id":"resume","revision":2,
        "target":{"kind":"session"},"expected_wait_ms":1000,"worker_url":engine.url})).await;
    assert_eq!(resume["known_scope"].as_array().unwrap().len(), 1);
    assert!(engine
        .state
        .lock()
        .unwrap()
        .commands
        .iter()
        .all(|c| c["kind"] != "submit"));
    api(&router,json!({"kind":"session_event","task":task("task"),"event":"request_observed","event_id":"other","revision":3,
        "branch_id":"other","branch_revision":1,"request_id":"r","request_incarnation":"two","input_ids":[1,2,3,5],"worker_url":engine.url})).await;
    let frozen = query(&router, "task", "resume").await;
    assert_eq!(frozen["snapshot_revision"], 1);
    assert_eq!(frozen["known_scope"].as_array().unwrap().len(), 1);
    api(&router,json!({"kind":"session_event","task":task("task"),"event":"tool_done","event_id":"done","revision":4,
        "target":{"kind":"branch","branch_id":"main"},"worker_url":engine.url})).await;
    let view = ready(&router, "task", "resume").await;
    assert_eq!(view["operations"][0]["ready_scope"]["end_token"], 4);
    let stale = reqwest::Client::new()
        .post(format!("{}/v1/kv/session/events", router.url))
        .bearer_auth("test-control")
        .json(&json!({"task":task("task"),"event":"tool_done","event_id":"late","revision":2}))
        .send()
        .await
        .unwrap();
    assert_eq!(stale.status(), 400);
}

#[tokio::test]
async fn backend_owner_restart_stays_unknown_without_a_fresh_domain_assertion() {
    let engine = Engine::start().await;
    let mut config = orchestration();
    config.backend_owner=Some(serde_json::from_value(json!({"admin_url":"http://127.0.0.1:1",
        "domain":{"cluster":"c","tenant_id":"tenant-a","model_namespace":"weights-v1","storage_namespace":"pool-a","layout_id":"full:p1"},
        "owner_epoch":"restarted-owner","fresh_domain_assertion":null,"private_missing_probe_key":"unused-owned-probe",
        "max_ttl_ms":60000,"max_objects":64,"max_objects_per_claim":16,"max_claims_per_object":8,"max_inflight_batches":2,"batch_size":16})).unwrap());
    let router = router_with(&[&engine], false, Some(config)).await;
    start(&router, "task").await;
    let caps = api(&router, json!({"kind":"capabilities"})).await;
    assert_eq!(caps["backend_protection"]["recovery_unknown"], true);
    assert!(caps["backend_protection"]["supported_protection"]
        .as_array()
        .unwrap()
        .is_empty());
    let result=api(&router,json!({"kind":"submit","task":task("task"),"message_id":"pin","action":"sglang.pin","target":prefix(vec![1,2]),
        "location":{"tier":"storage"},"mode":"acquire","claim_id":"keep","revision":1,"ttl_seconds":10,"required_protection":"backend_soft_pin"})).await;
    assert!(result["claims"][0]["claim"].is_null());
    assert_eq!(result["claims"][0]["error"], "RecoveryRequired");
    assert!(reqwest::Client::new()
        .post(format!("{}/generate", router.url))
        .json(&json!({"input_ids":[1,2]}))
        .send()
        .await
        .unwrap()
        .status()
        .is_success());
}

#[tokio::test]
async fn hostile_time_revision_and_token_bounds_do_not_terminate_the_control_actor() {
    let engine = Engine::start().await;
    let router = router(&[&engine], false, true).await;
    let client = reqwest::Client::new();
    for body in [
        json!({"kind":"session_event","task":task("task"),"event":"resume_expected","event_id":"huge-time","revision":1,"expected_wait_ms":u64::MAX}),
        json!({"kind":"session_event","task":task("task"),"event":"tool_done","event_id":"float-time","revision":1,"available_ms":1e200}),
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
        let hashes = sgl_router::state::kv_events::compute_block_hashes(&ids, 1);
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
