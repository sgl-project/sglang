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
    time::{Duration, Instant},
};
use tokio::task::JoinHandle;

#[derive(Default)]
struct EngineState {
    operations: BTreeMap<String, Value>,
    commands: Vec<Value>,
    http_commands: Vec<Value>,
    capability_gets: usize,
    inference: Vec<Value>,
    resident: u64,
    partial: Option<u64>,
    running: bool,
    queued: bool,
    read_batch: bool,
    batch_error: Option<String>,
    batch_remove_after_scope: bool,
    batch_gap_after_scope: bool,
    batch_generation_after_scope: bool,
    batch_filtered_page: bool,
    cancelled: bool,
    claims: BTreeMap<String, Value>,
    claim_deadlines: BTreeMap<String, Instant>,
    pin_refusal: Option<String>,
    lose_pin_submit_once: bool,
    fail_claim: bool,
    fail_pin_release: bool,
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
    delayed_capabilities: Option<Arc<DelayedQuery>>,
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
    let (reply, delayed) = {
        let mut s = state.lock().unwrap();
        s.capability_gets += 1;
        (capabilities_reply(&s), s.delayed_capabilities.take())
    };
    if let Some(delayed) = delayed {
        delayed.started.notify_one();
        delayed.release.notified().await;
    }
    Json(reply)
}
fn capabilities_reply(s: &EngineState) -> Value {
    let mut reply = json!({"success":true,"data":{"engine_id":"mock-engine","engine_epoch":s.epoch,"namespace":"weights-v1",
        "cache_salt_encoding":"owner_assertion","executor":{"cache_generation":1},"public_protocol":{"actions":{"sglang.prefetch":["1.0"],"sglang.demote":["1.0"],"sglang.pin":["1.0"]},
        "paths":{"sglang.prefetch":[["storage","host"],["host","gpu"]],"sglang.demote":[["gpu","storage"],["gpu","host"],["host","storage"]]},
        "storage_backend":"mooncake","backend_namespace":"pool-a","storage_targets":true,"prefix":{"namespace":"weights-v1","layout_id":"full:p1","page_size":1,"components":["full"],"hash_scheme":"sglang.sha256.u32le.i64.v1"}}}});
    if s.feedback && !s.hide_feedback {
        reply["data"]["feedback"] = json!({"query":"events","version":"1.0","source_id":"control:mock","source_epoch":s.feedback_epoch,
            "recovery":"query_known_scopes","cache_watermark":cache_watermark(&s)});
    }
    if s.read_batch {
        reply["data"]["read_batch"] = json!({"version":"1.0","max_queries":8,
            "commands":["capabilities","query","residency","claim","events"]});
    }
    reply
}
async fn command(
    State(state): State<Arc<Mutex<EngineState>>>,
    Json(body): Json<Value>,
) -> (axum::http::StatusCode, Json<Value>) {
    state.lock().unwrap().http_commands.push(body.clone());
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
    if body["kind"] == "read_batch" {
        let mode;
        {
            let mut s = state.lock().unwrap();
            s.commands.push(body.clone());
            if !s.read_batch || body["context"]["engine_epoch"] != s.epoch {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(json!({"success":false,"error":{"code":"stale_epoch"}})),
                );
            }
            mode = s.batch_error.clone();
        }
        let queries = body["queries"].as_array().unwrap();
        assert!((1..=8).contains(&queries.len()));
        let mut results = Vec::new();
        let mut saw_scope = false;
        for query in queries {
            assert!(matches!(
                query["kind"].as_str(),
                Some("capabilities" | "query" | "residency" | "claim" | "events")
            ));
            if query["kind"] == "capabilities" {
                assert!(query.get("context").is_none());
                results.push(capabilities_reply(&state.lock().unwrap())["data"].clone());
                continue;
            }
            assert_eq!(query["context"], body["context"]);
            if query["kind"] == "events"
                && !query["cursor"].is_null()
                && query["cursor"]["source_epoch"] != state.lock().unwrap().feedback_epoch
            {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(json!({"success":false,"error":{"code":"stale_cursor"}})),
                );
            }
            let (status, reply) = command_now(state.clone(), query.clone());
            if !status.is_success() {
                return (status, reply);
            }
            let mut data = reply.0["data"].clone();
            if query["kind"] == "events" && !saw_scope && state.lock().unwrap().batch_filtered_page
            {
                state.lock().unwrap().batch_filtered_page = false;
                data["items"] = json!([]);
                data["earliest_sequence"] = json!(1);
                data["next_cursor"]["sequence"] =
                    json!(data["latest_sequence"].as_u64().unwrap().saturating_sub(1));
            }
            if query["kind"] == "residency" || query["kind"] == "claim" {
                saw_scope = true;
                let mut s = state.lock().unwrap();
                if s.batch_remove_after_scope {
                    s.batch_remove_after_scope = false;
                    s.resident = 0;
                    s.cache_version += 1;
                    feedback_event(&mut s, "invalidation", json!({"tiers":["host"]}));
                }
                if s.batch_gap_after_scope {
                    s.feedback_gap = true;
                }
                if s.batch_generation_after_scope {
                    s.batch_generation_after_scope = false;
                    s.native_generation += 1;
                    s.cache_version = 0;
                    feedback_event(&mut s, "invalidation", json!({"tiers":["host"]}));
                }
            }
            results.push(data);
        }
        if mode.as_deref() == Some("partial") {
            results.pop();
        }
        if mode.as_deref() == Some("oversized") {
            results[0]["padding"] = json!("x".repeat(1_100_000));
        }
        return (
            StatusCode::OK,
            Json(json!({"success":true,"data":{
            "version":if mode.as_deref()==Some("version"){"2.0"}else{"1.0"},"results":results}})),
        );
    }
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
        if s.fail_claim {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({"success":false})),
            );
        }
        let id = body["claim_id"].as_str().unwrap();
        if s.claim_deadlines
            .get(id)
            .is_some_and(|deadline| Instant::now() >= *deadline)
        {
            if let Some(claim) = s.claims.get_mut(id) {
                if claim["state"] == "active" {
                    claim["state"] = json!("expired");
                    claim["effective_protection"] = Value::Null;
                }
            }
        }
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
        let mut result = json!({"ref":{"context":body["context"],"message_id":id,"action_id":"transfer","operation_id":format!("op-{id}")},
            "sequence":1,"update":{"status":if s.queued{"queued"}else if s.running{"running"}else{"completed"},"quiescent":!s.running},"public_result":{"state_version":99_999}});
        if action["action_type"] == "sglang.pin" {
            let payload = &action["payload"];
            if payload["mode"] == "release" && s.fail_pin_release {
                return (
                    StatusCode::SERVICE_UNAVAILABLE,
                    Json(json!({"success":false})),
                );
            }
            if !s.operations.contains_key(&id) {
                if let Some(refusal) = &s.pin_refusal {
                    result["update"] = json!({"status":refusal,"quiescent":true});
                    if refusal == "missing" {
                        s.resident = 0;
                    }
                } else {
                    let epoch = s.epoch.clone();
                    let claim_id = payload["claim_id"].as_str().unwrap().to_string();
                    s.claims.insert(claim_id.clone(),json!({"owner_epoch":epoch,"revision":payload["revision"],
                "state":if payload["mode"]=="release"{"released"}else{"active"},"effective_protection":if payload["mode"]=="release"{Value::Null}else{json!("eviction_exclusion")},"actual_scope":[{"start_token":0,"end_token":count}]}));
                    if let Some(ttl) = payload["ttl_seconds"].as_f64() {
                        s.claim_deadlines
                            .insert(claim_id, Instant::now() + Duration::from_secs_f64(ttl));
                    }
                }
            }
        }
        let fresh = !s.operations.contains_key(&id);
        s.operations.entry(id).or_insert(result);
        if fresh && !s.running && !s.queued && action["action_type"] == "sglang.prefetch" {
            s.resident = s.partial.unwrap_or(count);
        }
        if s.lose_submit_once || (s.lose_pin_submit_once && action["action_type"] == "sglang.pin") {
            s.lose_submit_once = false;
            s.lose_pin_submit_once = false;
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
    let engine = feedback_engine().await;
    let router = router_with(&[&engine], false, Some(config)).await;
    start(&router, "a").await;
    start(&router, "b").await;
    tokio::time::timeout(Duration::from_secs(3), async {
        loop {
            let view = feedback_view(&router).await;
            if view["owners"]
                .as_array()
                .is_some_and(|owners| !owners.is_empty())
                && view["owners"][0]["error"].is_null()
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
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
    {
        let mut state = engine.state.lock().unwrap();
        feedback_event(
            &mut state,
            "invalidation",
            json!({"reason":"storage_attachment_changed","tiers":["storage"],"semantics":"owner_storage_observations_only"}),
        );
    }
    let b = tokio::time::timeout(Duration::from_secs(3), async {
        loop {
            let view = query(&router, "b", "pin").await;
            if view["claims"][0]["claim"]["phase"] == "unknown" {
                break view;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    assert!(b["claims"][0]["claim"]["effective_protection"].is_null());
    assert_eq!(backend.lock().unwrap().0.len(), 4);
    // A cached backend-owner receipt cannot silently repair attachment history.
    assert_eq!(
        query(&router, "b", "pin").await["claims"][0]["claim"]["phase"],
        "unknown"
    );
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
async fn measured_policy_keeps_late_preparation_and_learns_confirmed_scope() {
    let engine = Engine::start().await;
    let mut cfg = orchestration();
    cfg.preparation_policy = Some(sgl_kv_orchestrator::preparation::PreparationConfig {
        min_samples: 1,
        max_lead_ms: 200,
        ..Default::default()
    });
    // Keep preparation useful despite a lead estimate beyond the remaining
    // window; one millisecond cannot survive a real HTTP discovery round trip.
    cfg.initial_prefetch_ms_per_token = 100.0;
    let router = router_with(&[&engine], false, Some(cfg)).await;
    start(&router, "measured").await;
    api(&router,json!({"kind":"session_event","task":task("measured"),"event":"request_observed","event_id":"observed","revision":1,
        "branch_id":"main","branch_revision":1,"request_id":"r","request_incarnation":"one","input_ids":[1,2,3,4],"worker_url":engine.url})).await;
    let view = api(&router,json!({"kind":"session_event","task":task("measured"),"event":"resume_expected","event_id":"resume","revision":2,
        "target":{"kind":"session"},"expected_wait_ms":100,"worker_url":engine.url})).await;
    assert_eq!(view["preparation_estimate"]["lead_ms"], 200);
    assert_eq!(view["preparation_estimate"]["samples"], 0);
    assert_eq!(view["operations"].as_array().unwrap().len(), 1);
    ready(&router, "measured", "resume").await;
    let caps = api(&router, json!({"kind":"capabilities"})).await;
    assert_eq!(caps["preparation_cost_groups"], 1);
    assert_eq!(caps["preparation_policy"]["max_lead_ms"], 200);
}

#[tokio::test]
async fn measured_policy_advances_early_tool_done_without_reusing_old_scope() {
    let engine = Engine::start().await;
    let mut cfg = orchestration();
    cfg.preparation_policy = Some(Default::default());
    let router = router_with(&[&engine], false, Some(cfg)).await;
    start(&router, "early").await;
    api(&router,json!({"kind":"session_event","task":task("early"),"event":"request_observed","event_id":"observed","revision":1,
        "branch_id":"main","branch_revision":1,"request_id":"r","request_incarnation":"one","input_ids":[1,2,3,4],"worker_url":engine.url})).await;
    api(&router,json!({"kind":"session_event","task":task("early"),"event":"resume_expected","event_id":"resume","revision":2,
        "target":{"kind":"session"},"expected_wait_ms":10000,"worker_url":engine.url})).await;
    let done = api(&router,json!({"kind":"session_event","task":task("early"),"event":"tool_done","event_id":"done","revision":3,
        "target":{"kind":"session"},"available_ms":0,"worker_url":engine.url})).await;
    assert_eq!(done["state"], "on_demand_fallback");
    ready(&router, "early", "resume").await;
    api(&router,json!({"kind":"session_event","task":task("early"),"event":"request_observed","event_id":"grown","revision":4,
        "branch_id":"main","branch_revision":2,"request_id":"r2","request_incarnation":"two","input_ids":[1,2,3,4,5,6],"worker_url":engine.url})).await;
    let grown = api(&router,json!({"kind":"session_event","task":task("early"),"event":"tool_done","event_id":"grown-done","revision":5,
        "target":{"kind":"session"},"available_ms":10000,"worker_url":engine.url})).await;
    assert_eq!(grown["known_scope"][0]["end_token"], 6);
    let grown = ready(&router, "early", "grown-done").await;
    assert_eq!(grown["operations"][0]["ready_scope"]["end_token"], 6);
}

fn auto_pin_config(ttl_ms: u64, max_tokens: u64) -> OrchestrationConfig {
    let mut cfg = orchestration();
    cfg.preparation_policy = Some(sgl_kv_orchestrator::preparation::PreparationConfig {
        ready_pin_ttl_ms: ttl_ms,
        auto_pin_max_tokens: max_tokens,
        ..Default::default()
    });
    cfg
}
async fn lifecycle_prepare(router: &TestRouter, engine: &Engine, id: &str, tokens: Vec<u32>) {
    api(router,json!({"kind":"session_event","task":task(id),"event":"resume_expected","event_id":"prepare","revision":2,
        "target":prefix(tokens),"expected_wait_ms":100,"worker_url":engine.url})).await;
}
async fn pin_view(router: &TestRouter, id: &str, expected: &str) -> Value {
    tokio::time::timeout(Duration::from_secs(4), async {
        loop {
            let view = query(router, id, "prepare").await;
            if view["operations"][0]["auto_pin"]["state"] == expected {
                return view;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap()
}

fn window_config() -> OrchestrationConfig {
    let mut cfg = orchestration();
    cfg.preparation_policy = Some(Default::default());
    cfg.poll_ms = 10_000;
    cfg.initial_prefetch_ms_per_token = 100.0;
    cfg
}
fn resume_event(id: &str, wait_ms: u64, engine: &Engine) -> Value {
    json!({"kind":"session_event","task":task(id),"event":"resume_expected",
        "event_id":"prepare","revision":2,"target":prefix(vec![1,2,3,4]),
        "expected_wait_ms":wait_ms,"worker_url":engine.url})
}

#[tokio::test]
async fn absolute_resume_window_includes_mailbox_wait_and_keeps_replay_identity() {
    for measured in [false, true] {
        let engine = Engine::start().await;
        let mut cfg = window_config();
        if !measured {
            cfg.preparation_policy = None;
            cfg.initial_prefetch_ms_per_token = 0.5;
        }
        let router = Arc::new(router_with(&[&engine], false, Some(cfg)).await);
        start(&router, "block").await;
        start(&router, "late").await;
        engine.state.lock().unwrap().partial = Some(2);
        submit(
            &router,
            "block",
            "prepare",
            prefix(vec![9, 8, 7, 6]),
            &engine,
        )
        .await;
        ready(&router, "block", "prepare").await;
        engine.state.lock().unwrap().partial = None;
        let delayed = Arc::new(DelayedQuery::default());
        engine.state.lock().unwrap().delayed_query = Some(delayed.clone());
        let r = router.clone();
        let blocker = tokio::spawn(async move { query(&r, "block", "prepare").await });
        tokio::time::timeout(Duration::from_secs(2), delayed.started.notified())
            .await
            .unwrap();
        let body = resume_event("late", 100, &engine);
        let r = router.clone();
        let request = body.clone();
        let queued = tokio::spawn(async move { api(&r, request).await });
        tokio::time::sleep(Duration::from_millis(180)).await;
        delayed.release.notify_one();
        blocker.await.unwrap();
        let receipt = queued.await.unwrap();
        if measured {
            assert_eq!(receipt["state"], "on_demand_fallback");
            assert_eq!(engine.state.lock().unwrap().operations.len(), 1);
        } else {
            assert!(receipt["operations"].is_array());
            ready(&router, "late", "prepare").await;
            assert_eq!(engine.state.lock().unwrap().operations.len(), 2);
        }
        tokio::time::sleep(Duration::from_millis(30)).await;
        assert_eq!(api(&router, body.clone()).await, receipt);
    }
}

#[tokio::test]
async fn absolute_resume_due_does_not_slide_by_mailbox_time() {
    let engine = Engine::start().await;
    let mut cfg = window_config();
    cfg.initial_prefetch_ms_per_token = 70.0;
    let router = Arc::new(router_with(&[&engine], false, Some(cfg)).await);
    start(&router, "block").await;
    start(&router, "due").await;
    engine.state.lock().unwrap().partial = Some(2);
    submit(
        &router,
        "block",
        "prepare",
        prefix(vec![9, 8, 7, 6]),
        &engine,
    )
    .await;
    ready(&router, "block", "prepare").await;
    engine.state.lock().unwrap().partial = None;
    let delayed = Arc::new(DelayedQuery::default());
    engine.state.lock().unwrap().delayed_query = Some(delayed.clone());
    let r = router.clone();
    let blocker = tokio::spawn(async move { query(&r, "block", "prepare").await });
    tokio::time::timeout(Duration::from_secs(2), delayed.started.notified())
        .await
        .unwrap();
    let sent = tokio::time::Instant::now();
    let body = resume_event("due", 1000, &engine);
    let r = router.clone();
    let queued = tokio::spawn(async move { api(&r, body).await });
    tokio::time::sleep(Duration::from_millis(300)).await;
    delayed.release.notify_one();
    blocker.await.unwrap();
    queued.await.unwrap();
    tokio::time::sleep_until(sent + Duration::from_millis(780)).await;
    assert_eq!(
        query(&router, "due", "prepare").await["operations"][0]["ready_scope"]["end_token"],
        4
    );
}

#[tokio::test]
async fn discovery_delay_expires_unattempted_preparation_without_engine_submit() {
    let engine = Engine::start().await;
    let router = Arc::new(router_with(&[&engine], false, Some(window_config())).await);
    start(&router, "discovery").await;
    tokio::time::sleep(Duration::from_millis(20)).await;
    let delayed = Arc::new(DelayedQuery::default());
    engine.state.lock().unwrap().delayed_capabilities = Some(delayed.clone());
    let r = router.clone();
    let body = resume_event("discovery", 100, &engine);
    let queued = tokio::spawn(async move { api(&r, body).await });
    delayed.started.notified().await;
    tokio::time::sleep(Duration::from_millis(150)).await;
    delayed.release.notify_one();
    queued.await.unwrap();
    let late = query(&router, "discovery", "prepare").await;
    assert_eq!(
        late["operations"][0]["policy_fallback"],
        "late_before_submit"
    );
    assert_eq!(late["quiescent"], true);
    assert!(late["operations"][0]["error"].is_null());
    assert!(engine.state.lock().unwrap().operations.is_empty());
}

#[tokio::test]
async fn expired_preparation_can_take_future_or_explicit_shared_interest() {
    for explicit in [false, true] {
        let engine = Engine::start().await;
        let router = router_with(&[&engine], false, Some(window_config())).await;
        start(&router, "old").await;
        start(&router, "new").await;
        tokio::time::sleep(Duration::from_millis(20)).await;
        api(&router, resume_event("old", 100, &engine)).await;
        tokio::time::sleep(Duration::from_millis(150)).await;
        let expired = query(&router, "old", "prepare").await;
        assert_eq!(
            expired["operations"][0]["policy_fallback"],
            "late_before_submit"
        );
        assert_eq!(expired["quiescent"], true);
        engine.state.lock().unwrap().running = true;
        let admitted = if explicit {
            submit(&router, "new", "prepare", prefix(vec![1, 2, 3, 4]), &engine).await
        } else {
            api(&router, resume_event("new", 200, &engine)).await
        };
        assert_eq!(
            admitted["operations"][0]["child_id"],
            expired["operations"][0]["child_id"]
        );
        let running = query(&router, "new", "prepare").await;
        assert_eq!(running["quiescent"], false);
        assert!(running["operations"][0]["policy_fallback"].is_null());
        assert_eq!(query(&router, "old", "prepare").await["quiescent"], false);
        {
            let mut s = engine.state.lock().unwrap();
            s.running = false;
            s.resident = 4;
            let result = s.operations.values_mut().next().unwrap();
            result["sequence"] = json!(2);
            result["update"] = json!({"status":"completed","quiescent":true});
        }
        ready(&router, "new", "prepare").await;
        assert_eq!(engine.state.lock().unwrap().operations.len(), 1);
    }
}

#[tokio::test]
async fn automatic_preparation_falls_back_while_cancelled_episode_waits_for_ack() {
    for event_name in ["resume_expected", "tool_done"] {
        for drained in [false, true] {
            let engine = Engine::start().await;
            engine.state.lock().unwrap().running = true;
            let router = router_with(&[&engine], false, Some(window_config())).await;
            start(&router, "old").await;
            start(&router, "new").await;
            tokio::time::sleep(Duration::from_millis(20)).await;
            submit(&router, "old", "prepare", prefix(vec![1, 2, 3, 4]), &engine).await;
            query(&router, "old", "prepare").await;
            api(
                &router,
                json!({"kind":"cancel","task":task("old"),"message_id":"prepare"}),
            )
            .await;
            if drained {
                assert_eq!(query(&router, "old", "prepare").await["quiescent"], true);
            }
            let old_wire = engine
                .state
                .lock()
                .unwrap()
                .operations
                .keys()
                .next()
                .unwrap()
                .clone();
            let before = engine
                .state
                .lock()
                .unwrap()
                .commands
                .iter()
                .filter(|c| c["kind"] == "submit")
                .count();
            let mut body = resume_event("new", 1000, &engine);
            body["event"] = json!(event_name);
            if event_name == "tool_done" {
                body["available_ms"] = json!(10000);
            }
            let fallback = api(&router, body.clone()).await;
            assert_eq!(fallback["state"], "on_demand_fallback");
            assert_eq!(fallback["reason"], "cancelled_preparation_draining");
            assert_eq!(api(&router, body).await, fallback);
            let finish = event(&router, "new", "task_finished", 3, "finish").await;
            assert!(finish["intents"].as_array().unwrap().is_empty());
            assert_eq!(
                engine
                    .state
                    .lock()
                    .unwrap()
                    .commands
                    .iter()
                    .filter(|c| c["kind"] == "submit")
                    .count(),
                before
            );
            let old = query(&router, "old", "prepare").await;
            assert_eq!(old["quiescent"], true);
            assert_eq!(
                old["operations"][0]["result"]["update"]["status"],
                "cancelled"
            );
            api(
                &router,
                json!({"kind":"ack","task":task("old"),"message_id":"prepare"}),
            )
            .await;
            engine.state.lock().unwrap().running = false;
            start(&router, "fresh").await;
            api(&router, resume_event("fresh", 500, &engine)).await;
            let fresh = ready(&router, "fresh", "prepare").await;
            let new_wire = fresh["operations"][0]["result"]["ref"]["message_id"]
                .as_str()
                .unwrap();
            assert_ne!(new_wire, old_wire);
            assert_eq!(engine.state.lock().unwrap().operations.len(), 2);
        }
    }
}

#[tokio::test]
async fn cancelled_episode_conflict_stays_an_error_for_explicit_or_legacy_preparation() {
    for measured in [false, true] {
        let engine = Engine::start().await;
        engine.state.lock().unwrap().running = true;
        let mut cfg = window_config();
        if !measured {
            cfg.preparation_policy = None;
            cfg.initial_prefetch_ms_per_token = 0.5;
        }
        let router = router_with(&[&engine], false, Some(cfg)).await;
        start(&router, "old").await;
        start(&router, "new").await;
        tokio::time::sleep(Duration::from_millis(20)).await;
        submit(&router, "old", "prepare", prefix(vec![1, 2, 3, 4]), &engine).await;
        query(&router, "old", "prepare").await;
        api(
            &router,
            json!({"kind":"cancel","task":task("old"),"message_id":"prepare"}),
        )
        .await;
        let body = if measured {
            // An explicit caller cannot opt into fallback by adding an event.
            json!({"kind":"submit","task":task("new"),"message_id":"prepare",
                "event":"resume_expected","expected_wait_ms":1000,
                "action":"sglang.prefetch","target":prefix(vec![1,2,3,4]),"worker_url":engine.url,
                "source":{"tier":"storage"},"destination":{"tier":"host"}})
        } else {
            resume_event("new", 1000, &engine)
        };
        let reply: Value = reqwest::Client::new()
            .post(format!("{}/v1/kv/control", router.url))
            .bearer_auth("test-control")
            .json(&body)
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(reply["success"], false);
        assert_eq!(reply["error"]["reason"], "Conflict");
        assert_eq!(engine.state.lock().unwrap().operations.len(), 1);
    }
}

#[tokio::test]
async fn cancelled_child_fallback_is_atomic_and_cannot_reschedule_other_live_children() {
    let engine = Engine::start().await;
    engine.state.lock().unwrap().running = true;
    let router = router_with(&[&engine], false, Some(window_config())).await;
    for id in ["cancelled", "live", "fanout"] {
        start(&router, id).await;
    }
    tokio::time::sleep(Duration::from_millis(20)).await;
    submit(
        &router,
        "cancelled",
        "prepare",
        prefix(vec![1, 2, 3, 4]),
        &engine,
    )
    .await;
    query(&router, "cancelled", "prepare").await;
    api(
        &router,
        json!({"kind":"cancel","task":task("cancelled"),"message_id":"prepare"}),
    )
    .await;
    let mut live = resume_event("live", 5000, &engine);
    live["target"] = prefix(vec![5, 6, 7, 8]);
    api(&router, live).await;
    for (index, ids) in [vec![1, 2, 3, 4], vec![5, 6, 7, 8], vec![9, 10, 11, 12]]
        .into_iter()
        .enumerate()
    {
        api(
            &router,
            json!({"kind":"session_event","task":task("fanout"),"event":"request_observed",
            "event_id":format!("observed-{index}"),"revision":index+1,
            "branch_id":format!("branch-{index}"),"branch_revision":1,
            "request_id":format!("request-{index}"),"request_incarnation":"one",
            "input_ids":ids,"worker_url":engine.url}),
        )
        .await;
    }
    let mut fanout = resume_event("fanout", 1000, &engine);
    fanout["revision"] = json!(4);
    fanout["target"] = json!({"kind":"session"});
    let fallback = api(&router, fanout).await;
    assert_eq!(fallback["state"], "on_demand_fallback");
    assert_eq!(fallback["reason"], "cancelled_preparation_draining");
    let finish = event(&router, "fanout", "task_finished", 5, "finish").await;
    assert!(finish["intents"].as_array().unwrap().is_empty());
    // A failed fanout's 1s window must not replace the existing 5s schedule.
    tokio::time::sleep(Duration::from_millis(800)).await;
    assert!(query(&router, "live", "prepare").await["operations"][0]["result"].is_null());
    assert_eq!(engine.state.lock().unwrap().operations.len(), 1);
    for id in ["cancelled", "live"] {
        api(
            &router,
            json!({"kind":"cancel","task":task(id),"message_id":"prepare"}),
        )
        .await;
        assert_eq!(query(&router, id, "prepare").await["quiescent"], true);
        api(
            &router,
            json!({"kind":"ack","task":task(id),"message_id":"prepare"}),
        )
        .await;
    }
}

#[tokio::test]
async fn cancelled_preparation_does_not_reopen_for_a_new_predicted_interest() {
    let engine = Engine::start().await;
    engine.state.lock().unwrap().running = true;
    let router = router_with(&[&engine], false, Some(window_config())).await;
    start(&router, "old").await;
    start(&router, "new").await;
    submit(&router, "old", "prepare", prefix(vec![1, 2, 3, 4]), &engine).await;
    query(&router, "old", "prepare").await;
    api(
        &router,
        json!({"kind":"cancel","task":task("old"),"message_id":"prepare"}),
    )
    .await;
    let cancelled = query(&router, "old", "prepare").await;
    assert_eq!(
        cancelled["operations"][0]["result"]["update"]["status"],
        "cancelled"
    );
    let reply: Value = reqwest::Client::new()
        .post(format!("{}/v1/kv/control", router.url))
        .bearer_auth("test-control")
        .json(&resume_event("new", 1000, &engine))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(reply["success"], true);
    assert_eq!(reply["data"]["state"], "on_demand_fallback");
    assert_eq!(reply["data"]["reason"], "cancelled_preparation_draining");
    assert_eq!(engine.state.lock().unwrap().operations.len(), 1);
}

#[tokio::test]
async fn attempted_lost_response_keeps_wire_identity_after_window_and_skips_new_pin() {
    let engine = Engine::start().await;
    engine.state.lock().unwrap().lose_submit_once = true;
    let mut cfg = window_config();
    cfg.preparation_policy.as_mut().unwrap().ready_pin_ttl_ms = 1000;
    cfg.preparation_policy.as_mut().unwrap().auto_pin_max_tokens = 4;
    let router = router_with(&[&engine], false, Some(cfg)).await;
    start(&router, "attempted").await;
    tokio::time::sleep(Duration::from_millis(20)).await;
    api(&router, resume_event("attempted", 100, &engine)).await;
    assert_eq!(
        query(&router, "attempted", "prepare").await["quiescent"],
        false
    );
    tokio::time::sleep(Duration::from_millis(150)).await;
    let done = ready(&router, "attempted", "prepare").await;
    assert!(done["operations"][0]["auto_pin"].is_null());
    let s = engine.state.lock().unwrap();
    let submits: Vec<_> = s
        .commands
        .iter()
        .filter(|c| c["kind"] == "submit")
        .collect();
    assert!(submits.len() >= 2);
    assert!(submits
        .iter()
        .all(|c| c["envelope"]["message_id"] == submits[0]["envelope"]["message_id"]));
    assert_eq!(s.operations.len(), 1);
    assert!(s.claims.is_empty());
}

#[tokio::test]
async fn existing_ready_pin_is_kept_after_resume_window_until_normal_cleanup() {
    let engine = Engine::start().await;
    let mut cfg = window_config();
    cfg.preparation_policy.as_mut().unwrap().ready_pin_ttl_ms = 1000;
    cfg.preparation_policy.as_mut().unwrap().auto_pin_max_tokens = 4;
    let router = router_with(&[&engine], false, Some(cfg)).await;
    start(&router, "retained").await;
    api(&router, resume_event("retained", 100, &engine)).await;
    pin_view(&router, "retained", "active").await;
    tokio::time::sleep(Duration::from_millis(150)).await;
    assert_eq!(
        pin_view(&router, "retained", "active").await["operations"][0]["auto_pin"]["state"],
        "active"
    );
    event(&router, "retained", "task_finished", 3, "finish").await;
    assert_eq!(
        pin_view(&router, "retained", "released").await["quiescent"],
        true
    );
}

#[tokio::test]
async fn pin_child_discovery_cannot_issue_a_new_claim_after_resume_window() {
    let engine = Engine::start().await;
    let mut cfg = window_config();
    cfg.preparation_policy.as_mut().unwrap().ready_pin_ttl_ms = 1000;
    cfg.preparation_policy.as_mut().unwrap().auto_pin_max_tokens = 4;
    let router = Arc::new(router_with(&[&engine], false, Some(cfg)).await);
    start(&router, "pin-discovery").await;
    tokio::time::sleep(Duration::from_millis(20)).await;
    api(&router, resume_event("pin-discovery", 250, &engine)).await;
    let scope = Arc::new(DelayedQuery::default());
    engine.state.lock().unwrap().delayed_query = Some(scope.clone());
    let r = router.clone();
    let query_task = tokio::spawn(async move { query(&r, "pin-discovery", "prepare").await });
    tokio::time::timeout(Duration::from_secs(2), scope.started.notified())
        .await
        .unwrap();
    let caps = Arc::new(DelayedQuery::default());
    engine.state.lock().unwrap().delayed_capabilities = Some(caps.clone());
    scope.release.notify_one();
    tokio::time::timeout(Duration::from_secs(2), caps.started.notified())
        .await
        .unwrap();
    tokio::time::sleep(Duration::from_millis(300)).await;
    caps.release.notify_one();
    let view = query_task.await.unwrap();
    assert_eq!(view["quiescent"], true);
    let s = engine.state.lock().unwrap();
    assert!(s.claims.is_empty());
    assert!(s
        .commands
        .iter()
        .filter(|c| c["kind"] == "submit")
        .all(|c| c["envelope"]["actions"][0]["action_type"] != "sglang.pin"));
}

#[tokio::test]
async fn read_batch_negotiates_opt_in_and_preserves_old_engine_fallback() {
    for (measured, supported) in [(false, true), (true, false), (true, true)] {
        let engine = Engine::start().await;
        {
            let mut s = engine.state.lock().unwrap();
            s.read_batch = supported;
            s.feedback = true;
        }
        let mut cfg = orchestration();
        if measured {
            cfg.preparation_policy = Some(Default::default());
        }
        let router = router_with(&[&engine], false, Some(cfg)).await;
        start(&router, "batch").await;
        submit(
            &router,
            "batch",
            "prepare",
            prefix(vec![1, 2, 3, 4]),
            &engine,
        )
        .await;
        ready(&router, "batch", "prepare").await;
        tokio::time::sleep(Duration::from_millis(60)).await;
        ready(&router, "batch", "prepare").await;
        let s = engine.state.lock().unwrap();
        let batches: Vec<_> = s
            .commands
            .iter()
            .filter(|c| c["kind"] == "read_batch")
            .collect();
        assert_eq!(!batches.is_empty(), measured && supported);
        if measured && supported {
            assert!(batches.iter().any(|b| b["queries"]
                .as_array()
                .unwrap()
                .iter()
                .any(|q| q["kind"] == "query")));
            assert!(batches
                .iter()
                .any(|b| b["queries"].as_array().unwrap().len() == 4));
            for batch in batches {
                let kinds: Vec<_> = batch["queries"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|q| q["kind"].as_str().unwrap())
                    .collect();
                assert!(
                    kinds == ["capabilities", "events", "query", "residency", "events"]
                        || kinds == ["capabilities", "events", "residency", "events"]
                );
            }
        }
    }
}

#[tokio::test]
async fn read_batch_ongoing_partial_observation_uses_one_http_rpc() {
    let engine = Engine::start().await;
    {
        let mut s = engine.state.lock().unwrap();
        s.read_batch = true;
        s.feedback = true;
        s.partial = Some(2);
    }
    let mut cfg = orchestration();
    cfg.preparation_policy = Some(Default::default());
    cfg.poll_ms = 10_000;
    let router = router_with(&[&engine], false, Some(cfg)).await;
    start(&router, "batch").await;
    submit(
        &router,
        "batch",
        "prepare",
        prefix(vec![1, 2, 3, 4]),
        &engine,
    )
    .await;
    ready(&router, "batch", "prepare").await;
    // Let the interval's first immediate tick settle. The next tick is ten
    // seconds away; partial coverage deliberately bypasses the ready cache.
    tokio::time::sleep(Duration::from_millis(20)).await;
    let gets;
    {
        let mut s = engine.state.lock().unwrap();
        s.http_commands.clear();
        gets = s.capability_gets;
    }
    assert_eq!(
        query(&router, "batch", "prepare").await["operations"][0]["ready_scope"]["end_token"],
        2
    );
    let s = engine.state.lock().unwrap();
    assert_eq!(s.capability_gets, gets);
    assert_eq!(s.http_commands.len(), 1);
    assert_eq!(s.http_commands[0]["kind"], "read_batch");
    assert_eq!(s.http_commands[0]["queries"].as_array().unwrap().len(), 4);
}

#[tokio::test]
async fn read_batch_queries_queued_receipts_and_current_partial_residency() {
    let engine = Engine::start().await;
    {
        let mut s = engine.state.lock().unwrap();
        s.read_batch = true;
        s.feedback = true;
        s.queued = true;
    }
    let mut cfg = orchestration();
    cfg.preparation_policy = Some(Default::default());
    let router = router_with(&[&engine], false, Some(cfg)).await;
    start(&router, "batch").await;
    submit(
        &router,
        "batch",
        "prepare",
        prefix(vec![1, 2, 3, 4]),
        &engine,
    )
    .await;
    let queued = query(&router, "batch", "prepare").await;
    assert_eq!(
        queued["operations"][0]["result"]["update"]["status"],
        "queued"
    );
    {
        let mut s = engine.state.lock().unwrap();
        s.queued = false;
        s.resident = 2;
        let result = s.operations.values_mut().next().unwrap();
        result["sequence"] = json!(2);
        result["update"] = json!({"status":"completed","quiescent":true});
    }
    let partial = ready(&router, "batch", "prepare").await;
    assert_eq!(partial["operations"][0]["ready_scope"]["end_token"], 2);
    assert_eq!(partial["operations"][0]["result"]["sequence"], 2);
}

#[tokio::test]
async fn read_batch_retains_pin_claim_ttl_and_new_release_messages() {
    let engine = Engine::start().await;
    {
        let mut s = engine.state.lock().unwrap();
        s.read_batch = true;
        s.feedback = true;
    }
    let router = router_with(&[&engine], false, Some(auto_pin_config(2000, 4))).await;
    start(&router, "batch-pin").await;
    lifecycle_prepare(&router, &engine, "batch-pin", vec![1, 2, 3, 4]).await;
    pin_view(&router, "batch-pin", "active").await;
    event(&router, "batch-pin", "task_finished", 3, "finish").await;
    assert_eq!(
        pin_view(&router, "batch-pin", "released").await["quiescent"],
        true
    );
    let s = engine.state.lock().unwrap();
    assert!(s.commands.iter().any(|c| c["kind"] == "read_batch"
        && c["queries"]
            .as_array()
            .unwrap()
            .iter()
            .any(|q| q["kind"] == "claim")));
    assert!(s.commands.iter().any(|c| c["kind"] == "query"
        && c["message_id"]
            .as_str()
            .is_some_and(|id| id.ends_with("-release"))));
    drop(s);
    start(&router, "batch-ttl").await;
    lifecycle_prepare(&router, &engine, "batch-ttl", vec![5, 6, 7, 8]).await;
    pin_view(&router, "batch-ttl", "active").await;
    {
        let mut s = engine.state.lock().unwrap();
        for deadline in s.claim_deadlines.values_mut() {
            *deadline = Instant::now();
        }
    }
    assert_eq!(
        pin_view(&router, "batch-ttl", "expired").await["quiescent"],
        true
    );
}

#[tokio::test]
async fn read_batch_rejects_partial_version_and_oversized_replies_then_recovers() {
    for mode in ["partial", "version", "oversized"] {
        let engine = Engine::start().await;
        {
            let mut s = engine.state.lock().unwrap();
            s.read_batch = true;
            s.feedback = true;
            s.batch_error = Some(mode.into());
        }
        let mut cfg = orchestration();
        cfg.preparation_policy = Some(Default::default());
        let router = router_with(&[&engine], false, Some(cfg)).await;
        start(&router, "batch").await;
        submit(
            &router,
            "batch",
            "prepare",
            prefix(vec![1, 2, 3, 4]),
            &engine,
        )
        .await;
        let failed = tokio::time::timeout(Duration::from_secs(3), async {
            loop {
                let view = query(&router, "batch", "prepare").await;
                if view["operations"][0]["error"].is_string() {
                    break view;
                }
                tokio::time::sleep(Duration::from_millis(60)).await;
            }
        })
        .await
        .unwrap();
        assert!(
            failed["operations"][0]["ready_scope"].is_null(),
            "{mode}: {failed}"
        );
        engine.state.lock().unwrap().batch_error = None;
        assert_eq!(
            ready(&router, "batch", "prepare").await["operations"][0]["ready_scope"]["end_token"],
            4
        );
    }
}

#[tokio::test]
async fn read_batch_rebinds_feedback_epoch_and_downgrades_transport_without_stale_ready() {
    let engine = Engine::start().await;
    {
        let mut s = engine.state.lock().unwrap();
        s.read_batch = true;
        s.feedback = true;
    }
    let mut cfg = orchestration();
    cfg.preparation_policy = Some(Default::default());
    let router = router_with(&[&engine], false, Some(cfg)).await;
    start(&router, "batch").await;
    submit(
        &router,
        "batch",
        "prepare",
        prefix(vec![1, 2, 3, 4]),
        &engine,
    )
    .await;
    ready(&router, "batch", "prepare").await;
    {
        let mut s = engine.state.lock().unwrap();
        s.feedback_epoch = "journal-2".into();
        s.feedback_sequence = 0;
        s.feedback_events.clear();
    }
    tokio::time::sleep(Duration::from_millis(60)).await;
    ready(&router, "batch", "prepare").await;
    {
        let mut s = engine.state.lock().unwrap();
        s.read_batch = false;
        s.resident = 0;
    }
    tokio::time::sleep(Duration::from_millis(60)).await;
    let absent = query(&router, "batch", "prepare").await;
    assert!(absent["operations"][0]["ready_scope"].is_null());
    engine.state.lock().unwrap().resident = 4;
    ready(&router, "batch", "prepare").await;
    let batches = engine
        .state
        .lock()
        .unwrap()
        .commands
        .iter()
        .filter(|c| c["kind"] == "read_batch")
        .count();
    tokio::time::sleep(Duration::from_millis(60)).await;
    ready(&router, "batch", "prepare").await;
    assert_eq!(
        engine
            .state
            .lock()
            .unwrap()
            .commands
            .iter()
            .filter(|c| c["kind"] == "read_batch")
            .count(),
        batches
    );
    engine.state.lock().unwrap().epoch = "epoch-2".into();
    tokio::time::sleep(Duration::from_millis(60)).await;
    assert!(query(&router, "batch", "prepare").await["operations"][0]["ready_scope"].is_null());
}

#[tokio::test]
async fn read_batch_rejects_scope_crossing_native_removal_or_feedback_gap() {
    for mode in ["removal", "gap", "generation"] {
        let engine = Engine::start().await;
        {
            let mut s = engine.state.lock().unwrap();
            s.read_batch = true;
            s.feedback = true;
        }
        let mut cfg = orchestration();
        cfg.preparation_policy = Some(Default::default());
        let router = router_with(&[&engine], false, Some(cfg)).await;
        start(&router, "batch").await;
        submit(
            &router,
            "batch",
            "prepare",
            prefix(vec![1, 2, 3, 4]),
            &engine,
        )
        .await;
        ready(&router, "batch", "prepare").await;
        {
            let mut s = engine.state.lock().unwrap();
            s.batch_gap_after_scope = mode == "gap";
            s.batch_remove_after_scope = mode == "removal";
            s.batch_generation_after_scope = mode == "generation";
        }
        tokio::time::sleep(Duration::from_millis(60)).await;
        let failed = tokio::time::timeout(Duration::from_secs(3), async {
            loop {
                let view = query(&router, "batch", "prepare").await;
                if view["operations"][0]["ready_scope"].is_null() {
                    break view;
                }
                tokio::time::sleep(Duration::from_millis(60)).await;
            }
        })
        .await
        .unwrap();
        assert!(
            failed["operations"][0]["ready_scope"].is_null(),
            "{mode}: {failed}"
        );
        {
            let mut s = engine.state.lock().unwrap();
            s.resident = 4;
            s.batch_gap_after_scope = false;
        }
        ready(&router, "batch", "prepare").await;
    }
}

#[tokio::test]
async fn read_batch_catches_up_full_and_filtered_feedback_pages_before_ready() {
    for filtered in [false, true] {
        let engine = Engine::start().await;
        {
            let mut s = engine.state.lock().unwrap();
            s.read_batch = true;
            s.feedback = true;
            if filtered {
                s.feedback_sequence = 10;
                s.batch_filtered_page = true;
            } else {
                for _ in 0..33 {
                    feedback_event(&mut s, "operation", json!({}));
                }
            }
        }
        let mut cfg = orchestration();
        cfg.preparation_policy = Some(Default::default());
        let router = router_with(&[&engine], false, Some(cfg)).await;
        start(&router, "batch").await;
        submit(
            &router,
            "batch",
            "prepare",
            prefix(vec![1, 2, 3, 4]),
            &engine,
        )
        .await;
        // Background discovery may already advance a page. Every accepted
        // positive result must nevertheless be backed by a caught-up batch.
        ready(&router, "batch", "prepare").await;
        let feedback = api(&router, json!({"kind":"feedback"})).await;
        assert!(feedback["owners"][0]["error"].is_null());
        assert_eq!(
            feedback["owners"][0]["cursor"]["sequence"],
            if filtered { 10 } else { 33 }
        );
    }
}

#[tokio::test]
async fn measured_policy_reuses_terminal_receipt_but_refreshes_current_residency() {
    for measured in [false, true] {
        let engine = Engine::start().await;
        let mut cfg = orchestration();
        if measured {
            cfg.preparation_policy = Some(Default::default());
        }
        let router = router_with(&[&engine], false, Some(cfg)).await;
        start(&router, "receipt").await;
        submit(
            &router,
            "receipt",
            "prepare",
            prefix(vec![1, 2, 3, 4]),
            &engine,
        )
        .await;
        ready(&router, "receipt", "prepare").await;
        let count = |kind: &str| {
            engine
                .state
                .lock()
                .unwrap()
                .commands
                .iter()
                .filter(|command| command["kind"] == kind)
                .count()
        };
        let queries = count("query");
        let residency = count("residency");
        assert!(queries > 0);
        for _ in 0..3 {
            // Cross the existing advisory freshness interval, so advance must
            // obtain a current scoped query instead of its short cached view.
            tokio::time::sleep(Duration::from_millis(60)).await;
            assert_eq!(
                query(&router, "receipt", "prepare").await["operations"][0]["ready_scope"]
                    ["end_token"],
                4
            );
        }
        assert!(count("residency") >= residency + 3);
        if measured {
            assert_eq!(count("query"), queries);
        } else {
            assert!(count("query") >= queries + 3);
        }
        engine.state.lock().unwrap().resident = 0;
        tokio::time::sleep(Duration::from_millis(60)).await;
        let absent = query(&router, "receipt", "prepare").await;
        assert_eq!(
            absent["operations"][0]["result"]["update"]["status"],
            "completed"
        );
        assert!(absent["operations"][0]["ready_scope"].is_null());
    }
}

#[tokio::test]
async fn measured_policy_keeps_querying_queued_quiescent_receipts() {
    let engine = Engine::start().await;
    engine.state.lock().unwrap().queued = true;
    let mut cfg = orchestration();
    cfg.preparation_policy = Some(Default::default());
    let router = router_with(&[&engine], false, Some(cfg)).await;
    start(&router, "queued").await;
    submit(
        &router,
        "queued",
        "prepare",
        prefix(vec![1, 2, 3, 4]),
        &engine,
    )
    .await;
    let queued = query(&router, "queued", "prepare").await;
    assert_eq!(
        queued["operations"][0]["result"]["update"]["status"],
        "queued"
    );
    assert_eq!(
        queued["operations"][0]["result"]["update"]["quiescent"],
        true
    );
    assert!(queued["operations"][0]["ready_scope"].is_null());
    let queries;
    {
        let mut state = engine.state.lock().unwrap();
        queries = state
            .commands
            .iter()
            .filter(|command| command["kind"] == "query")
            .count();
        state.queued = false;
        state.resident = 4;
        let result = state.operations.values_mut().next().unwrap();
        result["sequence"] = json!(2);
        result["update"] = json!({"status":"completed","quiescent":true});
    }
    let completed = ready(&router, "queued", "prepare").await;
    assert_eq!(completed["operations"][0]["result"]["sequence"], 2);
    assert_eq!(completed["operations"][0]["ready_scope"]["end_token"], 4);
    assert!(
        engine
            .state
            .lock()
            .unwrap()
            .commands
            .iter()
            .filter(|command| command["kind"] == "query")
            .count()
            > queries
    );
}

#[tokio::test]
async fn measured_pin_receipt_reuse_keeps_claim_queries_and_new_release_query() {
    let engine = Engine::start().await;
    let router = router_with(&[&engine], false, Some(auto_pin_config(3000, 4))).await;
    start(&router, "pin-receipt").await;
    lifecycle_prepare(&router, &engine, "pin-receipt", vec![1, 2, 3, 4]).await;
    let active = pin_view(&router, "pin-receipt", "active").await;
    let pin_id = active["operations"][0]["auto_pin"]["child_id"]
        .as_str()
        .unwrap()
        .to_owned();
    let count = |kind: &str, id: Option<&str>| {
        engine
            .state
            .lock()
            .unwrap()
            .commands
            .iter()
            .filter(|command| {
                command["kind"] == kind && id.is_none_or(|id| command["message_id"] == id)
            })
            .count()
    };
    let queries = count("query", Some(&pin_id));
    let claims = count("claim", None);
    assert!(queries > 0);
    for _ in 0..3 {
        tokio::time::sleep(Duration::from_millis(60)).await;
        assert_eq!(
            query(&router, "pin-receipt", "prepare").await["operations"][0]["auto_pin"]["state"],
            "active"
        );
    }
    assert_eq!(count("query", Some(&pin_id)), queries);
    assert!(count("claim", None) >= claims + 3);
    event(&router, "pin-receipt", "task_finished", 3, "finish").await;
    let released = pin_view(&router, "pin-receipt", "released").await;
    assert_eq!(released["quiescent"], true);
    let release_id = format!("{pin_id}-release");
    assert!(engine.state.lock().unwrap().commands.iter().any(|command| {
        command["kind"] == "submit"
            && command["envelope"]["message_id"] == release_id
            && command["envelope"]["actions"][0]["payload"]["mode"] == "release"
    }));
    assert!(count("query", Some(&release_id)) > 0);
}

#[tokio::test]
async fn lifecycle_ready_pin_shares_parent_interests_and_recovers_lost_acquire_reply() {
    let engine = Engine::start().await;
    engine.state.lock().unwrap().lose_pin_submit_once = true;
    engine.state.lock().unwrap().claims.insert(
        "foreign".into(),
        json!({"owner_epoch":"epoch-1","state":"active","revision":1}),
    );
    let router = router_with(&[&engine], false, Some(auto_pin_config(3000, 4))).await;
    start(&router, "a").await;
    start(&router, "b").await;
    lifecycle_prepare(&router, &engine, "a", vec![1, 2, 3, 4]).await;
    lifecycle_prepare(&router, &engine, "b", vec![1, 2, 3, 4]).await;
    let view = pin_view(&router, "a", "active").await;
    assert_eq!(view["operations"].as_array().unwrap().len(), 1);
    assert_eq!(view["operations"][0]["ready_scope"]["end_token"], 4);
    let child = view["operations"][0]["auto_pin"]["child_id"].clone();
    assert_eq!(
        pin_view(&router, "b", "active").await["operations"][0]["auto_pin"]["child_id"],
        child
    );
    api(
        &router,
        json!({"kind":"cancel","task":task("a"),"message_id":"prepare"}),
    )
    .await;
    api(
        &router,
        json!({"kind":"ack","task":task("a"),"message_id":"prepare"}),
    )
    .await;
    assert_eq!(
        pin_view(&router, "b", "active").await["operations"][0]["auto_pin"]["child_id"],
        child
    );
    // Sending ordinary inference does not end the preparation interest/lease.
    let response = reqwest::Client::new()
        .post(format!("{}/v1/chat/completions", router.url))
        .json(&json!({"model":MODEL,"messages":[{"role":"user","content":"keep claim"}],
            "kv_context":{"session_id":"b","incarnation":"task-1","branch_id":"main","revision":3,"branch_revision":1,"request_id":"r","request_incarnation":"one"}}))
        .send()
        .await
        .unwrap();
    assert!(response.status().is_success());
    assert_eq!(
        pin_view(&router, "b", "active").await["operations"][0]["auto_pin"]["claim"]["state"],
        "active"
    );
    event(&router, "b", "task_finished", 4, "finish").await;
    let done = pin_view(&router, "b", "released").await;
    assert_eq!(done["quiescent"], true);
    api(
        &router,
        json!({"kind":"ack","task":task("b"),"message_id":"prepare"}),
    )
    .await;
    start(&router, "c").await;
    lifecycle_prepare(&router, &engine, "c", vec![1, 2, 3, 4]).await;
    let next = pin_view(&router, "c", "active").await;
    assert_ne!(next["operations"][0]["auto_pin"]["child_id"], child);
    event(&router, "c", "task_finished", 3, "finish").await;
    pin_view(&router, "c", "released").await;
    api(
        &router,
        json!({"kind":"ack","task":task("c"),"message_id":"prepare"}),
    )
    .await;
    let state = engine.state.lock().unwrap();
    assert_eq!(state.claims.len(), 3);
    assert_eq!(state.claims["foreign"]["state"], "active");
    let ids: std::collections::BTreeSet<_> = state
        .commands
        .iter()
        .filter(|c| {
            c["kind"] == "submit"
                && c["envelope"]["actions"][0]["action_type"] == "sglang.pin"
                && c["envelope"]["actions"][0]["payload"]["mode"] == "acquire"
        })
        .map(|c| c["envelope"]["message_id"].clone().to_string())
        .collect();
    assert_eq!(ids.len(), 2);
}

#[tokio::test]
async fn lifecycle_ready_pin_budget_and_engine_refusals_keep_readiness_distinct() {
    for refusal in [None, Some("deferred"), Some("missing")] {
        let engine = Engine::start().await;
        engine.state.lock().unwrap().pin_refusal = refusal.map(str::to_owned);
        let budget = if refusal.is_none() { 3 } else { 4 };
        let router = router_with(&[&engine], false, Some(auto_pin_config(1000, budget))).await;
        start(&router, "task").await;
        lifecycle_prepare(&router, &engine, "task", vec![1, 2, 3, 4]).await;
        let view = if refusal.is_some() {
            pin_view(&router, "task", "not_acquired").await
        } else {
            ready(&router, "task", "prepare").await
        };
        if refusal.is_none() {
            assert_eq!(
                view["operations"][0]["auto_pin_error"],
                "auto PIN token capacity"
            );
            assert!(view["operations"][0]["auto_pin"].is_null());
        } else {
            let denied = pin_view(&router, "task", "not_acquired").await;
            assert_eq!(
                denied["operations"][0]["auto_pin"]["result"]["update"]["status"],
                refusal.unwrap()
            );
            assert_eq!(denied["quiescent"], true);
        }
        if refusal == Some("missing") {
            assert!(
                query(&router, "task", "prepare").await["operations"][0]["ready_scope"].is_null()
            );
        } else {
            assert_eq!(view["operations"][0]["ready_scope"]["end_token"], 4);
        }
        assert!(engine.state.lock().unwrap().claims.is_empty());
        api(
            &router,
            json!({"kind":"ack","task":task("task"),"message_id":"prepare"}),
        )
        .await;
    }
}

#[tokio::test]
async fn lifecycle_ready_pin_expiry_is_not_reacquired_by_poll_or_new_interest() {
    let engine = Engine::start().await;
    let router = router_with(&[&engine], false, Some(auto_pin_config(200, 4))).await;
    start(&router, "a").await;
    lifecycle_prepare(&router, &engine, "a", vec![1, 2, 3, 4]).await;
    pin_view(&router, "a", "active").await;
    tokio::time::sleep(Duration::from_millis(250)).await;
    let expired = pin_view(&router, "a", "expired").await;
    assert_eq!(expired["quiescent"], true);
    start(&router, "b").await;
    lifecycle_prepare(&router, &engine, "b", vec![1, 2, 3, 4]).await;
    pin_view(&router, "b", "expired").await;
    let state = engine.state.lock().unwrap();
    assert_eq!(
        state
            .commands
            .iter()
            .filter(|c| c["kind"] == "submit"
                && c["envelope"]["actions"][0]["action_type"] == "sglang.pin"
                && c["envelope"]["actions"][0]["payload"]["mode"] == "acquire")
            .count(),
        1
    );
}

#[tokio::test]
async fn lifecycle_ready_pin_charges_both_prefix_and_payload_material() {
    let engine = Engine::start().await;
    let mut cfg = auto_pin_config(1000, 4);
    // Four tokens occupy 32 logical bytes across the parent's two owned copies;
    // another 32 for its PIN child must not exceed the shared 48-byte budget.
    cfg.limits.max_token_material_bytes = 48;
    let router = router_with(&[&engine], false, Some(cfg)).await;
    start(&router, "task").await;
    lifecycle_prepare(&router, &engine, "task", vec![1, 2, 3, 4]).await;
    let view = ready(&router, "task", "prepare").await;
    assert_eq!(
        view["operations"][0]["auto_pin_error"],
        "auto PIN operation/material/claim capacity"
    );
    assert!(view["operations"][0]["auto_pin"].is_null());
    assert!(engine.state.lock().unwrap().claims.is_empty());
}

#[tokio::test]
async fn refused_explicit_pin_renewal_still_queries_the_existing_claim() {
    let engine = Engine::start().await;
    let router = router(&[&engine], false, true).await;
    start(&router, "task").await;
    for (message, mode, revision) in [("keep", "acquire", 1), ("renew", "renew", 2)] {
        if mode == "renew" {
            engine.state.lock().unwrap().pin_refusal = Some("failed".into());
        }
        api(&router,json!({"kind":"submit","task":task("task"),"message_id":message,"action":"sglang.pin","target":prefix(vec![1,2,3,4]),
            "location":{"tier":"host"},"mode":mode,"claim_id":"keep","revision":revision,"ttl_seconds":10,"required_protection":"eviction_exclusion"})).await;
        let view = query(&router, "task", message).await;
        assert_eq!(view["operations"][0]["claim"]["state"], "active");
        if mode == "renew" {
            assert_eq!(
                view["operations"][0]["result"]["update"]["status"],
                "failed"
            );
        }
    }
    engine.state.lock().unwrap().pin_refusal = None;
    event(&router, "task", "task_finished", 1, "finish").await;
    assert_eq!(
        query(&router, "task", "keep").await["operations"][0]["claim"]["state"],
        "released"
    );
}

#[tokio::test]
async fn lifecycle_ready_pin_unknown_cleanup_blocks_ack_and_retains_budget() {
    let engine = Engine::start().await;
    let router = router_with(&[&engine], false, Some(auto_pin_config(3000, 4))).await;
    start(&router, "a").await;
    lifecycle_prepare(&router, &engine, "a", vec![1, 2, 3, 4]).await;
    pin_view(&router, "a", "active").await;
    engine.state.lock().unwrap().fail_claim = true;
    let cancelled = api(
        &router,
        json!({"kind":"cancel","task":task("a"),"message_id":"prepare"}),
    )
    .await;
    assert_eq!(cancelled["quiescent"], false);
    let unknown = query(&router, "a", "prepare").await;
    assert_eq!(unknown["quiescent"], false);
    assert!(unknown["operations"][0]["auto_pin"]["error"].is_string());
    let response: Value = reqwest::Client::new()
        .post(format!("{}/v1/kv/control", router.url))
        .bearer_auth("test-control")
        .json(&json!({"kind":"ack","task":task("a"),"message_id":"prepare"}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(response["success"], false);
    {
        let mut state = engine.state.lock().unwrap();
        state.fail_claim = false;
        state.fail_pin_release = true;
    }
    assert_eq!(query(&router, "a", "prepare").await["quiescent"], false);
    start(&router, "b").await;
    lifecycle_prepare(&router, &engine, "b", vec![5, 6, 7, 8]).await;
    assert_eq!(
        ready(&router, "b", "prepare").await["operations"][0]["auto_pin_error"],
        "auto PIN token capacity"
    );
    engine.state.lock().unwrap().fail_pin_release = false;
    assert_eq!(pin_view(&router, "a", "released").await["quiescent"], true);
    api(
        &router,
        json!({"kind":"ack","task":task("a"),"message_id":"prepare"}),
    )
    .await;
}

#[tokio::test]
async fn lifecycle_host_source_is_opt_in_and_disabled_ready_pin_remains_absent() {
    for measured in [false, true] {
        let engine = Engine::start().await;
        let mut cfg = orchestration();
        if measured {
            cfg.preparation_policy = Some(Default::default());
        }
        let router = router_with(&[&engine], false, Some(cfg)).await;
        start(&router, "task").await;
        api(&router,json!({"kind":"session_event","task":task("task"),"event":"tool_wait","event_id":"cold","revision":1,
            "target":prefix(vec![1,2,3,4]),"worker_url":engine.url,"pressure":true,"expected_wait_ms":10000,"source":{"tier":"host"}})).await;
        query(&router, "task", "cold").await;
        lifecycle_prepare(&router, &engine, "task", vec![5, 6, 7, 8]).await;
        let view = ready(&router, "task", "prepare").await;
        assert!(view["operations"][0]["auto_pin"].is_null());
        let state = engine.state.lock().unwrap();
        let demote = state
            .commands
            .iter()
            .find(|c| {
                c["kind"] == "submit"
                    && c["envelope"]["actions"][0]["action_type"] == "sglang.demote"
            })
            .unwrap();
        assert_eq!(
            demote["envelope"]["actions"][0]["payload"]["source"]["tier"],
            if measured { "host" } else { "gpu" }
        );
        assert!(state.claims.is_empty());
    }
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

#[tokio::test]
async fn new_lifecycle_and_actor_episodes_use_fresh_wire_ids_after_cached_completion() {
    let engine = Engine::start().await;
    let mut logical = None;
    let mut wire_ids = std::collections::BTreeSet::new();
    for tasks in [vec!["a", "b"], vec!["c"]] {
        let router = router(&[&engine], false, true).await;
        for id in tasks {
            start(&router, id).await;
            let view = submit(&router, id, "prepare", prefix(vec![1, 2, 3, 4]), &engine).await;
            if let Some(expected) = &logical {
                assert_eq!(&view["operations"][0]["child_id"], expected);
            } else {
                logical = Some(view["operations"][0]["child_id"].clone());
            }
            ready(&router, id, "prepare").await;
            event(&router, id, "task_finished", 1, "finish").await;
            query(&router, id, "prepare").await;
            api(
                &router,
                json!({"kind":"ack","task":task(id),"message_id":"prepare"}),
            )
            .await;
            let mut state = engine.state.lock().unwrap();
            let command = state
                .commands
                .iter()
                .rev()
                .find(|c| c["kind"] == "submit")
                .unwrap();
            assert!(wire_ids.insert(
                command["envelope"]["message_id"]
                    .as_str()
                    .unwrap()
                    .to_owned()
            ));
            // Engine retains its COMPLETED record while the physical copy is
            // gone. The next episode must perform a new preparation.
            state.resident = 0;
        }
    }
    assert_eq!(wire_ids.len(), 3);
    assert_eq!(engine.state.lock().unwrap().operations.len(), 3);
}

#[tokio::test]
async fn measured_lifecycle_keeps_host_copy_when_resumption_is_close() {
    let engine = Engine::start().await;
    let mut cfg = orchestration();
    cfg.preparation_policy = Some(Default::default());
    let router = router_with(&[&engine], false, Some(cfg)).await;
    start(&router, "near").await;
    let near = api(&router,json!({"kind":"session_event","task":task("near"),"event":"tool_wait","event_id":"near","revision":1,
        "target":prefix(vec![1,2,3,4]),"worker_url":engine.url,"pressure":true,"expected_wait_ms":100,"source":{"tier":"host"}})).await;
    assert_eq!(near["state"], "skipped");
    assert_eq!(near["reason"], "insufficient expected cold residency");
    assert!(engine
        .state
        .lock()
        .unwrap()
        .commands
        .iter()
        .all(|command| { command["envelope"]["actions"][0]["action_type"] != "sglang.demote" }));
    let distant = api(&router,json!({"kind":"session_event","task":task("near"),"event":"tool_wait","event_id":"distant","revision":2,
        "target":prefix(vec![1,2,3,4]),"worker_url":engine.url,"pressure":true,"expected_wait_ms":10000,"source":{"tier":"host"}})).await;
    assert_eq!(distant["operations"].as_array().unwrap().len(), 1);
    query(&router, "near", "distant").await;
    assert!(engine
        .state
        .lock()
        .unwrap()
        .commands
        .iter()
        .any(|command| { command["envelope"]["actions"][0]["action_type"] == "sglang.demote" }));
}

#[tokio::test]
async fn measured_completed_observation_is_reused_only_within_its_poll_window() {
    let engine = Engine::start().await;
    let mut cfg = orchestration();
    cfg.poll_ms = 1000;
    cfg.preparation_policy = Some(Default::default());
    let router = router_with(&[&engine], false, Some(cfg)).await;
    start(&router, "observed").await;
    submit(
        &router,
        "observed",
        "prepare",
        prefix(vec![1, 2, 3, 4]),
        &engine,
    )
    .await;
    ready(&router, "observed", "prepare").await;
    let count = || {
        engine
            .state
            .lock()
            .unwrap()
            .commands
            .iter()
            .filter(|command| command["kind"] == "residency")
            .count()
    };
    let initial = count();
    for _ in 0..5 {
        let view = query(&router, "observed", "prepare").await;
        assert_eq!(view["operations"][0]["ready_scope"]["end_token"], 4);
        assert!(view["operations"][0]["observation_age_ms"].is_number());
    }
    assert_eq!(count(), initial);
    engine.state.lock().unwrap().fail_residency = true;
    tokio::time::sleep(Duration::from_millis(1100)).await;
    let view = query(&router, "observed", "prepare").await;
    assert!(view["operations"][0]["ready_scope"].is_null());
    assert!(count() > initial);
    api(
        &router,
        json!({"kind":"cancel","task":task("observed"),"message_id":"prepare"}),
    )
    .await;
    assert_eq!(
        query(&router, "observed", "prepare").await["quiescent"],
        true
    );
}
