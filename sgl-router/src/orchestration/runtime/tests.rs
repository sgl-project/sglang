use super::*;
use crate::discovery::{WorkerId, WorkerSpec};

fn capabilities() -> Value {
    json!({"engine_id":"engine","engine_epoch":"epoch","namespace":"weights",
        "public_protocol":{"actions":{"sglang.prefetch":["1.0"]},
        "paths":{"sglang.prefetch":[["storage","host"]]},"storage_backend":"mooncake","backend_namespace":"pool",
        "prefix":{"layout_id":"full:p1","page_size":1,"components":["full"],"hash_scheme":"sglang.sha256.u32le.i64.v1"}}})
}

fn actor(url: &str) -> Actor {
    let config = serde_json::from_value(json!({"caller_id":"test","control_bearer":"test",
        "queue_capacity":32,"max_operations":64,"max_workers":4,"max_engine_bytes":1048576,"rpc_timeout_ms":1000,
        "poll_ms":50,"operation_ttl_ms":10000,"initial_prefetch_ms_per_token":1.0,"lead_margin_ms":20,
        "max_resume_prefetch_ms":100,"preparation_policy":{},
        "limits":{"max_sessions":32,"max_branches":32,"max_history_revisions":64,"max_requests":64,
        "max_content_nodes":1024,"max_token_material_bytes":1048576,"max_tombstones":32,"max_claims":32,
        "max_intents":64,"max_identifier_bytes":128,"max_components":8}})).unwrap();
    let registry = Arc::new(WorkerRegistry::default());
    registry
        .add(WorkerSpec {
            id: WorkerId("engine".into()),
            url: url.into(),
            model_ids: vec![ModelId("model".into())],
            ..Default::default()
        })
        .unwrap();
    let mut actor = Actor::new(
        config,
        registry,
        ModelId("model".into()),
        None,
        Arc::new(RwLock::new(Published::default())),
        Arc::new(AtomicU64::new(0)),
    );
    actor.bind_caps(url, Capabilities::parse(capabilities()).unwrap());
    actor
}

async fn pending(actor: &mut Actor, token: u32) -> (String, DemandId) {
    let url = actor.workers()[0].url.clone();
    let task = SessionIdentity {
        caller_id: "test".into(),
        session_id: format!("task-{token}"),
        incarnation: "one".into(),
    };
    actor.index.start_session(task.clone(), None).unwrap();
    let demand = DemandId {
        task: task.clone(),
        id: "prepare".into(),
    };
    let prefix = PrefixTarget {
        kind: "prefix".into(),
        compatibility: actor.caps[&url].compatibility.clone(),
        start_token: 0,
        end_token: 1,
        token_ids: Some(vec![token]),
        blocks: vec![],
    };
    let reply = actor.submit(json!({"action":"sglang.prefetch","event":"resume_expected","target":prefix,
        "expected_wait_ms":10000,"source":{"tier":"storage"},"destination":{"tier":"host"},"worker_url":url}),
        task, demand.clone(), Some(Instant::now()+Duration::from_secs(10)), true).await.unwrap();
    let id = reply["operations"][0]["child_id"]
        .as_str()
        .unwrap()
        .to_string();
    actor.jobs.get_mut(&id).unwrap().due = Instant::now() - Duration::from_secs(1);
    (id, demand)
}

fn healthy(actor: &Actor) -> BTreeSet<String> {
    actor.workers().iter().map(|w| w.url.clone()).collect()
}

#[tokio::test]
async fn due_priority_selects_earliest_and_allows_reopened_quiescent_preparation() {
    let mut actor = actor("http://127.0.0.1:1");
    let (delayed, _) = pending(&mut actor, 0).await;
    let (terminal, _) = pending(&mut actor, 1).await;
    let (earlier, _) = pending(&mut actor, 2).await;
    let (later, _) = pending(&mut actor, 3).await;
    let (reopened, _) = pending(&mut actor, 4).await;
    let now = Instant::now();
    actor.jobs.get_mut(&delayed).unwrap().due = now + Duration::from_secs(1);
    let job = actor.jobs.get_mut(&terminal).unwrap();
    job.attempted = true;
    job.submitted = true;
    job.quiescent = true;
    job.due = now - Duration::from_secs(4);
    actor.jobs.get_mut(&earlier).unwrap().due = now - Duration::from_secs(2);
    actor.jobs.get_mut(&later).unwrap().due = now - Duration::from_secs(1);
    let job = actor.jobs.get_mut(&reopened).unwrap();
    job.due = now - Duration::from_secs(3);
    job.quiescent = true;
    job.late_before_submit = true;
    actor.book.mark_quiescent(&reopened).unwrap();
    let healthy = healthy(&actor);
    assert_eq!(actor.next_job(&healthy), Some((reopened.clone(), true)));
    actor.jobs.get_mut(&reopened).unwrap().attempted = true;
    assert!(!actor.next_job(&healthy).unwrap().1);
    assert_eq!(actor.next_job(&healthy), Some((earlier, true)));
}

#[tokio::test]
async fn due_priority_excludes_work_without_live_unattempted_host_demand() {
    for mode in [
        "future",
        "window",
        "expired",
        "released",
        "unhealthy",
        "cancelled",
        "attempted",
        "submitted",
        "demote",
        "host_source",
        "gpu_destination",
    ] {
        let mut actor = actor("http://127.0.0.1:1");
        let (id, demand) = pending(&mut actor, 0).await;
        let mut healthy = healthy(&actor);
        let past = Instant::now() - Duration::from_secs(1);
        match mode {
            "future" => {
                actor.jobs.get_mut(&id).unwrap().due = Instant::now() + Duration::from_secs(1)
            }
            "window" => actor.intents.get_mut(&demand).unwrap().resume_at = Some(past),
            "expired" => actor.intents.get_mut(&demand).unwrap().expires = past,
            "released" => actor.intents.get_mut(&demand).unwrap().released = true,
            "unhealthy" => healthy.clear(),
            "cancelled" => {
                actor.release(&demand);
                assert!(actor.book.job(&id).unwrap().cancel_requested);
            }
            "attempted" => actor.jobs.get_mut(&id).unwrap().attempted = true,
            "submitted" => actor.jobs.get_mut(&id).unwrap().submitted = true,
            "demote" => actor.jobs.get_mut(&id).unwrap().action = "sglang.demote".into(),
            "host_source" => {
                actor.jobs.get_mut(&id).unwrap().payload["source"]["tier"] = json!("host")
            }
            "gpu_destination" => {
                actor.jobs.get_mut(&id).unwrap().payload["destination"]["tier"] = json!("gpu")
            }
            _ => unreachable!(),
        }
        assert_eq!(actor.next_job(&healthy), Some((id, false)), "{mode}");
    }
}

#[tokio::test]
async fn due_priority_round_robin_covers_even_and_nine_job_sets_under_continuous_pressure() {
    for count in [6, 9] {
        let mut actor = actor("http://127.0.0.1:1");
        let mut entries = Vec::new();
        for token in 0..count {
            entries.push(pending(&mut actor, token).await);
        }
        for (id, _) in &entries[..entries.len() - 1] {
            let job = actor.jobs.get_mut(id).unwrap();
            job.attempted = true;
            job.submitted = true;
        }
        actor.release(&entries[0].1);
        actor.jobs.get_mut(&entries[1].0).unwrap().submitted = false; // Lost reply.
        let pin = actor.jobs.get_mut(&entries[2].0).unwrap();
        pin.action = "sglang.pin".into();
        pin.deadline = Instant::now() - Duration::from_secs(1);
        let urgent = entries.last().unwrap().0.clone();
        let expected: Vec<_> = actor.jobs.keys().cloned().collect();
        let healthy = healthy(&actor);
        let mut visited = Vec::new();
        // A failure before Engine submit can leave the same job eligible on
        // every turn. Cancellation, PIN expiry and lost replies still get RR.
        for turn in 0..2 * count {
            actor.tick_cursor += 1;
            let (id, priority) = actor.next_job(&healthy).unwrap();
            if turn % 2 == 0 {
                assert!(priority);
                assert_eq!(id, urgent);
            } else {
                assert!(!priority);
                visited.push(id);
            }
        }
        assert_eq!(visited, expected);
        assert_eq!(actor.job_rr_cursor, count as usize);
        actor.jobs.clear();
        assert!(actor.next_job(&healthy).is_none());
        assert_eq!(actor.job_rr_cursor, count as usize);
    }
}

#[tokio::test]
async fn legacy_job_round_robin_keeps_the_original_tick_cursor() {
    let mut actor = actor("http://127.0.0.1:1");
    for token in 0..6 {
        pending(&mut actor, token).await;
    }
    actor.preparation_costs = None;
    let expected: Vec<_> = actor.jobs.keys().cloned().collect();
    let healthy = healthy(&actor);
    for tick in 1..=12 {
        actor.tick_cursor = tick;
        assert_eq!(
            actor.next_job(&healthy),
            Some((expected[tick % 6].clone(), false))
        );
    }
    assert_eq!(actor.job_rr_cursor, 0);
}

#[tokio::test]
async fn due_priority_polls_one_job_before_the_standalone_worker_probe() {
    use axum::{
        extract::State,
        http::StatusCode,
        routing::{get, post},
        Json, Router,
    };
    use std::sync::Mutex;
    type Calls = Arc<Mutex<Vec<String>>>;
    async fn caps(State(calls): State<Calls>) -> Json<Value> {
        calls.lock().unwrap().push("capabilities".into());
        Json(json!({"success":true,"data":capabilities()}))
    }
    async fn fail_submit(
        State(calls): State<Calls>,
        Json(body): Json<Value>,
    ) -> (StatusCode, Json<Value>) {
        calls
            .lock()
            .unwrap()
            .push(body["kind"].as_str().unwrap().into());
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"success":false,"error":"injected before receipt"})),
        )
    }
    for measured in [true, false] {
        let calls = Calls::default();
        let app = Router::new()
            .route("/v1/kv/capabilities", get(caps))
            .route("/v1/kv/control", post(fail_submit))
            .with_state(calls.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        let mut actor = actor(&url);
        let (first, _) = pending(&mut actor, 0).await;
        pending(&mut actor, 1).await;
        actor.jobs.get_mut(&first).unwrap().due -= Duration::from_secs(1);
        if !measured {
            actor.preparation_costs = None;
        }
        actor.pump().await;
        let expected = if measured {
            vec!["capabilities", "submit", "capabilities"]
        } else {
            vec!["capabilities", "capabilities", "submit"]
        };
        assert_eq!(*calls.lock().unwrap(), expected);
        assert_eq!(actor.jobs.values().filter(|job| job.attempted).count(), 1);
        server.abort();
    }
}
