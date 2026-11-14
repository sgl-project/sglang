// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Engine endpoints answered by one worker: `/server_info`, `/model_info`
//! (and their `/get_` aliases) and `/health_generate`, as SGLang engines
//! serve them. The worker is a healthy one that chat can route to (under P/D
//! a prefill with a decode peer), picked at random. The worker's status and
//! body pass through; only a transport failure becomes a router error.

use crate::buckets_reorg::ReadyPools;
use crate::discovery::{ModelId, WorkerMode};
use crate::policies::registry::PdPoolResolver;
use crate::server::app_context::{AppContext, ChatRouting};
use crate::server::error::ApiError;
use crate::server::routes::chat::pool_error;
use crate::workers::worker::Worker;
use axum::extract::State;
use axum::http::header::{AUTHORIZATION, CONTENT_TYPE};
use axum::http::HeaderMap;
use axum::response::{IntoResponse, Response};
use rand::seq::{IteratorRandom, SliceRandom};
use std::sync::Arc;

pub async fn server_info(State(ctx): State<Arc<AppContext>>, headers: HeaderMap) -> Response {
    forward_to_first_stage(&ctx, &headers, "/server_info").await
}

pub async fn model_info(State(ctx): State<Arc<AppContext>>, headers: HeaderMap) -> Response {
    forward_to_first_stage(&ctx, &headers, "/model_info").await
}

/// Runs a generation on one plain worker, or on one prefill worker and one of
/// its decode peers at once, and answers with the prefill's failure if any,
/// else with the decode's response.
pub async fn health_generate(State(ctx): State<Arc<AppContext>>, headers: HeaderMap) -> Response {
    const PATH: &str = "/health_generate";
    let (first, decode) = match select(&ctx) {
        Ok(selected) => selected,
        Err(e) => return e.into_response(),
    };
    let Some(decode) = decode else {
        return forward(&ctx, &headers, &first, PATH)
            .await
            .unwrap_or_else(IntoResponse::into_response);
    };
    let (prefill, decode) = tokio::join!(
        forward(&ctx, &headers, &first, PATH),
        forward(&ctx, &headers, &decode, PATH),
    );
    match (prefill, decode) {
        (Err(e), _) => e.into_response(),
        (Ok(prefill), _) if !prefill.status().is_success() => prefill,
        (Ok(_), decode) => decode.unwrap_or_else(IntoResponse::into_response),
    }
}

/// Forwards to a plain worker, or a prefill worker under P/D.
async fn forward_to_first_stage(ctx: &AppContext, headers: &HeaderMap, path: &str) -> Response {
    let model = ModelId(ctx.config.model.id.clone());
    match pools(ctx, &model) {
        Ok(pools) => forward(ctx, headers, &pick(&pools.firsts), path)
            .await
            .unwrap_or_else(IntoResponse::into_response),
        Err(e) => e.into_response(),
    }
}

/// One plain worker, or one prefill worker and one of its decode peers.
fn select(ctx: &AppContext) -> Result<(Arc<Worker>, Option<Arc<Worker>>), ApiError> {
    let model = ModelId(ctx.config.model.id.clone());
    let ReadyPools { firsts, decoders } = pools(ctx, &model)?;
    let first = pick(&firsts);
    if first.mode() != WorkerMode::Prefill {
        return Ok((first, None));
    }
    let peers: Vec<_> = decoders
        .into_iter()
        .filter(|d| d.version_group() == first.version_group())
        .collect();
    if peers.is_empty() {
        return Err(ApiError::NoDecodeWorkersAvailable { model: model.0 });
    }
    Ok((first, Some(pick(&peers))))
}

/// The pools chat routes over. Under bucket routing, those of one bucket with
/// ready workers, so a pair never spans two buckets; with none ready the
/// answer is `no_healthy_workers`, as `/readyz` sees it, without the per-stage
/// detail.
fn pools(ctx: &AppContext, model: &ModelId) -> Result<ReadyPools, ApiError> {
    match &ctx.chat_routing {
        ChatRouting::Legacy => {
            let resolver = PdPoolResolver::new(Arc::clone(&ctx.registry));
            let firsts = resolver
                .prefill_candidates(model)
                .map_err(|e| pool_error(e, model))?;
            // Decoders matter only to `/health_generate`, which reports their absence.
            let decoders = resolver
                .decode_candidates(model)
                .unwrap_or_default()
                .into_iter()
                .filter(|w| w.mode() == WorkerMode::Decode)
                .collect();
            Ok(ReadyPools { firsts, decoders })
        }
        ChatRouting::Reorg(resolvers) => resolvers
            .get(model)
            .into_iter()
            .flat_map(|resolver| &resolver.buckets)
            .map(|bucket| bucket.ready_pools(&ctx.registry, model))
            .filter(|pools| !pools.firsts.is_empty())
            .choose(&mut rand::thread_rng())
            .ok_or_else(|| ApiError::NoHealthyWorkers {
                model: model.0.clone(),
            }),
    }
}

fn pick(candidates: &[Arc<Worker>]) -> Arc<Worker> {
    Arc::clone(
        candidates
            .choose(&mut rand::thread_rng())
            .expect("callers pass a non-empty pool"),
    )
}

async fn forward(
    ctx: &AppContext,
    headers: &HeaderMap,
    worker: &Worker,
    path: &str,
) -> Result<Response, ApiError> {
    let url = reqwest::Url::parse(&worker.url)
        .and_then(|base| base.join(path))
        .map_err(|e| ApiError::WorkerMisconfigured {
            worker: worker.url.clone(),
            source: e.into(),
        })?;
    let mut request = ctx
        .proxy
        .admin_client()
        .get(url.clone())
        .timeout(ctx.proxy.request_timeout);
    if let Some(auth) = headers.get(AUTHORIZATION) {
        request = request.header(AUTHORIZATION, auth);
    }
    let upstream = request.send().await.map_err(|e| {
        if e.is_timeout() {
            ApiError::UpstreamTimeout {
                worker: url.clone(),
            }
        } else {
            ApiError::UpstreamUnreachable {
                worker: url.clone(),
                source: e.into(),
            }
        }
    })?;
    let status = upstream.status();
    let content_type = upstream.headers().get(CONTENT_TYPE).cloned();
    let body = upstream
        .bytes()
        .await
        .map_err(|_| ApiError::UpstreamStatus { status })?;
    let mut response = (status, body).into_response();
    if let Some(content_type) = content_type {
        response.headers_mut().insert(CONTENT_TYPE, content_type);
    }
    Ok(response)
}

#[cfg(test)]
mod tests {
    use crate::buckets_reorg::{Bucket, BucketGroups, BucketResolver, EngineGroup};
    use crate::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
    use crate::health::circuit_breaker::CircuitBreakerConfig;
    use crate::policies_reorg::power_of_two::PowerOfTwoPolicy;
    use crate::policies_reorg::Policy;
    use crate::server::app_context::AppContext;
    use crate::server::app_context::ChatRouting;
    use crate::state::load_monitor::engine_reported_load::EngineReportedLoadTable;
    use crate::workers::worker::WireProtocol;
    use axum::body::Body;
    use axum::http::header::{AUTHORIZATION, CONTENT_TYPE};
    use axum::http::{HeaderMap, Request, StatusCode};
    use axum::routing::get;
    use axum::Router;
    use http_body_util::BodyExt;
    use std::sync::Arc;
    use tokio::net::TcpListener;
    use tower::ServiceExt;

    /// A fake engine whose `/server_info`, `/model_info` and
    /// `/health_generate` answer `status` with a body naming `name` and the
    /// caller's `Authorization` header.
    async fn spawn_engine(name: &'static str, status: StatusCode) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let answer = move |path: &'static str| {
            move |headers: HeaderMap| async move {
                let auth = headers
                    .get(AUTHORIZATION)
                    .map(|v| v.to_str().unwrap().to_string());
                (
                    status,
                    axum::Json(serde_json::json!({"worker": name, "path": path, "auth": auth})),
                )
            }
        };
        let app = Router::new()
            .route("/server_info", get(answer("/server_info")))
            .route("/model_info", get(answer("/model_info")))
            .route("/health_generate", get(answer("/health_generate")));
        tokio::spawn(async move { axum::serve(listener, app).await });
        format!("http://127.0.0.1:{port}")
    }

    fn ctx_with(workers: &[(&str, WorkerMode)]) -> Arc<AppContext> {
        let ctx = AppContext::stub();
        for (i, (url, mode)) in workers.iter().enumerate() {
            ctx.registry
                .add(WorkerSpec {
                    id: WorkerId(format!("w-{i}")),
                    url: (*url).to_string(),
                    mode: *mode,
                    bootstrap_port: (*mode == WorkerMode::Prefill).then_some(8998),
                    model_ids: vec![ModelId("stub-model".into())],
                    ..Default::default()
                })
                .expect("worker accepted");
        }
        Arc::new(ctx)
    }

    async fn get_json(ctx: Arc<AppContext>, uri: &str) -> (StatusCode, serde_json::Value) {
        let req = Request::builder()
            .uri(uri)
            .header(AUTHORIZATION, "Bearer k")
            .body(Body::empty())
            .unwrap();
        let res = crate::server::app::build_router(ctx)
            .oneshot(req)
            .await
            .unwrap();
        let status = res.status();
        assert_eq!(res.headers()[CONTENT_TYPE], "application/json");
        let body = res.into_body().collect().await.unwrap().to_bytes();
        (status, serde_json::from_slice(&body).unwrap())
    }

    #[tokio::test]
    async fn info_paths_forward_to_a_worker_with_the_callers_auth() {
        let plain = spawn_engine("plain", StatusCode::OK).await;
        let ctx = ctx_with(&[(&plain, WorkerMode::Plain)]);
        for (uri, path) in [
            ("/server_info", "/server_info"),
            ("/get_server_info", "/server_info"),
            ("/model_info", "/model_info"),
            ("/get_model_info", "/model_info"),
            ("/health_generate", "/health_generate"),
        ] {
            let (status, body) = get_json(Arc::clone(&ctx), uri).await;
            assert_eq!(status, StatusCode::OK, "{uri}");
            assert_eq!(body["path"], path, "{uri}");
            assert_eq!(body["auth"], "Bearer k", "{uri}");
        }
    }

    #[tokio::test]
    async fn info_paths_use_a_prefill_worker_under_pd() {
        let prefill = spawn_engine("prefill", StatusCode::OK).await;
        let decode = spawn_engine("decode", StatusCode::OK).await;
        let ctx = ctx_with(&[
            (&decode, WorkerMode::Decode),
            (&prefill, WorkerMode::Prefill),
        ]);
        let (_, body) = get_json(ctx, "/server_info").await;
        assert_eq!(body["worker"], "prefill");
    }

    #[tokio::test]
    async fn pd_health_generate_fails_when_either_side_fails() {
        let ok_prefill = spawn_engine("prefill", StatusCode::OK).await;
        let bad_prefill = spawn_engine("prefill", StatusCode::SERVICE_UNAVAILABLE).await;
        let ok_decode = spawn_engine("decode", StatusCode::OK).await;
        let bad_decode = spawn_engine("decode", StatusCode::SERVICE_UNAVAILABLE).await;
        for (prefill, decode, want_status, want_worker) in [
            (&ok_prefill, &ok_decode, StatusCode::OK, "decode"),
            (
                &bad_prefill,
                &ok_decode,
                StatusCode::SERVICE_UNAVAILABLE,
                "prefill",
            ),
            (
                &ok_prefill,
                &bad_decode,
                StatusCode::SERVICE_UNAVAILABLE,
                "decode",
            ),
        ] {
            let ctx = ctx_with(&[
                (prefill.as_str(), WorkerMode::Prefill),
                (decode.as_str(), WorkerMode::Decode),
            ]);
            let (status, body) = get_json(ctx, "/health_generate").await;
            assert_eq!(status, want_status, "{prefill} {decode}");
            assert_eq!(body["worker"], want_worker, "{prefill} {decode}");
        }
    }

    #[tokio::test]
    async fn empty_fleet_is_a_503() {
        let (status, body) = get_json(ctx_with(&[]), "/server_info").await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(body["error"]["code"], "no_healthy_workers");
    }

    /// A worker whose circuit breaker is open is skipped while a healthy one
    /// remains, so a probe does not flap between the two.
    #[tokio::test]
    async fn worker_with_open_breaker_is_skipped() {
        let down = spawn_engine("down", StatusCode::SERVICE_UNAVAILABLE).await;
        let up = spawn_engine("up", StatusCode::OK).await;
        let ctx = AppContext::stub();
        for (id, url) in [("down", &down), ("up", &up)] {
            ctx.registry
                .add_with_cb(
                    WorkerSpec {
                        id: WorkerId(id.into()),
                        url: url.clone(),
                        model_ids: vec![ModelId("stub-model".into())],
                        ..Default::default()
                    },
                    Some(CircuitBreakerConfig::default()),
                    WireProtocol::default(),
                )
                .expect("worker accepted");
        }
        let breaker = &ctx.registry.get(&WorkerId("down".into())).unwrap().breaker;
        while breaker.allow() {
            breaker.record_failure();
        }
        let ctx = Arc::new(ctx);
        for _ in 0..32 {
            let (status, body) = get_json(Arc::clone(&ctx), "/health_generate").await;
            assert_eq!(status, StatusCode::OK);
            assert_eq!(body["worker"], "up");
        }
    }

    /// Under bucket routing only bucket members answer, as chat would route;
    /// a fleet with no ready bucket is a 503.
    #[tokio::test]
    async fn bucket_routing_answers_from_bucket_members_only() {
        let inside = spawn_engine("inside", StatusCode::OK).await;
        let outside = spawn_engine("outside", StatusCode::OK).await;
        let mut ctx = Arc::into_inner(ctx_with(&[
            (&inside, WorkerMode::Plain),
            (&outside, WorkerMode::Plain),
        ]))
        .unwrap();
        let bucket_of = |id: &str| {
            let policy: Arc<dyn Policy> =
                Arc::new(PowerOfTwoPolicy::new(EngineReportedLoadTable::new()));
            let group = EngineGroup {
                worker_ids: Some([WorkerId(id.into())].into()),
                ..EngineGroup::new(policy)
            };
            let resolver =
                BucketResolver::new(vec![Bucket::new("b", BucketGroups::Plain(group))]).unwrap();
            ChatRouting::Reorg([(ModelId("stub-model".into()), resolver)].into())
        };
        ctx.chat_routing = bucket_of("w-0");
        let ctx = Arc::new(ctx);
        for _ in 0..32 {
            let (_, body) = get_json(Arc::clone(&ctx), "/server_info").await;
            assert_eq!(body["worker"], "inside");
        }

        let mut ctx = Arc::into_inner(ctx_with(&[(&outside, WorkerMode::Plain)])).unwrap();
        ctx.chat_routing = bucket_of("absent");
        let (status, body) = get_json(Arc::new(ctx), "/health_generate").await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(body["error"]["code"], "no_healthy_workers");
    }

    #[tokio::test]
    async fn unreachable_worker_is_a_502() {
        let port = std::net::TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        let url = format!("http://127.0.0.1:{port}");
        let ctx = ctx_with(&[(&url, WorkerMode::Plain)]);
        let (status, body) = get_json(ctx, "/model_info").await;
        assert_eq!(status, StatusCode::BAD_GATEWAY);
        assert_eq!(body["error"]["code"], "upstream_unreachable");
    }
}
