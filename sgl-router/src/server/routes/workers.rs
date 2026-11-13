// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::discovery::{ModelId, WorkerId, WorkerMode};
use crate::server::app_context::AppContext;
use crate::server::error::ApiError;
use crate::workers::control::{
    AcceptedOperation, WorkerControlError, WorkerControlService, WorkerInput, WorkerListFilter,
    WorkerSource, WorkerView,
};
use axum::extract::rejection::{JsonRejection, QueryRejection};
use axum::extract::{Path, Query, State};
use axum::http::header::{IF_MATCH, LOCATION};
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Deserialize;
use std::sync::Arc;

#[derive(Debug, Default, Deserialize)]
pub struct WorkerListQuery {
    model_id: Option<String>,
    mode: Option<WorkerMode>,
    source: Option<WorkerSource>,
}

pub async fn create_worker(
    State(context): State<Arc<AppContext>>,
    input: Result<Json<WorkerInput>, JsonRejection>,
) -> Result<Response, ApiError> {
    let accepted = control(&context)?.create(worker_input(input)?)?;
    let location = HeaderValue::from_str(&format!("/workers/{}", accepted.worker_id))
        .map_err(|error| ApiError::Internal(error.into()))?;
    Ok(accepted_response(accepted, Some(location)))
}

pub async fn list_workers(
    State(context): State<Arc<AppContext>>,
    query: Result<Query<WorkerListQuery>, QueryRejection>,
) -> Result<Json<Vec<WorkerView>>, ApiError> {
    let Query(query) =
        query.map_err(|error| ApiError::BadRequest(format!("invalid worker filter: {error}")))?;
    let filter = WorkerListFilter {
        model_id: query.model_id.map(ModelId),
        mode: query.mode,
        source: query.source,
    };
    Ok(Json(control(&context)?.list(&filter)))
}

pub async fn get_worker(
    State(context): State<Arc<AppContext>>,
    Path(id): Path<String>,
) -> Result<Json<WorkerView>, ApiError> {
    let id = WorkerId(id);
    control(&context)?
        .get(&id)
        .map(Json)
        .ok_or_else(|| WorkerControlError::NotFound(id).into())
}

pub async fn replace_worker(
    State(context): State<Arc<AppContext>>,
    Path(id): Path<String>,
    headers: HeaderMap,
    input: Result<Json<WorkerInput>, JsonRejection>,
) -> Result<Response, ApiError> {
    let accepted = control(&context)?.replace(
        &WorkerId(id),
        worker_input(input)?,
        expected_revision(&headers)?,
    )?;
    Ok(accepted_response(accepted, None))
}

pub async fn remove_worker(
    State(context): State<Arc<AppContext>>,
    Path(id): Path<String>,
    headers: HeaderMap,
) -> Result<Response, ApiError> {
    let accepted = control(&context)?.remove(&WorkerId(id), expected_revision(&headers)?)?;
    Ok(accepted_response(accepted, None))
}

fn control(context: &AppContext) -> Result<&Arc<WorkerControlService>, ApiError> {
    context
        .worker_control
        .as_ref()
        .ok_or_else(|| WorkerControlError::Unavailable.into())
}

fn worker_input(
    input: Result<Json<WorkerInput>, JsonRejection>,
) -> Result<WorkerInput, WorkerControlError> {
    input.map(|Json(input)| input).map_err(|error| {
        WorkerControlError::Invalid(format!("invalid worker specification: {error}"))
    })
}

fn expected_revision(headers: &HeaderMap) -> Result<u64, WorkerControlError> {
    let raw = headers
        .get(IF_MATCH)
        .ok_or(WorkerControlError::MissingRevision)?
        .to_str()
        .map_err(|_| WorkerControlError::InvalidRevision)?;
    raw.trim_matches('"')
        .parse()
        .map_err(|_| WorkerControlError::InvalidRevision)
}

fn accepted_response(accepted: AcceptedOperation, location: Option<HeaderValue>) -> Response {
    let mut response = (StatusCode::ACCEPTED, Json(accepted)).into_response();
    if let Some(location) = location {
        response.headers_mut().insert(LOCATION, location);
    }
    response
}

#[cfg(test)]
mod tests {
    use crate::discovery::{DiscoveryEvent, ModelId, WorkerId, WorkerMode, WorkerSpec};
    use crate::server::app::build_router;
    use crate::server::app_context::AppContext;
    use crate::workers::control::{ExternalWorkerSource, OperationState, WorkerControlService};
    use axum::body::Body;
    use axum::http::{Method, Request, StatusCode};
    use http_body_util::BodyExt;
    use serde_json::{json, Value};
    use std::sync::Arc;
    use tower::ServiceExt;

    async fn spawn_fake_worker() -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let app = axum::Router::new()
            .route(
                "/model_info",
                axum::routing::get(|| async { axum::Json(json!({"served_model_name":"m"})) }),
            )
            .route(
                "/server_info",
                axum::routing::get(|| async { axum::Json(json!({"served_model_name":"m"})) }),
            );
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        format!("http://127.0.0.1:{port}")
    }

    async fn spawn_slow_fake_worker() -> (String, Arc<tokio::sync::Notify>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let started = Arc::new(tokio::sync::Notify::new());
        let route = |started: Arc<tokio::sync::Notify>| {
            axum::routing::get(move || {
                let started = Arc::clone(&started);
                async move {
                    started.notify_one();
                    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                    axum::Json(json!({"served_model_name":"m"}))
                }
            })
        };
        let app = axum::Router::new()
            .route("/model_info", route(Arc::clone(&started)))
            .route("/server_info", route(Arc::clone(&started)));
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        (format!("http://127.0.0.1:{port}"), started)
    }

    async fn body_json(response: axum::response::Response) -> Value {
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        serde_json::from_slice(&bytes).unwrap()
    }

    fn context_with_control() -> (
        Arc<AppContext>,
        Arc<WorkerControlService>,
        tokio::sync::mpsc::Receiver<crate::workers::control::WorkerControlCommand>,
    ) {
        let mut context = AppContext::stub();
        let (control, rx) =
            WorkerControlService::new(Arc::clone(&context.registry), ExternalWorkerSource::Static);
        context.worker_control = Some(Arc::clone(&control));
        (Arc::new(context), control, rx)
    }

    fn json_request(method: Method, uri: &str, body: Value) -> Request<Body> {
        Request::builder()
            .method(method)
            .uri(uri)
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap()
    }

    #[tokio::test]
    async fn create_and_get_worker_through_public_router() {
        let (context, _control, _control_rx) = context_with_control();
        let app = build_router(context);
        let response = app
            .clone()
            .oneshot(json_request(
                Method::POST,
                "/workers",
                json!({"url":"http://worker-1:30000","mode":"plain"}),
            ))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::ACCEPTED);
        let location = response
            .headers()
            .get("location")
            .unwrap()
            .to_str()
            .unwrap()
            .to_owned();
        let accepted = body_json(response).await;
        let worker_id = accepted["worker_id"].as_str().unwrap();
        assert_eq!(location, format!("/workers/{worker_id}"));
        assert_eq!(accepted["revision"], 1);

        let response = app
            .oneshot(
                Request::builder()
                    .uri(format!("/workers/{worker_id}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let worker = body_json(response).await;
        assert_eq!(worker["source"], "api");
        assert_eq!(worker["operation"]["state"], "pending");
    }

    #[tokio::test]
    async fn replace_requires_current_if_match_revision() {
        let (context, control, _control_rx) = context_with_control();
        let created = control
            .create(crate::workers::control::WorkerInput {
                url: "http://worker-1:30000".into(),
                mode: WorkerMode::Plain,
                model_ids: vec![ModelId("m".into())],
                bootstrap_port: None,
                version_group: None,
                services: Default::default(),
            })
            .unwrap();
        control.mark_succeeded(&created.operation_id);
        let response = build_router(context)
            .oneshot(
                json_request(
                    Method::PUT,
                    &format!("/workers/{}", created.worker_id),
                    json!({"url":"http://worker-1:30000","mode":"decode"}),
                )
                .map(|body| body),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert_eq!(
            response.headers()["x-router-error-code"],
            "missing_worker_revision"
        );

        let request = Request::builder()
            .method(Method::PUT)
            .uri(format!("/workers/{}", created.worker_id))
            .header("content-type", "application/json")
            .header("if-match", "0")
            .body(Body::from(
                json!({"url":"http://worker-1:30000","mode":"decode"}).to_string(),
            ))
            .unwrap();
        let response = build_router(Arc::new({
            let mut context = AppContext::stub();
            context.worker_control = Some(control);
            context
        }))
        .oneshot(request)
        .await
        .unwrap();
        assert_eq!(response.status(), StatusCode::CONFLICT);
        assert_eq!(response.headers()["x-router-error-code"], "stale_revision");
    }

    #[tokio::test]
    async fn delete_rejects_discovery_managed_worker() {
        let (context, _control, _control_rx) = context_with_control();
        context
            .registry
            .add(WorkerSpec {
                id: WorkerId("discovered".into()),
                url: "http://worker-1:30000".into(),
                mode: WorkerMode::Plain,
                model_ids: vec![ModelId("m".into())],
                ..Default::default()
            })
            .unwrap();
        let response = build_router(context)
            .oneshot(
                Request::builder()
                    .method(Method::DELETE)
                    .uri("/workers/discovered")
                    .header("if-match", "1")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::CONFLICT);
        assert_eq!(
            response.headers()["x-router-error-code"],
            "worker_not_api_managed"
        );
    }

    #[tokio::test]
    async fn workers_api_is_unavailable_without_control_service() {
        let response = build_router(Arc::new(AppContext::stub()))
            .oneshot(
                Request::builder()
                    .uri("/workers")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            response.headers()["x-router-error-code"],
            "worker_control_unavailable"
        );
    }

    #[tokio::test]
    async fn invalid_worker_json_uses_router_error_contract() {
        let (context, _control, _control_rx) = context_with_control();
        let response = build_router(context)
            .oneshot(json_request(
                Method::POST,
                "/workers",
                json!({"url":"http://worker-1:30000","mode":"unknown"}),
            ))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert_eq!(
            response.headers()["x-router-error-code"],
            "invalid_worker_spec"
        );
    }

    #[tokio::test]
    async fn invalid_worker_filter_uses_router_error_contract() {
        let (context, _control, _control_rx) = context_with_control();
        let response = build_router(context)
            .oneshot(
                Request::builder()
                    .uri("/workers?mode=unknown")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert_eq!(response.headers()["x-router-error-code"], "bad_request");
        let body = body_json(response).await;
        assert_eq!(body["error"]["code"], "bad_request");
    }

    #[tokio::test]
    async fn public_router_create_and_remove_reach_worker_manager() {
        let worker_url = spawn_fake_worker().await;
        let mut context = AppContext::stub();
        let registry = Arc::clone(&context.registry);
        let (control, control_rx) =
            WorkerControlService::new(Arc::clone(&registry), ExternalWorkerSource::Static);
        context.worker_control = Some(Arc::clone(&control));
        let context = Arc::new(context);
        let (discovery_tx, discovery_rx) = tokio::sync::mpsc::channel(1);
        let manager = tokio::spawn(crate::workers::manager::run_with_config_and_control(
            discovery_rx,
            control_rx,
            Arc::clone(&control),
            None,
            None,
            None,
            None,
        ));
        let app = build_router(context);

        let response = app
            .clone()
            .oneshot(json_request(
                Method::POST,
                "/workers",
                json!({"url":worker_url,"mode":"plain"}),
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::ACCEPTED);
        let accepted = body_json(response).await;
        let worker_id = accepted["worker_id"].as_str().unwrap().to_owned();
        tokio::time::timeout(std::time::Duration::from_secs(2), async {
            loop {
                if registry.get(&WorkerId(worker_id.clone())).is_some() {
                    break;
                }
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("worker registration");

        let response = app
            .oneshot(
                Request::builder()
                    .method(Method::DELETE)
                    .uri(format!("/workers/{worker_id}"))
                    .header("if-match", "1")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::ACCEPTED);
        tokio::time::timeout(std::time::Duration::from_secs(2), async {
            loop {
                if control
                    .get(&WorkerId(worker_id.clone()))
                    .is_some_and(|view| view.deleted)
                {
                    break;
                }
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("worker removal");
        assert!(registry.get(&WorkerId(worker_id)).is_none());

        drop(discovery_tx);
        manager.abort();
    }

    #[tokio::test]
    async fn late_discovery_does_not_duplicate_an_api_worker_url() {
        let worker_url = spawn_fake_worker().await;
        let barrier_url = spawn_fake_worker().await;
        let mut context = AppContext::stub();
        let registry = Arc::clone(&context.registry);
        let (control, control_rx) =
            WorkerControlService::new(Arc::clone(&registry), ExternalWorkerSource::Kubernetes);
        context.worker_control = Some(Arc::clone(&control));
        let (discovery_tx, discovery_rx) = tokio::sync::mpsc::channel(4);
        let manager = tokio::spawn(crate::workers::manager::run_with_config_and_control(
            discovery_rx,
            control_rx,
            Arc::clone(&control),
            None,
            None,
            None,
            None,
        ));

        let response = build_router(Arc::new(context))
            .oneshot(json_request(
                Method::POST,
                "/workers",
                json!({"url":worker_url,"mode":"plain"}),
            ))
            .await
            .unwrap();
        let accepted = body_json(response).await;
        let api_id = WorkerId(accepted["worker_id"].as_str().unwrap().to_owned());
        tokio::time::timeout(std::time::Duration::from_secs(2), async {
            while registry.get(&api_id).is_none() {
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("API worker registration");

        let discovered_id = WorkerId("discovered-duplicate".into());
        discovery_tx
            .send(DiscoveryEvent::Added(WorkerSpec {
                id: discovered_id.clone(),
                url: worker_url.clone(),
                mode: WorkerMode::Plain,
                model_ids: vec![ModelId("m".into())],
                ..Default::default()
            }))
            .await
            .unwrap();
        discovery_tx
            .send(DiscoveryEvent::ModeChanged {
                id: discovered_id,
                mode: WorkerMode::Plain,
            })
            .await
            .unwrap();
        let barrier_id = WorkerId("discovery-barrier".into());
        discovery_tx
            .send(DiscoveryEvent::Added(WorkerSpec {
                id: barrier_id.clone(),
                url: barrier_url,
                mode: WorkerMode::Plain,
                model_ids: vec![ModelId("m".into())],
                ..Default::default()
            }))
            .await
            .unwrap();
        tokio::time::timeout(std::time::Duration::from_secs(2), async {
            while registry.get(&barrier_id).is_none() {
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("discovery barrier registration");

        assert_eq!(
            registry
                .all()
                .iter()
                .filter(|worker| worker.url == worker_url)
                .count(),
            1,
        );
        manager.abort();
    }

    #[tokio::test]
    async fn api_create_does_not_race_pending_discovery_for_the_same_url() {
        let (worker_url, introspection_started) = spawn_slow_fake_worker().await;
        let barrier_url = spawn_fake_worker().await;
        let mut context = AppContext::stub();
        let registry = Arc::clone(&context.registry);
        let (control, control_rx) =
            WorkerControlService::new(Arc::clone(&registry), ExternalWorkerSource::Kubernetes);
        context.worker_control = Some(Arc::clone(&control));
        let (discovery_tx, discovery_rx) = tokio::sync::mpsc::channel(2);
        let manager = tokio::spawn(crate::workers::manager::run_with_config_and_control(
            discovery_rx,
            control_rx,
            Arc::clone(&control),
            None,
            None,
            None,
            None,
        ));

        let discovered_id = WorkerId("discovered-pending".into());
        discovery_tx
            .send(DiscoveryEvent::Added(WorkerSpec {
                id: discovered_id.clone(),
                url: worker_url.clone(),
                mode: WorkerMode::Plain,
                model_ids: vec![ModelId("m".into())],
                ..Default::default()
            }))
            .await
            .unwrap();
        tokio::time::timeout(
            std::time::Duration::from_secs(2),
            introspection_started.notified(),
        )
        .await
        .expect("discovery introspection start");

        let response = build_router(Arc::new(context))
            .oneshot(json_request(
                Method::POST,
                "/workers",
                json!({"url":worker_url,"mode":"plain"}),
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::ACCEPTED);
        let accepted = body_json(response).await;
        let api_id = WorkerId(accepted["worker_id"].as_str().unwrap().to_owned());

        tokio::time::timeout(std::time::Duration::from_secs(2), async {
            loop {
                let operation_finished = control.get(&api_id).is_some_and(|worker| {
                    worker.operation.is_some_and(|operation| {
                        matches!(
                            operation.state,
                            OperationState::Succeeded | OperationState::Failed
                        )
                    })
                });
                if registry.get(&discovered_id).is_some() && operation_finished {
                    break;
                }
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("discovery and API operations");

        assert_eq!(
            registry
                .all()
                .iter()
                .filter(|worker| worker.url == worker_url)
                .count(),
            1,
        );
        assert_eq!(
            control
                .get(&api_id)
                .and_then(|worker| worker.operation)
                .map(|operation| operation.state),
            Some(OperationState::Failed),
        );

        let replacement_id = WorkerId("discovered-replacement".into());
        discovery_tx
            .send(DiscoveryEvent::Added(WorkerSpec {
                id: replacement_id.clone(),
                url: worker_url,
                mode: WorkerMode::Plain,
                model_ids: vec![ModelId("m".into())],
                ..Default::default()
            }))
            .await
            .unwrap();
        discovery_tx
            .send(DiscoveryEvent::Removed { id: discovered_id })
            .await
            .unwrap();
        discovery_tx
            .send(DiscoveryEvent::ModeChanged {
                id: replacement_id.clone(),
                mode: WorkerMode::Plain,
            })
            .await
            .unwrap();
        let barrier_id = WorkerId("post-conflict-barrier".into());
        discovery_tx
            .send(DiscoveryEvent::Added(WorkerSpec {
                id: barrier_id.clone(),
                url: barrier_url,
                mode: WorkerMode::Plain,
                model_ids: vec![ModelId("m".into())],
                ..Default::default()
            }))
            .await
            .unwrap();
        tokio::time::timeout(std::time::Duration::from_secs(2), async {
            while registry.get(&barrier_id).is_none() {
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("post-conflict discovery barrier");
        assert!(registry.get(&replacement_id).is_some());
        manager.abort();
    }
}
