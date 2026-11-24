//! `--api-key` / `--admin-api-key` enforcement, as Python's
//! `add_api_key_middleware` and `decide_request_auth`
//! (`sglang.srt.utils.auth`) apply them to the HTTP server.

use std::collections::HashMap;
use std::sync::{Arc, LazyLock};

use axum::{
    Json, Router,
    extract::{Request, State},
    handler::Handler,
    http::{Method, StatusCode, header::AUTHORIZATION},
    middleware::Next,
    response::{IntoResponse, Response},
    routing::MethodRouter,
};
use serde::Deserialize;

use crate::api_server::handlers::ADMIN_KEY_MISSING;
use crate::message::config::ApiKeys;

/// Python's `AuthLevel`.
#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(rename_all = "snake_case")]
enum AuthLevel {
    Normal,
    AdminOptional,
    AdminForce,
}

/// Every HTTP route sglang serves and its auth level, the table the Python
/// server reads too.
static ROUTE_LEVELS: LazyLock<HashMap<String, AuthLevel>> = LazyLock::new(|| {
    serde_json::from_str(include_str!(
        "../../../../../python/sglang/srt/utils/auth_levels.json"
    ))
    .expect("auth_levels.json")
});

/// The level of a served `path`: its exact entry, else the declared template
/// it matches (`{name}` takes one segment, `{name:path}` the rest), as
/// FastAPI's route match gives Python; `NORMAL` when nothing declares it.
fn level(path: &str) -> AuthLevel {
    if let Some(level) = ROUTE_LEVELS.get(path) {
        return *level;
    }
    ROUTE_LEVELS
        .iter()
        .find(|(route, _)| route.contains('{') && template_matches(route, path))
        .map_or(AuthLevel::Normal, |(_, level)| *level)
}

/// Whether the declared `template` serves `path`.
fn template_matches(template: &str, path: &str) -> bool {
    let mut segments = path.split('/');
    for pattern in template.split('/') {
        if pattern.starts_with('{') && pattern.ends_with(":path}") {
            return segments.next().is_some();
        }
        let matched = match segments.next() {
            Some(segment) if pattern.starts_with('{') => !segment.is_empty(),
            Some(segment) => segment == pattern,
            None => false,
        };
        if !matched {
            return false;
        }
    }
    segments.next().is_none()
}

pub(super) fn is_admin_force(path: &str) -> bool {
    level(path) == AuthLevel::AdminForce
}

/// Whether `auth_levels.json` declares `path` as `level` looks it up.
#[cfg(test)]
pub(super) fn has_level(path: &str) -> bool {
    ROUTE_LEVELS.contains_key(path)
}

/// The declared level of the route `path`, its parameters written as axum's
/// `{name}` / `{*name}` or FastAPI's `{name}` / `{name:path}`.
fn declared_level(path: &str) -> Option<AuthLevel> {
    fn bare_params(path: &str) -> String {
        path.split('/')
            .map(|segment| match segment.strip_prefix('{') {
                Some(param) => {
                    let name = param.trim_start_matches('*').trim_end_matches('}');
                    format!("{{{}}}", name.split(':').next().unwrap_or(name))
                }
                None => segment.to_string(),
            })
            .collect::<Vec<_>>()
            .join("/")
    }
    let path = bare_params(path);
    ROUTE_LEVELS
        .iter()
        .find(|(route, _)| bare_params(route) == path)
        .map(|(_, level)| *level)
}

/// `Router::route` for one of sglang's own routes, which `auth_levels.json`
/// must declare: an undeclared one panics as the router is built, so the tests
/// that build it fail. Routes an embedding package adds (`HttpLayer`) skip this
/// and are `NORMAL`, as an undeclared route is in Python.
pub(super) trait DeclaredRoute<S> {
    fn declared_route(self, path: &str, method_router: MethodRouter<S>) -> Self;
}

impl<S: Clone + Send + Sync + 'static> DeclaredRoute<S> for Router<S> {
    fn declared_route(self, path: &str, method_router: MethodRouter<S>) -> Self {
        declared_level(path).unwrap_or_else(|| {
            panic!("route {path} has no auth level in python/sglang/srt/utils/auth_levels.json")
        });
        self.route(path, method_router)
    }
}

/// Wrap `app` in the key check. It applies without keys too: the forced
/// admin routes are refused unless an admin key is configured.
pub(super) fn apply(app: Router, keys: &ApiKeys) -> Router {
    app.layer(axum::middleware::from_fn_with_state(
        Arc::new(keys.clone()),
        check,
    ))
}

/// Key the fallback of `app`, a router merged from keyed and open routes (the
/// merge keeps an open fallback): an unknown path needs the API key, as in
/// Python, where unmatched paths are `NORMAL` -- also a path Python serves as
/// an admin route, which this server would only answer with 404.
pub(super) fn guard_fallback(app: Router, keys: &ApiKeys) -> Router {
    app.fallback(
        (|| async { StatusCode::NOT_FOUND }).layer(axum::middleware::from_fn_with_state(
            Arc::new(keys.clone()),
            check_unserved,
        )),
    )
}

/// Python's 403 for an admin-only operation on a server without an admin key.
pub(super) fn admin_key_missing() -> Response {
    (
        StatusCode::FORBIDDEN,
        Json(serde_json::json!({ "error": ADMIN_KEY_MISSING })),
    )
        .into_response()
}

async fn check(State(keys): State<Arc<ApiKeys>>, request: Request, next: Next) -> Response {
    let level = level(request.uri().path());
    check_at(level, &keys, request, next).await
}

async fn check_unserved(
    State(keys): State<Arc<ApiKeys>>,
    request: Request,
    next: Next,
) -> Response {
    check_at(AuthLevel::Normal, &keys, request, next).await
}

async fn check_at(level: AuthLevel, keys: &ApiKeys, request: Request, next: Next) -> Response {
    let authorization = request
        .headers()
        .get(AUTHORIZATION)
        .and_then(|value| value.to_str().ok());
    match decide(
        request.method(),
        request.uri().path(),
        level,
        authorization,
        keys,
    ) {
        Decision::Allowed => next.run(request).await,
        Decision::Unauthorized => (
            StatusCode::UNAUTHORIZED,
            Json(serde_json::json!({ "error": "Unauthorized" })),
        )
            .into_response(),
        Decision::AdminKeyMissing => admin_key_missing(),
    }
}

#[derive(Debug, PartialEq)]
enum Decision {
    Allowed,
    Unauthorized,
    AdminKeyMissing,
}

/// Python's `decide_request_auth`.
fn decide(
    method: &Method,
    path: &str,
    level: AuthLevel,
    authorization: Option<&str>,
    keys: &ApiKeys,
) -> Decision {
    // Probes and scrapes never carry a key.
    if method == Method::OPTIONS
        || path.starts_with("/health")
        || path.starts_with("/ready")
        || path.starts_with("/metrics")
    {
        return Decision::Allowed;
    }
    let required = match level {
        AuthLevel::AdminForce => match keys.admin_api_key.as_deref() {
            Some(key) => Some(key),
            None => return Decision::AdminKeyMissing,
        },
        // The admin key when configured (the API key then does not unlock
        // these), else the API key.
        AuthLevel::AdminOptional => keys.admin_api_key.as_deref().or(keys.api_key.as_deref()),
        // The API key alone; the admin key does not unlock these.
        AuthLevel::Normal => keys.api_key.as_deref(),
    };
    if required.is_none_or(|key| bearer_matches(authorization, key)) {
        Decision::Allowed
    } else {
        Decision::Unauthorized
    }
}

/// `Authorization: Bearer <key>`, the scheme case-insensitive and the token
/// compared in constant time, as Python's `_check_bearer_token`.
fn bearer_matches(authorization: Option<&str>, key: &str) -> bool {
    let Some((scheme, token)) = authorization.and_then(|value| value.split_once(' ')) else {
        return false;
    };
    scheme.eq_ignore_ascii_case("bearer") && secret_matches(token.as_bytes(), key)
}

/// Whether `presented` is `secret`, compared in constant time.
pub(super) fn secret_matches(presented: &[u8], secret: &str) -> bool {
    presented.len() == secret.len()
        && presented
            .iter()
            .zip(secret.bytes())
            .fold(0u8, |diff, (a, b)| diff | (a ^ b))
            == 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{body::Body, routing::get};
    use std::future::IntoFuture;
    use tower::ServiceExt;

    const API: &str = "sk-api";
    const ADMIN: &str = "sk-admin";

    fn keys(api_key: Option<&str>, admin_api_key: Option<&str>) -> ApiKeys {
        ApiKeys::new(api_key.map(Into::into), admin_api_key.map(Into::into))
    }

    /// `decide_request_auth`'s answers, recorded by Python's `test_auth.py`
    /// (`render_auth_decisions_fixture`).
    #[test]
    fn decisions_match_python() {
        #[derive(Deserialize)]
        struct Case {
            api_key: Option<String>,
            admin_api_key: Option<String>,
            method: String,
            path: String,
            level: AuthLevel,
            statuses: Vec<u16>,
        }
        #[derive(Deserialize)]
        struct Fixture {
            authorizations: Vec<Option<String>>,
            cases: Vec<Case>,
        }
        let fixture: Fixture =
            serde_json::from_str(include_str!("../../../testdata/auth_decisions.json")).unwrap();
        for case in fixture.cases {
            let keys = ApiKeys::new(case.api_key.clone(), case.admin_api_key.clone());
            let method = Method::from_bytes(case.method.as_bytes()).unwrap();
            assert_eq!(case.statuses.len(), fixture.authorizations.len());
            for (authorization, status) in fixture.authorizations.iter().zip(case.statuses) {
                let decision = decide(
                    &method,
                    &case.path,
                    case.level,
                    authorization.as_deref(),
                    &keys,
                );
                let expected = match status {
                    200 => Decision::Allowed,
                    401 => Decision::Unauthorized,
                    403 => Decision::AdminKeyMissing,
                    _ => panic!("unexpected status {status}"),
                };
                assert_eq!(
                    decision, expected,
                    "{:?} {:?} {} {} {:?} {authorization:?}",
                    case.api_key, case.admin_api_key, case.method, case.path, case.level
                );
            }
        }
    }

    #[test]
    fn levels_come_from_the_shared_table() {
        assert_eq!(level("/start_profile"), AuthLevel::AdminForce);
        assert_eq!(level("/flush_cache"), AuthLevel::AdminOptional);
        assert_eq!(level("/generate"), AuthLevel::Normal);
        assert_eq!(
            level(crate::api_server::core::FORWARD_PEER_PATH),
            AuthLevel::AdminOptional
        );
        assert_eq!(level("/nope"), AuthLevel::Normal);
        // A declared template serves its parameterized paths at its level.
        assert_eq!(level("/dumper/status"), AuthLevel::AdminForce);
        assert_eq!(level("/dumper/a/b"), AuthLevel::Normal);
        assert_eq!(level("/v1/models/org/model"), AuthLevel::Normal);
        assert_eq!(level("/metrics/a/b"), AuthLevel::Normal);
        // axum's parameter syntax finds FastAPI's.
        assert_eq!(
            declared_level("/v1/models/{model}"),
            Some(AuthLevel::Normal)
        );
        assert_eq!(declared_level("/metrics/{*path}"), Some(AuthLevel::Normal));
        assert_eq!(declared_level("/nope"), None);
    }

    #[test]
    #[should_panic(expected = "has no auth level")]
    fn an_undeclared_route_stops_the_server() {
        let _: Router = Router::new().declared_route("/nope", get(|| async { "ok" }));
    }

    /// A DP control relay passes a keyed peer's check: the relay presents the
    /// admin key, which the peer requires on its relay route once one is
    /// configured; a relay without it is refused.
    #[tokio::test]
    async fn the_control_relay_presents_the_admin_key_to_its_peers() {
        use crate::api_server::core::{FORWARD_PEER_PATH, peer_endpoints};
        use crate::message::config::{DeploymentInfo, RustServerEndpoint};
        use crate::message::io_struct::{ControlRequest, FlushCacheReqInput};
        use axum::routing::post;

        let both = keys(Some(API), Some(ADMIN));
        let peer = apply(
            Router::new().route(FORWARD_PEER_PATH, post(|| async { "ok" })),
            &both,
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(axum::serve(listener, peer).into_future());
        let info = DeploymentInfo {
            dp_rank: Some(0),
            rust_server_endpoints: vec![RustServerEndpoint {
                dp_rank: 1,
                endpoint: url,
                peer_token: None,
            }],
            ..Default::default()
        };
        let flush =
            || ControlRequest::FlushCacheReqInput(FlushCacheReqInput::new("r".into(), None));
        for (case, relay_keys, accepted) in [
            ("admin key", both.clone(), true),
            ("API key only", keys(Some(API), None), false),
            ("no key", ApiKeys::default(), false),
        ] {
            let peers = peer_endpoints(&info, 0, &relay_keys).unwrap();
            let outcome = peers[0].control(&flush).await.remove(0).outcome;
            match outcome {
                Ok(_) => assert!(accepted, "{case}: must be refused"),
                Err(error) => {
                    assert!(!accepted, "{case}: {error}");
                    assert_eq!(error.http_status(), StatusCode::UNAUTHORIZED);
                }
            }
        }
    }

    #[tokio::test]
    async fn rejections_are_pythons() {
        let app = Router::new()
            .route("/generate", get(|| async { "ok" }))
            .route("/start_profile", get(|| async { "ok" }));
        let api_only = keys(Some(API), None);
        let app = guard_fallback(apply(app, &api_only), &api_only);
        let send = |uri: &str, authorization: Option<String>| {
            let mut request = Request::builder().uri(uri);
            if let Some(value) = authorization {
                request = request.header(AUTHORIZATION, value);
            }
            app.clone().oneshot(request.body(Body::empty()).unwrap())
        };
        let body = |response: Response| async {
            axum::body::to_bytes(response.into_body(), 1024)
                .await
                .unwrap()
        };

        let rejected = send("/generate", None).await.unwrap();
        assert_eq!(rejected.status(), StatusCode::UNAUTHORIZED);
        assert_eq!(&body(rejected).await[..], br#"{"error":"Unauthorized"}"#);

        let accepted = send("/generate", Some(format!("Bearer {API}")))
            .await
            .unwrap();
        assert_eq!(accepted.status(), StatusCode::OK);

        let forced = send("/start_profile", Some(format!("Bearer {API}")))
            .await
            .unwrap();
        assert_eq!(forced.status(), StatusCode::FORBIDDEN);
        let error: serde_json::Value = serde_json::from_slice(&body(forced).await).unwrap();
        assert_eq!(error["error"], ADMIN_KEY_MISSING);

        // A wrong method is refused at the route's level before the 405, as in Python.
        let wrong_method = Request::post("/start_profile")
            .header(AUTHORIZATION, format!("Bearer {API}"))
            .body(Body::empty())
            .unwrap();
        let wrong_method = app.clone().oneshot(wrong_method).await.unwrap();
        assert_eq!(wrong_method.status(), StatusCode::FORBIDDEN);

        // Python's admin route, not served here: NORMAL, so the API key gets the 404.
        let unserved = send("/pause_generation", Some(format!("Bearer {API}")))
            .await
            .unwrap();
        assert_eq!(unserved.status(), StatusCode::NOT_FOUND);
    }
}
