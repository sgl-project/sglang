//! `--api-key` / `--admin-api-key` enforcement, as Python's
//! `add_api_key_middleware` and `decide_request_auth`
//! (`sglang.srt.utils.auth`) apply them to the HTTP server.

use std::sync::Arc;

use axum::{
    Json, Router,
    extract::{Request, State},
    handler::Handler,
    http::{Method, StatusCode, header::AUTHORIZATION},
    middleware::Next,
    response::{IntoResponse, Response},
};

use crate::message::config::ApiKeys;

/// The served routes Python marks `@auth_level(AuthLevel.ADMIN_OPTIONAL)`;
/// every other route is `NORMAL`. None of them is `ADMIN_FORCE`. The DP
/// control relay carries those same operations, so it needs their key too.
const ADMIN_OPTIONAL_PATHS: &[&str] = &[
    crate::api_server::core::FORWARD_PEER_PATH,
    "/flush_cache",
    "/start_profile",
    "/stop_profile",
    "/weights_checker",
    "/abort_request",
    "/hicache/storage-backend",
    "/hicache/storage-backend/clear",
];

/// Wrap `app` in the key check. Python installs its middleware only when a
/// key is configured, so without one `app` is returned untouched.
pub(super) fn apply(app: Router, keys: &ApiKeys) -> Router {
    match configured(keys) {
        Some(keys) => app.layer(axum::middleware::from_fn_with_state(keys, check)),
        None => app,
    }
}

/// Key the fallback of `app`, a router merged from keyed and open routes (the
/// merge keeps an open fallback): an unknown path needs the API key, as in
/// Python, where unmatched paths are `NORMAL`.
pub(super) fn guard_fallback(app: Router, keys: &ApiKeys) -> Router {
    match configured(keys) {
        Some(keys) => app.fallback(
            (|| async { StatusCode::NOT_FOUND })
                .layer(axum::middleware::from_fn_with_state(keys, check)),
        ),
        None => app,
    }
}

fn configured(keys: &ApiKeys) -> Option<Arc<ApiKeys>> {
    (keys.api_key.is_some() || keys.admin_api_key.is_some()).then(|| Arc::new(keys.clone()))
}

async fn check(State(keys): State<Arc<ApiKeys>>, request: Request, next: Next) -> Response {
    let authorization = request
        .headers()
        .get(AUTHORIZATION)
        .and_then(|value| value.to_str().ok());
    if allowed(request.method(), request.uri().path(), authorization, &keys) {
        next.run(request).await
    } else {
        (
            StatusCode::UNAUTHORIZED,
            Json(serde_json::json!({ "error": "Unauthorized" })),
        )
            .into_response()
    }
}

/// Python's `decide_request_auth` for the `NORMAL` and `ADMIN_OPTIONAL`
/// levels.
fn allowed(method: &Method, path: &str, authorization: Option<&str>, keys: &ApiKeys) -> bool {
    // Probes and scrapes never carry a key.
    if method == Method::OPTIONS
        || path.starts_with("/health")
        || path.starts_with("/ready")
        || path.starts_with("/metrics")
    {
        return true;
    }
    let required = if ADMIN_OPTIONAL_PATHS.contains(&path) {
        // The admin key when configured (the API key then does not unlock
        // these), else the API key.
        keys.admin_api_key.as_deref().or(keys.api_key.as_deref())
    } else {
        // The API key alone; the admin key does not unlock these.
        keys.api_key.as_deref()
    };
    required.is_none_or(|key| bearer_matches(authorization, key))
}

/// `Authorization: Bearer <key>`, the scheme case-insensitive and the token
/// compared in constant time, as Python's `_check_bearer_token`.
fn bearer_matches(authorization: Option<&str>, key: &str) -> bool {
    let Some((scheme, token)) = authorization.and_then(|value| value.split_once(' ')) else {
        return false;
    };
    scheme.eq_ignore_ascii_case("bearer")
        && token.len() == key.len()
        && token
            .bytes()
            .zip(key.bytes())
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

    /// Python's `decide_request_auth` table for every key configuration, at
    /// both levels, with no / a wrong / the API / the admin key.
    #[test]
    fn decisions_match_python() {
        let wrong = "Bearer nope".to_string();
        let api = format!("Bearer {API}");
        let admin = format!("Bearer {ADMIN}");
        let presented = [
            None,
            Some(wrong.as_str()),
            Some(api.as_str()),
            Some(admin.as_str()),
        ];
        // (api_key, admin_api_key, path, [allowed for each presented key])
        let cases = [
            (None, None, "/generate", [true, true, true, true]),
            (None, None, "/flush_cache", [true, true, true, true]),
            (Some(API), None, "/generate", [false, false, true, false]),
            (Some(API), None, "/flush_cache", [false, false, true, false]),
            (None, Some(ADMIN), "/generate", [true, true, true, true]),
            (
                None,
                Some(ADMIN),
                "/flush_cache",
                [false, false, false, true],
            ),
            (
                Some(API),
                Some(ADMIN),
                "/v1/chat/completions",
                [false, false, true, false],
            ),
            (
                Some(API),
                Some(ADMIN),
                "/abort_request",
                [false, false, false, true],
            ),
            (
                Some(API),
                Some(ADMIN),
                "/weights_checker",
                [false, false, false, true],
            ),
            (
                Some(API),
                Some(ADMIN),
                "/start_profile",
                [false, false, false, true],
            ),
            (
                Some(API),
                Some(ADMIN),
                "/stop_profile",
                [false, false, false, true],
            ),
            (
                Some(API),
                Some(ADMIN),
                "/hicache/storage-backend",
                [false, false, false, true],
            ),
            (
                Some(API),
                Some(ADMIN),
                "/hicache/storage-backend/clear",
                [false, false, false, true],
            ),
        ];
        for (api_key, admin_api_key, path, expected) in cases {
            let keys = keys(api_key, admin_api_key);
            for (authorization, expected) in presented.iter().zip(expected) {
                assert_eq!(
                    allowed(&Method::POST, path, *authorization, &keys),
                    expected,
                    "{api_key:?} {admin_api_key:?} {path} {authorization:?}"
                );
            }
        }
        let both = keys(Some(API), Some(ADMIN));
        for path in [
            "/health",
            "/health_generate",
            "/metrics",
            "/metrics/x",
            "/ready",
        ] {
            assert!(allowed(&Method::GET, path, None, &both), "{path}");
        }
        assert!(allowed(&Method::OPTIONS, "/generate", None, &both));
        // Unknown paths are NORMAL, as Python's unmatched routes.
        assert!(!allowed(&Method::GET, "/nope", None, &both));
    }

    #[test]
    fn bearer_parsing_matches_python() {
        assert!(bearer_matches(Some("Bearer sk"), "sk"));
        assert!(bearer_matches(Some("bEaReR sk"), "sk"));
        assert!(!bearer_matches(Some("Bearer  sk"), "sk"));
        assert!(!bearer_matches(Some("Bearer sk "), "sk"));
        assert!(!bearer_matches(Some("Basic sk"), "sk"));
        assert!(!bearer_matches(Some("Bearer"), "sk"));
        assert!(!bearer_matches(Some("Bearer s"), "sk"));
        assert!(!bearer_matches(None, "sk"));
        // An empty key is no key.
        assert!(allowed(
            &Method::POST,
            "/generate",
            None,
            &keys(Some(""), Some(""))
        ));
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
    async fn rejection_is_pythons_401() {
        let app = Router::new().route("/generate", get(|| async { "ok" }));
        let app = apply(app, &keys(Some(API), None));
        let send = |authorization: Option<String>| {
            let mut request = Request::builder().uri("/generate");
            if let Some(value) = authorization {
                request = request.header(AUTHORIZATION, value);
            }
            app.clone().oneshot(request.body(Body::empty()).unwrap())
        };

        let rejected = send(None).await.unwrap();
        assert_eq!(rejected.status(), StatusCode::UNAUTHORIZED);
        let body = axum::body::to_bytes(rejected.into_body(), 1024)
            .await
            .unwrap();
        assert_eq!(&body[..], br#"{"error":"Unauthorized"}"#);

        let accepted = send(Some(format!("Bearer {API}"))).await.unwrap();
        assert_eq!(accepted.status(), StatusCode::OK);
    }
}
