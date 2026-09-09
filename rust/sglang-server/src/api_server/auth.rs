//! API-key authorization for the public HTTP router.
//!
//! The policy intentionally mirrors `sglang.srt.utils.auth`. Keep the pure
//! decision function independent of handlers and model state so changes to the
//! Python policy can be compared here directly.

use std::{fmt, sync::Arc};

use axum::{
    Json, Router,
    extract::{Request, State},
    http::{HeaderValue, Method, StatusCode, header::AUTHORIZATION},
    middleware::{self, Next},
    response::{IntoResponse, Response},
};
use constant_time_eq::constant_time_eq;
use serde::Serialize;

/// Secrets used only by the HTTP authorization layer.
///
/// This type deliberately does not implement `Serialize`, and its `Debug`
/// implementation reports presence only. Empty strings are normalized to an
/// unset key, matching Python's truthiness check; whitespace is preserved.
#[derive(Clone, Default)]
pub(crate) struct AuthConfig {
    api_key: Option<Arc<str>>,
    admin_api_key: Option<Arc<str>>,
}

impl AuthConfig {
    pub(crate) fn new(api_key: Option<String>, admin_api_key: Option<String>) -> Self {
        Self {
            api_key: normalize_key(api_key),
            admin_api_key: normalize_key(admin_api_key),
        }
    }

    fn needs_layer(&self, level: AuthLevel) -> bool {
        match level {
            AuthLevel::Normal => self.api_key.is_some(),
            AuthLevel::AdminOptional => self.api_key.is_some() || self.admin_api_key.is_some(),
            // An unconfigured forced-admin route must still reject with 403.
            AuthLevel::AdminForce => true,
        }
    }
}

fn normalize_key(key: Option<String>) -> Option<Arc<str>> {
    key.filter(|key| !key.is_empty()).map(Arc::from)
}

impl fmt::Debug for AuthConfig {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AuthConfig")
            .field("api_key", &SecretPresence(self.api_key.is_some()))
            .field(
                "admin_api_key",
                &SecretPresence(self.admin_api_key.is_some()),
            )
            .finish()
    }
}

struct SecretPresence(bool);

impl fmt::Debug for SecretPresence {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(if self.0 { "<configured>" } else { "<unset>" })
    }
}

/// Authorization policy attached to a group of endpoints.
// The admin levels deliberately ship with the policy engine before Rust owns
// any admin routes; keeping them here pins Python parity for those future route
// groups without pretending they are mounted today.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AuthLevel {
    /// Requires `api_key` when it is configured.
    Normal,
    /// Requires the admin key when present, otherwise the regular key when
    /// present, and is public when neither exists.
    AdminOptional,
    /// Always requires an admin key; returns 403 if no admin key is configured.
    AdminForce,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AuthDecision {
    Allow,
    Deny(StatusCode),
}

/// Decide whether one request is authorized without invoking any HTTP handler.
pub(crate) fn decide_request_auth(
    method: &Method,
    path: &str,
    authorization: Option<&HeaderValue>,
    config: &AuthConfig,
    level: AuthLevel,
) -> AuthDecision {
    // Kubernetes probes, Prometheus scraping, and CORS preflight stay public.
    if method == Method::OPTIONS || path.starts_with("/health") || path.starts_with("/metrics") {
        return AuthDecision::Allow;
    }

    match level {
        AuthLevel::AdminForce => match &config.admin_api_key {
            None => AuthDecision::Deny(StatusCode::FORBIDDEN),
            Some(key) if bearer_matches(authorization, key) => AuthDecision::Allow,
            Some(_) => AuthDecision::Deny(StatusCode::UNAUTHORIZED),
        },
        AuthLevel::AdminOptional => {
            let expected = config
                .admin_api_key
                .as_deref()
                .or(config.api_key.as_deref());
            match expected {
                None => AuthDecision::Allow,
                Some(key) if bearer_matches(authorization, key) => AuthDecision::Allow,
                Some(_) => AuthDecision::Deny(StatusCode::UNAUTHORIZED),
            }
        }
        AuthLevel::Normal => match &config.api_key {
            None => AuthDecision::Allow,
            Some(key) if bearer_matches(authorization, key) => AuthDecision::Allow,
            Some(_) => AuthDecision::Deny(StatusCode::UNAUTHORIZED),
        },
    }
}

/// Install authorization on all routes and the fallback already present in
/// `router`. Callers must merge internal-only routers after this helper.
///
/// For policies that are unconditionally public under the current config, the
/// router is returned unchanged so the default configuration has no per-request
/// middleware cost.
pub(crate) fn protect<S>(router: Router<S>, config: Arc<AuthConfig>, level: AuthLevel) -> Router<S>
where
    S: Clone + Send + Sync + 'static,
{
    if !config.needs_layer(level) {
        return router;
    }

    router.layer(middleware::from_fn_with_state(
        AuthMiddlewareState { config, level },
        authorize,
    ))
}

#[derive(Clone)]
struct AuthMiddlewareState {
    config: Arc<AuthConfig>,
    level: AuthLevel,
}

async fn authorize(
    State(state): State<AuthMiddlewareState>,
    request: Request,
    next: Next,
) -> Response {
    let decision = decide_request_auth(
        request.method(),
        request.uri().path(),
        request.headers().get(AUTHORIZATION),
        &state.config,
        state.level,
    );

    match decision {
        AuthDecision::Allow => next.run(request).await,
        AuthDecision::Deny(status) => denial_response(status),
    }
}

fn bearer_matches(authorization: Option<&HeaderValue>, expected: &str) -> bool {
    let Some(value) = authorization else {
        return false;
    };
    let bytes = value.as_bytes();
    let Some(space) = bytes.iter().position(|byte| *byte == b' ') else {
        return false;
    };
    let (scheme, token_with_separator) = bytes.split_at(space);
    let token = &token_with_separator[1..];

    scheme.eq_ignore_ascii_case(b"bearer") && constant_time_eq(token, expected.as_bytes())
}

#[derive(Serialize)]
struct AuthError {
    error: &'static str,
}

fn denial_response(status: StatusCode) -> Response {
    let error = if status == StatusCode::FORBIDDEN {
        "Forbidden"
    } else {
        "Unauthorized"
    };
    (status, Json(AuthError { error })).into_response()
}

#[cfg(test)]
mod tests {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    use axum::{
        Router,
        body::{Body, to_bytes},
        http::{Request as HttpRequest, header::CONTENT_TYPE},
        routing::{get, post},
    };
    use serde::Deserialize;
    use tower::ServiceExt;

    use super::*;

    const API_KEY: &str = "customer-secret";
    const ADMIN_KEY: &str = "admin-secret";

    fn config(api_key: Option<&str>, admin_api_key: Option<&str>) -> AuthConfig {
        AuthConfig::new(
            api_key.map(ToOwned::to_owned),
            admin_api_key.map(ToOwned::to_owned),
        )
    }

    fn header(value: &'static str) -> HeaderValue {
        HeaderValue::from_static(value)
    }

    fn decide(
        config: &AuthConfig,
        level: AuthLevel,
        authorization: Option<&HeaderValue>,
    ) -> AuthDecision {
        decide_request_auth(&Method::GET, "/v1/models", authorization, config, level)
    }

    #[test]
    fn policy_matrix_matches_python() {
        struct Case {
            name: &'static str,
            config: AuthConfig,
            level: AuthLevel,
            accepted_token: Option<&'static str>,
            missing: AuthDecision,
        }

        let cases = [
            Case {
                name: "no keys normal",
                config: config(None, None),
                level: AuthLevel::Normal,
                accepted_token: None,
                missing: AuthDecision::Allow,
            },
            Case {
                name: "no keys admin optional",
                config: config(None, None),
                level: AuthLevel::AdminOptional,
                accepted_token: None,
                missing: AuthDecision::Allow,
            },
            Case {
                name: "no keys admin force",
                config: config(None, None),
                level: AuthLevel::AdminForce,
                accepted_token: None,
                missing: AuthDecision::Deny(StatusCode::FORBIDDEN),
            },
            Case {
                name: "api key only normal",
                config: config(Some(API_KEY), None),
                level: AuthLevel::Normal,
                accepted_token: Some(API_KEY),
                missing: AuthDecision::Deny(StatusCode::UNAUTHORIZED),
            },
            Case {
                name: "api key only admin optional",
                config: config(Some(API_KEY), None),
                level: AuthLevel::AdminOptional,
                accepted_token: Some(API_KEY),
                missing: AuthDecision::Deny(StatusCode::UNAUTHORIZED),
            },
            Case {
                name: "api key only admin force",
                config: config(Some(API_KEY), None),
                level: AuthLevel::AdminForce,
                accepted_token: None,
                missing: AuthDecision::Deny(StatusCode::FORBIDDEN),
            },
            Case {
                name: "admin key only normal",
                config: config(None, Some(ADMIN_KEY)),
                level: AuthLevel::Normal,
                accepted_token: None,
                missing: AuthDecision::Allow,
            },
            Case {
                name: "admin key only admin optional",
                config: config(None, Some(ADMIN_KEY)),
                level: AuthLevel::AdminOptional,
                accepted_token: Some(ADMIN_KEY),
                missing: AuthDecision::Deny(StatusCode::UNAUTHORIZED),
            },
            Case {
                name: "admin key only admin force",
                config: config(None, Some(ADMIN_KEY)),
                level: AuthLevel::AdminForce,
                accepted_token: Some(ADMIN_KEY),
                missing: AuthDecision::Deny(StatusCode::UNAUTHORIZED),
            },
            Case {
                name: "both keys normal",
                config: config(Some(API_KEY), Some(ADMIN_KEY)),
                level: AuthLevel::Normal,
                accepted_token: Some(API_KEY),
                missing: AuthDecision::Deny(StatusCode::UNAUTHORIZED),
            },
            Case {
                name: "both keys admin optional",
                config: config(Some(API_KEY), Some(ADMIN_KEY)),
                level: AuthLevel::AdminOptional,
                accepted_token: Some(ADMIN_KEY),
                missing: AuthDecision::Deny(StatusCode::UNAUTHORIZED),
            },
            Case {
                name: "both keys admin force",
                config: config(Some(API_KEY), Some(ADMIN_KEY)),
                level: AuthLevel::AdminForce,
                accepted_token: Some(ADMIN_KEY),
                missing: AuthDecision::Deny(StatusCode::UNAUTHORIZED),
            },
        ];

        for case in cases {
            assert_eq!(
                decide(&case.config, case.level, None),
                case.missing,
                "{} without a token",
                case.name
            );

            if let Some(token) = case.accepted_token {
                let authorization = HeaderValue::from_str(&format!("Bearer {token}")).unwrap();
                assert_eq!(
                    decide(&case.config, case.level, Some(&authorization)),
                    AuthDecision::Allow,
                    "{} with its required token",
                    case.name
                );
            }
        }
    }

    #[test]
    fn key_roles_are_not_interchangeable() {
        let config = config(Some(API_KEY), Some(ADMIN_KEY));
        let api = header("Bearer customer-secret");
        let admin = header("Bearer admin-secret");

        assert_eq!(
            decide(&config, AuthLevel::Normal, Some(&admin)),
            AuthDecision::Deny(StatusCode::UNAUTHORIZED)
        );
        assert_eq!(
            decide(&config, AuthLevel::AdminOptional, Some(&api)),
            AuthDecision::Deny(StatusCode::UNAUTHORIZED)
        );
        assert_eq!(
            decide(&config, AuthLevel::AdminForce, Some(&api)),
            AuthDecision::Deny(StatusCode::UNAUTHORIZED)
        );
    }

    #[test]
    fn public_prefixes_and_options_always_bypass_auth() {
        let config = config(Some(API_KEY), None);
        for path in [
            "/health",
            "/health_generate",
            "/health/ready",
            "/metrics",
            "/metrics/prometheus",
        ] {
            assert_eq!(
                decide_request_auth(&Method::GET, path, None, &config, AuthLevel::Normal),
                AuthDecision::Allow,
                "{path}"
            );
        }
        assert_eq!(
            decide_request_auth(
                &Method::OPTIONS,
                "/v1/models",
                None,
                &config,
                AuthLevel::Normal
            ),
            AuthDecision::Allow
        );
    }

    #[test]
    fn public_bypass_precedes_forced_admin_rejection() {
        let config = config(None, None);
        for (method, path) in [
            (Method::GET, "/health"),
            (Method::GET, "/metrics"),
            (Method::OPTIONS, "/admin"),
        ] {
            assert_eq!(
                decide_request_auth(&method, path, None, &config, AuthLevel::AdminForce),
                AuthDecision::Allow
            );
        }
    }

    #[test]
    fn bearer_parsing_matches_python_split_semantics() {
        let spaced_config = config(Some(" secret"), None);
        let cases = [
            ("Bearer  secret", AuthDecision::Allow),
            ("bearer  secret", AuthDecision::Allow),
            ("BEARER  secret", AuthDecision::Allow),
            (
                "Bearer secret",
                AuthDecision::Deny(StatusCode::UNAUTHORIZED),
            ),
            (
                "Bearer   secret",
                AuthDecision::Deny(StatusCode::UNAUTHORIZED),
            ),
            (
                "Basic  secret",
                AuthDecision::Deny(StatusCode::UNAUTHORIZED),
            ),
            ("Bearer", AuthDecision::Deny(StatusCode::UNAUTHORIZED)),
            ("", AuthDecision::Deny(StatusCode::UNAUTHORIZED)),
        ];

        for (value, expected) in cases {
            let value = HeaderValue::from_str(value).unwrap();
            assert_eq!(
                decide(&spaced_config, AuthLevel::Normal, Some(&value)),
                expected,
                "{value:?}"
            );
        }

        let trailing_config = config(Some("secret"), None);
        let trailing = header("Bearer secret ");
        assert_eq!(
            decide(&trailing_config, AuthLevel::Normal, Some(&trailing)),
            AuthDecision::Deny(StatusCode::UNAUTHORIZED)
        );
    }

    #[test]
    fn empty_keys_are_unset_but_whitespace_is_preserved() {
        let empty = AuthConfig::new(Some(String::new()), Some(String::new()));
        assert_eq!(decide(&empty, AuthLevel::Normal, None), AuthDecision::Allow);
        assert_eq!(
            decide(&empty, AuthLevel::AdminOptional, None),
            AuthDecision::Allow
        );
        assert_eq!(
            decide(&empty, AuthLevel::AdminForce, None),
            AuthDecision::Deny(StatusCode::FORBIDDEN)
        );

        let whitespace = config(Some(" "), None);
        let authorization = header("Bearer  ");
        assert_eq!(
            decide(&whitespace, AuthLevel::Normal, None),
            AuthDecision::Deny(StatusCode::UNAUTHORIZED)
        );
        assert_eq!(
            decide(&whitespace, AuthLevel::Normal, Some(&authorization)),
            AuthDecision::Allow
        );
    }

    #[test]
    fn debug_output_reports_presence_without_secrets() {
        let config = config(Some(API_KEY), Some(ADMIN_KEY));
        let debug = format!("{config:?}");
        assert_eq!(
            debug,
            "AuthConfig { api_key: <configured>, admin_api_key: <configured> }"
        );
        assert!(!debug.contains(API_KEY));
        assert!(!debug.contains(ADMIN_KEY));
    }

    async fn counted_handler(counter: Arc<AtomicUsize>) -> &'static str {
        counter.fetch_add(1, Ordering::Relaxed);
        "handled"
    }

    fn counted_router(counter: Arc<AtomicUsize>) -> Router {
        Router::new()
            .route(
                "/protected",
                get({
                    let counter = counter.clone();
                    move || counted_handler(counter.clone())
                }),
            )
            .fallback(move || counted_handler(counter.clone()))
    }

    async fn call(app: Router, request: HttpRequest<Body>) -> Response {
        app.oneshot(request).await.unwrap()
    }

    async fn body(response: Response) -> String {
        String::from_utf8(
            to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap()
                .to_vec(),
        )
        .unwrap()
    }

    #[tokio::test]
    async fn layer_rejects_before_handler_and_returns_exact_json() {
        let counter = Arc::new(AtomicUsize::new(0));
        let app = protect(
            counted_router(counter.clone()),
            Arc::new(config(Some(API_KEY), None)),
            AuthLevel::Normal,
        );

        let response = call(
            app,
            HttpRequest::builder()
                .uri("/protected")
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        assert_eq!(response.headers()[CONTENT_TYPE], "application/json");
        assert_eq!(body(response).await, r#"{"error":"Unauthorized"}"#);
        assert_eq!(counter.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn layer_allows_valid_token_exactly_once() {
        let counter = Arc::new(AtomicUsize::new(0));
        let app = protect(
            counted_router(counter.clone()),
            Arc::new(config(Some(API_KEY), None)),
            AuthLevel::Normal,
        );

        let response = call(
            app,
            HttpRequest::builder()
                .uri("/protected")
                .header(AUTHORIZATION, "Bearer customer-secret")
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(body(response).await, "handled");
        assert_eq!(counter.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn layer_protects_existing_fallback() {
        let counter = Arc::new(AtomicUsize::new(0));
        let app = protect(
            counted_router(counter.clone()),
            Arc::new(config(Some(API_KEY), None)),
            AuthLevel::Normal,
        );

        let unauthorized = call(
            app.clone(),
            HttpRequest::builder()
                .uri("/unknown")
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(unauthorized.status(), StatusCode::UNAUTHORIZED);
        assert_eq!(counter.load(Ordering::Relaxed), 0);

        let authorized = call(
            app,
            HttpRequest::builder()
                .uri("/unknown")
                .header(AUTHORIZATION, "Bearer customer-secret")
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(authorized.status(), StatusCode::OK);
        assert_eq!(counter.load(Ordering::Relaxed), 1);
    }

    #[derive(Deserialize)]
    struct Payload {
        value: String,
    }

    #[tokio::test]
    async fn layer_rejects_before_json_extraction() {
        async fn json_handler(Json(payload): Json<Payload>) -> String {
            payload.value
        }

        let app = protect(
            Router::new().route("/json", post(json_handler)),
            Arc::new(config(Some(API_KEY), None)),
            AuthLevel::Normal,
        );
        let response = call(
            app,
            HttpRequest::builder()
                .method(Method::POST)
                .uri("/json")
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from("not-json"))
                .unwrap(),
        )
        .await;

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        assert_eq!(body(response).await, r#"{"error":"Unauthorized"}"#);
    }

    #[tokio::test]
    async fn forced_admin_without_configuration_returns_exact_403() {
        let counter = Arc::new(AtomicUsize::new(0));
        let app = protect(
            counted_router(counter.clone()),
            Arc::new(config(Some(API_KEY), None)),
            AuthLevel::AdminForce,
        );
        let response = call(
            app,
            HttpRequest::builder()
                .uri("/protected")
                .header(AUTHORIZATION, "Bearer customer-secret")
                .body(Body::empty())
                .unwrap(),
        )
        .await;

        assert_eq!(response.status(), StatusCode::FORBIDDEN);
        assert_eq!(response.headers()[CONTENT_TYPE], "application/json");
        assert_eq!(body(response).await, r#"{"error":"Forbidden"}"#);
        assert_eq!(counter.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn normal_router_without_api_key_is_unchanged() {
        let counter = Arc::new(AtomicUsize::new(0));
        let app = protect(
            counted_router(counter.clone()),
            Arc::new(config(None, Some(ADMIN_KEY))),
            AuthLevel::Normal,
        );
        let response = call(
            app,
            HttpRequest::builder()
                .uri("/protected")
                .body(Body::empty())
                .unwrap(),
        )
        .await;

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(counter.load(Ordering::Relaxed), 1);
    }
}
