//! The API-key gate, as one transport-generic tower layer (retires the
//! api_server TODO(auth)). Mirrors the Python NORMAL auth level
//! (`sglang.srt.utils.auth.decide_request_auth`): when an `api_key` is
//! configured, every route requires `Authorization: Bearer <api_key>` except
//! OPTIONS and the `/health*` / `/metrics*` prefixes (k8s probes and
//! Prometheus scrape without secrets) — plus the gRPC HealthCheck method,
//! the same probe by its RPC path. The rust surface has no admin endpoints,
//! so `admin_api_key` plays no role here.
//!
//! Rejections match each wire: HTTP answers Python's exact 401 body
//! (`{"error":"Unauthorized"}`); gRPC answers trailers-only UNAUTHENTICATED
//! (a status in headers, empty body).

use std::sync::Arc;
use std::task::{Context, Poll};

use tower::{Layer, Service};

use super::RejectionBody;

#[derive(Clone)]
pub(crate) struct ApiKeyAuthLayer {
    api_key: Arc<str>,
}

impl ApiKeyAuthLayer {
    pub(crate) fn new(api_key: &str) -> Self {
        ApiKeyAuthLayer {
            api_key: api_key.into(),
        }
    }
}

impl<S> Layer<S> for ApiKeyAuthLayer {
    type Service = ApiKeyAuth<S>;
    fn layer(&self, inner: S) -> Self::Service {
        ApiKeyAuth {
            inner,
            api_key: self.api_key.clone(),
        }
    }
}

#[derive(Clone)]
pub(crate) struct ApiKeyAuth<S> {
    inner: S,
    api_key: Arc<str>,
}

/// Constant-time bearer-token check (Python uses `secrets.compare_digest`).
fn bearer_matches(authorization: Option<&http::HeaderValue>, expected: &str) -> bool {
    let Some(value) = authorization.and_then(|v| v.to_str().ok()) else {
        return false;
    };
    let Some((scheme, token)) = value.split_once(' ') else {
        return false;
    };
    if !scheme.eq_ignore_ascii_case("bearer") {
        return false;
    }
    let (token, expected) = (token.as_bytes(), expected.as_bytes());
    let mut diff = token.len() ^ expected.len();
    for i in 0..token.len().max(expected.len()) {
        let a = token.get(i).copied().unwrap_or(0);
        let b = expected.get(i).copied().unwrap_or(0);
        diff |= usize::from(a ^ b);
    }
    diff == 0
}

/// The always-open surface: OPTIONS, the health/metrics prefixes (Python
/// parity), and the gRPC health probe by its method path.
fn exempt(method: &http::Method, path: &str) -> bool {
    method == http::Method::OPTIONS
        || path.starts_with("/health")
        || path.starts_with("/metrics")
        || path == "/sglang.api.v1.SglangApi/HealthCheck"
}

fn reject<RB: RejectionBody>(is_grpc: bool) -> http::Response<RB> {
    if is_grpc {
        // Trailers-only response: gRPC statuses ride the headers of an
        // empty-body HTTP 200. 16 = UNAUTHENTICATED.
        http::Response::builder()
            .status(http::StatusCode::OK)
            .header(http::header::CONTENT_TYPE, "application/grpc")
            .header("grpc-status", "16")
            .header("grpc-message", "Unauthorized")
            .body(RB::empty())
            .expect("static rejection response builds")
    } else {
        // Python's exact body: ORJSONResponse({"error": "Unauthorized"}, 401).
        http::Response::builder()
            .status(http::StatusCode::UNAUTHORIZED)
            .header(http::header::CONTENT_TYPE, "application/json")
            .body(RB::from_static(b"{\"error\":\"Unauthorized\"}"))
            .expect("static rejection response builds")
    }
}

impl<S, B, RB> Service<http::Request<B>> for ApiKeyAuth<S>
where
    S: Service<http::Request<B>, Response = http::Response<RB>>,
    S::Future: Send + 'static,
    RB: RejectionBody,
{
    type Response = S::Response;
    type Error = S::Error;
    type Future =
        futures::future::Either<std::future::Ready<Result<Self::Response, Self::Error>>, S::Future>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: http::Request<B>) -> Self::Future {
        use futures::future::Either;
        if exempt(req.method(), req.uri().path())
            || bearer_matches(
                req.headers().get(http::header::AUTHORIZATION),
                &self.api_key,
            )
        {
            return Either::Right(self.inner.call(req));
        }
        let is_grpc = req
            .headers()
            .get(http::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .is_some_and(|v| v.starts_with("application/grpc"));
        Either::Left(std::future::ready(Ok(reject(is_grpc))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bearer_check_accepts_only_the_exact_key() {
        let h = |s: &str| Some(http::HeaderValue::from_str(s).unwrap());
        assert!(bearer_matches(h("Bearer sk-1").as_ref(), "sk-1"));
        assert!(bearer_matches(h("bearer sk-1").as_ref(), "sk-1"));
        assert!(!bearer_matches(h("Bearer sk-2").as_ref(), "sk-1"));
        assert!(!bearer_matches(h("Bearer sk-11").as_ref(), "sk-1"));
        assert!(!bearer_matches(h("Basic sk-1").as_ref(), "sk-1"));
        assert!(!bearer_matches(h("sk-1").as_ref(), "sk-1"));
        assert!(!bearer_matches(None, "sk-1"));
    }

    #[test]
    fn exemptions_match_python_and_the_grpc_probe() {
        let get = http::Method::GET;
        assert!(exempt(&get, "/health"));
        assert!(exempt(&get, "/health_generate"));
        assert!(exempt(&get, "/metrics"));
        assert!(exempt(&http::Method::OPTIONS, "/generate"));
        assert!(exempt(
            &http::Method::POST,
            "/sglang.api.v1.SglangApi/HealthCheck"
        ));
        assert!(!exempt(&get, "/generate"));
        assert!(!exempt(&get, "/v1/chat/completions"));
        assert!(!exempt(
            &http::Method::POST,
            "/sglang.api.v1.SglangApi/Generate"
        ));
    }

    /// Through the layered HTTP service: no header -> Python's exact 401
    /// body; the right key -> through; /health exempt without a key.
    #[tokio::test]
    async fn layered_service_gate() {
        use http_body_util::BodyExt;
        use tower::ServiceExt;

        use crate::api_server::http::{HttpBody, empty, text_response};

        let app = || {
            ApiKeyAuthLayer::new("sk-1").layer(tower::service_fn(
                |_req: http::Request<HttpBody>| async {
                    Ok::<_, std::convert::Infallible>(text_response(
                        http::StatusCode::OK,
                        "ok".to_string(),
                    ))
                },
            ))
        };
        let get = |path: &str, auth: Option<&str>| {
            let mut b = http::Request::builder().uri(path);
            if let Some(a) = auth {
                b = b.header(http::header::AUTHORIZATION, a);
            }
            b.body(empty()).unwrap()
        };

        let denied = app().oneshot(get("/generate", None)).await.unwrap();
        assert_eq!(denied.status(), http::StatusCode::UNAUTHORIZED);
        let body = denied.into_body().collect().await.unwrap().to_bytes();
        assert_eq!(&body[..], b"{\"error\":\"Unauthorized\"}");

        let wrong = app()
            .oneshot(get("/generate", Some("Bearer nope")))
            .await
            .unwrap();
        assert_eq!(wrong.status(), http::StatusCode::UNAUTHORIZED);

        let allowed = app()
            .oneshot(get("/generate", Some("Bearer sk-1")))
            .await
            .unwrap();
        assert_eq!(allowed.status(), http::StatusCode::OK);

        let probe = app().oneshot(get("/health", None)).await.unwrap();
        assert_eq!(probe.status(), http::StatusCode::OK);
    }

    /// The two rejection wire shapes: Python's 401 body on HTTP, a
    /// trailers-only UNAUTHENTICATED on gRPC.
    #[test]
    fn rejection_shapes() {
        let http_reject: http::Response<crate::api_server::http::HttpBody> = reject(false);
        assert_eq!(http_reject.status(), http::StatusCode::UNAUTHORIZED);
        let grpc_reject: http::Response<tonic::body::Body> = reject(true);
        assert_eq!(grpc_reject.status(), http::StatusCode::OK);
        assert_eq!(
            grpc_reject.headers().get("grpc-status").unwrap(),
            &http::HeaderValue::from_static("16")
        );
    }
}
