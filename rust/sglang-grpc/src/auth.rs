use std::sync::Arc;
use std::task::{Context, Poll};

use subtle::ConstantTimeEq;
use tonic::codegen::{Service, http};
use tonic::service::interceptor::InterceptedService;
use tonic::{Request, Status, service::Interceptor};

#[derive(Clone)]
pub struct ApiKeyAuth {
    api_key: Option<Arc<str>>,
    admin_api_key: Option<Arc<str>>,
}

#[derive(Clone, Copy)]
enum AuthLevel {
    Public,
    Normal,
    Admin,
}

fn auth_level(path: &str) -> AuthLevel {
    match path.strip_prefix("/sglang.runtime.v1.SglangService/") {
        Some("HealthCheck") => AuthLevel::Public,
        Some(
            "TextGenerate" | "Generate" | "TextEmbed" | "Embed" | "Classify" | "Tokenize"
            | "Detokenize" | "WatchEngineState" | "GetModelInfo" | "GetServerInfo" | "ListModels"
            | "GetLoad" | "ChatComplete" | "Complete" | "OpenAIEmbed" | "OpenAIClassify" | "Score"
            | "Rerank",
        ) => AuthLevel::Normal,
        // New RPCs require admin credentials until explicitly classified.
        _ => AuthLevel::Admin,
    }
}

#[derive(Clone)]
pub struct AuthService<S> {
    inner: InterceptedService<S, ApiKeyAuth>,
}

impl<S> AuthService<S> {
    pub fn new(service: S, auth: ApiKeyAuth) -> Self {
        Self {
            inner: InterceptedService::new(service, auth),
        }
    }
}

impl<S: tonic::server::NamedService> tonic::server::NamedService for AuthService<S> {
    const NAME: &'static str = S::NAME;
}

impl<S, B> Service<http::Request<B>> for AuthService<S>
where
    InterceptedService<S, ApiKeyAuth>: Service<http::Request<B>>,
{
    type Response = <InterceptedService<S, ApiKeyAuth> as Service<http::Request<B>>>::Response;
    type Error = <InterceptedService<S, ApiKeyAuth> as Service<http::Request<B>>>::Error;
    type Future = <InterceptedService<S, ApiKeyAuth> as Service<http::Request<B>>>::Future;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, mut request: http::Request<B>) -> Self::Future {
        let level = auth_level(request.uri().path());
        request.extensions_mut().insert(level);
        self.inner.call(request)
    }
}

impl ApiKeyAuth {
    pub fn new(api_key: Option<String>, admin_api_key: Option<String>) -> Self {
        Self {
            api_key: api_key.filter(|key| !key.is_empty()).map(Arc::from),
            admin_api_key: admin_api_key.filter(|key| !key.is_empty()).map(Arc::from),
        }
    }
}

impl Interceptor for ApiKeyAuth {
    fn call(&mut self, request: Request<()>) -> Result<Request<()>, Status> {
        let expected = match request
            .extensions()
            .get::<AuthLevel>()
            .unwrap_or(&AuthLevel::Admin)
        {
            AuthLevel::Public => None,
            AuthLevel::Normal => self.api_key.as_ref(),
            AuthLevel::Admin => self.admin_api_key.as_ref().or(self.api_key.as_ref()),
        };
        let Some(expected) = expected else {
            return Ok(request);
        };
        let mut values = request.metadata().get_all("authorization").iter();
        let token = values
            .next()
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.split_once(' '))
            .filter(|(scheme, _)| scheme.eq_ignore_ascii_case("bearer"))
            .map(|(_, token)| token);
        if values.next().is_none()
            && token.is_some_and(|token| bool::from(token.as_bytes().ct_eq(expected.as_bytes())))
        {
            Ok(request)
        } else {
            Err(Status::unauthenticated("Unauthorized"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bearer_tokens_are_required_without_changing_unauthenticated_mode() {
        for value in [
            None,
            Some("Bearer wrong"),
            Some("Basic secret"),
            Some("Bearer secret "),
        ] {
            let mut request = Request::new(());
            if let Some(value) = value {
                request
                    .metadata_mut()
                    .insert("authorization", value.parse().unwrap());
            }
            assert_eq!(
                ApiKeyAuth::new(Some("secret".into()), None)
                    .call(request)
                    .unwrap_err()
                    .code(),
                tonic::Code::Unauthenticated,
            );
        }
        for value in ["Bearer secret", "bearer secret", "BEARER secret"] {
            let mut request = Request::new(());
            request
                .metadata_mut()
                .insert("authorization", value.parse().unwrap());
            assert!(
                ApiKeyAuth::new(Some("secret".into()), None)
                    .call(request)
                    .is_ok()
            );
        }
        for key in [None, Some(String::new())] {
            assert!(ApiKeyAuth::new(key, None).call(Request::new(())).is_ok());
        }
    }

    #[test]
    fn ambiguous_credentials_are_rejected() {
        let mut request = Request::new(());
        for value in ["Bearer secret", "Bearer wrong"] {
            request
                .metadata_mut()
                .append("authorization", value.parse().unwrap());
        }
        assert_eq!(
            ApiKeyAuth::new(Some("secret".into()), None)
                .call(request)
                .unwrap_err()
                .code(),
            tonic::Code::Unauthenticated,
        );
    }

    #[test]
    fn unknown_paths_and_missing_classification_require_admin_credentials() {
        for path in [
            "/sglang.runtime.v1.SglangService/NewControl",
            "/other.Service/HealthCheck",
            "/sglang.runtime.v1.SglangService/HealthCheck/extra",
        ] {
            let mut request = Request::new(());
            request.extensions_mut().insert(auth_level(path));
            request
                .metadata_mut()
                .insert("authorization", "Bearer user".parse().unwrap());
            assert_eq!(
                ApiKeyAuth::new(Some("user".into()), Some("admin".into()))
                    .call(request)
                    .unwrap_err()
                    .code(),
                tonic::Code::Unauthenticated,
            );
        }
        assert_eq!(
            ApiKeyAuth::new(None, Some("admin".into()))
                .call(Request::new(()))
                .unwrap_err()
                .code(),
            tonic::Code::Unauthenticated,
        );
    }
}
