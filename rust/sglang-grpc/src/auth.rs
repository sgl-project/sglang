use std::sync::Arc;

use subtle::ConstantTimeEq;
use tonic::{Request, Status, service::Interceptor};

#[derive(Clone)]
pub struct ApiKeyAuth(Option<Arc<str>>);

impl ApiKeyAuth {
    pub fn new(api_key: Option<String>) -> Self {
        Self(api_key.filter(|key| !key.is_empty()).map(Arc::from))
    }
}

impl Interceptor for ApiKeyAuth {
    fn call(&mut self, request: Request<()>) -> Result<Request<()>, Status> {
        let Some(expected) = &self.0 else {
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
                ApiKeyAuth::new(Some("secret".into()))
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
            assert!(ApiKeyAuth::new(Some("secret".into())).call(request).is_ok());
        }
        for key in [None, Some(String::new())] {
            assert!(ApiKeyAuth::new(key).call(Request::new(())).is_ok());
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
            ApiKeyAuth::new(Some("secret".into()))
                .call(request)
                .unwrap_err()
                .code(),
            tonic::Code::Unauthenticated,
        );
    }
}
