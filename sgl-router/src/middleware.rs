use actix_web::{
    dev::{forward_ready, Service, ServiceRequest, ServiceResponse, Transform},
    Error, HttpMessage, HttpRequest,
};
use futures_util::future::LocalBoxFuture;
use std::future::{ready, Ready};

/// Generate OpenAI-compatible request ID based on endpoint
fn generate_request_id(path: &str) -> String {
    let prefix = if path.contains("/chat/completions") {
        "chatcmpl-"
    } else if path.contains("/completions") {
        "cmpl-"
    } else if path.contains("/generate") {
        "gnt-"
    } else {
        "req-"
    };

    // Generate a random string similar to OpenAI's format
    let random_part: String = (0..24)
        .map(|_| {
            let chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
            chars
                .chars()
                .nth(rand::random::<usize>() % chars.len())
                .unwrap()
        })
        .collect();

    format!("{}{}", prefix, random_part)
}

/// Extract request ID from request extensions or generate a new one
pub fn get_request_id(req: &HttpRequest) -> String {
    req.extensions()
        .get::<String>()
        .cloned()
        .unwrap_or_else(|| generate_request_id(req.path()))
}

/// Middleware for injecting request ID into request extensions
pub struct RequestIdMiddleware {
    headers: Vec<String>,
}

impl RequestIdMiddleware {
    pub fn new(headers: Vec<String>) -> Self {
        Self { headers }
    }
}

impl<S, B> Transform<S, ServiceRequest> for RequestIdMiddleware
where
    S: Service<ServiceRequest, Response = ServiceResponse<B>, Error = Error>,
    S::Future: 'static,
    B: 'static,
{
    type Response = ServiceResponse<B>;
    type Error = Error;
    type InitError = ();
    type Transform = RequestIdMiddlewareService<S>;
    type Future = Ready<Result<Self::Transform, Self::InitError>>;

    fn new_transform(&self, service: S) -> Self::Future {
        ready(Ok(RequestIdMiddlewareService {
            service,
            headers: self.headers.clone(),
        }))
    }
}

pub struct RequestIdMiddlewareService<S> {
    service: S,
    headers: Vec<String>,
}

impl<S, B> Service<ServiceRequest> for RequestIdMiddlewareService<S>
where
    S: Service<ServiceRequest, Response = ServiceResponse<B>, Error = Error>,
    S::Future: 'static,
    B: 'static,
{
    type Response = ServiceResponse<B>;
    type Error = Error;
    type Future = LocalBoxFuture<'static, Result<Self::Response, Self::Error>>;

    forward_ready!(service);

    fn call(&self, req: ServiceRequest) -> Self::Future {
        // Extract request ID from headers or generate new one
        let mut request_id = None;

        for header_name in &self.headers {
            if let Some(header_value) = req.headers().get(header_name) {
                if let Ok(value) = header_value.to_str() {
                    request_id = Some(value.to_string());
                    break;
                }
            }
        }

        let request_id = request_id.unwrap_or_else(|| generate_request_id(req.path()));

        // Insert request ID into request extensions
        req.extensions_mut().insert(request_id);

        let fut = self.service.call(req);
        Box::pin(async move { fut.await })
    }
}
