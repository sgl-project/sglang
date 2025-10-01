use serde::{Deserialize, Serialize};

wasmtime::component::bindgen!({
    path: "src/wasm/wit",
    world: "sgl-router",
});

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WitMiddlewareHeader {
    pub name: String,
    pub value: String,
}

// onRequest
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WitMiddlewareHeaderOnRequest {
    pub method: String,
    pub path: String,
    pub query: String,
    pub headers: Vec<WitMiddlewareHeader>,
    pub body: Vec<u8>,
    #[serde(rename = "request-id")]
    pub request_id: String,
    #[serde(rename = "now-epoch-ms")]
    pub now_epoch_ms: u64,
}

impl WitMiddlewareHeaderOnRequest {
    pub fn to_wit_request(&self) -> sgl::router::middleware_types::Request {
        sgl::router::middleware_types::Request {
            method: self.method.clone(),
            path: self.path.clone(),
            query: self.query.clone(),
            headers: self
                .headers
                .iter()
                .map(|h| sgl::router::middleware_types::Header {
                    name: h.name.clone(),
                    value: h.value.clone(),
                })
                .collect(),
            body: self.body.clone(),
            request_id: self.request_id.clone(),
            now_epoch_ms: self.now_epoch_ms,
        }
    }
}

// onResponse
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WitMiddlewareHeaderOnResponse {
    pub status: u16,
    pub headers: Vec<WitMiddlewareHeader>,
    pub body: Vec<u8>,
}

// Return actions
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum WitMiddlewareAction {
    Continue,
    Reject(u16),
    Modify(WitMiddlewareModifyAction),
}

impl WitMiddlewareAction {
    pub fn from_wit(action: sgl::router::middleware_types::Action) -> Self {
        use sgl::router::middleware_types::Action;
        match action {
            Action::Continue => Self::Continue,
            Action::Reject(status) => Self::Reject(status),
            Action::Modify(modify) => Self::Modify(WitMiddlewareModifyAction {
                status: modify.status,
                headers_set: modify
                    .headers_set
                    .into_iter()
                    .map(|h| WitMiddlewareHeader {
                        name: h.name,
                        value: h.value,
                    })
                    .collect(),
                headers_add: modify
                    .headers_add
                    .into_iter()
                    .map(|h| WitMiddlewareHeader {
                        name: h.name,
                        value: h.value,
                    })
                    .collect(),
                headers_remove: modify.headers_remove,
                body_replace: modify.body_replace,
            }),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WitMiddlewareModifyAction {
    pub status: Option<u16>,
    #[serde(rename = "headers-set")]
    pub headers_set: Vec<WitMiddlewareHeader>,
    #[serde(rename = "headers-add")]
    pub headers_add: Vec<WitMiddlewareHeader>,
    #[serde(rename = "headers-remove")]
    pub headers_remove: Vec<String>,
    #[serde(rename = "body-replace")]
    pub body_replace: Option<Vec<u8>>,
}
