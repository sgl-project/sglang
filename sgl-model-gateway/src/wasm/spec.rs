//! WebAssembly Interface Bindings and Type Conversions
//!
//! Contains wasmtime component bindings generated from interface definitions,
//! and helper functions to convert between Axum HTTP types and interface types.

use axum::http::{header, HeaderMap, HeaderValue};

wasmtime::component::bindgen!({
    path: "src/wasm/interface",
    world: "sgl-model-gateway",
    imports: { default: async | trappable },
    exports: { default: async },
});

/// Build WebAssembly headers from Axum HeaderMap
pub fn build_wasm_headers_from_axum_headers(
    headers: &HeaderMap,
) -> Vec<sgl::model_gateway::middleware_types::Header> {
    let mut wasm_headers = Vec::new();
    for (name, value) in headers.iter() {
        if let Ok(value_str) = value.to_str() {
            wasm_headers.push(sgl::model_gateway::middleware_types::Header {
                name: name.as_str().to_string(),
                value: value_str.to_string(),
            });
        }
    }
    wasm_headers
}

/// Apply ModifyAction header modifications to Axum HeaderMap
pub fn apply_modify_action_to_headers(
    headers: &mut HeaderMap,
    modify: &sgl::model_gateway::middleware_types::ModifyAction,
) {
    // Apply headers_set
    for header_mod in &modify.headers_set {
        if let (Ok(name), Ok(value)) = (
            header_mod.name.parse::<header::HeaderName>(),
            header_mod.value.parse::<HeaderValue>(),
        ) {
            headers.insert(name, value);
        }
    }
    // Apply headers_add
    for header_mod in &modify.headers_add {
        if let (Ok(name), Ok(value)) = (
            header_mod.name.parse::<header::HeaderName>(),
            header_mod.value.parse::<HeaderValue>(),
        ) {
            headers.append(name, value);
        }
    }
    // Apply headers_remove
    for name_str in &modify.headers_remove {
        if let Ok(name) = name_str.parse::<header::HeaderName>() {
            headers.remove(name);
        }
    }
}
