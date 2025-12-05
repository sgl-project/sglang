//! WASM Guest Logging Example for sgl-model-gateway
//!
//! This example demonstrates logging and tracing middleware
//! for sgl-model-gateway using the WebAssembly Component Model.
//!
//! Features:
//! - Request tracking and tracing headers
//! - Response status code conversion

wit_bindgen::generate!({
    path: "../../../src/wasm/interface",
    world: "sgl-model-gateway",
});

use exports::sgl::router::{
    middleware_on_request::Guest as OnRequestGuest,
    middleware_on_response::Guest as OnResponseGuest,
};
use sgl::router::middleware_types::{Action, Header, ModifyAction, Request, Response};

/// Main middleware implementation
struct Middleware;

// Helper function to create header
fn create_header(name: &str, value: &str) -> Header {
    Header {
        name: name.to_string(),
        value: value.to_string(),
    }
}

// Implement on-request interface
impl OnRequestGuest for Middleware {
    fn on_request(req: Request) -> Action {
        let mut modify_action = ModifyAction {
            status: None,
            headers_set: vec![],
            headers_add: vec![],
            headers_remove: vec![],
            body_replace: None,
        };

        // Request Logging and Tracing
        // Add tracing headers with request ID
        modify_action
            .headers_add
            .push(create_header("x-request-id", &req.request_id));
        modify_action
            .headers_add
            .push(create_header("x-wasm-processed", "true"));
        modify_action.headers_add.push(create_header(
            "x-processed-at",
            &req.now_epoch_ms.to_string(),
        ));

        // Add custom header for API requests
        if req.path.starts_with("/api") || req.path.starts_with("/v1") {
            modify_action
                .headers_add
                .push(create_header("x-api-route", "true"));
        }

        Action::Modify(modify_action)
    }
}

// Implement on-response interface
impl OnResponseGuest for Middleware {
    fn on_response(resp: Response) -> Action {
        // Status code conversion: Convert 500 to 503 for better client handling
        if resp.status == 500 {
            let modify_action = ModifyAction {
                status: Some(503),
                headers_set: vec![],
                headers_add: vec![],
                headers_remove: vec![],
                body_replace: None,
            };
            Action::Modify(modify_action)
        } else {
            // No modification needed
            Action::Continue
        }
    }
}

// Export the component
export!(Middleware);
