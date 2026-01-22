//! WASM Component Type System
//!
//! Provides generic input/output types for WASM component execution
//! based on attach points.

use wasmtime::{component::ResourceTable, StoreLimits};
use wasmtime_wasi::{WasiCtx, WasiCtxView, WasiView};

use crate::wasm::{
    module::{MiddlewareAttachPoint, WasmModuleAttachPoint},
    spec::sgl::model_gateway::middleware_types,
};

/// Generic input type for WASM component execution
///
/// This enum represents all possible input types that can be passed
/// to a WASM component, determined by the attach_point.
#[derive(Debug, Clone)]
pub enum WasmComponentInput {
    /// Middleware OnRequest input
    MiddlewareRequest(middleware_types::Request),
    /// Middleware OnResponse input
    MiddlewareResponse(middleware_types::Response),
}

/// Generic output type from WASM component execution
///
/// This enum represents all possible output types that can be returned
/// from a WASM component, determined by the attach_point.
#[derive(Debug, Clone)]
pub enum WasmComponentOutput {
    /// Middleware Action output
    MiddlewareAction(middleware_types::Action),
}

impl WasmComponentInput {
    /// Create input based on attach_point and raw data
    ///
    /// This helper function validates that the attach_point matches
    /// the expected input type.
    pub fn from_attach_point(attach_point: &WasmModuleAttachPoint) -> Result<Self, String> {
        match attach_point {
            WasmModuleAttachPoint::Middleware(MiddlewareAttachPoint::OnRequest) => {
                // OnRequest expects a Request type, but we can't construct it here
                // The caller should use MiddlewareRequest variant directly
                Err("OnRequest requires MiddlewareRequest input. Use WasmComponentInput::MiddlewareRequest directly.".to_string())
            }
            WasmModuleAttachPoint::Middleware(MiddlewareAttachPoint::OnResponse) => {
                // OnResponse expects a Response type
                Err("OnResponse requires MiddlewareResponse input. Use WasmComponentInput::MiddlewareResponse directly.".to_string())
            }
            WasmModuleAttachPoint::Middleware(MiddlewareAttachPoint::OnError) => {
                Err("OnError attach point not yet implemented".to_string())
            }
        }
    }

    /// Get the expected attach_point for this input type
    pub fn expected_attach_point(&self) -> WasmModuleAttachPoint {
        match self {
            WasmComponentInput::MiddlewareRequest(_) => {
                WasmModuleAttachPoint::Middleware(MiddlewareAttachPoint::OnRequest)
            }
            WasmComponentInput::MiddlewareResponse(_) => {
                WasmModuleAttachPoint::Middleware(MiddlewareAttachPoint::OnResponse)
            }
        }
    }
}

impl WasmComponentOutput {
    /// Get the attach_point that produced this output type
    pub fn from_attach_point(attach_point: &WasmModuleAttachPoint) -> Result<Self, String> {
        match attach_point {
            WasmModuleAttachPoint::Middleware(MiddlewareAttachPoint::OnRequest) => {
                // This would be set after execution
                Err("Cannot create output before execution".to_string())
            }
            WasmModuleAttachPoint::Middleware(MiddlewareAttachPoint::OnResponse) => {
                Err("Cannot create output before execution".to_string())
            }
            WasmModuleAttachPoint::Middleware(MiddlewareAttachPoint::OnError) => {
                Err("OnError attach point not yet implemented".to_string())
            }
        }
    }
}

pub struct WasiState {
    pub ctx: WasiCtx,
    pub table: ResourceTable,
    pub limits: StoreLimits,
}

impl WasiView for WasiState {
    fn ctx(&mut self) -> WasiCtxView<'_> {
        WasiCtxView {
            ctx: &mut self.ctx,
            table: &mut self.table,
        }
    }
}
