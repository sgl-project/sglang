//! WebAssembly (WASM) module support for SGL Model Gateway
//!
//! This module re-exports the smg-wasm crate and provides HTTP API routes.

// Re-export everything from smg-wasm
pub use smg_wasm::*;

// Local HTTP API routes (depends on app-specific types)
pub mod route;
