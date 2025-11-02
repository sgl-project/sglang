//! WebAssembly (WASM) module support for sgl-router
//!
//! This module provides WASM component execution capabilities using the WebAssembly Component Model (WIT).
//! It supports middleware execution at various attach points (OnRequest, OnResponse) with async support.

pub mod config;
pub mod errors;
pub mod module;
pub mod module_manager;
pub mod route;
pub mod runtime;
pub mod spec;
pub mod types;
