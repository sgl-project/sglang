//! Parser module for function calls and reasoning extraction
//!
//! This module provides parsing operations for model output, including:
//! - Function call extraction from text
//! - Reasoning separation from normal text

mod handlers;

pub use handlers::{parse_function_call, parse_reasoning};
