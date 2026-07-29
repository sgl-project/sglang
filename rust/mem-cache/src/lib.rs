//! Rust radix tree core for SGLang's KV cache, exposed to Python as `mem_cache`.
// Panics are the tree core's contract: asserts mirror the python reference's
// invariants.
// TODO(Jialin): burn down these allows; boundary panics convert to PyErr.
#![allow(
    dead_code,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::print_stdout,
    clippy::unimplemented,
    clippy::unreachable
)]

mod components;
mod node;
#[cfg(feature = "python-extension")]
mod python_bindings;
#[cfg(test)]
#[path = "tests/test_utils.rs"]
pub(crate) mod test_utils;
mod unified_lru_list;
mod unified_tree_core;
