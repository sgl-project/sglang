//! Rust radix tree core for SGLang's KV cache, exposed to Python as `mem_cache`.
// TODO(Jialin): Replace recoverable panics with explicit Rust errors and map
// them to PyErr at the Python boundary.
#![allow(
    dead_code,
    unsafe_op_in_unsafe_fn,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::owned_cow,
    clippy::panic,
    clippy::print_stdout,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::unimplemented,
    clippy::unreachable,
    clippy::useless_conversion
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
