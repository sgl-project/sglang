//! sglang-server's integration tests: a whole frontend over real sockets,
//! with the test standing in for the scheduler through `sglang-server-test`.
//! One binary, laid out as `src/` is.

mod api_server;
mod common;
mod utils;
