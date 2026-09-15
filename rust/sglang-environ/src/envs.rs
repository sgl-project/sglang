//! Every environment variable the Rust crates read, declared once. The
//! declaration owns the name and the default; call sites only do
//! `envs::SGLANG_FOO.get()`, mirroring Python's `envs.SGLANG_FOO.get()`.

use crate::{EnvBool, EnvInt, EnvU64, EnvUsize};

// HTTP server
pub const SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION: EnvBool =
    EnvBool::new("SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION", true);
pub const SGLANG_HEALTH_CHECK_TIMEOUT: EnvInt = EnvInt::new("SGLANG_HEALTH_CHECK_TIMEOUT", 20);
pub const SGLANG_MAX_BATCH_REQS_PER_HTTP_REQ: EnvInt =
    EnvInt::new("SGLANG_MAX_BATCH_REQS_PER_HTTP_REQ", 4096);

// Disaggregation bootstrap
pub const SGLANG_DISAGGREGATION_BOOTSTRAP_ENTRY_CLEANUP_INTERVAL: EnvInt = EnvInt::new(
    "SGLANG_DISAGGREGATION_BOOTSTRAP_ENTRY_CLEANUP_INTERVAL",
    120,
);

// gRPC server
pub const SGLANG_TONIC_PAYLOAD: EnvUsize = EnvUsize::new("SGLANG_TONIC_PAYLOAD", 64 * 1024 * 1024);

// Multimodal preprocessing
pub const REQUEST_TIMEOUT: EnvU64 = EnvU64::new("REQUEST_TIMEOUT", 3);
/// 0 means "size the pool to the machine" (see `sglang_mm::common::pool`).
pub const SGL_MM_RS_THREADS: EnvUsize = EnvUsize::new("SGL_MM_RS_THREADS", 0);
