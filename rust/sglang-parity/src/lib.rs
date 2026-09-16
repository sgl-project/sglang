//! End-to-end parity testing for SGLang implementations.
//!
//! API suites supply requests and validation for an explicit check target. This library
//! owns service lifetimes, transport, declared comparison rules, and reports.
//! Raw JSON is intentional at the API boundary: server DTOs could discard
//! unknown fields or fill defaults and conceal compatibility differences.
//!
//! ```
//! use serde_json::json;
//! use sglang_parity::compare_json;
//!
//! let differences = compare_json(&json!({"tokens": [1, 2]}), &json!({"tokens": [1, 3]}));
//! assert_eq!(differences[0].path, "/tokens/1");
//! // An omitted field is different from an explicit null.
//! assert!(!compare_json(&json!({}), &json!({"value": null})).is_empty());
//! ```

pub mod artifacts;
pub mod compare;
pub mod environment;
pub mod http;
pub mod plan;
pub mod process;
mod progress;
pub mod report;
pub mod runner;
pub mod sse;

pub use compare::{ComparisonRules, ComparisonScope, Difference, Violation, compare_json};
pub use http::{CaptureMode, HttpCase, HttpObservation, HttpRequest, Isolation};
pub use plan::{
    CheckTarget, ExecutionPlan, ProfilePlan, ProfileSpec, Requirements, ResolvedProfile,
};
pub use process::{Implementation, ServerConfig};
pub use runner::{
    EquivalenceValue, HttpSuite, PreparedResponse, Report, ResponsePolicy, RunConfig, RunError,
    describe, describe_plan, run, run_plan,
};
