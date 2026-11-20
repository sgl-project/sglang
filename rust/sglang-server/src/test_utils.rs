//! What `sglang-server-test` builds its scheduler-less server on. Not a
//! stable API: nothing else may depend on it.

// ---- The server: the runtime, and the launch a test gives it ----

pub use crate::message::config::{DeploymentInfo, RuntimeConfig, RustServerEndpoint};
pub use crate::utils::runtime::{Runtime, start};

// ---- The scheduler's side of the ring: the frames a test answers with ----

pub use crate::message::response::{
    BatchHeader, frame_control_result, frame_decode_batch_cols, frame_error,
};

// ---- Helpers a model package's tests call as they are ----

pub use crate::api_server::http::pre_generate::test_utils::{PreGenerateAborts, pre_generate_app};
pub use crate::utils::logging::{CapturedLogs, capture_logs};

// ---- Launch-argument types an integration test must name to set ----
//
// `ServerArgs` and its fields are public, so a test sets most arguments in
// place. These are the field types it has to construct, which the crate
// does not otherwise export.

pub use crate::message::config::{ApiKeys, PreferredSamplingParams};
