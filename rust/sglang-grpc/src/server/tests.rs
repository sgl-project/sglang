use super::{
    DEFAULT_GRPC_MAX_MESSAGE_SIZE, openai_status_code, resolve_max_message_size,
    terminal_error_status,
};
use crate::bridge::TerminalError;
use std::collections::HashMap;
use tonic::Code;

#[test]
fn openai_status_code_uses_forwarded_status_when_present() {
    let meta_info = HashMap::from([(String::from("status_code"), String::from("429"))]);
    assert_eq!(openai_status_code(&meta_info, 200), 429);
}

#[test]
fn openai_status_code_falls_back_when_missing_or_invalid() {
    assert_eq!(openai_status_code(&HashMap::new(), 200), 200);

    let meta_info = HashMap::from([(String::from("status_code"), String::from("not-an-int"))]);
    assert_eq!(openai_status_code(&meta_info, 200), 200);
}

#[test]
fn terminal_error_status_maps_channel_full_to_resource_exhausted() {
    let status = terminal_error_status(TerminalError::ChannelFull {
        rid: "rid".to_string(),
    });

    assert_eq!(status.code(), Code::ResourceExhausted);
}

#[test]
fn terminal_error_status_maps_abort_to_cancelled() {
    let status = terminal_error_status(TerminalError::Aborted {
        rid: "rid".to_string(),
    });

    assert_eq!(status.code(), Code::Cancelled);
}

// SAFETY: env vars are process-global; bundle all SGLANG_TONIC_PAYLOAD cases into one
// serial test so they don't race each other under `cargo test`'s default parallelism.
#[test]
fn resolve_max_message_size_honors_env_var() {
    const VAR: &str = "SGLANG_TONIC_PAYLOAD";

    // Unset → default.
    // SAFETY: single-threaded test mutating process env (see note above).
    unsafe {
        std::env::remove_var(VAR);
    }
    assert_eq!(resolve_max_message_size(), DEFAULT_GRPC_MAX_MESSAGE_SIZE);

    // Valid override → honored verbatim.
    unsafe {
        std::env::set_var(VAR, "1048576");
    }
    assert_eq!(resolve_max_message_size(), 1_048_576);

    // Invalid string → warn + fall back to default.
    unsafe {
        std::env::set_var(VAR, "not-a-number");
    }
    assert_eq!(resolve_max_message_size(), DEFAULT_GRPC_MAX_MESSAGE_SIZE);

    // Zero → treated as invalid, fall back to default.
    unsafe {
        std::env::set_var(VAR, "0");
    }
    assert_eq!(resolve_max_message_size(), DEFAULT_GRPC_MAX_MESSAGE_SIZE);

    unsafe {
        std::env::remove_var(VAR);
    }
}

#[tokio::test]
async fn start_profile_preserves_optional_steps_and_rejects_nonpositive_limits() {
    use crate::proto::sglang_service_server::SglangService;
    use prost::Message;
    use pyo3::prelude::*;
    use pyo3::types::PyModule;
    use std::sync::Arc;
    use tonic::Request;

    Python::initialize();
    let handle = Python::attach(|py| {
        PyModule::from_code(
            py,
            c"
class Runtime:
    def __init__(self):
        self.calls = []

    def start_profile(self, output_dir, callback, num_steps):
        self.calls.append((output_dir, num_steps))
        callback(b'{\"message\": \"Profiling started.\"}', finished=True)
",
            c"profile_test.py",
            c"profile_test",
        )
        .unwrap()
        .getattr("Runtime")
        .unwrap()
        .call0()
        .unwrap()
        .unbind()
    });
    let service = super::SglangServiceImpl {
        bridge: Arc::new(crate::bridge::PyBridge::new(
            Python::attach(|py| handle.clone_ref(py)),
            None,
            1024,
            16,
            tokio::runtime::Handle::current(),
        )),
        response_timeout: std::time::Duration::from_secs(5),
    };
    // Empty bytes are a valid legacy request with no optional fields.
    let legacy = crate::proto::StartProfileRequest::decode(&[][..]).unwrap();
    assert_eq!(legacy.num_steps, None);
    service.start_profile(Request::new(legacy)).await.unwrap();
    for steps in [1, 10, i32::MAX] {
        let request = crate::proto::StartProfileRequest {
            output_dir: Some("/tmp/profile".into()),
            num_steps: Some(steps),
        };
        let decoded =
            crate::proto::StartProfileRequest::decode(request.encode_to_vec().as_slice()).unwrap();
        service.start_profile(Request::new(decoded)).await.unwrap();
    }
    for steps in [0, -1, i32::MIN] {
        let error = service
            .start_profile(Request::new(crate::proto::StartProfileRequest {
                output_dir: None,
                num_steps: Some(steps),
            }))
            .await
            .unwrap_err();
        assert_eq!(error.code(), Code::InvalidArgument);
    }
    Python::attach(|py| {
        let calls: Vec<(Option<String>, Option<i32>)> =
            handle.getattr(py, "calls").unwrap().extract(py).unwrap();
        assert_eq!(
            calls,
            vec![
                (None, None),
                (Some("/tmp/profile".into()), Some(1)),
                (Some("/tmp/profile".into()), Some(10)),
                (Some("/tmp/profile".into()), Some(i32::MAX)),
            ]
        );
    });
}
