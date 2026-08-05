use super::{
    ChoiceTracker, DEFAULT_GRPC_MAX_MESSAGE_SIZE, GenerationTerminal, generation_terminal,
    openai_status_code, resolve_max_message_size, terminal_error_status,
};
use crate::bridge::TerminalError;
use crate::proto;
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

#[test]
fn choice_tracker_requires_one_terminal_per_choice() {
    let mut tracker = ChoiceTracker::new(2);

    assert!(!tracker.observe(0, false, false).unwrap());
    assert!(!tracker.observe(0, true, false).unwrap());
    assert!(tracker.observe(1, true, true).unwrap());
    assert!(tracker.observe(1, false, false).is_err());
    assert!(ChoiceTracker::new(2).observe(2, false, false).is_err());
    assert!(ChoiceTracker::new(2).observe(0, true, true).is_err());
}

#[test]
fn generation_terminal_maps_stop_and_scheduler_error() {
    let stop = serde_json::json!({"type": "stop", "matched": "END"});
    match generation_terminal(Some(&stop)) {
        GenerationTerminal::Finish(finish) => {
            assert_eq!(finish.reason, proto::FinishReason::Stop as i32);
            assert!(matches!(
                finish.stop_reason.and_then(|reason| reason.reason),
                Some(proto::stop_reason::Reason::MatchedString(value)) if value == "END"
            ));
        }
        GenerationTerminal::Error(_) => panic!("expected a finish terminal"),
    }

    let error = serde_json::json!({
        "type": "error",
        "status_code": 503,
        "message": "busy"
    });
    match generation_terminal(Some(&error)) {
        GenerationTerminal::Error(error) => {
            assert_eq!(error.code, proto::GenerationErrorCode::Unavailable as i32);
            assert!(error.retryable);
            assert_eq!(error.message, "busy");
        }
        GenerationTerminal::Finish(_) => panic!("expected an error terminal"),
    }
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
