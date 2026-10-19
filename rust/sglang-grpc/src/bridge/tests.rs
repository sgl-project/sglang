use super::*;

#[test]
fn terminal_error_messages_include_request_id() {
    let error = TerminalError::ClientDisconnected {
        rid: "rid".to_string(),
    };
    assert!(error.message().contains("rid"));
}

#[test]
fn unavailable_callback_preserves_shutdown_rejection() {
    assert_eq!(
        callback_error_status("shutting down".into(), Some(503)).code(),
        tonic::Code::Unavailable
    );
    assert_eq!(
        callback_error_status("other error".into(), None).code(),
        tonic::Code::Internal
    );
}
