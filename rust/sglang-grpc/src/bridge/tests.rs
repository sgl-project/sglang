use super::*;

#[test]
fn terminal_error_messages_include_request_id() {
    let error = TerminalError::ClientDisconnected {
        rid: "rid".to_string(),
    };
    assert!(error.message().contains("rid"));
}
