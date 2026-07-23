use super::*;

#[test]
fn terminal_error_messages_include_request_id() {
    let error = TerminalError::ClientDisconnected {
        rid: "rid".to_string(),
    };
    assert!(error.message().contains("rid"));
}

#[tokio::test]
async fn stale_request_key_cannot_abort_reused_request_id() {
    Python::initialize();
    let runtime_handle = Python::attach(|py| PyDict::new(py).clone().unbind().into_any());
    let bridge = PyBridge::new(
        runtime_handle,
        None,
        1,
        1,
        tokio::runtime::Handle::current(),
    );
    let (sender, _receiver) = tokio::sync::mpsc::channel(1);
    let old_key = RequestKey {
        rid: "reused".to_string(),
        incarnation: 1,
    };
    let current_key = RequestKey {
        rid: "reused".to_string(),
        incarnation: 2,
    };
    {
        let mut state = lock_or_recover(bridge.state.as_ref(), "state");
        state.channels.insert(
            current_key.rid.clone(),
            ActiveChannel {
                incarnation: current_key.incarnation,
                sender,
            },
        );
        state.pending_sends.insert(old_key.clone());
    }

    // The dict has no `abort` method, so this also proves the stale path never
    // invokes the Python abort callback for the new incarnation.
    bridge.abort_request(&old_key).unwrap();

    let state = lock_or_recover(bridge.state.as_ref(), "state");
    assert_eq!(
        state.channels.get(current_key.rid()).unwrap().incarnation,
        current_key.incarnation
    );
    assert!(!state.pending_sends.contains(&old_key));
}

#[test]
fn metadata_modes_preserve_legacy_json_and_parse_typed_values() {
    Python::initialize();
    Python::attach(|py| {
        let chunk = PyDict::new(py);
        let meta = PyDict::new(py);
        meta.set_item("number", 7).unwrap();
        meta.set_item("text", "hello").unwrap();
        meta.set_item("array", vec![1, 2]).unwrap();
        chunk.set_item("meta_info", meta).unwrap();

        let legacy = extract_legacy_meta_info(&chunk);
        assert_eq!(legacy["number"], "7");
        assert_eq!(legacy["text"], "\"hello\"");
        assert_eq!(legacy["array"], "[1, 2]");

        let typed = extract_typed_meta_info(&chunk);
        assert_eq!(typed["number"], serde_json::json!(7));
        assert_eq!(typed["text"], serde_json::json!("hello"));
        assert_eq!(typed["array"], serde_json::json!([1, 2]));
    });
}
