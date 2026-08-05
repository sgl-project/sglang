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
                preserve_on_explicit_abort: false,
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
fn explicit_abort_keeps_choice_aware_channel_for_terminal_errors() {
    let (sender, _receiver) = tokio::sync::mpsc::channel(1);
    let key = RequestKey {
        rid: "multi-choice".to_string(),
        incarnation: 1,
    };
    let mut state = BridgeState::default();
    state.channels.insert(
        key.rid.clone(),
        ActiveChannel {
            incarnation: key.incarnation,
            sender,
            preserve_on_explicit_abort: true,
        },
    );

    finalize_explicit_abort_locked(&mut state, &key);

    assert!(state.channels.contains_key(key.rid()));
    assert!(!state.terminal_errors.contains_key(&key));
}
