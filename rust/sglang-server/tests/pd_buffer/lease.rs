use super::*;

#[test]
fn four_lease_classes_and_three_native_stages_transition_at_most_once() {
    let profile = PdProfileV1::load_embedded().expect("profile");
    let source_tracker = TableUseTracker::new();
    let destination_tracker = TableUseTracker::new();
    let ledger = CapacityLedger::new(
        &profile,
        source_tracker.clone(),
        destination_tracker.clone(),
    );
    let handle = ledger.reserve(reservation(1)).expect("reserve leases");

    assert!(ledger.begin_stage(handle, TransferStage::Aux).is_err());
    for stage in [
        TransferStage::Kv,
        TransferStage::Aux,
        TransferStage::Completion,
    ] {
        ledger.begin_stage(handle, stage).expect("begin stage");
        assert!(ledger.begin_stage(handle, stage).is_err());
        ledger.finish_stage(handle, stage).expect("finish stage");
    }

    assert_eq!(
        ledger.release_source_safe(handle).expect("source release"),
        TransitionResult::Applied
    );
    assert_eq!(
        ledger
            .release_source_safe(handle)
            .expect("duplicate source"),
        TransitionResult::AlreadyApplied
    );
    assert_eq!(
        ledger
            .handoff_destination(handle)
            .expect("destination handoff"),
        TransitionResult::Applied
    );
    assert_eq!(
        ledger
            .handoff_destination(handle)
            .expect("duplicate handoff"),
        TransitionResult::AlreadyApplied
    );
    assert_eq!(
        ledger.release_terminal(handle).expect("terminal release"),
        TransitionResult::Applied
    );
    assert_eq!(
        ledger.release_terminal(handle).expect("duplicate terminal"),
        TransitionResult::AlreadyApplied
    );
    let snapshot = ledger.snapshot();
    assert_eq!(snapshot.active_rooms, 0);
    assert_eq!(snapshot.release_actions, 3);
    assert_eq!(snapshot.handoff_actions, 1);
}

#[test]
fn post_submit_abort_quarantines_resources_until_explicit_native_safety() {
    let profile = PdProfileV1::load_embedded().expect("profile");
    let source_tracker = TableUseTracker::new();
    let destination_tracker = TableUseTracker::new();
    let ledger = CapacityLedger::new(
        &profile,
        source_tracker.clone(),
        destination_tracker.clone(),
    );
    let handle = ledger.reserve(reservation(2)).expect("reserve leases");
    ledger
        .begin_stage(handle, TransferStage::Kv)
        .expect("submit KV");
    assert_eq!(
        ledger.quarantine(handle).expect("quarantine"),
        TransitionResult::Applied
    );
    assert_eq!(
        ledger.quarantine(handle).expect("duplicate quarantine"),
        TransitionResult::AlreadyApplied
    );
    assert_eq!(ledger.snapshot().quarantined_rooms, 1);
    assert_eq!(source_tracker.snapshot().quarantined, 1);
    assert!(ledger.reserve(reservation(2)).is_err());

    assert_eq!(
        ledger
            .resolve_quarantine(handle)
            .expect("native safety release"),
        TransitionResult::Applied
    );
    assert_eq!(
        ledger.resolve_quarantine(handle).expect("duplicate safety"),
        TransitionResult::AlreadyApplied
    );
    assert_eq!(ledger.snapshot().active_rooms, 0);
    assert_eq!(source_tracker.snapshot().quarantined, 0);
    assert_eq!(destination_tracker.snapshot().quarantined, 0);
}
