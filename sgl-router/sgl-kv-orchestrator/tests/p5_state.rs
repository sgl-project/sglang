//! Behavioral tests at the pure API boundary, using P1's compatibility fixture.
use sgl_kv_orchestrator::{
    models::{Compatibility, SessionIdentity},
    session_index::*,
    target_resolver::*,
};
use std::sync::Arc;

fn limits() -> SessionLimits {
    SessionLimits {
        max_sessions: 16,
        max_branches: 32,
        max_history_revisions: 64,
        max_requests: 64,
        max_content_nodes: 64,
        max_token_material_bytes: 4096,
        max_tombstones: 32,
        max_claims: 32,
        max_intents: 32,
        max_identifier_bytes: 128,
        max_components: 8,
    }
}
fn compatibility() -> Compatibility {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../fixtures/actions.json")).unwrap();
    serde_json::from_value(fixture[0]["action"]["payload"]["target"]["compatibility"].clone())
        .unwrap()
}
fn key(digest: &str) -> ContentKey {
    ContentKey::new(compatibility(), digest.into()).unwrap()
}
fn session(name: &str) -> SessionIdentity {
    SessionIdentity {
        caller_id: "caller".into(),
        session_id: name.into(),
        incarnation: "task-1".into(),
    }
}
fn request(id: &str, incarnation: &str) -> RequestIdentity {
    RequestIdentity {
        request_id: id.into(),
        incarnation: incarnation.into(),
    }
}
fn register(
    index: &mut SessionIndex,
    digest: &str,
    parent: Option<&ContentKey>,
    end: u64,
    tokens: Option<Vec<u32>>,
) -> ContentKey {
    let content = key(digest);
    index
        .register_content(ContentRegistration {
            key: content.clone(),
            parent: parent.cloned(),
            end_token: end,
            suffix_tokens: tokens,
        })
        .unwrap();
    content
}
fn association(
    task: &SessionIdentity,
    branch: &str,
    session_revision: u64,
    branch_revision: u64,
    content: &ContentKey,
    request: Option<RequestIdentity>,
) -> Association {
    Association {
        session: task.clone(),
        session_revision,
        branch_id: branch.into(),
        branch_revision,
        content: content.clone(),
        parent_frontier: None,
        request,
        provenance: AssociationProvenance::RenderedRequest,
    }
}
fn bind(
    index: &mut SessionIndex,
    task: &SessionIdentity,
    branch: &str,
    session_revision: u64,
    branch_revision: u64,
    content: &ContentKey,
    request: Option<RequestIdentity>,
) -> Arc<BranchRevision> {
    index
        .associate(association(
            task,
            branch,
            session_revision,
            branch_revision,
            content,
            request,
        ))
        .unwrap()
}
#[test]
fn freezes_all_known_branches_and_preserves_historical_request_bindings() {
    let mut index = SessionIndex::new(limits());
    let task = session("chat");
    index.start_session(task.clone(), None).unwrap();
    let root = register(&mut index, "root", None, 4, Some(vec![1, 2, 3, 4]));
    let root_revision = bind(
        &mut index,
        &task,
        "main",
        1,
        1,
        &root,
        Some(request("r", "sample-1")),
    );
    let left = register(&mut index, "left", Some(&root), 8, Some(vec![5, 6, 7, 8]));
    let left_revision = bind(
        &mut index,
        &task,
        "left",
        2,
        1,
        &left,
        Some(request("r", "sample-2")),
    );
    let right = register(
        &mut index,
        "right",
        Some(&root),
        8,
        Some(vec![9, 10, 11, 12]),
    );
    let right_revision = bind(&mut index, &task, "right", 3, 1, &right, None);
    assert!(Arc::ptr_eq(
        left_revision.content().parent().unwrap(),
        right_revision.content().parent().unwrap()
    ));
    assert_eq!(index.usage().token_material_bytes, 3 * 4 * 4);
    let frozen = index.resolve(Selector::Session(task.clone()));
    assert_eq!(frozen.snapshot_revision(), Some(3));
    assert_eq!(frozen.branch_set().len(), 3);
    let expected = frozen.branch_set();
    let rerendered = register(
        &mut index,
        "rerendered",
        None,
        4,
        Some(vec![100, 101, 102, 103]),
    );
    bind(
        &mut index,
        &task,
        "main",
        4,
        2,
        &rerendered,
        Some(request("r", "sample-3")),
    );
    bind(&mut index, &task, "later", 5, 1, &root, None);
    assert_eq!(frozen.branch_set(), expected);
    assert_eq!(frozen.snapshot_revision(), Some(3));
    assert_eq!(
        index
            .resolve(Selector::Session(task.clone()))
            .known_scope()
            .len(),
        4
    );
    let historical = index.resolve(Selector::Request {
        session: task.clone(),
        request: request("r", "sample-1"),
    });
    assert!(Arc::ptr_eq(
        historical.known_scope()[0].snapshot().unwrap(),
        &root_revision
    ));
    let latest = index.resolve(Selector::Branch {
        session: task,
        branch_id: "main".into(),
        revision: None,
    });
    assert_eq!(latest.known_scope()[0].scope().key(), &rerendered);
    assert_eq!(
        index
            .materialize(
                frozen
                    .known_scope()
                    .iter()
                    .find(|s| s.scope().key() == &left)
                    .unwrap()
                    .scope()
            )
            .unwrap()
            .token_ids,
        Some(vec![1, 2, 3, 4, 5, 6, 7, 8])
    );
}

#[test]
fn cache_hit_association_needs_no_stored_and_shared_cleanup_does_not_delete_content() {
    let mut index = SessionIndex::new(limits());
    let a = session("a");
    let b = session("b");
    index.start_session(a.clone(), None).unwrap();
    index.start_session(b.clone(), None).unwrap();
    let shared = register(&mut index, "shared", None, 4, Some(vec![1, 2, 3, 4]));
    bind(&mut index, &a, "main", 1, 1, &shared, None);
    let mut hit = association(&b, "main", 1, 1, &shared, Some(request("hit", "1")));
    hit.provenance = AssociationProvenance::ConfirmedCacheHit;
    index.associate(hit).unwrap(); // No Stored or any physical observation involved.
    assert_eq!(index.usage().content_nodes, 1);
    assert_eq!(index.usage().token_material_bytes, 16);
    index.finish_task(&a, 2).unwrap();
    let target = index.resolve(Selector::Session(b.clone()));
    assert!(target.is_complete());
    assert_eq!(
        index
            .materialize(target.known_scope()[0].scope())
            .unwrap()
            .token_ids,
        Some(vec![1, 2, 3, 4])
    );
    index.finish_task(&b, 2).unwrap();
    index.collect_garbage();
    assert_eq!(index.usage().content_nodes, 1); // Frozen operation still needs this node.
    drop(target);
    index.collect_garbage();
    assert_eq!(index.usage().content_nodes, 0);
    assert_eq!(index.usage().token_material_bytes, 0);
}

#[test]
fn evicted_and_absent_material_report_exact_gaps_without_erasing_known_associations() {
    let mut index = SessionIndex::new(limits());
    let task = session("task");
    index.start_session(task.clone(), None).unwrap();
    let root = register(&mut index, "root", None, 4, Some(vec![1, 2, 3, 4]));
    bind(&mut index, &task, "root", 1, 1, &root, None);
    let leaf = register(&mut index, "leaf", Some(&root), 8, None);
    bind(&mut index, &task, "leaf", 2, 1, &leaf, None);
    let target = index.resolve(Selector::Branch {
        session: task.clone(),
        branch_id: "leaf".into(),
        revision: None,
    });
    assert_eq!(
        target.materialization_required(),
        &[MaterializationGap::Tokens {
            content: leaf.clone(),
            start_token: 4,
            end_token: 8
        }]
    );
    assert!(index.evict_material(&root));
    let gaps = index
        .materialize(target.known_scope()[0].scope())
        .unwrap_err();
    assert_eq!(gaps.len(), 2);
    assert_eq!(
        index.resolve(Selector::Session(task)).known_scope().len(),
        2
    );
    register(&mut index, "root", None, 4, Some(vec![1, 2, 3, 4]));
    register(&mut index, "leaf", Some(&root), 8, Some(vec![5, 6, 7, 8]));
    assert!(index.materialize(target.known_scope()[0].scope()).is_ok());
    let middle = index.resolve(Selector::Prefix {
        content: leaf.clone(),
        start_token: 4,
        end_token: 8,
    });
    assert_eq!(
        middle.materialization_required(),
        &[MaterializationGap::FullPrefixRequired {
            start_token: 4,
            end_token: 8
        }]
    );
    let invalid = index.resolve(Selector::Prefix {
        content: leaf,
        start_token: 0,
        end_token: 12,
    });
    assert!(invalid.known_scope().is_empty());
    assert_eq!(invalid.missing_scope().len(), 1);
    assert_eq!(
        index
            .resolve(Selector::Prefix {
                content: key("absent"),
                start_token: 0,
                end_token: 4
            })
            .missing_scope(),
        &[MissingScope::Content(key("absent"))]
    );
}

#[test]
fn task_cleanup_is_scoped_and_reincarnation_rejects_late_events() {
    let mut index = SessionIndex::new(limits());
    let parent = session("parent");
    let child = session("child");
    let sibling = session("sibling");
    index.start_session(parent.clone(), None).unwrap();
    index
        .start_session(child.clone(), Some(parent.clone()))
        .unwrap();
    index
        .start_session(sibling.clone(), Some(parent.clone()))
        .unwrap();
    let content = register(&mut index, "shared", None, 4, Some(vec![1, 2, 3, 4]));
    for task in [&parent, &child, &sibling] {
        bind(
            &mut index,
            task,
            "main",
            1,
            1,
            &content,
            Some(request("r", "1")),
        );
        index
            .track_resource(task, ResourceKind::Claim, "claim".into())
            .unwrap();
        index
            .track_resource(task, ResourceKind::Intent, "intent".into())
            .unwrap();
    }
    let cleanup = index.finish_task(&child, 2).unwrap();
    assert_eq!(cleanup.tasks, vec![child.clone()]);
    assert_eq!(cleanup.associations.len(), 1);
    assert_eq!(cleanup.requests.len(), 1);
    assert_eq!(
        cleanup.claims,
        vec![OwnedResourceId {
            task: child.clone(),
            id: "claim".into()
        }]
    );
    assert_eq!(
        cleanup.intents,
        vec![OwnedResourceId {
            task: child.clone(),
            id: "intent".into()
        }]
    );
    assert_eq!(index.usage().claims, 2);
    assert_eq!(index.usage().intents, 2);
    let mut child2 = child.clone();
    child2.incarnation = "task-2".into();
    index
        .start_session(child2.clone(), Some(parent.clone()))
        .unwrap();
    bind(&mut index, &child2, "main", 1, 1, &content, None);
    assert_eq!(
        index
            .associate(association(&child, "main", 99, 99, &content, None))
            .unwrap_err(),
        IndexError::RetiredIncarnation
    );
    assert_eq!(
        index.start_session(child.clone(), None),
        Err(IndexError::RetiredIncarnation)
    );
    assert!(index.finish_task(&child, 100).unwrap().tasks.is_empty());
    assert!(index.session_revision(&child2).is_some());
    let cleanup = index.finish_task(&parent, 2).unwrap();
    assert_eq!(cleanup.tasks, vec![parent.clone()]);
    assert!(index.session_revision(&child2).is_some());
    assert!(index.session_revision(&sibling).is_some());

    // An independent active tree demonstrates explicit cascade across descendants.
    let root = session("root");
    let c = session("c");
    let g = session("g");
    index.start_session(root.clone(), None).unwrap();
    index.start_session(c.clone(), Some(root.clone())).unwrap();
    index.start_session(g.clone(), Some(c.clone())).unwrap();
    let cleanup = index.finish_task_with_cascade(&root, 1, true).unwrap();
    assert_eq!(cleanup.tasks.len(), 3);
    assert!(
        cleanup.tasks.contains(&root) && cleanup.tasks.contains(&c) && cleanup.tasks.contains(&g)
    );
    assert!(index.session_revision(&sibling).is_some());
}

#[test]
fn revisions_and_request_incarnations_cannot_rebind_immutable_snapshots() {
    let mut index = SessionIndex::new(limits());
    let task = session("task");
    index.start_session(task.clone(), None).unwrap();
    let content = register(&mut index, "root", None, 4, Some(vec![1, 2, 3, 4]));
    let first = bind(
        &mut index,
        &task,
        "main",
        10,
        3,
        &content,
        Some(request("r", "1")),
    );
    let retry = bind(
        &mut index,
        &task,
        "main",
        10,
        3,
        &content,
        Some(request("r", "1")),
    );
    assert!(Arc::ptr_eq(&first, &retry));
    assert_eq!(
        index
            .associate(association(&task, "main", 11, 2, &content, None))
            .unwrap_err(),
        IndexError::StaleRevision
    );
    assert_eq!(
        index
            .associate(association(&task, "other", 9, 1, &content, None))
            .unwrap_err(),
        IndexError::StaleRevision
    );
    assert_eq!(
        index
            .associate(association(
                &task,
                "other",
                11,
                1,
                &content,
                Some(request("r", "1"))
            ))
            .unwrap_err(),
        IndexError::Conflict
    );
    bind(
        &mut index,
        &task,
        "other",
        11,
        1,
        &content,
        Some(request("r", "2")),
    );
    assert_eq!(
        index
            .associate(association(&task, "main", 12, 3, &content, None))
            .unwrap_err(),
        IndexError::Conflict
    );
    assert_eq!(index.usage().history_revisions, 2);
    assert_eq!(index.session_revision(&task), Some(11));
    assert!(index.forget_request(&task, &request("r", "1"), 12).unwrap());
    assert!(index.session_revision(&task).is_some());
    assert!(index.request(&task, &request("r", "2")).is_some());
}

#[test]
fn parent_frontier_uses_shared_ancestor_not_generated_suffix_concatenation() {
    let mut index = SessionIndex::new(limits());
    let task = session("task");
    index.start_session(task.clone(), None).unwrap();
    let root = register(&mut index, "root", None, 4, Some(vec![1, 2, 3, 4]));
    bind(&mut index, &task, "root", 1, 1, &root, None);
    let old = register(&mut index, "old", Some(&root), 8, Some(vec![5, 6, 7, 8]));
    let old_snapshot = bind(&mut index, &task, "old", 2, 1, &old, None);
    let rendered = register(
        &mut index,
        "rendered",
        Some(&root),
        8,
        Some(vec![9, 10, 11, 12]),
    );
    let mut a = association(&task, "new", 3, 1, &rendered, None);
    a.parent_frontier = Some(ParentFrontier {
        snapshot: old_snapshot.id().clone(),
        end_token: 4,
    });
    index.associate(a).unwrap();
    let target = index.resolve(Selector::Branch {
        session: task.clone(),
        branch_id: "new".into(),
        revision: None,
    });
    assert_eq!(
        index
            .materialize(target.known_scope()[0].scope())
            .unwrap()
            .token_ids,
        Some(vec![1, 2, 3, 4, 9, 10, 11, 12])
    );
    let mut invalid = association(&task, "bad", 4, 1, &rendered, None);
    invalid.parent_frontier = Some(ParentFrontier {
        snapshot: old_snapshot.id().clone(),
        end_token: 8,
    });
    assert_eq!(
        index.associate(invalid).unwrap_err(),
        IndexError::InvalidPrefix
    );
}

#[test]
fn compatibility_is_part_of_identity_and_different_layouts_cannot_share_parents() {
    let base = key("same-digest");
    let mut empty_salt = compatibility();
    empty_salt.cache_salt = Some(String::new());
    assert_eq!(
        base,
        ContentKey::new(empty_salt, "same-digest".into()).unwrap()
    );
    for mutate in [0, 1, 2, 3, 4, 5] {
        let mut c = compatibility();
        match mutate {
            0 => c.namespace = "other-weights".into(),
            1 => c.cache_salt = Some("tenant".into()),
            2 => c.layout_id = "different-layout".into(),
            3 => c.page_size = 8,
            4 => c.components = vec!["other".into()],
            _ => c.hash_scheme = "other-hash".into(),
        }
        assert_ne!(base, ContentKey::new(c, "same-digest".into()).unwrap());
    }
    let mut c = compatibility();
    c.components = vec!["k".into(), "v".into()];
    let a = ContentKey::new(c.clone(), "digest".into()).unwrap();
    c.components.reverse();
    assert_eq!(a, ContentKey::new(c, "digest".into()).unwrap());
    let mut index = SessionIndex::new(limits());
    let root = register(&mut index, "root", None, 4, Some(vec![1, 2, 3, 4]));
    let mut c = compatibility();
    c.cache_salt = Some("other".into());
    assert_eq!(
        index.register_content(ContentRegistration {
            key: ContentKey::new(c, "leaf".into()).unwrap(),
            parent: Some(root),
            end_token: 8,
            suffix_tokens: Some(vec![5, 6, 7, 8])
        }),
        Err(IndexError::InvalidPrefix)
    );
}

#[test]
fn bounds_refuse_without_losing_active_metadata_and_reclaim_only_unreferenced_history() {
    let mut config = limits();
    config.max_history_revisions = 1;
    config.max_content_nodes = 1;
    config.max_branches = 1;
    let mut index = SessionIndex::new(config);
    let task = session("task");
    index.start_session(task.clone(), None).unwrap();
    let content = register(&mut index, "root", None, 4, Some(vec![1, 2, 3, 4]));
    let old = bind(
        &mut index,
        &task,
        "main",
        1,
        1,
        &content,
        Some(request("r", "1")),
    );
    let frozen = index.resolve(Selector::Session(task.clone()));
    assert_eq!(
        index
            .associate(association(&task, "main", 2, 2, &content, None))
            .unwrap_err(),
        IndexError::Capacity("history_revisions")
    );
    assert_eq!(
        index
            .associate(association(&task, "other", 2, 1, &content, None))
            .unwrap_err(),
        IndexError::Capacity("branches")
    );
    assert_eq!(
        index.register_content(ContentRegistration {
            key: key("new"),
            parent: Some(content.clone()),
            end_token: 8,
            suffix_tokens: Some(vec![5, 6, 7, 8])
        }),
        Err(IndexError::Capacity("content_nodes"))
    );
    assert_eq!(index.session_revision(&task), Some(1));
    index.forget_request(&task, &request("r", "1"), 2).unwrap();
    drop(old);
    assert_eq!(
        index
            .associate(association(&task, "main", 3, 2, &content, None))
            .unwrap_err(),
        IndexError::Capacity("history_revisions")
    );
    drop(frozen);
    bind(&mut index, &task, "main", 3, 2, &content, None);
    assert_eq!(index.usage().history_revisions, 1);
    assert!(index.branch(&task, "main", Some(1)).is_none());
}

#[test]
fn content_insertion_reclaims_unused_material_at_either_limit() {
    for node_limit in [true, false] {
        let mut config = limits();
        if node_limit {
            config.max_content_nodes = 2;
        } else {
            config.max_token_material_bytes = 32;
        }
        let mut index = SessionIndex::new(config);
        let root = register(&mut index, "root", None, 4, Some(vec![1, 2, 3, 4]));
        let hold = index.content(&root).unwrap();
        let unused = register(&mut index, "unused", None, 4, Some(vec![9; 4]));
        // The parent is live only through the insertion's own reference.
        drop(hold);
        let leaf = register(&mut index, "child", Some(&root), 8, Some(vec![5, 6, 7, 8]));
        assert!(index.content(&unused).is_none());
        let prefix = index.resolve(Selector::Prefix {
            content: leaf,
            start_token: 0,
            end_token: 8,
        });
        assert_eq!(
            index
                .materialize(prefix.known_scope()[0].scope())
                .unwrap()
                .token_ids,
            Some(vec![1, 2, 3, 4, 5, 6, 7, 8])
        );
        assert_eq!(index.usage().token_material_bytes, 32);
    }
}

#[test]
fn material_request_resource_and_tombstone_limits_fail_atomically() {
    let mut config = limits();
    config.max_sessions = 2;
    config.max_token_material_bytes = 16;
    config.max_requests = 1;
    config.max_claims = 1;
    config.max_intents = 1;
    config.max_tombstones = 1;
    let mut index = SessionIndex::new(config);
    let a = session("a");
    let b = session("b");
    index.start_session(a.clone(), None).unwrap();
    index.start_session(b.clone(), None).unwrap();
    assert_eq!(
        index.start_session(session("c"), None),
        Err(IndexError::Capacity("sessions"))
    );
    let root = register(&mut index, "root", None, 4, Some(vec![1, 2, 3, 4]));
    bind(&mut index, &a, "main", 1, 1, &root, Some(request("r", "1")));
    assert_eq!(
        index.register_content(ContentRegistration {
            key: key("leaf"),
            parent: Some(root.clone()),
            end_token: 8,
            suffix_tokens: Some(vec![5, 6, 7, 8])
        }),
        Err(IndexError::Capacity("token_material_bytes"))
    );
    assert_eq!(index.usage().content_nodes, 1);
    assert_eq!(
        index
            .associate(association(
                &b,
                "main",
                1,
                1,
                &root,
                Some(request("r", "2"))
            ))
            .unwrap_err(),
        IndexError::Capacity("requests")
    );
    index
        .track_resource(&a, ResourceKind::Claim, "c".into())
        .unwrap();
    index
        .track_resource(&a, ResourceKind::Intent, "i".into())
        .unwrap();
    assert_eq!(
        index.track_resource(&b, ResourceKind::Claim, "c".into()),
        Err(IndexError::Capacity("claims"))
    );
    assert_eq!(
        index.track_resource(&b, ResourceKind::Intent, "i".into()),
        Err(IndexError::Capacity("intents"))
    );
    index.finish_task(&a, 2).unwrap();
    index
        .track_resource(&b, ResourceKind::Claim, "c".into())
        .unwrap();
    assert_eq!(
        index.finish_task(&b, 1).unwrap_err(),
        IndexError::Capacity("tombstones")
    );
    assert_eq!(index.usage().claims, 1);
    assert_eq!(index.session_revision(&b), Some(0));
    assert_eq!(
        index.start_session(a.clone(), None),
        Err(IndexError::RetiredIncarnation)
    );
    let mut other = b.clone();
    other.incarnation = "new".into();
    assert_eq!(
        index.start_session(other, None),
        Err(IndexError::ActiveIncarnation)
    );
    let mut huge = session("x");
    huge.session_id = "x".repeat(129);
    assert_eq!(
        index.start_session(huge, None),
        Err(IndexError::InvalidIdentity)
    );
}

#[test]
fn material_refill_cannot_change_an_immutable_content_node_after_eviction() {
    let mut index = SessionIndex::new(limits());
    let task = session("task");
    index.start_session(task.clone(), None).unwrap();
    let content = register(&mut index, "root", None, 4, Some(vec![1, 2, 3, 4]));
    bind(&mut index, &task, "main", 1, 1, &content, None);
    assert!(index.evict_material(&content));
    assert_eq!(
        index.register_content(ContentRegistration {
            key: content.clone(),
            parent: None,
            end_token: 4,
            suffix_tokens: Some(vec![5, 6, 7, 8])
        }),
        Err(IndexError::Conflict)
    );
    let target = index.resolve(Selector::Session(task));
    assert!(!target.is_complete());
    assert_eq!(index.usage().token_material_bytes, 0);
    register(&mut index, "root", None, 4, Some(vec![1, 2, 3, 4]));
    assert_eq!(
        index
            .materialize(target.known_scope()[0].scope())
            .unwrap()
            .token_ids,
        Some(vec![1, 2, 3, 4])
    );
}

#[test]
fn long_task_history_reclaims_superseded_revisions_but_holds_operation_dependencies() {
    let mut config = limits();
    config.max_history_revisions = 2;
    // Incoming content needs a staging slot while the current branch is active.
    config.max_content_nodes = 3;
    config.max_token_material_bytes = 48;
    let mut index = SessionIndex::new(config);
    let task = session("long");
    index.start_session(task.clone(), None).unwrap();
    let initial = register(&mut index, "initial", None, 4, Some(vec![1, 2, 3, 4]));
    bind(&mut index, &task, "main", 1, 1, &initial, None);
    let in_flight = index.resolve(Selector::Session(task.clone()));
    for revision in 2..=100 {
        let content = register(
            &mut index,
            &format!("rerender-{revision}"),
            None,
            4,
            Some(vec![revision as u32; 4]),
        );
        bind(
            &mut index, &task, "main", revision, revision, &content, None,
        );
        index.collect_garbage();
        assert_eq!(index.usage().history_revisions, 2);
        assert_eq!(index.usage().content_nodes, 2);
        assert_eq!(index.usage().token_material_bytes, 32);
    }
    assert_eq!(
        index
            .materialize(in_flight.known_scope()[0].scope())
            .unwrap()
            .token_ids,
        Some(vec![1, 2, 3, 4])
    );
    drop(in_flight);
    index.collect_garbage();
    assert_eq!(index.usage().history_revisions, 1);
    assert_eq!(index.usage().token_material_bytes, 16);
}

#[test]
fn scope_resolution_reports_empty_unknown_and_collected_history_explicitly() {
    let mut index = SessionIndex::new(limits());
    let task = session("task");
    assert_eq!(
        index
            .resolve(Selector::Session(task.clone()))
            .missing_scope(),
        &[MissingScope::Session(task.clone())]
    );
    index.start_session(task.clone(), None).unwrap();
    let empty = index.resolve(Selector::Session(task.clone()));
    assert_eq!(empty.snapshot_revision(), Some(0));
    assert_eq!(
        empty.missing_scope(),
        &[MissingScope::EmptySession(task.clone())]
    );
    let request_id = request("missing", "1");
    assert_eq!(
        index
            .resolve(Selector::Request {
                session: task.clone(),
                request: request_id.clone()
            })
            .missing_scope(),
        &[MissingScope::Request {
            session: task.clone(),
            request: request_id
        }]
    );
    let content = register(&mut index, "root", None, 4, None);
    bind(&mut index, &task, "main", 1, 1, &content, None);
    bind(&mut index, &task, "main", 2, 2, &content, None);
    index.collect_garbage();
    assert_eq!(
        index
            .resolve(Selector::Branch {
                session: task.clone(),
                branch_id: "main".into(),
                revision: Some(1)
            })
            .missing_scope(),
        &[MissingScope::Branch {
            session: task,
            branch_id: "main".into(),
            revision: Some(1)
        }]
    );
    let large = register(&mut index, "too-large-for-p1", None, 65_540, None);
    let target = index.resolve(Selector::Prefix {
        content: large,
        start_token: 0,
        end_token: 65_540,
    });
    assert!(target.materialization_required().iter().any(|gap| matches!(
        gap,
        MaterializationGap::ProtocolTokenLimit {
            end_token: 65_540,
            max_tokens: 65_536
        }
    )));
    assert!(index.materialize(target.known_scope()[0].scope()).is_err());
}
