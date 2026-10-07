use std::sync::Mutex;

use super::*;
use crate::components::{FULL, MAMBA, SWA};
use crate::unified_tree_core::UnifiedTreeCore;

fn full_factory(_: &TreeComponentArgument<'_>) -> Result<FullComponent, ComponentInitError> {
    Ok(FullComponent)
}

#[test]
fn registration_requires_explicit_replacement_and_preserves_kind() {
    let registry = TreeComponentRegistry::default();
    for name in ["", " \t"] {
        assert!(matches!(
            registry.register_tree_component(name, FULL, full_factory, false),
            Err(ComponentInitError::EmptyFactoryKey)
        ));
    }
    assert!(matches!(
        registry.register_tree_component("full_default", FULL, full_factory, false),
        Err(ComponentInitError::DuplicateFactoryKey(_))
    ));
    assert!(matches!(
        registry.register_tree_component("full_default", SWA, full_factory, true),
        Err(ComponentInitError::ComponentTypeMismatch {
            expected: FULL,
            actual: SWA
        })
    ));
    registry
        .register_tree_component("full_default", FULL, full_factory, true)
        .unwrap();
    registry
        .register_tree_component("custom_full", FULL, full_factory, false)
        .unwrap();
    let mut metadata = registry.registered_tree_components();
    metadata.clear();
    assert_eq!(
        registry.registered_tree_components(),
        HashMap::from([
            ("full_default".to_owned(), FULL),
            ("swa_default".to_owned(), SWA),
            ("mamba_default".to_owned(), MAMBA),
            ("custom_full".to_owned(), FULL),
        ])
    );
    assert_eq!(
        TreeComponentRegistry::default()
            .registered_tree_components()
            .len(),
        3
    );
}

fn exercise_snapshot_replacement<K: TreeComponentKey>() {
    let registry = Arc::new(TreeComponentRegistry::default());
    let events = Arc::new(Mutex::new(Vec::new()));
    let old_events = Arc::clone(&events);
    registry
        .register_tree_component(
            "swa_default",
            SWA,
            move |argument: &TreeComponentArgument<'_>| {
                old_events.lock().unwrap().push("old");
                Ok(SwaComponent::new(argument.params))
            },
            true,
        )
        .unwrap();
    let weak_registry = Arc::downgrade(&registry);
    let factory_events = Arc::clone(&events);
    registry
        .register_tree_component(
            "full_default",
            FULL,
            move |argument: &TreeComponentArgument<'_>| {
                assert_eq!(argument.is_bigram, K::IS_BIGRAM);
                assert_eq!(argument.component_type, FULL);
                let registry = weak_registry.upgrade().unwrap();
                drop(
                    registry
                        .factories
                        .try_write()
                        .expect("factory runs without the registry lock"),
                );
                factory_events.lock().unwrap().push("full");
                let new_events = Arc::clone(&factory_events);
                registry
                    .register_tree_component(
                        "swa_default",
                        SWA,
                        move |argument: &TreeComponentArgument<'_>| {
                            new_events.lock().unwrap().push("new");
                            Ok(SwaComponent::new(argument.params))
                        },
                        true,
                    )
                    .unwrap();
                Ok(FullComponent)
            },
            true,
        )
        .unwrap();

    let params = CacheInitParams {
        swa_sliding_window_size: Some(4),
        ..Default::default()
    };
    let keys = ["full_default".to_owned(), "swa_default".to_owned()];
    let first = registry
        .snapshot(&keys)
        .unwrap()
        .create_components::<K>(&params)
        .unwrap();
    assert_eq!(*events.lock().unwrap(), ["full", "old"]);
    let second = registry
        .snapshot(&keys)
        .unwrap()
        .create_components::<K>(&params)
        .unwrap();
    assert_eq!(*events.lock().unwrap(), ["full", "old", "full", "new"]);
    assert!(!Arc::ptr_eq(&first[1], &second[1]));
}

#[test]
fn snapshots_are_atomic_and_factories_can_register_replacements() {
    exercise_snapshot_replacement::<Vec<i64>>();
    exercise_snapshot_replacement::<Vec<(i64, i64)>>();
}

#[test]
fn factory_selection_rejects_unknown_duplicate_and_invalid_layouts() {
    let registry = TreeComponentRegistry::default();
    registry
        .register_tree_component("custom_full", FULL, full_factory, false)
        .unwrap();
    for keys in [
        vec![""],
        vec!["missing"],
        vec!["full_default", "full_default"],
        vec!["full_default", "custom_full"],
    ] {
        let keys = keys.into_iter().map(str::to_owned).collect::<Vec<_>>();
        assert!(registry.snapshot(&keys).is_err());
    }
    for keys in [
        vec![],
        vec!["swa_default"],
        vec!["swa_default", "full_default"],
        vec!["full_default", "mamba_default", "swa_default"],
    ] {
        let keys = keys.into_iter().map(str::to_owned).collect::<Vec<_>>();
        let snapshot = registry.snapshot(&keys).unwrap();
        let result = UnifiedTreeCore::<Vec<i64>>::with_component_factory_snapshot(
            CacheInitParams::default(),
            snapshot,
        );
        assert!(
            result
                .err()
                .unwrap()
                .to_string()
                .contains("component sets are supported")
        );
    }
}

#[test]
fn factories_validate_requested_and_produced_kinds_and_key_mode() {
    let registry = TreeComponentRegistry::default();
    registry
        .register_tree_component("wrong_result", SWA, full_factory, false)
        .unwrap();
    let params = CacheInitParams::default();
    let mut argument = TreeComponentArgument {
        component_type: SWA,
        params: &params,
        is_bigram: false,
    };
    assert!(matches!(
        registry.create_tree_component::<Vec<i64>>("full_default", &argument),
        Err(ComponentInitError::ComponentTypeMismatch {
            expected: SWA,
            actual: FULL
        })
    ));
    assert!(matches!(
        registry.create_tree_component::<Vec<i64>>("wrong_result", &argument),
        Err(ComponentInitError::ComponentTypeMismatch {
            expected: SWA,
            actual: FULL
        })
    ));
    argument.component_type = FULL;
    argument.is_bigram = true;
    assert!(matches!(
        registry.create_tree_component::<Vec<i64>>("full_default", &argument),
        Err(ComponentInitError::KeyModeMismatch)
    ));
}

#[test]
fn invalid_configuration_and_factory_errors_are_returned() {
    let registry = TreeComponentRegistry::default();
    registry
        .register_tree_component(
            "fails",
            FULL,
            |_: &TreeComponentArgument<'_>| {
                Err::<FullComponent, _>(ComponentInitError::InvalidConfiguration(
                    "factory rejected configuration",
                ))
            },
            false,
        )
        .unwrap();
    let params = CacheInitParams::default();
    for (key, component_type) in [
        ("fails", FULL),
        ("swa_default", SWA),
        ("mamba_default", MAMBA),
    ] {
        let argument = TreeComponentArgument {
            component_type,
            params: &params,
            is_bigram: false,
        };
        assert!(matches!(
            registry.create_tree_component::<Vec<i64>>(key, &argument),
            Err(ComponentInitError::InvalidConfiguration(_))
        ));
    }
    let params = CacheInitParams {
        page_size: 0,
        ..Default::default()
    };
    let argument = TreeComponentArgument {
        component_type: FULL,
        params: &params,
        is_bigram: false,
    };
    assert!(matches!(
        registry.create_tree_component::<Vec<i64>>("full_default", &argument),
        Err(ComponentInitError::InvalidConfiguration(
            "page_size must be at least 1"
        ))
    ));
}

#[test]
fn ordered_factories_reject_zero_component_configuration() {
    for (key, params) in [
        (
            "swa_default",
            CacheInitParams {
                swa_sliding_window_size: Some(0),
                ..Default::default()
            },
        ),
        (
            "mamba_default",
            CacheInitParams {
                mamba_cache_chunk_size: Some(0),
                ..Default::default()
            },
        ),
    ] {
        let result = UnifiedTreeCore::<Vec<i64>>::with_component_factories(
            params,
            vec!["full_default".to_owned(), key.to_owned()],
        );
        assert!(matches!(
            result,
            Err(ComponentInitError::InvalidConfiguration(_))
        ));
    }
}

#[test]
#[should_panic(expected = "swa_sliding_window_size must be positive")]
fn ordinary_plain_constructor_preserves_the_component_assertion() {
    UnifiedTreeCore::<Vec<i64>>::new(
        CacheInitParams {
            swa_sliding_window_size: Some(0),
            ..Default::default()
        },
        vec![FULL, SWA],
    );
}

#[test]
#[should_panic(expected = "swa_sliding_window_size must be positive")]
fn ordinary_bigram_constructor_preserves_the_component_assertion() {
    UnifiedTreeCore::<Vec<(i64, i64)>>::new(
        CacheInitParams {
            swa_sliding_window_size: Some(0),
            ..Default::default()
        },
        vec![FULL, SWA],
    );
}
