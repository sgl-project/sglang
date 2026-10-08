use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, Weak};

use super::*;
use crate::components::{FULL, SWA};
use crate::unified_tree_core::UnifiedTreeCore;

fn full_factory(_: &TreeComponentArgument<'_>) -> FullComponent {
    FullComponent
}

struct KeyedFactoryForTest {
    calls: Arc<Mutex<Vec<(ComponentType, bool)>>>,
}

impl<K: ChildKeyType> TreeComponentFactory<K> for KeyedFactoryForTest {
    fn create(&self, argument: &TreeComponentArgument<'_>) -> TreeComponentInstance<K> {
        assert_eq!(argument.is_bigram, K::IS_BIGRAM);
        self.calls
            .lock()
            .unwrap()
            .push((argument.component_type, K::IS_BIGRAM));
        match argument.component_type {
            FULL => Arc::new(FullComponent),
            SWA => Arc::new(SwaComponent::new(argument.params)),
            _ => unreachable!(),
        }
    }
}

#[test]
fn one_factory_key_supports_multiple_component_and_key_types() {
    let registry = TreeComponentRegistry::default();
    let calls = Arc::new(Mutex::new(Vec::new()));
    registry
        .register_tree_component(
            "shared_factory",
            KeyedFactoryForTest {
                calls: Arc::clone(&calls),
            },
            false,
        )
        .unwrap();
    let selection = registry
        .resolve(
            &[FULL, SWA],
            &HashMap::from([
                (FULL, "shared_factory".to_owned()),
                (SWA, "shared_factory".to_owned()),
            ]),
        )
        .unwrap();
    let params = CacheInitParams {
        swa_sliding_window_size: Some(4),
        ..Default::default()
    };
    selection.create::<Vec<i64>>(FULL, &params);
    selection.create::<Vec<i64>>(SWA, &params);
    selection.create::<Vec<(i64, i64)>>(FULL, &params);
    selection.create::<Vec<(i64, i64)>>(SWA, &params);
    assert_eq!(
        *calls.lock().unwrap(),
        [(FULL, false), (SWA, false), (FULL, true), (SWA, true)]
    );
}

#[test]
fn registration_requires_explicit_replacement() {
    let registry = TreeComponentRegistry::default();
    for key in ["", " \t"] {
        assert!(matches!(
            registry.register_tree_component(key, full_factory, false),
            Err(TreeComponentRegistryError::EmptyKey)
        ));
    }
    assert!(matches!(
        registry.register_tree_component("full_default", full_factory, false),
        Err(TreeComponentRegistryError::DuplicateKey(_))
    ));
    registry
        .register_tree_component("full_default", full_factory, true)
        .unwrap();
    registry
        .register_tree_component("custom_full", full_factory, false)
        .unwrap();
    assert!(
        registry
            .resolve(&[FULL], &HashMap::from([(FULL, "custom_full".to_owned())]))
            .is_ok()
    );
}

#[test]
fn selection_validates_slots_and_keys_before_invoking_factories() {
    let registry = TreeComponentRegistry::default();
    for (component_types, overrides, expected) in [
        (
            vec![FULL, FULL],
            HashMap::new(),
            "duplicate component type Full",
        ),
        (
            vec![FULL],
            HashMap::from([(SWA, "swa_default".to_owned())]),
            "component Swa is not enabled",
        ),
        (
            vec![FULL],
            HashMap::from([(FULL, "".to_owned())]),
            "component factory key must be non-empty",
        ),
        (
            vec![FULL],
            HashMap::from([(FULL, "missing".to_owned())]),
            "unknown component factory",
        ),
    ] {
        assert!(
            registry
                .resolve(&component_types, &overrides)
                .err()
                .unwrap()
                .to_string()
                .contains(expected)
        );
    }
}

fn exercise_selection<K: TreeComponentKey>() {
    let registry = Arc::new(TreeComponentRegistry::default());
    let events = Arc::new(Mutex::new(Vec::new()));
    let old_events = Arc::clone(&events);
    registry
        .register_tree_component(
            "swa_default",
            move |argument: &TreeComponentArgument<'_>| {
                old_events.lock().unwrap().push("old");
                SwaComponent::new(argument.params)
            },
            true,
        )
        .unwrap();
    let weak_registry = Arc::downgrade(&registry);
    let factory_events = Arc::clone(&events);
    registry
        .register_tree_component(
            "full_default",
            move |argument: &TreeComponentArgument<'_>| {
                assert_eq!(argument.component_type, FULL);
                assert_eq!(argument.is_bigram, K::IS_BIGRAM);
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
                        move |argument: &TreeComponentArgument<'_>| {
                            new_events.lock().unwrap().push("new");
                            SwaComponent::new(argument.params)
                        },
                        true,
                    )
                    .unwrap();
                FullComponent
            },
            true,
        )
        .unwrap();

    let params = CacheInitParams {
        swa_sliding_window_size: Some(4),
        ..Default::default()
    };
    let selection = registry.resolve(&[FULL, SWA], &HashMap::new()).unwrap();
    selection.create::<K>(FULL, &params);
    selection.create::<K>(SWA, &params);
    assert_eq!(*events.lock().unwrap(), ["full", "old"]);
    let next = registry.resolve(&[FULL, SWA], &HashMap::new()).unwrap();
    next.create::<K>(FULL, &params);
    next.create::<K>(SWA, &params);
    assert_eq!(*events.lock().unwrap(), ["full", "old", "full", "new"]);
}

#[test]
fn resolved_factories_are_consistent_across_registration_changes() {
    exercise_selection::<Vec<i64>>();
    exercise_selection::<Vec<(i64, i64)>>();
}

struct FactoryDropForTest {
    registry: Weak<TreeComponentRegistry>,
    drops: Arc<AtomicUsize>,
}

impl Drop for FactoryDropForTest {
    fn drop(&mut self) {
        let registry = self.registry.upgrade().unwrap();
        drop(
            registry
                .factories
                .try_write()
                .expect("factory drops without the registry lock"),
        );
        registry
            .register_tree_component("from_drop", full_factory, false)
            .unwrap();
        self.drops.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn replacement_drops_old_factories_outside_the_write_lock() {
    let registry = Arc::new(TreeComponentRegistry::default());
    let drops = Arc::new(AtomicUsize::new(0));
    let probe = FactoryDropForTest {
        registry: Arc::downgrade(&registry),
        drops: Arc::clone(&drops),
    };
    registry
        .register_tree_component(
            "replaceable",
            move |_: &TreeComponentArgument<'_>| {
                let _ = &probe;
                FullComponent
            },
            false,
        )
        .unwrap();
    registry
        .register_tree_component("replaceable", full_factory, true)
        .unwrap();
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[test]
fn ordinary_native_construction_resolves_registered_defaults() {
    let thread = std::thread::current().id();
    let calls = Arc::new(AtomicUsize::new(0));
    let observed = Arc::clone(&calls);
    register_tree_component(
        "full_default",
        move |_: &TreeComponentArgument<'_>| {
            if std::thread::current().id() == thread {
                calls.fetch_add(1, Ordering::SeqCst);
            }
            FullComponent
        },
        true,
    )
    .unwrap();
    UnifiedTreeCore::<Vec<i64>>::new(CacheInitParams::default(), vec![FULL]);
    UnifiedTreeCore::<Vec<(i64, i64)>>::new(CacheInitParams::default(), vec![FULL]);
    register_tree_component("full_default", full_factory, true).unwrap();
    assert_eq!(observed.load(Ordering::SeqCst), 2);
}

#[test]
#[should_panic(expected = "component factory returned the wrong kind for Swa")]
fn selected_factory_rejects_a_different_produced_kind() {
    let registry = TreeComponentRegistry::default();
    registry
        .register_tree_component("wrong_result", full_factory, false)
        .unwrap();
    let selection = registry
        .resolve(&[SWA], &HashMap::from([(SWA, "wrong_result".to_owned())]))
        .unwrap();
    selection.create::<Vec<i64>>(SWA, &CacheInitParams::default());
}
