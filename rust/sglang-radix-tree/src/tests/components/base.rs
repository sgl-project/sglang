use super::*;
use crate::components::{FULL, FullComponent, MAMBA, MambaComponent, SWA, SwaComponent};
use crate::unified_tree_core::CacheInitParams;
use std::sync::{Arc, Mutex};

// Exercise trait defaults and trace delegated FULL evictions.
struct DefaultComponentForTest {
    component_type: ComponentType,
    count_scale: usize,
    removed: Mutex<Vec<NodeId>>,
}

impl Default for DefaultComponentForTest {
    fn default() -> Self {
        Self {
            component_type: FULL,
            count_scale: 1,
            removed: Mutex::default(),
        }
    }
}

impl TreeComponent<Vec<i64>> for DefaultComponentForTest {
    fn component_type(&self) -> ComponentType {
        self.component_type
    }

    fn reclaimable_value_len(&self, node: &Node<Vec<i64>>) -> usize {
        node.device_reclaimable_value_len(self.component_type) * self.count_scale
    }

    fn on_node_removed(&self, node_id: NodeId) {
        self.removed.lock().unwrap().push(node_id);
    }

    fn create_match_validator(
        &self,
        _tree_core: &UnifiedTreeCore<Vec<i64>>,
        match_device_only: bool,
    ) -> Box<dyn FnMut(&UnifiedTreeCore<Vec<i64>>, NodeIdx_) -> bool> {
        unimplemented!()
    }

    fn redistribute_on_node_split(
        &self,
        tree_core: &mut UnifiedTreeCore<Vec<i64>>,
        new_parent_id: NodeIdx_,
        child_id: NodeIdx_,
    ) {
        unimplemented!()
    }

    fn evict_component(
        &self,
        tree_core: &mut UnifiedTreeCore<Vec<i64>>,
        node_id: NodeIdx_,
        device_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
        host_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
        cache_actions: &mut Vec<CacheAction>,
        target: EvictLayer,
    ) -> (usize, usize) {
        cache_actions.push(CacheAction::MambaEvictExcessPathStates {
            tail_node_id: tree_core.arena.node(node_id).id,
        });
        FullComponent.evict_component(
            tree_core,
            node_id,
            device_frees,
            host_frees,
            cache_actions,
            target,
        )
    }

    fn evict_device_start(&self, tree_core: &mut UnifiedTreeCore<Vec<i64>>, request_cnt: usize) {
        unimplemented!()
    }

    fn evict_device_next_node(
        &self,
        tree_core: &mut UnifiedTreeCore<Vec<i64>>,
        tracker: &mut HashMap<ComponentType, usize>,
        device_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
        host_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
        cache_actions: &mut Vec<CacheAction>,
    ) -> Option<NodeIdx_> {
        unimplemented!()
    }

    fn evict_device_end(&self, tree_core: &mut UnifiedTreeCore<Vec<i64>>) {
        unimplemented!()
    }

    fn acquire_component_lock(
        &self,
        tree_core: &mut UnifiedTreeCore<Vec<i64>>,
        node_id: NodeIdx_,
        result: IncLockRefResult,
        lock_host: bool,
    ) -> IncLockRefResult {
        unimplemented!()
    }

    fn release_component_lock(
        &self,
        tree_core: &mut UnifiedTreeCore<Vec<i64>>,
        node_id: NodeIdx_,
        params: &DecLockRefParams,
        lock_host: bool,
    ) {
        unimplemented!()
    }
}

#[test]
fn insert_overlap_default_consumes_nothing() {
    let mut tc: UnifiedTreeCore<Vec<i64>> =
        UnifiedTreeCore::new(CacheInitParams::default(), vec![FULL]);
    let root = tc.arena.root();
    let consumed_from = DefaultComponentForTest::default().update_component_on_insert_overlap(
        &mut tc,
        root,
        /* prefix_len = */ 3,
        /* total_prefix_len = */ 0,
        Tensor::from_slice(&[0i64, 1, 2]),
        &InsertParams {
            rotation_base: None,
            key: &vec![0, 1, 2],
            namespace: Default::default(),
            value: Tensor::from_slice(&[0i64, 1, 2]),
            mamba_value: None,
            prev_prefix_len: 0,
            swa_evicted_seqlen: 0,
            swa_branching_seqlen: None,
            inserted_len: 0,
            priority: 0,
            session_id: None,
            track_adopted_ranges: false,
        },
        false,
        &mut InsertResult::default(),
        &mut Vec::new(),
    );
    // Nothing consumed: the whole overlap stays freeable as duplicates.
    assert_eq!(consumed_from, 3);
}

#[test]
fn finalize_match_result_default_returns_result_unchanged() {
    let tc: UnifiedTreeCore<Vec<i64>> =
        UnifiedTreeCore::new(CacheInitParams::default(), vec![FULL]);
    let result = MatchResult {
        last_device_node_id: 3,
        best_match_node_id: 7,
        host_hit_length: 11,
        ..tc.empty_match_result()
    };
    let out = DefaultComponentForTest::default().finalize_match_result_in_tree_core(
        &tc,
        result,
        tc.arena.root(),
        tc.arena.root(),
        &MatchPrefixParams {
            key: &Vec::new(),
            namespace: Default::default(),
        },
        &[],
        0,
    );
    assert_eq!(out.last_device_node_id, 3);
    assert_eq!(out.best_match_node_id, 7);
    assert_eq!(out.host_hit_length, 11);
}

#[test]
fn drive_host_eviction_default_is_a_noop() {
    let mut tc: UnifiedTreeCore<Vec<i64>> =
        UnifiedTreeCore::new(CacheInitParams::default(), vec![FULL]);
    let mut tracker = HashMap::from([(FULL, 5usize)]);
    let mut device_frees = HashMap::new();
    let mut host_frees = HashMap::new();
    DefaultComponentForTest::default().drive_host_eviction(
        &mut tc,
        /* num_tokens = */ 100,
        &mut tracker,
        &mut device_frees,
        &mut host_frees,
        &mut Vec::new(),
    );
    assert_eq!(tracker[&FULL], 5);
    assert!(device_frees.is_empty());
    assert!(host_frees.is_empty());
}

// Component types.

#[test]
fn idx_matches_discriminants() {
    assert_eq!(ComponentType::Full.idx(), 0);
    assert_eq!(ComponentType::Swa.idx(), 1);
    assert_eq!(ComponentType::Mamba.idx(), 2);
}

fn action_core() -> UnifiedTreeCore<Vec<i64>> {
    UnifiedTreeCore::new_with_components_(
        CacheInitParams {
            enable_hicache: true,
            ..Default::default()
        },
        vec![Arc::new(DefaultComponentForTest::default())],
    )
}

fn action_node(
    core: &mut UnifiedTreeCore<Vec<i64>>,
    parent: NodeIdx_,
    key: i64,
    device: bool,
    host: bool,
) -> NodeIdx_ {
    let node = core.arena.alloc_child(parent, vec![key], 0, None).unwrap();
    if device {
        core.arena
            .set_device_value(node, FULL, Tensor::from_slice(&[key]));
        core.inc_evictable_size(FULL, 1);
    }
    if host {
        core.arena
            .set_host_value(node, FULL, Tensor::from_slice(&[key + 100]));
    }
    core.update_evictable_leaf_sets_(node);
    core.update_evictable_leaf_sets_(parent);
    core.update_full_coexisting_host_tracking_(node);
    node
}

fn action_node_ids(actions: &[CacheAction]) -> Vec<usize> {
    actions
        .iter()
        .map(|action| match action {
            CacheAction::MambaEvictExcessPathStates { tail_node_id } => *tail_node_id,
            _ => panic!("unexpected component action"),
        })
        .collect()
}

#[test]
fn leaf_eviction_results_preserve_component_actions() {
    for demote in [false, true] {
        let mut core = action_core();
        let root = core.arena.root();
        let node = action_node(&mut core, root, 1, true, demote);
        let node_id = core.arena.node(node).id;
        let result = if demote {
            core.demote(node_id).unwrap()
        } else {
            core.evict_device_leaf(node_id, false).unwrap().1
        };
        assert_eq!(action_node_ids(&result.cache_actions), [node_id]);
        assert_eq!(result.device_frees[&FULL].len(), 1);
        assert_eq!(result.tracker[&FULL], 1);
    }
}

#[test]
fn subtree_cleanup_returns_child_actions_before_parent_actions() {
    let mut core = action_core();
    let root = core.arena.root();
    let parent = action_node(&mut core, root, 1, true, false);
    let child = action_node(&mut core, parent, 2, false, true);
    let parent_id = core.arena.node(parent).id;
    let child_id = core.arena.node(child).id;
    let (dropped, result) = core.drop_subtree_no_host(parent_id).unwrap();
    assert!(dropped);
    assert_eq!(
        action_node_ids(&result.cache_actions),
        [child_id, parent_id]
    );
    assert_eq!(result.device_frees[&FULL].len(), 1);
    assert_eq!(result.host_frees[&FULL].len(), 1);
}

#[test]
fn host_drivers_forward_component_actions_in_order() {
    for coexisting in [false, true] {
        let mut core = action_core();
        let root = core.arena.root();
        let nodes = [
            action_node(&mut core, root, 1, coexisting, true),
            action_node(&mut core, root, 2, coexisting, true),
        ];
        let ids = nodes.map(|node| core.arena.node(node).id);
        let mut tracker = HashMap::from([(FULL, 0)]);
        let mut device_frees = HashMap::new();
        let mut host_frees = HashMap::new();
        let mut actions = Vec::new();
        if coexisting {
            FullComponent.reclaim_coexisting_host_values(
                &mut core,
                2,
                &mut tracker,
                &mut device_frees,
                &mut host_frees,
                &mut actions,
            );
        } else {
            FullComponent.drive_host_eviction(
                &mut core,
                2,
                &mut tracker,
                &mut device_frees,
                &mut host_frees,
                &mut actions,
            );
        }
        assert_eq!(action_node_ids(&actions), ids);
        assert_eq!(tracker[&FULL], 2);
        assert!(device_frees.is_empty());
        assert_eq!(host_frees[&FULL].len(), 2);
    }
}

#[test]
fn reused_drivers_count_the_configured_components_backing() {
    for component_type in [FULL, SWA, MAMBA] {
        let params = CacheInitParams {
            swa_sliding_window_size: Some(4),
            mamba_cache_chunk_size: Some(1),
            ..Default::default()
        };
        let counted = Arc::new(DefaultComponentForTest {
            component_type,
            count_scale: 2,
            ..Default::default()
        });
        let mut components: Vec<Arc<dyn TreeComponent<Vec<i64>> + Send + Sync>> = Vec::new();
        if component_type != FULL {
            components.push(Arc::new(FullComponent));
        }
        components.push(counted.clone());
        let driver: Box<dyn TreeComponent<Vec<i64>>> = if component_type == FULL {
            Box::new(FullComponent)
        } else if component_type == SWA {
            Box::new(SwaComponent::new(&params))
        } else {
            Box::new(MambaComponent::new(&params))
        };
        let mut core = UnifiedTreeCore::new_with_components_(params, components);
        let root = core.arena.root();
        let node = core.add_new_node_(
            root,
            vec![1, 2, 3, 4],
            &Tensor::from_slice(&[10i64, 11, 12, 13]),
            0,
            None,
        );
        let node_id = core.arena.node(node).id;
        let logical_len = if component_type == MAMBA { 1 } else { 4 };
        if component_type != FULL {
            assert_eq!(counted.reclaimable_value_len(core.arena.node(node)), 0);
            core.set_component_device_value(
                node_id,
                component_type,
                Tensor::from_slice(&vec![20i64; logical_len]),
            )
            .unwrap();
        }
        let counted_len = logical_len * 2;
        assert_eq!(
            core.arena
                .node(node)
                .device_reclaimable_value_len(component_type),
            logical_len
        );
        assert_eq!(counted.reclaimable_value_len(core.arena.node(root)), 0);
        assert_eq!(core.component_evictable_size(component_type), counted_len);
        core.try_sanity_check(&[], &[]).unwrap();

        // These ordinary drivers must consult their configured wrapper's count.
        let receipt =
            driver.acquire_component_lock(&mut core, node, IncLockRefResult::default(), false);
        driver.acquire_component_lock(&mut core, node, IncLockRefResult::default(), false);
        assert_eq!(core.component_evictable_size(component_type), 0);
        assert_eq!(core.component_protected_size(component_type), counted_len);
        driver.release_component_lock(&mut core, node, &receipt.to_dec_params(), false);
        assert_eq!(core.component_protected_size(component_type), counted_len);
        driver.release_component_lock(&mut core, node, &receipt.to_dec_params(), false);
        assert_eq!(core.component_evictable_size(component_type), counted_len);
        assert_eq!(core.component_protected_size(component_type), 0);
        core.try_sanity_check(&[], &[]).unwrap();
        let mut device_frees = HashMap::new();
        let (freed, _) = driver.evict_component(
            &mut core,
            node,
            &mut device_frees,
            &mut HashMap::new(),
            &mut Vec::new(),
            EvictLayer::Device,
        );
        assert_eq!(freed, counted_len);
        assert_eq!(core.component_evictable_size(component_type), 0);
        assert_eq!(device_frees[&component_type].len(), 1);
    }
}

#[test]
fn removal_notifies_components_only_after_success() {
    let observer = Arc::new(DefaultComponentForTest::default());
    let mut core =
        UnifiedTreeCore::new_with_components_(CacheInitParams::default(), vec![observer.clone()]);
    let root = core.arena.root();
    let parent = core.arena.alloc_child(root, vec![1], 0, None).unwrap();
    let child = core.arena.alloc_child(parent, vec![2], 0, None).unwrap();
    let parent_id = core.arena.node(parent).id;
    let child_id = core.arena.node(child).id;
    let rejected = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        core.remove_leaf_from_parent_(parent)
    }));
    assert!(rejected.is_err());
    assert!(observer.removed.lock().unwrap().is_empty());
    assert!(core.arena.resolve(parent_id).is_ok());
    core.remove_leaf_from_parent_(child);
    core.remove_leaf_from_parent_(parent);
    assert_eq!(*observer.removed.lock().unwrap(), [child_id, parent_id]);
    assert!(core.arena.resolve(child_id).is_err());
    assert!(core.arena.resolve(parent_id).is_err());
}
