use super::*;
use crate::components::FULL;
use crate::unified_tree_core::CacheInitParams;

// Test-only component exercising the trait defaults; abstract hooks stay unimplemented.
struct DefaultComponentForTest;

impl TreeComponent<Vec<i64>> for DefaultComponentForTest {
    fn component_type(&self) -> ComponentType {
        FULL
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
        target: EvictLayer,
    ) -> (usize, usize) {
        unimplemented!()
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
        params: Option<&DecLockRefParams>,
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
    let consumed_from = DefaultComponentForTest.update_component_on_insert_overlap(
        &mut tc,
        root,
        /* prefix_len = */ 3,
        /* total_prefix_len = */ 0,
        Tensor::from_slice(&[0i64, 1, 2]),
        &InsertParams {
            key: &vec![0, 1, 2],
            namespace: Default::default(),
            value: Tensor::from_slice(&[0i64, 1, 2]),
            mamba_value: None,
            prev_prefix_len: 0,
            swa_evicted_seqlen: 0,
            chunked: false,
            priority: 0,
            track_adopted_ranges: false,
        },
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
    let out = DefaultComponentForTest.finalize_match_result_in_tree_core(
        &tc,
        result,
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
    DefaultComponentForTest.drive_host_eviction(
        &mut tc,
        /* num_tokens = */ 100,
        &mut tracker,
        &mut device_frees,
        &mut host_frees,
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
