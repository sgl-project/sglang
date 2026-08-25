use std::sync::Mutex;

use tch::Tensor;

use super::*;
use crate::components::{FULL, MAMBA, SWA};
use crate::node::ValueSlotIdx;
use crate::test_utils::{accumulate_step, action_kinds};

fn core() -> UnifiedTreeCore<Vec<i64>> {
    UnifiedTreeCore::new(CacheInitParams::default(), vec![FULL])
}

// Records every refresh_lru dispatch; unrelated hooks stay unimplemented.
#[derive(Default)]
struct RecordingComponentForTest {
    refreshes: Mutex<Vec<(LRURefreshPhase, NodeIdx_)>>,
    host_eviction_calls: Mutex<Vec<(&'static str, usize)>>,
}

impl TreeComponent<Vec<i64>> for RecordingComponentForTest {
    fn component_type(&self) -> ComponentType {
        SWA
    }

    fn refresh_lru(
        &self,
        _tree_core: &mut UnifiedTreeCore<Vec<i64>>,
        phase: LRURefreshPhase,
        node_id: NodeIdx_,
    ) {
        self.refreshes.lock().unwrap().push((phase, node_id));
    }

    fn create_match_validator(
        &self,
        _tree_core: &UnifiedTreeCore<Vec<i64>>,
        _match_device_only: bool,
    ) -> Box<dyn FnMut(&UnifiedTreeCore<Vec<i64>>, NodeIdx_) -> bool> {
        Box::new(|_, _| true)
    }

    fn redistribute_on_node_split(
        &self,
        _tree_core: &mut UnifiedTreeCore<Vec<i64>>,
        _new_parent_id: NodeIdx_,
        _child_id: NodeIdx_,
    ) {
    }

    fn evict_component(
        &self,
        _tree_core: &mut UnifiedTreeCore<Vec<i64>>,
        _node_id: NodeIdx_,
        _device_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
        _host_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
        _target: EvictLayer,
    ) -> (usize, usize) {
        unimplemented!()
    }

    fn evict_device_start(&self, _tree_core: &mut UnifiedTreeCore<Vec<i64>>, _request_cnt: usize) {
        unimplemented!()
    }

    fn evict_device_next_node(
        &self,
        _tree_core: &mut UnifiedTreeCore<Vec<i64>>,
        _tracker: &mut HashMap<ComponentType, usize>,
        _device_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
        _host_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
    ) -> Option<NodeIdx_> {
        unimplemented!()
    }

    fn evict_device_end(&self, _tree_core: &mut UnifiedTreeCore<Vec<i64>>) {
        unimplemented!()
    }

    fn acquire_component_lock(
        &self,
        _tree_core: &mut UnifiedTreeCore<Vec<i64>>,
        _node_id: NodeIdx_,
        _result: IncLockRefResult,
        _lock_host: bool,
    ) -> IncLockRefResult {
        unimplemented!()
    }

    fn release_component_lock(
        &self,
        _tree_core: &mut UnifiedTreeCore<Vec<i64>>,
        _node_id: NodeIdx_,
        _params: Option<&DecLockRefParams>,
        _lock_host: bool,
    ) {
        unimplemented!()
    }

    fn reclaim_coexisting_host_values(
        &self,
        _tree_core: &mut UnifiedTreeCore<Vec<i64>>,
        _num_tokens: usize,
        tracker: &mut HashMap<ComponentType, usize>,
        _device_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
        _host_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
    ) {
        let tracked = tracker[&SWA];
        self.host_eviction_calls
            .lock()
            .unwrap()
            .push(("reclaim", tracked));
        tracker.insert(SWA, tracked + 2);
    }

    fn drive_host_eviction(
        &self,
        _tree_core: &mut UnifiedTreeCore<Vec<i64>>,
        _num_tokens: usize,
        tracker: &mut HashMap<ComponentType, usize>,
        _device_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
        _host_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
    ) {
        let tracked = tracker[&SWA];
        self.host_eviction_calls
            .lock()
            .unwrap()
            .push(("drive", tracked));
        tracker.insert(SWA, tracked + 3);
    }
}

// Counts every match-validator invocation; unrelated hooks stay unimplemented.
#[derive(Default)]
struct CountingComponentForTest {
    validator_calls: Arc<Mutex<usize>>,
}

impl TreeComponent<Vec<i64>> for CountingComponentForTest {
    fn component_type(&self) -> ComponentType {
        SWA
    }

    fn refresh_lru(
        &self,
        _tree_core: &mut UnifiedTreeCore<Vec<i64>>,
        _phase: LRURefreshPhase,
        _node_id: NodeIdx_,
    ) {
    }

    fn create_match_validator(
        &self,
        _tree_core: &UnifiedTreeCore<Vec<i64>>,
        _match_device_only: bool,
    ) -> Box<dyn FnMut(&UnifiedTreeCore<Vec<i64>>, NodeIdx_) -> bool> {
        let calls = Arc::clone(&self.validator_calls);
        Box::new(move |_, _| {
            *calls.lock().unwrap() += 1;
            true
        })
    }

    fn redistribute_on_node_split(
        &self,
        _tree_core: &mut UnifiedTreeCore<Vec<i64>>,
        _new_parent_id: NodeIdx_,
        _child_id: NodeIdx_,
    ) {
        unimplemented!()
    }

    fn evict_component(
        &self,
        _tree_core: &mut UnifiedTreeCore<Vec<i64>>,
        _node_id: NodeIdx_,
        _device_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
        _host_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
        _target: EvictLayer,
    ) -> (usize, usize) {
        unimplemented!()
    }

    fn evict_device_start(&self, _tree_core: &mut UnifiedTreeCore<Vec<i64>>, _request_cnt: usize) {
        unimplemented!()
    }

    fn evict_device_next_node(
        &self,
        _tree_core: &mut UnifiedTreeCore<Vec<i64>>,
        _tracker: &mut HashMap<ComponentType, usize>,
        _device_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
        _host_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
    ) -> Option<NodeIdx_> {
        unimplemented!()
    }

    fn evict_device_end(&self, _tree_core: &mut UnifiedTreeCore<Vec<i64>>) {
        unimplemented!()
    }

    fn acquire_component_lock(
        &self,
        _tree_core: &mut UnifiedTreeCore<Vec<i64>>,
        _node_id: NodeIdx_,
        _result: IncLockRefResult,
        _lock_host: bool,
    ) -> IncLockRefResult {
        unimplemented!()
    }

    fn release_component_lock(
        &self,
        _tree_core: &mut UnifiedTreeCore<Vec<i64>>,
        _node_id: NodeIdx_,
        _params: Option<&DecLockRefParams>,
        _lock_host: bool,
    ) {
        unimplemented!()
    }
}

// Swa-flavored stub driver: any dispatched call panics as unimplemented.
// A Mamba-slot double with internal priority 0; its release panics so a
// test can pin that dec_swa_lock_only dispatches lower-priority releases.
struct LowPriorityComponentForTest;

impl TreeComponent<Vec<i64>> for LowPriorityComponentForTest {
    fn component_type(&self) -> ComponentType {
        MAMBA
    }

    fn eviction_priority(&self, _is_leaf: bool) -> i64 {
        0
    }

    fn create_match_validator(
        &self,
        _tree_core: &UnifiedTreeCore<Vec<i64>>,
        _match_device_only: bool,
    ) -> Box<dyn FnMut(&UnifiedTreeCore<Vec<i64>>, NodeIdx_) -> bool> {
        unimplemented!()
    }

    fn redistribute_on_node_split(
        &self,
        _tree_core: &mut UnifiedTreeCore<Vec<i64>>,
        _new_parent_id: NodeIdx_,
        _child_id: NodeIdx_,
    ) {
    }

    fn evict_component(
        &self,
        _tree_core: &mut UnifiedTreeCore<Vec<i64>>,
        _node_id: NodeIdx_,
        _device_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
        _host_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
        _target: EvictLayer,
    ) -> (usize, usize) {
        unimplemented!()
    }

    fn evict_device_start(&self, _tree_core: &mut UnifiedTreeCore<Vec<i64>>, _request_cnt: usize) {
        unimplemented!()
    }

    fn evict_device_next_node(
        &self,
        _tree_core: &mut UnifiedTreeCore<Vec<i64>>,
        _tracker: &mut HashMap<ComponentType, usize>,
        _device_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
        _host_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
    ) -> Option<NodeIdx_> {
        unimplemented!()
    }

    fn evict_device_end(&self, _tree_core: &mut UnifiedTreeCore<Vec<i64>>) {
        unimplemented!()
    }

    fn acquire_component_lock(
        &self,
        _tree_core: &mut UnifiedTreeCore<Vec<i64>>,
        _node_id: NodeIdx_,
        _result: IncLockRefResult,
        _lock_host: bool,
    ) -> IncLockRefResult {
        unimplemented!()
    }

    fn release_component_lock(
        &self,
        _tree_core: &mut UnifiedTreeCore<Vec<i64>>,
        _node_id: NodeIdx_,
        params: Option<&DecLockRefParams>,
        lock_host: bool,
    ) {
        assert!(!lock_host);
        assert!(params.is_some_and(|p| p.swa_uuid_for_lock.is_some()));
        panic!("low-priority release dispatched");
    }
}

struct SwaComponentForTest;

impl TreeComponent<Vec<i64>> for SwaComponentForTest {
    fn component_type(&self) -> ComponentType {
        SWA
    }

    fn create_match_validator(
        &self,
        _tree_core: &UnifiedTreeCore<Vec<i64>>,
        _match_device_only: bool,
    ) -> Box<dyn FnMut(&UnifiedTreeCore<Vec<i64>>, NodeIdx_) -> bool> {
        unimplemented!()
    }

    fn redistribute_on_node_split(
        &self,
        _tree_core: &mut UnifiedTreeCore<Vec<i64>>,
        _new_parent_id: NodeIdx_,
        _child_id: NodeIdx_,
    ) {
    }

    fn evict_component(
        &self,
        _tree_core: &mut UnifiedTreeCore<Vec<i64>>,
        _node_id: NodeIdx_,
        _device_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
        _host_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
        _target: EvictLayer,
    ) -> (usize, usize) {
        unimplemented!()
    }

    fn evict_device_start(&self, _tree_core: &mut UnifiedTreeCore<Vec<i64>>, _request_cnt: usize) {
        unimplemented!()
    }

    fn evict_device_next_node(
        &self,
        _tree_core: &mut UnifiedTreeCore<Vec<i64>>,
        _tracker: &mut HashMap<ComponentType, usize>,
        _device_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
        _host_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
    ) -> Option<NodeIdx_> {
        unimplemented!()
    }

    fn evict_device_end(&self, _tree_core: &mut UnifiedTreeCore<Vec<i64>>) {
        unimplemented!()
    }

    fn acquire_component_lock(
        &self,
        _tree_core: &mut UnifiedTreeCore<Vec<i64>>,
        _node_id: NodeIdx_,
        _result: IncLockRefResult,
        _lock_host: bool,
    ) -> IncLockRefResult {
        unimplemented!()
    }

    fn release_component_lock(
        &self,
        _tree_core: &mut UnifiedTreeCore<Vec<i64>>,
        _node_id: NodeIdx_,
        _params: Option<&DecLockRefParams>,
        _lock_host: bool,
    ) {
        unimplemented!()
    }
}

// Swa-flavored driver with working eviction hooks: evict_component frees the
// Swa slot and records the call; eviction priorities are configurable.
struct SwaEvictionComponentForTest {
    leaf_priority: i64,
    internal_priority: i64,
    evictions: Mutex<Vec<(NodeIdx_, EvictLayer)>>,
}

impl SwaEvictionComponentForTest {
    fn new(leaf_priority: i64, internal_priority: i64) -> Self {
        SwaEvictionComponentForTest {
            leaf_priority,
            internal_priority,
            evictions: Mutex::new(Vec::new()),
        }
    }
}

impl TreeComponent<Vec<i64>> for SwaEvictionComponentForTest {
    fn component_type(&self) -> ComponentType {
        SWA
    }

    fn eviction_priority(&self, is_leaf: bool) -> i64 {
        if is_leaf {
            self.leaf_priority
        } else {
            self.internal_priority
        }
    }

    fn evict_component(
        &self,
        tree_core: &mut UnifiedTreeCore<Vec<i64>>,
        node_id: NodeIdx_,
        device_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
        host_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
        target: EvictLayer,
    ) -> (usize, usize) {
        self.evictions.lock().unwrap().push((node_id, target));
        let node = tree_core.arena.node_mut(node_id);
        let mut device_freed = 0;
        let mut host_freed = 0;
        if target.contains(EvictLayer::Device)
            && let Some(value) = node.values[SWA.idx()].value.take()
        {
            device_freed = value.size()[0] as usize;
            device_frees.entry(SWA).or_default().push(value);
        }
        if target.contains(EvictLayer::Host)
            && let Some(value) = node.state_mut_(ValueSlotIdx::host(SWA)).value.take()
        {
            host_freed = value.size()[0] as usize;
            host_frees.entry(SWA).or_default().push(value);
        }
        (device_freed, host_freed)
    }

    fn create_match_validator(
        &self,
        _tree_core: &UnifiedTreeCore<Vec<i64>>,
        _match_device_only: bool,
    ) -> Box<dyn FnMut(&UnifiedTreeCore<Vec<i64>>, NodeIdx_) -> bool> {
        unimplemented!()
    }

    fn redistribute_on_node_split(
        &self,
        _tree_core: &mut UnifiedTreeCore<Vec<i64>>,
        _new_parent_id: NodeIdx_,
        _child_id: NodeIdx_,
    ) {
    }

    fn evict_device_start(&self, _tree_core: &mut UnifiedTreeCore<Vec<i64>>, _request_cnt: usize) {
        unimplemented!()
    }

    fn evict_device_next_node(
        &self,
        _tree_core: &mut UnifiedTreeCore<Vec<i64>>,
        _tracker: &mut HashMap<ComponentType, usize>,
        _device_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
        _host_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
    ) -> Option<NodeIdx_> {
        unimplemented!()
    }

    fn evict_device_end(&self, _tree_core: &mut UnifiedTreeCore<Vec<i64>>) {
        unimplemented!()
    }

    fn acquire_component_lock(
        &self,
        _tree_core: &mut UnifiedTreeCore<Vec<i64>>,
        _node_id: NodeIdx_,
        _result: IncLockRefResult,
        _lock_host: bool,
    ) -> IncLockRefResult {
        unimplemented!()
    }

    fn release_component_lock(
        &self,
        _tree_core: &mut UnifiedTreeCore<Vec<i64>>,
        _node_id: NodeIdx_,
        _params: Option<&DecLockRefParams>,
        _lock_host: bool,
    ) {
        unimplemented!()
    }
}

// A locked non-root anchor; component dispatch bypasses roots entirely.
fn locked_anchor_for_dispatch(tc: &mut UnifiedTreeCore<Vec<i64>>) -> NodeIdx_ {
    let root = tc.arena.root();
    let n1 = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![1, 11],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    tc.arena
        .set_device_value(n1, FULL, Tensor::from_slice(&[0i64, 1]));
    tc.component_state_mut(FULL).evictable_size = 2;
    tc.inc_lock_ref(tc.arena.node(n1).id);
    n1
}

#[test]
fn dec_lock_ref_skip_swa_skips_the_swa_component() {
    let mut tc = core();
    let n1 = locked_anchor_for_dispatch(&mut tc);
    tc.register_component_(Arc::new(SwaComponentForTest));
    // The skipped Swa driver is never dispatched, so its stub cannot panic.
    tc.dec_lock_ref(
        tc.arena.node(n1).id,
        /* params = */ None,
        /* skip_swa = */ true,
    );
    assert_eq!(tc.arena.device_lock_ref(n1, FULL), 0);
}

#[test]
#[should_panic(expected = "not implemented")]
fn inc_lock_ref_reaches_every_component() {
    let mut tc = core();
    tc.register_component_(Arc::new(SwaComponentForTest));
    let root = tc.arena.root();
    let n1 = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![1, 11],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    tc.arena
        .set_device_value(n1, FULL, Tensor::from_slice(&[0i64, 1]));
    tc.component_state_mut(FULL).evictable_size = 2;
    tc.inc_lock_ref(tc.arena.node(n1).id);
}

#[test]
#[should_panic(expected = "not implemented")]
fn dec_lock_ref_without_skip_swa_reaches_every_component() {
    let mut tc = core();
    let n1 = locked_anchor_for_dispatch(&mut tc);
    tc.register_component_(Arc::new(SwaComponentForTest));
    tc.dec_lock_ref(
        tc.arena.node(n1).id,
        /* params = */ None,
        /* skip_swa = */ false,
    );
}

#[test]
fn set_component_device_value_sizes_by_the_value_length() {
    let mut tc = core();
    tc.register_component_(Arc::new(SwaComponentForTest));
    let root = tc.arena.root();
    let node = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![1, 2, 3],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    tc.set_component_device_value(
        tc.arena.node(node).id,
        SWA,
        Tensor::from_slice(&[7i64, 8, 9]),
    );
    assert!(
        tc.arena
            .device_value(node, SWA)
            .equal(&Tensor::from_slice(&[7i64, 8, 9]))
    );
    assert_eq!(tc.evictable_size_(SWA), 3);
}

#[test]
#[should_panic(expected = "slot already set")]
fn set_component_device_value_rejects_an_occupied_slot() {
    let mut tc = core();
    tc.register_component_(Arc::new(SwaComponentForTest));
    let root = tc.arena.root();
    let node = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![1],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    tc.set_component_device_value(tc.arena.node(node).id, SWA, Tensor::from_slice(&[7i64]));
    tc.device_lru_list_mut(SWA).remove_node(node);
    tc.set_component_device_value(tc.arena.node(node).id, SWA, Tensor::from_slice(&[8i64]));
}

#[test]
#[should_panic(expected = "Swa component is not enabled")]
fn set_component_device_value_rejects_a_disabled_component() {
    let mut tc = core();
    let root = tc.arena.root();
    tc.set_component_device_value(tc.arena.node(root).id, SWA, Tensor::from_slice(&[1i64]));
}

#[test]
#[should_panic(expected = "low-priority release dispatched")]
fn dec_swa_lock_only_dispatches_lower_priority_releases() {
    let mut tc: UnifiedTreeCore<Vec<i64>> = UnifiedTreeCore::new(
        CacheInitParams {
            swa_sliding_window_size: Some(4),
            ..Default::default()
        },
        vec![FULL, SWA],
    );
    tc.register_component_(Arc::new(LowPriorityComponentForTest));
    let root = tc.arena.root();
    let mut device_frees = HashMap::new();
    let mut host_frees = HashMap::new();
    tc.dec_swa_lock_only(
        tc.arena.node(root).id,
        Some(7),
        &mut device_frees,
        &mut host_frees,
    );
}

#[test]
fn dec_swa_lock_only_returns_device_frees_in_the_device_dict() {
    let params = CacheInitParams {
        swa_sliding_window_size: Some(2),
        ..Default::default()
    };
    let mut tc: UnifiedTreeCore<Vec<i64>> = UnifiedTreeCore::new(params, vec![FULL, SWA]);
    let root = tc.arena.root();
    let a = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![1],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    tc.arena
        .set_device_value(a, FULL, Tensor::from_slice(&[9i64]));
    tc.set_component_device_value(tc.arena.node(a).id, SWA, Tensor::from_slice(&[7i64]));
    let swa = SwaComponent::new(&CacheInitParams {
        swa_sliding_window_size: Some(2),
        ..Default::default()
    });
    let result = swa.acquire_component_lock(
        &mut tc,
        a,
        IncLockRefResult::default(),
        /* lock_host = */ false,
    );
    let mut device_frees = HashMap::new();
    let mut host_frees = HashMap::new();
    tc.dec_swa_lock_only(
        tc.arena.node(a).id,
        result.swa_uuid_for_lock,
        &mut device_frees,
        &mut host_frees,
    );
    // The fully unlocked D-leaf's SWA value is device-evicted on release;
    // the freed span is reported as the node's Full indices.
    assert!(!tc.arena.has_device_value(a, SWA));
    assert_eq!(device_frees[&SWA].len(), 1);
    assert!(device_frees[&SWA][0].equal(&Tensor::from_slice(&[9i64])));
    assert!(host_frees.is_empty());
}

#[test]
fn next_swa_uuid_counts_up_from_two() {
    let mut tc = core();
    assert_eq!(tc.next_swa_uuid_(), 2);
    assert_eq!(tc.next_swa_uuid_(), 3);
}

#[test]
fn inc_lock_ref_result_defaults_carry_no_uuids() {
    let result = IncLockRefResult::default();
    assert_eq!(result.swa_uuid_for_lock, None);
    assert_eq!(result.swa_uuid_for_host_lock, None);
}

// A tree with the Swa stub registered and a node carrying an SWA device value.
fn swa_valued_node(tc: &mut UnifiedTreeCore<Vec<i64>>) -> NodeIdx_ {
    tc.register_component_(Arc::new(SwaComponentForTest));
    let root = tc.arena.root();
    let a = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![1],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    tc.arena.node_mut(a).values[SWA.idx()].value = Some(Tensor::from_slice(&[0i64]));
    a
}

#[test]
fn for_each_component_lru_visits_valued_aux_components_only() {
    let mut tc = core();
    let a = swa_valued_node(&mut tc);
    // A Full device value must not draw an LRU visit (Full uses leaf sets).
    tc.arena
        .set_device_value(a, FULL, Tensor::from_slice(&[0i64]));
    let mut visited = Vec::new();
    tc.for_each_component_lru_(
        a,
        &mut |lru, node_id| {
            lru.insert_mru(node_id);
            visited.push(node_id);
        },
        EvictLayer::Device,
        /* skip_existing = */ false,
    );
    assert_eq!(visited, vec![a]);
    assert!(tc.device_lru_list(SWA).in_list(Some(a)));
    assert!(!tc.device_lru_list(FULL).in_list(Some(a)));
}

#[test]
fn for_each_component_lru_skips_valueless_components() {
    let mut tc = core();
    tc.register_component_(Arc::new(SwaComponentForTest));
    let root = tc.arena.root();
    let a = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![1],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    let mut visits = 0;
    tc.for_each_component_lru_(
        a,
        &mut |_, _| visits += 1,
        EvictLayer::Device,
        /* skip_existing = */ false,
    );
    assert_eq!(visits, 0);
}

#[test]
fn for_each_component_lru_targets_the_host_tier() {
    let mut tc = core();
    tc.register_component_(Arc::new(SwaComponentForTest));
    let root = tc.arena.root();
    let a = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![1],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    tc.arena
        .node_mut(a)
        .state_mut_(ValueSlotIdx::host(SWA))
        .value = Some(Tensor::from_slice(&[0i64]));
    let mut visited = Vec::new();
    tc.for_each_component_lru_(
        a,
        &mut |lru, node_id| {
            lru.insert_mru(node_id);
            visited.push(node_id);
        },
        EvictLayer::Host,
        /* skip_existing = */ false,
    );
    assert_eq!(visited, vec![a]);
    assert!(tc.host_lru_list(SWA).in_list(Some(a)));
    // The device-tier walk sees no device value on the node.
    let mut device_visits = 0;
    tc.for_each_component_lru_(
        a,
        &mut |_, _| device_visits += 1,
        EvictLayer::Device,
        /* skip_existing = */ false,
    );
    assert_eq!(device_visits, 0);
}

#[test]
fn for_each_component_lru_skip_existing_spares_listed_nodes() {
    let mut tc = core();
    let a = swa_valued_node(&mut tc);
    tc.device_lru_list_mut(SWA).insert_mru(a);
    let mut visits = 0;
    tc.for_each_component_lru_(
        a,
        &mut |_, _| visits += 1,
        EvictLayer::Device,
        /* skip_existing = */ true,
    );
    assert_eq!(visits, 0);
    // Without the flag, the listed node is visited again.
    tc.for_each_component_lru_(
        a,
        &mut |_, _| visits += 1,
        EvictLayer::Device,
        /* skip_existing = */ false,
    );
    assert_eq!(visits, 1);
}

#[test]
fn new_node_allocates_a_half_linked_stamped_node() {
    let mut tc = core();
    let root = tc.arena.root();
    let before = tc.arena.node(root).last_access_counter;
    let a = tc.new_node_(
        /* key = */ vec![5],
        root,
        /* priority = */ 7,
        /* hit_count = */ 3,
        /* creation_counter = */ None,
        /* extra_key = */ None,
    );
    let node = tc.arena.node(a);
    assert_eq!(node.priority, 7);
    assert_eq!(node.hit_count, 3);
    assert_eq!(node.key, vec![5]);
    assert_eq!(node.parent(), root);
    assert!(node.children.is_empty());
    // Half-linked: the parent's child map does not know the node yet.
    assert!(tc.arena.root_child(None, &[5]).is_none());
    // Creation and access stamps share one fresh tick.
    assert_eq!(node.creation_counter, node.last_access_counter);
    assert!(node.last_access_counter > before);
}

#[test]
fn new_node_ids_are_distinct_live_slots() {
    let mut tc = core();
    let root = tc.arena.root();
    let a = tc.new_node_(
        /* key = */ vec![1],
        root,
        /* priority = */ 0,
        /* hit_count = */ 0,
        /* creation_counter = */ None,
        /* extra_key = */ None,
    );
    let b = tc.new_node_(
        /* key = */ vec![2],
        root,
        /* priority = */ 0,
        /* hit_count = */ 0,
        /* creation_counter = */ None,
        /* extra_key = */ None,
    );
    assert_ne!(a, b);
    assert_eq!(tc.arena.resolve(tc.arena.node(a).id), a);
    assert_eq!(tc.arena.resolve(tc.arena.node(b).id), b);
}

// Chain root -> c with a 3-atom key and FULL device value, seeded as a D-leaf.
fn split_setup(tc: &mut UnifiedTreeCore<Vec<i64>>) -> NodeIdx_ {
    let root = tc.arena.root();
    let c = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![1, 2, 3],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    tc.arena
        .set_device_value(c, FULL, Tensor::from_slice(&[10i64, 11, 12]));
    tc.evictable_device_leaves.add(c);
    c
}

#[test]
fn split_wires_the_new_node_between_parent_and_child() {
    let mut tc = core();
    let root = tc.arena.root();
    let c = split_setup(&mut tc);
    let (new_node, action) = tc.split_node_(c, /* split_len = */ 2);
    assert!(action.is_none());
    assert_eq!(tc.arena.node(new_node).key, vec![1, 2]);
    assert_eq!(tc.arena.node(c).key, vec![3]);
    assert_eq!(tc.arena.root_child(None, &[1]).as_ref(), Some(&new_node));
    assert_eq!(
        tc.arena.node(new_node).children.get(&(None, vec![3])),
        Some(&c)
    );
    assert_eq!(tc.arena.node(new_node).parent(), root);
    assert_eq!(tc.arena.node(c).parent(), new_node);
}

#[test]
fn split_redistributes_the_device_value_and_locks() {
    let mut tc = core();
    let c = split_setup(&mut tc);
    tc.arena
        .node_mut(c)
        .set_lock_ref_(ValueSlotIdx::device(FULL), 2);
    let (new_node, _) = tc.split_node_(c, /* split_len = */ 2);
    assert_eq!(tc.arena.device_value_len(new_node, FULL), 2);
    assert_eq!(tc.arena.device_value_len(c, FULL), 1);
    assert_eq!(tc.arena.device_lock_ref(new_node, FULL), 2);
    assert_eq!(tc.arena.device_lock_ref(c, FULL), 2);
}

#[test]
fn split_copies_stats_and_restamps_the_child() {
    let mut tc = core();
    let c = split_setup(&mut tc);
    tc.arena.node_mut(c).hit_count = 5;
    let creation = tc.arena.node(c).creation_counter;
    let access_before = tc.arena.node(c).last_access_counter;
    let (new_node, _) = tc.split_node_(c, /* split_len = */ 2);
    // The prefix node inherits hits and creation; the child gets a fresh access tick.
    assert_eq!(tc.arena.node(new_node).hit_count, 5);
    assert_eq!(tc.arena.node(new_node).creation_counter, creation);
    assert!(tc.arena.node(c).last_access_counter > access_before);
}

#[test]
fn split_propagates_the_child_priority() {
    let mut tc = core();
    let c = split_setup(&mut tc);
    tc.arena.node_mut(c).priority = 7;
    let (new_node, _) = tc.split_node_(c, /* split_len = */ 2);
    assert_eq!(tc.arena.node(new_node).priority, 7);
    assert_eq!(tc.arena.node(c).priority, 7);
}

#[test]
fn split_updates_the_leaf_sets() {
    let mut tc = core();
    let c = split_setup(&mut tc);
    let (new_node, _) = tc.split_node_(c, /* split_len = */ 2);
    // The child stays the D-leaf; the prefix node has a valued child.
    assert!(tc.evictable_device_leaves.contains(c));
    assert!(!tc.evictable_device_leaves.contains(new_node));
}

#[test]
fn split_readmits_aux_lru_cells() {
    let mut tc = core();
    tc.register_component_(Arc::new(SwaComponentForTest));
    let c = split_setup(&mut tc);
    tc.arena.node_mut(c).values[SWA.idx()].value = Some(Tensor::from_slice(&[0i64]));
    tc.device_lru_list_mut(SWA).insert_mru(c);
    // A second listed node makes the child's detach-and-readmit observable.
    let root = tc.arena.root();
    let s = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![9],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    tc.device_lru_list_mut(SWA).insert_mru(s);
    let (new_node, _) = tc.split_node_(c, /* split_len = */ 2);
    // The child re-enters the SWA LRU at MRU; the value-less prefix node does not.
    assert!(tc.device_lru_list(SWA).in_list(Some(c)));
    assert!(!tc.device_lru_list(SWA).in_list(Some(new_node)));
    assert_eq!(tc.device_lru_list(SWA).get_lru_where(|_| true), Some(s));
}

#[test]
#[should_panic(expected = "split_node_: the parent's page entry must map to the split child")]
fn split_panics_when_the_parent_entry_is_not_the_child() {
    let mut tc = core();
    let root = tc.arena.root();
    let c = split_setup(&mut tc);
    // A corrupted parent map (the page entry no longer points at c) must fail loudly.
    let imposter = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![9],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    tc.arena.insert_child_edge(root, vec![1], imposter);
    tc.split_node_(c, /* split_len = */ 2);
}

#[test]
#[should_panic(expected = "split_at: split_idx 3 out of range (0, 3)")]
fn split_panics_on_a_boundary_split() {
    let mut tc = core();
    let c = split_setup(&mut tc);
    tc.split_node_(c, /* split_len = */ 3);
}

#[test]
#[should_panic(expected = "split_node_: split_len 0 must be a nonzero page multiple")]
fn split_panics_on_a_zero_split_len() {
    let mut tc = core();
    let c = split_setup(&mut tc);
    tc.split_node_(c, /* split_len = */ 0);
}

#[test]
fn add_new_node_creates_a_valued_child() {
    let mut tc = core();
    let root = tc.arena.root();
    let mut source = Tensor::from_slice(&[10i64, 11]);
    let a = tc.add_new_node_(
        root,
        /* key = */ vec![1, 2],
        &source,
        /* priority = */ 3,
        /* extra_key = */ None,
    );
    let node = tc.arena.node(a);
    assert_eq!(node.key, vec![1, 2]);
    assert_eq!(node.parent(), root);
    assert_eq!(node.priority, 3);
    assert_eq!(tc.arena.root_child(None, &[1]).as_ref(), Some(&a));
    assert_eq!(tc.evictable_size_(FULL), 2);
    assert!(tc.evictable_device_leaves.contains(a));
    assert!(!tc.evictable_device_leaves.contains(root));
    // The stored value is a deep copy of the insert slice.
    let _ = source.fill_(99);
    assert!(
        tc.arena
            .device_value(a, FULL)
            .equal(&Tensor::from_slice(&[10i64, 11]))
    );
}

#[test]
fn add_new_node_retires_the_parent_from_the_leaf_set() {
    let mut tc = core();
    let root = tc.arena.root();
    let p = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![1],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    tc.arena
        .set_device_value(p, FULL, Tensor::from_slice(&[0i64]));
    tc.evictable_device_leaves.add(p);
    let value = Tensor::from_slice(&[10i64]);
    let a = tc.add_new_node_(
        p,
        /* key = */ vec![2],
        &value,
        /* priority = */ 0,
        /* extra_key = */ None,
    );
    // The new leaf takes over; the parent now has a valued child.
    assert!(tc.evictable_device_leaves.contains(a));
    assert!(!tc.evictable_device_leaves.contains(p));
}

#[test]
#[should_panic(expected = "already has a child on the new node's page")]
fn add_new_node_panics_when_the_page_is_taken() {
    let mut tc = core();
    let root = tc.arena.root();
    let value = Tensor::from_slice(&[10i64]);
    tc.add_new_node_(
        root,
        /* key = */ vec![1],
        &value,
        /* priority = */ 0,
        /* extra_key = */ None,
    );
    tc.add_new_node_(
        root,
        /* key = */ vec![1],
        &value,
        /* priority = */ 0,
        /* extra_key = */ None,
    );
}

#[test]
fn unevict_restores_the_value_and_the_leaf_sets() {
    // Chain root -> p (valued) -> c (evicted): p is the D-leaf until c revives.
    let mut tc = core();
    let root = tc.arena.root();
    let p = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![1],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    let c = tc
        .arena
        .alloc_child(
            p,
            /* key = */ vec![2],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    tc.arena
        .set_device_value(p, FULL, Tensor::from_slice(&[0i64]));
    tc.evictable_device_leaves.add(p);
    let mut fresh = Tensor::from_slice(&[20i64]);
    tc.unevict_node_on_insert_(c, &fresh);
    assert_eq!(tc.evictable_size_(FULL), 1);
    assert!(tc.evictable_device_leaves.contains(c));
    assert!(!tc.evictable_device_leaves.contains(p));
    // The restored value is a deep copy of the fresh indices.
    let _ = fresh.fill_(99);
    assert!(
        tc.arena
            .device_value(c, FULL)
            .equal(&Tensor::from_slice(&[20i64]))
    );
}

#[test]
#[should_panic(expected = "slot already set")]
fn unevict_panics_on_a_node_that_still_has_its_value() {
    let mut tc = core();
    let root = tc.arena.root();
    let a = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![1],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    tc.arena
        .set_device_value(a, FULL, Tensor::from_slice(&[0i64]));
    tc.unevict_node_on_insert_(a, &Tensor::from_slice(&[1i64]));
}

fn match_params(key: &Vec<i64>) -> MatchPrefixParams<'_, Vec<i64>> {
    MatchPrefixParams {
        key,
        extra_key: None,
    }
}

// root -> a (key [1,2], kv [10,11]) -> b (key [3], kv [12]).
fn matched_chain(tc: &mut UnifiedTreeCore<Vec<i64>>) -> (NodeIdx_, NodeIdx_) {
    let root = tc.arena.root();
    let a = tc.add_new_node_(
        root,
        /* key = */ vec![1, 2],
        &Tensor::from_slice(&[10i64, 11]),
        /* priority = */ 0,
        /* extra_key = */ None,
    );
    let b = tc.add_new_node_(
        a,
        /* key = */ vec![3],
        &Tensor::from_slice(&[12i64]),
        /* priority = */ 0,
        /* extra_key = */ None,
    );
    (a, b)
}

#[test]
fn match_prefix_returns_the_full_hit() {
    let mut tc = core();
    let (_a, b) = matched_chain(&mut tc);
    let result = tc.match_prefix(&match_params(&vec![1, 2, 3]));
    assert!(
        result
            .device_indices
            .equal(&Tensor::from_slice(&[10i64, 11, 12]))
    );
    assert_eq!(result.last_device_node_id, tc.arena.node(b).id);
    assert_eq!(result.last_host_node_id, tc.arena.node(b).id);
    assert_eq!(result.best_match_node_id, tc.arena.node(b).id);
    assert_eq!(result.host_hit_length, 0);
    assert!(result.cache_actions.is_empty());
}

#[test]
fn match_prefix_stops_at_the_matched_depth() {
    let mut tc = core();
    let (a, _b) = matched_chain(&mut tc);
    let result = tc.match_prefix(&match_params(&vec![1, 2, 9]));
    assert!(
        result
            .device_indices
            .equal(&Tensor::from_slice(&[10i64, 11]))
    );
    assert_eq!(result.best_match_node_id, tc.arena.node(a).id);
}

#[test]
fn match_prefix_miss_anchors_at_the_root() {
    let mut tc = core();
    matched_chain(&mut tc);
    let root = tc.arena.root();
    let result = tc.match_prefix(&match_params(&vec![9]));
    assert_eq!(result.device_indices.numel(), 0);
    assert_eq!(result.best_match_node_id, tc.arena.node(root).id);
    assert_eq!(result.last_device_node_id, tc.arena.node(root).id);
}

#[test]
fn match_prefix_splits_on_a_partial_match() {
    let mut tc = core();
    let (a, _b) = matched_chain(&mut tc);
    let result = tc.match_prefix(&match_params(&vec![1, 9]));
    // The walk split a at 1: the new prefix node holds [10] and anchors the result.
    let prefix_node = result.best_match_node_id;
    assert_ne!(prefix_node, tc.arena.node(a).id);
    assert!(result.device_indices.equal(&Tensor::from_slice(&[10i64])));
    assert_eq!(tc.arena.node(tc.arena.resolve(prefix_node)).key, vec![1]);
    assert_eq!(tc.arena.node(a).key, vec![2]);
    assert_eq!(tc.arena.node(a).parent(), tc.arena.resolve(prefix_node));
}

#[test]
fn match_prefix_stops_at_a_dead_node() {
    // An evicted, unbackuped child ends the traversal before it.
    let mut tc = core();
    let root = tc.arena.root();
    let a = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![1, 2],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    let _ = a;
    let result = tc.match_prefix(&match_params(&vec![1, 2]));
    assert_eq!(result.device_indices.numel(), 0);
    assert_eq!(result.best_match_node_id, tc.arena.node(root).id);
}

#[test]
fn match_prefix_page_aligns_the_query() {
    let params = CacheInitParams {
        page_size: 2,
        ..Default::default()
    };
    let mut tc: UnifiedTreeCore<Vec<i64>> = UnifiedTreeCore::new(params, vec![FULL]);
    let root = tc.arena.root();
    let a = tc.add_new_node_(
        root,
        /* key = */ vec![1, 2],
        &Tensor::from_slice(&[10i64, 11]),
        /* priority = */ 0,
        /* extra_key = */ None,
    );
    // The trailing partial page is dropped before the walk.
    let result = tc.match_prefix(&match_params(&vec![1, 2, 3]));
    assert!(
        result
            .device_indices
            .equal(&Tensor::from_slice(&[10i64, 11]))
    );
    assert_eq!(result.best_match_node_id, tc.arena.node(a).id);
}

#[test]
fn match_prefix_restamps_the_matched_path_newest_first() {
    let mut tc = core();
    let (a, b) = matched_chain(&mut tc);
    let before = tc.arena.node(b).last_access_counter;
    tc.match_prefix(&match_params(&vec![1, 2, 3]));
    let root_id = tc.arena.root();
    let root_tick = tc.arena.node(root_id).last_access_counter;
    let a_tick = tc.arena.node(a).last_access_counter;
    let b_tick = tc.arena.node(b).last_access_counter;
    assert!(b_tick > before);
    assert!(b_tick > a_tick);
    assert!(a_tick > root_tick);
}

#[test]
fn insert_first_write_creates_the_namespace() {
    let mut tc = core();
    matched_chain(&mut tc);
    tc.insert(&InsertParams {
        extra_key: Some("lora-1"),
        ..insert_params(&vec![7, 8], &[40, 41])
    });
    // The namespace is isolated under its own root edges, created by the write.
    assert!(tc.arena.namespace_exists(Some("lora-1")));
    let result = tc.match_prefix(&MatchPrefixParams {
        key: &vec![1, 2],
        extra_key: Some("lora-1"),
    });
    assert_eq!(result.device_indices.numel(), 0);
    assert_eq!(result.best_match_node_id, tc.root_node_handle(None));
}

#[test]
fn match_prefix_empty_query_anchors_at_the_root() {
    let mut tc = core();
    let root_handle = tc.root_node_handle(None);
    for extra_key in [Some("lora-1"), None] {
        let result = tc.match_prefix(&MatchPrefixParams {
            key: &vec![],
            extra_key,
        });
        assert_eq!(result.best_match_node_id, root_handle);
        assert_eq!(result.last_device_node_id, root_handle);
        assert_eq!(result.last_host_node_id, root_handle);
    }
    // A read never creates a namespace.
    assert!(!tc.arena.namespace_exists(Some("lora-1")));
}

#[test]
fn match_prefix_skips_host_only_nodes_without_hicache() {
    // Chain a(valued) -> b(evicted but backuped): the walk passes b but the
    // device-only validator keeps the boundary at a.
    let mut tc = core();
    let (a, b) = matched_chain(&mut tc);
    let taken = tc.arena.take_device_value(b, FULL);
    tc.arena.set_host_value(b, FULL, taken);
    let result = tc.match_prefix(&match_params(&vec![1, 2, 3]));
    assert!(
        result
            .device_indices
            .equal(&Tensor::from_slice(&[10i64, 11]))
    );
    assert_eq!(result.best_match_node_id, tc.arena.node(a).id);
    assert_eq!(result.last_device_node_id, tc.arena.node(a).id);
}

#[test]
fn match_prefix_with_hicache_advances_best_match_onto_host_nodes() {
    // Chain a(valued) -> b(evicted but backuped): the device anchor stays at
    // a while the hicache validators carry the best match onto b.
    let mut tc = core();
    tc.set_hicache_enabled();
    let (a, b) = matched_chain(&mut tc);
    let taken = tc.arena.take_device_value(b, FULL);
    tc.arena.set_host_value(b, FULL, taken);
    let result = tc.match_prefix(&match_params(&vec![1, 2, 3]));
    assert!(
        result
            .device_indices
            .equal(&Tensor::from_slice(&[10i64, 11]))
    );
    assert_eq!(result.last_device_node_id, tc.arena.node(a).id);
    assert_eq!(result.best_match_node_id, tc.arena.node(b).id);
    assert_eq!(result.last_host_node_id, tc.arena.node(b).id);
    assert_eq!(result.host_hit_length, 1);
    assert!(result.cache_actions.is_empty());
}

#[test]
fn match_prefix_with_hicache_restamps_the_host_best_match() {
    // Chain a(valued) -> b(host-only): the restamp walk anchors at the
    // consensus best match b, not the shallower device anchor a.
    let mut tc = core();
    tc.set_hicache_enabled();
    let (a, b) = matched_chain(&mut tc);
    let taken = tc.arena.take_device_value(b, FULL);
    tc.arena.set_host_value(b, FULL, taken);
    let before = tc.arena.node(b).last_access_counter;
    tc.match_prefix(&match_params(&vec![1, 2, 3]));
    let a_tick = tc.arena.node(a).last_access_counter;
    let b_tick = tc.arena.node(b).last_access_counter;
    assert!(b_tick > before);
    assert!(b_tick > a_tick);
}

#[test]
fn match_prefix_with_hicache_sums_the_host_span_length() {
    let mut tc = core();
    tc.set_hicache_enabled();
    let (a, b) = matched_chain(&mut tc);
    let c = tc.add_new_node_(
        b,
        /* key = */ vec![4, 5],
        &Tensor::from_slice(&[13i64, 14]),
        /* priority = */ 0,
        /* extra_key = */ None,
    );
    let taken = tc.arena.take_device_value(b, FULL);
    tc.arena.set_host_value(b, FULL, taken);
    let taken = tc.arena.take_device_value(c, FULL);
    tc.arena.set_host_value(c, FULL, taken);
    let result = tc.match_prefix(&match_params(&vec![1, 2, 3, 4, 5]));
    assert!(
        result
            .device_indices
            .equal(&Tensor::from_slice(&[10i64, 11]))
    );
    assert_eq!(result.last_device_node_id, tc.arena.node(a).id);
    assert_eq!(result.best_match_node_id, tc.arena.node(c).id);
    assert_eq!(result.last_host_node_id, tc.arena.node(c).id);
    assert_eq!(result.host_hit_length, 3);
}

#[test]
fn match_walk_runs_every_validator_at_a_host_only_node() {
    // Chain a(valued) -> b(host-only): Full's device-only validator is
    // false at b, yet the aux validator must still observe b.
    let mut tc = core();
    let counter = Arc::new(CountingComponentForTest::default());
    tc.register_component_(counter.clone());
    let (_a, b) = matched_chain(&mut tc);
    let taken = tc.arena.take_device_value(b, FULL);
    tc.arena.set_host_value(b, FULL, taken);
    tc.match_prefix(&match_params(&vec![1, 2, 3]));
    assert_eq!(*counter.validator_calls.lock().unwrap(), 2);
}

#[test]
fn match_walk_runs_every_device_validator_under_hicache() {
    // Both hicache folds observe both nodes: Full's device validator is
    // false at the host-only b, yet the aux device validator still runs there.
    let mut tc = core();
    tc.set_hicache_enabled();
    let counter = Arc::new(CountingComponentForTest::default());
    tc.register_component_(counter.clone());
    let (_a, b) = matched_chain(&mut tc);
    let taken = tc.arena.take_device_value(b, FULL);
    tc.arena.set_host_value(b, FULL, taken);
    tc.match_prefix(&match_params(&vec![1, 2, 3]));
    assert_eq!(*counter.validator_calls.lock().unwrap(), 4);
}

#[test]
fn match_end_refresh_anchors_at_the_consensus_best_match() {
    // Chain a(valued) -> b(host-only): the device anchor stays at a while
    // the MatchEnd refresh dispatches on the consensus best match b.
    let mut tc = core();
    tc.set_hicache_enabled();
    let recorder = Arc::new(RecordingComponentForTest::default());
    tc.register_component_(recorder.clone());
    let (a, b) = matched_chain(&mut tc);
    let taken = tc.arena.take_device_value(b, FULL);
    tc.arena.set_host_value(b, FULL, taken);
    let result = tc.match_prefix(&match_params(&vec![1, 2, 3]));
    assert_eq!(result.last_device_node_id, tc.arena.node(a).id);
    assert_eq!(result.best_match_node_id, tc.arena.node(b).id);
    let refreshes = recorder.refreshes.lock().unwrap();
    assert!(
        refreshes
            .iter()
            .any(|&(phase, node)| phase == LRURefreshPhase::MatchEnd && node == b)
    );
}

// A [Full, Swa] core with the given sliding window (page size 1).
fn swa_match_core(window: usize) -> UnifiedTreeCore<Vec<i64>> {
    UnifiedTreeCore::new(
        CacheInitParams {
            swa_sliding_window_size: Some(window),
            ..Default::default()
        },
        vec![FULL, SWA],
    )
}

// Stamp a key-covering SWA device value on the node and list it in the SWA LRU.
fn set_swa_device_and_list(tc: &mut UnifiedTreeCore<Vec<i64>>, node: NodeIdx_) {
    let len = tc.arena.node(node).key.atom_len();
    tc.arena.node_mut(node).values[SWA.idx()].value = Some(Tensor::from_slice(&vec![0i64; len]));
    tc.device_lru_list_mut(SWA).insert_mru(node);
}

#[test]
fn match_walk_swa_state_resets_at_a_full_rejected_node() {
    // Chain a(SWA on) -> b(Full host-only, SWA tombstone) -> c(SWA on,
    // below the window): Full's validator rejects b, but the SWA validator
    // must still observe b so its window run restarts before c.
    let mut tc = swa_match_core(/* window = */ 2);
    let root = tc.arena.root();
    let a = tc.add_new_node_(
        root,
        /* key = */ vec![1, 2],
        &Tensor::from_slice(&[10i64, 11]),
        /* priority = */ 0,
        /* extra_key = */ None,
    );
    let b = tc.add_new_node_(
        a,
        /* key = */ vec![3],
        &Tensor::from_slice(&[12i64]),
        /* priority = */ 0,
        /* extra_key = */ None,
    );
    let c = tc.add_new_node_(
        b,
        /* key = */ vec![4],
        &Tensor::from_slice(&[13i64]),
        /* priority = */ 0,
        /* extra_key = */ None,
    );
    let taken = tc.arena.take_device_value(b, FULL);
    tc.arena.set_host_value(b, FULL, taken);
    set_swa_device_and_list(&mut tc, a);
    set_swa_device_and_list(&mut tc, c);
    let result = tc.match_prefix(&match_params(&vec![1, 2, 3, 4]));
    assert_eq!(result.best_match_node_id, tc.arena.node(a).id);
    assert_eq!(result.last_device_node_id, tc.arena.node(a).id);
    assert!(
        result
            .device_indices
            .equal(&Tensor::from_slice(&[10i64, 11]))
    );
}

#[test]
fn match_prefix_swa_tombstone_holds_the_best_match_below_the_window() {
    // a(SWA on) -> t(Full on, SWA tombstone) -> c(SWA on, span 1 < window):
    // the best match stays at a even though Full accepts the deeper nodes.
    let mut tc = swa_match_core(/* window = */ 2);
    let root = tc.arena.root();
    let a = tc.add_new_node_(
        root,
        /* key = */ vec![1, 2],
        &Tensor::from_slice(&[10i64, 11]),
        /* priority = */ 0,
        /* extra_key = */ None,
    );
    let t = tc.add_new_node_(
        a,
        /* key = */ vec![3],
        &Tensor::from_slice(&[12i64]),
        /* priority = */ 0,
        /* extra_key = */ None,
    );
    let c = tc.add_new_node_(
        t,
        /* key = */ vec![4],
        &Tensor::from_slice(&[13i64]),
        /* priority = */ 0,
        /* extra_key = */ None,
    );
    set_swa_device_and_list(&mut tc, a);
    set_swa_device_and_list(&mut tc, c);
    let result = tc.match_prefix(&match_params(&vec![1, 2, 3, 4]));
    assert_eq!(result.best_match_node_id, tc.arena.node(a).id);
    assert!(
        result
            .device_indices
            .equal(&Tensor::from_slice(&[10i64, 11]))
    );
}

#[test]
fn match_prefix_swa_best_match_advances_at_the_window() {
    // Same shape, but c spans the whole window: c revalidates and takes
    // the best match past the SWA tombstone.
    let mut tc = swa_match_core(/* window = */ 2);
    let root = tc.arena.root();
    let a = tc.add_new_node_(
        root,
        /* key = */ vec![1, 2],
        &Tensor::from_slice(&[10i64, 11]),
        /* priority = */ 0,
        /* extra_key = */ None,
    );
    let t = tc.add_new_node_(
        a,
        /* key = */ vec![3],
        &Tensor::from_slice(&[12i64]),
        /* priority = */ 0,
        /* extra_key = */ None,
    );
    let c = tc.add_new_node_(
        t,
        /* key = */ vec![4, 5],
        &Tensor::from_slice(&[13i64, 14]),
        /* priority = */ 0,
        /* extra_key = */ None,
    );
    set_swa_device_and_list(&mut tc, a);
    set_swa_device_and_list(&mut tc, c);
    let result = tc.match_prefix(&match_params(&vec![1, 2, 3, 4, 5]));
    assert_eq!(result.best_match_node_id, tc.arena.node(c).id);
    assert!(
        result
            .device_indices
            .equal(&Tensor::from_slice(&[10i64, 11, 12, 13, 14]))
    );
}

#[test]
fn match_end_refresh_moves_the_swa_window_run() {
    let mut tc = swa_match_core(/* window = */ 2);
    let root = tc.arena.root();
    let a = tc.add_new_node_(
        root,
        /* key = */ vec![1, 2],
        &Tensor::from_slice(&[10i64, 11]),
        /* priority = */ 0,
        /* extra_key = */ None,
    );
    let b = tc.add_new_node_(
        a,
        /* key = */ vec![3],
        &Tensor::from_slice(&[12i64]),
        /* priority = */ 0,
        /* extra_key = */ None,
    );
    let c = tc.add_new_node_(
        b,
        /* key = */ vec![4],
        &Tensor::from_slice(&[13i64]),
        /* priority = */ 0,
        /* extra_key = */ None,
    );
    let s = tc.add_new_node_(
        root,
        /* key = */ vec![9],
        &Tensor::from_slice(&[19i64]),
        /* priority = */ 0,
        /* extra_key = */ None,
    );
    for node in [c, b, a, s] {
        set_swa_device_and_list(&mut tc, node);
    }
    let result = tc.match_prefix(&match_params(&vec![1, 2, 3, 4]));
    assert_eq!(result.best_match_node_id, tc.arena.node(c).id);
    // The walk window is sliding_window_size + page_size = 3: c, b, and
    // the straddling a become the MRU run; the sentinel s stays behind.
    let order: Vec<NodeIdx_> = tc.device_lru_list(SWA).iter().collect();
    assert_eq!(order, vec![c, b, a, s]);
}

#[test]
fn match_prefix_with_hicache_reports_swa_host_hits() {
    // a(SWA device) -> b(Full host-only, SWA host-only): the best match
    // advances onto b and finalize reports b's SWA host span.
    let mut tc = swa_match_core(/* window = */ 4);
    tc.set_hicache_enabled();
    let root = tc.arena.root();
    let a = tc.add_new_node_(
        root,
        /* key = */ vec![1, 2],
        &Tensor::from_slice(&[10i64, 11]),
        /* priority = */ 0,
        /* extra_key = */ None,
    );
    let b = tc.add_new_node_(
        a,
        /* key = */ vec![3],
        &Tensor::from_slice(&[12i64]),
        /* priority = */ 0,
        /* extra_key = */ None,
    );
    let taken = tc.arena.take_device_value(b, FULL);
    tc.arena.set_host_value(b, FULL, taken);
    set_swa_device_and_list(&mut tc, a);
    tc.arena
        .node_mut(b)
        .state_mut_(ValueSlotIdx::host(SWA))
        .value = Some(Tensor::from_slice(&[0i64]));
    let result = tc.match_prefix(&match_params(&vec![1, 2, 3]));
    assert_eq!(result.best_match_node_id, tc.arena.node(b).id);
    assert_eq!(result.last_device_node_id, tc.arena.node(a).id);
    assert_eq!(result.host_hit_length, 1);
    assert_eq!(result.swa_host_hit_length, 1);
}

#[test]
fn touch_node_walkdown_keeps_the_swa_lru_order() {
    let mut tc = swa_match_core(/* window = */ 2);
    let root = tc.arena.root();
    let a = tc.add_new_node_(
        root,
        /* key = */ vec![1],
        &Tensor::from_slice(&[10i64]),
        /* priority = */ 0,
        /* extra_key = */ None,
    );
    let b = tc.add_new_node_(
        root,
        /* key = */ vec![2],
        &Tensor::from_slice(&[11i64]),
        /* priority = */ 0,
        /* extra_key = */ None,
    );
    set_swa_device_and_list(&mut tc, a);
    set_swa_device_and_list(&mut tc, b);
    // The SWA walk-down refresh is a no-op: touching the LRU-tail node
    // must not move it (window-bounded refresh runs at match/insert end).
    tc.touch_node_(a);
    let order: Vec<NodeIdx_> = tc.device_lru_list(SWA).iter().collect();
    assert_eq!(order, vec![b, a]);
}

#[test]
fn repeated_deep_swa_matches_keep_the_tree_sane() {
    let mut tc = swa_match_core(/* window = */ 2);
    let root = tc.arena.root();
    let a = tc.add_new_node_(
        root,
        /* key = */ vec![1, 2],
        &Tensor::from_slice(&[10i64, 11]),
        /* priority = */ 0,
        /* extra_key = */ None,
    );
    let b = tc.add_new_node_(
        a,
        /* key = */ vec![3],
        &Tensor::from_slice(&[12i64]),
        /* priority = */ 0,
        /* extra_key = */ None,
    );
    let c = tc.add_new_node_(
        b,
        /* key = */ vec![4],
        &Tensor::from_slice(&[13i64]),
        /* priority = */ 0,
        /* extra_key = */ None,
    );
    let s = tc.add_new_node_(
        root,
        /* key = */ vec![9],
        &Tensor::from_slice(&[19i64]),
        /* priority = */ 0,
        /* extra_key = */ None,
    );
    // The size-accounted store keeps the SWA bookkeeping sanity-checkable.
    for node in [c, b, a, s] {
        let len = tc.arena.node(node).key.atom_len();
        tc.set_component_device_value(
            tc.arena.node(node).id,
            SWA,
            Tensor::from_slice(&vec![0i64; len]),
        );
    }
    for _ in 0..3 {
        let result = tc.match_prefix(&match_params(&vec![1, 2, 3, 4]));
        assert!(
            result
                .device_indices
                .equal(&Tensor::from_slice(&[10i64, 11, 12, 13]))
        );
        tc.sanity_check(&[], &[]);
    }
}

#[test]
fn swa_host_backed_node_advances_best_match_but_keeps_the_device_anchor() {
    let mut tc = swa_match_core(/* window = */ 2);
    tc.set_hicache_enabled();
    // A wired host SWA pool makes host-only SWA gate the device match again.
    tc.set_has_swa_host_pool();
    let root = tc.arena.root();
    let a = tc.add_new_node_(
        root,
        /* key = */ vec![1, 2],
        &Tensor::from_slice(&[10i64, 11]),
        /* priority = */ 0,
        /* extra_key = */ None,
    );
    let b = tc.add_new_node_(
        a,
        /* key = */ vec![3],
        &Tensor::from_slice(&[12i64]),
        /* priority = */ 0,
        /* extra_key = */ None,
    );
    set_swa_device_and_list(&mut tc, a);
    tc.arena
        .node_mut(b)
        .state_mut_(ValueSlotIdx::host(SWA))
        .value = Some(Tensor::from_slice(&[0i64]));
    tc.host_lru_list_mut(SWA).insert_mru(b);
    let result = tc.match_prefix(&match_params(&vec![1, 2, 3]));
    assert!(
        result
            .device_indices
            .equal(&Tensor::from_slice(&[10i64, 11]))
    );
    assert_eq!(result.last_device_node_id, tc.arena.node(a).id);
    assert_eq!(result.best_match_node_id, tc.arena.node(b).id);
    assert_eq!(result.host_hit_length, 0);
    assert_eq!(result.swa_host_hit_length, 1);
}

fn insert_params<'k>(key: &'k Vec<i64>, value: &[i64]) -> InsertParams<'k, Vec<i64>> {
    InsertParams {
        key,
        extra_key: None,
        value: Tensor::from_slice(value),
        mamba_value: None,
        prev_prefix_len: 0,
        swa_evicted_seqlen: 0,
        chunked: false,
        priority: 0,
    }
}

#[test]
fn insert_creates_a_leaf_and_matches_back() {
    let mut tc = core();
    let result = tc.insert(&insert_params(&vec![1, 2, 3], &[10, 11, 12]));
    assert_eq!(result.prefix_len, 0);
    assert!(!result.mamba_exist);
    assert!(result.cache_actions.is_empty());
    assert_eq!(tc.evictable_size_(FULL), 3);
    let matched = tc.match_prefix(&match_params(&vec![1, 2, 3]));
    assert!(
        matched
            .device_indices
            .equal(&Tensor::from_slice(&[10i64, 11, 12]))
    );
}

#[test]
fn insert_full_overlap_frees_the_duplicates() {
    let mut tc = core();
    tc.insert(&insert_params(&vec![1, 2, 3], &[10, 11, 12]));
    let result = tc.insert(&insert_params(&vec![1, 2, 3], &[20, 21, 22]));
    assert_eq!(result.prefix_len, 3);
    // Nothing was consumed by any component: the whole overlap is duplicate.
    let [CacheAction::FreeDeviceKV(freed)] = result.cache_actions.as_slice() else {
        panic!(
            "expected one FreeDeviceKV action, got {:?}",
            action_kinds(&result.cache_actions)
        );
    };
    assert_eq!(freed.len(), 1);
    assert!(freed[0].equal(&Tensor::from_slice(&[20i64, 21, 22])));
}

#[test]
fn insert_prev_prefix_len_narrows_the_dup_window() {
    let mut tc = core();
    tc.insert(&insert_params(&vec![1, 2, 3], &[10, 11, 12]));
    let result = tc.insert(&InsertParams {
        prev_prefix_len: 2,
        ..insert_params(&vec![1, 2, 3], &[20, 21, 22])
    });
    let [CacheAction::FreeDeviceKV(freed)] = result.cache_actions.as_slice() else {
        panic!(
            "expected one FreeDeviceKV action, got {:?}",
            action_kinds(&result.cache_actions)
        );
    };
    assert!(freed[0].equal(&Tensor::from_slice(&[22i64])));
}

#[test]
fn insert_extends_an_existing_prefix() {
    let mut tc = core();
    tc.insert(&insert_params(&vec![1, 2, 3], &[10, 11, 12]));
    let result = tc.insert(&insert_params(&vec![1, 2, 3, 4, 5], &[20, 21, 22, 13, 14]));
    assert_eq!(result.prefix_len, 3);
    // The existing node's overlap is duplicate; the new suffix is kept.
    let [CacheAction::FreeDeviceKV(freed)] = result.cache_actions.as_slice() else {
        panic!(
            "expected one FreeDeviceKV action, got {:?}",
            action_kinds(&result.cache_actions)
        );
    };
    assert_eq!(freed.len(), 1);
    assert!(freed[0].equal(&Tensor::from_slice(&[20i64, 21, 22])));
    let matched = tc.match_prefix(&match_params(&vec![1, 2, 3, 4, 5]));
    assert!(
        matched
            .device_indices
            .equal(&Tensor::from_slice(&[10i64, 11, 12, 13, 14]))
    );
}

#[test]
fn insert_prev_prefix_len_spans_a_multi_node_walk() {
    let mut tc = core();
    tc.insert(&insert_params(&vec![1, 2, 3], &[10, 11, 12]));
    tc.insert(&insert_params(&vec![1, 2, 3, 4, 5], &[20, 21, 22, 13, 14]));
    // The request already matched [1,2,3]: only the second node's overlap
    // is duplicate.
    let result = tc.insert(&InsertParams {
        prev_prefix_len: 3,
        ..insert_params(&vec![1, 2, 3, 4, 5], &[30, 31, 32, 33, 34])
    });
    assert_eq!(result.prefix_len, 5);
    let [CacheAction::FreeDeviceKV(freed)] = result.cache_actions.as_slice() else {
        panic!(
            "expected one FreeDeviceKV action, got {:?}",
            action_kinds(&result.cache_actions)
        );
    };
    assert_eq!(freed.len(), 1);
    assert!(freed[0].equal(&Tensor::from_slice(&[33i64, 34])));
}

#[test]
fn insert_prev_prefix_len_narrows_mid_node_on_a_multi_node_walk() {
    let mut tc = core();
    tc.insert(&insert_params(&vec![1, 2, 3], &[10, 11, 12]));
    tc.insert(&insert_params(&vec![1, 2, 3, 4, 5], &[20, 21, 22, 13, 14]));
    // prev_prefix_len 4 lands mid second node: only its last token is duplicate.
    let result = tc.insert(&InsertParams {
        prev_prefix_len: 4,
        ..insert_params(&vec![1, 2, 3, 4, 5], &[30, 31, 32, 33, 34])
    });
    assert_eq!(result.prefix_len, 5);
    let [CacheAction::FreeDeviceKV(freed)] = result.cache_actions.as_slice() else {
        panic!(
            "expected one FreeDeviceKV action, got {:?}",
            action_kinds(&result.cache_actions)
        );
    };
    assert_eq!(freed.len(), 1);
    assert!(freed[0].equal(&Tensor::from_slice(&[34i64])));
}

#[test]
fn insert_splits_on_a_partial_overlap() {
    let mut tc = core();
    tc.insert(&insert_params(&vec![1, 2, 3], &[10, 11, 12]));
    let result = tc.insert(&insert_params(&vec![1, 2, 9], &[20, 21, 29]));
    assert_eq!(result.prefix_len, 2);
    // Both suffixes live under the split prefix node.
    let matched = tc.match_prefix(&match_params(&vec![1, 2, 9]));
    assert!(
        matched
            .device_indices
            .equal(&Tensor::from_slice(&[10i64, 11, 29]))
    );
    let matched = tc.match_prefix(&match_params(&vec![1, 2, 3]));
    assert!(
        matched
            .device_indices
            .equal(&Tensor::from_slice(&[10i64, 11, 12]))
    );
}

#[test]
fn insert_unevicts_a_tombstoned_node() {
    let mut tc = core();
    tc.insert(&insert_params(&vec![1, 2], &[10, 11]));
    let a = tc
        .match_prefix(&match_params(&vec![1, 2]))
        .best_match_node_id;
    let _ = tc.arena.take_device_value(tc.arena.resolve(a), FULL);
    tc.component_state_mut(FULL).evictable_size = 0;
    tc.evictable_device_leaves.discard(tc.arena.resolve(a));
    let result = tc.insert(&insert_params(&vec![1, 2], &[20, 21]));
    assert_eq!(result.prefix_len, 2);
    // The fresh KV revives the node; nothing is duplicate.
    assert!(result.cache_actions.is_empty());
    assert!(
        tc.arena
            .device_value(tc.arena.resolve(a), FULL)
            .equal(&Tensor::from_slice(&[20i64, 21]))
    );
    assert_eq!(tc.evictable_size_(FULL), 2);
    assert!(tc.evictable_device_leaves.contains(tc.arena.resolve(a)));
}

#[test]
fn insert_priority_floor_applies_along_the_path() {
    let mut tc = core();
    tc.insert(&insert_params(&vec![1, 2], &[10, 11]));
    let a = tc
        .match_prefix(&match_params(&vec![1, 2]))
        .best_match_node_id;
    tc.insert(&InsertParams {
        priority: 5,
        ..insert_params(&vec![1, 2], &[20, 21])
    });
    assert_eq!(tc.arena.node(tc.arena.resolve(a)).priority, 5);
}

#[test]
fn insert_chunked_skips_the_hit_count() {
    let mut tc = core();
    tc.insert(&insert_params(&vec![1, 2], &[10, 11]));
    let a = tc
        .match_prefix(&match_params(&vec![1, 2]))
        .best_match_node_id;
    let hits_before = tc.arena.node(tc.arena.resolve(a)).hit_count;
    tc.insert(&InsertParams {
        chunked: true,
        ..insert_params(&vec![1, 2], &[20, 21])
    });
    assert_eq!(tc.arena.node(tc.arena.resolve(a)).hit_count, hits_before);
}

#[test]
fn insert_extension_bumps_the_traversed_node_hit_count_once() {
    let mut tc = core();
    tc.insert(&insert_params(&vec![1, 2, 3], &[10, 11, 12]));
    let a = tc
        .match_prefix(&match_params(&vec![1, 2, 3]))
        .best_match_node_id;
    let hits_before = tc.arena.node(tc.arena.resolve(a)).hit_count;
    tc.insert(&insert_params(&vec![1, 2, 3, 4, 5], &[20, 21, 22, 13, 14]));
    assert_eq!(
        tc.arena.node(tc.arena.resolve(a)).hit_count,
        hits_before + 1
    );
}

#[test]
fn insert_full_overlap_bumps_the_hit_count_once() {
    let mut tc = core();
    tc.insert(&insert_params(&vec![1, 2, 3], &[10, 11, 12]));
    let a = tc
        .match_prefix(&match_params(&vec![1, 2, 3]))
        .best_match_node_id;
    let hits_before = tc.arena.node(tc.arena.resolve(a)).hit_count;
    // The walk already counted the full overlap; the target is no new leaf.
    tc.insert(&insert_params(&vec![1, 2, 3], &[20, 21, 22]));
    assert_eq!(
        tc.arena.node(tc.arena.resolve(a)).hit_count,
        hits_before + 1
    );
}

#[test]
fn insert_threshold_crossing_emits_the_backup_kv_action() {
    let params = CacheInitParams {
        write_through_threshold: 1,
        ..Default::default()
    };
    let mut tc: UnifiedTreeCore<Vec<i64>> = UnifiedTreeCore::new(params, vec![FULL]);
    tc.set_hicache_enabled();
    let result = tc.insert(&insert_params(&vec![1, 2, 3], &[10, 11, 12]));
    let leaf = tc
        .match_prefix(&match_params(&vec![1, 2, 3]))
        .best_match_node_id;
    let backups: Vec<_> = result
        .cache_actions
        .iter()
        .filter_map(|action| match action {
            CacheAction::BackupKV(backup) => Some(backup.node_ids.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(backups, vec![vec![leaf]]);
}

#[test]
fn mark_write_through_pending_stamps_the_node_id_as_the_ack() {
    let mut tc = core();
    tc.insert(&insert_params(&vec![1], &[10]));
    let leaf = tc.match_prefix(&match_params(&vec![1])).best_match_node_id;
    tc.mark_write_through_pending(leaf);
    assert_eq!(
        tc.arena
            .node(tc.arena.resolve(leaf))
            .write_through_pending_id,
        Some(leaf)
    );
}

#[test]
fn finish_write_through_clears_only_the_matching_ack() {
    let mut tc = core();
    tc.insert(&insert_params(&vec![1], &[10]));
    let leaf = tc.match_prefix(&match_params(&vec![1])).best_match_node_id;
    tc.mark_write_through_pending(leaf);
    tc.finish_write_through(vec![leaf], /* ack_id = */ 999_999);
    assert_eq!(
        tc.arena
            .node(tc.arena.resolve(leaf))
            .write_through_pending_id,
        Some(leaf)
    );
    tc.finish_write_through(vec![leaf], /* ack_id = */ leaf);
    assert_eq!(
        tc.arena
            .node(tc.arena.resolve(leaf))
            .write_through_pending_id,
        None
    );
}

#[test]
fn backup_kv_action_chains_unbacked_ancestors_first() {
    let mut tc = core();
    tc.insert(&insert_params(&vec![1], &[10]));
    tc.insert(&insert_params(&vec![1, 2], &[10, 11]));
    tc.insert(&insert_params(&vec![1, 2, 3], &[10, 11, 12]));
    let a = tc.match_prefix(&match_params(&vec![1])).best_match_node_id;
    let b = tc
        .match_prefix(&match_params(&vec![1, 2]))
        .best_match_node_id;
    let c = tc
        .match_prefix(&match_params(&vec![1, 2, 3]))
        .best_match_node_id;
    // a is backuped: the chain stops there and orders ancestors first.
    tc.arena
        .set_host_value(tc.arena.resolve(a), FULL, Tensor::from_slice(&[20i64]));
    let action = tc.build_backup_kv_action_(
        tc.arena.node(tc.arena.resolve(c)),
        /* write_back = */ false,
    );
    assert_eq!(action.node_ids, vec![b, c]);
    let action = tc.build_backup_kv_action_(
        tc.arena.node(tc.arena.resolve(c)),
        /* write_back = */ true,
    );
    assert_eq!(action.node_ids, vec![c]);
}

#[test]
fn split_of_a_pending_node_transfers_the_ack_and_emits_the_replace_action() {
    let mut tc = core();
    tc.insert(&insert_params(&vec![1, 2, 3], &[10, 11, 12]));
    let node = tc
        .match_prefix(&match_params(&vec![1, 2, 3]))
        .best_match_node_id;
    tc.mark_write_through_pending(node);
    let (new_node, action) = tc.split_node_(tc.arena.resolve(node), /* split_len = */ 1);
    assert_eq!(tc.arena.node(new_node).write_through_pending_id, Some(node));
    assert_eq!(
        tc.arena
            .node(tc.arena.resolve(node))
            .write_through_pending_id,
        Some(node)
    );
    match action {
        Some(CacheAction::ReplaceWriteThroughOnNodeSplit {
            ack_id,
            old_node_id,
            new_node_id,
            new_child_node_id,
        }) => {
            assert_eq!(ack_id, node);
            assert_eq!(old_node_id, node);
            assert_eq!(new_node_id, tc.arena.node(new_node).id);
            assert_eq!(new_child_node_id, node);
        }
        other => panic!("expected the replace action, got {:?}", other.is_some()),
    }
}

#[test]
fn insert_does_not_hash_without_storage() {
    let mut tc = UnifiedTreeCore::new(
        CacheInitParams {
            page_size: 2,
            ..CacheInitParams::default()
        },
        vec![FULL],
    );
    tc.insert(&insert_params(&vec![1, 2], &[10, 11]));
    let leaf = tc
        .match_prefix(&match_params(&vec![1, 2]))
        .best_match_node_id;
    assert_eq!(tc.arena.node(tc.arena.resolve(leaf)).hash_value, None);
}

#[test]
fn insert_hashes_pages_chained_from_the_parent_when_storage_is_on() {
    // Expected values are literals produced by the python native hash.
    let mut tc = UnifiedTreeCore::new(
        CacheInitParams {
            page_size: 2,
            ..CacheInitParams::default()
        },
        vec![FULL],
    );
    tc.set_enable_storage(true);
    tc.insert(&insert_params(&vec![1, 2], &[10, 11]));
    tc.insert(&insert_params(&vec![1, 2, 7, 8], &[10, 11, 12, 13]));
    let parent = tc
        .match_prefix(&match_params(&vec![1, 2]))
        .best_match_node_id;
    assert_eq!(
        tc.arena.node(tc.arena.resolve(parent)).hash_value,
        Some(vec![
            "34fb5c825de7ca4aea6e712f19d439c1da0c92c37b423936c5f618545ca4fa1f".to_string()
        ])
    );
    let child = tc
        .match_prefix(&match_params(&vec![1, 2, 7, 8]))
        .best_match_node_id;
    assert_eq!(
        tc.arena.node(tc.arena.resolve(child)).hash_value,
        Some(vec![
            "0bfa9b9c6fd727c7410b6d42b753439911022d34cc6ef99ac43ed7724aa48a75".to_string()
        ])
    );
    // The prefix walk concatenates the chain in root-to-node order.
    assert_eq!(
        tc.arena.prefix_hash_values(Some(tc.arena.resolve(child))),
        vec![
            "34fb5c825de7ca4aea6e712f19d439c1da0c92c37b423936c5f618545ca4fa1f".to_string(),
            "0bfa9b9c6fd727c7410b6d42b753439911022d34cc6ef99ac43ed7724aa48a75".to_string(),
        ]
    );
    assert_eq!(
        tc.arena.prefix_hash_values(Some(tc.arena.resolve(parent))),
        vec!["34fb5c825de7ca4aea6e712f19d439c1da0c92c37b423936c5f618545ca4fa1f".to_string()]
    );
}

fn events_core(page_size: usize) -> UnifiedTreeCore<Vec<i64>> {
    UnifiedTreeCore::new(
        CacheInitParams {
            page_size,
            enable_kv_cache_events: true,
            ..CacheInitParams::default()
        },
        vec![FULL],
    )
}

#[test]
fn take_events_is_empty_when_events_are_disabled() {
    let mut tc = core();
    tc.insert(&insert_params(&vec![1, 2], &[10, 11]));
    assert_eq!(tc.take_events(), Vec::new());
}

#[test]
fn insert_emits_one_block_stored_per_page_with_the_chained_parent() {
    let mut tc = events_core(2);
    tc.insert(&insert_params(&vec![1, 2, 7, 8], &[10, 11, 12, 13]));
    let hashes = crate::node::get_hash_str::<Vec<i64>>(&[1, 2, 7, 8], None, 2);
    assert_eq!(
        tc.take_events(),
        vec![
            KvCacheEvent::BlockStored {
                block_hash: crate::node::hash_str_to_int64(&hashes[0]),
                parent_block_hash: None,
                token_ids: vec![1, 2],
                medium: StorageMedium::Gpu,
            },
            KvCacheEvent::BlockStored {
                block_hash: crate::node::hash_str_to_int64(&hashes[1]),
                parent_block_hash: Some(crate::node::hash_str_to_int64(&hashes[0])),
                token_ids: vec![7, 8],
                medium: StorageMedium::Gpu,
            },
        ]
    );
    // Events hash lazily even though the storage tier is off.
    let leaf = tc
        .match_prefix(&match_params(&vec![1, 2, 7, 8]))
        .best_match_node_id;
    assert_eq!(
        tc.arena.node(tc.arena.resolve(leaf)).hash_value,
        Some(hashes)
    );
}

#[test]
fn eviction_emits_block_removed_with_all_page_hashes() {
    let mut tc = events_core(2);
    tc.insert(&insert_params(&vec![1, 2, 7, 8], &[10, 11, 12, 13]));
    let _ = tc.take_events();
    let mut tracker = HashMap::from([(FULL, 0)]);
    let (mut device_frees, mut host_frees) = (HashMap::new(), HashMap::new());
    tc.evict_device_start(FULL, 100);
    loop {
        let (node, step) = tc.evict_device_next_node(FULL, &tracker);
        accumulate_step(step, &mut tracker, &mut device_frees, &mut host_frees);
        let Some(node) = node else { break };
        let (_, step) = tc.evict_device_leaf(node, /* is_write_back = */ false);
        accumulate_step(step, &mut tracker, &mut device_frees, &mut host_frees);
    }
    tc.evict_device_end(FULL);
    let hashes = crate::node::get_hash_str::<Vec<i64>>(&[1, 2, 7, 8], None, 2);
    assert_eq!(
        tc.take_events(),
        vec![KvCacheEvent::BlockRemoved {
            block_hashes: hashes
                .iter()
                .map(|h| crate::node::hash_str_to_int64(h))
                .collect(),
            medium: StorageMedium::Gpu,
        }]
    );
}

#[test]
fn bigram_insert_events_carry_pair_token_payloads() {
    let mut tc = UnifiedTreeCore::<Vec<(i64, i64)>>::new(
        CacheInitParams {
            enable_kv_cache_events: true,
            ..CacheInitParams::default()
        },
        vec![FULL],
    );
    let key: Vec<(i64, i64)> = vec![(1, 2), (2, 3)];
    tc.insert(&InsertParams {
        key: &key,
        extra_key: None,
        value: Tensor::from_slice(&[10i64, 11]),
        mamba_value: None,
        prev_prefix_len: 0,
        swa_evicted_seqlen: 0,
        chunked: false,
        priority: 0,
    });
    let hashes = crate::node::get_hash_str::<Vec<(i64, i64)>>(&key, None, 1);
    assert_eq!(
        tc.take_events(),
        vec![
            KvCacheEvent::BlockStored {
                block_hash: crate::node::hash_str_to_int64(&hashes[0]),
                parent_block_hash: None,
                token_ids: vec![(1, 2)],
                medium: StorageMedium::Gpu,
            },
            KvCacheEvent::BlockStored {
                block_hash: crate::node::hash_str_to_int64(&hashes[1]),
                parent_block_hash: Some(crate::node::hash_str_to_int64(&hashes[0])),
                token_ids: vec![(2, 3)],
                medium: StorageMedium::Gpu,
            },
        ]
    );
}

#[test]
fn finish_write_through_emits_cpu_stored_events() {
    let mut tc = events_core(1);
    tc.set_hicache_enabled();
    tc.insert(&insert_params(&vec![1], &[10]));
    let leaf = tc.match_prefix(&match_params(&vec![1])).best_match_node_id;
    let _ = tc.take_events();
    tc.finish_write_through(vec![leaf], leaf);
    let hashes = crate::node::get_hash_str::<Vec<i64>>(&[1], None, 1);
    assert_eq!(
        tc.take_events(),
        vec![KvCacheEvent::BlockStored {
            block_hash: crate::node::hash_str_to_int64(&hashes[0]),
            parent_block_hash: None,
            token_ids: vec![1],
            medium: StorageMedium::Cpu,
        }]
    );
}

// A demoted (host-only) single-token leaf with the event queue drained.
fn demoted_events_leaf(tc: &mut UnifiedTreeCore<Vec<i64>>) -> NodeIdx_ {
    tc.set_hicache_enabled();
    tc.insert(&insert_params(&vec![1], &[10]));
    let leaf = tc.match_prefix(&match_params(&vec![1])).best_match_node_id;
    tc.commit_backup(leaf, Tensor::from_slice(&[100i64]), HashMap::new());
    tc.demote(leaf);
    let _ = tc.take_events();
    tc.arena.resolve(leaf)
}

#[test]
fn host_eviction_emits_a_cpu_block_removed() {
    let mut tc = events_core(1);
    demoted_events_leaf(&mut tc);
    tc.drive_host_eviction(FULL, 1);
    let hashes = crate::node::get_hash_str::<Vec<i64>>(&[1], None, 1);
    assert_eq!(
        tc.take_events(),
        vec![KvCacheEvent::BlockRemoved {
            block_hashes: vec![crate::node::hash_str_to_int64(&hashes[0])],
            medium: StorageMedium::Cpu,
        }]
    );
}

#[test]
fn load_back_commit_emits_gpu_stored_events() {
    let mut tc = events_core(1);
    let leaf = demoted_events_leaf(&mut tc);
    let (kv_xfer, comp_xfers) = tc.build_load_back_spec(tc.arena.node(leaf).id, None);
    tc.commit_load_back(
        tc.arena.node(leaf).id,
        Tensor::from_slice(&[50i64]),
        kv_xfer,
        comp_xfers,
    );
    let hashes = crate::node::get_hash_str::<Vec<i64>>(&[1], None, 1);
    assert_eq!(
        tc.take_events(),
        vec![KvCacheEvent::BlockStored {
            block_hash: crate::node::hash_str_to_int64(&hashes[0]),
            parent_block_hash: None,
            token_ids: vec![1],
            medium: StorageMedium::Gpu,
        }]
    );
}

#[test]
fn unevict_on_insert_emits_a_gpu_stored_event() {
    let mut tc = events_core(1);
    demoted_events_leaf(&mut tc);
    tc.insert(&insert_params(&vec![1], &[60]));
    let hashes = crate::node::get_hash_str::<Vec<i64>>(&[1], None, 1);
    assert_eq!(
        tc.take_events(),
        vec![KvCacheEvent::BlockStored {
            block_hash: crate::node::hash_str_to_int64(&hashes[0]),
            parent_block_hash: None,
            token_ids: vec![1],
            medium: StorageMedium::Gpu,
        }]
    );
}

#[test]
fn drop_subtree_emits_removals_for_host_descendants_then_the_leaf() {
    let mut tc = UnifiedTreeCore::new(
        CacheInitParams {
            is_write_back: true,
            enable_hicache: true,
            enable_kv_cache_events: true,
            ..CacheInitParams::default()
        },
        vec![FULL],
    );
    tc.insert(&insert_params(&vec![1, 2], &[10, 11]));
    let leaf = tc
        .match_prefix(&match_params(&vec![1, 2]))
        .best_match_node_id;
    let child = tc
        .arena
        .alloc_child(
            tc.arena.resolve(leaf),
            /* key = */ vec![3, 4],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    tc.arena
        .set_host_value(child, FULL, Tensor::from_slice(&[20i64, 21]));
    tc.update_evictable_leaf_sets_(child);
    tc.update_evictable_leaf_sets_(tc.arena.resolve(leaf));
    let _ = tc.take_events();
    let (dropped, _step) = tc.drop_subtree_no_host(leaf);
    assert!(dropped);
    // The leaf hashed lazily at its insert store event; the host-only
    // child hashes lazily at removal, chaining from the leaf.
    let leaf_hashes = crate::node::get_hash_str::<Vec<i64>>(&[1, 2], None, 1);
    let child_hashes =
        crate::node::get_hash_str::<Vec<i64>>(&[3, 4], leaf_hashes.last().map(String::as_str), 1);
    assert_eq!(
        tc.take_events(),
        vec![
            KvCacheEvent::BlockRemoved {
                block_hashes: child_hashes
                    .iter()
                    .map(|h| crate::node::hash_str_to_int64(h))
                    .collect(),
                medium: StorageMedium::Cpu,
            },
            KvCacheEvent::BlockRemoved {
                block_hashes: leaf_hashes
                    .iter()
                    .map(|h| crate::node::hash_str_to_int64(h))
                    .collect(),
                medium: StorageMedium::Gpu,
            },
        ]
    );
}

#[test]
fn all_cleared_event_queues_and_take_drains() {
    let mut tc = events_core(1);
    tc.record_all_cleared_event();
    assert_eq!(tc.take_events(), vec![KvCacheEvent::AllBlocksCleared]);
    assert_eq!(tc.take_events(), Vec::new());
}

#[test]
fn split_insert_stores_only_the_new_block_chained_to_the_split_parent() {
    let mut tc = events_core(2);
    tc.insert(&insert_params(&vec![1, 2, 3, 4], &[10, 11, 12, 13]));
    let _ = tc.take_events();
    tc.insert(&insert_params(&vec![1, 2, 5, 6], &[20, 21, 22, 23]));
    let base_hashes = crate::node::get_hash_str::<Vec<i64>>(&[1, 2, 3, 4], None, 2);
    let leaf_hashes =
        crate::node::get_hash_str::<Vec<i64>>(&[5, 6], Some(base_hashes[0].as_str()), 2);
    // Only the diverging suffix is stored; the matched prefix is not re-published.
    assert_eq!(
        tc.take_events(),
        vec![KvCacheEvent::BlockStored {
            block_hash: crate::node::hash_str_to_int64(&leaf_hashes[0]),
            parent_block_hash: Some(crate::node::hash_str_to_int64(&base_hashes[0])),
            token_ids: vec![5, 6],
            medium: StorageMedium::Gpu,
        }]
    );
    // The split divided the page hashes between the two fragments.
    let parent = tc
        .match_prefix(&match_params(&vec![1, 2]))
        .best_match_node_id;
    assert_eq!(
        tc.arena.node(tc.arena.resolve(parent)).hash_value,
        Some(vec![base_hashes[0].clone()])
    );
    let child = tc
        .match_prefix(&match_params(&vec![1, 2, 3, 4]))
        .best_match_node_id;
    assert_eq!(
        tc.arena.node(tc.arena.resolve(child)).hash_value,
        Some(vec![base_hashes[1].clone()])
    );
}

#[test]
fn finish_write_through_after_a_split_publishes_both_fragments() {
    let mut tc = events_core(2);
    tc.set_hicache_enabled();
    tc.insert(&insert_params(&vec![1, 2, 3, 4], &[10, 11, 12, 13]));
    let leaf = tc
        .match_prefix(&match_params(&vec![1, 2, 3, 4]))
        .best_match_node_id;
    tc.mark_write_through_pending(leaf);
    let _ = tc.take_events();
    let result = tc.insert(&insert_params(&vec![1, 2, 5, 6], &[20, 21, 22, 23]));
    let new_node_id = result
        .cache_actions
        .iter()
        .find_map(|action| match action {
            CacheAction::ReplaceWriteThroughOnNodeSplit {
                ack_id,
                new_node_id,
                ..
            } => {
                assert_eq!(*ack_id, leaf);
                Some(*new_node_id)
            }
            _ => None,
        })
        .expect("the split relocates the pending write-through");
    // Nothing reaches the host tier before the ack.
    assert!(tc.take_events().iter().all(|event| !matches!(
        event,
        KvCacheEvent::BlockStored {
            medium: StorageMedium::Cpu,
            ..
        }
    )));
    tc.commit_backup(
        new_node_id,
        Tensor::from_slice(&[100i64, 101]),
        HashMap::new(),
    );
    tc.commit_backup(leaf, Tensor::from_slice(&[102i64, 103]), HashMap::new());
    tc.finish_write_through(vec![new_node_id, leaf], /* ack_id = */ leaf);
    let hashes = crate::node::get_hash_str::<Vec<i64>>(&[1, 2, 3, 4], None, 2);
    assert_eq!(
        tc.take_events(),
        vec![
            KvCacheEvent::BlockStored {
                block_hash: crate::node::hash_str_to_int64(&hashes[0]),
                parent_block_hash: None,
                token_ids: vec![1, 2],
                medium: StorageMedium::Cpu,
            },
            KvCacheEvent::BlockStored {
                block_hash: crate::node::hash_str_to_int64(&hashes[1]),
                parent_block_hash: Some(crate::node::hash_str_to_int64(&hashes[0])),
                token_ids: vec![3, 4],
                medium: StorageMedium::Cpu,
            },
        ]
    );
    // The matching ack cleared the pending mark on both fragments.
    assert_eq!(
        tc.arena
            .node(tc.arena.resolve(new_node_id))
            .write_through_pending_id,
        None
    );
    assert_eq!(
        tc.arena
            .node(tc.arena.resolve(leaf))
            .write_through_pending_id,
        None
    );
}

#[test]
fn prefetch_anchor_info_maps_the_namespace() {
    let mut tc = core();
    tc.insert(&insert_params(&vec![1, 2], &[10, 11]));
    tc.insert(&InsertParams {
        extra_key: Some("chat"),
        ..insert_params(&vec![7, 8], &[20, 21])
    });
    let plain = tc
        .match_prefix(&match_params(&vec![1, 2]))
        .best_match_node_id;
    assert_eq!(tc.prefetch_anchor_info(plain), None);
    let salted = tc
        .match_prefix(&MatchPrefixParams {
            key: &vec![7, 8],
            extra_key: Some("chat"),
        })
        .best_match_node_id;
    assert_eq!(tc.prefetch_anchor_info(salted), Some("chat".to_string()));
    // A root anchor carries no namespace: the single root serves them all.
    let root = tc.arena.root();
    assert_eq!(tc.prefetch_anchor_info(tc.arena.node(root).id), None);
    // A node minted by a split inherits the namespace.
    tc.insert(&InsertParams {
        extra_key: Some("chat"),
        ..insert_params(&vec![7], &[30])
    });
    let split_mid = tc
        .match_prefix(&MatchPrefixParams {
            key: &vec![7],
            extra_key: Some("chat"),
        })
        .best_match_node_id;
    assert_ne!(split_mid, salted);
    assert_eq!(tc.prefetch_anchor_info(split_mid), Some("chat".to_string()));
}

#[test]
fn mamba_core_constructs_through_the_factory() {
    let tc = UnifiedTreeCore::<Vec<i64>>::new(
        CacheInitParams {
            mamba_cache_chunk_size: Some(256),
            ..CacheInitParams::default()
        },
        vec![FULL, MAMBA],
    );
    assert_eq!(tc.components.len(), 2);
}

#[test]
fn mamba_sizes_and_flatten_read_the_component_state() {
    let mut tc = UnifiedTreeCore::<Vec<i64>>::new(
        CacheInitParams {
            mamba_cache_chunk_size: Some(256),
            ..CacheInitParams::default()
        },
        vec![FULL, MAMBA],
    );
    assert_eq!(tc.mamba_evictable_size(), 0);
    assert_eq!(tc.mamba_protected_size(), 0);
    assert_eq!(tc.all_mamba_values_flatten().numel(), 0);
    let root = tc.arena.root();
    let a = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![1],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    let b = tc
        .arena
        .alloc_child(
            a,
            /* key = */ vec![2],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    tc.arena
        .set_device_value(a, MAMBA, Tensor::from_slice(&[7i64]));
    tc.arena
        .set_device_value(b, MAMBA, Tensor::from_slice(&[9i64]));
    let mut slots = Vec::<i64>::try_from(tc.all_mamba_values_flatten()).unwrap();
    slots.sort_unstable();
    assert_eq!(slots, vec![7, 9]);
    // A Full-only tree reports empty mamba state.
    let full_only = core();
    assert_eq!(full_only.mamba_evictable_size(), 0);
    assert_eq!(full_only.all_mamba_values_flatten().numel(), 0);
}

#[test]
fn prefetch_node_accessors_cover_gate_and_hash_chain() {
    let mut tc = UnifiedTreeCore::new(
        CacheInitParams {
            page_size: 2,
            enable_hicache: true,
            ..CacheInitParams::default()
        },
        vec![FULL],
    );
    tc.set_enable_storage(true);
    tc.insert(&insert_params(&vec![1, 2], &[10, 11]));
    tc.insert(&insert_params(&vec![1, 2, 7, 8], &[10, 11, 12, 13]));
    let child = tc
        .match_prefix(&match_params(&vec![1, 2, 7, 8]))
        .best_match_node_id;

    assert!(!tc.node_backuped(child));
    assert!(!tc.is_root(child));
    assert_eq!(
        tc.get_last_hash_value(child).as_deref(),
        Some("0bfa9b9c6fd727c7410b6d42b753439911022d34cc6ef99ac43ed7724aa48a75")
    );
    assert_eq!(
        tc.get_prefix_hash_values(child),
        vec!["34fb5c825de7ca4aea6e712f19d439c1da0c92c37b423936c5f618545ca4fa1f".to_string()]
    );

    tc.commit_backup(child, Tensor::from_slice(&[102i64, 103]), HashMap::new());
    assert!(tc.node_backuped(child));

    // Roots have no hashes of their own.
    let root = tc.arena.root();
    assert!(tc.is_root(tc.arena.node(root).id));
    assert_eq!(tc.get_last_hash_value(tc.arena.node(root).id), None);
    assert_eq!(
        tc.get_prefix_hash_values(tc.arena.node(root).id),
        Vec::<String>::new()
    );
}

#[test]
fn storage_backup_spec_is_none_for_an_unbackuped_node() {
    let mut tc = core();
    tc.insert(&insert_params(&vec![1, 2], &[10, 11]));
    let leaf = tc
        .match_prefix(&match_params(&vec![1, 2]))
        .best_match_node_id;
    assert!(
        tc.build_storage_backup_spec(tc.arena.node(tc.arena.resolve(leaf)).id, true)
            .is_none()
    );
}

#[test]
fn storage_backup_spec_gathers_the_chained_node() {
    let mut tc = UnifiedTreeCore::new(
        CacheInitParams {
            page_size: 2,
            enable_hicache: true,
            ..CacheInitParams::default()
        },
        vec![FULL],
    );
    tc.set_enable_storage(true);
    tc.insert(&insert_params(&vec![1, 2], &[10, 11]));
    tc.insert(&insert_params(&vec![1, 2, 7, 8], &[10, 11, 12, 13]));
    let parent = tc
        .match_prefix(&match_params(&vec![1, 2]))
        .best_match_node_id;
    let child = tc
        .match_prefix(&match_params(&vec![1, 2, 7, 8]))
        .best_match_node_id;
    tc.commit_backup(parent, Tensor::from_slice(&[100i64, 101]), HashMap::new());
    tc.commit_backup(child, Tensor::from_slice(&[102i64, 103]), HashMap::new());

    let spec = tc
        .build_storage_backup_spec(tc.arena.node(tc.arena.resolve(child)).id, true)
        .unwrap();
    assert!(spec.host_value.equal(&Tensor::from_slice(&[102i64, 103])));
    assert_eq!(spec.token_ids, vec![7, 8]);
    assert_eq!(
        spec.hash_value,
        Some(vec![
            "0bfa9b9c6fd727c7410b6d42b753439911022d34cc6ef99ac43ed7724aa48a75".to_string()
        ])
    );
    assert_eq!(
        spec.prefix_keys,
        Some(vec![
            "34fb5c825de7ca4aea6e712f19d439c1da0c92c37b423936c5f618545ca4fa1f".to_string()
        ])
    );
    assert!(spec.comp_xfers.is_empty());

    let spec = tc
        .build_storage_backup_spec(tc.arena.node(tc.arena.resolve(child)).id, false)
        .unwrap();
    assert_eq!(spec.prefix_keys, None);
}

#[test]
fn prefix_hash_walk_stops_below_an_unhashed_ancestor() {
    let mut tc = core();
    let (a, b) = matched_chain(&mut tc);
    tc.arena.node_mut(b).hash_value = Some(vec!["b0".to_string()]);
    assert_eq!(tc.arena.node(a).hash_value, None);
    assert_eq!(tc.arena.prefix_hash_values(Some(b)), vec!["b0".to_string()]);
    assert_eq!(tc.arena.prefix_hash_values(None), Vec::<String>::new());
}

#[test]
fn insert_host_attaches_a_host_only_leaf_under_the_root() {
    let mut tc = core();
    let root = tc.arena.root();
    let result = tc.insert_host(
        tc.arena.node(root).id,
        /* extra_key = */ None,
        vec![1, 2],
        Tensor::from_slice(&[100i64, 101]),
        vec!["h0".to_string(), "h1".to_string()],
    );
    assert_eq!(result.prefix_len, 0);
    assert_eq!(result.total_len, 2);
    let new_node = result.inserted_host_node.unwrap();
    let node = tc.arena.node(tc.arena.resolve(new_node));
    assert!(node.evicted() && node.backuped());
    assert!(
        node.host_value(FULL)
            .equal(&Tensor::from_slice(&[100i64, 101]))
    );
    assert_eq!(
        node.hash_value,
        Some(vec!["h0".to_string(), "h1".to_string()])
    );
    assert!(
        tc.evictable_host_leaves
            .contains(tc.arena.resolve(new_node))
    );
    tc.sanity_check(&[], &[]);
}

#[test]
fn insert_host_walks_the_match_and_attaches_only_the_suffix() {
    let mut tc = core();
    tc.insert(&insert_params(&vec![1, 2], &[10, 11]));
    let root = tc.arena.root();
    let result = tc.insert_host(
        tc.arena.node(root).id,
        /* extra_key = */ None,
        vec![1, 2, 3, 4],
        Tensor::from_slice(&[100i64, 101, 102, 103]),
        vec!["h0", "h1", "h2", "h3"]
            .into_iter()
            .map(String::from)
            .collect(),
    );
    assert_eq!(result.prefix_len, 2);
    assert_eq!(result.total_len, 4);
    let new_node = tc
        .arena
        .node(tc.arena.resolve(result.inserted_host_node.unwrap()));
    assert!(
        new_node
            .host_value(FULL)
            .equal(&Tensor::from_slice(&[102i64, 103]))
    );
    assert_eq!(
        new_node.hash_value,
        Some(vec!["h2".to_string(), "h3".to_string()])
    );
}

#[test]
fn insert_host_splits_a_host_chain_and_divides_the_hash() {
    let mut tc = core();
    let root = tc.arena.root();
    tc.insert_host(
        tc.arena.node(root).id,
        /* extra_key = */ None,
        vec![1, 2, 3],
        Tensor::from_slice(&[100i64, 101, 102]),
        vec!["h0", "h1", "h2"]
            .into_iter()
            .map(String::from)
            .collect(),
    );
    let result = tc.insert_host(
        tc.arena.node(root).id,
        /* extra_key = */ None,
        vec![1, 9],
        Tensor::from_slice(&[200i64, 201]),
        vec!["g0".to_string(), "g1".to_string()],
    );
    assert_eq!(result.prefix_len, 1);
    let new_node = tc
        .arena
        .node(tc.arena.resolve(result.inserted_host_node.unwrap()));
    assert!(
        new_node
            .host_value(FULL)
            .equal(&Tensor::from_slice(&[201i64]))
    );
    assert_eq!(new_node.hash_value, Some(vec!["g1".to_string()]));
    // The split divided the chain's host value and hash at the boundary.
    let split_parent = new_node.parent();
    let parent = tc.arena.node(split_parent);
    assert!(
        parent
            .host_value(FULL)
            .equal(&Tensor::from_slice(&[100i64]))
    );
    assert_eq!(parent.hash_value, Some(vec!["h0".to_string()]));
    tc.sanity_check(&[], &[]);
}

#[test]
fn insert_host_hash_slices_by_pages_not_atoms() {
    let params = CacheInitParams {
        page_size: 2,
        ..Default::default()
    };
    let mut tc: UnifiedTreeCore<Vec<i64>> = UnifiedTreeCore::new(params, vec![FULL]);
    let root = tc.arena.root();
    tc.insert_host(
        tc.arena.node(root).id,
        /* extra_key = */ None,
        vec![1, 2],
        Tensor::from_slice(&[100i64, 101]),
        vec!["h0".to_string()],
    );
    let result = tc.insert_host(
        tc.arena.node(root).id,
        /* extra_key = */ None,
        vec![1, 2, 3, 4],
        Tensor::from_slice(&[200i64, 201, 202, 203]),
        vec!["g0".to_string(), "g1".to_string()],
    );
    assert_eq!(result.prefix_len, 2);
    let new_node = tc
        .arena
        .node(tc.arena.resolve(result.inserted_host_node.unwrap()));
    // Two matched atoms are ONE page: only g0 is consumed.
    assert_eq!(new_node.hash_value, Some(vec!["g1".to_string()]));
    assert!(
        new_node
            .host_value(FULL)
            .equal(&Tensor::from_slice(&[202i64, 203]))
    );
}

#[test]
fn insert_host_full_match_reports_only_a_backuped_node() {
    let mut tc = core();
    tc.insert(&insert_params(&vec![1, 2], &[10, 11]));
    let leaf = tc
        .match_prefix(&match_params(&vec![1, 2]))
        .best_match_node_id;
    let root = tc.arena.root();
    // The device-only match reports no host node.
    let result = tc.insert_host(
        tc.arena.node(root).id,
        /* extra_key = */ None,
        vec![1, 2],
        Tensor::from_slice(&[100i64, 101]),
        vec!["h0".to_string(), "h1".to_string()],
    );
    assert_eq!(result.prefix_len, 2);
    assert_eq!(result.inserted_host_node, None);
    // Once backuped, the same insert reports the node.
    tc.arena.set_host_value(
        tc.arena.resolve(leaf),
        FULL,
        Tensor::from_slice(&[20i64, 21]),
    );
    let result = tc.insert_host(
        tc.arena.node(root).id,
        /* extra_key = */ None,
        vec![1, 2],
        Tensor::from_slice(&[100i64, 101]),
        vec!["h0".to_string(), "h1".to_string()],
    );
    assert_eq!(result.inserted_host_node, Some(leaf));
}

#[test]
#[ignore]
#[should_panic(expected = "insert_host: parent")]
fn insert_host_panics_on_a_colliding_page() {
    // TODO: unconstructible today — the insert_host walk (like the python one)
    // follows any child on the suffix page instead of breaking at a dead node,
    // so the add path never sees an occupied page; the assert is defensive-only.
    let mut tc = core();
    tc.set_hicache_enabled();
    tc.insert(&insert_params(&vec![1, 2], &[10, 11]));
    let root = tc.arena.root();
    // A host suffix colliding with the device-valued child's first page.
    tc.insert_host(
        tc.arena.node(root).id,
        /* extra_key = */ None,
        vec![1, 9],
        Tensor::from_slice(&[100i64, 101]),
        vec!["h0".to_string(), "h1".to_string()],
    );
}

#[test]
fn insert_host_empty_key_is_a_noop() {
    let mut tc = core();
    let root = tc.arena.root();
    let result = tc.insert_host(
        tc.arena.node(root).id,
        /* extra_key = */ None,
        vec![],
        Tensor::from_slice(&[0i64; 0]),
        vec![],
    );
    assert_eq!(result.prefix_len, 0);
    assert!(result.mamba_exist);
    assert_eq!(result.inserted_host_node, None);
    assert_eq!(tc.arena.len(), 1);
}

#[test]
fn commit_backup_attaches_the_host_value() {
    let mut tc = core();
    tc.insert(&insert_params(&vec![1, 2], &[10, 11]));
    let leaf = tc
        .match_prefix(&match_params(&vec![1, 2]))
        .best_match_node_id;
    tc.commit_backup(leaf, Tensor::from_slice(&[100i64, 101]), HashMap::new());
    let node = tc.arena.node(tc.arena.resolve(leaf));
    assert!(node.backuped());
    assert!(
        node.host_value(FULL)
            .equal(&Tensor::from_slice(&[100i64, 101]))
    );
}

#[test]
fn build_backup_spec_reads_the_device_value() {
    let mut tc = core();
    tc.insert(&insert_params(&vec![1, 2], &[10, 11]));
    let leaf = tc
        .match_prefix(&match_params(&vec![1, 2]))
        .best_match_node_id;
    let (device_value, comp_xfers) = tc.build_backup_spec(leaf);
    assert!(device_value.equal(&Tensor::from_slice(&[10i64, 11])));
    assert!(comp_xfers.is_empty());
}

#[test]
fn build_backup_spec_skips_full_kv_for_an_already_backuped_node() {
    let mut tc = core();
    tc.insert(&insert_params(&vec![1, 2], &[10, 11]));
    let leaf = tc
        .match_prefix(&match_params(&vec![1, 2]))
        .best_match_node_id;

    tc.commit_backup(leaf, Tensor::from_slice(&[100i64, 101]), HashMap::new());
    assert!(tc.arena.node(tc.arena.resolve(leaf)).backuped());

    let (device_value, comp_xfers) = tc.build_backup_spec(leaf);
    assert_eq!(device_value.numel(), 0);
    assert!(comp_xfers.is_empty());
}

#[test]
fn commit_backup_preserves_full_kv_when_host_indices_are_empty() {
    let mut tc = core();
    tc.insert(&insert_params(&vec![1, 2], &[10, 11]));
    let leaf = tc
        .match_prefix(&match_params(&vec![1, 2]))
        .best_match_node_id;

    tc.commit_backup(leaf, Tensor::from_slice(&[100i64, 101]), HashMap::new());
    tc.commit_backup(leaf, Tensor::from_slice(&[] as &[i64]), HashMap::new());

    let node = tc.arena.node(tc.arena.resolve(leaf));
    assert!(
        node.host_value(FULL)
            .equal(&Tensor::from_slice(&[100i64, 101]))
    );
}

// Two backuped device nodes [1,2] -> [3,4]; returns (parent, child) ids.
fn backuped_chain(tc: &mut UnifiedTreeCore<Vec<i64>>) -> (NodeIdx_, NodeIdx_) {
    tc.insert(&insert_params(&vec![1, 2], &[10, 11]));
    tc.insert(&insert_params(&vec![1, 2, 3, 4], &[10, 11, 12, 13]));
    let parent = tc
        .match_prefix(&match_params(&vec![1, 2]))
        .best_match_node_id;
    let child = tc
        .match_prefix(&match_params(&vec![1, 2, 3, 4]))
        .best_match_node_id;
    tc.commit_backup(parent, Tensor::from_slice(&[20i64, 21]), HashMap::new());
    tc.commit_backup(child, Tensor::from_slice(&[22i64, 23]), HashMap::new());
    (tc.arena.resolve(parent), tc.arena.resolve(child))
}

// Demote `node_id` (device release of a backuped node), discarding the frees.
fn demote_node(tc: &mut UnifiedTreeCore<Vec<i64>>, node_id: NodeIdx_) {
    tc.demote(tc.arena.node(node_id).id);
}

#[test]
fn demote_releases_the_device_value_and_keeps_the_host_copy() {
    let mut tc = core();
    let (_parent, child) = backuped_chain(&mut tc);
    demote_node(&mut tc, child);
    let node = tc.arena.node(child);
    assert!(node.evicted() && node.backuped());
    assert_eq!(tc.full_evictable_size(), 2);
    tc.sanity_check(&[], &[]);
}

#[test]
fn build_load_back_spec_collects_the_evicted_chain_ancestors_first() {
    let mut tc = core();
    let (parent, child) = backuped_chain(&mut tc);
    demote_node(&mut tc, child);
    demote_node(&mut tc, parent);
    let (kv_xfer, comp_xfers) =
        tc.build_load_back_spec(tc.arena.node(child).id, /* req = */ None);
    assert_eq!(kv_xfer.name, PoolName::Kv);
    assert!(
        kv_xfer
            .host_indices
            .unwrap()
            .equal(&Tensor::from_slice(&[20i64, 21, 22, 23]))
    );
    assert!(kv_xfer.device_indices.is_none());
    assert_eq!(
        kv_xfer.nodes_to_load,
        Some(vec![tc.arena.node(parent).id, tc.arena.node(child).id])
    );
    assert!(comp_xfers.is_empty());
}

#[test]
fn build_load_back_spec_returns_an_empty_transfer_for_a_device_backed_node() {
    let mut tc = core();
    let (_parent, child) = backuped_chain(&mut tc);
    let (kv_xfer, comp_xfers) =
        tc.build_load_back_spec(tc.arena.node(child).id, /* req = */ None);
    let host_indices = kv_xfer.host_indices.unwrap();
    assert_eq!(host_indices.numel(), 0);
    assert_eq!(host_indices.kind(), Kind::Int64);
    assert_eq!(kv_xfer.nodes_to_load, Some(vec![]));
    assert!(comp_xfers.is_empty());
}

#[test]
fn commit_load_back_reattaches_device_slices_and_restores_the_match() {
    let mut tc = core();
    let (parent, child) = backuped_chain(&mut tc);
    demote_node(&mut tc, child);
    demote_node(&mut tc, parent);
    let (kv_xfer, comp_xfers) =
        tc.build_load_back_spec(tc.arena.node(child).id, /* req = */ None);
    let actions = tc.commit_load_back(
        tc.arena.node(child).id,
        Tensor::from_slice(&[50i64, 51, 52, 53]),
        kv_xfer,
        comp_xfers,
    );
    assert!(actions.is_empty());
    assert!(
        tc.arena
            .device_value(parent, FULL)
            .equal(&Tensor::from_slice(&[50i64, 51]))
    );
    assert!(
        tc.arena
            .device_value(child, FULL)
            .equal(&Tensor::from_slice(&[52i64, 53]))
    );
    assert_eq!(tc.full_evictable_size(), 4);
    // The orchestrator re-locks the loaded path right after commit; that lock walk
    // also re-evaluates the parent's transient D-leaf membership.
    tc.inc_lock_ref(tc.arena.node(child).id);
    tc.sanity_check(&[], &[(1, tc.arena.node(child).id)]);
    tc.dec_lock_ref(
        tc.arena.node(child).id,
        /* params = */ None,
        /* skip_swa = */ false,
    );
    tc.finish_load_back(tc.arena.node(child).id);
    tc.sanity_check(&[], &[]);
    let result = tc.match_prefix(&match_params(&vec![1, 2, 3, 4]));
    assert!(
        result
            .device_indices
            .equal(&Tensor::from_slice(&[50i64, 51, 52, 53]))
    );
}

#[test]
fn device_eviction_and_demote_skip_a_load_back_pinned_chain() {
    let mut tc = core();
    let (parent, child) = backuped_chain(&mut tc);
    demote_node(&mut tc, child);
    demote_node(&mut tc, parent);
    let (kv_xfer, comp_xfers) =
        tc.build_load_back_spec(tc.arena.node(child).id, /* req = */ None);
    tc.commit_load_back(
        tc.arena.node(child).id,
        Tensor::from_slice(&[50i64, 51, 52, 53]),
        kv_xfer,
        comp_xfers,
    );
    // The pin alone keeps the in-flight chain out of device eviction.
    tc.evict_device_start(FULL, 4);
    let (next, _) = tc.evict_device_next_node(FULL, &HashMap::new());
    assert_eq!(next, None);
    tc.evict_device_end(FULL);
    tc.demote(tc.arena.node(child).id);
    assert!(tc.arena.has_device_value(child, FULL));
    tc.finish_load_back(tc.arena.node(child).id);
    // The ack re-arms leaf-set membership and demotion.
    tc.evict_device_start(FULL, 4);
    let (next, _) = tc.evict_device_next_node(FULL, &HashMap::new());
    assert_eq!(next, Some(tc.arena.node(child).id));
    tc.evict_device_end(FULL);
    tc.demote(tc.arena.node(child).id);
    assert!(!tc.arena.has_device_value(child, FULL));
    tc.sanity_check(&[], &[]);
}

#[test]
fn component_has_host_value_only_tracks_the_demote_and_load_back_cycle() {
    let mut tc = core();
    tc.insert(&insert_params(&vec![1], &[10]));
    let leaf = tc.match_prefix(&match_params(&vec![1])).best_match_node_id;
    assert!(!tc.component_has_host_value_only(leaf, FULL));
    tc.commit_backup(leaf, Tensor::from_slice(&[20i64]), HashMap::new());
    // Device value still present: backuped but not host-only.
    assert!(!tc.component_has_host_value_only(leaf, FULL));
    let leaf_idx = tc.arena.resolve(leaf);
    demote_node(&mut tc, leaf_idx);
    assert!(tc.component_has_host_value_only(leaf, FULL));
    let (kv_xfer, comp_xfers) = tc.build_load_back_spec(leaf, /* req = */ None);
    tc.commit_load_back(leaf, Tensor::from_slice(&[30i64]), kv_xfer, comp_xfers);
    assert!(!tc.component_has_host_value_only(leaf, FULL));
    tc.finish_load_back(leaf);
    tc.sanity_check(&[], &[]);
}

#[test]
#[should_panic(expected = "!node.evicted() && node.backuped()")]
fn demote_panics_on_an_unbackuped_node() {
    let mut tc = core();
    tc.insert(&insert_params(&vec![1], &[10]));
    let leaf = tc.match_prefix(&match_params(&vec![1])).best_match_node_id;
    tc.demote(leaf);
}

#[test]
fn match_prefix_with_hicache_splits_a_host_only_backuped_node() {
    let mut tc = core();
    tc.set_hicache_enabled();
    tc.insert(&insert_params(&vec![1, 2, 3, 4], &[10, 11, 12, 13]));
    let leaf = tc
        .match_prefix(&match_params(&vec![1, 2, 3, 4]))
        .best_match_node_id;
    tc.commit_backup(
        leaf,
        Tensor::from_slice(&[100i64, 101, 102, 103]),
        HashMap::new(),
    );
    tc.demote(leaf);
    // The partial match splits the host-only node; the host prefix stays usable.
    let result = tc.match_prefix(&match_params(&vec![1, 2, 9]));
    assert_eq!(result.device_indices.numel(), 0);
    let root = tc.arena.root();
    assert_eq!(result.last_device_node_id, tc.arena.node(root).id);
    assert_eq!(result.host_hit_length, 2);
    assert_eq!(result.best_match_node_id, result.last_host_node_id);
    let parent = tc.arena.resolve(result.best_match_node_id);
    let child = tc.arena.node(parent).children[&(None, vec![3])];
    {
        let parent_node = tc.arena.node(parent);
        assert_eq!(parent_node.key, vec![1, 2]);
        assert!(parent_node.evicted() && parent_node.backuped());
        assert!(
            parent_node
                .host_value(FULL)
                .equal(&Tensor::from_slice(&[100i64, 101]))
        );
    }
    let child_node = tc.arena.node(child);
    assert_eq!(child_node.key, vec![3, 4]);
    assert!(child_node.evicted() && child_node.backuped());
    assert!(
        child_node
            .host_value(FULL)
            .equal(&Tensor::from_slice(&[102i64, 103]))
    );
    tc.sanity_check(&[], &[]);
}

#[test]
fn mixed_backup_evict_insert_keeps_the_leaf_sets_disjoint() {
    let mut tc = core();
    tc.set_hicache_enabled();
    tc.insert(&insert_params(&vec![1, 2], &[10, 11]));
    tc.insert(&insert_params(&vec![101, 102], &[20, 21]));
    tc.insert(&insert_params(&vec![201, 202], &[30, 31]));
    tc.insert(&insert_params(&vec![301, 302], &[40, 41]));
    tc.insert(&insert_params(&vec![401, 402], &[50, 51]));
    // Backing up (and thereby re-stamping) the first three chains leaves the
    // two unbacked chains as the oldest eviction victims.
    let first = tc
        .match_prefix(&match_params(&vec![1, 2]))
        .best_match_node_id;
    tc.commit_backup(first, Tensor::from_slice(&[100i64, 101]), HashMap::new());
    let second = tc
        .match_prefix(&match_params(&vec![101, 102]))
        .best_match_node_id;
    tc.commit_backup(second, Tensor::from_slice(&[102i64, 103]), HashMap::new());
    let third = tc
        .match_prefix(&match_params(&vec![201, 202]))
        .best_match_node_id;
    tc.commit_backup(third, Tensor::from_slice(&[104i64, 105]), HashMap::new());
    let mut tracker = HashMap::from([(FULL, 0)]);
    let (mut df, mut hf) = (HashMap::new(), HashMap::new());
    tc.evict_device_start(FULL, /* request_cnt = */ 4);
    loop {
        let (leaf, step) = tc.evict_device_next_node(FULL, &tracker);
        accumulate_step(step, &mut tracker, &mut df, &mut hf);
        let Some(leaf) = leaf else { break };
        let (_, step) = tc.evict_device_leaf(leaf, /* is_write_back = */ false);
        accumulate_step(step, &mut tracker, &mut df, &mut hf);
    }
    tc.evict_device_end(FULL);
    assert_eq!(tracker[&FULL], 4);
    // The unbacked chains died outright.
    assert_eq!(
        tc.match_prefix(&match_params(&vec![301, 302]))
            .device_indices
            .numel(),
        0
    );
    assert_eq!(
        tc.match_prefix(&match_params(&vec![401, 402]))
            .device_indices
            .numel(),
        0
    );
    tc.insert(&insert_params(&vec![501, 502], &[60, 61]));
    tc.insert(&insert_params(&vec![601, 602], &[70, 71]));
    tc.insert(&insert_params(&vec![701, 702], &[80, 81]));
    // D-leaf / H-leaf membership stays mutually exclusive after the mixed traffic.
    for node in tc.collect_all_nodes_() {
        assert!(
            !(tc.evictable_device_leaves.contains(node) && tc.evictable_host_leaves.contains(node))
        );
    }
    tc.sanity_check(&[], &[]);
}

fn write_back_core() -> UnifiedTreeCore<Vec<i64>> {
    UnifiedTreeCore::new(
        CacheInitParams {
            is_write_back: true,
            ..Default::default()
        },
        vec![FULL],
    )
}

// A device-on unbacked leaf [1,2] with a host-only child [3,4] under it.
fn unbacked_leaf_with_host_child(tc: &mut UnifiedTreeCore<Vec<i64>>) -> (NodeIdx_, NodeIdx_) {
    tc.insert(&insert_params(&vec![1, 2], &[10, 11]));
    let leaf = tc
        .match_prefix(&match_params(&vec![1, 2]))
        .best_match_node_id;
    let child = tc
        .arena
        .alloc_child(
            tc.arena.resolve(leaf),
            /* key = */ vec![3, 4],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    tc.arena
        .set_host_value(child, FULL, Tensor::from_slice(&[20i64, 21]));
    tc.update_evictable_leaf_sets_(child);
    tc.update_evictable_leaf_sets_(tc.arena.resolve(leaf));
    (tc.arena.resolve(leaf), child)
}

#[test]
fn drop_subtree_no_host_frees_the_leaf_and_its_host_descendants() {
    let mut tc = write_back_core();
    let (leaf, _child) = unbacked_leaf_with_host_child(&mut tc);
    let mut tracker = HashMap::from([(FULL, 0)]);
    let (mut df, mut hf) = (HashMap::new(), HashMap::new());
    let (dropped, step) = tc.drop_subtree_no_host(tc.arena.node(leaf).id);
    accumulate_step(step, &mut tracker, &mut df, &mut hf);
    assert!(dropped);
    // Under EvictLayer::All only device tokens enter the tracker; host
    // frees ride the host_frees tensors.
    assert_eq!(tracker[&FULL], 2);
    assert_eq!(df[&FULL].len(), 1);
    assert!(df[&FULL][0].equal(&Tensor::from_slice(&[10i64, 11])));
    assert_eq!(hf[&FULL].len(), 1);
    assert!(hf[&FULL][0].equal(&Tensor::from_slice(&[20i64, 21])));
    assert_eq!(tc.arena.len(), 1);
    let result = tc.match_prefix(&match_params(&vec![1, 2]));
    assert_eq!(result.device_indices.numel(), 0);
    tc.sanity_check(&[], &[]);
}

#[test]
fn drop_subtree_no_host_removes_a_deeper_host_chain_child_first() {
    let mut tc = write_back_core();
    let (leaf, child) = unbacked_leaf_with_host_child(&mut tc);
    let grandchild = tc
        .arena
        .alloc_child(
            child,
            /* key = */ vec![5],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    tc.arena
        .set_host_value(grandchild, FULL, Tensor::from_slice(&[22i64]));
    tc.update_evictable_leaf_sets_(grandchild);
    tc.update_evictable_leaf_sets_(child);
    let mut tracker = HashMap::from([(FULL, 0)]);
    let (mut df, mut hf) = (HashMap::new(), HashMap::new());
    let (dropped, step) = tc.drop_subtree_no_host(tc.arena.node(leaf).id);
    accumulate_step(step, &mut tracker, &mut df, &mut hf);
    assert!(dropped);
    assert_eq!(tracker[&FULL], 2);
    assert_eq!(hf[&FULL].len(), 2);
    assert_eq!(tc.arena.len(), 1);
    tc.sanity_check(&[], &[]);
}

#[test]
fn drop_subtree_no_host_bails_on_a_locked_descendant() {
    let mut tc = write_back_core();
    let (leaf, child) = unbacked_leaf_with_host_child(&mut tc);
    tc.arena
        .node_mut(child)
        .set_lock_ref_(ValueSlotIdx::host(FULL), 1);
    let mut tracker = HashMap::from([(FULL, 0)]);
    let (mut df, mut hf) = (HashMap::new(), HashMap::new());
    let (dropped, step) = tc.drop_subtree_no_host(tc.arena.node(leaf).id);
    accumulate_step(step, &mut tracker, &mut df, &mut hf);
    assert!(!dropped);
    assert_eq!(tracker[&FULL], 0);
    assert!(df.is_empty() && hf.is_empty());
    assert_eq!(tc.arena.len(), 3);
}

#[test]
fn drop_subtree_no_host_bails_on_a_host_locked_root() {
    let mut tc = write_back_core();
    let (leaf, _child) = unbacked_leaf_with_host_child(&mut tc);
    tc.arena
        .node_mut(leaf)
        .set_lock_ref_(ValueSlotIdx::host(FULL), 1);
    let (dropped, _step) = tc.drop_subtree_no_host(tc.arena.node(leaf).id);
    assert!(!dropped);
    assert_eq!(tc.arena.len(), 3);
}

#[test]
#[should_panic(expected = "is not a D-leaf")]
fn drop_subtree_no_host_panics_on_a_non_device_leaf() {
    let mut tc = write_back_core();
    tc.insert(&insert_params(&vec![1, 2], &[10, 11]));
    tc.insert(&insert_params(&vec![1, 2, 3, 4], &[10, 11, 12, 13]));
    let parent = tc
        .match_prefix(&match_params(&vec![1, 2]))
        .best_match_node_id;
    tc.drop_subtree_no_host(parent);
}

#[test]
#[should_panic(expected = "assertion failed")]
fn drop_subtree_no_host_panics_on_a_backuped_leaf() {
    let mut tc = write_back_core();
    tc.insert(&insert_params(&vec![1, 2], &[10, 11]));
    let leaf = tc
        .match_prefix(&match_params(&vec![1, 2]))
        .best_match_node_id;
    tc.commit_backup(leaf, Tensor::from_slice(&[20i64, 21]), HashMap::new());
    tc.drop_subtree_no_host(leaf);
}

#[test]
fn write_back_eviction_frees_the_device_value_exactly_once() {
    // The backup pass must not free the device value the DMA still reads;
    // the post-ack pass demotes and frees it exactly once.
    let mut tc = write_back_core();
    tc.insert(&insert_params(&vec![1], &[10]));
    let leaf = tc.match_prefix(&match_params(&vec![1])).best_match_node_id;
    let mut tracker = HashMap::from([(FULL, 0)]);
    let (mut df, mut hf) = (HashMap::new(), HashMap::new());
    let (action, step) = tc.evict_device_leaf(leaf, true);
    accumulate_step(step, &mut tracker, &mut df, &mut hf);
    assert_eq!(action.unwrap().node_ids, vec![leaf]);
    assert!(df.is_empty() && hf.is_empty());
    tc.commit_backup(leaf, Tensor::from_slice(&[20i64]), HashMap::new());
    let (action, step) = tc.evict_device_leaf(leaf, true);
    accumulate_step(step, &mut tracker, &mut df, &mut hf);
    assert!(action.is_none());
    assert_eq!(tracker[&FULL], 1);
    assert_eq!(df[&FULL].len(), 1);
    assert!(df[&FULL][0].equal(&Tensor::from_slice(&[10i64])));
    assert!(
        tc.arena.node(tc.arena.resolve(leaf)).evicted()
            && tc.arena.node(tc.arena.resolve(leaf)).backuped()
    );
    tc.sanity_check(&[], &[]);
}

#[test]
fn insert_empty_key_is_a_noop_and_mints_no_namespace_root() {
    let mut tc = core();
    let result = tc.insert(&insert_params(&vec![], &[]));
    assert_eq!(result.prefix_len, 0);
    // Vacuously true: there is no sequence whose mamba state could be missing.
    assert!(result.mamba_exist);
    assert_eq!(tc.arena.len(), 1);
    // A namespaced empty insert never creates the namespace root.
    let result = tc.insert(&InsertParams {
        extra_key: Some("ghost"),
        ..insert_params(&vec![], &[])
    });
    assert!(result.mamba_exist);
    assert_eq!(tc.arena.len(), 1);
    assert!(!tc.arena.namespace_exists(Some("ghost")));
}

#[test]
fn root_node_handle_is_namespace_independent() {
    let mut tc = core();
    let root_handle = tc.arena.node(tc.arena.root()).id;
    assert_eq!(tc.root_node_handle(None), root_handle);
    // The single root serves every namespace, seen or not.
    assert_eq!(tc.root_node_handle(Some("ghost")), root_handle);
    assert!(!tc.arena.namespace_exists(Some("ghost")));
    tc.insert(&InsertParams {
        extra_key: Some("chat"),
        ..insert_params(&vec![1], &[10])
    });
    assert_eq!(tc.root_node_handle(Some("chat")), root_handle);
}

#[test]
fn get_hash_values_reads_the_nodes_own_hashes() {
    let mut tc = core();
    let (a, _b) = matched_chain(&mut tc);
    assert_eq!(
        tc.get_hash_values(tc.arena.node(a).id),
        Vec::<String>::new()
    );
    tc.arena.node_mut(a).hash_value = Some(vec!["h0".to_string(), "h1".to_string()]);
    assert_eq!(
        tc.get_hash_values(tc.arena.node(a).id),
        vec!["h0".to_string(), "h1".to_string()]
    );
}

#[test]
fn insert_empty_key_still_touches_the_existing_root() {
    let mut tc = core();
    tc.insert(&InsertParams {
        priority: 7,
        ..insert_params(&vec![], &[])
    });
    assert_eq!(tc.arena.node(tc.arena.root()).priority, 7);
    // A namespaced empty insert touches the same single root.
    tc.insert(&InsertParams {
        extra_key: Some("chat"),
        priority: 9,
        ..insert_params(&vec![], &[])
    });
    assert_eq!(tc.arena.node(tc.arena.root()).priority, 9);
}

#[test]
fn insert_into_a_named_namespace_is_isolated() {
    let mut tc = core();
    tc.insert(&InsertParams {
        extra_key: Some("lora-1"),
        ..insert_params(&vec![1, 2], &[10, 11])
    });
    // The default namespace stays empty; the named namespace hits.
    let miss = tc.match_prefix(&match_params(&vec![1, 2]));
    assert_eq!(miss.device_indices.numel(), 0);
    let hit = tc.match_prefix(&MatchPrefixParams {
        key: &vec![1, 2],
        extra_key: Some("lora-1"),
    });
    assert!(hit.device_indices.equal(&Tensor::from_slice(&[10i64, 11])));
}

fn page2_core() -> UnifiedTreeCore<Vec<i64>> {
    UnifiedTreeCore::new(
        CacheInitParams {
            page_size: 2,
            ..Default::default()
        },
        vec![FULL],
    )
}

#[test]
fn insert_sub_page_key_is_a_noop_and_mints_no_namespace_root() {
    let mut tc = page2_core();
    let result = tc.insert(&insert_params(&vec![1], &[10]));
    assert_eq!(result.prefix_len, 0);
    assert!(result.mamba_exist);
    assert_eq!(tc.arena.len(), 1);
    let result = tc.insert(&InsertParams {
        extra_key: Some("ghost"),
        ..insert_params(&vec![1], &[10])
    });
    assert!(result.mamba_exist);
    assert_eq!(tc.arena.len(), 1);
    assert!(!tc.arena.namespace_exists(Some("ghost")));
}

#[test]
fn insert_page_size_two_drops_the_unaligned_tail() {
    let mut tc = page2_core();
    let result = tc.insert(&insert_params(&vec![1, 2, 3], &[10, 11, 12]));
    assert_eq!(result.prefix_len, 0);
    assert_eq!(tc.evictable_size_(FULL), 2);
    let leaf = tc
        .match_prefix(&match_params(&vec![1, 2]))
        .best_match_node_id;
    assert_eq!(tc.arena.node(tc.arena.resolve(leaf)).key, vec![1, 2]);
    assert!(
        tc.arena
            .device_value(tc.arena.resolve(leaf), FULL)
            .equal(&Tensor::from_slice(&[10i64, 11]))
    );
    tc.sanity_check(&[], &[]);
}

#[test]
fn insert_page_size_two_splits_mid_page_divergence_at_the_page_boundary() {
    let mut tc = page2_core();
    tc.insert(&insert_params(&vec![1, 2, 3, 4], &[10, 11, 12, 13]));
    // The keys share 3 atoms but pages quantize the overlap down to 2.
    let result = tc.insert(&insert_params(&vec![1, 2, 3, 9], &[20, 21, 22, 29]));
    assert_eq!(result.prefix_len, 2);
    let [CacheAction::FreeDeviceKV(freed)] = result.cache_actions.as_slice() else {
        panic!(
            "expected one FreeDeviceKV action, got {:?}",
            action_kinds(&result.cache_actions)
        );
    };
    assert_eq!(freed.len(), 1);
    assert!(freed[0].equal(&Tensor::from_slice(&[20i64, 21])));
    let prefix = tc
        .match_prefix(&match_params(&vec![1, 2]))
        .best_match_node_id;
    assert_eq!(tc.arena.node(tc.arena.resolve(prefix)).key, vec![1, 2]);
    let matched = tc.match_prefix(&match_params(&vec![1, 2, 3, 4]));
    assert!(
        matched
            .device_indices
            .equal(&Tensor::from_slice(&[10i64, 11, 12, 13]))
    );
    let matched = tc.match_prefix(&match_params(&vec![1, 2, 3, 9]));
    assert!(
        matched
            .device_indices
            .equal(&Tensor::from_slice(&[10i64, 11, 22, 29]))
    );
    tc.sanity_check(&[], &[]);
}

#[test]
fn match_prefix_page_size_two_splits_at_a_page_boundary() {
    let mut tc = page2_core();
    tc.insert(&insert_params(&vec![1, 2, 3, 4], &[10, 11, 12, 13]));
    // The query shares 3 atoms; pages quantize the split down to 2.
    let result = tc.match_prefix(&match_params(&vec![1, 2, 3, 9]));
    assert!(
        result
            .device_indices
            .equal(&Tensor::from_slice(&[10i64, 11]))
    );
    assert_eq!(
        tc.arena
            .node(tc.arena.resolve(result.best_match_node_id))
            .key,
        vec![1, 2]
    );
    // The split child stays reachable through its own page key.
    let matched = tc.match_prefix(&match_params(&vec![1, 2, 3, 4]));
    assert!(
        matched
            .device_indices
            .equal(&Tensor::from_slice(&[10i64, 11, 12, 13]))
    );
    tc.sanity_check(&[], &[]);
}

#[test]
fn match_prefix_page_size_two_sub_page_query_is_an_empty_match() {
    let mut tc = page2_core();
    tc.insert(&insert_params(&vec![1, 2], &[10, 11]));
    let root = tc.arena.root();
    let result = tc.match_prefix(&match_params(&vec![1]));
    assert_eq!(result.device_indices.numel(), 0);
    assert_eq!(result.best_match_node_id, tc.arena.node(root).id);
}

#[test]
fn evict_walk_page_size_two_empties_the_tree() {
    let mut tc = page2_core();
    tc.insert(&insert_params(&vec![1, 2, 3, 4], &[10, 11, 12, 13]));
    tc.insert(&insert_params(&vec![5, 6], &[14, 15]));
    let mut tracker = HashMap::from([(FULL, 0)]);
    let (mut df, mut hf) = (HashMap::new(), HashMap::new());
    tc.evict_device_start(FULL, /* request_cnt = */ 100);
    let mut evicted = 0;
    loop {
        let (leaf, step) = tc.evict_device_next_node(FULL, &tracker);
        accumulate_step(step, &mut tracker, &mut df, &mut hf);
        let Some(leaf) = leaf else { break };
        let (_, step) = tc.evict_device_leaf(leaf, /* is_write_back = */ false);
        accumulate_step(step, &mut tracker, &mut df, &mut hf);
        evicted += 1;
    }
    tc.evict_device_end(FULL);
    assert_eq!(evicted, 2);
    assert_eq!(tracker[&FULL], 6);
    assert_eq!(tc.arena.len(), 1);
    tc.sanity_check(&[], &[]);
}

#[test]
fn evict_and_detach_frees_the_device_value_and_tracks() {
    let mut tc = core();
    let root = tc.arena.root();
    let a = tc.add_new_node_(
        root,
        /* key = */ vec![1, 2],
        &Tensor::from_slice(&[10, 11]),
        /* priority = */ 0,
        /* extra_key = */ None,
    );
    // A mechanical LRU entry pins the detach branch (Full itself never lists).
    tc.device_lru_list_mut(FULL).insert_mru(a);
    let mut tracker = HashMap::from([(FULL, 0)]);
    let (mut df, mut hf) = (HashMap::new(), HashMap::new());
    let (device_freed, host_freed) = tc.evict_component_and_detach_lru_(
        a,
        FULL,
        &mut df,
        &mut hf,
        EvictLayer::Device,
        Some(&mut tracker),
    );
    assert_eq!((device_freed, host_freed), (2, 0));
    assert_eq!(tracker[&FULL], 2);
    assert_eq!(df[&FULL].len(), 1);
    assert!(!tc.device_lru_list(FULL).in_list(Some(a)));
}

#[test]
fn evict_and_detach_host_targets_the_host_tier() {
    let mut tc = core();
    let root = tc.arena.root();
    let a = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![1, 2],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    tc.arena
        .set_host_value(a, FULL, Tensor::from_slice(&[20i64, 21]));
    let mut tracker = HashMap::from([(FULL, 0)]);
    let (mut df, mut hf) = (HashMap::new(), HashMap::new());
    let (device_freed, host_freed) = tc.evict_component_and_detach_lru_(
        a,
        FULL,
        &mut df,
        &mut hf,
        EvictLayer::Host,
        Some(&mut tracker),
    );
    assert_eq!((device_freed, host_freed), (0, 2));
    assert_eq!(tracker[&FULL], 2);
    assert!(df.is_empty());
    assert_eq!(hf[&FULL].len(), 1);
}

#[test]
fn remove_leaf_from_parent_unlinks_and_recycles() {
    let mut tc = core();
    let root = tc.arena.root();
    let a = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![1, 2],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    tc.remove_leaf_from_parent_(a);
    assert!(tc.arena.node(root).children.is_empty());
    assert_eq!(tc.arena.len(), 1);
}

#[test]
#[should_panic(expected = "remove_leaf_from_parent_: a deletable leaf")]
fn remove_leaf_from_parent_panics_on_an_internal_node() {
    let mut tc = core();
    let root = tc.arena.root();
    let a = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![1],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    tc.arena
        .alloc_child(
            a,
            /* key = */ vec![2],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    tc.remove_leaf_from_parent_(a);
}

// Chain root -> a -> b where a is a valueless tombstone; b's deletion
// hands the walk a's id.
fn tombstone_chain(tc: &mut UnifiedTreeCore<Vec<i64>>) -> NodeIdx_ {
    let root = tc.arena.root();
    let a = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![1],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    let b = tc
        .arena
        .alloc_child(
            a,
            /* key = */ vec![2],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    tc.remove_leaf_from_parent_(b);
    a
}

fn delete_walk(tc: &mut UnifiedTreeCore<Vec<i64>>, from: NodeIdx_) {
    let mut tracker = HashMap::from([(FULL, 0)]);
    let (mut df, mut hf) = (HashMap::new(), HashMap::new());
    tc.iteratively_delete_tombstone_leaf_(from, &mut tracker, &mut df, &mut hf);
}

#[test]
fn tombstone_walk_deletes_childless_valueless_ancestors() {
    let mut tc = core();
    let a = tombstone_chain(&mut tc);
    delete_walk(&mut tc, a);
    assert_eq!(tc.arena.len(), 1);
    let root = tc.arena.root();
    assert!(tc.arena.node(root).children.is_empty());
}

#[test]
fn tombstone_walk_keeps_a_device_valued_ancestor_as_a_leaf() {
    let mut tc = core();
    let a = tombstone_chain(&mut tc);
    tc.arena
        .set_device_value(a, FULL, Tensor::from_slice(&[10i64]));
    delete_walk(&mut tc, a);
    assert_eq!(tc.arena.len(), 2);
    assert!(tc.evictable_device_leaves.contains(a));
}

#[test]
fn tombstone_walk_keeps_a_host_backed_ancestor_as_an_h_leaf() {
    let mut tc = core();
    let a = tombstone_chain(&mut tc);
    tc.arena
        .set_host_value(a, FULL, Tensor::from_slice(&[10i64]));
    delete_walk(&mut tc, a);
    assert_eq!(tc.arena.len(), 2);
    assert!(tc.evictable_host_leaves.contains(a));
    assert!(!tc.evictable_device_leaves.contains(a));
}

#[test]
fn tombstone_walk_stops_at_a_locked_ancestor() {
    let mut tc = core();
    let a = tombstone_chain(&mut tc);
    tc.arena
        .node_mut(a)
        .set_lock_ref_(ValueSlotIdx::device(FULL), 1);
    delete_walk(&mut tc, a);
    assert_eq!(tc.arena.len(), 2);
}

#[test]
fn tombstone_walk_cascades_multiple_levels() {
    let mut tc = core();
    let root = tc.arena.root();
    let a = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![1],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    let b = tc
        .arena
        .alloc_child(
            a,
            /* key = */ vec![2],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    let c = tc
        .arena
        .alloc_child(
            b,
            /* key = */ vec![3],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    tc.remove_leaf_from_parent_(c);
    delete_walk(&mut tc, b);
    assert_eq!(tc.arena.len(), 1);
}

#[test]
fn tombstone_walk_stops_at_a_host_locked_ancestor() {
    // A host lock (e.g. an in-flight write-back) pins a valueless ancestor.
    let mut tc = core();
    let a = tombstone_chain(&mut tc);
    tc.arena
        .node_mut(a)
        .set_lock_ref_(ValueSlotIdx::host(FULL), 1);
    delete_walk(&mut tc, a);
    assert_eq!(tc.arena.len(), 2);
}

#[test]
fn tombstone_walk_sweeps_orphaned_aux_device_data() {
    let mut tc = core();
    let recorder = Arc::new(SwaEvictionComponentForTest::new(
        /* leaf_priority = */ 0, /* internal_priority = */ 2,
    ));
    tc.register_component_(recorder.clone());
    let a = tombstone_chain(&mut tc);
    tc.arena.node_mut(a).values[SWA.idx()].value = Some(Tensor::from_slice(&[0i64, 1]));
    tc.device_lru_list_mut(SWA).insert_mru(a);
    let mut tracker = HashMap::new();
    let (mut df, mut hf) = (HashMap::new(), HashMap::new());
    tc.iteratively_delete_tombstone_leaf_(a, &mut tracker, &mut df, &mut hf);
    // The orphaned Swa device data is evicted before the node is deleted.
    assert_eq!(tc.arena.len(), 1);
    assert_eq!(
        *recorder.evictions.lock().unwrap(),
        vec![(a, EvictLayer::Device)]
    );
    assert_eq!(tracker[&SWA], 2);
    assert_eq!(df[&SWA].len(), 1);
    assert!(df[&SWA][0].equal(&Tensor::from_slice(&[0i64, 1])));
    assert!(!tc.device_lru_list(SWA).in_list(Some(a)));
}

#[test]
fn tombstone_walk_sweeps_orphaned_aux_host_data() {
    let mut tc = core();
    let recorder = Arc::new(SwaEvictionComponentForTest::new(
        /* leaf_priority = */ 0, /* internal_priority = */ 2,
    ));
    tc.register_component_(recorder.clone());
    let a = tombstone_chain(&mut tc);
    tc.arena
        .node_mut(a)
        .state_mut_(ValueSlotIdx::host(SWA))
        .value = Some(Tensor::from_slice(&[5i64]));
    tc.host_lru_list_mut(SWA).insert_mru(a);
    let mut tracker = HashMap::new();
    let (mut df, mut hf) = (HashMap::new(), HashMap::new());
    tc.iteratively_delete_tombstone_leaf_(a, &mut tracker, &mut df, &mut hf);
    // The orphaned Swa host data is evicted before the node is deleted.
    assert_eq!(tc.arena.len(), 1);
    assert_eq!(
        *recorder.evictions.lock().unwrap(),
        vec![(a, EvictLayer::Host)]
    );
    assert_eq!(tracker[&SWA], 1);
    assert_eq!(hf[&SWA].len(), 1);
    assert!(hf[&SWA][0].equal(&Tensor::from_slice(&[5i64])));
    assert!(!tc.host_lru_list(SWA).in_list(Some(a)));
}

#[test]
fn cascade_tombstones_full_after_the_component_sweep() {
    // Full's device value is cleared by the cascade, not by evict_component
    // (aux components read it while freeing); Full-only trees sweep nothing.
    let mut tc = core();
    let root = tc.arena.root();
    let a = tc.add_new_node_(
        root,
        /* key = */ vec![1],
        &Tensor::from_slice(&[10]),
        /* priority = */ 0,
        /* extra_key = */ None,
    );
    let mut tracker = HashMap::new();
    let (mut df, mut hf) = (HashMap::new(), HashMap::new());
    tc.cascade_evict_(
        a,
        BASE_COMPONENT_TYPE,
        &mut tracker,
        &mut df,
        &mut hf,
        EvictLayer::Device,
    );
    assert!(!tc.arena.has_device_value(a, FULL));
    // The trigger itself is excluded from the sweep: nothing freed here.
    assert!(df.is_empty());
    assert!(tracker.is_empty());
    assert!(!tc.evictable_device_leaves.contains(a));
}

#[test]
#[should_panic(expected = "cascade_evict_: EvictLayer::All is not a single layer")]
fn cascade_rejects_the_all_layer() {
    let mut tc = core();
    let root = tc.arena.root();
    let a = tc.add_new_node_(
        root,
        /* key = */ vec![1],
        &Tensor::from_slice(&[10]),
        /* priority = */ 0,
        /* extra_key = */ None,
    );
    let mut tracker = HashMap::new();
    let (mut df, mut hf) = (HashMap::new(), HashMap::new());
    tc.cascade_evict_(
        a,
        BASE_COMPONENT_TYPE,
        &mut tracker,
        &mut df,
        &mut hf,
        EvictLayer::All,
    );
}

#[test]
fn cascade_host_target_keeps_the_device_value() {
    let mut tc = core();
    let root = tc.arena.root();
    let a = tc.add_new_node_(
        root,
        /* key = */ vec![1],
        &Tensor::from_slice(&[10]),
        /* priority = */ 0,
        /* extra_key = */ None,
    );
    tc.arena
        .set_host_value(a, FULL, Tensor::from_slice(&[20i64]));
    let mut tracker = HashMap::new();
    let (mut df, mut hf) = (HashMap::new(), HashMap::new());
    tc.cascade_evict_(
        a,
        BASE_COMPONENT_TYPE,
        &mut tracker,
        &mut df,
        &mut hf,
        EvictLayer::Host,
    );
    // No tombstone on a host-target cascade; the D-leaf survives.
    assert!(tc.arena.has_device_value(a, FULL));
    assert!(tc.evictable_device_leaves.contains(a));
}

#[test]
fn cascade_moves_a_host_backed_node_into_the_h_leaf_set() {
    let mut tc = core();
    let root = tc.arena.root();
    let a = tc.add_new_node_(
        root,
        /* key = */ vec![1],
        &Tensor::from_slice(&[10]),
        /* priority = */ 0,
        /* extra_key = */ None,
    );
    tc.arena
        .set_host_value(a, FULL, Tensor::from_slice(&[20i64]));
    let mut tracker = HashMap::new();
    let (mut df, mut hf) = (HashMap::new(), HashMap::new());
    tc.cascade_evict_(
        a,
        BASE_COMPONENT_TYPE,
        &mut tracker,
        &mut df,
        &mut hf,
        EvictLayer::Device,
    );
    // Evicted but backuped: the node leaves the D-set and joins the H-set.
    assert!(!tc.evictable_device_leaves.contains(a));
    assert!(tc.evictable_host_leaves.contains(a));
}

// A D-leaf carrying both a Full device value and an Swa device value.
fn cascade_aux_setup(tc: &mut UnifiedTreeCore<Vec<i64>>) -> NodeIdx_ {
    let root = tc.arena.root();
    let a = tc.add_new_node_(
        root,
        /* key = */ vec![1],
        &Tensor::from_slice(&[10]),
        /* priority = */ 0,
        /* extra_key = */ None,
    );
    tc.arena.node_mut(a).values[SWA.idx()].value = Some(Tensor::from_slice(&[0i64]));
    a
}

#[test]
fn cascade_sweeps_an_equal_priority_aux_component() {
    // Swa leaf priority 0 equals Full's trigger priority: swept, not spared.
    let mut tc = core();
    let recorder = Arc::new(SwaEvictionComponentForTest::new(
        /* leaf_priority = */ 0, /* internal_priority = */ 2,
    ));
    tc.register_component_(recorder.clone());
    let a = cascade_aux_setup(&mut tc);
    let mut tracker = HashMap::new();
    let (mut df, mut hf) = (HashMap::new(), HashMap::new());
    tc.cascade_evict_(
        a,
        BASE_COMPONENT_TYPE,
        &mut tracker,
        &mut df,
        &mut hf,
        EvictLayer::Device,
    );
    assert_eq!(
        *recorder.evictions.lock().unwrap(),
        vec![(a, EvictLayer::Device)]
    );
    assert!(tc.arena.node(a).values[SWA.idx()].value.is_none());
    assert!(!tc.arena.has_device_value(a, FULL));
    assert_eq!(tracker[&SWA], 1);
    assert!(df[&SWA][0].equal(&Tensor::from_slice(&[0i64])));
}

#[test]
fn cascade_sweeps_a_lower_priority_aux_component() {
    let mut tc = core();
    let recorder = Arc::new(SwaEvictionComponentForTest::new(
        /* leaf_priority = */ -1, /* internal_priority = */ 2,
    ));
    tc.register_component_(recorder.clone());
    let a = cascade_aux_setup(&mut tc);
    let mut tracker = HashMap::new();
    let (mut df, mut hf) = (HashMap::new(), HashMap::new());
    tc.cascade_evict_(
        a,
        BASE_COMPONENT_TYPE,
        &mut tracker,
        &mut df,
        &mut hf,
        EvictLayer::Device,
    );
    assert_eq!(
        *recorder.evictions.lock().unwrap(),
        vec![(a, EvictLayer::Device)]
    );
    assert!(tc.arena.node(a).values[SWA.idx()].value.is_none());
}

#[test]
fn cascade_spares_a_higher_priority_aux_component() {
    // Swa leaf priority 1 outranks Full's trigger priority 0: kept intact.
    let mut tc = core();
    let recorder = Arc::new(SwaEvictionComponentForTest::new(
        /* leaf_priority = */ 1, /* internal_priority = */ 2,
    ));
    tc.register_component_(recorder.clone());
    let a = cascade_aux_setup(&mut tc);
    let mut tracker = HashMap::new();
    let (mut df, mut hf) = (HashMap::new(), HashMap::new());
    tc.cascade_evict_(
        a,
        BASE_COMPONENT_TYPE,
        &mut tracker,
        &mut df,
        &mut hf,
        EvictLayer::Device,
    );
    assert!(recorder.evictions.lock().unwrap().is_empty());
    assert!(tc.arena.node(a).values[SWA.idx()].value.is_some());
    // The trigger's deferred Full tombstone still lands.
    assert!(!tc.arena.has_device_value(a, FULL));
}

#[test]
fn cascade_spares_a_locked_component_of_equal_internal_priority() {
    // The Swa lock is a legit pin: leaf-collapse flattened priorities, but
    // its true internal priority matches the trigger's.
    let mut tc = core();
    let recorder = Arc::new(SwaEvictionComponentForTest::new(
        /* leaf_priority = */ 0, /* internal_priority = */ 2,
    ));
    tc.register_component_(recorder.clone());
    let a = cascade_aux_setup(&mut tc);
    tc.arena.node_mut(a).values[SWA.idx()].lock_ref = 1;
    let mut tracker = HashMap::new();
    let (mut df, mut hf) = (HashMap::new(), HashMap::new());
    tc.cascade_evict_(
        a,
        BASE_COMPONENT_TYPE,
        &mut tracker,
        &mut df,
        &mut hf,
        EvictLayer::Device,
    );
    assert!(recorder.evictions.lock().unwrap().is_empty());
    assert!(tc.arena.node(a).values[SWA.idx()].value.is_some());
    assert_eq!(tc.arena.node(a).values[SWA.idx()].lock_ref, 1);
}

#[test]
#[should_panic(expected = "a Swa device lock strands node")]
fn cascade_panics_on_a_locked_lower_internal_priority_component() {
    // A lock on a strictly-lower-priority tier is a real strand.
    let mut tc = core();
    tc.register_component_(Arc::new(SwaEvictionComponentForTest::new(
        /* leaf_priority = */ 0, /* internal_priority = */ 1,
    )));
    let a = cascade_aux_setup(&mut tc);
    tc.arena.node_mut(a).values[SWA.idx()].lock_ref = 1;
    let mut tracker = HashMap::new();
    let (mut df, mut hf) = (HashMap::new(), HashMap::new());
    tc.cascade_evict_(
        a,
        BASE_COMPONENT_TYPE,
        &mut tracker,
        &mut df,
        &mut hf,
        EvictLayer::Device,
    );
}

// An H-tier aux carrier: a raw child holding only an SWA host value.
fn cascade_host_aux_setup(tc: &mut UnifiedTreeCore<Vec<i64>>) -> NodeIdx_ {
    let root = tc.arena.root();
    let a = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![1],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    tc.arena.set_host_value(a, SWA, Tensor::from_slice(&[0i64]));
    a
}

#[test]
fn cascade_host_spares_a_locked_component_of_equal_internal_priority() {
    // The Swa host lock is a legit pin: its true internal priority matches
    // the trigger's, so the host cascade skips it and keeps the host value.
    let mut tc = core();
    let recorder = Arc::new(SwaEvictionComponentForTest::new(
        /* leaf_priority = */ 0, /* internal_priority = */ 2,
    ));
    tc.register_component_(recorder.clone());
    let a = cascade_host_aux_setup(&mut tc);
    tc.arena
        .node_mut(a)
        .set_lock_ref_(ValueSlotIdx::host(SWA), 1);
    let mut tracker = HashMap::new();
    let (mut df, mut hf) = (HashMap::new(), HashMap::new());
    tc.cascade_evict_(
        a,
        BASE_COMPONENT_TYPE,
        &mut tracker,
        &mut df,
        &mut hf,
        EvictLayer::Host,
    );
    assert!(recorder.evictions.lock().unwrap().is_empty());
    assert!(tc.arena.has_host_value(a, SWA));
    assert_eq!(tc.arena.host_lock_ref(a, SWA), 1);
}

#[test]
#[should_panic(expected = "a Swa host lock strands node")]
fn cascade_host_panics_on_a_locked_lower_internal_priority_component() {
    // A host lock on a strictly-lower-priority tier is a real strand.
    let mut tc = core();
    tc.register_component_(Arc::new(SwaEvictionComponentForTest::new(
        /* leaf_priority = */ 0, /* internal_priority = */ 1,
    )));
    let a = cascade_host_aux_setup(&mut tc);
    tc.arena
        .node_mut(a)
        .set_lock_ref_(ValueSlotIdx::host(SWA), 1);
    let mut tracker = HashMap::new();
    let (mut df, mut hf) = (HashMap::new(), HashMap::new());
    tc.cascade_evict_(
        a,
        BASE_COMPONENT_TYPE,
        &mut tracker,
        &mut df,
        &mut hf,
        EvictLayer::Host,
    );
}

#[test]
fn evict_walk_and_driver_empty_the_tree_end_to_end() {
    // The full eviction loop: insert three leaves, walk them in LRU
    // order, evict each through the driver, and end with a bare root.
    let mut tc = core();
    tc.insert(&insert_params(&vec![1, 2], &[10, 11]));
    tc.insert(&insert_params(&vec![3], &[12]));
    tc.insert(&insert_params(&vec![4], &[13]));
    let mut tracker = HashMap::from([(FULL, 0)]);
    let (mut df, mut hf) = (HashMap::new(), HashMap::new());
    tc.evict_device_start(FULL, /* request_cnt = */ 100);
    let mut evicted = 0;
    loop {
        let (leaf, step) = tc.evict_device_next_node(FULL, &tracker);
        accumulate_step(step, &mut tracker, &mut df, &mut hf);
        let Some(leaf) = leaf else { break };
        let (backup, step) = tc.evict_device_leaf(leaf, /* is_write_back = */ false);
        accumulate_step(step, &mut tracker, &mut df, &mut hf);
        assert!(backup.is_none());
        evicted += 1;
    }
    tc.evict_device_end(FULL);
    assert_eq!(evicted, 3);
    assert_eq!(tracker[&FULL], 4);
    assert_eq!(df[&FULL].len(), 3);
    assert_eq!(tc.arena.len(), 1);
    assert_eq!(tc.evictable_device_leaves.len(), 0);
    assert_eq!(tc.evictable_size_(FULL), 0);
}

#[test]
fn evict_driver_readmits_the_parent_into_the_walk() {
    // A two-level chain evicts leaf-first: the child's eviction turns the
    // prefix node into the next walkable D-leaf.
    let mut tc = core();
    tc.insert(&insert_params(&vec![1, 2, 3], &[10, 11, 12]));
    tc.insert(&insert_params(&vec![1, 2, 9], &[10, 11, 29]));
    let mut tracker = HashMap::from([(FULL, 0)]);
    let (mut df, mut hf) = (HashMap::new(), HashMap::new());
    tc.evict_device_start(FULL, /* request_cnt = */ 100);
    let mut evicted = 0;
    loop {
        let (leaf, step) = tc.evict_device_next_node(FULL, &tracker);
        accumulate_step(step, &mut tracker, &mut df, &mut hf);
        let Some(leaf) = leaf else { break };
        let (_, step) = tc.evict_device_leaf(leaf, false);
        accumulate_step(step, &mut tracker, &mut df, &mut hf);
        evicted += 1;
    }
    tc.evict_device_end(FULL);
    // Two suffix leaves plus the readmitted split-prefix node.
    assert_eq!(evicted, 3);
    assert_eq!(tracker[&FULL], 4);
    assert_eq!(tc.arena.len(), 1);
}

#[test]
fn release_all_component_layers_scrubs_a_host_leaf_from_both_leaf_sets() {
    // Host leaves only reach this helper via subtree drops; pin the discard.
    let mut tc = core();
    let (_a, b) = matched_chain(&mut tc);
    let taken = tc.arena.take_device_value(b, FULL);
    tc.arena.set_host_value(b, FULL, taken);
    tc.update_evictable_leaf_sets_(b);
    assert!(tc.evictable_host_leaves.contains(b));

    let mut tracker = HashMap::new();
    let (mut device_frees, mut host_frees) = (HashMap::new(), HashMap::new());
    tc.release_all_component_layers_(
        b,
        StorageMedium::Cpu,
        &mut tracker,
        &mut device_frees,
        &mut host_frees,
    );
    assert!(!tc.evictable_host_leaves.contains(b));
    assert!(!tc.evictable_device_leaves.contains(b));
    assert!(host_frees[&FULL][0].equal(&Tensor::from_slice(&[12i64])));
}

#[test]
fn evict_driver_deletes_through_a_tombstone_parent() {
    // A leaf under a valueless internal node cascades the delete upward.
    let mut tc = core();
    let root = tc.arena.root();
    let a = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![1],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    let b = tc.add_new_node_(
        a,
        /* key = */ vec![2],
        &Tensor::from_slice(&[10]),
        /* priority = */ 0,
        /* extra_key = */ None,
    );
    tc.evict_device_leaf(tc.arena.node(b).id, false);
    assert_eq!(tc.arena.len(), 1);
}

#[test]
fn insert_after_eviction_reuses_the_slot_but_never_the_handle() {
    let mut tc = core();
    tc.insert(&insert_params(&vec![1, 2, 3], &[10, 11, 12]));
    tc.insert(&insert_params(&vec![4], &[13]));
    let leaf = tc.match_prefix(&match_params(&vec![4])).best_match_node_id;
    let leaf_idx = tc.arena.resolve(leaf);
    tc.evict_device_leaf(leaf, /* is_write_back = */ false);
    assert_eq!(tc.arena.len(), 2);
    // The freed handle no longer resolves.
    assert!(tc.arena.try_resolve(leaf).is_none());
    // The splitting insert allocates its prefix node into the freed slot.
    tc.insert(&insert_params(&vec![1, 2, 9], &[20, 21, 29]));
    assert_eq!(tc.arena.len(), 4);
    let prefix = tc
        .match_prefix(&match_params(&vec![1, 2]))
        .best_match_node_id;
    // Slot recycled, but the stale handle can never alias the new node.
    assert_eq!(tc.arena.resolve(prefix), leaf_idx);
    assert_ne!(prefix, leaf);
}

#[test]
#[should_panic(expected = "is not allocated")]
fn stale_handle_panics_after_its_node_is_freed() {
    let mut tc = core();
    tc.insert(&insert_params(&vec![4], &[13]));
    let leaf = tc.match_prefix(&match_params(&vec![4])).best_match_node_id;
    tc.evict_device_leaf(leaf, /* is_write_back = */ false);
    tc.inc_lock_ref(leaf);
}

#[test]
#[should_panic(expected = "is not allocated")]
fn pre_reset_handle_panics_after_reset() {
    let mut tc = core();
    tc.insert(&insert_params(&vec![4], &[13]));
    let leaf = tc.match_prefix(&match_params(&vec![4])).best_match_node_id;
    tc.reset();
    tc.arena.resolve(leaf);
}

#[test]
#[should_panic(expected = "is not a D-leaf")]
fn evict_driver_rejects_a_non_leaf() {
    let mut tc = core();
    tc.insert(&insert_params(&vec![1, 2, 3], &[10, 11, 12]));
    tc.insert(&insert_params(&vec![1, 2, 9], &[10, 11, 29]));
    // The split-prefix node has valued children: not a D-leaf.
    let prefix = tc
        .match_prefix(&match_params(&vec![1, 2]))
        .best_match_node_id;
    tc.evict_device_leaf(prefix, false);
}

#[test]
fn evict_driver_write_back_returns_the_backup_action_for_an_unbacked_leaf() {
    let mut tc = core();
    tc.insert(&insert_params(&vec![1], &[10]));
    let leaf = tc.match_prefix(&match_params(&vec![1])).best_match_node_id;
    let mut tracker = HashMap::from([(FULL, 0)]);
    let (mut df, mut hf) = (HashMap::new(), HashMap::new());
    let (action, step) = tc.evict_device_leaf(leaf, /* is_write_back = */ true);
    accumulate_step(step, &mut tracker, &mut df, &mut hf);
    // Write-back carries only the leaf itself; nothing is freed yet.
    assert_eq!(action.unwrap().node_ids, vec![leaf]);
    assert!(!tc.arena.node(tc.arena.resolve(leaf)).evicted());
    assert_eq!(tracker[&FULL], 0);
    assert!(df.is_empty() && hf.is_empty());
}

#[test]
fn evict_driver_demotes_a_backuped_leaf_to_host_only() {
    let mut tc = core();
    tc.insert(&insert_params(&vec![1], &[10]));
    let leaf = tc.match_prefix(&match_params(&vec![1])).best_match_node_id;
    tc.arena
        .set_host_value(tc.arena.resolve(leaf), FULL, Tensor::from_slice(&[20i64]));
    let mut tracker = HashMap::from([(FULL, 0)]);
    let (mut df, mut hf) = (HashMap::new(), HashMap::new());
    let (action, step) = tc.evict_device_leaf(leaf, false);
    accumulate_step(step, &mut tracker, &mut df, &mut hf);
    assert!(action.is_none());
    // The node stays in the tree, now host-only.
    let node = tc.arena.node(tc.arena.resolve(leaf));
    assert!(node.evicted() && node.backuped());
    assert_eq!(tracker[&FULL], 1);
    assert_eq!(df[&FULL].len(), 1);
    assert!(hf.is_empty());
    assert!(!tc.evictable_device_leaves.contains(tc.arena.resolve(leaf)));
    assert!(tc.evictable_host_leaves.contains(tc.arena.resolve(leaf)));
    tc.sanity_check(&[], &[]);
}

#[test]
fn evict_device_leaf_step_counts_are_independent_of_prior_evictions() {
    let mut tc = core();
    tc.insert(&insert_params(&vec![1, 2, 3], &[10, 11, 12]));
    tc.insert(&insert_params(&vec![4], &[13]));
    let first = tc
        .match_prefix(&match_params(&vec![1, 2, 3]))
        .best_match_node_id;
    let second = tc.match_prefix(&match_params(&vec![4])).best_match_node_id;
    let (_, step) = tc.evict_device_leaf(first, /* is_write_back = */ false);
    assert_eq!(step.tracker[&FULL], 3);
    // The second step reports only its own leaf, not a running total.
    let (_, step) = tc.evict_device_leaf(second, /* is_write_back = */ false);
    assert_eq!(step.tracker[&FULL], 1);
    assert_eq!(step.device_frees[&FULL].len(), 1);
}

#[test]
fn empty_tree_device_evict_walk_returns_nothing_for_each_component() {
    let mut tc = core();
    tc.evict_device_start(FULL, /* request_cnt = */ 10);
    let (leaf, step) = tc.evict_device_next_node(FULL, &HashMap::from([(FULL, 0)]));
    assert_eq!(leaf, None);
    assert!(step.tracker.is_empty());
    assert!(step.device_frees.is_empty() && step.host_frees.is_empty());
    tc.evict_device_end(FULL);
    assert_eq!(tc.evictable_size_(FULL), 0);

    let mut tc = swa_match_core(/* window = */ 4);
    tc.evict_device_start(SWA, /* request_cnt = */ 10);
    let (leaf, step) = tc.evict_device_next_node(SWA, &HashMap::from([(FULL, 0), (SWA, 0)]));
    assert_eq!(leaf, None);
    assert!(step.tracker.is_empty());
    assert!(step.device_frees.is_empty() && step.host_frees.is_empty());
    tc.evict_device_end(SWA);
    assert_eq!(tc.evictable_size_(SWA), 0);

    let mut tc = UnifiedTreeCore::<Vec<i64>>::new(
        CacheInitParams {
            mamba_cache_chunk_size: Some(256),
            ..CacheInitParams::default()
        },
        vec![FULL, MAMBA],
    );
    tc.evict_device_start(MAMBA, /* request_cnt = */ 10);
    let (leaf, step) = tc.evict_device_next_node(MAMBA, &HashMap::from([(FULL, 0), (MAMBA, 0)]));
    assert_eq!(leaf, None);
    assert!(step.tracker.is_empty());
    assert!(step.device_frees.is_empty() && step.host_frees.is_empty());
    tc.evict_device_end(MAMBA);
    assert_eq!(tc.evictable_size_(MAMBA), 0);
}

#[test]
fn empty_match_result_anchors_all_boundaries_at_the_root() {
    let tc = core();
    let root = tc.arena.root();
    let result = tc.empty_match_result();
    assert_eq!(result.last_device_node_id, tc.arena.node(root).id);
    assert_eq!(result.last_host_node_id, tc.arena.node(root).id);
    assert_eq!(result.best_match_node_id, tc.arena.node(root).id);
    assert_eq!(result.host_hit_length, 0);
    assert_eq!(result.device_indices.numel(), 0);
    assert_eq!(result.device_indices.kind(), Kind::Int64);
}

#[test]
fn touch_node_stamps_a_fresh_access_tick() {
    let mut tc = core();
    let root = tc.arena.root();
    let a = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![1],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    let created_at = tc.arena.node(a).last_access_counter;
    tc.touch_node_(a);
    let first_touch = tc.arena.node(a).last_access_counter;
    assert!(first_touch > created_at);
    tc.touch_node_(a);
    assert!(tc.arena.node(a).last_access_counter > first_touch);
}

#[test]
#[should_panic(expected = "not implemented")]
fn touch_node_refreshes_aux_component_lrus() {
    let mut tc = core();
    tc.register_component_(Arc::new(SwaComponentForTest));
    let root = tc.arena.root();
    let a = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![1],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    tc.touch_node_(a);
}

#[test]
fn touch_node_on_a_root_only_stamps_the_tick() {
    // Even with an aux component present, a root touch skips the refresh loop.
    let mut tc = core();
    tc.register_component_(Arc::new(SwaComponentForTest));
    let root = tc.arena.root();
    let before = tc.arena.node(root).last_access_counter;
    tc.touch_node_(root);
    assert!(tc.arena.node(root).last_access_counter > before);
}

#[test]
fn inc_hit_count_bumps_and_stays_quiet_without_hicache() {
    let mut tc = core();
    let root = tc.arena.root();
    let a = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![1],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    tc.arena
        .set_device_value(a, FULL, Tensor::from_slice(&[0i64]));
    assert!(!tc.inc_hit_count_and_check_(a, /* chunked = */ false));
    assert_eq!(tc.arena.node(a).hit_count, 1);
    // The tree defaults keep the host tier off and the threshold at 256.
    assert!(!tc.enable_hicache);
    assert_eq!(tc.write_through_threshold, 256);
}

#[test]
fn inc_hit_count_skips_evicted_or_chunked_nodes() {
    let mut tc = core();
    let root = tc.arena.root();
    let evicted = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![1],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    let chunked = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![2],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    tc.arena
        .set_device_value(chunked, FULL, Tensor::from_slice(&[0i64]));
    assert!(!tc.inc_hit_count_and_check_(evicted, /* chunked = */ false));
    assert_eq!(tc.arena.node(evicted).hit_count, 0);
    assert!(!tc.inc_hit_count_and_check_(chunked, /* chunked = */ true));
    assert_eq!(tc.arena.node(chunked).hit_count, 0);
}

#[test]
fn inc_hit_count_is_a_noop_in_write_back_mode() {
    let params = CacheInitParams {
        is_write_back: true,
        ..Default::default()
    };
    let mut tc: UnifiedTreeCore<Vec<i64>> = UnifiedTreeCore::new(params, vec![FULL]);
    let root = tc.arena.root();
    let a = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![1],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    tc.arena
        .set_device_value(a, FULL, Tensor::from_slice(&[0i64]));
    assert!(!tc.inc_hit_count_and_check_(a, /* chunked = */ false));
    assert_eq!(tc.arena.node(a).hit_count, 0);
}

#[test]
fn inc_hit_count_fires_the_write_through_check() {
    let mut tc = core();
    tc.set_hicache_enabled();
    tc.write_through_threshold = 2;
    let root = tc.arena.root();
    let a = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![1],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    tc.arena
        .set_device_value(a, FULL, Tensor::from_slice(&[0i64]));
    assert!(!tc.inc_hit_count_and_check_(a, /* chunked = */ false));
    assert!(tc.inc_hit_count_and_check_(a, /* chunked = */ false));
    // A backuped node never re-fires.
    tc.arena
        .set_host_value(a, FULL, Tensor::from_slice(&[0i64]));
    assert!(!tc.inc_hit_count_and_check_(a, /* chunked = */ false));
    assert_eq!(tc.arena.node(a).hit_count, 3);
}

#[test]
#[should_panic(expected = "Swa component is not enabled")]
fn dispatch_panics_for_a_disabled_component() {
    let mut tc = core();
    tc.evict_device_start(SWA, /* request_cnt = */ 1);
}

#[test]
fn registry_and_map_share_the_drivers() {
    let tc = core();
    let component = &tc.components[0];
    assert_eq!(component.component_type(), FULL);
    assert!(Arc::ptr_eq(
        component,
        tc.components_by_type[FULL.idx()].as_ref().unwrap()
    ));
}

#[test]
#[should_panic(expected = "at least one component type is required")]
fn new_rejects_an_empty_component_list() {
    UnifiedTreeCore::<Vec<i64>>::new(CacheInitParams::default(), vec![]);
}

#[test]
#[should_panic(expected = "duplicate component type Full")]
fn new_rejects_duplicate_component_types() {
    UnifiedTreeCore::<Vec<i64>>::new(CacheInitParams::default(), vec![FULL, FULL]);
}

#[test]
#[should_panic(expected = "the base (Full) component is required")]
fn new_requires_the_full_component() {
    UnifiedTreeCore::<Vec<i64>>::new(CacheInitParams::default(), vec![SWA]);
}

#[test]
fn new_resolves_the_eviction_policy() {
    let mut tc = UnifiedTreeCore::new(
        CacheInitParams {
            eviction_policy: "FIFO".to_string(),
            ..Default::default()
        },
        vec![FULL],
    );
    let root = tc.arena.root();
    let a = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![1],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    tc.arena.node_mut(a).creation_counter = 7;
    assert_eq!(
        tc.eviction_strategy.get_priority(tc.arena.node(a)),
        crate::unified_lru_list::PriorityKey(7, 0)
    );
}

#[test]
fn new_builds_independent_lru_lists() {
    let mut tc = core();
    let root = tc.arena.root();
    let n1 = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![1],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    tc.device_lru_list_mut(FULL).insert_mru(n1);
    // Every component/tier list exists as an independent container.
    assert!(tc.device_lru_list(FULL).in_list(Some(n1)));
    assert!(!tc.host_lru_list(FULL).in_list(Some(n1)));
    assert!(!tc.device_lru_list(SWA).in_list(Some(n1)));
}

#[test]
fn update_adds_an_unlocked_device_valued_leaf() {
    let mut tc = core();
    let root = tc.arena.root();
    let n1 = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![1],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    tc.arena
        .set_device_value(n1, FULL, Tensor::from_slice(&[10i64]));
    tc.update_evictable_leaf_sets_(n1);
    assert!(tc.evictable_device_leaves.contains(n1));
    assert!(!tc.evictable_host_leaves.contains(n1));
}

#[test]
fn update_discards_a_node_that_stops_being_a_device_leaf() {
    let mut tc = core();
    let root = tc.arena.root();
    let n1 = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![1],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    tc.arena
        .set_device_value(n1, FULL, Tensor::from_slice(&[10i64]));
    tc.update_evictable_leaf_sets_(n1);
    assert!(tc.evictable_device_leaves.contains(n1));
    let _ = tc.arena.take_device_value(n1, FULL);
    tc.update_evictable_leaf_sets_(n1);
    assert!(!tc.evictable_device_leaves.contains(n1));
}

#[test]
fn device_leaf_excludes_the_root() {
    let mut tc = core();
    let root = tc.arena.root();
    // Root keys are empty, so a key-aligned device value is the empty tensor.
    let empty: [i64; 0] = [];
    tc.arena
        .set_device_value(root, FULL, Tensor::from_slice(&empty));
    tc.update_evictable_leaf_sets_(root);
    assert!(!tc.evictable_device_leaves.contains(root));
    assert!(!tc.evictable_host_leaves.contains(root));
}

#[test]
fn device_leaf_excludes_locked_node_until_unlocked() {
    let mut tc = core();
    let root = tc.arena.root();
    let n1 = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![1],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    tc.arena
        .set_device_value(n1, FULL, Tensor::from_slice(&[10i64]));
    tc.arena
        .node_mut(n1)
        .set_lock_ref_(ValueSlotIdx::device(FULL), 1);
    tc.update_evictable_leaf_sets_(n1);
    assert!(!tc.evictable_device_leaves.contains(n1));
    tc.arena
        .node_mut(n1)
        .set_lock_ref_(ValueSlotIdx::device(FULL), 0);
    tc.update_evictable_leaf_sets_(n1);
    assert!(tc.evictable_device_leaves.contains(n1));
}

#[test]
fn device_leaf_excludes_node_locked_by_another_component() {
    let mut tc = core();
    let root = tc.arena.root();
    let n1 = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![1],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    tc.arena
        .set_device_value(n1, FULL, Tensor::from_slice(&[10i64]));
    // The lock check spans every component, not just Full.
    tc.arena.node_mut(n1).values[SWA.idx()].lock_ref = 1;
    tc.update_evictable_leaf_sets_(n1);
    assert!(!tc.evictable_device_leaves.contains(n1));
}

#[test]
fn device_leaf_excludes_parent_with_a_device_valued_child() {
    let mut tc = core();
    let root = tc.arena.root();
    let parent = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![1, 2],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    let child = tc
        .arena
        .alloc_child(
            parent,
            /* key = */ vec![3],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    tc.arena
        .set_device_value(parent, FULL, Tensor::from_slice(&[10i64, 11]));
    tc.arena
        .set_device_value(child, FULL, Tensor::from_slice(&[30i64]));
    tc.update_evictable_leaf_sets_(parent);
    tc.update_evictable_leaf_sets_(child);
    assert!(!tc.evictable_device_leaves.contains(parent));
    assert!(tc.evictable_device_leaves.contains(child));
    // Once the child's device value is evicted, the parent becomes the D-leaf.
    let _ = tc.arena.take_device_value(child, FULL);
    tc.update_evictable_leaf_sets_(parent);
    tc.update_evictable_leaf_sets_(child);
    assert!(tc.evictable_device_leaves.contains(parent));
    assert!(!tc.evictable_device_leaves.contains(child));
}

#[test]
fn update_discards_a_node_that_stops_being_a_host_leaf() {
    let mut tc = core();
    let root = tc.arena.root();
    let n1 = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![1],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    tc.arena
        .set_host_value(n1, FULL, Tensor::from_slice(&[10i64]));
    tc.update_evictable_leaf_sets_(n1);
    assert!(tc.evictable_host_leaves.contains(n1));
    // Loading the device value back means the node is no longer evicted.
    tc.arena
        .set_device_value(n1, FULL, Tensor::from_slice(&[20i64]));
    tc.update_evictable_leaf_sets_(n1);
    assert!(!tc.evictable_host_leaves.contains(n1));
}

#[test]
fn host_leaf_excludes_the_root() {
    let mut tc = core();
    let root = tc.arena.root();
    // Root keys are empty, so a key-aligned host value is the empty tensor.
    let empty: [i64; 0] = [];
    tc.arena
        .set_host_value(root, FULL, Tensor::from_slice(&empty));
    tc.update_evictable_leaf_sets_(root);
    assert!(!tc.evictable_host_leaves.contains(root));
}

#[test]
fn host_leaf_true_for_evicted_backuped_childless_node() {
    let mut tc = core();
    let root = tc.arena.root();
    let n1 = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![1],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    tc.arena
        .set_host_value(n1, FULL, Tensor::from_slice(&[10i64]));
    tc.update_evictable_leaf_sets_(n1);
    assert!(tc.evictable_host_leaves.contains(n1));
    assert!(!tc.evictable_device_leaves.contains(n1));
}

#[test]
fn device_valued_node_is_not_a_host_leaf() {
    let mut tc = core();
    let root = tc.arena.root();
    let n1 = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![1],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    tc.arena
        .set_device_value(n1, FULL, Tensor::from_slice(&[10i64]));
    tc.arena
        .set_host_value(n1, FULL, Tensor::from_slice(&[20i64]));
    tc.update_evictable_leaf_sets_(n1);
    assert!(tc.evictable_device_leaves.contains(n1));
    assert!(!tc.evictable_host_leaves.contains(n1));
}

#[test]
fn host_leaf_excludes_node_with_children() {
    let mut tc = core();
    let root = tc.arena.root();
    let parent = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![1],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    let _child = tc
        .arena
        .alloc_child(
            parent,
            /* key = */ vec![2],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    tc.arena
        .set_host_value(parent, FULL, Tensor::from_slice(&[10i64]));
    tc.update_evictable_leaf_sets_(parent);
    assert!(!tc.evictable_host_leaves.contains(parent));
}

#[test]
fn host_leaf_excludes_host_locked_node_until_unlocked() {
    let mut tc = core();
    let root = tc.arena.root();
    let n1 = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![1],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    tc.arena
        .set_host_value(n1, FULL, Tensor::from_slice(&[10i64]));
    tc.arena
        .node_mut(n1)
        .set_lock_ref_(ValueSlotIdx::host(FULL), 1);
    tc.update_evictable_leaf_sets_(n1);
    assert!(!tc.evictable_host_leaves.contains(n1));
    tc.arena
        .node_mut(n1)
        .set_lock_ref_(ValueSlotIdx::host(FULL), 0);
    tc.update_evictable_leaf_sets_(n1);
    assert!(tc.evictable_host_leaves.contains(n1));
}

#[test]
fn host_leaf_excludes_node_host_locked_by_another_component() {
    let mut tc = core();
    let root = tc.arena.root();
    let n1 = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![1],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    tc.arena
        .set_host_value(n1, FULL, Tensor::from_slice(&[10i64]));
    // The host-lock check spans every component, not just Full.
    tc.arena
        .node_mut(n1)
        .state_mut_(ValueSlotIdx::host(SWA))
        .lock_ref = 1;
    tc.update_evictable_leaf_sets_(n1);
    assert!(!tc.evictable_host_leaves.contains(n1));
}

#[test]
fn tombstone_without_backup_is_in_neither_set() {
    let mut tc = core();
    let root = tc.arena.root();
    let n1 = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![1],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    tc.update_evictable_leaf_sets_(n1);
    assert!(!tc.evictable_device_leaves.contains(n1));
    assert!(!tc.evictable_host_leaves.contains(n1));
}

#[test]
#[should_panic(expected = "out of bounds")]
fn update_panics_on_missing_node() {
    let mut tc = core();
    tc.update_evictable_leaf_sets_(NodeIdx_(999));
}

#[test]
fn evict_host_leaf_frees_host_values_and_credits_the_tracker() {
    let mut tc = core();
    let root = tc.arena.root();
    let n1 = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![1, 2],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    tc.arena
        .set_host_value(n1, FULL, Tensor::from_slice(&[10i64, 11]));
    tc.evictable_host_leaves.add(n1);
    let mut tr = HashMap::from([(FULL, 0)]);
    let (mut df, mut hf) = (HashMap::new(), HashMap::new());
    tc.evict_host_leaf_(n1, &mut tr, &mut df, &mut hf);
    assert_eq!(tr[&FULL], 2);
    assert_eq!(hf[&FULL].len(), 1);
    assert!(df.is_empty());
    assert!(tc.evictable_host_leaves.is_empty());
    assert_eq!(tc.arena.len(), 1);
    tc.sanity_check(&[], &[]);
}

#[test]
#[should_panic(expected = "is not an H-leaf")]
fn evict_host_leaf_panics_on_a_device_valued_node() {
    let mut tc = core();
    let root = tc.arena.root();
    let n1 = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![1],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    tc.arena
        .set_device_value(n1, FULL, Tensor::from_slice(&[10i64]));
    let mut tr = HashMap::new();
    let (mut df, mut hf) = (HashMap::new(), HashMap::new());
    tc.evict_host_leaf_(n1, &mut tr, &mut df, &mut hf);
}

#[test]
fn evict_host_leaf_cascades_tombstone_ancestors() {
    let mut tc = core();
    let root = tc.arena.root();
    let t = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![1],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    let h = tc
        .arena
        .alloc_child(
            t,
            /* key = */ vec![2],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    tc.arena
        .set_host_value(h, FULL, Tensor::from_slice(&[20i64]));
    tc.evictable_host_leaves.add(h);
    let mut tr = HashMap::from([(FULL, 0)]);
    let (mut df, mut hf) = (HashMap::new(), HashMap::new());
    tc.evict_host_leaf_(h, &mut tr, &mut df, &mut hf);
    // The valueless ancestor t is deleted by the tombstone walk.
    assert_eq!(tc.arena.len(), 1);
    assert_eq!(tr[&FULL], 1);
    tc.sanity_check(&[], &[]);
}

#[test]
fn drive_host_eviction_is_a_noop_for_an_absent_component() {
    let mut tc = core();
    let root = tc.arena.root();
    let n1 = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![1],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    tc.arena
        .set_host_value(n1, FULL, Tensor::from_slice(&[10i64]));
    tc.evictable_host_leaves.add(n1);
    let mut tr = HashMap::from([(SWA, 0)]);
    let (mut df, mut hf) = (HashMap::new(), HashMap::new());
    accumulate_step(
        tc.drive_host_eviction(SWA, /* num_tokens = */ 100),
        &mut tr,
        &mut df,
        &mut hf,
    );
    assert_eq!(tr[&SWA], 0);
    assert!(hf.is_empty());
    assert!(tc.evictable_host_leaves.contains(n1));
}

#[test]
fn drive_host_eviction_keeps_zero_delta_tracker_entries() {
    let mut tc = core();
    let result = tc.drive_host_eviction(FULL, /* num_tokens = */ 10);
    assert_eq!(result.tracker, HashMap::from([(FULL, 0)]));
    assert!(result.device_frees.is_empty());
    assert!(result.host_frees.is_empty());
}

#[test]
fn drive_host_eviction_dispatches_reclaim_only_under_write_back() {
    let mut tc = UnifiedTreeCore::new(
        CacheInitParams {
            is_write_back: true,
            ..CacheInitParams::default()
        },
        vec![FULL],
    );
    let recorder = Arc::new(RecordingComponentForTest::default());
    tc.register_component_(recorder.clone());

    let result = tc.drive_host_eviction(SWA, /* num_tokens = */ 10);

    assert_eq!(
        *recorder.host_eviction_calls.lock().unwrap(),
        vec![("reclaim", 0), ("drive", 2)]
    );
    assert_eq!(result.tracker, HashMap::from([(SWA, 5)]));

    recorder.host_eviction_calls.lock().unwrap().clear();
    tc.is_write_back = false;
    let result = tc.drive_host_eviction(SWA, /* num_tokens = */ 10);

    assert_eq!(
        *recorder.host_eviction_calls.lock().unwrap(),
        vec![("drive", 0)]
    );
    assert_eq!(result.tracker, HashMap::from([(SWA, 3)]));
}

#[test]
fn drive_host_eviction_default_reclaim_hook_is_a_noop() {
    let mut tc = UnifiedTreeCore::new(
        CacheInitParams {
            is_write_back: true,
            ..CacheInitParams::default()
        },
        vec![FULL],
    );
    tc.register_component_(Arc::new(SwaComponentForTest));

    let result = tc.drive_host_eviction(SWA, /* num_tokens = */ 10);

    assert_eq!(result.tracker, HashMap::from([(SWA, 0)]));
    assert!(result.device_frees.is_empty());
    assert!(result.host_frees.is_empty());
}

#[test]
fn evict_layer_contains_matches_intflag_membership() {
    assert!(EvictLayer::Device.contains(EvictLayer::Device));
    assert!(!EvictLayer::Device.contains(EvictLayer::Host));
    assert!(EvictLayer::Host.contains(EvictLayer::Host));
    assert!(!EvictLayer::Host.contains(EvictLayer::Device));
    assert!(EvictLayer::All.contains(EvictLayer::Device));
    assert!(EvictLayer::All.contains(EvictLayer::Host));
    assert!(EvictLayer::All.contains(EvictLayer::All));
    assert!(!EvictLayer::Device.contains(EvictLayer::All));
    assert!(!EvictLayer::Host.contains(EvictLayer::All));
}

#[test]
fn reset_restores_a_fresh_tree() {
    let mut tc = core();
    tc.insert(&insert_params(&vec![1, 2, 3], &[10, 11, 12]));
    tc.insert(&InsertParams {
        extra_key: Some("chat"),
        ..insert_params(&vec![7, 8], &[20, 21])
    });
    let matched = tc.match_prefix(&match_params(&vec![1, 2, 3]));
    tc.inc_lock_ref(matched.best_match_node_id);
    assert_eq!(tc.protected_size(), 3);
    // Seed aux LRU, host LRU, and host-leaf state so the reset must clear each.
    let root = tc.arena.root();
    let d = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![9],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    let h = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![12],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    tc.device_lru_list_mut(SWA).insert_mru(d);
    tc.host_lru_list_mut(SWA).insert_mru(h);
    tc.evictable_host_leaves.add(h);
    tc.reset();
    assert_eq!(tc.arena.len(), 1);
    assert_eq!(tc.evictable_size(), 0);
    assert_eq!(tc.protected_size(), 0);
    assert_eq!(tc.total_size(), (0, 0));
    assert!(tc.evictable_device_leaves.is_empty());
    assert!(tc.evictable_host_leaves.is_empty());
    assert_eq!(tc.device_lru_list(FULL).len(), 0);
    assert_eq!(tc.device_lru_list(SWA).len(), 0);
    assert_eq!(tc.host_lru_list(SWA).len(), 0);
    let matched = tc.match_prefix(&match_params(&vec![1, 2, 3]));
    assert_eq!(matched.device_indices.numel(), 0);
    // The tree accepts fresh inserts after the reset.
    tc.insert(&insert_params(&vec![4, 5], &[30, 31]));
    assert_eq!(tc.evictable_size(), 2);
}

#[test]
fn size_accessors_mirror_the_full_component_state() {
    let mut tc = core();
    tc.insert(&insert_params(&vec![1, 2, 3], &[10, 11, 12]));
    assert_eq!(tc.evictable_size(), 3);
    assert_eq!(tc.full_evictable_size(), 3);
    assert_eq!(tc.protected_size(), 0);
    assert_eq!(tc.component_evictable_size(FULL), 3);
    let matched = tc.match_prefix(&match_params(&vec![1, 2, 3]));
    tc.inc_lock_ref(matched.best_match_node_id);
    assert_eq!(tc.protected_size(), 3);
    assert_eq!(tc.full_protected_size(), 3);
    assert_eq!(tc.evictable_size(), 0);
}

#[test]
fn swa_size_accessors_mirror_the_swa_component_state() {
    let mut tc = core();
    tc.component_state_mut(SWA).evictable_size = 2;
    tc.component_state_mut(SWA).protected_size = 1;
    assert_eq!(tc.swa_evictable_size(), 2);
    assert_eq!(tc.swa_protected_size(), 1);
    // The Full accessors read their own slot, untouched by the SWA seed.
    assert_eq!(tc.full_evictable_size(), 0);
    assert_eq!(tc.full_protected_size(), 0);
}

#[test]
fn total_size_spans_namespaces_and_aux_values() {
    let mut tc = core();
    tc.insert(&insert_params(&vec![1, 2, 3], &[10, 11, 12]));
    tc.insert(&InsertParams {
        extra_key: Some("chat"),
        ..insert_params(&vec![7, 8], &[20, 21])
    });
    assert_eq!(tc.total_size(), (5, 0));
    // An SWA-valued node adds to the aux total only.
    tc.register_component_(Arc::new(SwaComponentForTest));
    let root = tc.arena.root();
    let a = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![9, 10, 11],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    tc.arena.node_mut(a).values[SWA.idx()].value = Some(Tensor::from_slice(&[0i64, 1, 2]));
    assert_eq!(tc.total_size(), (5, 3));
}

// Zip a canary walk into sorted (slot, position, prev_slot) rows; emission order is not a contract.
fn sorted_canary_rows(walk: KvCanaryWalkResult) -> Vec<(i64, i64, i64)> {
    let mut rows: Vec<(i64, i64, i64)> = walk
        .slot_indices
        .into_iter()
        .zip(walk.positions)
        .zip(walk.prev_slot_indices)
        .map(|((slot, position), prev)| (slot, position, prev))
        .collect();
    rows.sort_unstable();
    rows
}

#[test]
fn walk_for_kv_canary_on_an_empty_tree_emits_nothing() {
    let tc = core();
    assert_eq!(
        sorted_canary_rows(tc.walk_for_kv_canary(false, false)),
        Vec::<(i64, i64, i64)>::new()
    );
}

#[test]
fn walk_for_kv_canary_chains_slots_across_namespaces() {
    let mut tc = core();
    tc.insert(&insert_params(&vec![1, 2, 3], &[10, 11, 12]));
    tc.insert(&InsertParams {
        extra_key: Some("chat"),
        ..insert_params(&vec![7, 8], &[20, 21])
    });
    assert_eq!(
        sorted_canary_rows(tc.walk_for_kv_canary(false, false)),
        vec![
            (10, 0, -1),
            (11, 1, 10),
            (12, 2, 11),
            (20, 0, -1),
            (21, 1, 20)
        ]
    );
}

#[test]
fn walk_for_kv_canary_unlocked_only_skips_locked_nodes_but_keeps_the_chain() {
    let mut tc = core();
    let (a, _b) = matched_chain(&mut tc);
    tc.inc_lock_ref(tc.arena.node(a).id);
    assert_eq!(
        sorted_canary_rows(tc.walk_for_kv_canary(true, false)),
        vec![(12, 2, 11)]
    );
    assert_eq!(
        sorted_canary_rows(tc.walk_for_kv_canary(false, false)),
        vec![(10, 0, -1), (11, 1, 10), (12, 2, 11)]
    );
}

#[test]
fn walk_for_kv_canary_spans_device_evicted_nodes_without_emitting_them() {
    let mut tc = core();
    let (a, _b) = matched_chain(&mut tc);
    let _ = tc.arena.take_device_value(a, FULL);
    assert_eq!(
        sorted_canary_rows(tc.walk_for_kv_canary(false, false)),
        vec![(12, 2, -1)]
    );
}

#[test]
fn walk_for_kv_canary_swa_resident_only_skips_swa_tombstoned_nodes() {
    let mut tc = core();
    let (a, _b) = matched_chain(&mut tc);
    tc.register_component_(Arc::new(SwaComponentForTest));
    tc.arena
        .set_device_value(a, SWA, Tensor::from_slice(&[0i64, 1]));
    assert_eq!(
        sorted_canary_rows(tc.walk_for_kv_canary(false, true)),
        vec![(10, 0, -1), (11, 1, 10)]
    );
}

#[test]
fn walk_for_kv_canary_swa_filter_is_inert_without_the_swa_component() {
    let mut tc = core();
    let (_a, _b) = matched_chain(&mut tc);
    assert_eq!(
        sorted_canary_rows(tc.walk_for_kv_canary(false, true)),
        vec![(10, 0, -1), (11, 1, 10), (12, 2, 11)]
    );
}

#[test]
fn walk_for_kv_canary_unlocked_only_gates_on_the_swa_lock_under_the_swa_filter() {
    let mut tc = core();
    let (a, _b) = matched_chain(&mut tc);
    tc.register_component_(Arc::new(SwaComponentForTest));
    tc.arena
        .set_device_value(a, SWA, Tensor::from_slice(&[0i64, 1]));
    // A node can hold Full KV for a running request while its SWA slots are unused.
    tc.arena.node_mut(a).values[FULL.idx()].lock_ref = 1;
    assert_eq!(
        sorted_canary_rows(tc.walk_for_kv_canary(true, true)),
        vec![(10, 0, -1), (11, 1, 10)]
    );
    // A held SWA lock excludes the node even with Full unlocked.
    tc.arena.node_mut(a).values[FULL.idx()].lock_ref = 0;
    tc.arena.node_mut(a).values[SWA.idx()].lock_ref = 1;
    assert_eq!(
        sorted_canary_rows(tc.walk_for_kv_canary(true, true)),
        Vec::<(i64, i64, i64)>::new()
    );
}

#[test]
fn get_component_device_value_reads_the_full_value() {
    let mut tc = core();
    tc.insert(&insert_params(&vec![1, 2, 3], &[10, 11, 12]));
    let leaf = tc
        .match_prefix(&match_params(&vec![1, 2, 3]))
        .best_match_node_id;
    assert!(
        tc.get_component_device_value(leaf, FULL)
            .unwrap()
            .equal(&Tensor::from_slice(&[10i64, 11, 12]))
    );
    let _ = tc.arena.take_device_value(tc.arena.resolve(leaf), FULL);
    assert!(tc.get_component_device_value(leaf, FULL).is_none());
}

#[test]
#[should_panic(expected = "Swa component is not enabled")]
fn get_component_device_value_panics_on_an_unregistered_component() {
    let tc = core();
    let root = tc.arena.root();
    tc.get_component_device_value(tc.arena.node(root).id, SWA);
}

#[test]
fn get_component_device_value_reads_the_registered_components_slot() {
    let mut tc = core();
    tc.register_component_(Arc::new(SwaComponentForTest));
    let (a, _b) = matched_chain(&mut tc);
    assert!(
        tc.get_component_device_value(tc.arena.node(a).id, SWA)
            .is_none()
    );
    tc.arena
        .set_device_value(a, SWA, Tensor::from_slice(&[5i64, 6]));
    assert_eq!(
        Vec::<i64>::try_from(
            tc.get_component_device_value(tc.arena.node(a).id, SWA)
                .unwrap()
        )
        .unwrap(),
        vec![5, 6]
    );
}

#[test]
fn component_evictable_size_is_zero_for_an_absent_component() {
    assert_eq!(core().component_evictable_size(SWA), 0);
}

#[test]
fn component_evictable_size_reads_the_registered_components_state() {
    let mut tc = core();
    tc.register_component_(Arc::new(SwaComponentForTest));
    tc.component_state_mut(SWA).evictable_size = 7;
    assert_eq!(tc.component_evictable_size(SWA), 7);
}

#[test]
fn component_protected_size_is_zero_for_an_absent_component() {
    assert_eq!(core().component_protected_size(SWA), 0);
}

#[test]
fn component_protected_size_reads_the_registered_components_state() {
    let mut tc = core();
    tc.register_component_(Arc::new(SwaComponentForTest));
    tc.component_state_mut(SWA).protected_size = 7;
    assert_eq!(tc.component_protected_size(SWA), 7);
}

#[test]
fn is_full_device_evicted_flips_when_the_value_tombstones() {
    let mut tc = core();
    tc.insert(&insert_params(&vec![1, 2, 3], &[10, 11, 12]));
    let leaf = tc
        .match_prefix(&match_params(&vec![1, 2, 3]))
        .best_match_node_id;
    assert!(!tc.is_full_device_evicted(leaf));
    let _ = tc.arena.take_device_value(tc.arena.resolve(leaf), FULL);
    assert!(tc.is_full_device_evicted(leaf));
}

#[test]
fn set_component_device_value_stores_and_restamps_the_lru() {
    let mut tc = core();
    tc.register_component_(Arc::new(SwaComponentForTest));
    let root = tc.arena.root();
    let a = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![1, 2],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    assert!(!tc.arena.has_device_value(a, SWA));
    tc.set_component_device_value(tc.arena.node(a).id, SWA, Tensor::from_slice(&[5i64, 6]));
    assert!(
        tc.arena
            .device_value(a, SWA)
            .equal(&Tensor::from_slice(&[5i64, 6]))
    );
    assert_eq!(tc.evictable_size_(SWA), 2);
    assert!(tc.device_lru_list(SWA).in_list(Some(a)));
    assert_eq!(tc.device_lru_list(SWA).len(), 1);
    assert!(!tc.host_lru_list(SWA).in_list(Some(a)));
}

#[test]
fn set_component_device_value_migrates_the_node_off_the_host_lru() {
    let mut tc = core();
    tc.register_component_(Arc::new(SwaComponentForTest));
    let root = tc.arena.root();
    let a = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![1],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    tc.host_lru_list_mut(SWA).insert_mru(a);
    tc.set_component_device_value(tc.arena.node(a).id, SWA, Tensor::from_slice(&[5i64]));
    assert!(!tc.host_lru_list(SWA).in_list(Some(a)));
    assert_eq!(tc.host_lru_list(SWA).len(), 0);
    assert_eq!(tc.device_lru_list(SWA).len(), 1);
}

#[test]
#[should_panic(expected = "auxiliary components only")]
fn set_component_device_value_rejects_the_base_component() {
    let mut tc = core();
    let root = tc.arena.root();
    tc.set_component_device_value(
        tc.arena.node(root).id,
        BASE_COMPONENT_TYPE,
        Tensor::from_slice(&[1i64]),
    );
}

#[test]
fn collect_full_device_indices_concatenates_in_root_order() {
    let mut tc = core();
    tc.insert(&insert_params(&vec![1, 2, 3], &[10, 11, 12]));
    tc.insert(&insert_params(&vec![1, 2, 3, 4, 5], &[20, 21, 22, 13, 14]));
    let leaf = tc
        .match_prefix(&match_params(&vec![1, 2, 3, 4, 5]))
        .best_match_node_id;
    let parent = tc.arena.node(tc.arena.resolve(leaf)).parent();
    let root = tc.arena.root();
    assert!(
        tc.collect_full_device_indices(leaf, tc.arena.node(root).id)
            .equal(&Tensor::from_slice(&[10i64, 11, 12, 13, 14]))
    );
    assert!(
        tc.collect_full_device_indices(leaf, tc.arena.node(parent).id)
            .equal(&Tensor::from_slice(&[13i64, 14]))
    );
    assert_eq!(
        tc.collect_full_device_indices(tc.arena.node(root).id, tc.arena.node(root).id)
            .numel(),
        0
    );
}

#[test]
#[should_panic(expected = "value: Full/device slot has no value")]
fn collect_full_device_indices_panics_on_an_evicted_path() {
    let mut tc = core();
    tc.insert(&insert_params(&vec![1, 2, 3], &[10, 11, 12]));
    tc.insert(&insert_params(&vec![1, 2, 3, 4, 5], &[20, 21, 22, 13, 14]));
    let leaf = tc
        .match_prefix(&match_params(&vec![1, 2, 3, 4, 5]))
        .best_match_node_id;
    let parent = tc.arena.node(tc.arena.resolve(leaf)).parent();
    let _ = tc.arena.take_device_value(parent, FULL);
    let root = tc.arena.root();
    let _ = tc.collect_full_device_indices(leaf, tc.arena.node(root).id);
}

#[test]
fn all_values_flatten_spans_namespaces() {
    let mut tc = core();
    assert_eq!(tc.all_values_flatten().numel(), 0);
    tc.insert(&insert_params(&vec![1, 2, 3], &[10, 11, 12]));
    tc.insert(&insert_params(&vec![1, 2, 3, 4, 5], &[20, 21, 22, 13, 14]));
    tc.insert(&InsertParams {
        extra_key: Some("chat"),
        ..insert_params(&vec![7, 8], &[20, 21])
    });
    let (sorted, _) = tc.all_values_flatten().sort(0, /* descending = */ false);
    assert!(sorted.equal(&Tensor::from_slice(&[10i64, 11, 12, 13, 14, 20, 21])));
}

#[test]
fn collect_all_nodes_visits_every_root_subtree() {
    let mut tc = core();
    tc.insert(&insert_params(&vec![1, 2, 3], &[10, 11, 12]));
    tc.insert(&InsertParams {
        extra_key: Some("chat"),
        ..insert_params(&vec![7, 8], &[20, 21])
    });
    let mut nodes = tc.collect_all_nodes_();
    nodes.sort();
    assert_eq!(nodes.len(), tc.arena.len());
    assert_eq!(nodes, vec![NodeIdx_(0), NodeIdx_(1), NodeIdx_(2)]);
}

#[test]
fn pretty_format_renders_every_namespace_and_component() {
    let mut tc = core();
    tc.insert(&insert_params(&vec![1, 2, 3], &[10, 11, 12]));
    tc.insert(&insert_params(&vec![1, 2, 3, 4, 5], &[20, 21, 22, 13, 14]));
    tc.insert(&InsertParams {
        extra_key: Some("chat"),
        ..insert_params(&vec![7], &[30])
    });
    let leaf = tc
        .match_prefix(&match_params(&vec![1, 2, 3, 4, 5]))
        .best_match_node_id;
    let _ = tc.arena.take_device_value(tc.arena.resolve(leaf), FULL);
    tc.register_component_(Arc::new(SwaComponentForTest));
    tc.arena.node_mut(NodeIdx_(1)).values[SWA.idx()].value = Some(Tensor::from_slice(&[0i64]));
    // Sibling render order follows HashMap iteration, so pin the line set.
    let mut lines: Vec<String> = tc.pretty_format_().lines().map(str::to_string).collect();
    lines.sort();
    let mut expected: Vec<String> = [
        " [0] 0 full_lock=1 Full=no Swa=no",
        "   [3] 1 full_lock=0 Full=yes Swa=no",
        "   [1] 3 full_lock=0 Full=yes Swa=yes",
        "     [2] 2 full_lock=0 Full=no Swa=no",
    ]
    .map(str::to_string)
    .to_vec();
    expected.sort();
    assert_eq!(lines, expected);
}

// A healthy multi-namespace tree with a split for the sanity pins.
fn sane_tree() -> UnifiedTreeCore<Vec<i64>> {
    let mut tc = core();
    tc.insert(&insert_params(&vec![1, 2, 3], &[10, 11, 12]));
    tc.insert(&insert_params(&vec![1, 2, 3, 4, 5], &[20, 21, 22, 13, 14]));
    tc.insert(&insert_params(&vec![1, 2, 9], &[30, 31, 39]));
    tc.insert(&InsertParams {
        extra_key: Some("chat"),
        ..insert_params(&vec![7, 8], &[40, 41])
    });
    tc
}

#[test]
fn sanity_check_passes_on_a_healthy_tree() {
    let mut tc = sane_tree();
    tc.sanity_check(&[], &[]);
    let leaf = tc
        .match_prefix(&match_params(&vec![1, 2, 9]))
        .best_match_node_id;
    tc.inc_lock_ref(leaf);
    tc.sanity_check(&[(1, leaf)], &[(2, leaf)]);
    tc.dec_lock_ref(
        tc.arena.node(tc.arena.resolve(leaf)).id,
        /* params = */ None,
        /* skip_swa = */ false,
    );
    tc.sanity_check(&[], &[]);
}

#[test]
fn sanity_check_passes_after_the_eviction_walk() {
    let mut tc = sane_tree();
    let mut tracker = HashMap::from([(FULL, 0)]);
    let (mut df, mut hf) = (HashMap::new(), HashMap::new());
    tc.evict_device_start(FULL, /* request_cnt = */ 100);
    loop {
        let (leaf, step) = tc.evict_device_next_node(FULL, &tracker);
        accumulate_step(step, &mut tracker, &mut df, &mut hf);
        let Some(leaf) = leaf else { break };
        let (_, step) = tc.evict_device_leaf(leaf, /* is_write_back = */ false);
        accumulate_step(step, &mut tracker, &mut df, &mut hf);
    }
    tc.evict_device_end(FULL);
    tc.sanity_check(&[], &[]);
    // The emptied "chat" namespace leaves nothing behind; only the root survives.
    assert_eq!(tc.arena.len(), 1);
    assert!(!tc.arena.namespace_exists(Some("chat")));
}

#[test]
#[should_panic(expected = "D-leaf missing")]
fn sanity_check_detects_a_missing_device_leaf() {
    let mut tc = sane_tree();
    let leaf = tc
        .match_prefix(&match_params(&vec![1, 2, 9]))
        .best_match_node_id;
    tc.evictable_device_leaves.discard(tc.arena.resolve(leaf));
    tc.sanity_check(&[], &[]);
}

#[test]
#[should_panic(expected = "D-leaf extra")]
fn sanity_check_detects_an_extra_device_leaf() {
    let mut tc = sane_tree();
    let leaf = tc
        .match_prefix(&match_params(&vec![1, 2, 9]))
        .best_match_node_id;
    let parent = tc.arena.node(tc.arena.resolve(leaf)).parent();
    tc.evictable_device_leaves.add(parent);
    tc.sanity_check(&[], &[]);
}

#[test]
#[should_panic(expected = "[Size]")]
fn sanity_check_detects_size_drift() {
    let mut tc = sane_tree();
    tc.component_state_mut(FULL).evictable_size = 999;
    tc.sanity_check(&[], &[]);
}

#[test]
#[should_panic(expected = "dead: no Full device and no Full host")]
fn sanity_check_detects_a_dead_node() {
    let mut tc = sane_tree();
    let leaf = tc
        .match_prefix(&match_params(&vec![1, 2, 9]))
        .best_match_node_id;
    let _ = tc.arena.take_device_value(tc.arena.resolve(leaf), FULL);
    tc.sanity_check(&[], &[]);
}

#[test]
#[should_panic(expected = "device present but parent")]
fn sanity_check_detects_an_evicted_parent_prefix() {
    let mut tc = sane_tree();
    let leaf = tc
        .match_prefix(&match_params(&vec![1, 2, 9]))
        .best_match_node_id;
    let parent = tc.arena.node(tc.arena.resolve(leaf)).parent();
    let _ = tc.arena.take_device_value(parent, FULL);
    tc.sanity_check(&[], &[]);
}

#[test]
#[should_panic(expected = "evicted but lock_ref")]
fn sanity_check_detects_a_locked_tombstone() {
    let mut tc = sane_tree();
    let leaf = tc
        .match_prefix(&match_params(&vec![1, 2, 9]))
        .best_match_node_id;
    tc.inc_lock_ref(leaf);
    let _ = tc.arena.take_device_value(tc.arena.resolve(leaf), FULL);
    tc.sanity_check(&[], &[]);
}

#[test]
#[should_panic(expected = "Full device LRU not empty")]
fn sanity_check_detects_full_lru_pollution() {
    let mut tc = sane_tree();
    tc.device_lru_list_mut(FULL).insert_mru(NodeIdx_(0));
    tc.sanity_check(&[], &[]);
}

#[test]
#[should_panic(expected = "device LRU mismatch at node")]
fn sanity_check_detects_an_aux_lru_mismatch() {
    let mut tc = sane_tree();
    let leaf = tc
        .match_prefix(&match_params(&vec![1, 2, 9]))
        .best_match_node_id;
    tc.register_component_(Arc::new(SwaComponentForTest));
    tc.arena.node_mut(tc.arena.resolve(leaf)).values[SWA.idx()].value =
        Some(Tensor::from_slice(&[0i64, 0, 0]));
    tc.sanity_check(&[], &[]);
}

#[test]
#[should_panic(expected = "write_through node 7 not in tree")]
fn sanity_check_detects_an_untracked_ongoing_node() {
    let tc = sane_tree();
    tc.sanity_check(&[(7, 999)], &[]);
}

#[test]
#[should_panic(expected = "load_back node 8 lock_ref=0")]
fn sanity_check_detects_an_unlocked_ongoing_node() {
    let mut tc = sane_tree();
    let leaf = tc
        .match_prefix(&match_params(&vec![1, 2, 9]))
        .best_match_node_id;
    tc.sanity_check(&[], &[(8, leaf)]);
}

#[test]
#[should_panic(expected = "[Root] root 0 holds a Full device value")]
fn sanity_check_detects_a_valued_root() {
    let mut tc = sane_tree();
    tc.arena.node_mut(NodeIdx_(0)).values[FULL.idx()].value = Some(Tensor::from_slice(&[0i64]));
    tc.sanity_check(&[], &[]);
}

#[test]
#[should_panic(expected = "[Root] root 0 Full lock_ref=0")]
fn sanity_check_detects_an_unlocked_root() {
    let mut tc = sane_tree();
    tc.arena.node_mut(NodeIdx_(0)).values[FULL.idx()].lock_ref = 0;
    tc.sanity_check(&[], &[]);
}

#[test]
#[should_panic(expected = "[Root] root 0 has a parent pointer")]
fn sanity_check_detects_a_parented_root() {
    let mut tc = sane_tree();
    tc.arena.node_mut(NodeIdx_(0)).parent = Some(NodeIdx_(1));
    tc.sanity_check(&[], &[]);
}

// A tree with the Swa stub registered and the root's Swa lock backfilled.
fn swa_locked_roots_tree() -> UnifiedTreeCore<Vec<i64>> {
    let mut tc = sane_tree();
    tc.register_component_(Arc::new(SwaComponentForTest));
    tc.arena.node_mut(tc.arena.root()).values[SWA.idx()].lock_ref = 1;
    tc
}

#[test]
#[should_panic(expected = "[Root] root 0 holds a Swa device value")]
fn sanity_check_detects_an_aux_device_valued_root() {
    let mut tc = swa_locked_roots_tree();
    tc.arena.node_mut(NodeIdx_(0)).values[SWA.idx()].value = Some(Tensor::from_slice(&[0i64]));
    tc.sanity_check(&[], &[]);
}

#[test]
#[should_panic(expected = "[Root] root 0 holds a Swa host value")]
fn sanity_check_detects_an_aux_host_valued_root() {
    let mut tc = swa_locked_roots_tree();
    tc.arena
        .node_mut(NodeIdx_(0))
        .state_mut_(ValueSlotIdx::host(SWA))
        .value = Some(Tensor::from_slice(&[0i64]));
    tc.sanity_check(&[], &[]);
}

#[test]
#[should_panic(expected = "[Root] root 0 Swa lock_ref=0")]
fn sanity_check_detects_an_unlocked_aux_root() {
    let mut tc = swa_locked_roots_tree();
    tc.arena.node_mut(NodeIdx_(0)).values[SWA.idx()].lock_ref = 0;
    tc.sanity_check(&[], &[]);
}

#[test]
#[should_panic(expected = "[Tree] child")]
fn sanity_check_detects_a_broken_parent_pointer() {
    let mut tc = sane_tree();
    let leaf = tc
        .match_prefix(&match_params(&vec![1, 2, 9]))
        .best_match_node_id;
    let leaf_idx = tc.arena.resolve(leaf);
    tc.arena.node_mut(leaf_idx).parent = Some(NodeIdx_(0));
    tc.sanity_check(&[], &[]);
}

#[test]
#[should_panic(expected = "device present but Full.value=None")]
fn sanity_check_detects_aux_device_without_full() {
    let mut tc = sane_tree();
    let leaf = tc
        .match_prefix(&match_params(&vec![1, 2, 9]))
        .best_match_node_id;
    tc.register_component_(Arc::new(SwaComponentForTest));
    tc.arena.node_mut(tc.arena.resolve(leaf)).values[SWA.idx()].value =
        Some(Tensor::from_slice(&[0i64]));
    let _ = tc.arena.take_device_value(tc.arena.resolve(leaf), FULL);
    tc.sanity_check(&[], &[]);
}

#[test]
#[should_panic(expected = "host present but Full.host_value=None")]
fn sanity_check_detects_aux_host_without_full_host() {
    let mut tc = sane_tree();
    let leaf = tc
        .match_prefix(&match_params(&vec![1, 2, 9]))
        .best_match_node_id;
    tc.register_component_(Arc::new(SwaComponentForTest));
    tc.arena
        .node_mut(tc.arena.resolve(leaf))
        .state_mut_(ValueSlotIdx::host(SWA))
        .value = Some(Tensor::from_slice(&[0i64]));
    tc.sanity_check(&[], &[]);
}

#[test]
#[should_panic(expected = "backed up but parent")]
fn sanity_check_detects_an_unbacked_parent_prefix() {
    let mut tc = sane_tree();
    let leaf = tc
        .match_prefix(&match_params(&vec![1, 2, 9]))
        .best_match_node_id;
    tc.arena
        .node_mut(tc.arena.resolve(leaf))
        .state_mut_(ValueSlotIdx::host(FULL))
        .value = Some(Tensor::from_slice(&[30i64]));
    tc.sanity_check(&[], &[]);
}

#[test]
fn sanity_check_accepts_a_write_back_child_backed_up_before_its_parent() {
    // Write-back backs up leaf-first, so an unbacked parent is legal.
    let params = CacheInitParams {
        is_write_back: true,
        ..Default::default()
    };
    let mut tc: UnifiedTreeCore<Vec<i64>> = UnifiedTreeCore::new(params, vec![FULL]);
    tc.insert(&insert_params(&vec![1, 2, 3], &[10, 11, 12]));
    tc.insert(&insert_params(&vec![1, 2, 3, 4, 5], &[20, 21, 22, 13, 14]));
    let leaf = tc
        .match_prefix(&match_params(&vec![1, 2, 3, 4, 5]))
        .best_match_node_id;
    tc.arena
        .node_mut(tc.arena.resolve(leaf))
        .state_mut_(ValueSlotIdx::host(FULL))
        .value = Some(Tensor::from_slice(&[13i64, 14]));
    // Register the host value set directly by the test.
    tc.update_full_coexisting_host_tracking_(tc.arena.resolve(leaf));
    tc.sanity_check(&[], &[]);
}

#[test]
#[should_panic(expected = "full_lock=0 < Swa_lock=5")]
fn sanity_check_detects_an_aux_lock_above_full() {
    let mut tc = sane_tree();
    let leaf = tc
        .match_prefix(&match_params(&vec![1, 2, 9]))
        .best_match_node_id;
    tc.register_component_(Arc::new(SwaComponentForTest));
    tc.arena.node_mut(tc.arena.resolve(leaf)).values[SWA.idx()].lock_ref = 5;
    tc.sanity_check(&[], &[]);
}

#[test]
#[should_panic(expected = "H-leaf missing")]
fn sanity_check_detects_a_missing_host_leaf() {
    let mut tc = sane_tree();
    let leaf = tc
        .match_prefix(&match_params(&vec![1, 2, 9]))
        .best_match_node_id;
    let parent = tc.arena.node(tc.arena.resolve(leaf)).parent();
    let _ = tc.arena.take_device_value(tc.arena.resolve(leaf), FULL);
    tc.arena
        .node_mut(tc.arena.resolve(leaf))
        .state_mut_(ValueSlotIdx::host(FULL))
        .value = Some(Tensor::from_slice(&[30i64]));
    tc.arena
        .node_mut(parent)
        .state_mut_(ValueSlotIdx::host(FULL))
        .value = Some(Tensor::from_slice(&[10i64, 11]));
    tc.sanity_check(&[], &[]);
}

#[test]
#[should_panic(expected = "H-leaf extra")]
fn sanity_check_detects_an_extra_host_leaf() {
    let mut tc = sane_tree();
    let leaf = tc
        .match_prefix(&match_params(&vec![1, 2, 9]))
        .best_match_node_id;
    tc.evictable_host_leaves.add(tc.arena.resolve(leaf));
    tc.sanity_check(&[], &[]);
}

#[test]
#[should_panic(expected = "in both sets")]
fn sanity_check_detects_a_leaf_in_both_sets() {
    let mut tc = sane_tree();
    let leaf = tc
        .match_prefix(&match_params(&vec![1, 2, 9]))
        .best_match_node_id;
    tc.evictable_host_leaves.add(tc.arena.resolve(leaf));
    tc.sanity_check(&[], &[]);
}

#[test]
#[should_panic(expected = "stale nodes in device_leaves")]
fn sanity_check_detects_a_stale_device_leaf() {
    let mut tc = sane_tree();
    tc.evictable_device_leaves.add(NodeIdx_(999));
    tc.sanity_check(&[], &[]);
}

#[test]
#[should_panic(expected = "stale nodes in host_leaves")]
fn sanity_check_detects_a_stale_host_leaf() {
    let mut tc = sane_tree();
    tc.evictable_host_leaves.add(NodeIdx_(999));
    tc.sanity_check(&[], &[]);
}

#[test]
#[should_panic(expected = "Full host LRU not empty")]
fn sanity_check_detects_full_host_lru_pollution() {
    let mut tc = sane_tree();
    tc.host_lru_list_mut(FULL).insert_mru(NodeIdx_(0));
    tc.sanity_check(&[], &[]);
}

#[test]
#[should_panic(expected = "host LRU mismatch at node")]
fn sanity_check_detects_an_aux_host_lru_mismatch() {
    let mut tc = sane_tree();
    let leaf = tc
        .match_prefix(&match_params(&vec![1, 2, 9]))
        .best_match_node_id;
    tc.register_component_(Arc::new(SwaComponentForTest));
    let leaf_idx2 = tc.arena.resolve(leaf);
    tc.host_lru_list_mut(SWA).insert_mru(leaf_idx2);
    tc.sanity_check(&[], &[]);
}

#[test]
#[should_panic(expected = "in both device and host LRU")]
fn sanity_check_detects_an_aux_node_in_both_lrus() {
    let mut tc = sane_tree();
    let leaf = tc
        .match_prefix(&match_params(&vec![1, 2, 9]))
        .best_match_node_id;
    tc.register_component_(Arc::new(SwaComponentForTest));
    tc.arena.node_mut(tc.arena.resolve(leaf)).values[SWA.idx()].value =
        Some(Tensor::from_slice(&[0i64, 0, 0]));
    let leaf_idx = tc.arena.resolve(leaf);
    tc.device_lru_list_mut(SWA).insert_mru(leaf_idx);
    tc.host_lru_list_mut(SWA).insert_mru(leaf_idx);
    tc.sanity_check(&[], &[]);
}

#[test]
#[should_panic(expected = "device LRU: tree=0 != lru=1")]
fn sanity_check_detects_device_lru_length_drift() {
    let mut tc = sane_tree();
    tc.register_component_(Arc::new(SwaComponentForTest));
    tc.device_lru_list_mut(SWA).insert_mru(NodeIdx_(999));
    tc.sanity_check(&[], &[]);
}

#[test]
#[should_panic(expected = "host LRU: tree=0 != lru=1")]
fn sanity_check_detects_host_lru_length_drift() {
    let mut tc = sane_tree();
    tc.register_component_(Arc::new(SwaComponentForTest));
    tc.host_lru_list_mut(SWA).insert_mru(NodeIdx_(999));
    tc.sanity_check(&[], &[]);
}

#[test]
#[should_panic(expected = "[device][Swa] list=0 != len=1")]
fn sanity_check_wires_the_device_list_integrity_walk() {
    let mut tc = sane_tree();
    tc.register_component_(Arc::new(SwaComponentForTest));
    tc.device_lru_list_mut(SWA).bump_len_for_test();
    tc.sanity_check(&[], &[]);
}

#[test]
#[should_panic(expected = "[host][Swa] list=0 != len=1")]
fn sanity_check_wires_the_host_list_integrity_walk() {
    let mut tc = sane_tree();
    tc.register_component_(Arc::new(SwaComponentForTest));
    tc.host_lru_list_mut(SWA).bump_len_for_test();
    tc.sanity_check(&[], &[]);
}

#[test]
#[should_panic(expected = "[device][Full] list=0 != len=1")]
fn sanity_check_wires_the_full_device_list_integrity_walk() {
    let mut tc = sane_tree();
    tc.device_lru_list_mut(FULL).bump_len_for_test();
    tc.sanity_check(&[], &[]);
}

#[test]
#[should_panic(expected = "protected=999")]
fn sanity_check_detects_protected_size_drift() {
    let mut tc = sane_tree();
    tc.component_state_mut(FULL).protected_size = 999;
    tc.sanity_check(&[], &[]);
}

#[test]
#[should_panic(expected = "write_through node 9 lock_ref=0")]
fn sanity_check_detects_an_unlocked_write_through() {
    let mut tc = sane_tree();
    let leaf = tc
        .match_prefix(&match_params(&vec![1, 2, 9]))
        .best_match_node_id;
    tc.sanity_check(&[(9, leaf)], &[]);
}

#[test]
#[should_panic(expected = "load_back node 10 not in tree")]
fn sanity_check_detects_an_untracked_load_back() {
    let tc = sane_tree();
    tc.sanity_check(&[], &[(10, 999)]);
}

#[test]
fn match_prefix_on_an_unknown_namespace_allocates_nothing() {
    let mut tc = core();
    tc.insert(&insert_params(&vec![1, 2, 3], &[10, 11, 12]));
    let arena_len = tc.arena.len();
    let result = tc.match_prefix(&MatchPrefixParams {
        key: &vec![1, 2, 3],
        extra_key: Some("ghost"),
    });
    assert_eq!(result.device_indices.numel(), 0);
    assert_eq!(tc.arena.len(), arena_len);
    // The empty result anchors at the default root (the namespace has no root).
    let default_root = tc.arena.root();
    assert_eq!(result.best_match_node_id, tc.arena.node(default_root).id);
}

#[test]
fn refresh_dispatches_fire_per_walk_phase_in_a_namespace() {
    let mut tc = core();
    let recorder = Arc::new(RecordingComponentForTest::default());
    tc.register_component_(recorder.clone());
    tc.insert(&InsertParams {
        extra_key: Some("chat"),
        ..insert_params(&vec![7, 8], &[40, 41])
    });
    // The deeper insert walks down through the existing [7,8] node.
    tc.insert(&InsertParams {
        extra_key: Some("chat"),
        ..insert_params(&vec![7, 8, 9], &[40, 41, 42])
    });
    let leaf = tc
        .match_prefix(&MatchPrefixParams {
            key: &vec![7, 8],
            extra_key: Some("chat"),
        })
        .best_match_node_id;
    let refreshes = recorder.refreshes.lock().unwrap();
    assert!(!refreshes.is_empty());
    assert!(
        refreshes
            .iter()
            .any(|&(phase, node)| phase == LRURefreshPhase::Walkdown
                && node == tc.arena.resolve(leaf))
    );
    assert!(refreshes.iter().any(
        |&(phase, node)| phase == LRURefreshPhase::InsertEnd && node == tc.arena.resolve(leaf)
    ));
    assert!(
        refreshes
            .iter()
            .any(|&(phase, node)| phase == LRURefreshPhase::MatchEnd
                && node == tc.arena.resolve(leaf))
    );
}

#[test]
#[should_panic(expected = "orphaned live nodes")]
fn sanity_check_detects_an_orphaned_node() {
    let mut tc = sane_tree();
    let root = tc.arena.root();
    // A parented node missing from its parent's child map is unreachable.
    tc.new_node_(
        /* key = */ vec![99],
        root,
        /* priority = */ 0,
        /* hit_count = */ 0,
        /* creation_counter = */ None,
        /* extra_key = */ None,
    );
    tc.sanity_check(&[], &[]);
}

#[test]
#[should_panic(expected = "not mapped under its own child key")]
fn sanity_check_detects_a_reverse_map_mismatch() {
    let mut tc = sane_tree();
    let leaf = tc
        .match_prefix(&match_params(&vec![1, 2, 9]))
        .best_match_node_id;
    let leaf_idx3 = tc.arena.resolve(leaf);
    let parent = tc.arena.node(leaf_idx3).parent();
    let key = tc.arena.node(leaf_idx3).key.child_key(1);
    let parent_node = tc.arena.node_mut(parent);
    parent_node.children.remove(&(None, key));
    parent_node.children.insert((None, vec![99]), leaf_idx3);
    tc.sanity_check(&[], &[]);
}

#[test]
#[should_panic(expected = "Full value length")]
fn sanity_check_detects_a_value_length_mismatch() {
    let mut tc = sane_tree();
    let leaf = tc
        .match_prefix(&match_params(&vec![1, 2, 9]))
        .best_match_node_id;
    tc.arena.node_mut(tc.arena.resolve(leaf)).values[FULL.idx()].value =
        Some(Tensor::from_slice(&[7i64, 8, 9, 10]));
    tc.sanity_check(&[], &[]);
}

#[test]
#[should_panic(expected = "Full host value length")]
fn sanity_check_detects_a_host_value_length_mismatch() {
    let mut tc = sane_tree();
    let leaf = tc
        .match_prefix(&match_params(&vec![1, 2, 9]))
        .best_match_node_id;
    let parent = tc.arena.node(tc.arena.resolve(leaf)).parent();
    tc.arena
        .node_mut(parent)
        .state_mut_(ValueSlotIdx::host(FULL))
        .value = Some(Tensor::from_slice(&[10i64, 11]));
    tc.arena
        .node_mut(tc.arena.resolve(leaf))
        .state_mut_(ValueSlotIdx::host(FULL))
        .value = Some(Tensor::from_slice(&[7i64, 8, 9, 10]));
    tc.sanity_check(&[], &[]);
}

#[test]
#[should_panic(expected = "has an empty key")]
fn sanity_check_detects_an_empty_key() {
    let mut tc = sane_tree();
    let leaf = tc
        .match_prefix(&match_params(&vec![1, 2, 9]))
        .best_match_node_id;
    tc.arena.node_mut(tc.arena.resolve(leaf)).key = vec![];
    tc.sanity_check(&[], &[]);
}

#[test]
#[should_panic(expected = "key is not page-aligned")]
fn sanity_check_detects_an_unaligned_key() {
    let mut tc = page2_core();
    tc.insert(&insert_params(&vec![1, 2, 3, 4], &[10, 11, 12, 13]));
    let leaf = tc
        .match_prefix(&match_params(&vec![1, 2, 3, 4]))
        .best_match_node_id;
    tc.arena.node_mut(tc.arena.resolve(leaf)).key = vec![1];
    tc.sanity_check(&[], &[]);
}

// Corrupt the [1,2,9] leaf's child map to point back at its own parent.
fn cyclic_child_map_tree() -> UnifiedTreeCore<Vec<i64>> {
    let mut tc = sane_tree();
    let leaf = tc
        .match_prefix(&match_params(&vec![1, 2, 9]))
        .best_match_node_id;
    let parent = tc.arena.node(tc.arena.resolve(leaf)).parent();
    tc.arena
        .node_mut(tc.arena.resolve(leaf))
        .children
        .insert((None, vec![50]), parent);
    tc
}

#[test]
fn collect_all_nodes_terminates_on_a_cyclic_child_map() {
    let tc = cyclic_child_map_tree();
    let nodes = tc.collect_all_nodes_();
    assert_eq!(nodes.len(), tc.arena.len());
}

#[test]
#[should_panic(expected = "[Tree] child")]
fn sanity_check_reports_a_cyclic_child_map_without_hanging() {
    let tc = cyclic_child_map_tree();
    tc.sanity_check(&[], &[]);
}

#[test]
#[should_panic(expected = "host LRU mismatch")]
fn sanity_check_detects_a_host_locked_value_missing_from_the_lru() {
    let mut tc = sane_tree();
    let leaf = tc
        .match_prefix(&match_params(&vec![1, 2, 9]))
        .best_match_node_id;
    let parent = tc.arena.node(tc.arena.resolve(leaf)).parent();
    tc.register_component_(Arc::new(SwaComponentForTest));
    // The arena was built Full-only; give the root the stub's lock too.
    tc.arena.node_mut(tc.arena.root()).values[SWA.idx()].lock_ref = 1;
    tc.arena
        .node_mut(parent)
        .state_mut_(ValueSlotIdx::host(FULL))
        .value = Some(Tensor::from_slice(&[10i64, 11]));
    let leaf_node = tc.arena.node_mut(tc.arena.resolve(leaf));
    leaf_node.state_mut_(ValueSlotIdx::host(FULL)).value = Some(Tensor::from_slice(&[30i64]));
    leaf_node.state_mut_(ValueSlotIdx::host(SWA)).value = Some(Tensor::from_slice(&[30i64]));
    leaf_node.state_mut_(ValueSlotIdx::host(SWA)).lock_ref = 1;
    tc.sanity_check(&[], &[]);
}

// A backed-up leaf whose unlocked Swa value is host-only (no device value).
fn host_only_aux_leaf(tc: &mut UnifiedTreeCore<Vec<i64>>) -> NodeIdx_ {
    let leaf = tc
        .match_prefix(&match_params(&vec![1, 2, 9]))
        .best_match_node_id;
    let parent = tc.arena.node(tc.arena.resolve(leaf)).parent();
    tc.register_component_(Arc::new(SwaComponentForTest));
    // The arena was built Full-only; give the root the stub's lock too.
    tc.arena.node_mut(tc.arena.root()).values[SWA.idx()].lock_ref = 1;
    tc.arena
        .node_mut(parent)
        .state_mut_(ValueSlotIdx::host(FULL))
        .value = Some(Tensor::from_slice(&[10i64, 11]));
    let leaf_node = tc.arena.node_mut(tc.arena.resolve(leaf));
    leaf_node.state_mut_(ValueSlotIdx::host(FULL)).value = Some(Tensor::from_slice(&[30i64]));
    leaf_node.state_mut_(ValueSlotIdx::host(SWA)).value = Some(Tensor::from_slice(&[30i64]));
    // Register the host values set directly by the test.
    tc.update_full_coexisting_host_tracking_(parent);
    tc.update_full_coexisting_host_tracking_(tc.arena.resolve(leaf));
    tc.arena.resolve(leaf)
}

#[test]
fn sanity_check_accepts_an_unlocked_host_only_value_in_the_lru() {
    let mut tc = sane_tree();
    let leaf = host_only_aux_leaf(&mut tc);
    tc.host_lru_list_mut(SWA).insert_mru(leaf);
    tc.sanity_check(&[], &[]);
}

#[test]
#[should_panic(expected = "host LRU mismatch")]
fn sanity_check_detects_a_host_only_value_missing_from_the_lru() {
    let mut tc = sane_tree();
    host_only_aux_leaf(&mut tc);
    tc.sanity_check(&[], &[]);
}

#[test]
#[should_panic(expected = "EvictLayer::All is not a single layer")]
fn for_each_component_lru_rejects_the_all_layer() {
    let mut tc = core();
    let root = tc.arena.root();
    let a = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![1],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    tc.register_component_(Arc::new(SwaComponentForTest));
    tc.for_each_component_lru_(
        a,
        &mut |_, _| {},
        EvictLayer::All,
        /* skip_existing = */ false,
    );
}

#[test]
fn insert_unevicts_a_tombstoned_deep_node() {
    let mut tc = core();
    tc.insert(&insert_params(&vec![1, 2, 3], &[10, 11, 12]));
    tc.insert(&insert_params(&vec![1, 2, 3, 4, 5], &[20, 21, 22, 13, 14]));
    let leaf = tc
        .match_prefix(&match_params(&vec![1, 2, 3, 4, 5]))
        .best_match_node_id;
    let _ = tc.arena.take_device_value(tc.arena.resolve(leaf), FULL);
    tc.component_state_mut(FULL).evictable_size = 3;
    tc.evictable_device_leaves.discard(tc.arena.resolve(leaf));
    let result = tc.insert(&insert_params(&vec![1, 2, 3, 4, 5], &[30, 31, 32, 33, 34]));
    assert_eq!(result.prefix_len, 5);
    // The revived leaf takes its own span of the fresh KV, not the key head.
    assert!(
        tc.arena
            .device_value(tc.arena.resolve(leaf), FULL)
            .equal(&Tensor::from_slice(&[33i64, 34]))
    );
    let [CacheAction::FreeDeviceKV(freed)] = result.cache_actions.as_slice() else {
        panic!(
            "expected one FreeDeviceKV action, got {:?}",
            action_kinds(&result.cache_actions)
        );
    };
    assert!(freed[0].equal(&Tensor::from_slice(&[30i64, 31, 32])));
}

#[test]
fn insert_ragged_key_onto_an_existing_prefix_page_size_two() {
    let mut tc = page2_core();
    tc.insert(&insert_params(&vec![1, 2], &[10, 11]));
    let result = tc.insert(&insert_params(&vec![1, 2, 3], &[20, 21, 22]));
    assert_eq!(result.prefix_len, 2);
    // The ragged tail never enters: no new leaf, the aligned span is duplicate.
    assert_eq!(tc.arena.len(), 2);
    let [CacheAction::FreeDeviceKV(freed)] = result.cache_actions.as_slice() else {
        panic!(
            "expected one FreeDeviceKV action, got {:?}",
            action_kinds(&result.cache_actions)
        );
    };
    assert!(freed[0].equal(&Tensor::from_slice(&[20i64, 21])));
}

#[test]
fn insert_ragged_key_traverses_a_node_before_the_tail_page_size_two() {
    let mut tc = page2_core();
    tc.insert(&insert_params(&vec![1, 2], &[10, 11]));
    tc.insert(&insert_params(&vec![1, 2, 3, 4], &[20, 21, 12, 13]));
    let result = tc.insert(&insert_params(&vec![1, 2, 3, 4, 5], &[30, 31, 32, 33, 34]));
    assert_eq!(result.prefix_len, 4);
    assert_eq!(tc.arena.len(), 3);
    let [
        CacheAction::FreeDeviceKV(freed_head),
        CacheAction::FreeDeviceKV(freed_tail),
    ] = result.cache_actions.as_slice()
    else {
        panic!(
            "expected one FreeDeviceKV per walked node, got {:?}",
            action_kinds(&result.cache_actions)
        );
    };
    assert!(freed_head[0].equal(&Tensor::from_slice(&[30i64, 31])));
    assert!(freed_tail[0].equal(&Tensor::from_slice(&[32i64, 33])));
}

#[test]
fn match_ragged_query_stops_at_the_aligned_window_page_size_two() {
    let mut tc = page2_core();
    tc.insert(&insert_params(&vec![1, 2, 3, 4], &[10, 11, 12, 13]));
    // The node key runs past the query's aligned window; the ragged atom matches too.
    let result = tc.match_prefix(&match_params(&vec![1, 2, 3]));
    assert!(
        result
            .device_indices
            .equal(&Tensor::from_slice(&[10i64, 11]))
    );
}

#[test]
fn begin_insert_empty_key_completes_in_one_step() {
    let mut tc = core();
    let step = tc.begin_insert(&insert_params(&vec![], &[]));
    assert!(step.actions.is_empty());
    let result = step.result.expect("an empty insert completes immediately");
    assert_eq!(result.prefix_len, 0);
    assert!(result.mamba_exist);
    assert!(!tc.has_ongoing_insert());
}

#[test]
fn deferrable_dup_frees_ride_the_final_step_without_suspension() {
    let mut tc = core();
    tc.insert(&insert_params(&vec![1, 2, 3], &[10, 11, 12]));
    let step = tc.begin_insert(&insert_params(&vec![1, 2, 3, 4], &[20, 21, 22, 13]));
    assert_eq!(action_kinds(&step.actions), vec!["FreeDeviceKV"]);
    let result = step
        .result
        .expect("a deferrable-only walk completes in one step");
    assert_eq!(result.prefix_len, 3);
    assert!(result.cache_actions.is_empty());
    assert!(!tc.has_ongoing_insert());
    assert!(tc.end_insert().is_empty());
}

fn suspended_walk_core() -> (UnifiedTreeCore<Vec<i64>>, NodeIdx_, InsertStepResult) {
    let mut tc: UnifiedTreeCore<Vec<i64>> = UnifiedTreeCore::new(
        CacheInitParams {
            write_through_threshold: 2,
            ..Default::default()
        },
        vec![FULL],
    );
    tc.set_hicache_enabled();
    tc.insert(&insert_params(&vec![1, 2, 3], &[10, 11, 12]));
    let a = tc
        .match_prefix(&match_params(&vec![1, 2, 3]))
        .best_match_node_id;
    let step = tc.begin_insert(&insert_params(&vec![1, 2, 3, 4, 5], &[20, 21, 22, 13, 14]));
    let a_idx = tc.arena.resolve(a);
    (tc, a_idx, step)
}

#[test]
fn walk_backup_crossing_suspends_then_resume_completes() {
    let (mut tc, a, step) = suspended_walk_core();
    assert!(step.result.is_none());
    assert!(tc.has_ongoing_insert());
    assert_eq!(
        action_kinds(&step.actions),
        vec!["FreeDeviceKV", "BackupKV"]
    );
    let CacheAction::BackupKV(backup) = &step.actions[1] else {
        unreachable!();
    };
    assert_eq!(backup.node_ids, vec![tc.arena.node(a).id]);

    let done = tc.resume_insert();
    assert!(done.actions.is_empty());
    let result = done.result.expect("the resumed walk completes");
    assert_eq!(result.prefix_len, 3);
    assert!(!tc.has_ongoing_insert());
    assert!(tc.end_insert().is_empty());
    tc.sanity_check(&[], &[]);
}

#[test]
#[should_panic(expected = "concurrent insert walks")]
fn begin_insert_rejects_a_concurrent_walk() {
    let (mut tc, _, step) = suspended_walk_core();
    assert!(step.result.is_none());
    let _ = tc.begin_insert(&insert_params(&vec![9], &[90]));
}

#[test]
#[should_panic(expected = "no in-flight insert")]
fn resume_insert_without_a_walk_panics() {
    let mut tc = core();
    let _ = tc.resume_insert();
}

#[test]
fn end_insert_aborts_the_suspended_walk() {
    let (mut tc, _, step) = suspended_walk_core();
    assert!(step.result.is_none());
    // The barrier flushed everything pending; the abort drain is empty.
    assert!(tc.end_insert().is_empty());
    assert!(!tc.has_ongoing_insert());
    assert!(tc.end_insert().is_empty());
    // The single-flight slot is clear: a fresh insert starts normally.
    let result = tc.insert(&insert_params(&vec![9], &[90]));
    assert_eq!(result.prefix_len, 0);
}

#[test]
fn resume_insert_completes_after_an_on_path_host_leaf_is_evicted() {
    let mut tc: UnifiedTreeCore<Vec<i64>> = UnifiedTreeCore::new(
        CacheInitParams {
            write_through_threshold: 2,
            ..Default::default()
        },
        vec![FULL],
    );
    tc.set_hicache_enabled();
    tc.insert(&insert_params(&vec![1, 2, 3, 4], &[10, 11, 12, 13]));
    let top = tc
        .match_prefix(&match_params(&vec![1, 2, 3, 4]))
        .best_match_node_id;
    let root = tc.arena.root();
    let h_leaf = tc
        .insert_host(
            tc.arena.node(root).id,
            /* extra_key = */ None,
            vec![1, 2, 3, 4, 5, 6, 7, 8],
            Tensor::from_slice(&[100i64, 101, 102, 103, 104, 105, 106, 107]),
            vec!["h0", "h1", "h2", "h3", "h4", "h5", "h6", "h7"]
                .into_iter()
                .map(String::from)
                .collect(),
        )
        .inserted_host_node
        .unwrap();
    let step = tc.begin_insert(&insert_params(
        &vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        &[20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
    ));
    assert!(step.result.is_none());
    assert_eq!(
        action_kinds(&step.actions),
        vec!["FreeDeviceKV", "BackupKV"]
    );
    // The barrier's backup host-evicts the on-path H-leaf before committing.
    let mut tracker = HashMap::from([(FULL, 0)]);
    let (mut df, mut hf) = (HashMap::new(), HashMap::new());
    tc.evict_host_leaf_(tc.arena.resolve(h_leaf), &mut tracker, &mut df, &mut hf);
    assert_eq!(tracker[&FULL], 4);
    tc.commit_backup(
        top,
        Tensor::from_slice(&[100i64, 101, 102, 103]),
        HashMap::new(),
    );
    let done = tc.resume_insert();
    let result = done.result.expect("the resumed walk completes");
    assert_eq!(result.prefix_len, 4);
    assert!(!tc.has_ongoing_insert());
    assert!(tc.arena.try_resolve(h_leaf).is_none());
    // The recreated suffix is top's single child, spanning the whole gap.
    let top_idx = tc.arena.resolve(top);
    assert_eq!(tc.arena.node(top_idx).children.len(), 1);
    let suffix = *tc.arena.node(top_idx).children.values().next().unwrap();
    assert_eq!(tc.arena.node(suffix).key, vec![5, 6, 7, 8, 9, 10, 11, 12]);
    tc.sanity_check(&[], &[]);
}

#[test]
fn aborted_barrier_crossing_refires_on_the_next_insert() {
    let (mut tc, a, step) = suspended_walk_core();
    assert!(step.result.is_none());
    assert!(tc.end_insert().is_empty());
    assert!(!tc.has_ongoing_insert());
    // The abort never committed the backup, so the same crossing fires again.
    let step = tc.begin_insert(&insert_params(&vec![1, 2, 3, 4, 5], &[20, 21, 22, 13, 14]));
    assert!(step.result.is_none());
    let backups: Vec<_> = step
        .actions
        .iter()
        .filter_map(|action| match action {
            CacheAction::BackupKV(backup) => Some(backup.node_ids.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(backups, vec![vec![tc.arena.node(a).id]]);
    tc.commit_backup(
        tc.arena.node(a).id,
        Tensor::from_slice(&[100i64, 101, 102]),
        HashMap::new(),
    );
    let done = tc.resume_insert();
    assert_eq!(
        done.result.expect("the resumed walk completes").prefix_len,
        3
    );
    tc.sanity_check(&[], &[]);
}

#[test]
fn one_insert_walk_fires_two_crossings_around_a_backuped_middle() {
    let mut tc = core();
    tc.set_hicache_enabled();
    tc.insert(&insert_params(&vec![1, 2, 3, 4], &[10, 11, 12, 13]));
    let top = tc
        .match_prefix(&match_params(&vec![1, 2, 3, 4]))
        .best_match_node_id;
    // A storage prefetch host-inserts a backuped middle below the unbacked top.
    let root = tc.arena.root();
    let middle = tc
        .insert_host(
            tc.arena.node(root).id,
            /* extra_key = */ None,
            vec![1, 2, 3, 4, 5, 6, 7, 8],
            Tensor::from_slice(&[100i64, 101, 102, 103, 104, 105, 106, 107]),
            vec!["h0", "h1", "h2", "h3", "h4", "h5", "h6", "h7"]
                .into_iter()
                .map(String::from)
                .collect(),
        )
        .inserted_host_node
        .unwrap();
    // The device insert unevicts the middle and adds the unbacked deep leaf.
    tc.insert(&insert_params(
        &vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        &[20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
    ));
    let deep = tc
        .match_prefix(&match_params(&vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]))
        .best_match_node_id;
    tc.write_through_threshold = 2;
    let result = tc.insert(&insert_params(
        &vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        &[
            40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
        ],
    ));
    // Both crossings fire in walk order; the backuped middle joins neither chain.
    let backups: Vec<_> = result
        .cache_actions
        .iter()
        .filter_map(|action| match action {
            CacheAction::BackupKV(backup) => Some(backup.node_ids.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(backups, vec![vec![top], vec![deep]]);
    assert!(tc.arena.node(tc.arena.resolve(middle)).backuped());
}

#[test]
fn full_kv_hit_length_counts_the_split_fragment() {
    let mut tc = core();
    tc.insert(&insert_params(&vec![1, 2, 3, 4], &[10, 11, 12, 13]));
    let result = tc.match_prefix(&match_params(&vec![1, 2, 99]));
    // The mid-node partial match splits the node; the fragment still counts.
    assert_eq!(result.full_kv_hit_length, 2);
    assert_eq!(result.device_indices.size()[0], 2);
}

#[test]
fn dec_evictable_size_updates_only_the_addressed_component() {
    let mut tc = core();
    tc.component_state_mut(FULL).evictable_size = 5;
    tc.component_state_mut(SWA).evictable_size = 7;
    tc.dec_evictable_size(FULL, 2);
    assert_eq!(tc.evictable_size_(FULL), 3);
    assert_eq!(tc.evictable_size_(SWA), 7);
}

#[test]
fn dec_evictable_size_to_exactly_zero() {
    let mut tc = core();
    tc.component_state_mut(FULL).evictable_size = 4;
    tc.dec_evictable_size(FULL, 4);
    assert_eq!(tc.evictable_size_(FULL), 0);
}

#[test]
#[should_panic(expected = "dec_evictable_size: Full evictable size underflow")]
fn dec_evictable_size_panics_on_underflow() {
    let mut tc = core();
    tc.component_state_mut(FULL).evictable_size = 1;
    tc.dec_evictable_size(FULL, 2);
}

#[test]
fn size_helpers_move_tokens_for_the_addressed_component() {
    let mut tc = core();
    tc.inc_evictable_size(FULL, 4);
    tc.inc_protected_size(FULL, 3);
    assert_eq!(tc.evictable_size_(FULL), 4);
    assert_eq!(tc.protected_size_(FULL), 3);
    assert_eq!(tc.evictable_size_(SWA), 0);
    assert_eq!(tc.protected_size_(SWA), 0);
    tc.dec_protected_size(FULL, 2);
    assert_eq!(tc.protected_size_(FULL), 1);
}

#[test]
#[should_panic(expected = "dec_protected_size: Full protected size underflow")]
fn dec_protected_size_panics_on_underflow() {
    let mut tc = core();
    tc.dec_protected_size(FULL, 1);
}

#[test]
fn component_state_accessors_address_the_given_component() {
    let mut tc = core();
    tc.component_state_mut(FULL).evictable_size = 5;
    assert_eq!(tc.component_state(FULL).evictable_size, 5);
    assert_eq!(tc.component_state(SWA).evictable_size, 0);
}

#[test]
fn evict_walk_lifecycle_tracks_the_bookkeeping() {
    let mut tc = core();
    tc.set_evict_device_start(FULL, /* request_cnt = */ 7);
    assert!(tc.component_state(FULL).is_evict_device_ongoing);
    assert_eq!(tc.component_state(FULL).evict_device_request_cnt, 7);
    assert_eq!(tc.component_state(FULL).evict_device_cursor, None);
    tc.set_evict_device_end(FULL);
    assert!(!tc.component_state(FULL).is_evict_device_ongoing);
}

#[test]
#[should_panic(expected = "Full device eviction already in progress")]
fn set_evict_device_start_panics_when_already_ongoing() {
    let mut tc = core();
    tc.set_evict_device_start(FULL, /* request_cnt = */ 1);
    tc.set_evict_device_start(FULL, /* request_cnt = */ 1);
}

#[test]
#[should_panic(expected = "Full device eviction not started")]
fn set_evict_device_end_panics_before_a_walk() {
    let mut tc = core();
    tc.set_evict_device_end(FULL);
}

#[test]
fn lru_list_accessors_address_the_given_component_lists() {
    let mut tc = core();
    let root = tc.arena.root();
    let a = tc
        .arena
        .alloc_child(
            root,
            /* key = */ vec![1],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    tc.device_lru_list_mut(FULL).insert_mru(a);
    assert!(tc.device_lru_list(FULL).in_list(Some(a)));
    assert!(!tc.host_lru_list(FULL).in_list(Some(a)));
    // A disabled component's list exists but stays empty and independent.
    assert!(!tc.device_lru_list(SWA).in_list(Some(a)));
    assert_eq!(tc.device_lru_list(SWA).len(), 0);
    // The mutable host accessor addresses the same per-component list.
    tc.host_lru_list_mut(SWA).insert_mru(a);
    assert!(tc.host_lru_list(SWA).in_list(Some(a)));
    assert!(!tc.host_lru_list(FULL).in_list(Some(a)));
}

#[test]
fn reset_invalidates_every_prior_handle() {
    let mut tc = core();
    tc.insert(&insert_params(&vec![1, 2], &[10, 11]));
    let old_root = tc.root_node_handle(/* extra_key = */ None);
    let old_leaf = tc
        .match_prefix(&match_params(&vec![1, 2]))
        .best_match_node_id;
    tc.reset();
    // Handles are never re-minted, so pre-reset ones miss instead of aliasing.
    assert!(tc.arena.try_resolve(old_root).is_none());
    assert!(tc.arena.try_resolve(old_leaf).is_none());
    let new_root = tc.root_node_handle(/* extra_key = */ None);
    assert_ne!(new_root, old_root);
    assert_eq!(tc.arena.resolve(new_root), tc.arena.root());
    tc.insert(&insert_params(&vec![1, 2], &[10, 11]));
    assert_eq!(
        tc.match_prefix(&match_params(&vec![1, 2]))
            .device_indices
            .size()[0],
        2
    );
}

#[test]
#[should_panic(expected = "is not enabled")]
fn component_has_host_value_only_panics_on_a_disabled_component() {
    let tc = core();
    let root = tc.root_node_handle(/* extra_key = */ None);
    tc.component_has_host_value_only(root, SWA);
}

#[test]
fn reset_clears_an_ongoing_evict_walk() {
    let mut tc = core();
    tc.insert(&insert_params(&vec![1], &[10]));
    tc.evict_device_start(FULL, /* request_cnt = */ 4);
    assert!(tc.component_state(FULL).is_evict_device_ongoing);
    tc.reset();
    // Reset drops the walk bookkeeping with the tree; a fresh walk starts clean.
    assert!(!tc.component_state(FULL).is_evict_device_ongoing);
    tc.insert(&insert_params(&vec![1], &[10]));
    tc.evict_device_start(FULL, /* request_cnt = */ 4);
    tc.evict_device_end(FULL);
}

// Deterministic xorshift so the sequence test needs no rand dependency.
fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

// Fresh-KV insert params for the sequence test; donates a mamba slot when asked.
fn sequence_insert_params<'k>(
    key: &'k Vec<i64>,
    prev_prefix_len: usize,
    kv_next: &mut i64,
    mamba_next: &mut i64,
    mamba: bool,
) -> InsertParams<'k, Vec<i64>> {
    let depth = key.len();
    let kv: Vec<i64> = (0..depth).map(|i| *kv_next + i as i64).collect();
    *kv_next += depth as i64;
    let mamba_value = mamba.then(|| {
        *mamba_next += 1;
        Tensor::from_slice(&[*mamba_next])
    });
    InsertParams {
        key,
        extra_key: None,
        value: Tensor::from_slice(&kv),
        mamba_value,
        prev_prefix_len,
        swa_evicted_seqlen: 0,
        chunked: false,
        priority: 0,
    }
}

// Randomized op sequence with a per-step sanity_check; `page` sizes the keys,
// `mamba` donates one state slot per insert.
fn run_random_op_sequence(mut tc: UnifiedTreeCore<Vec<i64>>, page: usize, mamba: bool) {
    let cts: Vec<ComponentType> = tc.components.iter().map(|c| c.component_type()).collect();
    let mut rng = 0x9E3779B97F4A7C15u64;
    let mut kv_next = 1000i64;
    let mut mamba_next = 1i64;
    for step in 0..400 {
        let depth = page * (1 + (xorshift64(&mut rng) % 3) as usize);
        let key: Vec<i64> = (0..depth)
            .map(|_| 1 + (xorshift64(&mut rng) % 5) as i64)
            .collect();
        match xorshift64(&mut rng) % 4 {
            0 => {
                tc.insert(&sequence_insert_params(
                    &key,
                    0,
                    &mut kv_next,
                    &mut mamba_next,
                    mamba,
                ));
            }
            1 => {
                // Consecutive matches of the same key agree (idempotency).
                let first = tc.match_prefix(&match_params(&key)).device_indices.numel();
                let second = tc.match_prefix(&match_params(&key)).device_indices.numel();
                assert_eq!(first, second);
            }
            2 => {
                // Balanced lock round trip on whatever the key matches.
                let anchor = tc.match_prefix(&match_params(&key)).best_match_node_id;
                let lock = tc.inc_lock_ref(anchor);
                let params = DecLockRefParams {
                    swa_uuid_for_lock: lock.swa_uuid_for_lock,
                    swa_uuid_for_host_lock: lock.swa_uuid_for_host_lock,
                    skip_lock_node_ids: lock.skip_lock_node_ids,
                };
                tc.dec_lock_ref(anchor, Some(&params), /* skip_swa = */ false);
            }
            _ => {
                // Insert-while-locked churn, the cache_finished_req shape.
                let matched = tc.match_prefix(&match_params(&key));
                let anchor = matched.best_match_node_id;
                let matched_len = matched.device_indices.numel() as usize;
                let lock = tc.inc_lock_ref(anchor);
                tc.insert(&sequence_insert_params(
                    &key,
                    matched_len,
                    &mut kv_next,
                    &mut mamba_next,
                    mamba,
                ));
                let params = DecLockRefParams {
                    swa_uuid_for_lock: lock.swa_uuid_for_lock,
                    swa_uuid_for_host_lock: lock.swa_uuid_for_host_lock,
                    skip_lock_node_ids: lock.skip_lock_node_ids,
                };
                tc.dec_lock_ref(anchor, Some(&params), /* skip_swa = */ false);
            }
        }
        if step % 8 == 7 {
            for &ct in &cts {
                let mut tracker: HashMap<ComponentType, usize> =
                    cts.iter().map(|&c| (c, 0)).collect();
                let mut device_frees = HashMap::new();
                let mut host_frees = HashMap::new();
                tc.evict_device_start(ct, /* request_cnt = */ 3);
                loop {
                    let (next, step_result) = tc.evict_device_next_node(ct, &tracker);
                    accumulate_step(
                        step_result,
                        &mut tracker,
                        &mut device_frees,
                        &mut host_frees,
                    );
                    let Some(leaf) = next else { break };
                    let (_, evict_result) =
                        tc.evict_device_leaf(leaf, /* is_write_back = */ false);
                    accumulate_step(
                        evict_result,
                        &mut tracker,
                        &mut device_frees,
                        &mut host_frees,
                    );
                }
                tc.evict_device_end(ct);
            }
        }
        tc.sanity_check(&[], &[]);
    }
}

#[test]
fn random_op_sequence_holds_the_sanity_invariants() {
    run_random_op_sequence(
        swa_match_core(/* window = */ 4),
        1,
        /* mamba = */ false,
    );
}

#[test]
fn random_op_sequence_holds_on_a_mamba_core() {
    let tc = UnifiedTreeCore::<Vec<i64>>::new(
        CacheInitParams {
            page_size: 1,
            mamba_cache_chunk_size: Some(256),
            ..CacheInitParams::default()
        },
        vec![FULL, MAMBA],
    );
    run_random_op_sequence(tc, 1, /* mamba = */ true);
}

#[test]
fn random_op_sequence_holds_on_a_paged_swa_core() {
    let tc = UnifiedTreeCore::<Vec<i64>>::new(
        CacheInitParams {
            page_size: 2,
            swa_sliding_window_size: Some(8),
            ..CacheInitParams::default()
        },
        vec![FULL, SWA],
    );
    run_random_op_sequence(tc, 2, /* mamba = */ false);
}

// Drain every evictable FULL device leaf, as the orchestrator's evict loop does.
fn drain_full_device(tc: &mut UnifiedTreeCore<Vec<i64>>) {
    let mut tracker = HashMap::from([(FULL, 0)]);
    let (mut df, mut hf) = (HashMap::new(), HashMap::new());
    tc.evict_device_start(FULL, /* request_cnt = */ 1_000_000);
    loop {
        let (leaf, step) = tc.evict_device_next_node(FULL, &tracker);
        accumulate_step(step, &mut tracker, &mut df, &mut hf);
        let Some(leaf) = leaf else { break };
        let (_, step) = tc.evict_device_leaf(leaf, /* is_write_back = */ false);
        accumulate_step(step, &mut tracker, &mut df, &mut hf);
    }
    tc.evict_device_end(FULL);
}

#[test]
fn an_emptied_namespace_leaves_nothing_behind() {
    let mut tc = core();
    tc.insert(&InsertParams {
        extra_key: Some("salted"),
        ..insert_params(&vec![1, 2], &[10, 11])
    });
    let top = tc
        .match_prefix(&MatchPrefixParams {
            extra_key: Some("salted"),
            ..match_params(&vec![1, 2])
        })
        .best_match_node_id;
    drain_full_device(&mut tc);
    // The namespace's nodes evict like any others; its edge map drops with them.
    assert!(!tc.arena.namespace_exists(Some("salted")));
    assert!(tc.arena.try_resolve(top).is_none());
    assert_eq!(tc.arena.len(), 1);
    tc.sanity_check(&[], &[]);
    // A later insert respins the namespace from scratch.
    tc.insert(&InsertParams {
        extra_key: Some("salted"),
        ..insert_params(&vec![1, 2], &[10, 11])
    });
    assert!(tc.arena.namespace_exists(Some("salted")));
    tc.sanity_check(&[], &[]);
}

#[test]
fn namespaces_do_not_accumulate_across_salts() {
    let mut tc = core();
    for salt in 0..64 {
        let salt = format!("session-{salt}");
        tc.insert(&InsertParams {
            extra_key: Some(&salt),
            ..insert_params(&vec![1, 2], &[10, 11])
        });
        drain_full_device(&mut tc);
    }
    assert_eq!(tc.arena.len(), 1);
    assert!(!tc.arena.namespace_exists(Some("session-0")));
    tc.sanity_check(&[], &[]);
}

#[test]
fn a_zero_length_match_anchors_at_the_root() {
    let mut tc = core();
    tc.insert(&InsertParams {
        extra_key: Some("salted"),
        ..insert_params(&vec![1, 2], &[10, 11])
    });
    let anchor = tc
        .match_prefix(&MatchPrefixParams {
            extra_key: Some("salted"),
            ..match_params(&vec![9])
        })
        .best_match_node_id;
    assert_eq!(anchor, tc.root_node_handle(Some("salted")));
    // The root handle stays valid across a full namespace eviction.
    tc.inc_lock_ref(anchor);
    drain_full_device(&mut tc);
    tc.dec_lock_ref(
        anchor, /* params = */ None, /* skip_swa = */ false,
    );
    assert!(tc.arena.try_resolve(anchor).is_some());
    tc.sanity_check(&[], &[]);
}
