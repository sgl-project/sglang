use super::*;
use crate::components::FULL;
use crate::node::{NodeArena, NodeIdx_, ValueSlotIdx};

fn order(list: &UnifiedLRUList) -> Vec<NodeIdx_> {
    list.iter().collect()
}

#[test]
fn fresh_list_reads_are_empty() {
    let list = UnifiedLRUList::new(ValueSlotIdx::device(FULL));
    assert_eq!(list.get_lru_where(|_| true), None);
    assert_eq!(list.iter().count(), 0);
    assert_eq!(list.len(), 0);
    list.validate();
}

#[test]
fn insert_mru_orders_most_recent_first() {
    let mut list = UnifiedLRUList::new(ValueSlotIdx::device(FULL));
    list.insert_mru(NodeIdx_(10));
    list.insert_mru(NodeIdx_(20));
    list.insert_mru(NodeIdx_(30));
    assert_eq!(order(&list), vec![NodeIdx_(30), NodeIdx_(20), NodeIdx_(10)]);
    assert_eq!(list.len(), 3);
    assert!(list.in_list(Some(NodeIdx_(10))));
    list.validate();
}

#[test]
#[should_panic(expected = "already in the LRU list")]
fn insert_mru_panics_when_already_a_member() {
    let mut list = UnifiedLRUList::new(ValueSlotIdx::device(FULL));
    list.insert_mru(NodeIdx_(1));
    list.insert_mru(NodeIdx_(1));
}

#[test]
fn remove_node_updates_membership_immediately() {
    let mut list = UnifiedLRUList::new(ValueSlotIdx::device(FULL));
    list.insert_mru(NodeIdx_(10));
    list.insert_mru(NodeIdx_(20));
    list.insert_mru(NodeIdx_(30));
    list.remove_node(NodeIdx_(20));
    assert_eq!(order(&list), vec![NodeIdx_(30), NodeIdx_(10)]);
    assert!(!list.in_list(Some(NodeIdx_(20))));
    assert_eq!(list.len(), 2);
    list.validate();
}

#[test]
#[should_panic(expected = "index out of bounds")]
fn remove_node_panics_when_absent() {
    let mut list = UnifiedLRUList::new(ValueSlotIdx::device(FULL));
    list.remove_node(NodeIdx_(1));
}

#[test]
#[should_panic(expected = "not in the LRU list")]
fn remove_node_panics_on_a_node_removed_earlier() {
    let mut list = UnifiedLRUList::new(ValueSlotIdx::device(FULL));
    list.insert_mru(NodeIdx_(10));
    list.insert_mru(NodeIdx_(20));
    list.remove_node(NodeIdx_(10));
    // The cell is still allocated but reset; membership must gate the removal.
    list.remove_node(NodeIdx_(10));
}

#[test]
#[should_panic(expected = "not in the LRU list")]
fn remove_node_panics_on_an_unlisted_cell() {
    let mut list = UnifiedLRUList::new(ValueSlotIdx::device(FULL));
    list.insert_mru(NodeIdx_(10));
    list.remove_node(NodeIdx_(10));
    list.remove_node_(UnifiedLRUList::cell_of_(NodeIdx_(10)));
}

#[test]
#[should_panic(expected = "already in the LRU list")]
fn add_node_panics_on_a_linked_cell() {
    let mut list = UnifiedLRUList::new(ValueSlotIdx::device(FULL));
    list.insert_mru(NodeIdx_(10));
    list.add_node_(UnifiedLRUList::cell_of_(NodeIdx_(10)));
}

#[test]
fn removed_nodes_can_be_reinserted() {
    let mut list = UnifiedLRUList::new(ValueSlotIdx::device(FULL));
    list.insert_mru(NodeIdx_(10));
    list.insert_mru(NodeIdx_(20));
    list.remove_node(NodeIdx_(10));
    list.insert_mru(NodeIdx_(10));
    assert_eq!(order(&list), vec![NodeIdx_(10), NodeIdx_(20)]);
    list.validate();
}

#[test]
fn reset_node_mru_moves_a_member_to_the_front() {
    let mut list = UnifiedLRUList::new(ValueSlotIdx::device(FULL));
    list.insert_mru(NodeIdx_(10));
    list.insert_mru(NodeIdx_(20));
    list.insert_mru(NodeIdx_(30));
    list.reset_node_mru(NodeIdx_(10));
    assert_eq!(order(&list), vec![NodeIdx_(10), NodeIdx_(30), NodeIdx_(20)]);
    list.validate();
}

#[test]
#[should_panic(expected = "index out of bounds")]
fn reset_node_mru_panics_on_a_non_member() {
    let mut list = UnifiedLRUList::new(ValueSlotIdx::device(FULL));
    list.reset_node_mru(NodeIdx_(7));
}

// Arena chain root -> a -> b -> c plus a sibling, two atoms per key.
fn arena_chain() -> (
    NodeArena<Vec<i64>>,
    NodeIdx_,
    NodeIdx_,
    NodeIdx_,
    NodeIdx_,
    NodeIdx_,
) {
    let mut arena = NodeArena::new(vec![crate::components::FULL], /* page_size = */ 1);
    let root = arena.root();
    let a = arena
        .alloc_child(
            root,
            /* key = */ vec![1, 11],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    let b = arena
        .alloc_child(
            a,
            /* key = */ vec![2, 22],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    let c = arena
        .alloc_child(
            b,
            /* key = */ vec![3, 33],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    let other = arena
        .alloc_child(
            root,
            /* key = */ vec![9, 99],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    (arena, root, a, b, c, other)
}

#[test]
fn reset_parents_mru_reranks_included_nodes_deepest_first() {
    let (arena, _root, a, b, c, other) = arena_chain();
    let mut list = UnifiedLRUList::new(ValueSlotIdx::device(FULL));
    list.insert_mru(a);
    list.insert_mru(c);
    list.insert_mru(other);
    // b is excluded and skipped; c then a become the MRU run.
    list.reset_node_and_parents_mru(c, &arena, |node| node.idx != b);
    assert_eq!(order(&list), vec![c, a, other]);
    list.validate();
}

#[test]
fn reset_parents_mru_reranks_ancestors_when_the_deepest_is_excluded() {
    let (arena, _root, a, _b, c, other) = arena_chain();
    let mut list = UnifiedLRUList::new(ValueSlotIdx::device(FULL));
    list.insert_mru(a);
    list.insert_mru(other);
    // c and b are excluded; a alone becomes the new MRU head.
    list.reset_node_and_parents_mru(c, &arena, |node| node.idx == a);
    assert_eq!(order(&list), vec![a, other]);
    list.validate();
}

#[test]
fn reset_walks_are_noops_when_node_is_the_root() {
    let (arena, root, a, _b, _c, _other) = arena_chain();
    let mut list = UnifiedLRUList::new(ValueSlotIdx::device(FULL));
    list.insert_mru(a);
    list.reset_node_and_parents_mru(root, &arena, |_| true);
    list.reset_node_and_window_ancestors_mru(root, 4, &arena, |_| true);
    assert_eq!(order(&list), vec![a]);
    list.validate();
}

#[test]
#[should_panic(expected = "not in the LRU list")]
fn reset_parents_mru_panics_on_an_unlisted_included_node() {
    let (arena, _root, _a, _b, c, _other) = arena_chain();
    let mut list = UnifiedLRUList::new(ValueSlotIdx::device(FULL));
    list.insert_mru(c);
    list.remove_node(c);
    list.reset_node_and_parents_mru(c, &arena, |_| true);
}

#[test]
fn reset_window_ancestors_mru_stops_at_the_window() {
    let (arena, _root, a, b, c, other) = arena_chain();
    let mut list = UnifiedLRUList::new(ValueSlotIdx::device(FULL));
    list.insert_mru(b);
    list.insert_mru(c);
    list.insert_mru(a);
    list.insert_mru(other);
    // A window of 4 atoms covers c and b; a stays put beyond it.
    list.reset_node_and_window_ancestors_mru(c, 4, &arena, |_| true);
    assert_eq!(order(&list), vec![c, b, other, a]);
    list.validate();
}

#[test]
fn reset_window_ancestors_mru_includes_the_straddling_ancestor() {
    let (arena, _root, a, b, c, other) = arena_chain();
    let mut list = UnifiedLRUList::new(ValueSlotIdx::device(FULL));
    list.insert_mru(a);
    list.insert_mru(b);
    list.insert_mru(c);
    list.insert_mru(other);
    // A window of 5 atoms ends mid-a: the straddling ancestor is still included.
    list.reset_node_and_window_ancestors_mru(c, 5, &arena, |_| true);
    assert_eq!(order(&list), vec![c, b, a, other]);
    list.validate();
}

#[test]
fn reset_walks_stop_at_the_salted_chains_root() {
    let mut arena: NodeArena<Vec<i64>> =
        NodeArena::new(vec![crate::components::FULL], /* page_size = */ 1);
    let named = arena.root();
    let a = arena
        .alloc_child(
            named,
            /* key = */ vec![1, 11],
            /* priority = */ 0,
            Some("lora-1"),
        )
        .unwrap();
    let b = arena
        .alloc_child(
            a,
            /* key = */ vec![2, 22],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    let mut list = UnifiedLRUList::new(ValueSlotIdx::device(FULL));
    list.insert_mru(a);
    list.insert_mru(b);
    // Both walks terminate at the root without visiting it.
    list.reset_node_and_parents_mru(b, &arena, |_| true);
    assert_eq!(order(&list), vec![b, a]);
    list.reset_node_and_window_ancestors_mru(b, 100, &arena, |_| true);
    assert_eq!(order(&list), vec![b, a]);
    list.validate();
}

#[test]
fn get_lru_no_lock_returns_the_lru_most_unlocked_member() {
    let (mut arena, _root, a, b, c, _other) = arena_chain();
    let mut list = UnifiedLRUList::new(ValueSlotIdx::device(FULL));
    list.insert_mru(a);
    list.insert_mru(b);
    list.insert_mru(c);
    assert_eq!(list.get_lru_no_lock(&arena), Some(a));
    // A lock on the list's own slot hides the LRU end from the walker.
    arena
        .node_mut(a)
        .set_lock_ref_(ValueSlotIdx::device(FULL), 1);
    assert_eq!(list.get_lru_no_lock(&arena), Some(b));
    arena
        .node_mut(b)
        .set_lock_ref_(ValueSlotIdx::device(FULL), 1);
    arena
        .node_mut(c)
        .set_lock_ref_(ValueSlotIdx::device(FULL), 1);
    assert_eq!(list.get_lru_no_lock(&arena), None);
}

#[test]
fn get_prev_no_lock_skips_locked_members_toward_the_mru_end() {
    let (mut arena, _root, a, b, c, other) = arena_chain();
    let mut list = UnifiedLRUList::new(ValueSlotIdx::device(FULL));
    list.insert_mru(a);
    list.insert_mru(b);
    list.insert_mru(c);
    list.insert_mru(other);
    assert_eq!(list.get_prev_no_lock(a, &arena), Some(b));
    // The locked b is skipped; from the MRU end there is no predecessor left.
    arena
        .node_mut(b)
        .set_lock_ref_(ValueSlotIdx::device(FULL), 1);
    assert_eq!(list.get_prev_no_lock(a, &arena), Some(c));
    assert_eq!(list.get_prev_no_lock(other, &arena), None);
    // A lock on a different slot does not gate this list's walker.
    arena.node_mut(c).set_lock_ref_(ValueSlotIdx::host(FULL), 1);
    assert_eq!(list.get_prev_no_lock(a, &arena), Some(c));
}

#[test]
fn reset_window_accumulation_counts_excluded_nodes() {
    let (arena, _root, a, b, c, other) = arena_chain();
    let mut list = UnifiedLRUList::new(ValueSlotIdx::device(FULL));
    list.insert_mru(c);
    list.insert_mru(a);
    list.insert_mru(other);
    // b is excluded but its atoms still consume the window, keeping a out of reach.
    list.reset_node_and_window_ancestors_mru(c, 4, &arena, |node| node.idx != b);
    assert_eq!(order(&list), vec![c, other, a]);
    list.validate();
}

#[test]
#[should_panic(expected = "not in the LRU list")]
fn reset_node_mru_panics_on_a_node_removed_earlier() {
    let mut list = UnifiedLRUList::new(ValueSlotIdx::device(FULL));
    list.insert_mru(NodeIdx_(10));
    list.remove_node(NodeIdx_(10));
    // The cell is still allocated but unlisted; the gated read must reject it.
    list.reset_node_mru(NodeIdx_(10));
}

#[test]
fn in_list_is_false_for_none_and_non_members() {
    let mut list = UnifiedLRUList::new(ValueSlotIdx::device(FULL));
    assert!(!list.in_list(None));
    assert!(!list.in_list(Some(NodeIdx_(5))));
    list.insert_mru(NodeIdx_(5));
    assert!(list.in_list(Some(NodeIdx_(5))));
}

#[test]
fn get_lru_where_walks_from_the_tail() {
    let mut list = UnifiedLRUList::new(ValueSlotIdx::device(FULL));
    list.insert_mru(NodeIdx_(10));
    list.insert_mru(NodeIdx_(20));
    list.insert_mru(NodeIdx_(30));
    assert_eq!(list.get_lru_where(|_| true), Some(NodeIdx_(10)));
    assert_eq!(
        list.get_lru_where(|id| id != NodeIdx_(10)),
        Some(NodeIdx_(20))
    );
    assert_eq!(list.get_lru_where(|_| false), None);
}

#[test]
fn get_prev_where_walks_toward_the_head_from_a_member() {
    let mut list = UnifiedLRUList::new(ValueSlotIdx::device(FULL));
    list.insert_mru(NodeIdx_(10));
    list.insert_mru(NodeIdx_(20));
    list.insert_mru(NodeIdx_(30));
    // Order is [30, 20, 10]; 10's predecessors are 20 then 30.
    assert_eq!(
        list.get_prev_where(NodeIdx_(10), |_| true),
        Some(NodeIdx_(20))
    );
    assert_eq!(
        list.get_prev_where(NodeIdx_(10), |id| id != NodeIdx_(20)),
        Some(NodeIdx_(30))
    );
    assert_eq!(list.get_prev_where(NodeIdx_(30), |_| true), None);
}

#[test]
#[should_panic(expected = "index out of bounds")]
fn get_prev_where_panics_on_a_non_member() {
    let list = UnifiedLRUList::new(ValueSlotIdx::device(FULL));
    list.get_prev_where(NodeIdx_(7), |_| true);
}

#[test]
#[should_panic(expected = "not in the LRU list")]
fn get_prev_where_panics_on_a_node_removed_earlier() {
    let mut list = UnifiedLRUList::new(ValueSlotIdx::device(FULL));
    list.insert_mru(NodeIdx_(10));
    list.insert_mru(NodeIdx_(20));
    list.remove_node(NodeIdx_(10));
    // The cell is still allocated but unlisted; the gated read must reject it.
    list.get_prev_where(NodeIdx_(10), |_| true);
}

#[test]
fn get_prev_before_remove_keeps_the_walk_consistent() {
    let mut list = UnifiedLRUList::new(ValueSlotIdx::device(FULL));
    list.insert_mru(NodeIdx_(10));
    list.insert_mru(NodeIdx_(20));
    list.insert_mru(NodeIdx_(30));
    // The eviction-cursor contract: compute the predecessor, then remove.
    let next = list.get_prev_where(NodeIdx_(10), |_| true);
    list.remove_node(NodeIdx_(10));
    assert_eq!(next, Some(NodeIdx_(20)));
    assert!(list.in_list(next));
    assert_eq!(
        list.get_prev_where(NodeIdx_(20), |_| true),
        Some(NodeIdx_(30))
    );
    list.validate();
}

#[test]
fn insert_mru_grows_the_cell_table_one_by_one() {
    let mut list = UnifiedLRUList::new(ValueSlotIdx::device(FULL));
    list.insert_mru(NodeIdx_(0));
    list.insert_mru(NodeIdx_(1));
    list.insert_mru(NodeIdx_(2));
    assert_eq!(order(&list), vec![NodeIdx_(2), NodeIdx_(1), NodeIdx_(0)]);
    list.validate();
}

#[test]
#[should_panic(expected = "broken prev link")]
fn validate_panics_on_a_corrupted_prev_link() {
    let mut list = UnifiedLRUList::new(ValueSlotIdx::device(FULL));
    list.insert_mru(NodeIdx_(10));
    list.insert_mru(NodeIdx_(20));
    let cell = UnifiedLRUList::cell_of_(NodeIdx_(10));
    list.cells[cell.0].prev = cell;
    list.validate();
}

#[test]
#[should_panic(expected = "membership mismatch")]
fn validate_panics_on_a_linked_cell_without_the_flag() {
    let mut list = UnifiedLRUList::new(ValueSlotIdx::device(FULL));
    list.insert_mru(NodeIdx_(10));
    list.cells[UnifiedLRUList::cell_of_(NodeIdx_(10)).0].in_list = false;
    list.validate();
}

#[test]
#[should_panic(expected = "membership mismatch")]
fn validate_panics_on_a_flagged_unlinked_cell() {
    let mut list = UnifiedLRUList::new(ValueSlotIdx::device(FULL));
    list.insert_mru(NodeIdx_(10));
    list.insert_mru(NodeIdx_(20));
    list.remove_node(NodeIdx_(20));
    // The cell is reset but a stray flag claims membership.
    list.cells[UnifiedLRUList::cell_of_(NodeIdx_(20)).0].in_list = true;
    list.validate();
}

#[test]
fn reset_window_ancestors_mru_is_a_noop_on_a_zero_window() {
    let (arena, _root, a, _b, c, _other) = arena_chain();
    let mut list = UnifiedLRUList::new(ValueSlotIdx::device(FULL));
    list.insert_mru(a);
    list.insert_mru(c);
    list.reset_node_and_window_ancestors_mru(c, 0, &arena, |_| true);
    assert_eq!(order(&list), vec![c, a]);
    list.validate();
}

#[test]
#[should_panic(expected = "out of bounds")]
fn validate_panics_on_an_out_of_range_link() {
    let mut list = UnifiedLRUList::new(ValueSlotIdx::device(FULL));
    list.insert_mru(NodeIdx_(10));
    list.cells[UnifiedLRUList::cell_of_(NodeIdx_(10)).0].next = CellId(99);
    list.validate();
}

#[test]
#[should_panic(expected = "length mismatch")]
fn validate_panics_on_a_desynced_member_counter() {
    let mut list = UnifiedLRUList::new(ValueSlotIdx::device(FULL));
    list.insert_mru(NodeIdx_(10));
    list.len = 2;
    list.validate();
}

#[test]
fn len_drops_to_zero_after_all_members_removed() {
    let mut list = UnifiedLRUList::new(ValueSlotIdx::device(FULL));
    list.insert_mru(NodeIdx_(1));
    list.remove_node(NodeIdx_(1));
    assert_eq!(list.len(), 0);
    assert_eq!(list.iter().count(), 0);
    list.validate();
}

#[test]
fn check_linked_list_accepts_a_clean_list() {
    let mut lru = UnifiedLRUList::new(ValueSlotIdx::device(FULL));
    lru.insert_mru(NodeIdx_(0));
    lru.insert_mru(NodeIdx_(1));
    lru.insert_mru(NodeIdx_(2));
    let mut errors = Vec::new();
    lru.check_linked_list_("[t]", &mut errors);
    assert!(errors.is_empty());
}

#[test]
fn check_linked_list_reports_a_broken_prev() {
    let mut lru = UnifiedLRUList::new(ValueSlotIdx::device(FULL));
    lru.insert_mru(NodeIdx_(0));
    lru.insert_mru(NodeIdx_(1));
    lru.cells[UnifiedLRUList::cell_of_(NodeIdx_(0)).0].prev = UnifiedLRUList::cell_of_(NodeIdx_(0));
    let mut errors = Vec::new();
    lru.check_linked_list_("[t]", &mut errors);
    assert_eq!(errors.len(), 1);
    assert!(errors[0].contains("broken prev at node 0"));
}

#[test]
fn check_linked_list_reports_an_unflagged_member() {
    let mut lru = UnifiedLRUList::new(ValueSlotIdx::device(FULL));
    lru.insert_mru(NodeIdx_(0));
    lru.insert_mru(NodeIdx_(1));
    lru.cells[UnifiedLRUList::cell_of_(NodeIdx_(0)).0].in_list = false;
    let mut errors = Vec::new();
    lru.check_linked_list_("[t]", &mut errors);
    assert_eq!(errors.len(), 1);
    assert!(errors[0].contains("node 0 in list not flagged"));
}

#[test]
fn check_linked_list_reports_a_cycle() {
    let mut lru = UnifiedLRUList::new(ValueSlotIdx::device(FULL));
    lru.insert_mru(NodeIdx_(0));
    lru.insert_mru(NodeIdx_(1));
    // 0's next loops back to 1 instead of reaching the tail.
    lru.cells[UnifiedLRUList::cell_of_(NodeIdx_(0)).0].next = UnifiedLRUList::cell_of_(NodeIdx_(1));
    let mut errors = Vec::new();
    lru.check_linked_list_("[t]", &mut errors);
    assert!(errors.iter().any(|e| e.contains("cycle at node 1")));
}

#[test]
fn check_linked_list_reports_a_count_mismatch() {
    let mut lru = UnifiedLRUList::new(ValueSlotIdx::device(FULL));
    lru.insert_mru(NodeIdx_(0));
    lru.len = 2;
    let mut errors = Vec::new();
    lru.check_linked_list_("[t]", &mut errors);
    assert_eq!(errors.len(), 1);
    assert!(errors[0].contains("list=1 != len=2"));
}

#[test]
fn check_linked_list_reports_an_out_of_bounds_link() {
    let mut lru = UnifiedLRUList::new(ValueSlotIdx::device(FULL));
    lru.insert_mru(NodeIdx_(0));
    lru.cells[UnifiedLRUList::cell_of_(NodeIdx_(0)).0].next = CellId(999);
    let mut errors = Vec::new();
    lru.check_linked_list_("[t]", &mut errors);
    assert!(errors.iter().any(|e| e.contains("cell 999 out of bounds")));
}

#[test]
fn check_linked_list_reports_a_broken_tail_backlink() {
    let mut lru = UnifiedLRUList::new(ValueSlotIdx::device(FULL));
    lru.insert_mru(NodeIdx_(0));
    lru.cells[TAIL.0].prev = HEAD;
    let mut errors = Vec::new();
    lru.check_linked_list_("[t]", &mut errors);
    assert_eq!(errors.len(), 1);
    assert!(errors[0].contains("broken tail backlink"));
}

#[test]
fn check_linked_list_reports_a_flagged_unreachable_cell() {
    let mut lru = UnifiedLRUList::new(ValueSlotIdx::device(FULL));
    lru.insert_mru(NodeIdx_(0));
    lru.insert_mru(NodeIdx_(1));
    lru.remove_node(NodeIdx_(0));
    // Re-flag the unlinked cell without relinking it.
    lru.cells[UnifiedLRUList::cell_of_(NodeIdx_(0)).0].in_list = true;
    let mut errors = Vec::new();
    lru.check_linked_list_("[t]", &mut errors);
    assert!(
        errors
            .iter()
            .any(|e| e.contains("node 0 flagged but unreachable"))
    );
}

// Eviction priority keys.

// A node with distinct field values: last_access 5, creation 7, hits 3, priority 9.
fn arena_with_node() -> (NodeArena<Vec<i64>>, NodeIdx_) {
    let mut arena: NodeArena<Vec<i64>> = NodeArena::new(vec![FULL], /* page_size = */ 1);
    let root = arena.root();
    let a = arena
        .alloc_child(
            root,
            /* key = */ vec![1],
            /* priority = */ 9,
            /* extra_key = */ None,
        )
        .unwrap();
    let node = arena.node_mut(a);
    node.last_access_counter = 5;
    node.creation_counter = 7;
    node.hit_count = 3;
    (arena, NodeIdx_(a.0))
}

#[test]
fn each_strategy_maps_its_node_fields_into_the_key() {
    let (arena, a) = arena_with_node();
    let node = arena.node(NodeIdx_(a.0));
    assert_eq!(LruStrategy.get_priority(node), PriorityKey(5, 0));
    assert_eq!(LfuStrategy.get_priority(node), PriorityKey(3, 5));
    assert_eq!(FifoStrategy.get_priority(node), PriorityKey(7, 0));
    assert_eq!(MruStrategy.get_priority(node), PriorityKey(-5, 0));
    assert_eq!(FiloStrategy.get_priority(node), PriorityKey(-7, 0));
    assert_eq!(PriorityStrategy.get_priority(node), PriorityKey(9, 5));
}

#[test]
fn slru_segments_on_the_protected_threshold() {
    let (mut arena, a) = arena_with_node();
    let slru = SlruStrategy {
        protected_threshold: 2,
    };
    // 3 hits >= threshold 2: protected segment.
    assert_eq!(
        slru.get_priority(arena.node(NodeIdx_(a.0))),
        PriorityKey(1, 5)
    );
    // Exactly at the threshold counts as protected.
    arena.node_mut(NodeIdx_(a.0)).hit_count = 2;
    assert_eq!(
        slru.get_priority(arena.node(NodeIdx_(a.0))),
        PriorityKey(1, 5)
    );
    arena.node_mut(NodeIdx_(a.0)).hit_count = 1;
    assert_eq!(
        slru.get_priority(arena.node(NodeIdx_(a.0))),
        PriorityKey(0, 5)
    );
}

#[test]
fn get_eviction_strategy_resolves_each_policy_name() {
    let (arena, a) = arena_with_node();
    let node = arena.node(NodeIdx_(a.0));
    // Distinct node fields make each policy's key identify its strategy.
    let cases = [
        ("lru", PriorityKey(5, 0)),
        ("LFU", PriorityKey(3, 5)),
        ("fifo", PriorityKey(7, 0)),
        ("mru", PriorityKey(-5, 0)),
        ("filo", PriorityKey(-7, 0)),
        ("priority", PriorityKey(9, 5)),
        ("slru", PriorityKey(1, 5)),
    ];
    for (policy, expected) in cases {
        assert_eq!(
            get_eviction_strategy::<Vec<i64>>(policy).get_priority(node),
            expected,
            "policy {policy}"
        );
    }
}

#[test]
fn eviction_policy_names_are_case_insensitive() {
    let (arena, a) = arena_with_node();
    let node = arena.node(NodeIdx_(a.0));
    // Mixed-case names resolve to the same strategies as their lowercase forms.
    assert_eq!(
        get_eviction_strategy::<Vec<i64>>("LRU").get_priority(node),
        PriorityKey(5, 0)
    );
    assert_eq!(
        get_eviction_strategy::<Vec<i64>>("Priority").get_priority(node),
        PriorityKey(9, 5)
    );
}

#[test]
fn get_eviction_strategy_slru_default_threshold_is_two() {
    let (mut arena, a) = arena_with_node();
    let slru = get_eviction_strategy::<Vec<i64>>("slru");
    // Exactly 2 hits is protected under the factory default; 1 is not.
    arena.node_mut(NodeIdx_(a.0)).hit_count = 2;
    assert_eq!(
        slru.get_priority(arena.node(NodeIdx_(a.0))),
        PriorityKey(1, 5)
    );
    arena.node_mut(NodeIdx_(a.0)).hit_count = 1;
    assert_eq!(
        slru.get_priority(arena.node(NodeIdx_(a.0))),
        PriorityKey(0, 5)
    );
}

#[test]
#[should_panic(expected = "Unknown eviction policy: random. Supported policies:")]
fn get_eviction_strategy_panics_on_an_unknown_policy() {
    get_eviction_strategy::<Vec<i64>>("Random");
}

#[test]
fn priority_keys_order_lexicographically() {
    assert!(PriorityKey(0, 9) < PriorityKey(1, 0));
    assert!(PriorityKey(1, 2) < PriorityKey(1, 3));
}
