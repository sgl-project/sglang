//! Self-contained LRU order over `NodeIdx_`s: MRU at the head side, LRU at the
//! tail side. Node semantics stay with callers through predicates; the reset
//! walks read parent links from the arena.

use std::collections::HashSet;

use crate::node::ChildKeyType;
use crate::node::Node;
use crate::node::NodeArena;
use crate::node::{NodeIdx_, ValueSlotIdx};

/// Index into the cell table; distinct from `NodeIdx_` so shifted and unshifted
/// ids cannot be mixed.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
struct CellId(usize);

/// Head sentinel cell.
const HEAD: CellId = CellId(0);
/// Tail sentinel cell.
const TAIL: CellId = CellId(1);
/// Table offset: node ids map to cells after the sentinels.
const OFFSET: usize = 2;

/// One doubly-linked cell; a node's cell lives at `NodeIdx_ + OFFSET`.
#[derive(Clone, Copy, Default)]
struct Cell {
    prev: CellId,
    next: CellId,
    in_list: bool,
}

/// LRU list over `NodeIdx_`s, with head/tail sentinel cells keeping the link
/// operations branchless. External APIs take `NodeIdx_`s; internal (`_`-suffixed)
/// interfaces work on `CellId`s.
pub struct UnifiedLRUList {
    /// The (component × tier) value slot whose lock gates this list's walkers.
    slot: ValueSlotIdx,
    /// Cell table indexed by `NodeIdx_ + OFFSET`; cells 0/1 are the sentinels.
    cells: Vec<Cell>,
    /// Number of member cells, excluding the sentinels.
    len: usize,
}

impl UnifiedLRUList {
    pub fn new(slot: ValueSlotIdx) -> Self {
        UnifiedLRUList {
            slot,
            cells: vec![
                // Sentinels link to each other and stay permanently flagged so
                // the gated cell accessors admit them.
                Cell {
                    prev: HEAD,
                    next: TAIL,
                    in_list: true,
                },
                Cell {
                    prev: HEAD,
                    next: TAIL,
                    in_list: true,
                },
            ],
            len: 0,
        }
    }

    // ==== List operations ====

    fn add_node_after_(&mut self, prev: CellId, cell: CellId) {
        self.new_cell_(cell);
        let next = self.cell_(prev).next;
        self.connect_(cell, next);
        self.connect_(prev, cell);
    }

    fn add_node_(&mut self, cell: CellId) {
        self.add_node_after_(HEAD, cell);
    }

    fn remove_node_(&mut self, cell: CellId) {
        let Cell { prev, next, .. } = *self.cell_(cell);
        self.connect_(prev, next);
        // Unflag the cell; the stale prev/next are never read while unlisted.
        self.cell_mut_(cell).in_list = false;
        self.len -= 1;
    }

    // ==== NodeIdx_ <-> CellId plumbing ====

    /// The node's cell slot; the only `NodeIdx_` -> `CellId` crossing.
    fn cell_of_(node_id: NodeIdx_) -> CellId {
        CellId(node_id.0 + OFFSET)
    }

    /// The cell's node; the only `CellId` -> `NodeIdx_` crossing.
    fn node_of_(cell: CellId) -> NodeIdx_ {
        NodeIdx_(cell.0 - OFFSET)
    }

    /// The cell, asserting it is linked (sentinels always are).
    #[track_caller]
    fn cell_(&self, id: CellId) -> &Cell {
        let cell = &self.cells[id.0];
        assert!(
            cell.in_list,
            "node {} not in the LRU list",
            Self::node_of_(id)
        );
        cell
    }

    #[track_caller]
    fn cell_mut_(&mut self, id: CellId) -> &mut Cell {
        let cell = &mut self.cells[id.0];
        assert!(
            cell.in_list,
            "node {} not in the LRU list",
            Self::node_of_(id)
        );
        cell
    }

    /// Admit an unlisted cell: grow the table to cover it, then flag and count
    /// it before any connections.
    fn new_cell_(&mut self, cell: CellId) {
        if cell.0 >= self.cells.len() {
            self.cells.resize(cell.0 + 1, Cell::default());
        }
        assert!(
            !self.cells[cell.0].in_list,
            "new_cell_: cell {cell:?} already in the LRU list"
        );
        self.cells[cell.0].in_list = true;
        self.len += 1;
    }

    /// Whether the cell is linked into the list; safe on cells beyond the table.
    fn in_list_(&self, cell: CellId) -> bool {
        self.cells.get(cell.0).is_some_and(|cell| cell.in_list)
    }

    /// Link `a -> b`.
    fn connect_(&mut self, a: CellId, b: CellId) {
        self.cell_mut_(a).next = b;
        self.cell_mut_(b).prev = a;
    }

    /// Insert a node as the most-recently-used; panics if already a member.
    pub fn insert_mru(&mut self, node_id: NodeIdx_) {
        self.add_node_(Self::cell_of_(node_id));
    }

    /// Remove a member node, resetting its cell; panics if not a member.
    pub fn remove_node(&mut self, node_id: NodeIdx_) {
        self.remove_node_(Self::cell_of_(node_id));
    }

    /// Move a member node back to the most-recently-used position.
    pub fn reset_node_mru(&mut self, node_id: NodeIdx_) {
        let cell = Self::cell_of_(node_id);
        self.remove_node_(cell);
        self.add_node_(cell);
    }

    /// Re-rank the `should_include` nodes from `node_id` up to its root
    /// (exclusive) as the MRU run, deepest first.
    pub fn reset_node_and_parents_mru<K: ChildKeyType>(
        &mut self,
        node_id: NodeIdx_,
        arena: &NodeArena<K>,
        mut should_include: impl FnMut(&Node<K>) -> bool,
    ) {
        let mut prev = HEAD;
        let mut cur = node_id;
        loop {
            let node = arena.node(cur);
            let Some(parent) = node.try_parent() else {
                break;
            };
            if should_include(node) {
                let cell = Self::cell_of_(cur);
                self.remove_node_(cell);
                self.add_node_after_(prev, cell);
                prev = cell;
            }
            cur = parent;
        }
    }

    /// Like `reset_node_and_parents_mru`, stopping once `window_size` atoms
    /// are covered; excluded ancestors consume the window too.
    pub fn reset_node_and_window_ancestors_mru<K: ChildKeyType>(
        &mut self,
        node_id: NodeIdx_,
        window_size: usize,
        arena: &NodeArena<K>,
        mut should_include: impl FnMut(&Node<K>) -> bool,
    ) {
        let mut prev = HEAD;
        let mut accumulated = 0;
        let mut cur = node_id;
        while accumulated < window_size {
            let node = arena.node(cur);
            let Some(parent) = node.try_parent() else {
                break;
            };
            if should_include(node) {
                let cell = Self::cell_of_(cur);
                self.remove_node_(cell);
                self.add_node_after_(prev, cell);
                prev = cell;
            }
            accumulated += node.key.atom_len();
            cur = parent;
        }
    }

    /// Whether the node is a member (`None` is never a member).
    pub fn in_list(&self, node_id: Option<NodeIdx_>) -> bool {
        node_id.is_some_and(|id| self.in_list_(Self::cell_of_(id)))
    }

    /// The nearest predecessor of `cell` satisfying `pred`, walking toward the
    /// head; `cell` itself is excluded.
    fn get_prev_where_(
        &self,
        cell: CellId,
        mut pred: impl FnMut(NodeIdx_) -> bool,
    ) -> Option<NodeIdx_> {
        let mut cell = self.cell_(cell).prev;
        while cell != HEAD {
            let node = Self::node_of_(cell);
            if pred(node) {
                return Some(node);
            }
            cell = self.cell_(cell).prev;
        }
        None
    }

    /// The nearest predecessor of a member satisfying `pred`; panics if
    /// `node_id` is not a member.
    pub fn get_prev_where(
        &self,
        node_id: NodeIdx_,
        pred: impl FnMut(NodeIdx_) -> bool,
    ) -> Option<NodeIdx_> {
        self.get_prev_where_(Self::cell_of_(node_id), pred)
    }

    /// The least-recent member whose lock on the list's own slot is free.
    pub fn get_lru_no_lock<K: ChildKeyType>(&self, arena: &NodeArena<K>) -> Option<NodeIdx_> {
        self.get_lru_where(|id| arena.node(id).lock_ref_(self.slot) == 0)
    }

    /// The nearest more-recent member whose lock on the list's own slot is
    /// free, from `node_id`.
    pub fn get_prev_no_lock<K: ChildKeyType>(
        &self,
        node_id: NodeIdx_,
        arena: &NodeArena<K>,
    ) -> Option<NodeIdx_> {
        self.get_prev_where(node_id, |id| arena.node(id).lock_ref_(self.slot) == 0)
    }

    /// The least-recently-used member satisfying `pred`.
    pub fn get_lru_where(&self, pred: impl FnMut(NodeIdx_) -> bool) -> Option<NodeIdx_> {
        self.get_prev_where_(TAIL, pred)
    }

    /// Number of member cells, excluding the sentinels.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Materialize the current members from most to least recent.
    ///
    /// Inspection callers need an owned snapshot across the Python boundary;
    /// the linked-list iterator itself never escapes the Rust core.
    pub(crate) fn snapshot_node_ids(&self) -> Vec<NodeIdx_> {
        let mut node_ids = Vec::with_capacity(self.len);
        let mut cell = self.cell_(HEAD).next;
        while cell != TAIL {
            node_ids.push(Self::node_of_(cell));
            cell = self.cell_(cell).next;
        }
        node_ids
    }

    // ==== Test-only conveniences ====

    /// The members, MRU to LRU.
    #[cfg(test)]
    pub fn iter(&self) -> impl Iterator<Item = NodeIdx_> + '_ {
        let mut cell = self.cell_(HEAD).next;
        std::iter::from_fn(move || {
            if cell == TAIL {
                return None;
            }
            let node = Self::node_of_(cell);
            cell = self.cell_(cell).next;
            Some(node)
        })
    }

    /// Panics if the links, membership flags, or member counter are inconsistent.
    /// Reads cells raw: it inspects possibly-inconsistent state that the gated
    /// accessors would reject.
    #[cfg(test)]
    pub fn validate(&self) {
        let mut count = 0;
        let mut prev = HEAD;
        let mut cell = self.cells[HEAD.0].next;
        while cell != TAIL {
            assert!(
                cell.0 >= OFFSET && cell.0 < self.cells.len(),
                "validate: cell {cell:?} out of bounds"
            );
            assert_eq!(
                self.cells[cell.0].prev, prev,
                "validate: broken prev link at cell {cell:?}"
            );
            assert!(
                self.cells[cell.0].in_list,
                "validate: membership mismatch at cell {cell:?}"
            );
            count += 1;
            assert!(count <= self.len, "validate: cycle detected");
            prev = cell;
            cell = self.cells[cell.0].next;
        }
        assert_eq!(self.cells[TAIL.0].prev, prev, "validate: broken tail link");
        assert_eq!(count, self.len, "validate: length mismatch");
        let flagged = self
            .cells
            .iter()
            .skip(OFFSET)
            .filter(|cell| cell.in_list)
            .count();
        assert_eq!(flagged, self.len, "validate: membership mismatch");
    }
    /// Test-only: desynchronize `len` to force integrity errors.
    #[cfg(test)]
    pub(crate) fn bump_len_for_test(&mut self) {
        self.len += 1;
    }

    /// Walk a LRU doubly-linked list, collect integrity errors.
    pub(crate) fn check_linked_list_(&self, label: &str, errors: &mut Vec<String>) {
        let mut visited: HashSet<usize> = HashSet::new();
        let mut prev = HEAD;
        let mut x = self.cells[HEAD.0].next;
        while x != TAIL {
            if x.0 < OFFSET {
                errors.push(format!("{label} broken chain: link points at a sentinel"));
                break;
            }
            let Some(cell) = self.cells.get(x.0) else {
                errors.push(format!("{label} broken chain: cell {} out of bounds", x.0));
                break;
            };
            if cell.prev != prev {
                errors.push(format!("{label} broken prev at node {}", Self::node_of_(x)));
            }
            if !cell.in_list {
                errors.push(format!(
                    "{label} node {} in list not flagged",
                    Self::node_of_(x)
                ));
            }
            if !visited.insert(x.0) {
                errors.push(format!("{label} cycle at node {}", Self::node_of_(x)));
                break;
            }
            prev = x;
            x = cell.next;
        }
        // The tail backlink closes the list onto the last visited member.
        if x == TAIL && self.cells[TAIL.0].prev != prev {
            errors.push(format!("{label} broken tail backlink"));
        }
        // Every flagged member cell must be reachable from the head.
        for (idx, cell) in self.cells.iter().enumerate().skip(OFFSET) {
            if cell.in_list && !visited.contains(&idx) {
                errors.push(format!(
                    "{label} node {} flagged but unreachable",
                    idx - OFFSET
                ));
            }
        }
        if visited.len() != self.len {
            errors.push(format!(
                "{label} list={} != len={}",
                visited.len(),
                self.len
            ));
        }
    }
}

// Eviction priority keys.

/// Eviction-priority key, ordered lexicographically; lower evicts first.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct PriorityKey(pub i64, pub i64);

/// Ranks nodes for eviction; lower priority evicts first.
pub trait EvictionStrategy<K: ChildKeyType> {
    /// The node's eviction priority.
    fn get_priority(&self, node: &Node<K>) -> PriorityKey;
}

/// Least-recently-used.
pub struct LruStrategy;

impl<K: ChildKeyType> EvictionStrategy<K> for LruStrategy {
    fn get_priority(&self, node: &Node<K>) -> PriorityKey {
        PriorityKey(node.last_access_counter, 0)
    }
}

/// Least-frequently-used; LRU within a hit count.
pub struct LfuStrategy;

impl<K: ChildKeyType> EvictionStrategy<K> for LfuStrategy {
    fn get_priority(&self, node: &Node<K>) -> PriorityKey {
        PriorityKey(node.hit_count, node.last_access_counter)
    }
}

/// First-in-first-out over creation order.
pub struct FifoStrategy;

impl<K: ChildKeyType> EvictionStrategy<K> for FifoStrategy {
    fn get_priority(&self, node: &Node<K>) -> PriorityKey {
        PriorityKey(node.creation_counter, 0)
    }
}

/// Most-recently-used first.
pub struct MruStrategy;

impl<K: ChildKeyType> EvictionStrategy<K> for MruStrategy {
    fn get_priority(&self, node: &Node<K>) -> PriorityKey {
        PriorityKey(-node.last_access_counter, 0)
    }
}

/// First-in-last-out over creation order.
pub struct FiloStrategy;

impl<K: ChildKeyType> EvictionStrategy<K> for FiloStrategy {
    fn get_priority(&self, node: &Node<K>) -> PriorityKey {
        PriorityKey(-node.creation_counter, 0)
    }
}

/// Priority-aware: lower node priority evicts first, LRU within a priority.
pub struct PriorityStrategy;

impl<K: ChildKeyType> EvictionStrategy<K> for PriorityStrategy {
    fn get_priority(&self, node: &Node<K>) -> PriorityKey {
        PriorityKey(node.priority, node.last_access_counter)
    }
}

/// Segmented LRU: probationary nodes (hits below the threshold) evict before
/// protected ones, LRU within a segment.
pub struct SlruStrategy {
    pub protected_threshold: i64,
}

impl<K: ChildKeyType> EvictionStrategy<K> for SlruStrategy {
    fn get_priority(&self, node: &Node<K>) -> PriorityKey {
        PriorityKey(
            (node.hit_count >= self.protected_threshold) as i64,
            node.last_access_counter,
        )
    }
}

/// The strategy for an eviction-policy name.
pub fn get_eviction_strategy<K: ChildKeyType>(policy: &str) -> Box<dyn EvictionStrategy<K> + Send> {
    match policy.to_lowercase().as_str() {
        "lru" => Box::new(LruStrategy),
        "lfu" => Box::new(LfuStrategy),
        "fifo" => Box::new(FifoStrategy),
        "mru" => Box::new(MruStrategy),
        "filo" => Box::new(FiloStrategy),
        "priority" => Box::new(PriorityStrategy),
        "slru" => Box::new(SlruStrategy {
            protected_threshold: 2,
        }),
        other => panic!(
            "Unknown eviction policy: {other}. Supported policies: \
             'lru', 'lfu', 'fifo', 'mru', 'filo', 'priority', 'slru'."
        ),
    }
}
#[cfg(test)]
#[path = "tests/unified_lru_list.rs"]
mod tests;
