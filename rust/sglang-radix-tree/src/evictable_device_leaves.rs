//! The Full D-leaf set and its persistent eviction queue.
//!
//! The queue survives across device-eviction walks so a walk costs
//! O(k log L) for k victims instead of an O(L) rebuild over every D-leaf.
//! Entries are validated lazily when popped. Invariant between walks, while
//! not `needs_rebuild`: every member is in `admitted` or has a queue entry
//! whose key is at most its current priority. Priority increases therefore
//! need no hook; anything that can lower a member's priority calls `requeue`.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};

use crate::node::{NodeIdx_, NodeSet};
use crate::unified_lru_list::PriorityKey;

type Entry = Reverse<(PriorityKey, NodeIdx_)>;

/// Arbitrary headroom so tiny trees do not compact on every walk.
const COMPACTION_SLACK: usize = 64;

pub struct EvictableDeviceLeaves {
    members: NodeSet,
    /// Members that joined, or whose priority may have dropped, since the
    /// queue last absorbed them.
    admitted: Vec<NodeIdx_>,
    /// Per arena slot: the walk generation in which the slot last joined.
    joined_walk: Vec<u64>,
    /// Bumped at every walk start. A slot that joins mid-walk is not eligible
    /// until the next walk, matching a snapshot taken at walk start.
    walk: u64,
    queue: BinaryHeap<Entry>,
    /// Leaf parents re-admitted during the current walk; cleared at walk end.
    walk_parents: BinaryHeap<Entry>,
    /// `admitted` overflowed: rebuild the queue from `members` at walk start.
    needs_rebuild: bool,
}

impl Default for EvictableDeviceLeaves {
    fn default() -> Self {
        Self::new()
    }
}

impl EvictableDeviceLeaves {
    pub fn new() -> Self {
        Self {
            members: NodeSet::new(),
            admitted: Vec::new(),
            joined_walk: Vec::new(),
            walk: 0,
            queue: BinaryHeap::new(),
            walk_parents: BinaryHeap::new(),
            needs_rebuild: false,
        }
    }

    pub fn contains(&self, node_id: NodeIdx_) -> bool {
        self.members.contains(node_id)
    }

    pub fn add(&mut self, node_id: NodeIdx_) {
        if self.members.contains(node_id) {
            return;
        }
        self.members.add(node_id);
        if node_id.0 >= self.joined_walk.len() {
            self.joined_walk.resize(node_id.0 + 1, 0);
        }
        self.joined_walk[node_id.0] = self.walk;
        self.note_admitted(node_id);
    }

    /// Queue entries of a discarded slot are dropped lazily when popped.
    pub fn discard(&mut self, node_id: NodeIdx_) {
        self.members.discard(node_id);
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = NodeIdx_> + '_ {
        self.members.iter()
    }

    /// Re-key a member whose priority may have dropped since it was queued.
    pub fn requeue(&mut self, node_id: NodeIdx_) {
        if self.members.contains(node_id) {
            self.note_admitted(node_id);
        }
    }

    fn note_admitted(&mut self, node_id: NodeIdx_) {
        if self.needs_rebuild {
            return;
        }
        if self.admitted.len() >= self.compaction_bound() {
            self.admitted.clear();
            self.needs_rebuild = true;
            return;
        }
        self.admitted.push(node_id);
    }

    fn compaction_bound(&self) -> usize {
        2 * self.members.iter().len() + COMPACTION_SLACK
    }

    /// Start a walk: absorb admissions, or rebuild when the queue has grown
    /// past the compaction bound.
    pub fn begin_walk(&mut self, priority: impl Fn(NodeIdx_) -> PriorityKey) {
        self.walk += 1;
        self.walk_parents.clear();
        if self.needs_rebuild || self.queue.len() + self.admitted.len() > self.compaction_bound() {
            self.queue = self
                .members
                .iter()
                .map(|id| Reverse((priority(id), id)))
                .collect();
            self.admitted.clear();
            self.needs_rebuild = false;
            return;
        }
        for id in self.admitted.drain(..) {
            if self.members.contains(id) {
                self.queue.push(Reverse((priority(id), id)));
            }
        }
    }

    pub fn end_walk(&mut self) {
        self.walk_parents.clear();
    }

    /// Offer a parent that became a walk candidate after its child's eviction.
    pub fn push_walk_parent(&mut self, key: PriorityKey, node_id: NodeIdx_) {
        self.walk_parents.push(Reverse((key, node_id)));
    }

    /// The next walk candidate in (priority, slot) order across the queue and
    /// this walk's parents. The caller still checks walk eligibility.
    pub fn pop_next(&mut self, priority: impl Fn(NodeIdx_) -> PriorityKey) -> Option<NodeIdx_> {
        loop {
            let take_parent = match (self.walk_parents.peek(), self.queue.peek()) {
                (None, None) => return None,
                (Some(_), None) => true,
                (None, Some(_)) => false,
                (Some(parent), Some(queued)) => parent > queued,
            };
            if take_parent {
                let Reverse((_, id)) = self.walk_parents.pop().unwrap();
                return Some(id);
            }
            let Reverse((key, id)) = self.queue.pop().unwrap();
            if !self.members.contains(id) || self.joined_walk[id.0] == self.walk {
                continue;
            }
            let current = priority(id);
            if key < current {
                self.queue.push(Reverse((current, id)));
                continue;
            }
            // A higher key is stale: `requeue` queued a fresher entry.
            if key > current {
                continue;
            }
            return Some(id);
        }
    }

    /// Describe a broken queue invariant, if any.
    pub fn check_queue_invariant(
        &self,
        priority: impl Fn(NodeIdx_) -> PriorityKey,
    ) -> Option<String> {
        if self.needs_rebuild {
            return None;
        }
        let mut min_key: HashMap<NodeIdx_, PriorityKey> = HashMap::new();
        for Reverse((key, id)) in self.queue.iter() {
            min_key
                .entry(*id)
                .and_modify(|k| *k = (*k).min(*key))
                .or_insert(*key);
        }
        let admitted: HashSet<NodeIdx_> = self.admitted.iter().copied().collect();
        let uncovered: Vec<NodeIdx_> = self
            .members
            .iter()
            .filter(|id| {
                !admitted.contains(id) && min_key.get(id).is_none_or(|k| *k > priority(*id))
            })
            .take(5)
            .collect();
        (!uncovered.is_empty())
            .then(|| format!("D-leaf eviction queue misses members: {uncovered:?}"))
    }

    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.members.iter().len()
    }

    #[cfg(test)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[cfg(test)]
    pub fn queue_len(&self) -> usize {
        self.queue.len()
    }
}

#[cfg(test)]
#[path = "tests/evictable_device_leaves.rs"]
mod tests;
