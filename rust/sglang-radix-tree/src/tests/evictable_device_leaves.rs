use std::cell::Cell;
use std::collections::{BTreeSet, HashMap};

use super::*;

fn idx(i: usize) -> NodeIdx_ {
    NodeIdx_(i)
}

fn key(k: i64) -> PriorityKey {
    PriorityKey(k, 0)
}

/// Priorities by slot; a missing slot panics so stray lookups surface.
struct Priorities(HashMap<usize, i64>);

impl Priorities {
    fn of(&self, id: NodeIdx_) -> PriorityKey {
        key(self.0[&id.0])
    }
}

fn drain_walk(leaves: &mut EvictableDeviceLeaves, prio: &Priorities) -> Vec<usize> {
    leaves.begin_walk(|id| prio.of(id));
    let mut order = Vec::new();
    while let Some(id) = leaves.pop_next(|id| prio.of(id)) {
        order.push(id.0);
        leaves.discard(id);
    }
    leaves.end_walk();
    order
}

#[test]
fn pops_in_priority_then_slot_order() {
    let prio = Priorities(HashMap::from([(1, 30), (2, 10), (3, 20), (4, 10)]));
    let mut leaves = EvictableDeviceLeaves::new();
    for i in [1, 2, 3, 4] {
        leaves.add(idx(i));
    }
    assert_eq!(drain_walk(&mut leaves, &prio), vec![2, 4, 3, 1]);
}

#[test]
fn the_queue_survives_across_walks() {
    let prio = Priorities(HashMap::from([(1, 1), (2, 2), (3, 3)]));
    let mut leaves = EvictableDeviceLeaves::new();
    for i in [1, 2, 3] {
        leaves.add(idx(i));
    }
    for expected in [1, 2, 3] {
        leaves.begin_walk(|id| prio.of(id));
        let id = leaves.pop_next(|id| prio.of(id)).unwrap();
        assert_eq!(id.0, expected);
        leaves.discard(id);
        leaves.end_walk();
    }
}

#[test]
fn a_walk_scores_only_slots_admitted_since_the_last_walk() {
    let prio = Priorities((0..1000).map(|i| (i, i as i64)).collect());
    let calls = Cell::new(0usize);
    let counted = |id: NodeIdx_| {
        calls.set(calls.get() + 1);
        prio.of(id)
    };
    let mut leaves = EvictableDeviceLeaves::new();
    for i in 0..1000 {
        leaves.add(idx(i));
    }
    leaves.begin_walk(counted);
    leaves.end_walk();
    for expected in 0..10 {
        calls.set(0);
        leaves.begin_walk(counted);
        let id = leaves.pop_next(counted).unwrap();
        leaves.discard(id);
        leaves.end_walk();
        assert_eq!(id.0, expected);
        // One validation for the popped entry; no O(L) rescoring.
        assert!(calls.get() <= 2, "walk scored {} slots", calls.get());
    }
}

#[test]
fn a_slot_that_joins_mid_walk_waits_for_the_next_walk() {
    let prio = Priorities(HashMap::from([(1, 10), (2, 1)]));
    let mut leaves = EvictableDeviceLeaves::new();
    leaves.add(idx(1));
    leaves.begin_walk(|id| prio.of(id));
    // Lower priority than the queued slot, but it joined after the snapshot.
    leaves.add(idx(2));
    assert_eq!(leaves.pop_next(|id| prio.of(id)), Some(idx(1)));
    leaves.discard(idx(1));
    assert_eq!(leaves.pop_next(|id| prio.of(id)), None);
    leaves.end_walk();
    assert_eq!(drain_walk(&mut leaves, &prio), vec![2]);
}

#[test]
fn a_stale_entry_does_not_admit_a_slot_that_rejoins_mid_walk() {
    let mut prio = Priorities(HashMap::from([(1, 10), (2, 1)]));
    let mut leaves = EvictableDeviceLeaves::new();
    leaves.add(idx(1));
    leaves.add(idx(2));
    leaves.begin_walk(|id| prio.of(id));
    leaves.end_walk();
    // Slot 2 leaves the set, keeping its queued (1, 2) entry.
    leaves.discard(idx(2));
    leaves.begin_walk(|id| prio.of(id));
    // It rejoins mid-walk at a priority its stale entry would re-key to.
    prio.0.insert(2, 5);
    leaves.add(idx(2));
    assert_eq!(leaves.pop_next(|id| prio.of(id)), Some(idx(1)));
    leaves.discard(idx(1));
    assert_eq!(leaves.pop_next(|id| prio.of(id)), None);
    leaves.end_walk();
    assert_eq!(drain_walk(&mut leaves, &prio), vec![2]);
}

#[test]
fn a_raised_priority_is_rekeyed_without_a_hook() {
    let mut prio = Priorities(HashMap::from([(1, 1), (2, 2)]));
    let mut leaves = EvictableDeviceLeaves::new();
    leaves.add(idx(1));
    leaves.add(idx(2));
    leaves.begin_walk(|id| prio.of(id));
    leaves.end_walk();
    // An access moves slot 1 past slot 2 (LRU); no requeue is needed.
    prio.0.insert(1, 3);
    assert_eq!(drain_walk(&mut leaves, &prio), vec![2, 1]);
}

#[test]
fn a_lowered_priority_takes_effect_once_requeued() {
    let mut prio = Priorities(HashMap::from([(1, 1), (2, 2), (3, 3)]));
    let mut leaves = EvictableDeviceLeaves::new();
    for i in [1, 2, 3] {
        leaves.add(idx(i));
    }
    leaves.begin_walk(|id| prio.of(id));
    leaves.end_walk();
    // MRU-style: an access drops slot 3 below everything else.
    prio.0.insert(3, -1);
    leaves.requeue(idx(3));
    assert_eq!(drain_walk(&mut leaves, &prio), vec![3, 1, 2]);
}

#[test]
fn a_recycled_slot_is_ordered_by_its_new_priority() {
    let mut prio = Priorities(HashMap::from([(1, 1), (2, 5)]));
    let mut leaves = EvictableDeviceLeaves::new();
    leaves.add(idx(1));
    leaves.add(idx(2));
    leaves.begin_walk(|id| prio.of(id));
    leaves.end_walk();
    // Slot 1 is freed, then reused by a newer node.
    leaves.discard(idx(1));
    prio.0.insert(1, 9);
    leaves.add(idx(1));
    assert_eq!(drain_walk(&mut leaves, &prio), vec![2, 1]);
}

#[test]
fn walk_parents_merge_in_order_and_expire_with_the_walk() {
    let prio = Priorities(HashMap::from([(1, 1), (3, 3)]));
    let mut leaves = EvictableDeviceLeaves::new();
    leaves.add(idx(1));
    leaves.add(idx(3));
    leaves.begin_walk(|id| prio.of(id));
    leaves.push_walk_parent(key(2), idx(2));
    let mut order = Vec::new();
    while let Some(id) = leaves.pop_next(|id| prio.of(id)) {
        order.push(id.0);
        leaves.discard(id);
    }
    assert_eq!(order, vec![1, 2, 3]);
    leaves.push_walk_parent(key(7), idx(7));
    leaves.end_walk();
    leaves.begin_walk(|id| prio.of(id));
    assert_eq!(leaves.pop_next(|id| prio.of(id)), None);
    leaves.end_walk();
}

#[test]
fn queue_growth_stays_bounded() {
    let prio = Priorities(HashMap::from([(1, 1), (2, 2)]));
    let mut leaves = EvictableDeviceLeaves::new();
    leaves.add(idx(1));
    leaves.add(idx(2));
    for _ in 0..10_000 {
        leaves.requeue(idx(1));
        leaves.requeue(idx(2));
    }
    leaves.begin_walk(|id| prio.of(id));
    assert!(leaves.queue_len() <= 2 * leaves.len() + COMPACTION_SLACK);
    leaves.end_walk();
    assert_eq!(drain_walk(&mut leaves, &prio), vec![1, 2]);
}

#[test]
fn the_invariant_check_reports_a_missed_requeue() {
    let mut prio = Priorities(HashMap::from([(1, 5)]));
    let mut leaves = EvictableDeviceLeaves::new();
    leaves.add(idx(1));
    leaves.begin_walk(|id| prio.of(id));
    leaves.end_walk();
    assert_eq!(leaves.check_queue_invariant(|id| prio.of(id)), None);
    prio.0.insert(1, 0);
    assert!(leaves.check_queue_invariant(|id| prio.of(id)).is_some());
    leaves.requeue(idx(1));
    assert_eq!(leaves.check_queue_invariant(|id| prio.of(id)), None);
}

#[test]
fn a_member_handed_out_but_not_evicted_stays_queued() {
    let prio = Priorities(HashMap::from([(1, 1), (2, 2)]));
    let mut leaves = EvictableDeviceLeaves::new();
    leaves.add(idx(1));
    leaves.add(idx(2));
    // The caller stops the walk after one pop without evicting the victim.
    leaves.begin_walk(|id| prio.of(id));
    assert_eq!(leaves.pop_next(|id| prio.of(id)), Some(idx(1)));
    leaves.end_walk();
    assert_eq!(leaves.check_queue_invariant(|id| prio.of(id)), None);
    assert_eq!(drain_walk(&mut leaves, &prio), vec![1, 2]);
}

#[test]
fn a_duplicate_entry_is_handed_out_once_per_walk() {
    let prio = Priorities(HashMap::from([(1, 1)]));
    let mut leaves = EvictableDeviceLeaves::new();
    leaves.add(idx(1));
    leaves.begin_walk(|id| prio.of(id));
    leaves.end_walk();
    // An unchanged-priority requeue leaves two equal-key entries.
    leaves.requeue(idx(1));
    leaves.begin_walk(|id| prio.of(id));
    assert_eq!(leaves.pop_next(|id| prio.of(id)), Some(idx(1)));
    assert_eq!(leaves.pop_next(|id| prio.of(id)), None);
    leaves.end_walk();
    assert_eq!(drain_walk(&mut leaves, &prio), vec![1]);
}

/// Deterministic xorshift so the fuzz needs no extra crate.
struct Rng(u64);

impl Rng {
    fn below(&mut self, n: u64) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0 % n
    }
}

/// The pre-queue walk: snapshot every member at walk start, then pop members only.
fn rebuild_walk_order(
    members: &BTreeSet<usize>,
    prio: &HashMap<usize, i64>,
    victims: usize,
) -> Vec<usize> {
    let mut snapshot: Vec<(i64, usize)> = members.iter().map(|&i| (prio[&i], i)).collect();
    snapshot.sort();
    snapshot.into_iter().take(victims).map(|(_, i)| i).collect()
}

#[test]
fn eviction_order_matches_a_per_walk_rebuild() {
    for seed in 1..=20u64 {
        let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let mut prio: HashMap<usize, i64> = HashMap::new();
        let mut members: BTreeSet<usize> = BTreeSet::new();
        let mut leaves = EvictableDeviceLeaves::new();
        let mut tick = 0i64;
        for _ in 0..300 {
            for _ in 0..rng.below(8) {
                let slot = rng.below(64) as usize;
                match rng.below(4) {
                    0 | 1 => {
                        if !members.contains(&slot) {
                            tick += 1;
                            prio.insert(slot, tick);
                            members.insert(slot);
                            leaves.add(idx(slot));
                        }
                    }
                    2 => {
                        members.remove(&slot);
                        leaves.discard(idx(slot));
                    }
                    _ => {
                        if prio.contains_key(&slot) {
                            // Raise or lower; only a drop needs the hook.
                            let old = prio[&slot];
                            tick += 1;
                            let new = if rng.below(2) == 0 { tick } else { -tick };
                            prio.insert(slot, new);
                            if new < old {
                                leaves.requeue(idx(slot));
                            }
                        }
                    }
                }
            }
            let victims = rng.below(4) as usize;
            let expected = rebuild_walk_order(&members, &prio, victims);
            let p = Priorities(prio.clone());
            leaves.begin_walk(|id| p.of(id));
            let mut actual = Vec::new();
            while actual.len() < victims {
                let Some(id) = leaves.pop_next(|id| p.of(id)) else {
                    break;
                };
                actual.push(id.0);
                members.remove(&id.0);
                leaves.discard(id);
                // A slot joining mid-walk must not perturb this walk.
                let late = 64 + rng.below(64) as usize;
                if !members.contains(&late) && rng.below(3) == 0 {
                    tick += 1;
                    prio.insert(late, if rng.below(2) == 0 { tick } else { -tick });
                    members.insert(late);
                    leaves.add(idx(late));
                }
            }
            leaves.end_walk();
            assert_eq!(actual, expected, "seed {seed}");
            let p = Priorities(prio.clone());
            assert_eq!(
                leaves.check_queue_invariant(|id| p.of(id)),
                None,
                "seed {seed}"
            );
        }
    }
}
