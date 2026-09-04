use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

use tch::Tensor;

use super::*;
use crate::components::{FULL, MAMBA, SWA};
use crate::node::TreeCoreRuntimeError;

static COUNTED_KEY_CONSTRUCTIONS: AtomicUsize = AtomicUsize::new(0);

#[derive(Default)]
struct ByteCountingHasher {
    bytes_written: usize,
}

impl std::hash::Hasher for ByteCountingHasher {
    fn finish(&self) -> u64 {
        0
    }

    fn write(&mut self, bytes: &[u8]) {
        self.bytes_written += bytes.len();
    }
}

#[derive(Clone, Debug, Default, Eq, Hash, PartialEq)]
struct CountedKey(Vec<i64>);

impl AsRef<[i64]> for CountedKey {
    fn as_ref(&self) -> &[i64] {
        &self.0
    }
}

impl Borrow<[i64]> for CountedKey {
    fn borrow(&self) -> &[i64] {
        &self.0
    }
}

impl From<Vec<i64>> for CountedKey {
    fn from(value: Vec<i64>) -> Self {
        COUNTED_KEY_CONSTRUCTIONS.fetch_add(1, Ordering::Relaxed);
        Self(value)
    }
}

impl ChildKeyType for CountedKey {
    type Atom = i64;
    const IS_BIGRAM: bool = false;

    fn key_from(token_ids: Cow<'_, Vec<i64>>) -> Cow<'_, Self> {
        Cow::Owned(Self::from(token_ids.into_owned()))
    }

    fn hash_words(atom: &Self::Atom) -> impl Iterator<Item = u32> {
        std::iter::once(*atom as u32)
    }

    fn raw_token_ids(atoms: &[Self::Atom]) -> Cow<'_, [i64]> {
        Cow::Borrowed(atoms)
    }
}

#[test]
fn evicted_only_for_attached_nodes_without_full_device_value() {
    let mut parent: Node<Vec<i64>> = Node::new_root(/* id = */ 0);
    assert!(!parent.evicted());
    let mut child = Node::new_child(
        /* id = */ 1,
        /* key = */ vec![7],
        /* priority = */ 0,
    );
    // Detached (parent-less) nodes are root-like: never evicted.
    assert!(!child.evicted());
    parent
        .attach_child(&mut child, /* page_size = */ 1)
        .unwrap();
    assert!(child.evicted());
    child.set_device_value(FULL, Tensor::from_slice(&[1i64]));
    assert!(!child.evicted());
}

#[test]
fn get_last_hash_value_returns_the_final_page_hash() {
    let mut node: Node<Vec<i64>> = Node::new_child(
        /* id = */ 1,
        /* key = */ vec![7],
        /* priority = */ 0,
    );
    assert_eq!(node.get_last_hash_value(), None);
    node.hash_value = Some(Vec::new());
    assert_eq!(node.get_last_hash_value(), None);
    node.hash_value = Some(vec!["h0".to_string(), "h1".to_string()]);
    assert_eq!(node.get_last_hash_value(), Some("h1"));
}

#[test]
fn backuped_tracks_the_full_host_value() {
    let mut node: Node<Vec<i64>> = Node::new_child(
        /* id = */ 1,
        /* key = */ vec![7],
        /* priority = */ 0,
    );
    assert!(!node.backuped());
    node.set_host_value(FULL, Tensor::from_slice(&[1i64]));
    assert!(node.backuped());
}

#[test]
fn attach_child_links_both_sides() {
    let mut parent: Node<Vec<i64>> = Node::new_root(/* id = */ 0);
    let mut child = Node::new_child(
        /* id = */ 1,
        /* key = */ vec![7],
        /* priority = */ 0,
    );
    child.namespace = KeyNamespace::new(Some("ns"), None);
    parent
        .attach_child(&mut child, /* page_size = */ 1)
        .unwrap();
    assert_eq!(child.parent, Some(NodeIdx_(0)));
    // The edge key mirrors the child's namespace.
    assert_eq!(
        parent
            .children
            .get(&(KeyNamespace::new(Some("ns"), None), vec![7])),
        Some(&NodeIdx_(1))
    );
}

#[test]
fn parent_accessors_resolve_the_link() {
    let mut parent: Node<Vec<i64>> = Node::new_root(/* id = */ 0);
    let mut child = Node::new_child(
        /* id = */ 1,
        /* key = */ vec![7],
        /* priority = */ 0,
    );
    parent
        .attach_child(&mut child, /* page_size = */ 1)
        .unwrap();
    assert_eq!(child.parent(), NodeIdx_(0));
    assert_eq!(child.try_parent(), Some(NodeIdx_(0)));
    assert_eq!(parent.try_parent(), None);
}

#[test]
#[should_panic(expected = "node 0 is a root and has no parent")]
fn parent_panics_on_a_root() {
    let root: Node<Vec<i64>> = Node::new_root(/* id = */ 0);
    root.parent();
}

#[test]
#[should_panic(expected = "already attached")]
fn attach_child_panics_on_already_attached() {
    let mut parent: Node<Vec<i64>> = Node::new_root(/* id = */ 0);
    let mut child = Node::new_child(
        /* id = */ 1,
        /* key = */ vec![7],
        /* priority = */ 0,
    );
    parent
        .attach_child(&mut child, /* page_size = */ 1)
        .unwrap();
    // Re-attaching an already-attached node is an internal invariant violation.
    let mut other: Node<Vec<i64>> = Node::new_root(/* id = */ 2);
    let _ = other.attach_child(&mut child, /* page_size = */ 1);
}

#[test]
fn attach_child_rejects_duplicate_key() {
    let mut parent: Node<Vec<i64>> = Node::new_root(/* id = */ 0);
    let mut a = Node::new_child(
        /* id = */ 1,
        /* key = */ vec![7],
        /* priority = */ 0,
    );
    let mut b = Node::new_child(
        /* id = */ 2,
        /* key = */ vec![7],
        /* priority = */ 0,
    );
    parent.attach_child(&mut a, /* page_size = */ 1).unwrap();
    assert!(matches!(
        parent.attach_child(&mut b, /* page_size = */ 1),
        Err(TreeCoreRuntimeError::DuplicateChildKey { parent: p, .. }) if p == 0
    ));
    // b was rejected without mutation; a still holds key 7.
    assert_eq!(
        parent.children.get(&(KeyNamespace::default(), vec![7])),
        Some(&NodeIdx_(1))
    );
    assert_eq!(b.parent, None);
}

#[test]
fn detach_from_parent_unlinks_both_sides() {
    let mut parent: Node<Vec<i64>> = Node::new_root(/* id = */ 0);
    let mut child = Node::new_child(
        /* id = */ 1,
        /* key = */ vec![7],
        /* priority = */ 0,
    );
    parent
        .attach_child(&mut child, /* page_size = */ 1)
        .unwrap();
    child.detach_from_parent(&mut parent, /* page_size = */ 1);
    assert_eq!(child.parent, None);
    assert!(parent.children.is_empty());
}

#[test]
#[should_panic(expected = "no child")]
fn detach_from_parent_panics_on_broken_link() {
    let mut parent: Node<Vec<i64>> = Node::new_root(/* id = */ 0);
    // `orphan` is not registered under `parent` (never attached) — a broken link.
    let mut orphan = Node::new_child(
        /* id = */ 1,
        /* key = */ vec![7],
        /* priority = */ 0,
    );
    orphan.detach_from_parent(&mut parent, /* page_size = */ 1);
}

#[test]
#[should_panic(expected = "(found Some(NodeIdx_(1)))")]
fn detach_from_parent_panics_on_a_mismatched_child_id() {
    let mut parent: Node<Vec<i64>> = Node::new_root(/* id = */ 0);
    let mut a = Node::new_child(
        /* id = */ 1,
        /* key = */ vec![7],
        /* priority = */ 0,
    );
    parent.attach_child(&mut a, /* page_size = */ 1).unwrap();
    // `b` claims the same page key, but the parent's entry points at `a`.
    let mut b = Node::new_child(
        /* id = */ 2,
        /* key = */ vec![7],
        /* priority = */ 0,
    );
    b.parent = Some(NodeIdx_(0));
    b.detach_from_parent(&mut parent, /* page_size = */ 1);
}

#[test]
fn value_slot_idx_addresses_device_then_host() {
    assert_eq!(ValueSlotIdx::device(FULL).idx(), 0);
    assert_eq!(ValueSlotIdx::device(MAMBA).idx(), 2);
    assert_eq!(ValueSlotIdx::host(FULL).idx(), 3);
    assert_eq!(ValueSlotIdx::host(MAMBA).idx(), 5);
    assert!(!ValueSlotIdx::device(SWA).is_host());
    assert!(ValueSlotIdx::host(SWA).is_host());
    assert_eq!(ValueSlotIdx::host(SWA).component_type(), SWA);
}

#[test]
fn device_and_host_accessors_address_their_own_tiers() {
    let mut node: Node<Vec<i64>> = Node::new_child(
        /* id = */ 1,
        /* key = */ vec![7, 8],
        /* priority = */ 0,
    );
    assert!(!node.has_device_value(SWA));
    node.set_device_value(SWA, Tensor::from_slice(&[1i64, 2]));
    assert!(node.has_device_value(SWA));
    assert!(!node.has_host_value(SWA));
    assert_eq!(node.device_value_len(SWA), 2);
    assert_eq!(node.host_value_len(SWA), 0);

    node.set_host_value(SWA, Tensor::from_slice(&[3i64, 4]));
    assert_eq!(node.host_value(SWA).size()[0], 2);
    let taken = node.take_host_value(SWA);
    assert_eq!(taken.size()[0], 2);
    assert!(node.has_device_value(SWA));
}

#[test]
fn lock_predicates_scan_their_own_tier() {
    let mut node: Node<Vec<i64>> = Node::new_child(
        /* id = */ 1,
        /* key = */ vec![7],
        /* priority = */ 0,
    );
    assert!(!node.is_device_locked() && !node.is_host_locked());
    node.inc_device_lock_ref(MAMBA);
    assert!(node.is_device_locked() && !node.is_host_locked());
    node.dec_device_lock_ref(MAMBA);
    node.inc_host_lock_ref(FULL);
    assert!(!node.is_device_locked() && node.is_host_locked());
    assert_eq!(node.host_lock_ref(FULL), 1);
    assert_eq!(node.device_lock_ref(FULL), 0);
}

#[test]
fn load_back_pending_predicate_tracks_the_anchor() {
    let mut node: Node<Vec<i64>> = Node::new_child(
        /* id = */ 1,
        /* key = */ vec![7],
        /* priority = */ 0,
    );
    assert!(!node.is_load_back_pending());
    node.load_back_pending_id = Some(2);
    assert!(node.is_load_back_pending());
}

#[test]
#[should_panic(expected = "expects a single state slot")]
fn set_value_rejects_multi_row_mamba_states() {
    let mut node: Node<Vec<i64>> = Node::new_child(
        /* id = */ 1,
        /* key = */ vec![7],
        /* priority = */ 0,
    );
    node.set_device_value(MAMBA, Tensor::from_slice(&[1i64, 2]));
}

#[test]
#[should_panic(expected = "dec_lock_ref: Full/device lock_ref underflow on node 1")]
fn dec_lock_ref_panics_when_unlocked() {
    let mut node: Node<Vec<i64>> = Node::new_child(
        /* id = */ 1,
        /* key = */ vec![7],
        /* priority = */ 0,
    );
    node.dec_device_lock_ref(FULL);
}

#[test]
#[should_panic(expected = "value length differs from the key")]
fn set_value_rejects_a_length_mismatch() {
    let mut node: Node<Vec<i64>> = Node::new_child(
        /* id = */ 1,
        /* key = */ vec![7, 8],
        /* priority = */ 0,
    );
    node.set_device_value(FULL, Tensor::from_slice(&[1i64, 2, 3]));
}

#[test]
#[should_panic(expected = "slot already set")]
fn set_value_rejects_an_occupied_slot() {
    let mut node: Node<Vec<i64>> = Node::new_child(
        /* id = */ 1,
        /* key = */ vec![7],
        /* priority = */ 0,
    );
    node.set_device_value(FULL, Tensor::from_slice(&[1i64]));
    node.set_device_value(FULL, Tensor::from_slice(&[2i64]));
}

#[test]
#[should_panic(expected = "has no value")]
fn value_panics_on_an_empty_slot() {
    let node: Node<Vec<i64>> = Node::new_child(
        /* id = */ 1,
        /* key = */ vec![7],
        /* priority = */ 0,
    );
    node.device_value(FULL);
}

#[test]
#[should_panic(expected = "has no value")]
fn take_value_panics_on_an_empty_slot() {
    let mut node: Node<Vec<i64>> = Node::new_child(
        /* id = */ 1,
        /* key = */ vec![7],
        /* priority = */ 0,
    );
    let _ = node.take_device_value(FULL);
}

#[test]
fn try_accessors_return_none_when_unset() {
    let mut node: Node<Vec<i64>> = Node::new_child(
        /* id = */ 1,
        /* key = */ vec![7],
        /* priority = */ 0,
    );
    assert!(node.try_device_value(FULL).is_none());
    assert!(node.try_host_value(FULL).is_none());
    node.set_host_value(FULL, Tensor::from_slice(&[9i64]));
    assert!(node.try_device_value(FULL).is_none());
    assert_eq!(
        Vec::<i64>::try_from(node.try_host_value(FULL).unwrap()).unwrap(),
        vec![9]
    );
}

#[test]
fn redistribute_child_value_splits_head_and_tail() {
    let mut parent: Node<Vec<i64>> = Node::new_child(
        /* id = */ 1,
        /* key = */ vec![7, 8],
        /* priority = */ 0,
    );
    let mut child: Node<Vec<i64>> = Node::new_child(
        /* id = */ 2,
        /* key = */ vec![9],
        /* priority = */ 0,
    );
    // A mid-split child briefly holds its full pre-split value under the tail key.
    child.state_mut_(ValueSlotIdx::device(FULL)).value = Some(Tensor::from_slice(&[1i64, 2, 3]));
    Node::redistribute_child_device_value(&mut parent, &mut child, FULL, /* split_len = */ 2);
    assert_eq!(
        Vec::<i64>::try_from(parent.device_value(FULL)).unwrap(),
        vec![1, 2]
    );
    assert_eq!(
        Vec::<i64>::try_from(child.device_value(FULL)).unwrap(),
        vec![3]
    );
}

#[test]
#[should_panic(expected = "out of range")]
fn redistribute_child_value_rejects_a_boundary_split() {
    let mut parent: Node<Vec<i64>> = Node::new_child(
        /* id = */ 1,
        /* key = */ vec![7, 8, 9],
        /* priority = */ 0,
    );
    let mut child: Node<Vec<i64>> = Node::new_child(
        /* id = */ 2,
        /* key = */ vec![10],
        /* priority = */ 0,
    );
    child.state_mut_(ValueSlotIdx::device(FULL)).value = Some(Tensor::from_slice(&[1i64, 2, 3]));
    Node::redistribute_child_device_value(&mut parent, &mut child, FULL, /* split_len = */ 3);
}

#[test]
fn copy_device_lock_ref_copies_between_nodes() {
    let mut src: Node<Vec<i64>> = Node::new_child(
        /* id = */ 1,
        /* key = */ vec![7],
        /* priority = */ 0,
    );
    let mut dst: Node<Vec<i64>> = Node::new_child(
        /* id = */ 2,
        /* key = */ vec![8],
        /* priority = */ 0,
    );
    src.set_lock_ref_(ValueSlotIdx::device(SWA), 3);
    dst.copy_device_lock_ref(SWA, &src);
    assert_eq!(dst.device_lock_ref(SWA), 3);
    assert_eq!(dst.host_lock_ref(SWA), 0);
}

#[test]
fn mamba_single_state_skips_the_key_length_check() {
    let mut node: Node<Vec<i64>> = Node::new_child(
        /* id = */ 1,
        /* key = */ vec![7, 8, 9],
        /* priority = */ 0,
    );
    // One state slot regardless of the key length.
    node.set_device_value(MAMBA, Tensor::from_slice(&[42i64]));
    assert_eq!(node.device_value_len(MAMBA), 1);
}

#[test]
fn present_but_empty_tensor_has_value_true_len_zero() {
    // A zero-length tensor is still "present" (matches Python `value is not None`).
    let mut node: Node<Vec<i64>> = Node::new_root(/* id = */ 0);
    let empty: [i64; 0] = [];
    node.state_mut_(ValueSlotIdx::device(FULL)).value = Some(Tensor::from_slice(&empty));
    assert!(node.has_device_value(FULL));
    assert_eq!(node.device_value_len(FULL), 0);
}

#[test]
#[should_panic(expected = "out of range")]
fn redistribute_child_value_rejects_a_zero_split() {
    let mut parent: Node<Vec<i64>> = Node::new_child(
        /* id = */ 1,
        /* key = */ vec![7, 8, 9],
        /* priority = */ 0,
    );
    let mut child: Node<Vec<i64>> = Node::new_child(
        /* id = */ 2,
        /* key = */ vec![10],
        /* priority = */ 0,
    );
    child.state_mut_(ValueSlotIdx::device(FULL)).value = Some(Tensor::from_slice(&[1i64, 2, 3]));
    Node::redistribute_child_device_value(&mut parent, &mut child, FULL, /* split_len = */ 0);
}

#[test]
#[should_panic(expected = "no value")]
fn redistribute_child_value_panics_when_child_is_value_less() {
    let mut parent: Node<Vec<i64>> = Node::new_child(
        /* id = */ 1,
        /* key = */ vec![7, 8, 9],
        /* priority = */ 0,
    );
    let mut child: Node<Vec<i64>> = Node::new_child(
        /* id = */ 2,
        /* key = */ vec![10],
        /* priority = */ 0,
    );
    Node::redistribute_child_device_value(&mut parent, &mut child, FULL, /* split_len = */ 1);
}

// Node handles and per-slot value state.

#[test]
fn from_idx_round_trips_every_slot() {
    assert_eq!(ValueSlotIdx::from_idx(0), ValueSlotIdx::device(FULL));
    assert_eq!(ValueSlotIdx::from_idx(1), ValueSlotIdx::device(SWA));
    assert_eq!(ValueSlotIdx::from_idx(2), ValueSlotIdx::device(MAMBA));
    assert_eq!(ValueSlotIdx::from_idx(3), ValueSlotIdx::host(FULL));
    assert_eq!(ValueSlotIdx::from_idx(4), ValueSlotIdx::host(SWA));
    assert_eq!(ValueSlotIdx::from_idx(5), ValueSlotIdx::host(MAMBA));
}

#[test]
#[should_panic(expected = "from_idx: 6 is not a value-slot index")]
fn from_idx_panics_out_of_range() {
    ValueSlotIdx::from_idx(NUM_VALUE_SLOTS);
}

// Unigram and bigram child keys.

#[test]
fn raw_token_ids_borrow_unigram_atoms() {
    let raw = <Vec<i64> as ChildKeyType>::raw_token_ids(&[1, 2, 3]);
    assert_eq!(raw.as_ref(), &[1, 2, 3]);
    assert!(matches!(raw, Cow::Borrowed(_)));
}

#[test]
fn raw_token_ids_unzip_overlapping_bigram_atoms() {
    assert_eq!(
        <Vec<(i64, i64)> as ChildKeyType>::raw_token_ids(&[(1, 2), (2, 3), (3, 4)]).as_ref(),
        &[1, 2, 3, 4]
    );
    assert_eq!(
        <Vec<(i64, i64)> as ChildKeyType>::raw_token_ids(&[]).as_ref(),
        &[] as &[i64]
    );
}

#[test]
fn child_key_takes_the_first_page() {
    let key: Vec<i64> = vec![1, 2, 3];
    assert_eq!(key.child_key(/* page_size = */ 1), vec![1]);
    assert_eq!(key.child_key(/* page_size = */ 2), vec![1, 2]);
    let bigram: Vec<(i64, i64)> = vec![(1, 2), (2, 3)];
    assert_eq!(bigram.child_key(/* page_size = */ 1), vec![(1, 2)]);
}

#[test]
fn match_len_rounds_down_to_page_multiples() {
    let key: Vec<i64> = vec![1, 2, 3, 4];
    assert_eq!(
        key.match_len(
            /* start = */ 0,
            &vec![1, 2, 3, 9],
            /* page_size = */ 1
        ),
        3
    );
    assert_eq!(
        key.match_len(
            /* start = */ 0,
            &vec![1, 2, 3, 9],
            /* page_size = */ 2
        ),
        2
    );
    assert_eq!(
        key.match_len(
            /* start = */ 0,
            &vec![9, 2, 3, 4],
            /* page_size = */ 1
        ),
        0
    );
    assert_eq!(
        key.match_len(
            /* start = */ 0,
            &vec![1, 2, 3, 4],
            /* page_size = */ 1
        ),
        4
    );
    // The shorter key bounds the comparison.
    assert_eq!(
        key.match_len(/* start = */ 0, &vec![1, 2], /* page_size = */ 1),
        2
    );
}

#[test]
fn match_len_from_compares_the_tail_at_start() {
    let key: Vec<i64> = vec![1, 2, 3, 4];
    assert_eq!(
        key.match_len(
            /* start = */ 0,
            &vec![1, 2, 9],
            /* page_size = */ 1
        ),
        2
    );
    assert_eq!(
        key.match_len(
            /* start = */ 1,
            &vec![2, 3, 9],
            /* page_size = */ 1
        ),
        2
    );
    assert_eq!(
        key.match_len(/* start = */ 3, &vec![4], /* page_size = */ 1),
        1
    );
    assert_eq!(
        key.match_len(/* start = */ 1, &vec![9], /* page_size = */ 1),
        0
    );
}

#[test]
fn match_len_from_rounds_down_to_page_multiples() {
    let key: Vec<i64> = vec![1, 2, 3, 4, 5];
    assert_eq!(
        key.match_len(
            /* start = */ 1,
            &vec![2, 3, 4, 9],
            /* page_size = */ 2
        ),
        2
    );
    assert_eq!(
        key.match_len(/* start = */ 1, &vec![2, 9], /* page_size = */ 2),
        0
    );
    assert_eq!(
        key.match_len(
            /* start = */ 1,
            &vec![2, 3, 4, 5],
            /* page_size = */ 2
        ),
        4
    );
}

#[test]
fn match_len_from_is_empty_at_the_tail_boundary() {
    let key: Vec<i64> = vec![1, 2, 3];
    assert_eq!(
        key.match_len(
            /* start = */ 3,
            &vec![1, 2, 3],
            /* page_size = */ 1
        ),
        0
    );
}

#[test]
#[should_panic(expected = "match_len: start 4 beyond the key length 3")]
fn match_len_from_panics_beyond_the_key() {
    let key: Vec<i64> = vec![1, 2, 3];
    key.match_len(/* start = */ 4, &vec![1], /* page_size = */ 1);
}

#[test]
fn match_len_from_compares_bigram_atoms() {
    let key: Vec<(i64, i64)> = vec![(1, 2), (2, 3), (3, 4)];
    assert_eq!(
        key.match_len(
            /* start = */ 1,
            &vec![(2, 3), (3, 9)],
            /* page_size = */ 1
        ),
        1
    );
    assert_eq!(
        key.match_len(
            /* start = */ 1,
            &vec![(2, 9)],
            /* page_size = */ 1
        ),
        0
    );
}

#[test]
fn page_at_borrows_the_page_at_start() {
    let key: Vec<i64> = vec![1, 2, 3, 4];
    assert_eq!(key.page_at(/* start = */ 0, /* page_size = */ 2), &[1, 2]);
    assert_eq!(key.page_at(/* start = */ 2, /* page_size = */ 2), &[3, 4]);
    assert_eq!(key.page_at(/* start = */ 3, /* page_size = */ 1), &[4]);
    let bigram: Vec<(i64, i64)> = vec![(1, 2), (2, 3)];
    assert_eq!(
        bigram.page_at(/* start = */ 1, /* page_size = */ 1),
        &[(2, 3)]
    );
}

#[test]
#[should_panic(expected = "page_at: page [2, 4) reaches beyond the key length 3")]
fn page_at_panics_past_the_end() {
    let key: Vec<i64> = vec![1, 2, 3];
    key.page_at(/* start = */ 2, /* page_size = */ 2);
}

#[test]
fn page_at_keys_a_child_map_lookup() {
    let mut children: HashMap<Vec<i64>, usize> = HashMap::new();
    children.insert(vec![3, 4], 7);
    let key: Vec<i64> = vec![1, 2, 3, 4];
    assert_eq!(
        children.get(key.page_at(/* start = */ 2, /* page_size = */ 2)),
        Some(&7)
    );
    assert_eq!(
        children.get(key.page_at(/* start = */ 0, /* page_size = */ 2)),
        None
    );
}

#[test]
fn suffix_takes_the_tail_and_allows_the_boundary() {
    let key: Vec<i64> = vec![1, 2, 3];
    assert_eq!(key.suffix(1), vec![2, 3]);
    assert_eq!(key.suffix(3), Vec::<i64>::new());
    assert_eq!(key.suffix(0), vec![1, 2, 3]);
}

#[test]
#[should_panic(expected = "suffix: start 4 beyond the key length 3")]
fn suffix_panics_beyond_the_key() {
    let key: Vec<i64> = vec![1, 2, 3];
    key.suffix(4);
}

#[test]
fn page_aligned_truncates_to_whole_pages() {
    let key: Vec<i64> = vec![1, 2, 3];
    assert_eq!(key.page_aligned(/* page_size = */ 1), vec![1, 2, 3]);
    assert_eq!(key.page_aligned(/* page_size = */ 2), vec![1, 2]);
    assert_eq!(key.page_aligned(/* page_size = */ 4), Vec::<i64>::new());
}

#[test]
#[should_panic(expected = "child_key: key of 3 atoms is shorter than a page (4)")]
fn child_key_panics_on_a_key_shorter_than_a_page() {
    let key: Vec<i64> = vec![1, 2, 3];
    key.child_key(/* page_size = */ 4);
}

#[test]
fn split_at_partitions_the_key() {
    let key: Vec<i64> = vec![1, 2, 3];
    assert_eq!(key.split_at(/* split_idx = */ 1), (vec![1], vec![2, 3]));
    assert_eq!(key.split_at(/* split_idx = */ 2), (vec![1, 2], vec![3]));
}

#[test]
#[should_panic(expected = "split_at: split_idx 3 out of range (0, 3)")]
fn split_at_panics_on_the_tail_boundary() {
    let key: Vec<i64> = vec![1, 2, 3];
    key.split_at(/* split_idx = */ 3);
}

#[test]
#[should_panic(expected = "split_at: split_idx 0 out of range (0, 3)")]
fn split_at_panics_on_the_head_boundary() {
    let key: Vec<i64> = vec![1, 2, 3];
    key.split_at(/* split_idx = */ 0);
}

#[test]
#[should_panic(expected = "child_key: key of 0 atoms is shorter than a page (1)")]
fn child_key_panics_on_an_empty_key() {
    let key: Vec<i64> = vec![];
    key.child_key(/* page_size = */ 1);
}

#[test]
fn unigram_key_from_passes_ownership_through() {
    let ids = vec![1i64, 2, 3];
    assert!(matches!(
        <Vec<i64>>::key_from(Cow::Borrowed(&ids)),
        Cow::Borrowed(_)
    ));
    assert_eq!(
        <Vec<i64>>::key_from(Cow::Owned(vec![1, 2, 3])).into_owned(),
        vec![1, 2, 3]
    );
}

#[test]
fn bigram_key_from_pairs_overlapping_tokens() {
    assert_eq!(
        <Vec<(i64, i64)>>::key_from(Cow::Owned(vec![1, 2, 3, 4])).into_owned(),
        vec![(1, 2), (2, 3), (3, 4)]
    );
    assert_eq!(
        <Vec<(i64, i64)>>::key_from(Cow::Owned(vec![5, 6])).into_owned(),
        vec![(5, 6)]
    );
}

#[test]
fn bigram_key_from_is_empty_below_one_pair() {
    assert_eq!(
        <Vec<(i64, i64)>>::key_from(Cow::Owned(vec![])).into_owned(),
        Vec::<(i64, i64)>::new()
    );
    assert_eq!(
        <Vec<(i64, i64)>>::key_from(Cow::Owned(vec![7])).into_owned(),
        Vec::<(i64, i64)>::new()
    );
}

// Per-page hash chains.

// Expected values are literals produced by the python native hash
// (mem_cache/utils.py::get_hash_str over cpp_utils/hash_binding.cpp).

#[test]
fn unigram_pages_chain_within_the_node() {
    assert_eq!(
        get_hash_str::<Vec<i64>>(&[1, 2, 3, 4, 5], None, 2),
        vec![
            "34fb5c825de7ca4aea6e712f19d439c1da0c92c37b423936c5f618545ca4fa1f",
            "c57b445f90651b9a650e516ab2238c965b21af35608a31c303e6d9e407f2915c",
            "e1fd781b60f933e64fe17100c521f3130659b4761e25f7b79abcf81ae5aa23cf",
        ]
    );
}

#[test]
fn single_page_covers_the_whole_key() {
    assert_eq!(
        get_hash_str::<Vec<i64>>(&[1, 2, 3], None, 4),
        vec!["4636993d3e1da4e9d6b8f87b79e8f7c6d018580d52661950eabc3845c5897a4d"]
    );
}

#[test]
fn parent_hash_chains_across_nodes() {
    let parent = get_hash_str::<Vec<i64>>(&[1, 2, 3], None, 2);
    assert_eq!(
        parent,
        vec![
            "34fb5c825de7ca4aea6e712f19d439c1da0c92c37b423936c5f618545ca4fa1f",
            "3ac5d352be720e428633fe34fc74591b2b80060a7764e195d8f34068203ae98f",
        ]
    );
    assert_eq!(
        get_hash_str::<Vec<i64>>(&[7, 8], parent.last().map(String::as_str), 2),
        vec!["5d13a4cf14ad5f9dbd0da79d004bd2a339f633be6100067dc3987c1880bcf0dc"]
    );
}

#[test]
fn bigram_atoms_hash_as_word_pairs() {
    // Raw ids [1, 2, 3, 4, 5] as overlapping pairs, two pairs per page.
    let atoms: Vec<(i64, i64)> = vec![(1, 2), (2, 3), (3, 4), (4, 5)];
    assert_eq!(
        get_hash_str::<Vec<(i64, i64)>>(&atoms, None, 2),
        vec![
            "ede8ef26a097f0fd889f9f39f6f5af921370630b164c4a4fa28eb716b2df9269",
            "678c5307f2a02a2aaf8edbeb80ebac13663f70e14a4fc8d5d62b77e702841e4b",
        ]
    );
}

#[test]
fn empty_key_yields_no_pages() {
    assert_eq!(get_hash_str::<Vec<i64>>(&[], None, 2), Vec::<String>::new());
}

#[test]
#[should_panic(expected = "token id does not fit in uint32")]
fn oversized_token_id_is_rejected() {
    get_hash_str::<Vec<i64>>(&[1 << 33], None, 2);
}

#[test]
#[should_panic(expected = "token id does not fit in uint32")]
fn negative_token_id_is_rejected() {
    get_hash_str::<Vec<i64>>(&[-1], None, 2);
}

#[test]
fn empty_prior_hash_chains_nothing() {
    assert_eq!(
        get_hash_str::<Vec<i64>>(&[1, 2, 3], Some(""), 4),
        get_hash_str::<Vec<i64>>(&[1, 2, 3], None, 4)
    );
}

#[test]
fn hash_str_to_int64_takes_the_first_sixteen_hex_chars_signed() {
    let hex = format!("34fb5c825de7ca4a{}", "0".repeat(48));
    assert_eq!(hash_str_to_int64(&hex), 0x34fb5c825de7ca4a_i64);
    assert_eq!(hash_str_to_int64(&"f".repeat(64)), -1);
}

#[test]
#[should_panic(expected = "byte index 16 is out of bounds")]
fn hash_str_to_int64_panics_on_a_short_string() {
    hash_str_to_int64("abc");
}

#[test]
#[should_panic(expected = "hash must be a hex digest")]
fn hash_str_to_int64_panics_on_a_non_hex_string() {
    hash_str_to_int64(&"z".repeat(64));
}

#[test]
#[should_panic(expected = "prior hash contains a non-hex character")]
fn get_hash_str_panics_on_a_non_hex_prior() {
    get_hash_str::<Vec<i64>>(&[1, 2], Some(&"z".repeat(64)), 2);
}

#[test]
#[should_panic(expected = "prior hash must be a 64-char hex digest")]
fn get_hash_str_panics_on_a_wrong_length_prior() {
    get_hash_str::<Vec<i64>>(&[1, 2], Some("abcd"), 2);
}

#[test]
#[should_panic(expected = "page_size must be positive")]
fn get_hash_str_panics_on_a_zero_page_size() {
    get_hash_str::<Vec<i64>>(&[1, 2], None, 0);
}

#[test]
fn split_redistributes_page_hashes() {
    let hashes = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    let (head, tail) = split_node_hash_value(Some(hashes), 4, 2);
    assert_eq!(head, Some(vec!["a".to_string(), "b".to_string()]));
    assert_eq!(tail, Some(vec!["c".to_string()]));
}

#[test]
fn split_of_an_unhashed_node_stays_none() {
    assert_eq!(split_node_hash_value(None, 4, 2), (None, None));
}

// Node arena storage.

fn arena() -> NodeArena<Vec<i64>> {
    NodeArena::new(vec![FULL], /* page_size = */ 1)
}

#[test]
fn reset_installs_protected_valueless_root() -> Result<(), TreeCoreRuntimeError> {
    let arena = arena();
    let root_id = arena.root();
    let node = arena.node(root_id);
    assert!(node.is_root());
    assert!(node.is_leaf());
    assert!(node.values[FULL.idx()].value.is_none());
    assert_eq!(node.values[FULL.idx()].lock_ref, 1);
    assert_eq!(node.priority, i64::MIN);
    assert_eq!(arena.len(), 1);
    Ok(())
}

#[test]
fn reset_seeds_lock_ref_for_each_enabled_component() -> Result<(), TreeCoreRuntimeError> {
    let arena: NodeArena<Vec<i64>> = NodeArena::new(vec![FULL, SWA], /* page_size = */ 1);
    let root_id = arena.root();
    let root = arena.node(root_id);
    assert_eq!(root.values[FULL.idx()].lock_ref, 1);
    assert_eq!(root.values[SWA.idx()].lock_ref, 1);
    assert_eq!(root.values[MAMBA.idx()].lock_ref, 0);
    Ok(())
}

#[test]
fn alloc_free_recycles_slots() -> Result<(), TreeCoreRuntimeError> {
    let mut arena = arena();
    let root = arena.root();
    let a = arena.alloc_child(
        root,
        /* key = */ vec![1, 2],
        /* priority = */ 0,
        /* extra_key = */ None,
    )?;
    assert_eq!(arena.len(), 2);
    arena.free_leaf(a)?;
    assert_eq!(arena.len(), 1);
    let b = arena.alloc_child(
        root,
        /* key = */ vec![3, 4],
        /* priority = */ 0,
        /* extra_key = */ None,
    )?;
    assert_eq!(a, b);
    assert_eq!(arena.node(b).key, vec![3, 4]);
    Ok(())
}

#[test]
fn alloc_detached_reuses_a_freed_slot() -> Result<(), TreeCoreRuntimeError> {
    let mut arena = arena();
    let root = arena.root();
    let a = arena.alloc_child(
        root,
        /* key = */ vec![1],
        /* priority = */ 0,
        /* extra_key = */ None,
    )?;
    arena.free_leaf(a)?;
    let b = arena.alloc_detached(/* priority = */ 0);
    assert_eq!(b, a);
    assert_eq!(arena.len(), 2);
    Ok(())
}

#[test]
fn reset_clears_then_reinstalls_root() -> Result<(), TreeCoreRuntimeError> {
    let mut arena = arena();
    let root = arena.root();
    arena.alloc_child(
        root,
        /* key = */ vec![9],
        /* priority = */ 0,
        /* extra_key = */ None,
    )?;
    assert_eq!(arena.len(), 2);
    arena.reset();
    assert_eq!(arena.len(), 1);
    let root_id = arena.root();
    assert!(arena.node(root_id).is_root());
    Ok(())
}

#[test]
fn get_and_bump_access_counter_is_monotonic() {
    let mut arena = arena();
    // The root's construction consumed tick 1.
    assert_eq!(arena.get_and_bump_access_counter(), 2);
    assert_eq!(arena.get_and_bump_access_counter(), 3);
}

#[test]
fn salted_children_file_under_their_namespace() -> Result<(), TreeCoreRuntimeError> {
    let mut arena = arena();
    let root = arena.root();
    let a = arena.alloc_child(
        root,
        /* key = */ vec![7],
        /* priority = */ 0,
        Some("lora-1"),
    )?;
    assert_eq!(arena.node(a).namespace.extra_key(), Some("lora-1"));
    // The same page key resolves independently per namespace.
    let b = arena.alloc_child(
        root,
        /* key = */ vec![7],
        /* priority = */ 0,
        Some("lora-2"),
    )?;
    let c = arena.alloc_child(
        root,
        /* key = */ vec![7],
        /* priority = */ 0,
        None,
    )?;
    assert_eq!(arena.root_child(Some("lora-1"), &[7]), Some(a));
    assert_eq!(arena.root_child(Some("lora-2"), &[7]), Some(b));
    assert_eq!(arena.root_child(None, &[7]), Some(c));
    assert_eq!(arena.root_child(Some("ghost"), &[7]), None);
    Ok(())
}

#[test]
fn cache_salt_is_a_distinct_child_namespace_dimension() -> Result<(), TreeCoreRuntimeError> {
    let mut arena = arena();
    let root = arena.root();
    let first_namespace = KeyNamespaceRef::new(Some("bc"), Some("a"));
    let second_namespace = KeyNamespaceRef::new(Some("c"), Some("ab"));
    let first = arena.alloc_child_in_namespace(root, vec![7], 0, first_namespace)?;
    let second = arena.alloc_child_in_namespace(root, vec![7], 0, second_namespace)?;

    assert_eq!(
        arena.root_child_in_namespace(first_namespace, &[7]),
        Some(first)
    );
    assert_eq!(
        arena.root_child_in_namespace(second_namespace, &[7]),
        Some(second)
    );
    assert_eq!(arena.root_child(None, &[7]), None);
    assert_eq!(arena.node(first).namespace.as_ref(), first_namespace);
    assert_eq!(arena.node(second).namespace.as_ref(), second_namespace);
    assert_eq!(
        KeyNamespaceRef::new(None, Some("")).to_owned(),
        KeyNamespace::default()
    );
    Ok(())
}

#[test]
fn namespace_hashing_uses_the_cached_digest_but_equality_checks_strings() {
    let long_extra_key = "x".repeat(64 * 1024);
    let long_cache_salt = "y".repeat(64 * 1024);
    let namespace = KeyNamespace::new(Some(&long_extra_key), Some(&long_cache_salt));

    let mut owned_hasher = ByteCountingHasher::default();
    std::hash::Hash::hash(&namespace, &mut owned_hasher);
    assert_eq!(owned_hasher.bytes_written, size_of::<u64>());

    let mut borrowed_hasher = ByteCountingHasher::default();
    std::hash::Hash::hash(&namespace.as_ref(), &mut borrowed_hasher);
    assert_eq!(borrowed_hasher.bytes_written, size_of::<u64>());

    let first = KeyNamespaceRef {
        extra_key: Some("adapter-a"),
        cache_salt: Some("tenant-a"),
        hash: 7,
    };
    let second = KeyNamespaceRef {
        extra_key: Some("adapter-b"),
        cache_salt: Some("tenant-b"),
        hash: 7,
    };
    assert_ne!(first, second);

    let page = vec![1];
    let mut children: ChildMap<Vec<i64>> = ChildMap::with_hasher(RandomState::new());
    children.insert((first.to_owned(), page.clone()), NodeIdx_(1));
    children.insert((second.to_owned(), page.clone()), NodeIdx_(2));
    assert_eq!(
        children.get(&ChildEdgeRef::<Vec<i64>> {
            namespace: first,
            page: &page,
        }),
        Some(&NodeIdx_(1))
    );
    assert_eq!(
        children.get(&ChildEdgeRef::<Vec<i64>> {
            namespace: second,
            page: &page,
        }),
        Some(&NodeIdx_(2))
    );
}

#[test]
fn child_page_lookup_borrows_namespace_and_page_key() -> Result<(), TreeCoreRuntimeError> {
    let mut arena: NodeArena<CountedKey> = NodeArena::new(vec![FULL], /* page_size = */ 2);
    let root = arena.root();
    let default_child = arena.alloc_child(
        root,
        CountedKey(vec![1, 2, 3]),
        /* priority = */ 0,
        /* extra_key = */ None,
    )?;
    let namespaced_child = arena.alloc_child(
        root,
        CountedKey(vec![1, 2, 4]),
        /* priority = */ 0,
        Some("adapter-a"),
    )?;

    COUNTED_KEY_CONSTRUCTIONS.store(0, Ordering::Relaxed);
    for _ in 0..100 {
        assert_eq!(arena.root_child(None, &[1, 2]), Some(default_child));
        assert_eq!(
            arena.root_child(Some("adapter-a"), &[1, 2]),
            Some(namespaced_child)
        );
        assert_eq!(arena.root_child(Some("adapter-b"), &[1, 2]), None);
        assert_eq!(arena.root_child(None, &[2, 3]), None);
    }
    assert_eq!(COUNTED_KEY_CONSTRUCTIONS.load(Ordering::Relaxed), 0);
    Ok(())
}

#[test]
fn node_extra_key_propagates_down_the_chain() -> Result<(), TreeCoreRuntimeError> {
    let mut arena = arena();
    let root = arena.root();
    let child = arena.alloc_child(
        root,
        /* key = */ vec![7],
        /* priority = */ 0,
        Some("chat"),
    )?;
    let grandchild = arena.alloc_child(
        child,
        /* key = */ vec![8],
        /* priority = */ 0,
        /* extra_key = */ None,
    )?;
    assert_eq!(arena.node_extra_key(grandchild), Some("chat"));
    assert_eq!(arena.node_extra_key(arena.root()), None);
    let detached = arena.alloc_detached(/* priority = */ 0);
    assert_eq!(arena.node_extra_key(detached), None);
    Ok(())
}

#[test]
fn reset_clears_namespace_edges() {
    let mut arena = arena();
    let root = arena.root();
    arena
        .alloc_child(
            root,
            /* key = */ vec![7],
            /* priority = */ 0,
            Some("lora-1"),
        )
        .unwrap();
    assert_eq!(arena.len(), 2);
    arena.reset();
    assert_eq!(arena.len(), 1);
    assert!(!arena.namespace_exists(Some("lora-1")));
}

#[test]
fn alloc_child_sets_child_contract() -> Result<(), TreeCoreRuntimeError> {
    let mut arena = arena();
    let root = arena.root();
    let c = arena.alloc_child(
        root,
        /* key = */ vec![1, 2],
        /* priority = */ 7,
        /* extra_key = */ None,
    )?;
    let node = arena.node(c);
    assert!(!node.is_root());
    assert!(node.is_leaf());
    assert_eq!(node.parent, Some(root));
    assert_eq!(node.priority, 7);
    assert_eq!(node.key, vec![1, 2]);
    assert!(node.values[FULL.idx()].value.is_none());
    assert_eq!(node.values[FULL.idx()].lock_ref, 0);
    Ok(())
}

#[test]
fn alloc_stamps_self_id_and_a_fresh_access_tick() -> Result<(), TreeCoreRuntimeError> {
    let mut arena = arena();
    let root = arena.root();
    // Handles mint monotonically and the map resolves each back to its slot.
    assert_eq!(arena.node(root).id, 0);
    let a = arena.alloc_child(
        root,
        /* key = */ vec![1, 2],
        /* priority = */ 0,
        /* extra_key = */ None,
    )?;
    let b = arena.alloc_child(
        root,
        /* key = */ vec![3, 4],
        /* priority = */ 0,
        /* extra_key = */ None,
    )?;
    assert_eq!(arena.node(a).id, 1);
    assert_eq!(arena.node(b).id, 2);
    assert_eq!(arena.resolve(arena.node(a).id), a);
    // Construction stamps strictly increasing ticks: root, then a, then b;
    // both stamps share the node's single construction tick.
    let root_tick = arena.node(root).last_access_counter;
    let a_tick = arena.node(a).last_access_counter;
    let b_tick = arena.node(b).last_access_counter;
    assert!(root_tick > 0);
    assert!(root_tick < a_tick);
    assert!(a_tick < b_tick);
    assert_eq!(arena.node(root).creation_counter, root_tick);
    assert_eq!(arena.node(a).creation_counter, a_tick);
    assert_eq!(arena.node(b).creation_counter, b_tick);
    Ok(())
}

#[test]
fn reset_zeroes_access_counter() {
    let mut arena = arena();
    arena.get_and_bump_access_counter();
    arena.get_and_bump_access_counter();
    arena.reset();
    // The counter restarts and the fresh root's stamp consumes tick 1.
    assert_eq!(arena.get_and_bump_access_counter(), 2);
}

#[test]
fn node_pair_mut_returns_a_live_parent_child_pair() -> Result<(), TreeCoreRuntimeError> {
    let mut arena = arena();
    let root = arena.root();
    let parent = arena.alloc_child(
        root,
        /* key = */ vec![1],
        /* priority = */ 0,
        /* extra_key = */ None,
    )?;
    let child = arena.alloc_child(
        parent,
        /* key = */ vec![2],
        /* priority = */ 0,
        /* extra_key = */ None,
    )?;
    let parent_ext = arena.node(parent).id;
    let child_ext = arena.node(child).id;
    let (parent_node, child_node) = arena.node_pair_mut(parent, child);
    assert_eq!(parent_node.id, parent_ext);
    assert_eq!(child_node.id, child_ext);
    Ok(())
}

#[test]
#[should_panic(expected = "distinct nodes required")]
fn node_pair_mut_panics_on_same_id() {
    let mut arena = arena();
    let root = arena.root();
    let a = arena
        .alloc_child(
            root,
            /* key = */ vec![1],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    let _ = arena.node_pair_mut(a, a);
}

#[test]
#[should_panic(expected = "is not a child of")]
fn node_pair_mut_panics_when_not_parent_and_child() {
    let mut arena = arena();
    let root = arena.root();
    let a = arena
        .alloc_child(
            root,
            /* key = */ vec![1],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    let b = arena
        .alloc_child(
            root,
            /* key = */ vec![2],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    let _ = arena.node_pair_mut(a, b);
}

#[test]
#[should_panic(expected = "live node")]
fn node_pair_mut_panics_on_freed_id() {
    let mut arena = arena();
    let root = arena.root();
    let parent = arena
        .alloc_child(
            root,
            /* key = */ vec![1],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    let child = arena
        .alloc_child(
            parent,
            /* key = */ vec![2],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    arena.free_leaf(child).unwrap();
    let _ = arena.node_pair_mut(parent, child);
}

#[test]
#[should_panic(expected = "out of bounds")]
fn node_pair_mut_panics_on_out_of_bounds_id() {
    let mut arena = arena();
    let root = arena.root();
    let parent = arena
        .alloc_child(
            root,
            /* key = */ vec![1],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    let _ = arena.node_pair_mut(parent, NodeIdx_(999));
}

#[test]
#[should_panic(expected = "node 1 is not allocated")]
fn node_panics_on_a_freed_id() {
    let mut arena = arena();
    let root = arena.root();
    let a = arena
        .alloc_child(
            root,
            /* key = */ vec![1],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    arena.free_leaf(a).unwrap();
    arena.node(a);
}

#[test]
#[should_panic(expected = "node 1 is not allocated")]
fn node_mut_panics_on_a_freed_id() {
    let mut arena = arena();
    let root = arena.root();
    let a = arena
        .alloc_child(
            root,
            /* key = */ vec![1],
            /* priority = */ 0,
            /* extra_key = */ None,
        )
        .unwrap();
    arena.free_leaf(a).unwrap();
    arena.node_mut(a);
}

#[test]
#[should_panic(expected = "id 999 not in [0, 1)")]
fn node_panics_on_an_out_of_bounds_id() {
    let arena = arena(); // one slot: the root
    arena.node(NodeIdx_(999));
}

#[test]
#[should_panic(expected = "id 999 not in [0, 1)")]
fn node_mut_panics_on_an_out_of_bounds_id() {
    let mut arena = arena(); // one slot: the root
    arena.node_mut(NodeIdx_(999));
}

#[test]
fn out_of_bounds_id_returns_out_of_bound_err() {
    let mut arena = arena(); // one slot: the root
    let bogus = NodeIdx_(999);
    assert!(matches!(
        arena.free_leaf(bogus),
        Err(TreeCoreRuntimeError::NodeAccessOutOfBound { id, .. }) if id == bogus
    ));
    assert!(matches!(
        arena.alloc_child(bogus, /* key = */ vec![1], /* priority = */ 0, /* extra_key = */ None),
        Err(TreeCoreRuntimeError::NodeAccessOutOfBound { id, .. }) if id == bogus
    ));
}

#[test]
fn double_free_returns_err() -> Result<(), TreeCoreRuntimeError> {
    let mut arena = arena();
    let root = arena.root();
    let a = arena.alloc_child(
        root,
        /* key = */ vec![1],
        /* priority = */ 0,
        /* extra_key = */ None,
    )?;
    arena.free_leaf(a)?;
    assert!(matches!(
        arena.free_leaf(a),
        Err(TreeCoreRuntimeError::NodeDoubleFree { id }) if id == a
    ));
    // The rejected free did not re-push the slot onto the freelist.
    assert_eq!(arena.len(), 1);
    Ok(())
}

#[test]
fn free_root_returns_err() -> Result<(), TreeCoreRuntimeError> {
    let mut arena = arena();
    let root = arena.root();
    assert!(matches!(
        arena.free_leaf(root),
        Err(TreeCoreRuntimeError::RootNotFreeable { id }) if id == root
    ));
    // The root survives and stays accessible.
    assert!(arena.node(root).is_root());
    assert_eq!(arena.len(), 1);
    Ok(())
}

#[test]
fn free_the_root_returns_err() {
    let mut arena = arena();
    let r = arena.root();
    assert!(matches!(
        arena.free_leaf(r),
        Err(TreeCoreRuntimeError::RootNotFreeable { id }) if id == r
    ));
}

#[test]
fn freeing_the_last_salted_child_drops_the_namespace() -> Result<(), TreeCoreRuntimeError> {
    let mut arena = arena();
    let root = arena.root();
    let a = arena.alloc_child(
        root,
        /* key = */ vec![7],
        /* priority = */ 0,
        Some("chat"),
    )?;
    assert!(arena.namespace_exists(Some("chat")));
    arena.free_leaf(a)?;
    // An emptied salted namespace leaves nothing behind.
    assert!(!arena.namespace_exists(Some("chat")));
    assert_eq!(arena.len(), 1);
    Ok(())
}

#[test]
fn free_node_with_children_returns_err() -> Result<(), TreeCoreRuntimeError> {
    let mut arena = arena();
    let root = arena.root();
    let parent = arena.alloc_child(
        root,
        /* key = */ vec![1],
        /* priority = */ 0,
        /* extra_key = */ None,
    )?;
    // alloc_child wires this under `parent`, so `parent` is no longer a leaf.
    arena.alloc_child(
        parent,
        /* key = */ vec![2],
        /* priority = */ 0,
        /* extra_key = */ None,
    )?;
    assert!(matches!(
        arena.free_leaf(parent),
        Err(TreeCoreRuntimeError::FreeNonLeafNode { id, num_children })
            if id == parent && num_children == 1
    ));
    Ok(())
}

#[test]
fn alloc_child_wires_into_parent() -> Result<(), TreeCoreRuntimeError> {
    let mut arena = arena();
    let root = arena.root();
    let a = arena.alloc_child(
        root,
        /* key = */ vec![1, 2],
        /* priority = */ 0,
        /* extra_key = */ None,
    )?;
    // Both link sides are set: parent.children[first page] -> a, and a.parent -> root.
    assert_eq!(arena.root_child(None, &[1]).as_ref(), Some(&a));
    assert_eq!(arena.node(a).parent, Some(root));
    Ok(())
}

#[test]
fn free_leaf_detaches_from_parent() -> Result<(), TreeCoreRuntimeError> {
    let mut arena = arena();
    let root = arena.root();
    let a = arena.alloc_child(
        root,
        /* key = */ vec![1],
        /* priority = */ 0,
        /* extra_key = */ None,
    )?;
    assert!(arena.root_child(None, &[1]).is_some());
    arena.free_leaf(a)?;
    // The freed leaf is gone from its parent's children.
    assert!(arena.root_child(None, &[1]).is_none());
    assert_eq!(arena.len(), 1);
    Ok(())
}

#[test]
fn children_are_keyed_by_the_first_radix_page() -> Result<(), TreeCoreRuntimeError> {
    let mut arena: NodeArena<Vec<i64>> = NodeArena::new(vec![FULL], /* page_size = */ 1);
    let root = arena.root();
    let a = arena.alloc_child(
        root,
        /* key = */ vec![1, 2, 3],
        /* priority = */ 0,
        /* extra_key = */ None,
    )?;
    assert_eq!(arena.root_child(None, &[1]).as_ref(), Some(&a));
    assert!(arena.root_child(None, &[1, 2, 3]).is_none());
    // The freed leaf unlinks through the same page key.
    arena.free_leaf(a)?;
    assert!(arena.root_child(None, &[1]).is_none());
    Ok(())
}

#[test]
fn siblings_sharing_a_first_page_collide() -> Result<(), TreeCoreRuntimeError> {
    let mut arena: NodeArena<Vec<i64>> = NodeArena::new(vec![FULL], /* page_size = */ 1);
    let root = arena.root();
    arena.alloc_child(
        root,
        /* key = */ vec![1, 2],
        /* priority = */ 0,
        /* extra_key = */ None,
    )?;
    // A radix tree admits one child per page: same first page is a duplicate.
    assert!(
        arena
            .alloc_child(
                root,
                /* key = */ vec![1, 9],
                /* priority = */ 0,
                /* extra_key = */ None
            )
            .is_err()
    );
    Ok(())
}

#[test]
fn page_size_two_keys_children_by_two_atoms() -> Result<(), TreeCoreRuntimeError> {
    let mut arena: NodeArena<Vec<i64>> = NodeArena::new(vec![FULL], /* page_size = */ 2);
    let root = arena.root();
    let a = arena.alloc_child(
        root,
        /* key = */ vec![1, 2, 3, 4],
        /* priority = */ 0,
        /* extra_key = */ None,
    )?;
    assert_eq!(arena.root_child(None, &[1, 2]).as_ref(), Some(&a));
    Ok(())
}

#[test]
#[should_panic(expected = "get_and_batch_bump_access_counter: delta 0 must be positive")]
fn batch_bump_rejects_a_non_positive_delta() {
    let mut arena: NodeArena<Vec<i64>> = NodeArena::new(vec![FULL], /* page_size = */ 1);
    arena.get_and_batch_bump_access_counter(/* delta = */ 0);
}

#[test]
fn batch_bump_reserves_a_tick_range() {
    let mut arena: NodeArena<Vec<i64>> = NodeArena::new(vec![FULL], /* page_size = */ 1);
    let single = arena.get_and_bump_access_counter();
    let newest = arena.get_and_batch_bump_access_counter(/* delta = */ 3);
    assert_eq!(newest, single + 3);
    assert_eq!(arena.get_and_bump_access_counter(), newest + 1);
}

#[test]
fn alloc_child_rejects_duplicate_key() -> Result<(), TreeCoreRuntimeError> {
    let mut arena = arena();
    let root = arena.root();
    arena.alloc_child(
        root,
        /* key = */ vec![1],
        /* priority = */ 0,
        /* extra_key = */ None,
    )?;
    assert!(matches!(
        arena.alloc_child(root, /* key = */ vec![1], /* priority = */ 0, /* extra_key = */ None),
        Err(TreeCoreRuntimeError::DuplicateChildKey { parent, .. }) if parent == arena.node(root).id
    ));
    // The rejected add reserved no slot: root + the first child only.
    assert_eq!(arena.len(), 2);
    Ok(())
}

#[test]
fn failed_alloc_child_mints_no_id_and_keeps_the_freelist() -> Result<(), TreeCoreRuntimeError> {
    let mut arena = arena();
    let root = arena.root();
    arena.alloc_child(
        root,
        /* key = */ vec![1],
        /* priority = */ 0,
        /* extra_key = */ None,
    )?;
    let b = arena.alloc_child(
        root,
        /* key = */ vec![2],
        /* priority = */ 0,
        /* extra_key = */ None,
    )?;
    let last_minted = arena.node(b).id;
    arena.free_leaf(b)?;
    assert_eq!(arena.len(), 2);
    assert!(
        arena
            .alloc_child(
                root,
                /* key = */ vec![1],
                /* priority = */ 0,
                /* extra_key = */ None
            )
            .is_err()
    );
    assert_eq!(arena.len(), 2);
    // The failure leaked neither the peeked freelist slot nor a handle.
    let c = arena.alloc_child(
        root,
        /* key = */ vec![3],
        /* priority = */ 0,
        /* extra_key = */ None,
    )?;
    assert_eq!(c, b);
    assert_eq!(arena.node(c).id, last_minted + 1);
    Ok(())
}

#[test]
#[should_panic(expected = "is not allocated")]
fn resolve_panics_on_a_never_minted_handle() {
    let arena = arena();
    arena.root();
    arena.resolve(1_000_000);
}

#[test]
fn alloc_child_under_freed_parent_returns_err() -> Result<(), TreeCoreRuntimeError> {
    let mut arena = arena();
    let root = arena.root();
    let a = arena.alloc_child(
        root,
        /* key = */ vec![1],
        /* priority = */ 0,
        /* extra_key = */ None,
    )?;
    arena.free_leaf(a)?;
    assert!(matches!(
        arena.alloc_child(a, /* key = */ vec![2], /* priority = */ 0, /* extra_key = */ None),
        Err(TreeCoreRuntimeError::ParentNotAllocated { id }) if id == a
    ));
    // The rejected alloc_child consumed no slot.
    assert_eq!(arena.len(), 1);
    Ok(())
}

#[test]
fn free_reuses_slots_lifo() -> Result<(), TreeCoreRuntimeError> {
    let mut arena = arena();
    let root = arena.root();
    let a = arena.alloc_child(
        root,
        /* key = */ vec![1],
        /* priority = */ 0,
        /* extra_key = */ None,
    )?;
    let b = arena.alloc_child(
        root,
        /* key = */ vec![2],
        /* priority = */ 0,
        /* extra_key = */ None,
    )?;
    assert_eq!(arena.len(), 3);
    arena.free_leaf(a)?;
    arena.free_leaf(b)?;
    assert_eq!(arena.len(), 1);
    // Last freed is reused first.
    assert_eq!(
        arena.alloc_child(
            root,
            /* key = */ vec![3],
            /* priority = */ 0,
            /* extra_key = */ None
        )?,
        b
    );
    assert_eq!(
        arena.alloc_child(
            root,
            /* key = */ vec![4],
            /* priority = */ 0,
            /* extra_key = */ None
        )?,
        a
    );
    assert_eq!(arena.len(), 3);
    Ok(())
}

#[test]
fn arena_supports_bigram_key_type() -> Result<(), TreeCoreRuntimeError> {
    let mut arena: NodeArena<Vec<(i64, i64)>> =
        NodeArena::new(vec![FULL], /* page_size = */ 1);
    let root = arena.root();
    let a = arena.alloc_child(
        root,
        /* key = */ vec![(1, 2), (3, 4)],
        /* priority = */ 0,
        /* extra_key = */ None,
    )?;
    assert_eq!(arena.len(), 2);
    assert_eq!(arena.node(a).key, vec![(1, 2), (3, 4)]);
    Ok(())
}

#[test]
fn id_map_stays_consistent_across_free_and_realloc() -> Result<(), TreeCoreRuntimeError> {
    let mut arena = arena();
    let root = arena.root();
    let a = arena.alloc_child(
        root,
        /* key = */ vec![1],
        /* priority = */ 0,
        /* extra_key = */ None,
    )?;
    let b = arena.alloc_child(
        root,
        /* key = */ vec![2],
        /* priority = */ 0,
        /* extra_key = */ None,
    )?;
    let b_id = arena.node(b).id;
    arena.free_leaf(b)?;
    assert!(arena.try_resolve(b_id).is_none());
    // The freed slot is recycled with a fresh handle; the old one stays dead.
    let c = arena.alloc_child(
        root,
        /* key = */ vec![3],
        /* priority = */ 0,
        /* extra_key = */ None,
    )?;
    assert_eq!(c, b);
    assert_ne!(arena.node(c).id, b_id);
    assert!(arena.try_resolve(b_id).is_none());
    // Every live slot resolves back from its own handle.
    for idx in arena.live_ids().collect::<Vec<_>>() {
        assert_eq!(arena.resolve(arena.node(idx).id), idx);
    }
    let _ = a;
    Ok(())
}

// Eviction-eligible node set.

#[test]
fn add_then_contains_and_len() {
    let mut set = EvictableNodeSet::new();
    assert!(!set.contains(NodeIdx_(3)));
    assert!(set.is_empty());
    set.add(NodeIdx_(3));
    assert!(set.contains(NodeIdx_(3)));
    assert_eq!(set.len(), 1);
    assert!(!set.is_empty());
}

#[test]
fn add_is_idempotent() {
    let mut set = EvictableNodeSet::new();
    set.add(NodeIdx_(5));
    set.add(NodeIdx_(5));
    assert_eq!(set.len(), 1);
    assert_eq!(set.iter().collect::<Vec<_>>(), vec![NodeIdx_(5)]);
}

#[test]
fn discard_removes_the_member() {
    let mut set = EvictableNodeSet::new();
    set.add(NodeIdx_(2));
    set.discard(NodeIdx_(2));
    assert!(!set.contains(NodeIdx_(2)));
    assert_eq!(set.len(), 0);
    assert!(set.is_empty());
}

#[test]
fn add_sequential_ids_grows_the_slot_table_one_by_one() {
    // Arena NodeIds are sequential, so add(node_id == slots.len()) is the common case.
    let mut set = EvictableNodeSet::new();
    set.add(NodeIdx_(0));
    set.add(NodeIdx_(1));
    set.add(NodeIdx_(2));
    assert!(set.contains(NodeIdx_(0)));
    assert!(set.contains(NodeIdx_(1)));
    assert!(set.contains(NodeIdx_(2)));
    assert_eq!(set.len(), 3);
}

#[test]
fn discard_absent_is_noop() {
    let mut set = EvictableNodeSet::new();
    set.add(NodeIdx_(1));
    set.discard(NodeIdx_(9));
    set.discard(NodeIdx_(0));
    assert!(set.contains(NodeIdx_(1)));
    assert_eq!(set.len(), 1);
}

#[test]
fn discard_fixes_up_the_swapped_member_slot() {
    let mut set = EvictableNodeSet::new();
    set.add(NodeIdx_(10));
    set.add(NodeIdx_(20));
    set.add(NodeIdx_(30));
    // Removing the first member swap-moves the tail (30) into its slot.
    set.discard(NodeIdx_(10));
    assert!(!set.contains(NodeIdx_(10)));
    assert!(set.contains(NodeIdx_(20)));
    assert!(set.contains(NodeIdx_(30)));
    assert_eq!(set.len(), 2);
    // The moved member's slot stays consistent for a follow-up discard.
    set.discard(NodeIdx_(30));
    assert!(!set.contains(NodeIdx_(30)));
    assert!(set.contains(NodeIdx_(20)));
    assert_eq!(set.len(), 1);
}

#[test]
fn discard_the_tail_member() {
    let mut set = EvictableNodeSet::new();
    set.add(NodeIdx_(10));
    set.add(NodeIdx_(20));
    set.discard(NodeIdx_(20));
    assert!(set.contains(NodeIdx_(10)));
    assert!(!set.contains(NodeIdx_(20)));
    assert_eq!(set.len(), 1);
}

#[test]
fn re_add_after_discard() {
    let mut set = EvictableNodeSet::new();
    set.add(NodeIdx_(4));
    set.discard(NodeIdx_(4));
    set.add(NodeIdx_(4));
    assert!(set.contains(NodeIdx_(4)));
    assert_eq!(set.len(), 1);
}

#[test]
fn iter_yields_all_members() {
    let mut set = EvictableNodeSet::new();
    set.add(NodeIdx_(10));
    set.add(NodeIdx_(20));
    set.add(NodeIdx_(30));
    set.discard(NodeIdx_(20));
    let mut members = set.iter().collect::<Vec<_>>();
    members.sort_unstable();
    assert_eq!(members, vec![NodeIdx_(10), NodeIdx_(30)]);
}
