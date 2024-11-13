use sglang_router_rs::tree::RadixTree;

#[test]
fn test_new_tree() {
    let tree = RadixTree::new();
    assert_eq!(tree.root.count, 0);
    assert!(tree.root.children.is_empty());
    assert!(tree.root.ids.is_empty());
}

#[test]
fn test_single_insertion() {
    let mut tree = RadixTree::new();
    tree.insert(&[1, 2, 3]);

    assert_eq!(tree.root.count, 1);
    assert_eq!(tree.root.children.len(), 1);
    assert_eq!(tree.root.children[&1].ids, vec![1, 2, 3]);
    assert_eq!(tree.root.children[&1].count, 1);
}

#[test]
fn test_multiple_insertions_no_split() {
    let mut tree = RadixTree::new();
    tree.insert(&[1, 2, 3]);
    tree.insert(&[4, 5, 6]);

    assert_eq!(tree.root.count, 2);
    assert_eq!(tree.root.children.len(), 2);
    assert_eq!(tree.root.children[&1].ids, vec![1, 2, 3]);
    assert_eq!(tree.root.children[&4].ids, vec![4, 5, 6]);
}

#[test]
fn test_insertion_with_split() {
    let mut tree = RadixTree::new();
    tree.insert(&[1, 2, 3, 4]);
    tree.insert(&[1, 2, 5, 6]);

    assert_eq!(tree.root.count, 2);
    assert_eq!(tree.root.children.len(), 1);
    assert_eq!(tree.root.children[&1].ids, vec![1, 2]);
    assert_eq!(tree.root.children[&1].children.len(), 2);
    assert_eq!(tree.root.children[&1].children[&3].ids, vec![3, 4]);
    assert_eq!(tree.root.children[&1].children[&5].ids, vec![5, 6]);
}

#[test]
fn test_prefix_match_exact() {
    let mut tree = RadixTree::new();
    tree.insert(&[1, 2, 3, 4]);

    assert_eq!(tree.prefix_match(&[1, 2, 3, 4]), &[1, 2, 3, 4]);
}

#[test]
fn test_prefix_match_partial() {
    let mut tree = RadixTree::new();
    tree.insert(&[1, 2, 3, 4]);

    assert_eq!(tree.prefix_match(&[1, 2, 3, 5]), &[1, 2, 3]);
    assert_eq!(tree.prefix_match(&[1, 2, 5]), &[1, 2]);
    assert_eq!(tree.prefix_match(&[1, 5]), &[1]);
}

#[test]
fn test_prefix_match_no_match() {
    let mut tree = RadixTree::new();
    tree.insert(&[1, 2, 3, 4]);
    let empty_slices: &[u32] = &[];
    assert_eq!(tree.prefix_match(&[5, 6, 7]), empty_slices);
}

#[test]
fn test_delete_leaf() {
    let mut tree = RadixTree::new();
    tree.insert(&[1, 2, 3]);
    tree.delete(&[1, 2, 3]);

    assert_eq!(tree.root.count, 0);
    assert_eq!(tree.root.children.len(), 0);
}

#[test]
fn test_delete_with_siblings() {
    let mut tree = RadixTree::new();
    tree.insert(&[1, 2, 3]);
    tree.insert(&[1, 2, 4]);
    tree.delete(&[1, 2, 3]);

    assert_eq!(tree.root.count, 1);
    assert_eq!(tree.root.children[&1].children[&4].ids, vec![4]);
}

#[test]
fn test_multiple_operations() {
    let mut tree = RadixTree::new();

    // Insert several paths
    tree.insert(&[1, 2, 3]);
    tree.insert(&[1, 2, 4]);
    tree.insert(&[1, 5, 6]);

    // Verify structure
    assert_eq!(tree.root.count, 3);
    assert_eq!(tree.prefix_match(&[1, 2, 3]), &[1, 2, 3]);
    assert_eq!(tree.prefix_match(&[1, 2, 4]), &[1, 2, 4]);
    assert_eq!(tree.prefix_match(&[1, 5, 6]), &[1, 5, 6]);

    // Delete and verify
    tree.delete(&[1, 2, 3]);
    assert_eq!(tree.root.count, 2);
    assert_eq!(tree.prefix_match(&[1, 2, 3]), &[1, 2]); // Now only matches prefix
}

#[test]
#[should_panic(expected = "No match found")]
fn test_delete_nonexistent() {
    let mut tree = RadixTree::new();
    tree.insert(&[1, 2, 3]);
    tree.delete(&[4, 5, 6]); // Should panic
}

#[test]
fn test_empty_input() {
    let mut tree = RadixTree::new();
    let empty_slice: &[u32] = &[];
    tree.insert(empty_slice);
    assert_eq!(tree.prefix_match(empty_slice), empty_slice);
    tree.delete(empty_slice); // Should not panic
}
