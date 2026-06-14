import torch

from sglang.srt.mem_cache.cpp_radix_tree.radix_tree import RadixTreeCpp


def test_insert_and_match():
    print("Initializing RadixTreeCpp...")
    tree = RadixTreeCpp(
        disabled=False,
        host_size=None,  # Hierarchical Cache disabled
        page_size=1,  # Granularity
        write_through_threshold=1,
    )

    key1 = [11, 22, 33]
    indices1 = torch.tensor([100, 200, 300], dtype=torch.int64)

    print("\n--- Test 1: Full Insertion ---")
    print(f"Inserting key {key1} with indices {indices1.tolist()}")
    (
        ongoing_write,
        hit_len,
        new_indices,
        host_hit_len,
        dev_node,
        host_node,
    ) = tree.insert_and_match(key1, indices1)

    print(f"Result hit length prior to insert: {hit_len} (Expected: 0)")

    merged_indices = torch.cat(new_indices).tolist() if new_indices else []
    print(f"Result matched indices: {merged_indices} (Expected: {indices1.tolist()})")

    assert hit_len == 0
    assert merged_indices == indices1.tolist()

    print("\n--- Test 2: Partial overlap / Prefix extension ---")
    key2 = [11, 22, 33, 44, 55]  # shares 11, 22, 33
    indices2 = torch.tensor([100, 200, 300, 400, 500], dtype=torch.int64)
    print(f"Inserting extended key {key2} with indices {indices2.tolist()}")

    (
        ongoing_write2,
        hit_len2,
        new_indices2,
        host_hit_len2,
        dev_node2,
        host_node2,
    ) = tree.insert_and_match(key2, indices2)

    print(f"Result hit length prior to insert: {hit_len2} (Expected: 3)")
    merged_indices2 = torch.cat(new_indices2).tolist() if new_indices2 else []
    print(f"Result matched indices: {merged_indices2} (Expected: {indices2.tolist()})")

    assert hit_len2 == 3
    assert merged_indices2 == indices2.tolist()

    print("\n✅ Verification Successful! The C++ insert_and_match optimization works.")


if __name__ == "__main__":
    test_insert_and_match()
