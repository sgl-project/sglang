from typing import List
import pytest
from sgl_kernel.radix_tree import RadixTreeCpp
import torch

def _data(x: List[int]) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.int32)

def _merge(x: List[torch.Tensor]) -> torch.Tensor:
    return torch.cat(x) if x else torch.empty((0), dtype=torch.int32)

def test_radix_tree_gpu_simple():
    """
    Test the RadixTreeCpp class with CPU-only operations.
    """
    radix_tree = RadixTreeCpp(
        disabled=False,
        host_size=None,
        page_size=1,
        write_through_threshold=0,
    )

    _, matched_length = radix_tree.writing_through([1, 2, 3, 4, 5], _data([1, 2, 3, 4, 5]))
    assert matched_length == 0, "No indices is matched before writing through"

    _, matched_length = radix_tree.writing_through([1, 2, 3, 5], _data([6, 7, 8, 9]))
    assert matched_length == 3, "Three indices should match after writing through"

    indices_vec, _, _, _ = radix_tree.match_prefix([1, 2, 4])
    indices = _merge(indices_vec)
    assert indices.tolist() == [1, 2], "Matched indices should be [1, 2] (we don't update)"

    # due to LRU, we will evict the [4, 5] node
    indices_vec = radix_tree.evict(1)
    indices = _merge(indices_vec)
    assert indices.tolist() == [4, 5], "With LRU eviction policy, evicted indices should be [4, 5]"

    indices_vec, _, _, _ = radix_tree.match_prefix([1, 2, 3, 4, 5])
    indices = _merge(indices_vec)
    assert indices.tolist() == [1, 2, 3], "Matched indices should be [1, 2, 3] after eviction"

def test_radix_tree_gpu_with_lock():
    radix_tree = RadixTreeCpp(
        disabled=False,
        host_size=None,
        page_size=1,
        write_through_threshold=0,
    )

    radix_tree.writing_through([1, 2, 3, 4, 5], _data([1, 2, 3, 4, 5]))
    radix_tree.writing_through([1, 2, 3, 5], _data([6, 7, 8, 9]))

    _, _, last_device_node, _ = radix_tree.match_prefix([1, 2, 3, 4, 5])

    # lock the node to root
    radix_tree.lock_ref(last_device_node, True)

    # since [1, 2, 3, 4, 5] is locked, they should not be evicted
    # only the newly written [5] (indice = 9) is evictable and is evicted
    assert radix_tree.evictable_size() == 1
    assert radix_tree.protected_size() == 5
    indices_vec = radix_tree.evict(10000)

    indices = _merge(indices_vec)
    assert indices.tolist() == [9], "Wrong evicted indices, should be [9]"
    radix_tree.writing_through([1, 2, 6, 7, 8], _data([10, 11, 12, 13, 15]))

    # unlock the [1, 2, 3, 4, 5] chain
    radix_tree.lock_ref(last_device_node, False)
    assert radix_tree.evictable_size() == 8 # [1, 2] + ([3, 4, 5] | [6, 7, 8])
    assert radix_tree.protected_size() == 0

    radix_tree.reset()
    assert radix_tree.evictable_size() == 0
    assert radix_tree.protected_size() == 0

def test_radix_tree_cpu_write_through():
    radix_tree = RadixTreeCpp(
        disabled=False,
        host_size=6,
        page_size=1,
        write_through_threshold=1, # write through after 1 hit
    )

    # First write, hit count = 0
    write_through_list, _ = radix_tree.writing_through([1, 2, 3, 4, 5], _data([1, 2, 3, 4, 5]))
    assert len(write_through_list) == 0, "No indices should be written through"

    # Second write, hit count = 1, so the first node ([1, 2, 3, 4, 5]) needs writing through
    write_through_list, _ = radix_tree.writing_through(
        [1, 2, 3, 4, 5, 6, 7, 8],
        _data([1, 2, 3, 4, 5, 6, 7, 8])
    )
    assert len(write_through_list) == 1, "Only 1 node need to be written through"

    io_handle, device_indices, host_indices = write_through_list[0]
    assert device_indices.tolist() == [1, 2, 3, 4, 5], "All indices should match"

    # this means the writing through failed, which will free the host indices
    radix_tree.commit_writing_through(io_handle, False)

    # Now, we have node [1, 2, 3, 4, 5] annd [6, 7, 8] in the device
    # Since the host size is 6, we will only write through the first node
    write_through_list, _ = radix_tree.writing_through(
        [1, 2, 3, 4, 5, 6, 7, 8],
        _data([1, 2, 3, 4, 5, 6, 7, 8])
    )
    assert len(write_through_list) == 1, "Only 1 node can be written through"

    io_handle, device_indices, host_indices = write_through_list[0]
    assert device_indices.tolist() == [1, 2, 3, 4, 5], "All indices should match"

    # When undergoing io, the io nodes ([1, 2, 3, 4, 5]) must be locked and not evictable
    assert radix_tree.evictable_size() == 3, "There should be 3 evictable indices"
    assert radix_tree.protected_size() == 5, "There should be 5 protected indices"
    radix_tree.commit_writing_through(io_handle, True)

    # After IO, all indices can be evicted
    assert radix_tree.evictable_size() == 8, "All indices are evictable now"
    assert radix_tree.protected_size() == 0, "All indices are evictable now"

def test_radix_tree_cpu_load_onboard():
    radix_tree = RadixTreeCpp(
        disabled=False,
        host_size=10000,
        page_size=1,
        write_through_threshold=0, # always write through in this case
    )

    # write through the first node
    write_through_list, _ = radix_tree.writing_through(
        [1, 2, 3, 4],
        _data([1, 2, 3, 4])
    )
    assert len(write_through_list) == 1, "Only 1 node can be written through"
    io_handle, device_indices, host_indices = write_through_list[0]
    assert device_indices.tolist() == [1, 2, 3, 4], "All indices should match"
    radix_tree.commit_writing_through(io_handle, True)

    radix_tree.evict(4) # Evict the only node, now it is on cpu only
    assert radix_tree.evictable_size() == 0, "No evictable indices after eviction"
    assert radix_tree.protected_size() == 0, "No protected indices after eviction"

    indices_vec, host_hit_length, device_node, host_node = radix_tree.match_prefix([1, 2, 3])
    assert len(indices_vec) == 0, "No indices on device after eviction"
    assert host_hit_length == 3, "All indices should be matched on device"
    new_device_indices = _data([5, 6, 7])
    assert radix_tree.evictable_size() == 0, "No indices on device after eviction"
    assert radix_tree.protected_size() == 0, "No indices on device after eviction"

    # try to load onboard the node after match_prefix
    io_handle, _ = radix_tree.loading_onboard(host_node, new_device_indices)
    assert radix_tree.evictable_size() == 0, "No evictable indices since [1, 2, 3] is loading"
    assert radix_tree.protected_size() == 3, "3 loading indices are protected during loading"
    radix_tree.commit_loading_onboard(io_handle, True)

    assert radix_tree.evictable_size() == 3, "3 indices are evictable after loading"
    assert radix_tree.protected_size() == 0, "No protected indices after loading"

if __name__ == "__main__":
    pytest.main([__file__])
