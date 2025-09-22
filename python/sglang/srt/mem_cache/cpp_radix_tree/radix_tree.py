from __future__ import annotations

import os
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch
from torch.utils.cpp_extension import load

_abs_path = os.path.dirname(os.path.abspath(__file__))
radix_tree_cpp = load(
    name="radix_tree_cpp",
    sources=[
        f"{_abs_path}/tree_v2_binding.cpp",
        f"{_abs_path}/tree_v2_debug.cpp",
        f"{_abs_path}/tree_v2.cpp",
    ],
    extra_cflags=["-O3", "-std=c++20"],
)

if TYPE_CHECKING:

    class TreeNodeCpp:
        """
        A placeholder for the TreeNode class. Cannot be constructed elsewhere.
        """

    class IOHandle:
        """
        A placeholder for the IOHandle class. Cannot be constructed elsewhere.
        """

    class RadixTreeCpp:
        def __init__(
            self,
            disabled: bool,
            host_size: Optional[int],
            page_size: int,
            write_through_threshold: int,
        ):
            """
            Initializes the RadixTreeCpp instance.
            Args:
                disabled (bool): If True, the radix tree is disabled.
                host_size (Optional[int]): Size of the radix tree on the CPU. None means no CPU tree.
                page_size (int): Size of the page for the radix tree.
                write_through_threshold (int): Threshold for writing through from GPU to CPU.
            """
            self.tree = radix_tree_cpp.RadixTree(  # type: ignore
                disabled, host_size, page_size, write_through_threshold
            )

        def match_prefix(
            self, prefix: List[int]
        ) -> Tuple[List[torch.Tensor], int, TreeNodeCpp, TreeNodeCpp]:
            """
            Matches a prefix in the radix tree.
            Args:
                prefix (List[int]): The prefix to match.
            Returns:
                Tuple[List[torch.Tensor], TreeNodeCpp, TreeNodeCpp]:
                    0. A list of indices that is matched by the prefix on the GPU.
                    1. Sum length of the indices matched on the CPU.
                    2. The last node of the prefix matched on the GPU.
                    3. The last node of the prefix matched on the CPU.
            """
            return self.tree.match_prefix(prefix)

        def evict(self, num_tokens: int) -> List[torch.Tensor]:
            """
            Evicts a number of tokens from the radix tree.
            Args:
                num_tokens (int): The number of tokens to evict.
            Returns:
                List[torch.Tensor]: A list of indices that were evicted.
            """
            return self.tree.evict(num_tokens)

        def lock_ref(self, handle: TreeNodeCpp, lock: bool) -> None:
            """
            Locks or unlocks a reference to a tree node.
            After locking, the node will not be evicted from the radix tree.
            Args:
                handle (TreeNodeCpp): The tree node to lock or unlock.
                lock (bool): If True, locks the node; if False, unlocks it.
            """
            return self.tree.lock_ref(handle, lock)

        def writing_through(
            self, key: List[int], indices: torch.Tensor
        ) -> Tuple[List[Tuple[IOHandle, torch.Tensor, torch.Tensor]], int]:
            """
            Inserts a key-value pair into the radix tree and perform write-through check.
            Args:
                key (List[int]): The key to insert.
                indices (torch.Tensor): The value associated with the key.
            Returns:
                Tuple[List[Tuple[IOHandle, torch.Tensor, torch.Tensor]], int]:
                    0. A list of (IOHandle, device indices, host indices) tuples.
                       These IOhandles require write-through to the CPU in python side.
                    1. The number of indices that are matched on device.
            """
            return self.tree.writing_through(key, indices)

        def loading_onboard(
            self,
            host_node: TreeNodeCpp,
            new_device_indices: torch.Tensor,
        ) -> Tuple[IOHandle, List[torch.Tensor]]:
            """
            Updates the device indices of tree nodes within a range on the tree.
            Args:
                host_node (TreeNodeCpp): The tree node on the host, must be descendant of device_node.
                new_device_indices (torch.Tensor): The new device indices to set.
                    The length of this tensor must be exactly host indices length.
            Returns:
                Tuple[IOHandle, List[torch.Tensor]]:
                    0. An IOHandle that requires loading to the CPU in python side.
                    1. A list of host indices corresponding to the new device indices.
            """
            return self.tree.loading_onboard(host_node, new_device_indices)

        def commit_writing_through(self, handle: IOHandle, success: bool) -> None:
            """
            Commits the write-through process for a tree node.
            Args:
                handle (IOHandle): The IOHandle to commit.
                success (bool): If True, commits the write-through; if False, just indicates failure.
            """
            return self.tree.commit_writing_through(handle, success)

        def commit_loading_onboard(self, handle: IOHandle, success: bool) -> None:
            """
            Commits the load onboard process for tree nodes within a range on the tree.
            Args:
                handle (IOHandle): The IOHandle to commit.
                success (bool): If True, commits the load-onboard; if False, just indicates failure.
            """
            return self.tree.commit_loading_onboard(handle, success)

        def evictable_size(self) -> int:
            """
            Returns the size of the evictable part of the radix tree.
            This is the size of the part that can be evicted from the GPU (ref_count = 0).
            Returns:
                int: The size of the evictable part.
            """
            return self.tree.evictable_size()

        def protected_size(self) -> int:
            """
            Returns the size of the protected part of the radix tree.
            This is the size of the part that cannot be evicted from the GPU (ref_count > 0).
            Returns:
                int: The size of the protected part.
            """
            return self.tree.protected_size()

        def total_size(self) -> int:
            """
            Returns the total size of the radix tree (including CPU nodes).
            Returns:
                int: The total size of the radix tree.
            """
            return self.tree.total_size()

        def reset(self) -> None:
            """
            Resets the radix tree, clearing all nodes and indices.
            """
            return self.tree.reset()

        def debug_print(self) -> None:
            """
            Prints the internal state of the radix tree for debugging purposes.
            """
            return self.tree.debug_print()

else:
    # Real implementation of the classes for runtime
    RadixTreeCpp = radix_tree_cpp.RadixTree
    TreeNodeCpp = object
    IOHandle = object
