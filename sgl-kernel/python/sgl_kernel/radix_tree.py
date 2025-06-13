from typing import TYPE_CHECKING, Any, List, Tuple
import torch
import radix_tree_cpp

TreeNode = Any

if TYPE_CHECKING:
    class RadixTreeCpp(radix_tree_cpp.RadixTree):
        def __init__(
            self,
            disabled: bool,
            use_hicache: bool,
            page_size: int,
            host_size: int,
            write_threshold: int,
        ):
            """
            Initializes the RadixTreeCpp instance.
            Args:
                disabled (bool): If True, the radix tree is disabled.
                use_hicache (bool): If True, uses hierarchical cache.
                page_size (int): Size of the page for the radix tree.
                host_size (int): Size of the host memory tokens.
                write_threshold (int): Threshold for writing through from GPU to CPU.
            """
            super().__init__(disabled, use_hicache, page_size, host_size, write_threshold)
        def insert(self, key: List[int], indices: torch.Tensor) -> Tuple[List[TreeNode], int]:
            """
            Inserts a key-value pair into the radix tree.
            Args:
                key (List[int]): The key to insert.
                indices (torch.Tensor): The value associated with the key.
            Returns:
                Tuple[List[TreeNode], int]:
                    0. A list of tree nodes that need to be written through.
                    1. The number of indices that are matched on device.
            """
            return super().insert(key, indices)
        def match_prefix(self, prefix: List[int]) -> Tuple[List[torch.Tensor], int, TreeNode, TreeNode]:
            """
            Matches a prefix in the radix tree.
            Args:
                prefix (List[int]): The prefix to match.
            Returns:
                Tuple[List[torch.Tensor], TreeNode, TreeNode]:
                    0. A list of indices that is matched by the prefix.
                    1. Number of indices that reside on the GPU.
                    2. The last node of the prefix matched on the GPU.
                    3. The last node of the prefix matched on the CPU.
            """
            return super().match_prefix(prefix)
        def evict(self, num_tokens: int) -> List[torch.Tensor]:
            """
            Evicts a number of tokens from the radix tree.
            Args:
                num_tokens (int): The number of tokens to evict.
            Returns:
                List[torch.Tensor]: A list of indices that were evicted.
            """
            return super().evict(num_tokens)
        def lock_ref(self, handle: TreeNode, lock: bool) -> None:
            """
            Locks or unlocks a reference to a tree node.
            After locking, the node will not be evicted from the radix tree.
            Args:
                handle (TreeNode): The tree node to lock or unlock.
                lock (bool): If True, locks the node; if False, unlocks it.
            """
            return super().lock_ref(handle, lock)
        def start_write_through(self, handle: TreeNode) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Starts the write-through process for a tree node.
            This will just change the inner state of the tree,
            user should perform `write_through` in python side.
            Args:
                handle (TreeNode): The tree node to write through.
            Returns:
                Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                    0. Device indices of the node.
                    1. Host indices of the node.
            """
            return super().start_write_through(handle)
        def commit_write_through(self, handle: TreeNode, success: bool) -> None:
            """
            Commits the write-through process for a tree node.
            Args:
                handle (TreeNode): The tree node to commit.
                success (bool): If True, commits the write-through; if False, just indicates failure.
            """
            return super().commit_write_through(handle, success)
        def evictable_size(self) -> int:
            """
            Returns the size of the evictable part of the radix tree.
            This is the size of the part that can be evicted from the GPU (ref_count = 0).
            Returns:
                int: The size of the evictable part.
            """
            return super().evictable_size()
        def protected_size(self) -> int:
            """
            Returns the size of the protected part of the radix tree.
            This is the size of the part that cannot be evicted from the GPU (ref_count > 0).
            Returns:
                int: The size of the protected part.
            """
            return super().protected_size()
        def total_size(self) -> int:
            """
            Returns the total size of the radix tree (including CPU nodes).
            Returns:
                int: The total size of the radix tree.
            """
            return super().total_size()
else:
    RadixTreeCpp = radix_tree_cpp.RadixTree
