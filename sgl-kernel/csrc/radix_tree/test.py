from typing import List
from sgl_kernel.radix_tree import RadixTreeCpp
import torch


def _indices(l: List[int]):
    return torch.tensor(l, dtype=torch.int64)

@lambda f: ((f() and None) or f) if __name__ == "__main__" else f
def test_radix_tree():
    print("Testing RadixTreeCpp...")
    tree = RadixTreeCpp(
        False,
        False,
        1,
        0,
        0
    )

    tree.debug_print()
    tree.insert([1, 2], _indices([0, 1]))
    tree.insert([1, 3], _indices([2, 3]))
    tree.debug_print()
