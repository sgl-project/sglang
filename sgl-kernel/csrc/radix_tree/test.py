from typing import List

import torch
from sgl_kernel.radix_tree import RadixTreeCpp


def _indices(l: List[int]):
    return torch.tensor(l, dtype=torch.int64)


@lambda f: ((f() and None) or f) if __name__ == "__main__" else f
def test_radix_tree():
    print("Testing RadixTreeCpp...")
    tree = RadixTreeCpp(
        disabled=False,
        host_size=100,
        page_size=1,
        write_through_threshold=1,
    )

    def _print():
        print("=" * 80)
        tree.debug_print()
        print("=" * 80)

    def _call(result):
        print(result)
        _print()
        return result

    def _test_basic():
        x, l = _call(tree.writing_through([1, 2], _indices([1, 0])))
        assert l == 0 and len(x) == 0  # match length = 0 ; no writing through

        x, l = _call(tree.writing_through([1, 3, 4, 6], _indices([2, 3, 4, 5])))
        assert l == 1 and len(x) == 1  # match length = 1 ; writing through [1]

        y = _call(tree.evict(10000))
        assert len(y) == 2  # [3, 4, 5] and [0] are evicted

        # in this output, the hit count is not reset,
        # since the write through is committed as failure (False in arguments)
        _call(tree.commit_writing_through(x[0][0], False))

        # after this call, the tree is empty
        y = _call(tree.evict(10000))
        assert len(y) == 1  # [1] is now evicted

    def _test_split_when_writing_through():
        x, l = _call(tree.writing_through([1, 1, 1, 1, 1], _indices([0, 1, 2, 3, 4])))
        assert l == 0 and len(x) == 0  # match length = 0 ; no writing through

        # the indices should not be updated
        x, l = _call(tree.writing_through([1, 1, 1, 1], _indices([4, 5, 6, 7])))
        assert (
            l == 4 and len(x) == 1
        )  # match length = 4 ; writing through first 4 in one node

        a, b, c, d = _call(tree.match_prefix([1, 1, 1, 2, 2]))
        assert c == d  # the same node on CPU and GPU (since no writing through yet)
        assert b == 0  # no node on CPU yet
        assert torch.equal(a[0], torch.tensor([0, 1, 2], dtype=torch.int64))

        # complete the writing through
        _call(tree.commit_writing_through(x[0][0], True))
        _call(tree.reset())

    def _test_split_when_loading_onboard():
        x, l = _call(tree.writing_through([1, 1, 1, 1, 1], _indices([0, 1, 2, 3, 4])))
        assert l == 0 and len(x) == 0  # match length = 0 ; no writing through

        # run this to trigger host writing through
        x, l = _call(tree.writing_through([1, 1, 1, 1, 1], _indices([0, 1, 2, 3, 4])))
        assert (
            l == 5 and len(x) == 1
        )  # match length = 5 ; writing through all in one node

        # successfully writing through
        _call(tree.commit_writing_through(x[0][0], True))

        # evict all nodes. now node reside on CPU
        _call(tree.evict(10000))

        # no writing through, since the node is CPU backed already
        x, l = _call(tree.writing_through([1, 1], _indices([5, 6])))
        assert l == 0 and len(x) == 0  # match length = 0 ; no writing through

        x, l = _call(tree.writing_through([1, 1], _indices([0, 1])))
        assert l == 2 and len(x) == 0  # match length = 2 ; no writing through

        # now part of the node is both on GPU and CPU; while others on CPU only
        a, b, c, d = _call(tree.match_prefix([1, 1, 1, 1, 1]))
        assert c != d  # different node on CPU and GPU
        assert b == 3  # length 5 - 2 = 2 remain on CPU
        assert torch.equal(a[0], torch.tensor([5, 6], dtype=torch.int64))

        # now we can load the onboard node, with new device indice 7, 8, 9
        h, y = _call(tree.loading_onboard(c, d, _indices([7, 8, 9])))
        assert len(y) == 1  # one node is being loaded

        # split the node by calling another shorter match_prefix
        a, b, c, d = _call(tree.match_prefix([1, 1, 1, 1]))
        assert c == d  # the same node on CPU and GPU
        assert b == 0  # no node on CPU yet (on loading will not be counted)
        assert torch.equal(a[0], torch.tensor([5, 6], dtype=torch.int64))
        assert torch.equal(a[1], torch.tensor([7, 8], dtype=torch.int64))

        # commit the IOhandle; complete the loading
        _call(tree.commit_loading_onboard(h, True))
        _call(tree.reset())

    print("-" * 200)
    _test_basic()
    print("-" * 200)
    _test_split_when_writing_through()
    print("-" * 200)
    _test_split_when_loading_onboard()
