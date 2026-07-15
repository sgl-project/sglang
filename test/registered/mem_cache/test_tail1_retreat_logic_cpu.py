"""CPU unit test for the tail<2 retreat DECISION logic (no GPU, no server).
Mirrors the inline logic in MambaRadixCache._match_post_processor so we can lock the edge cases the GPU
aggregated run can't reach — especially R1 (an empty full_untruncated_fill_ids, as in PD-disagg decode,
must NOT retreat). For the upstream PR, extract the inline block into a helper and point this test at it.

Run: /home/onemount/miniconda3/bin/python test_tail1_retreat_logic_cpu.py
"""

from array import array

CHUNK = 64
MIN_SAFE_GDN_EXTEND = 2


class Node:
    def __init__(self, parent, end, has_ckpt):
        self.parent = parent
        self.end = end
        self.mamba_value = (
            end if has_ckpt else None
        )  # checkpoint slot id (truthy) or None


class Req:
    def __init__(self, fill_ids):
        self.full_untruncated_fill_ids = array("q", fill_ids)


def retreat(value, last_node, best_value_len, req, root, cow_mamba=True):
    """EXACT mirror of the patched _match_post_processor retreat block (R1+R2+R4)."""
    if (
        cow_mamba
        and req is not None
        and best_value_len > 0
        and last_node.mamba_value is not None
    ):
        fill = getattr(req, "full_untruncated_fill_ids", None)
        input_len = len(fill) if fill else None  # R1: empty array("q") -> None -> bail
        if input_len is None and hasattr(req, "get_fill_ids"):
            gf = req.get_fill_ids()
            input_len = len(gf) if gf else None
        if input_len is not None and (
            0
            < input_len - sum(len(v) for v in value[:best_value_len])
            < MIN_SAFE_GDN_EXTEND
        ):
            rn, rl = last_node, best_value_len
            while rl > 0:
                rn = rn.parent
                rl -= 1
                if rn is not None and rn.mamba_value is not None:
                    break
            if rl > 0 and rn is not None and rn.mamba_value is not None:
                return rn, rl
            return root, 0
    return last_node, best_value_len


def make_chain(n_chunks):
    """root -> ckpt@64 -> ckpt@128 -> ... ; value[i] = 64-token span per node; all chunk-aligned -> all ckpt."""
    root = Node(None, 0, False)
    nodes = [root]
    value = []
    for i in range(1, n_chunks + 1):
        nodes.append(Node(nodes[-1], i * CHUNK, True))
        value.append([0] * CHUNK)
    return root, nodes, value


def case(name, n_chunks, input_len, expect_retreat_to_len, cow_mamba=True, req=True):
    root, nodes, value = make_chain(n_chunks)
    last_node, bvl = nodes[-1], n_chunks  # full match on all checkpoint nodes
    r = Req(list(range(input_len))) if req else None
    ln, nbvl = retreat(value, last_node, bvl, r, root, cow_mamba)
    matched = sum(len(v) for v in value[:nbvl])
    tail = input_len - matched if req else None
    ok = nbvl == expect_retreat_to_len
    print(
        f"  {'PASS' if ok else 'FAIL'}  {name}: best_value_len {bvl}->{nbvl}  matched={matched}  "
        f"tail_after={tail}  (expected bvl={expect_retreat_to_len})"
    )
    return ok


# Edge cases the aggregated GPU run can't reach; each dict is kwargs for case().
CASES = [
    # full re-send L = k*64 + 1 -> tail_before=1 -> retreat one checkpoint (bvl k -> k-1)
    {
        "name": "tail=1 (L=129=2*64+1) retreats one ckpt",
        "n_chunks": 2,
        "input_len": 129,
        "expect_retreat_to_len": 1,
    },
    {
        "name": "tail=1 (L=705=11*64+1) retreats one ckpt",
        "n_chunks": 11,
        "input_len": 705,
        "expect_retreat_to_len": 10,
    },
    # tail=2 -> NO retreat (bvl unchanged) -> cache preserved
    {
        "name": "tail=2 (L=130) no retreat",
        "n_chunks": 2,
        "input_len": 130,
        "expect_retreat_to_len": 2,
    },
    {
        "name": "tail=64 (L=192) no retreat",
        "n_chunks": 2,
        "input_len": 192,
        "expect_retreat_to_len": 2,
    },
    # tail=0 -> perfect cache hit -> NO retreat (bvl unchanged)
    {
        "name": "tail=0 (L=128) no retreat",
        "n_chunks": 2,
        "input_len": 128,
        "expect_retreat_to_len": 2,
    },
    # single checkpoint + tail=1 -> no earlier ckpt -> fresh fallback (bvl=0)
    {
        "name": "single ckpt + tail=1 (L=65) -> fresh",
        "n_chunks": 1,
        "input_len": 65,
        "expect_retreat_to_len": 0,
    },
    # R1: empty full_untruncated_fill_ids (disagg decode shape) -> NO retreat despite cow_mamba
    {
        "name": "R1: empty fill_ids (disagg) -> NO retreat",
        "n_chunks": 11,
        "input_len": 0,
        "expect_retreat_to_len": 11,
    },
    # not a mamba COW match -> untouched
    {
        "name": "cow_mamba=False -> untouched",
        "n_chunks": 11,
        "input_len": 705,
        "expect_retreat_to_len": 11,
        "cow_mamba": False,
    },
    {
        "name": "req=None -> untouched",
        "n_chunks": 11,
        "input_len": 705,
        "expect_retreat_to_len": 11,
        "req": False,
    },
]


def test_tail1_retreat_edge_cases():
    """Pytest entry point: every tail<2 retreat edge case must hold."""
    assert all(case(**c) for c in CASES)


if __name__ == "__main__":
    print("tail<2 retreat decision — CPU edge-case lock:")
    results = [case(**c) for c in CASES]
    print(f"\n{sum(results)}/{len(results)} passed")
    assert all(results), "edge-case lock FAILED"
    print(
        "OK — tail=1 retreats, tail>=2 preserved, single-ckpt falls back fresh, empty-fill_ids/non-mamba untouched."
    )
