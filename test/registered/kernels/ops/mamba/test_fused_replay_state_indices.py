"""fused_replay_state_indices must be bit-identical to the unfused prep.

The unfused reference is the exact op sequence ``_replay_metadata`` used to
launch for the static hybrid pool:

    req_pool_indices[valid_bs:total_bs] = 0        # zero padded rows (side effect)
    mamba_indices = mapping[req_pool_indices]      # get_mamba_indices gather
    # identity v2p translate (static pool)
    mamba_indices[valid_bs:] = -1                  # padding sentinel
    state_indices[:total_bs].copy_(mamba_indices)

The two paths must agree bit-for-bit, INCLUDING the side effect of zeroing the
padded rows of the static ``req_pool_indices`` replay buffer — captured kernels
gather with that buffer, so a non-zeroed padded row is a delayed illegal memory
access, not a visible diff. Both paths run on guard-padded buffers across a
bs x num_padding matrix (non-power-of-two sizes exercise the BS_UPPER masking):

1. the produced state indices are identical over the whole ``[0, total_bs)``
   range (padding sentinel rows included);
2. the ``req_pool_indices`` buffer ends up identical (padded rows zeroed);
3. neither buffer is written beyond ``total_bs`` (guard tails stay intact).
"""

import unittest

import torch

from sglang.kernels.ops.mamba.mamba_state_indices_triton import (
    fused_replay_state_indices,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=5, stage="base-b-kernel-unit", runner_config="1-gpu-large")

# Guard tail appended to every buffer; must stay untouched by both paths.
_GUARD = 8
_GUARD_SENTINEL = -7777
# Poison for the out buffer so unwritten cells inside [0, total_bs) are caught.
_OUT_POISON = -12345

_REQ_POOL_SIZE = 160
_MAMBA_POOL_SIZE = 4096


def _reference_chain(
    req_pool_indices: torch.Tensor,
    mapping: torch.Tensor,
    out_buf: torch.Tensor,
    valid_bs: int,
    total_bs: int,
) -> None:
    """Replicates the _replay_metadata reference ops, in order, in place."""
    req_pool_indices[valid_bs:total_bs] = 0
    mamba_indices = mapping[req_pool_indices[:total_bs]]
    # static pool: _translate_mamba_indices is the identity
    mamba_indices[valid_bs:] = -1
    out_buf[: len(mamba_indices)].copy_(mamba_indices)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA (triton kernel)")
class TestFusedReplayStateIndices(CustomTestCase):
    def _run_case(self, total_bs: int, num_padding: int, seed: int) -> None:
        device = torch.device("cuda")
        gen = torch.Generator(device="cpu").manual_seed(seed)
        valid_bs = total_bs - num_padding

        # Production dtypes: req_pool_indices int64 (static replay buffer),
        # req_index_to_mamba_index_mapping int32, state_indices_list int32.
        req_pool = torch.randint(
            0, _REQ_POOL_SIZE, (total_bs + _GUARD,), generator=gen, dtype=torch.int64
        )
        req_pool[total_bs:] = _GUARD_SENTINEL
        mapping = torch.randint(
            0, _MAMBA_POOL_SIZE, (_REQ_POOL_SIZE,), generator=gen, dtype=torch.int32
        )
        out = torch.full((total_bs + _GUARD,), _OUT_POISON, dtype=torch.int32)

        req_pool_ref = req_pool.to(device)
        req_pool_fused = req_pool.to(device)
        mapping_dev = mapping.to(device)
        out_ref = out.to(device)
        out_fused = out.to(device)

        _reference_chain(
            req_pool_indices=req_pool_ref,
            mapping=mapping_dev,
            out_buf=out_ref,
            valid_bs=valid_bs,
            total_bs=total_bs,
        )
        returned = fused_replay_state_indices(
            req_pool_indices=req_pool_fused,
            mamba_index_mapping=mapping_dev,
            out_state_indices=out_fused,
            valid_bs=valid_bs,
            total_bs=total_bs,
        )
        torch.cuda.synchronize()

        case = f"{total_bs=} {num_padding=} {seed=}"
        # 1. state indices bit-identical over [0, total_bs), sentinels included
        self.assertTrue(
            torch.equal(out_ref[:total_bs], out_fused[:total_bs]),
            f"state indices mismatch ({case}):\n"
            f"  ref   {out_ref[:total_bs].tolist()}\n"
            f"  fused {out_fused[:total_bs].tolist()}",
        )
        # The returned view is what _replay_metadata forwards downstream.
        self.assertTrue(
            torch.equal(returned, out_fused[:total_bs]),
            f"returned view is not the filled buffer ({case})",
        )
        # 2. req_pool_indices side effect bit-identical (padded rows zeroed)
        self.assertTrue(
            torch.equal(req_pool_ref[:total_bs], req_pool_fused[:total_bs]),
            f"req_pool_indices mismatch ({case}):\n"
            f"  ref   {req_pool_ref[:total_bs].tolist()}\n"
            f"  fused {req_pool_fused[:total_bs].tolist()}",
        )
        # Explicit re-statement of the contract, independent of the reference:
        self.assertTrue(
            (req_pool_fused[valid_bs:total_bs] == 0).all(),
            f"padded req_pool_indices rows not zeroed ({case})",
        )
        self.assertTrue(
            (out_fused[valid_bs:total_bs] == -1).all(),
            f"padding sentinel rows not -1 ({case})",
        )
        self.assertFalse(
            (out_fused[:total_bs] == _OUT_POISON).any(),
            f"unwritten cells inside [0, total_bs) ({case})",
        )
        # 3. no out-of-range writes past total_bs (BS_UPPER > total_bs masking)
        for name, buf in (("req_pool", req_pool_fused), ("out", out_fused)):
            expected = _GUARD_SENTINEL if name == "req_pool" else _OUT_POISON
            self.assertTrue(
                (buf[total_bs:] == expected).all(),
                f"{name} guard tail clobbered ({case}): {buf[total_bs:].tolist()}",
            )

    def test_matrix(self):
        # Non-power-of-two sizes (7, 33) exercise the BS_UPPER in_range mask;
        # num_padding sweeps none / one / half / all-but-one padded rows.
        for total_bs in (1, 2, 7, 32, 33):
            paddings = sorted(
                {0, 1, total_bs // 2, total_bs - 1} & set(range(total_bs))
            )
            for num_padding in paddings:
                for seed in (0, 1, 2):
                    with self.subTest(
                        total_bs=total_bs, num_padding=num_padding, seed=seed
                    ):
                        self._run_case(
                            total_bs=total_bs, num_padding=num_padding, seed=seed
                        )

    def test_shared_mamba_slots(self):
        # Multiple requests mapping to the same mamba slot (mapping is not
        # injective in general) must gather identically on both paths.
        mapping_const = torch.full((_REQ_POOL_SIZE,), 3, dtype=torch.int32)
        device = torch.device("cuda")
        total_bs, num_padding = 7, 2
        valid_bs = total_bs - num_padding
        req_pool = torch.arange(total_bs + _GUARD, dtype=torch.int64)
        out = torch.full((total_bs + _GUARD,), _OUT_POISON, dtype=torch.int32)

        req_ref, req_fused = req_pool.to(device), req_pool.to(device)
        out_ref, out_fused = out.to(device), out.to(device)
        mapping_dev = mapping_const.to(device)

        _reference_chain(
            req_pool_indices=req_ref,
            mapping=mapping_dev,
            out_buf=out_ref,
            valid_bs=valid_bs,
            total_bs=total_bs,
        )
        fused_replay_state_indices(
            req_pool_indices=req_fused,
            mamba_index_mapping=mapping_dev,
            out_state_indices=out_fused,
            valid_bs=valid_bs,
            total_bs=total_bs,
        )
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(out_ref, out_fused))
        self.assertTrue(torch.equal(req_ref, req_fused))


if __name__ == "__main__":
    unittest.main()
