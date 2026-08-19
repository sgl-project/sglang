"""The packed Ulysses Q/K/V input exchange must be bit-identical to the
unpacked path. The collective is emulated in-process with exact
``all_to_all_single`` chunk semantics (rank r's j-th chunk goes to rank j's
r-th chunk); the pack kernel and unpack views run unmodified on CUDA."""

import unittest
from unittest.mock import patch

import torch

from sglang.multimodal_gen.runtime.layers import usp as usp_mod

_USP = "sglang.multimodal_gen.runtime.layers.usp"


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestPackedQKVInputA2A(unittest.TestCase):
    def _run_all_ranks(self, fn, world):
        sends, recvs = [], None

        def fake_a2a(x, role=None):
            if recvs is None:  # recording pass
                sends.append(x.detach().clone())
                return torch.empty_like(x)
            return recvs.pop(0).reshape(x.shape)

        with (
            patch(f"{_USP}._usp_all_to_all_single", fake_a2a),
            patch(f"{_USP}.get_ulysses_parallel_world_size", return_value=world),
        ):
            for r in range(world):
                fn(r)
            recvs = [
                torch.cat([s.flatten().chunk(world)[r] for s in sends])
                for r in range(world)
            ]
            return [fn(r) for r in range(world)]  # replay pass

    def test_packed_matches_unpacked_bitwise(self):
        for world, b, s_global, h_global, d in ((4, 1, 128, 8, 64), (2, 2, 48, 6, 32)):
            torch.manual_seed(1234)
            s_local, h_local = s_global // world, h_global // world
            full = [
                torch.randn(
                    b, s_global, h_global, d, dtype=torch.bfloat16, device="cuda"
                )
                for _ in range(3)
            ]
            shards = [
                tuple(t[:, r * s_local : (r + 1) * s_local].contiguous() for t in full)
                for r in range(world)
            ]
            packed = self._run_all_ranks(
                lambda r: usp_mod._usp_input_all_to_all_qkv(*shards[r]), world
            )
            for r in range(world):
                for i in range(3):
                    spec = full[i][:, :, r * h_local : (r + 1) * h_local].contiguous()
                    self.assertTrue(
                        torch.equal(packed[r][i], spec), f"rank{r} qkv[{i}]"
                    )
                    self.assertTrue(packed[r][i].is_contiguous())


if __name__ == "__main__":
    unittest.main()
