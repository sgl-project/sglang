import unittest

import torch

from sglang.multimodal_gen.runtime.models.dits.zimage import ZImageTransformer2DModel


class _RopeCacheHost(torch.nn.Module):
    """The Z-Image DiT's RoPE caches on a module without weights."""

    _get_rope_cache = ZImageTransformer2DModel._get_rope_cache
    _get_unified_freqs_cis = ZImageTransformer2DModel._get_unified_freqs_cis
    _pin_for_active_capture = ZImageTransformer2DModel._pin_for_active_capture

    def __init__(self):
        super().__init__()
        self._rope_caches = {}
        self._unified_freqs_cis = None


def _table(seed, rows, dim=16):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return tuple(
        torch.randn(rows, dim, device="cuda", generator=generator) for _ in range(2)
    )


def _expected_cache(table):
    return torch.cat([table[0].float(), table[1].float()], dim=-1)


@unittest.skipUnless(torch.cuda.is_available(), "the RoPE caches are CUDA-only")
class TestZImageRopeCache(unittest.TestCase):
    def test_new_table_at_a_freed_address_is_not_served_stale(self):
        # the server warmup's table is freed and the next request of the same
        # length allocates its own at that address
        host = _RopeCacheHost()
        host._get_rope_cache("unified", _table(0, 64))
        table = _table(1, 64)
        cache, positions = host._get_rope_cache("unified", table)
        self.assertTrue(torch.equal(cache, _expected_cache(table)))
        self.assertIsNone(positions)

    def test_unified_tables_are_built_once_per_request(self):
        host = _RopeCacheHost()
        x_freqs, cap_freqs = _table(0, 48), _table(1, 16)
        unified = host._get_unified_freqs_cis(x_freqs, cap_freqs)
        self.assertIs(host._get_unified_freqs_cis(x_freqs, cap_freqs), unified)
        for got, image, caption in zip(unified, x_freqs, cap_freqs):
            self.assertTrue(torch.equal(got, torch.cat([image, caption])))
        cache = host._get_rope_cache("unified", unified)
        self.assertIs(host._get_rope_cache("unified", unified), cache)

        # the next request has the same unified length from another split
        x_freqs, cap_freqs = _table(2, 32), _table(3, 32)
        unified = host._get_unified_freqs_cis(x_freqs, cap_freqs)
        for got, image, caption in zip(unified, x_freqs, cap_freqs):
            self.assertTrue(torch.equal(got, torch.cat([image, caption])))
        cache, _ = host._get_rope_cache("unified", unified)
        self.assertTrue(torch.equal(cache, _expected_cache(unified)))


if __name__ == "__main__":
    unittest.main()
