"""CPU contracts for DSpark's A3 cache ownership and speculative C2 ring.

The C2 test models CompressorV2's documented absolute-position cycle writes;
it does not substitute for executing the compiled operator on an A3 device.
"""

import dataclasses
import importlib.util
import math
import sys
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")


_support_spec = importlib.util.spec_from_file_location(
    "_dsv41_a3_cache_test_support",
    Path(__file__).with_name("test_dsv41_a3_cache.py"),
)
_support = importlib.util.module_from_spec(_support_spec)
sys.modules[_support_spec.name] = _support
_support_spec.loader.exec_module(_support)


def _pair_pool(num_draft_tokens):
    state = _support._load_definitions(
        _support.STATE_FILE,
        {"_lcm", "KVAndScore", "CompressStatePool"},
        {
            "__name__": __name__,
            "dataclasses": dataclasses,
            "gcd": math.gcd,
            "np": np,
            "torch": torch,
            "nullcontext": nullcontext,
            "GPU_MEMORY_TYPE_KV_CACHE": "kv_cache",
            "TorchMemorySaverAdapter": SimpleNamespace(
                create=lambda **_: SimpleNamespace(region=lambda _: nullcontext())
            ),
            "maybe_init_custom_mem_pool": lambda **_: (False, None, None),
        },
    )
    ring = _support._load_definitions(
        _support.ROOT / "python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py",
        {"get_compress_state_ring_size"},
        {},
    )["get_compress_state_ring_size"](2, num_draft_tokens)
    classes = _support._pool_classes()
    classes.update(
        CompressStatePool=state["CompressStatePool"],
        get_eager_max_batch_size=lambda _: 8,
    )
    owner = SimpleNamespace(
        num_req_slots=3,
        qk_nope_head_dim=448,
        qk_rope_head_dim=64,
        device="cpu",
        get_ring_size=lambda _: ring,
    )
    return classes["DSV4NPUTokenToKVPool"]._make_pair_state_pool(owner, False)


class TestA3DSparkCache(unittest.TestCase):
    def test_c2_rejected_suffix_does_not_overwrite_committed_pending_pair(self):
        for draft_tokens in (1, 3, 5, 6, 8, 16):
            pool = _pair_pool(draft_tokens)
            ring = pool.ring_size
            cache = pool.state_cache_3d
            self.assertGreaterEqual(ring, draft_tokens + 2)
            self.assertEqual(cache.shape[1:], (ring, 1024))
            self.assertGreaterEqual(cache.shape[0], 8)
            address = cache.data_ptr()
            # Exercise both parities and ring wrap, including a reject-all
            # lower bound (the real DSpark protocol always commits its anchor).
            for start in range(1, 2 * ring + 2):
                for accepted in range(draft_tokens + 1):
                    with self.subTest(
                        draft_tokens=draft_tokens, start=start, accepted=accepted
                    ):
                        cache.fill_(-999)
                        if start % 2:
                            cache[1, (start - 1) % ring].fill_(start - 1)
                        positions = torch.arange(start, start + draft_tokens)
                        req_ids = torch.ones_like(positions)
                        loc = pool.translate_from_req_position_to_state_loc(
                            req_ids, positions
                        )
                        pool.kv_score_buffer.kv_score[loc] = positions[:, None].float()
                        committed_end = start + accepted
                        if committed_end % 2:
                            # The next call reads only this committed even row.
                            self.assertTrue(
                                (
                                    cache[1, (committed_end - 1) % ring]
                                    == committed_end - 1
                                ).all()
                            )
                        # The replacement token overwrites its rejected slot;
                        # the other request's bank remains completely untouched.
                        cache[1, committed_end % ring].fill_(12345)
                        self.assertTrue((cache[2] == -999).all())
                        self.assertEqual(cache.data_ptr(), address)

    def test_uncompressed_draft_cache_is_bf16_on_a3(self):
        classes = _support._pool_classes()
        pool = object.__new__(classes["NPUDeepSeekV4SingleKVPool"])
        pool.layout = _support.NPUDSV4KVLayout.V4_BF16
        pool.store_dtype = torch.bfloat16
        pool.qk_nope_head_dim = 448
        pool.qk_rope_head_dim = 64
        pool.kernel_page_size = 128
        pool.size = 256
        pool.device = "cpu"
        buffer = pool.create_buffer(num_pages=0)
        self.assertEqual(buffer.shape, (3, 128, 1, 512))
        self.assertEqual(buffer.dtype, torch.bfloat16)
        self.assertEqual(pool.bytes_per_page_padded, 128 * 512 * 2)

    def test_accepted_only_bf16_store_keeps_rejected_rows_untouched(self):
        write = _support._pool_classes()["_write_dsv41_bf16_rows"]
        buffer = torch.full((3, 128, 1, 512), -17, dtype=torch.bfloat16)
        address = buffer.data_ptr()
        for accepted in (1, 3, 5):
            buffer.fill_(-17)
            positions = torch.arange(5)
            loc = torch.where(positions < accepted, positions + 128, -1)
            values = positions[:, None].expand(-1, 512).to(torch.bfloat16)
            write(buffer, loc, values)
            rows = buffer.flatten(0, 1)
            self.assertTrue(
                torch.equal(rows[128 : 128 + accepted, 0], values[:accepted])
            )
            self.assertTrue((rows[128 + accepted : 133] == -17).all())
            self.assertEqual(buffer.data_ptr(), address)


if __name__ == "__main__":
    unittest.main()
