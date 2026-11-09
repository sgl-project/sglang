import types
import unittest
from array import array
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers import schedule_batch  # noqa: E402
from sglang.srt.managers.embed_types import PositionalEmbeds  # noqa: E402
from sglang.srt.managers.schedule_batch import ScheduleBatch  # noqa: E402

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")

HIDDEN = 4


def _make_embeds(base, n, device):
    rows = torch.arange(n * HIDDEN, dtype=torch.float32).reshape(n, HIDDEN) + base
    return rows.to(device)


def _make_req(total_len, pre_len, extend_end, positions, base, device="cuda"):
    ids = array("q", range(100, 100 + total_len))
    return types.SimpleNamespace(
        full_untruncated_fill_ids=ids,
        origin_input_ids=ids,
        prefix_len=pre_len,
        extend_end=extend_end,
        extend_len=extend_end - pre_len,
        logprob_start_len=-1,
        extend_batch_idx=0,
        input_embeds=None,
        positional_embed_overrides=PositionalEmbeds(
            embeds=_make_embeds(base, len(positions), device), positions=positions
        ),
        multimodal_inputs=None,
        retracted_stain=False,
        already_computed=0,
        cached_tokens=0,
        _cache_breakdown_computed=False,
        host_hit_length=0,
        storage_hit_length=0,
        storage_hit_start=0,
        host_hit_is_storage=False,
        is_dllm=lambda: False,
        kv=types.SimpleNamespace(holds_mamba=False),
        is_retracted=False,
        materialized_host_hit_len=lambda: 0,
    )


def _run_prepare_for_extend(reqs):
    for r in reqs:
        r.get_fill_ids = lambda r=r: r.full_untruncated_fill_ids[: r.extend_end]
    batch = ScheduleBatch(reqs=reqs)
    batch.device = "cuda"
    batch.model_config = types.SimpleNamespace(
        is_encoder_decoder=False, vocab_size=1000
    )
    n = len(reqs)
    exec_ctx = types.SimpleNamespace(
        features=types.SimpleNamespace(enable_encoder_swa_bounded_replay=False),
        mamba=types.SimpleNamespace(enable_mamba_extra_buffer=False),
    )
    with (
        patch(
            "sglang.srt.managers.schedule_batch.get_exec",
            return_value=exec_ctx,
        ),
        patch(
            "sglang.srt.managers.schedule_batch.alloc_for_extend",
            return_value=(
                torch.zeros(1, dtype=torch.int64, device="cuda"),
                torch.arange(n, dtype=torch.int64, device="cuda"),
                torch.arange(n, dtype=torch.int64),
            ),
        ),
        patch(
            "sglang.srt.managers.schedule_batch.SamplingBatchInfo.from_schedule_batch",
            return_value=None,
        ),
    ):
        with patch.object(
            schedule_batch, "async_h2d", wraps=schedule_batch.async_h2d
        ) as h2d:
            batch.prepare_for_extend()
    torch.cuda.synchronize()
    return batch, h2d.call_count


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestPrepareForExtendEmbedOverrides(CustomTestCase):
    # async_h2d calls: one for the positions, plus one index copy per request
    # whose rows are only partly selected. A blocking copy never calls it.
    def _assert(self, batch, expected_rows, expected_positions):
        self.assertTrue(batch.replace_embeds.is_cuda)
        self.assertTrue(batch.replace_positions.is_cuda)
        self.assertEqual(batch.replace_positions.tolist(), expected_positions)
        self.assertTrue(torch.equal(batch.replace_embeds, expected_rows.to("cuda")))

    def test_all_rows_selected(self):
        req = _make_req(6, 0, 6, [1, 3], base=0)
        src = req.positional_embed_overrides.embeds
        batch, h2d_calls = _run_prepare_for_extend([req])
        self._assert(batch, src.cpu(), [1, 3])
        self.assertEqual(h2d_calls, 1)

    def test_cpu_embeds(self):
        # HTTP requests carry CPU embeds; the rows must still be gathered right.
        full = _make_req(6, 0, 6, [1, 3], base=0, device="cpu")
        full_src = full.positional_embed_overrides.embeds
        batch, h2d_calls = _run_prepare_for_extend([full])
        self._assert(batch, full_src, [1, 3])
        self.assertEqual(h2d_calls, 1)

        subset = _make_req(6, 3, 6, [1, 4, 5], base=0, device="cpu")
        subset_src = subset.positional_embed_overrides.embeds
        batch, _ = _run_prepare_for_extend([subset])
        self._assert(batch, subset_src[[1, 2]], [1, 2])

    def test_subset_from_cached_prefix(self):
        req = _make_req(6, 3, 6, [1, 4, 5], base=0)
        src = req.positional_embed_overrides.embeds.cpu()
        batch, h2d_calls = _run_prepare_for_extend([req])
        self.assertEqual(h2d_calls, 2)
        # Position 1 is inside the cached prefix; 4 and 5 map to 1 and 2.
        self._assert(batch, src[[1, 2]], [1, 2])

    def test_subset_from_chunk_boundary(self):
        req = _make_req(6, 0, 4, [0, 2, 5], base=0)
        src = req.positional_embed_overrides.embeds.cpu()
        batch, h2d_calls = _run_prepare_for_extend([req])
        self.assertEqual(h2d_calls, 2)
        # Position 5 lies beyond this chunk.
        self._assert(batch, src[[0, 1]], [0, 2])

    def test_multiple_requests_with_offsets(self):
        # r0: all rows, 4 extend tokens. r1: cached prefix of 2, subset.
        # r2: chunk boundary, subset.
        r0 = _make_req(4, 0, 4, [0, 3], base=0)
        r1 = _make_req(5, 2, 5, [0, 2, 4], base=100)
        r2 = _make_req(6, 0, 3, [1, 5], base=200)
        s0, s1, s2 = (r.positional_embed_overrides.embeds.cpu() for r in (r0, r1, r2))
        batch, h2d_calls = _run_prepare_for_extend([r0, r1, r2])
        # Index copies for r1 and r2 only, plus the positions.
        self.assertEqual(h2d_calls, 3)
        # input_id_pointer: r0 at 0 (len 4), r1 at 4 (len 3), r2 at 7 (len 3).
        expected_rows = torch.cat([s0, s1[[1, 2]], s2[[0]]], dim=0)
        self._assert(batch, expected_rows, [0, 3, 4, 6, 8])


if __name__ == "__main__":
    unittest.main()
