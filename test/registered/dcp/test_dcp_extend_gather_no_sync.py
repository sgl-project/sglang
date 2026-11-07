"""all_gather_kv_cache_for_mha_extend must not synchronize the device.

Regression guard: the per-request reorder used 0-dim CUDA tensors as Python
slice bounds, costing up to 4 * batch_size device-to-host syncs per layer per
extend step. The CPU test pins the row order; this one pins the sync property.
"""

import unittest

import torch

from sglang.srt import runtime_context as rc
from sglang.srt.layers.dcp.comm import all_gather_kv_cache_for_mha_extend
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=8, stage="base-b", runner_config="1-gpu-small")

KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64


class _SlotPool:
    def __init__(self, num_slots: int, device):
        self.kv_a = torch.randn(
            num_slots, 1, KV_LORA_RANK, dtype=torch.bfloat16, device=device
        )
        self.k_pe = torch.randn(
            num_slots, 1, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device=device
        )

    def get_mla_kv_buffer(self, attn_mqa, indices, dst_dtype=None):
        return self.kv_a[indices], self.k_pe[indices]


def _sync_error(fn):
    """The RuntimeError torch raises if `fn` synchronizes, or None."""
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        fn()
    except RuntimeError as exc:
        return exc
    finally:
        torch.cuda.set_sync_debug_mode("default")
        torch.cuda.synchronize()
    return None


@unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA device")
class TestMhaExtendGatherNoSync(CustomTestCase):
    def test_reorder_does_not_sync(self):
        device = torch.device("cuda")
        prefix_lens_cpu = [0, 7, 3, 12]
        extend_lens_cpu = [5, 1, 0, 9]
        pool = _SlotPool(sum(prefix_lens_cpu), device)
        kv_a = torch.randn(
            sum(extend_lens_cpu), KV_LORA_RANK, dtype=torch.bfloat16, device=device
        )
        k_pe = torch.randn(
            sum(extend_lens_cpu),
            1,
            QK_ROPE_HEAD_DIM,
            dtype=torch.bfloat16,
            device=device,
        )

        def _run():
            return all_gather_kv_cache_for_mha_extend(
                pool,
                None,
                torch.arange(sum(prefix_lens_cpu), device=device),
                torch.tensor(prefix_lens_cpu, dtype=torch.int32, device=device),
                prefix_lens_cpu,
                torch.tensor(extend_lens_cpu, dtype=torch.int32, device=device),
                kv_a,
                k_pe,
            )

        # dcp disabled: the collective is an identity cat, so only the
        # reorder (the part under test) runs on the device.
        with rc.get_parallel().override(dcp_enabled=False, dcp_size=1, dcp_rank=0):
            exc = _sync_error(_run)
            self.assertIsNone(exc, f"extend gather synchronized: {exc}")
            out_kv_a, out_k_pe = _run()
        self.assertEqual(out_kv_a.shape[0], sum(prefix_lens_cpu) + sum(extend_lens_cpu))
        self.assertEqual(out_k_pe.shape[0], out_kv_a.shape[0])


if __name__ == "__main__":
    unittest.main()
