"""CPU contracts for native-only sparse offload and actual query geometry."""

import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.budget import (
    resolve_sparse_kv_budget,
)
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.lim import (
    LIMDecodeAdapter,
)
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.mtp import (
    prepare_mtp_rows,
    sparse_query_width,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, stage="base-a")


class TestLIMState(CustomTestCase):
    def options(self, **kwargs):
        return dict(
            request_rows=3,
            graph_batch_capacity=8,
            context_len=32768,
            num_layers=2,
            top_k=2048,
            kv_bytes_per_token=1152,
            **kwargs,
        )

    def test_query_geometry_uses_actual_verify_width(self):
        for width in (1, 2, 3, 4, 5, 8, 14):
            batch = SimpleNamespace(
                forward_mode=ForwardMode.TARGET_VERIFY,
                spec_info=SimpleNamespace(num_tokens_per_req=width),
            )
            self.assertEqual(sparse_query_width(batch), width)
            batch.forward_mode = ForwardMode.DECODE
            self.assertEqual(sparse_query_width(batch), 1)

    def test_missing_verify_geometry_is_rejected(self):
        batch = SimpleNamespace(forward_mode=ForwardMode.TARGET_VERIFY, spec_info=None)
        with self.assertRaisesRegex(ValueError, "actual chain"):
            sparse_query_width(batch)

    def test_budget_defaults_to_configured_union_capacity(self):
        for width in (1, 2, 3, 4, 5, 8, 14):
            budget = resolve_sparse_kv_budget(**self.options(max_query_width=width))
            self.assertEqual(budget.cache_tokens, width * 2048)
            self.assertEqual(budget.selected_kv_bytes, 0)
            self.assertEqual(budget.policy_bytes, 0)

    def test_budget_rejects_insufficient_union_capacity_and_width(self):
        for width in (2, 3, 5, 14):
            with self.assertRaisesRegex(ValueError, "query width"):
                resolve_sparse_kv_budget(
                    **self.options(max_query_width=width, cache_tokens=2048)
                )
        for width in (0, 15):
            with self.assertRaisesRegex(ValueError, "query width"):
                resolve_sparse_kv_budget(**self.options(max_query_width=width))

    def test_graph_padding_grows_scratch_not_request_residency(self):
        small = resolve_sparse_kv_budget(
            **{**self.options(), "graph_batch_capacity": 3}
        )
        padded = resolve_sparse_kv_budget(**self.options())
        self.assertEqual(small.device_kv_bytes, padded.device_kv_bytes)
        self.assertEqual(small.slot_map_bytes, padded.slot_map_bytes)
        self.assertGreater(padded.copy_sfa_bytes, small.copy_sfa_bytes)
        with self.assertRaisesRegex(ValueError, "exceeding"):
            resolve_sparse_kv_budget(
                **self.options(hbm_budget_bytes=small.managed_hbm_bytes)
            )

    def test_adapter_budget_matches_actual_dynamic_query_allocations(self):
        prefix = "sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload."
        for width in (1, 3, 4, 8, 14):
            manager = SimpleNamespace(
                cache_capacity=width * 2048,
                size=3,
                batch_capacity=8,
                max_query_width=width,
                sparse_context_len=2048,
                max_context_len=32768,
                layer_num=2,
                device="cpu",
                store_dtype=torch.bfloat16,
                start_layer=0,
                paged_kv_cache=SimpleNamespace(
                    get_index_k_buffer=lambda _: torch.empty(
                        (256, 128, 1, 128), dtype=torch.bfloat16
                    )
                ),
            )
            pre, post = ModuleType("pre"), ModuleType("post")
            pre.fused_prepare_lim_metadata = Mock()
            post.fused_lim_copy_sfa_metadata = Mock()
            with (
                patch(prefix + "lim.load_fused_lim_operator", return_value=None),
                patch.object(
                    torch.ops.npu,
                    "lim_full_visible_max_query_width",
                    SimpleNamespace(default=lambda: 14),
                    create=True,
                ),
                patch.dict(
                    "sys.modules",
                    {
                        prefix + "npu_lim_input_metadata": pre,
                        prefix + "npu_lim_copy_sfa_metadata": post,
                    },
                ),
            ):
                adapter = LIMDecodeAdapter(manager)
            allocated = sum(
                t.numel() * t.element_size()
                for t in vars(adapter).values()
                if isinstance(t, torch.Tensor) and t is not adapter.key_scale
            )
            budget = resolve_sparse_kv_budget(**self.options(max_query_width=width))
            self.assertEqual(budget.adapter_bytes, allocated)

    def test_tp_suffix_does_not_extend_actual_query_windows(self):
        for width in (2, 3, 5, 8, 14):
            count = 2 * width + 3
            batch = SimpleNamespace(
                forward_mode=ForwardMode.TARGET_VERIFY,
                spec_info=SimpleNamespace(num_tokens_per_req=width),
                req_pool_indices=torch.tensor([1, 0]),
                seq_lens=torch.tensor([100, 0]),
                positions=torch.tensor(
                    list(range(100, 100 + width)) + [0] * (count - width)
                ),
                out_cache_loc=torch.tensor(list(range(width)) + [-1] * (count - width)),
            )
            req, pos, valid = prepare_mtp_rows(batch, count, 3, 1024)
            self.assertEqual(pos.shape, (2, width))
            self.assertEqual(pos[valid].tolist(), list(range(100, 100 + width)))
            self.assertEqual(req[valid].tolist(), [1] * width)


if __name__ == "__main__":
    unittest.main()
