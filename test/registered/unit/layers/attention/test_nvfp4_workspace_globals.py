"""Physical workspace values must not receive the checkpoint scales twice."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention import flashinfer_backend as backend
from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import NVFP4KVCacheMethod
from sglang.srt.layers.quantization.kvfp4_tensor import NVFP4KVQuantizeUtil
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def attention(q, k, v, **kwargs):
    k = k.float() * (kwargs.get("k_scale") or 1)
    v = v.float() * (kwargs.get("v_scale") or 1)
    scores = torch.einsum("qhd,khd->qhk", q, k) * kwargs["sm_scale"]
    return torch.einsum("qhk,khd->qhd", scores.softmax(-1), v), scores.logsumexp(-1)


class NumericalPagedWrapper:
    def forward_return_lse(self, q, kv, **kwargs):
        return attention(q, *kv, **kwargs)

    def forward(self, q, kv, **kwargs):
        return self.forward_return_lse(q, kv, **kwargs)[0]


class TestWorkspaceGlobals(CustomTestCase):
    def test_physical_prefill_and_cached_prefix_match_reference(self):
        recipe = NVFP4KVCacheMethod.__new__(NVFP4KVCacheMethod)
        recipe.k_scales_gpu, recipe.v_scales_gpu = (
            torch.tensor([2.0]),
            torch.tensor([4.0]),
        )
        packed_k = (
            torch.tensor([1, 3], dtype=torch.uint8).view(2, 1, 1).expand(2, 1, 128)
        )
        packed_v = (
            torch.tensor([2, 5], dtype=torch.uint8).view(2, 1, 1).expand(2, 1, 128)
        )
        with patch.object(
            NVFP4KVQuantizeUtil,
            "dequantize",
            side_effect=lambda x, sf, g: x.float() * sf * g,
        ):
            physical = recipe.dequantize_prev_kv(
                packed_k, torch.ones(2, 1, 128), packed_v, torch.ones(2, 1, 128), 0
            )
        torch.testing.assert_close(physical[0].float(), packed_k.float() * 2)
        torch.testing.assert_close(physical[1].float(), packed_v.float() * 4)
        q = torch.zeros(1, 1, 128)
        q[..., 0] = 1
        fresh_k, fresh_v = torch.ones_like(q), torch.ones_like(q) * 3
        layer = SimpleNamespace(
            layer_id=0,
            tp_q_head_num=1,
            tp_k_head_num=1,
            tp_v_head_num=1,
            head_dim=128,
            logit_cap=0,
            scaling=0.5,
            sliding_window_size=-1,
            is_cross_attention=False,
            attn_type=AttentionType.DECODER,
            k_scale_float=2.0,
            v_scale_float=4.0,
        )

        def merge(a, la, b, lb):
            total = torch.logaddexp(la, lb)
            return a * (la - total).exp()[..., None] + b * (lb - total).exp()[
                ..., None
            ], total

        for workspace in (False, True):
            for ragged in (False, True):
                with self.subTest(workspace=workspace, ragged=ragged):
                    attn = backend.FlashInferAttnBackend.__new__(
                        backend.FlashInferAttnBackend
                    )
                    attn.num_wrappers, attn.prefill_uses_dequant_workspace = (
                        1,
                        workspace,
                    )
                    attn.req_to_token_pool = SimpleNamespace(req_to_token=None)
                    attn.cpu_req_pool_indices, attn.page_size, attn.dq_page_table = (
                        [0],
                        1,
                        torch.tensor([1]),
                    )
                    attn.is_dllm_model = False
                    selected = tuple(x.float() for x in physical)
                    if not ragged:
                        # Paged prefill overlays physical fresh KV; the ragged
                        # route reads only the cached prefix from the workspace.
                        selected = (
                            torch.cat([selected[0], fresh_k]),
                            torch.cat([selected[1], fresh_v]),
                        )
                    kv = selected if workspace else (selected[0] / 2, selected[1] / 4)
                    attn.token_to_kv_pool = SimpleNamespace(
                        get_flashinfer_dequant_workspace_kv_buffer=lambda *a, **kw: kv,
                        get_kv_buffer=lambda _: kv,
                    )
                    attn.prefill_wrapper_ragged = SimpleNamespace(
                        forward_return_lse=lambda q, k, v, **kw: attention(
                            q, k, v, **kw
                        )
                    )
                    attn.forward_metadata = SimpleNamespace(
                        prefill_wrappers=[NumericalPagedWrapper()],
                        use_ragged=ragged,
                        extend_no_prefix=False,
                        multi_item_params=None,
                    )
                    batch = SimpleNamespace(
                        extend_prefix_lens_cpu=[2], extend_seq_lens_cpu=[1]
                    )
                    with (
                        get_parallel().override(attn_tp_size=1, attn_dcp_size=1),
                        patch.object(
                            backend, "_safe_merge_state", side_effect=merge, create=True
                        ),
                    ):
                        actual = attn.forward_extend(
                            q, fresh_k, fresh_v, layer, batch, save_kv_cache=False
                        )
                    keys, values = (x.float() for x in physical)
                    keys, values = (
                        torch.cat([keys, fresh_k]),
                        torch.cat([values, fresh_v]),
                    )
                    expected = attention(q, keys, values, sm_scale=layer.scaling)[0]
                    torch.testing.assert_close(actual, expected.view(1, -1))


if __name__ == "__main__":
    unittest.main()
