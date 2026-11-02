"""CPU contracts for physical-slot workspaces and scale-once attention reads."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.layers.attention import flashinfer_backend as fi
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=25, suite="base-a-test-cpu")


@pytest.mark.parametrize("ragged", [False, True])
def test_shared_prefix_uses_physical_slots_and_preserves_workspace_addresses(ragged):
    pool = object.__new__(MHATokenToKVPool)
    size, batch_size, dim = 160, 13, 8
    raw = torch.arange(size * dim, dtype=torch.float32).reshape(size, 1, dim)
    pool.get_raw_kv_buffer = lambda layer: (
        raw,
        raw + 10000,
        torch.ones(size, 1, 1),
        torch.ones(size, 1, 1),
    )
    pool.quant_method = SimpleNamespace(
        dequantize_prev_kv=lambda k, ks, v, vs, layer: (k + 1, v + 2)
    )
    pool.dq_k_buffer = torch.full_like(raw, float("nan"))
    pool.dq_v_buffer = torch.full_like(raw, float("nan"))
    before = (pool.dq_k_buffer.data_ptr(), pool.dq_v_buffer.data_ptr())
    requests = torch.zeros(batch_size, 84, dtype=torch.int32)
    prefix_lens = [80] * batch_size
    extend_lens = [i % 3 + 1 for i in range(batch_size)]
    for i, count in enumerate(extend_lens):
        requests[i, :80] = torch.arange(8, 88)
        requests[i, 80 : 80 + count] = torch.arange(100 + i * 4, 100 + i * 4 + count)
    current = (
        torch.arange(sum(extend_lens) * dim, dtype=torch.float32).reshape(-1, 1, dim)
        + 20000
    )
    # Logical prefix count exceeds the physical pool: compaction per request
    # would overflow even though every referenced physical slot fits.
    assert sum(prefix_lens) > size
    k, v = pool._prepare_dequant_extend_workspace(
        0,
        0,
        requests,
        list(range(batch_size)),
        prefix_lens,
        extend_lens,
        64,
        k_cur_fp8=None if ragged else current,
        v_cur_fp8=None if ragged else current + 1000,
    )
    assert (k.data_ptr(), v.data_ptr()) == before and k.shape == raw.shape
    torch.testing.assert_close(k[8:88], raw[8:88] + 1)
    torch.testing.assert_close(v[8:88], raw[8:88] + 10002)
    assert torch.isnan(k[:8]).all() and torch.isnan(v[88:100]).all()
    offset = 0
    for i, count in enumerate(extend_lens):
        loc = requests[i, 80 : 80 + count].long()
        if ragged:
            assert torch.isnan(k[loc]).all() and torch.isnan(v[loc]).all()
        else:
            torch.testing.assert_close(k[loc], current[offset : offset + count])
            torch.testing.assert_close(v[loc], current[offset : offset + count] + 1000)
        offset += count


@pytest.mark.parametrize("ragged", [False, True])
def test_metadata_uses_normal_physical_page_table(ragged):
    backend = object.__new__(fi.FlashInferAttnBackend)
    backend.prefill_uses_dequant_workspace = True
    backend.page_size = 64
    batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        req_pool_indices=torch.tensor([2, 1, 0]),
        extend_prefix_lens_cpu=[80, 80, 80],
        seq_lens_cpu=torch.tensor([82, 83, 84]),
    )
    backend._prepare_dequant_workspace_metadata_for_extend(batch, ragged)
    assert backend.dq_page_table is None and backend.dq_paged_kernel_lens is None
    assert backend.cpu_req_pool_indices.tolist() == [2, 1, 0]


@pytest.mark.parametrize("route", ["prefill", "ragged", "verify", "draft", "decode"])
@pytest.mark.parametrize("workspace", [True, False])
def test_attention_scales_match_the_kv_representation(route, workspace):
    backend = object.__new__(fi.FlashInferAttnBackend)
    backend.num_wrappers = 1
    backend.use_sliding_window_kv_pool = False
    backend.is_dllm_model = False
    backend.prefill_uses_dequant_workspace = workspace
    backend.decode_uses_dequant_workspace = workspace
    backend.page_size = 64
    backend.cpu_req_pool_indices = torch.tensor([0])
    backend.req_to_token_pool = SimpleNamespace(
        req_to_token=torch.arange(64).view(1, 64)
    )
    backend.kv_cache_quant_method = SimpleNamespace(needs_global_scale=lambda: True)
    wrapper = Mock()
    backend.forward_metadata = (
        fi.DecodeMetadata([wrapper])
        if route == "decode"
        else fi.PrefillMetadata([wrapper], route == "ragged", False)
    )
    generator = torch.Generator().manual_seed(43356)
    q = torch.randn(1, 1, 8, generator=generator)
    k = torch.randn(3, 1, 8, generator=generator)
    v = torch.randn(3, 1, 8, generator=generator)
    # Plain FP8 is scaled on read; a workspace already contains real values.
    scales = (0.125, 0.25)
    cache = (k, v) if workspace else (k / scales[0], v / scales[1])
    pool = SimpleNamespace(
        get_kv_buffer=Mock(return_value=cache),
        get_flashinfer_dequant_workspace_kv_buffer=Mock(return_value=cache),
        get_flashinfer_speculative_dequant_workspace_kv_buffer=Mock(return_value=cache),
        get_flashinfer_decode_dequant_workspace_kv_buffer=Mock(return_value=cache),
        set_kv_buffer=Mock(),
    )
    backend.token_to_kv_pool = pool
    layer = SimpleNamespace(
        layer_id=0,
        logit_cap=0.0,
        is_cross_attention=False,
        attn_type=AttentionType.DECODER,
        tp_q_head_num=1,
        tp_k_head_num=1,
        tp_v_head_num=1,
        head_dim=8,
        scaling=8**-0.5,
        sliding_window_size=-1,
        k_scale_float=scales[0],
        v_scale_float=scales[1],
    )
    modes = {
        "prefill": ForwardMode.EXTEND,
        "ragged": ForwardMode.EXTEND,
        "verify": ForwardMode.TARGET_VERIFY,
        "draft": ForwardMode.DRAFT_EXTEND_V2,
        "decode": ForwardMode.DECODE,
    }
    batch = SimpleNamespace(
        forward_mode=modes[route],
        out_cache_loc=torch.tensor([2]),
        out_cache_loc_is_physical=True,
        req_pool_indices=torch.tensor([0]),
        seq_lens=torch.tensor([3]),
        seq_lens_cpu=torch.tensor([3]),
        extend_prefix_lens_cpu=[2],
        extend_seq_lens_cpu=[1],
        spec_info=SimpleNamespace(num_tokens_per_req=1),
    )

    def reference(query, keys, values):
        weights = torch.softmax(query[:, 0] @ keys[:, 0].T * layer.scaling, dim=-1)
        return (weights @ values[:, 0]).view_as(query)

    def attend(query, kv, **kwargs):
        return reference(query, kv[0] * kwargs["k_scale"], kv[1] * kwargs["v_scale"])

    wrapper.forward.side_effect = attend
    wrapper.forward_return_lse.side_effect = lambda *args, **kw: (
        attend(*args, **kw),
        torch.zeros(1, 1),
    )
    backend.prefill_wrapper_ragged = Mock()
    backend.prefill_wrapper_ragged.forward_return_lse.return_value = (
        torch.zeros_like(q),
        torch.zeros(1, 1),
    )
    # Test the cached-prefix contribution independently of the ragged merge.
    with patch.object(
        fi,
        "_safe_merge_state",
        create=True,
        side_effect=lambda o1, s1, o2, s2: (o2, s2),
    ):
        method = backend.forward_decode if route == "decode" else backend.forward_extend
        result = method(q, k[-1:], v[-1:], layer, batch, save_kv_cache=False)
    torch.testing.assert_close(result, reference(q, k, v).view(1, 8), rtol=0, atol=0)
