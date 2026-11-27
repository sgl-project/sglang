"""Ascend window bounds and physical SWA page mapping through real operators."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace as NS

import pytest
import torch

from sglang.srt.hardware_backend.npu.attention.ascend_backend import (
    AscendAttnBackend,
    AscendAttnMaskBuilder,
)
from sglang.srt.hardware_backend.npu.attention.ascend_torch_native_backend import (
    AscendTorchNativeAttnBackend,
)
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=30, suite="extra-a-test-npu")


def fixture(window, identity=False):
    b = AscendAttnBackend.__new__(AscendAttnBackend)
    for name in (
        "use_mla",
        "use_fia",
        "use_fa",
        "use_quant_flash_attn",
        "use_flash_mla",
        "use_paged_mtp",
        "dense_mla_dcp_target",
        "dsa_dcp_target",
        "enable_sparsity_driven_kv_offload",
        "is_draft_worker",
        "is_dllm_model",
        "use_dsa",
        "use_alibi",
    ):
        setattr(b, name, False)
    b.device = "npu"
    b.page_size = 128
    b.max_context_len = 512
    b.speculative_num_draft_tokens = None
    b.q_head_num_padding = None
    b.attn_cp_size = b.dcp_size = 1
    b.is_hybrid_swa = True
    b.use_sliding_window_kv_pool = True
    b.sliding_window_size = window
    b.needs_cpu_seq_lens = True
    b.native_attn = AscendTorchNativeAttnBackend()
    b.ascend_attn_mask_builder = AscendAttnMaskBuilder(
        NS(dtype=torch.bfloat16), "npu", False, False
    )
    b.mask = b.ascend_attn_mask_builder.mask
    b.fia_mask = b.ascend_attn_mask_builder.fia_mask
    full = torch.arange(128, 640, dtype=torch.int32, device="npu")
    swa = full if identity else full + 384
    b.req_to_token = full[None, :]
    b.req_to_token_pool = NS(req_to_token=b.req_to_token)
    b.full_to_swa_index_mapping = torch.zeros(1152, dtype=torch.int64, device="npu")
    b.full_to_swa_index_mapping[full.long()] = swa.long()
    key = torch.zeros(9, 128, 2, 64, dtype=torch.bfloat16, device="npu")
    value = torch.full_like(key, -100)
    value.view(-1, 2, 64)[swa.long()] = torch.arange(512, device="npu").bfloat16()[
        :, None, None
    ]
    b.token_to_kv_pool = NS(
        get_key_buffer=lambda _: key,
        get_value_buffer=lambda _: value,
        translate_loc_from_full_to_swa=lambda loc: b.full_to_swa_index_mapping[loc],
    )
    layer = NS(
        layer_id=0,
        tp_q_head_num=4,
        tp_k_head_num=2,
        tp_v_head_num=2,
        qk_head_dim=64,
        v_head_dim=64,
        scaling=0.125,
        sliding_window_size=window,
        is_cross_attention=False,
        attn_type=AttentionType.DECODER,
        logit_cap=0,
        logit_capping_method="tanh",
    )
    return b, layer


def batch(length, qlen=1, mode=ForwardMode.EXTEND):
    return NS(
        batch_size=1,
        _original_batch_size=1,
        forward_mode=mode,
        seq_lens=torch.tensor([length], dtype=torch.int64, device="npu"),
        seq_lens_cpu=torch.tensor([length], dtype=torch.int64),
        req_pool_indices=torch.tensor([0], dtype=torch.int64, device="npu"),
        extend_prefix_lens=torch.tensor(
            [length - qlen], dtype=torch.int64, device="npu"
        ),
        extend_seq_lens=torch.tensor([qlen], dtype=torch.int64, device="npu"),
        extend_prefix_lens_cpu=[length - qlen],
        extend_seq_lens_cpu=[qlen],
        encoder_lens=None,
        spec_info=None,
        spec_algorithm=None,
        out_cache_loc=None,
        attn_cp_metadata=None,
        global_num_token_non_padded_cpu=qlen,
    )


def reference(length, qlen, window):
    qp = torch.arange(length - qlen, length, dtype=torch.float64)[:, None]
    kp = torch.arange(length, dtype=torch.float64)[None, :]
    visible = (kp <= qp) & (kp >= qp - window)
    return [(kp[0][row].mean().item()) for row in visible]


@pytest.fixture(autouse=True)
def npu_context():
    torch.npu.set_device(0)
    with get_context().override_server_args(device="npu"):
        yield


def test_graph_metadata_window_and_replay():
    backend, layer = fixture(63)
    backend.init_cuda_graph_state(1, 1)
    fb = batch(63, mode=ForwardMode.DECODE)
    backend.init_forward_metadata_out_graph(fb, in_capture=True)
    q = torch.zeros(1, 4, 64, dtype=torch.bfloat16, device="npu")
    for _ in range(3):
        backend.forward_decode_graph(q, None, None, layer, fb, save_kv_cache=False)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, auto_dispatch_capture=True):
        captured = backend.forward_decode_graph(
            q, None, None, layer, fb, save_kv_cache=False
        )
    for length in (63, 64, 65, 128, 129, 63):
        fb = batch(length, mode=ForwardMode.DECODE)
        backend.init_forward_metadata_out_graph(fb, in_capture=False)
        with ThreadPoolExecutor(
            max_workers=1, initializer=torch.npu.set_device, initargs=(0,)
        ) as executor:
            update = executor.submit(
                graph.update, cpu_update_input=[{"actual_seq_lengths_kv": [length]}]
            )
            graph.replay()
            update.result()
        torch.npu.synchronize()
        expected = reference(length, 1, 63)[0]
        torch.testing.assert_close(
            captured.cpu().float(),
            torch.full_like(captured.cpu().float(), expected),
            rtol=0,
            atol=0,
        )
        visible = (
            (~backend.forward_metadata.swa_mask[0, 0])
            .cpu()
            .nonzero()
            .flatten()
            .tolist()
        )
        assert visible == list(range(max(0, length - 1 - 63), length))


@pytest.mark.parametrize(
    "prefix,qlen,window",
    [(127, 1, 255), (128, 1, 255), (129, 1, 255), (127, 2, 63), (128, 2, 63)],
)
@pytest.mark.parametrize("identity", [False, True])
@pytest.mark.parametrize("cap", [0.0, 2.0])
def test_extend_mapped_pages_and_window(prefix, qlen, window, identity, cap):
    backend, layer = fixture(window, identity)
    layer.logit_cap = cap
    fb = batch(prefix + qlen, qlen)
    backend.init_forward_metadata(fb)
    q = torch.zeros(qlen, 4, 64, dtype=torch.bfloat16, device="npu")
    output = backend.forward_extend(q, None, None, layer, fb, save_kv_cache=False)
    expected = torch.tensor(reference(prefix + qlen, qlen, window))[:, None].expand(
        qlen, 256
    )
    torch.testing.assert_close(output.cpu().float(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("length", [64, 128, 129])
def test_eager_decode_mapped_window(length):
    backend, layer = fixture(63)
    fb = batch(length, mode=ForwardMode.DECODE)
    backend.init_forward_metadata(fb)
    q = torch.zeros(1, 4, 64, dtype=torch.bfloat16, device="npu")
    output = backend.forward_decode(q, None, None, layer, fb, save_kv_cache=False)
    expected = reference(length, 1, 63)[0]
    torch.testing.assert_close(
        output.cpu().float(),
        torch.full_like(output.cpu().float(), expected),
        rtol=0,
        atol=0,
    )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
