"""Small real CUDA metadata fixtures; no model runner or model weights required."""

from dataclasses import fields
from types import SimpleNamespace

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa.dsa_topk_backend import DSATopKBackend
from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_parallel

BS, NEXT_N, WIDTH, TOPK, POOL = 4, 6, 131072, 2048, 4
ROUNDS = (
    ([64, 128, 2048, 65530], [0, 2, 4, 6]),
    ([63, 129, 65539, 100001], [7, 5, 3, 1]),
    ([128, 61, 2051, 1023], [2, 6, 0, 4]),
)


def inputs(lengths, requests):
    return (
        torch.tensor(lengths, dtype=torch.int64, device="cuda"),
        torch.tensor(requests, dtype=torch.int64, device="cuda"),
    )


def make_backend(mode, seq, req, *, fusion=True):
    backend = object.__new__(DeepseekSparseAttnBackend)
    backend.device = torch.device("cuda")
    backend.device_sm_major = torch.cuda.get_device_capability()[0]
    backend.num_q_heads = 64
    backend.real_page_size = 64
    backend.dsa_index_topk = TOPK
    backend.dsa_index_kpool = POOL
    backend.speculative_num_draft_tokens = NEXT_N
    backend.dsa_drop_wide_page_table = False
    backend.dsa_decode_impl = "fa3"
    backend.dsa_prefill_impl = "fa3"
    backend.enable_auto_select_prefill_impl = False
    backend.token_to_kv_pool = SimpleNamespace(slots_per_page=64)
    # Only attention-dispatch state is synthetic; every metadata kernel is real.
    backend._is_in_breakable_cuda_graph = lambda: False
    backend._is_in_tc_piecewise_cuda_graph = lambda: False
    backend._get_device_sm = lambda: backend.device_sm_major * 10
    backend._is_blackwell = lambda: backend.device_sm_major == 10
    backend.dsa_topk_backend = DSATopKBackend.SGL_KERNEL
    backend.req_to_token = torch.arange(
        8 * WIDTH, device="cuda", dtype=torch.int32
    ).view(8, WIDTH)
    backend._arange_buf = torch.arange(
        BS * NEXT_N + 1, device="cuda", dtype=torch.int32
    )
    backend.decode_cuda_graph_metadata = {
        "page_table": torch.zeros(BS * NEXT_N, WIDTH, device="cuda", dtype=torch.int32),
        "cu_seqlens_q": backend._arange_buf,
    }
    with envs.SGLANG_EXPERIMENTAL_DSA_KPOOL_METADATA_FUSION.override(fusion):
        backend._init_kpool_metadata_fusion()
    with envs.SGLANG_OPT_USE_TOPK_V2.override(True):
        apply_metadata(backend, mode, seq, req)
    apply_metadata(backend, mode, seq, req)
    return backend


def apply_metadata(backend, mode, seq, req, spec_info=None):
    backend._apply_cuda_graph_metadata(
        bs=BS,
        req_pool_indices=req,
        seq_lens=seq,
        seq_lens_cpu=seq.cpu(),
        forward_mode=mode,
        spec_info=spec_info,
    )


def tensor_buffers(metadata):
    result = {}
    for field in fields(metadata):
        value = getattr(metadata, field.name)
        if isinstance(value, torch.Tensor):
            result[field.name] = value
    if metadata.kpool_write_plan is not None:
        for field in fields(metadata.kpool_write_plan):
            value = getattr(metadata.kpool_write_plan, field.name)
            if isinstance(value, torch.Tensor):
                result["kpool." + field.name] = value
    return result


def addresses(metadata):
    return {
        name: tensor.data_ptr() for name, tensor in tensor_buffers(metadata).items()
    }


def assert_metadata_equal(test, actual, expected):
    actual_buffers, expected_buffers = tensor_buffers(actual), tensor_buffers(expected)
    test.assertEqual(actual_buffers.keys(), expected_buffers.keys())
    for name, value in actual_buffers.items():
        reference = expected_buffers[name]
        if name == "topk_v2_plan":
            # Unused plan rows are intentionally uninitialized. Active rows are
            # compacted by atomicAdd, so compare them in request order.
            torch.testing.assert_close(value[0], reference[0])
            count = int(reference[0, 1].item())
            lhs, rhs = value[1 : count + 1], reference[1 : count + 1]
            torch.testing.assert_close(
                lhs[lhs[:, 0].argsort()], rhs[rhs[:, 0].argsort()]
            )
        elif name in ("page_table_1", "real_page_table", "pooled_real_page_table"):
            lengths = expected.cache_seqlens_int32
            lengths = lengths.repeat_interleave(value.shape[0] // lengths.numel())
            step = 1 if name == "page_table_1" else 64
            if name == "pooled_real_page_table":
                step *= POOL
            live = (
                torch.arange(value.shape[1], device=value.device)[None, :] * step
                < lengths[:, None]
            )
            torch.testing.assert_close(value[live], reference[live], msg=name)
        else:
            torch.testing.assert_close(value, reference, msg=name)


def capture_verify_metadata(backend, seq, req, *, dg_out_of_graph=False):
    backend.ingraph_verify_metadata_enabled = True
    backend.ingraph_verify_metadata_dg_out_of_graph = dg_out_of_graph
    batch = SimpleNamespace(
        batch_size=BS,
        forward_mode=ForwardMode.TARGET_VERIFY,
        seq_lens=seq,
        req_pool_indices=req,
    )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with get_parallel().override(dcp_enabled=False):
        with torch.cuda.stream(stream):
            # Compile kernels and prime the allocator before capture.
            backend.init_forward_metadata_in_graph(batch)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            backend.init_forward_metadata_in_graph(batch)
    torch.cuda.current_stream().wait_stream(stream)
    return graph
