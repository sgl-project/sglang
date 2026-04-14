import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist

from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend
from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.utils.cp_utils import prepare_context_parallel_metadata
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.utils import is_flashinfer_available
from sglang.test.test_utils import CustomTestCase, run_distributed_test


class MockModelRunner:
    def __init__(
        self,
        rank: int,
        world_size: int,
        page_size: int = 1,
        num_heads: int = 2,
        head_dim: int = 8,
    ):
        self.device = f"cuda:{rank}"
        self.dtype = torch.float16
        self.kv_cache_dtype = torch.float16
        self.is_hybrid_swa = False
        self.attention_chunk_size = None
        self.attn_cp_size = world_size
        self.attn_cp_rank = rank
        self.page_size = page_size
        self.sliding_window_size = None
        self.token_to_kv_pool_allocator = None

        max_batch_size = 8
        max_context_len = 2048

        hf_config = type(
            "HFConfig",
            (),
            {
                "architectures": ["LlamaForCausalLM"],
            },
        )()

        self.model_config = type(
            "ModelConfig",
            (),
            {
                "context_len": max_context_len,
                "is_multimodal": False,
                "attention_arch": AttentionArch.MHA,
                "is_encoder_decoder": False,
                "is_local_attention_model": False,
                "num_attention_heads": num_heads,
                "head_dim": head_dim,
                "hf_config": hf_config,
                "get_num_kv_heads": lambda _self, _tp_size: num_heads,
            },
        )()

        self.server_args = type(
            "ServerArgs",
            (),
            {
                "kv_cache_dtype": "auto",
                "speculative_eagle_topk": None,
                "speculative_num_draft_tokens": 0,
                "enable_deterministic_inference": False,
                "multi_item_scoring_delimiter": None,
                "disable_piecewise_cuda_graph": True,
                "dllm_algorithm": None,
                "dllm_algorithm_config": None,
                "max_running_requests": None,
                "model_path": None,
                "revision": None,
            },
        )()

        self.req_to_token_pool = type(
            "TokenPool",
            (),
            {
                "size": max_batch_size,
                "req_to_token": torch.zeros(
                    max_batch_size,
                    max_context_len,
                    dtype=torch.int32,
                    device=self.device,
                ),
            },
        )()

        self.token_to_kv_pool = MHATokenToKVPool(
            size=max_batch_size * max_context_len,
            page_size=page_size,
            dtype=self.dtype,
            head_num=num_heads,
            head_dim=head_dim,
            layer_num=1,
            device=self.device,
            enable_memory_saver=False,
        )


class _FakeCPGroup:
    def cp_all_gather_into_tensor_async(self, output, input_tensor, _stream):
        if hasattr(dist, "all_gather_into_tensor"):
            dist.all_gather_into_tensor(output, input_tensor.contiguous())
            return

        gathered = [torch.empty_like(input_tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, input_tensor.contiguous())
        output.copy_(torch.cat(gathered, dim=0))

    def all_gather(self, input_tensor, dim=0):
        gathered = [torch.empty_like(input_tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, input_tensor.contiguous())
        return torch.cat(gathered, dim=dim)


def _split_for_cp(tensor: torch.Tensor, cp_meta) -> torch.Tensor:
    pieces = list(torch.split(tensor, cp_meta.split_list, dim=0))
    return torch.cat([pieces[i] for i in cp_meta.zigzag_index], dim=0).contiguous()


def _write_req_to_token(model_runner: MockModelRunner, seq_len: int):
    req_to_token = (
        torch.arange(0, 1, dtype=torch.int32, device=model_runner.device)[:, None]
        * seq_len
        + torch.arange(0, seq_len, dtype=torch.int32, device=model_runner.device)[None, :]
        + model_runner.page_size
    )
    model_runner.req_to_token_pool.req_to_token[:1, :seq_len] = req_to_token


def _build_forward_batch(
    backend,
    input_ids: torch.Tensor,
    out_cache_loc: torch.Tensor,
    seq_len: int,
    extend_len: int,
    prefix_len: int,
    token_to_kv_pool: MHATokenToKVPool,
    req_to_token_pool,
    cp_meta=None,
):
    forward_batch = ForwardBatch(
        batch_size=1,
        input_ids=input_ids,
        out_cache_loc=out_cache_loc,
        seq_lens_sum=seq_len,
        forward_mode=ForwardMode.EXTEND,
        req_pool_indices=torch.tensor([0], device=input_ids.device),
        seq_lens=torch.tensor([seq_len], device=input_ids.device),
        seq_lens_cpu=torch.tensor([seq_len], device="cpu"),
        extend_prefix_lens=torch.tensor([prefix_len], device=input_ids.device),
        extend_prefix_lens_cpu=torch.tensor([prefix_len], device="cpu"),
        extend_seq_lens=torch.tensor([extend_len], device=input_ids.device),
        extend_seq_lens_cpu=torch.tensor([extend_len], device="cpu"),
        attn_backend=backend,
    )
    forward_batch.req_to_token_pool = req_to_token_pool
    forward_batch.token_to_kv_pool = token_to_kv_pool
    if cp_meta is not None:
        forward_batch.attn_cp_metadata = cp_meta
    return forward_batch


def _setup_prefix_kv_cache(
    token_to_kv_pool: MHATokenToKVPool,
    layer: RadixAttention,
    prefix_k: torch.Tensor,
    prefix_v: torch.Tensor,
):
    if prefix_k.numel() == 0:
        return

    # Keep the same "slot id" convention as req_to_token_pool in this test:
    # token position t is stored at slot (t + page_size).
    cache_loc = (
        torch.arange(prefix_k.shape[0], device=prefix_k.device, dtype=torch.int32)
        + token_to_kv_pool.page_size
    )
    token_to_kv_pool.set_kv_buffer(
        layer,
        cache_loc,
        prefix_k,
        prefix_v,
        layer.k_scale,
        layer.v_scale,
    )


def _run_flashinfer_cp_worker(
    rank: int,
    *,
    q_len: int,
    prefix_len: int,
    num_heads: int,
    head_dim: int,
    page_size: int = 1,
):
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"
    world_size = dist.get_world_size()
    total_len = prefix_len + q_len

    model_runner = MockModelRunner(
        rank=rank,
        world_size=world_size,
        page_size=page_size,
        num_heads=num_heads,
        head_dim=head_dim,
    )
    _write_req_to_token(model_runner, total_len)

    with patch(
        "sglang.srt.layers.attention.flashinfer_backend.get_attention_tp_size",
        return_value=1,
    ):
        backend = FlashInferAttnBackend(model_runner)
        ref_backend = TorchNativeAttnBackend(model_runner)
        layer = RadixAttention(
            num_heads=num_heads,
            head_dim=head_dim,
            scaling=1.0,
            num_kv_heads=num_heads,
            layer_id=0,
        )

        torch.manual_seed(1234)
        prefix_k = torch.randn(
            (prefix_len, num_heads, head_dim), dtype=torch.float16, device=device
        )
        prefix_v = torch.randn(
            (prefix_len, num_heads, head_dim), dtype=torch.float16, device=device
        )
        full_q = torch.randn(
            (q_len, num_heads, head_dim), dtype=torch.float16, device=device
        )
        full_k = torch.randn(
            (q_len, num_heads, head_dim), dtype=torch.float16, device=device
        )
        full_v = torch.randn(
            (q_len, num_heads, head_dim), dtype=torch.float16, device=device
        )

        cp_meta = prepare_context_parallel_metadata(q_len, rank, world_size, [total_len])
        local_q = _split_for_cp(full_q, cp_meta)
        local_k = _split_for_cp(full_k, cp_meta)
        local_v = _split_for_cp(full_v, cp_meta)
        # Keep out_cache_loc consistent with req_to_token: token position t maps to
        # slot (t + page_size).
        full_cache_loc = (
            torch.arange(prefix_len, total_len, dtype=torch.int32, device=device)
            + page_size
        )

        _setup_prefix_kv_cache(model_runner.token_to_kv_pool, layer, prefix_k, prefix_v)

        ref_forward_batch = _build_forward_batch(
            ref_backend,
            input_ids=torch.randint(0, 100, (1, q_len), device=device),
            out_cache_loc=full_cache_loc,
            seq_len=total_len,
            extend_len=q_len,
            prefix_len=prefix_len,
            token_to_kv_pool=model_runner.token_to_kv_pool,
            req_to_token_pool=model_runner.req_to_token_pool,
        )
        ref_output = ref_backend.forward_extend(
            full_q, full_k, full_v, layer, ref_forward_batch
        )
        expected_local = _split_for_cp(ref_output.view(q_len, -1), cp_meta)

        cp_forward_batch = _build_forward_batch(
            backend,
            # In production, CP splits q/k/v via hidden_states sharding, while
            # ForwardBatch metadata (including out_cache_loc) remains global.
            input_ids=torch.randint(0, 100, (1, q_len), device=device),
            out_cache_loc=full_cache_loc,
            seq_len=total_len,
            extend_len=q_len,
            prefix_len=prefix_len,
            token_to_kv_pool=model_runner.token_to_kv_pool,
            req_to_token_pool=model_runner.req_to_token_pool,
            cp_meta=cp_meta,
        )

        fake_cp_group = _FakeCPGroup()
        with patch(
            "sglang.srt.layers.utils.cp_utils.get_attention_cp_group",
            return_value=fake_cp_group,
        ):
            backend.init_forward_metadata(cp_forward_batch)
            output = backend.forward_extend(
                local_q, local_k, local_v, layer, cp_forward_batch
            )

    output = output.view(local_q.shape[0], -1)
    if not torch.allclose(output, expected_local, atol=1e-1, rtol=0.0):
        diff_mask = ~torch.isclose(output, expected_local, atol=1e-1, rtol=0.0)
        if diff_mask.any():
            mismatch = diff_mask.nonzero()[0]
            idx = tuple(mismatch.tolist())
            raise AssertionError(
                f"rank={rank} mismatch at {idx}: "
                f"got={output[idx].item()} expected={expected_local[idx].item()}"
            )


@unittest.skipIf(
    not torch.cuda.is_available() or not is_flashinfer_available(),
    "CUDA + flashinfer required",
)
class TestFlashInferBackend(CustomTestCase):
    def test_forward_extend_cp(self):
        if torch.cuda.device_count() < 2:
            self.skipTest("FlashInfer CP test requires at least 2 GPUs")

        run_distributed_test(
            _run_flashinfer_cp_worker,
            world_size=2,
            backend="nccl",
            q_len=128,
            prefix_len=0,
            num_heads=8,
            head_dim=64,
        )

    def test_forward_extend_cp_with_prefix(self):
        if torch.cuda.device_count() < 2:
            self.skipTest("FlashInfer CP test requires at least 2 GPUs")

        run_distributed_test(
            _run_flashinfer_cp_worker,
            world_size=2,
            backend="nccl",
            q_len=128,
            prefix_len=64,
            num_heads=8,
            head_dim=64,
        )

    def test_forward_extend_cp_with_prefix_page_size_16(self):
        if torch.cuda.device_count() < 2:
            self.skipTest("FlashInfer CP test requires at least 2 GPUs")

        run_distributed_test(
            _run_flashinfer_cp_worker,
            world_size=2,
            backend="nccl",
            q_len=128,
            prefix_len=64,
            num_heads=8,
            head_dim=64,
            page_size=16,
        )


if __name__ == "__main__":
    unittest.main()
