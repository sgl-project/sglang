"""Breakable prefill CUDA graphs under decode context parallelism.

Replay re-runs the break functions with their capture-time arguments, so the
DCP extend gather must take the live batch from the forward context, and the
runner must hand every replayed batch the bucket-sized gather buffer it
captured. Prefix hits need the cross-rank all-gather and stay eager.

    python -m pytest test/registered/unit/model_executor/test_prefill_cuda_graph_dcp.py -v
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.dcp.metadata import DecodeContextParallelMetadata
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
    PrefillCudaGraphRunner,
)
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    set_tc_piecewise_forward_context,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

KV_LORA_RANK = 4
ROPE_DIM = 2


def _metadata(num_tokens):
    return DecodeContextParallelMetadata(
        dcp_kv_indptr=torch.zeros(2, dtype=torch.int32),
        dcp_kv_buffer=torch.zeros((num_tokens, 1, KV_LORA_RANK + ROPE_DIM)),
        dcp_kv_indices=torch.arange(num_tokens, dtype=torch.int32),
        dcp_local_prefix_kv_indices=torch.zeros(0, dtype=torch.int32),
        dcp_extend_prefix_lens_sum=0,
    )


def _batch(num_tokens, prefix_lens):
    return SimpleNamespace(
        batch_size=len(prefix_lens),
        input_ids=torch.zeros(num_tokens, dtype=torch.int64),
        input_embeds=None,
        replace_embeds=None,
        forward_mode=SimpleNamespace(is_target_verify=lambda: False),
        capture_hidden_mode=CaptureHiddenMode.NULL,
        global_num_tokens_cpu=None,
        return_logprob=False,
        extend_prefix_lens_cpu=prefix_lens,
        extend_prefix_lens=torch.tensor(prefix_lens),
        seq_lens=torch.tensor([num_tokens]),
        extend_seq_lens=torch.tensor([num_tokens]),
        req_pool_indices=torch.zeros(1, dtype=torch.int64),
        seq_lens_sum=num_tokens,
        attn_dcp_metadata=None,
    )


def _runner(dcp_active):
    runner = PrefillCudaGraphRunner.__new__(PrefillCudaGraphRunner)
    runner._capture_req_slots = 4
    runner.enable_lora = False
    runner.capture_hidden_mode = CaptureHiddenMode.NULL
    runner.max_num_tokens = 32
    runner.capture_num_tokens = [8]
    runner.backend = SimpleNamespace()
    runner.prefill_backend_name = Backend.BREAKABLE
    runner.has_mha_companion_layers = False
    runner._is_full_backend = False
    runner._capture_chunked_prefix = False
    runner.use_captured_attn_metadata = False
    runner._dcp_extend_active = dcp_active
    runner._dcp_kv_buffers = {}
    return runner


class TestDcpAdmission(unittest.TestCase):
    def test_prefix_hits_stay_eager_under_dcp(self):
        fresh = _batch(8, [0])
        prefix_hit = _batch(8, [4])
        self.assertTrue(_runner(dcp_active=False).can_run_graph(prefix_hit))
        runner = _runner(dcp_active=True)
        self.assertTrue(runner.can_run_graph(fresh))
        self.assertFalse(runner.can_run_graph(prefix_hit))


class TestDcpReplayMetadata(unittest.TestCase):
    def test_replay_reuses_the_captured_gather_buffer(self):
        runner = _runner(dcp_active=True)
        captured = torch.full((8, 1, KV_LORA_RANK + ROPE_DIM), 7.0)
        runner._dcp_kv_buffers[8] = captured
        planned = []

        def plan(*args):
            planned.append(args)
            return _metadata(num_tokens=5)

        attn_backend = SimpleNamespace(
            req_to_token_pool=SimpleNamespace(req_to_token=torch.zeros((1, 8))),
            token_to_kv_pool=SimpleNamespace(
                get_kv_buffer_shape=lambda: (torch.Size([64, 1, 6]), None)
            ),
            init_forward_metadata=lambda batch: None,
            prepare_prefill_shared_read_snapshot=lambda batch, num_qo_tokens: None,
        )
        runner.model_runner = SimpleNamespace(
            attn_backend=attn_backend,
            model=SimpleNamespace(prepare_context_parallel_metadata_for_dcp=plan),
            kv_cache_dtype=torch.float32,
            device="cpu",
        )
        live = _batch(5, [0])
        static = _batch(8, [0])
        runner._prepare_forward_metadata_for_replay(live, static, num_tokens=8)

        self.assertEqual(len(planned), 1)
        self.assertIs(static.attn_dcp_metadata, live.attn_dcp_metadata)
        # Live indices, captured (bucket-sized) buffer.
        self.assertIs(static.attn_dcp_metadata.dcp_kv_buffer, captured)
        self.assertEqual(static.attn_dcp_metadata.dcp_kv_indices.numel(), 5)


class TestDcpExtendGatherBreak(unittest.TestCase):
    def test_gather_writes_the_forward_context_batch(self):
        from sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla import (
            bcg_dcp_extend_kv_gather,
        )

        capture_batch = _batch(8, [0])
        capture_batch.attn_dcp_metadata = _metadata(8)
        live_batch = _batch(8, [0])
        live_batch.attn_dcp_metadata = _metadata(8)
        k_nope = torch.ones((8, 1, KV_LORA_RANK))
        k_pe = torch.full((8, 1, ROPE_DIM), 2.0)
        with set_tc_piecewise_forward_context(
            live_batch,
            attention_layers=[],
            quant_config=None,
            moe_layers=[],
            moe_fusions=[],
        ):
            # Replay re-supplies the capture-time batch as the argument.
            bcg_dcp_extend_kv_gather(
                None, None, capture_batch, KV_LORA_RANK, k_nope, k_pe
            )
        self.assertTrue(
            torch.all(
                live_batch.attn_dcp_metadata.dcp_kv_buffer[..., :KV_LORA_RANK] == 1.0
            )
        )
        self.assertTrue(
            torch.all(
                live_batch.attn_dcp_metadata.dcp_kv_buffer[..., KV_LORA_RANK:] == 2.0
            )
        )
        self.assertTrue(torch.all(capture_batch.attn_dcp_metadata.dcp_kv_buffer == 0.0))


if __name__ == "__main__":
    unittest.main()
