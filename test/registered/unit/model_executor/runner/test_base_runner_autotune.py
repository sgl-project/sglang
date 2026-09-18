import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.runner import base_runner, flashinfer_autotune
from sglang.srt.runtime_context import get_context
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestExtendAutotune(CustomTestCase):
    def test_target_warmup_preserves_extend_and_verify_batches(self):
        for algorithm in (
            SpeculativeAlgorithm.NONE,
            SpeculativeAlgorithm.EAGLE3,
            SpeculativeAlgorithm.DFLASH,
        ):
            with (
                self.subTest(algorithm=algorithm),
                get_context().override_server_args(
                    speculative_num_steps=2,
                    speculative_eagle_topk=1,
                    speculative_num_draft_tokens=3,
                ),
            ):
                mr = SimpleNamespace(
                    device="cpu",
                    dtype=torch.float32,
                    is_generation=True,
                    is_draft_worker=False,
                    spec_algorithm=algorithm,
                    decode_num_tokens_per_req=lambda: 3,
                    model_config=SimpleNamespace(
                        hidden_size=4, is_multimodal=algorithm.is_dflash()
                    ),
                    model=SimpleNamespace(forward=MagicMock()),
                    attn_backend=SimpleNamespace(
                        extend_dummy_seqs_capped_by_req_pool=True,
                        get_cuda_graph_seq_len_fill_value=lambda: 1,
                        init_forward_metadata=MagicMock(),
                    ),
                    req_to_token_pool=SimpleNamespace(size=2),
                    lora_manager=None,
                    canary_manager=None,
                    tp_group=SimpleNamespace(barrier=lambda: None),
                    prepare_dummy_forward_batch=lambda batch: batch,
                    server_args=SimpleNamespace(moe_runner_backend="auto"),
                )
                buffers = base_runner._allocate_decode_buffers(
                    device=torch.device("cpu"),
                    max_bs=2,
                    max_num_token=8,
                    hidden_size=4,
                    vocab_size=8,
                    dtype=torch.float32,
                    dp_size=1,
                    pp_size=1,
                    is_encoder_decoder=False,
                    require_mlp_tp_gather=False,
                    seq_len_fill_value=1,
                    encoder_len_fill_value=0,
                    num_tokens_per_req=4,
                    cache_loc_dtype=torch.int64,
                    enable_mamba_track=False,
                    allocate_logits_buffer=False,
                )
                runner = SimpleNamespace(model_runner=mr)
                runner._dummy_run = base_runner.BaseRunner._dummy_run.__get__(runner)
                runner._alloc_dummy_decode_buffers = MagicMock(return_value=buffers)
                with (
                    patch.object(
                        base_runner, "require_mlp_tp_gather", return_value=False
                    ),
                    patch.object(
                        base_runner, "require_attn_tp_gather", return_value=False
                    ),
                    patch.object(
                        base_runner, "require_gathered_buffer", return_value=False
                    ),
                    patch.object(
                        flashinfer_autotune.envs.SGLANG_FLASHINFER_AUTOTUNE_EXTEND,
                        "get",
                        return_value=True,
                    ),
                    patch.object(
                        flashinfer_autotune, "max_prefill_buffer_tokens", return_value=8
                    ),
                    patch.object(
                        flashinfer_autotune,
                        "run_flashinfer_autotune_forward",
                        side_effect=lambda mr, fn, **kw: fn(),
                    ),
                    patch.object(torch.cuda, "empty_cache"),
                ):
                    flashinfer_autotune.maybe_flashinfer_autotune_extend(
                        runner, decode_num_tokens=3
                    )
                    mr.model.forward.assert_called_once()
                    batch = mr.model.forward.call_args.args[2]
                    self.assertEqual(batch.forward_mode, ForwardMode.EXTEND)
                    self.assertIsNone(batch.spec_info)
                    self.assertEqual(batch.input_ids.numel(), 8)
                    self.assertEqual(batch.extend_seq_lens_cpu, [4, 4])
                    self.assertEqual(batch.extend_prefix_lens_cpu, [0, 0])
                    self.assertEqual(batch.extend_start_loc.tolist(), [0, 4])
                    runner._dummy_run(batch_size=2, buffers=buffers)
                    batch = mr.model.forward.call_args.args[2]
                    if algorithm.is_speculative():
                        self.assertEqual(batch.forward_mode, ForwardMode.TARGET_VERIFY)
                        self.assertIsNotNone(batch.spec_info)
                        self.assertEqual(batch.input_ids.numel(), 6)
                    else:
                        self.assertEqual(batch.forward_mode, ForwardMode.DECODE)
                        self.assertIsNone(batch.spec_info)
                        self.assertEqual(batch.input_ids.numel(), 2)


if __name__ == "__main__":
    unittest.main()
