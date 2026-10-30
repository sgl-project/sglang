import itertools
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative import dflash_info, spec_utils
from sglang.srt.speculative.dspark_components.dspark_verify import TargetVerifyExecutor
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


class TestDSparkVerifyTracking(unittest.TestCase):
    def test_verify_refreshes_tracking_before_forward_batch(self):
        for compact, graph, lazy in itertools.product((False, True), repeat=3):
            with self.subTest(compact=compact, graph=graph, lazy=lazy):
                self._check_verify(compact, graph, lazy)

    def _check_verify(self, compact, graph, lazy):
        mapping = torch.arange(8, device="cuda").reshape(4, 2)
        batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            seq_lens_cpu=torch.tensor([7, 15]),
            seq_lens_sum=22,
            req_pool_indices=torch.tensor([3, 1], device="cuda"),
            reqs=[
                SimpleNamespace(kv=SimpleNamespace(mamba_next_track_idx=1)),
                SimpleNamespace(kv=SimpleNamespace(mamba_next_track_idx=None)),
            ],
            req_to_token_pool=SimpleNamespace(
                req_index_to_mamba_ping_pong_track_buffer_mapping=mapping
            ),
            mamba_lazy_spec_track_positions_cpu=[0, 1],
        )
        mode = SimpleNamespace(
            mamba=SimpleNamespace(
                enable_mamba_extra_buffer=True,
                enable_mamba_extra_buffer_lazy=lazy,
            )
        )
        runner = SimpleNamespace(
            decode_cuda_graph_runner=Mock() if graph else None,
            attn_backend=Mock(),
            model_config=SimpleNamespace(
                hf_text_config=SimpleNamespace(model_type="test")
            ),
        )
        worker = SimpleNamespace(model_runner=runner, forward_batch_generation=Mock())
        executor = TargetVerifyExecutor(
            target_worker=worker,
            gamma=1,
            verify_num_draft_tokens=2,
            model_runner=runner,
            kv_injector=None,
            tp_sync=None,
        )
        executor._verify_backend_self_adds_seq_lens_cache = False
        ids = torch.arange(4)
        window = SimpleNamespace(
            positions_2d=ids.reshape(2, 2),
            verify_cache_loc=ids,
            iteration_ids=None,
        )
        stale_fields = (
            "mamba_track_mask",
            "mamba_track_seqlens",
            "mamba_prefill_track_mask_cpu",
            "mamba_track_seqlens_cpu",
        )

        def init_new(prepared, *_args, **_kwargs):
            expected = torch.tensor([mapping[3, 0 if lazy else 1], -1], device="cuda")
            torch.testing.assert_close(prepared.mamba_track_indices, expected)
            for name in stale_fields:
                self.assertIsNone(getattr(prepared, name))
            return SimpleNamespace(mamba_track_indices=prepared.mamba_track_indices)

        with (
            patch.object(spec_utils, "get_exec", return_value=mode),
            patch.object(spec_utils, "_VERIFY_TRACK_INDICES", None),
            patch.object(dflash_info, "_is_npu", False),
            patch.object(dflash_info.ForwardBatch, "init_new", side_effect=init_new),
        ):
            # A later step can reuse request rows with a different slot mapping.
            for offset in (0, 100):
                mapping.add_(offset)
                batch.mamba_track_indices = torch.full((2,), -99, device="cuda")
                for name in stale_fields:
                    setattr(batch, name, object())
                if compact:
                    executor._run_ragged(
                        batch=batch,
                        layout=SimpleNamespace(verify_lens_cpu=[2, 2]),
                        ragged_window=SimpleNamespace(
                            verify_ids=ids,
                            positions=ids,
                            verify_cache_loc=ids,
                            window_index=None,
                        ),
                        sampling_info=None,
                        iteration_ids=None,
                    )
                else:
                    executor.run_non_compact(
                        batch=batch,
                        draft_input=None,
                        verify_ids_2d=ids.reshape(2, 2),
                        verify_window=window,
                        sampling_info=None,
                    )
                forward_batch = worker.forward_batch_generation.call_args.kwargs[
                    "forward_batch"
                ]
                if graph:
                    runner.decode_cuda_graph_runner.load_batch.assert_called_with(
                        forward_batch
                    )
                else:
                    runner.attn_backend.init_forward_metadata.assert_called_with(
                        forward_batch
                    )
                self.assertEqual(batch.seq_lens_cpu.tolist(), [7, 15])


if __name__ == "__main__":
    unittest.main()
