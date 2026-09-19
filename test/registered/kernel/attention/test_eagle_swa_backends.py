import unittest
from unittest.mock import patch

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_context
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.attention_unittest.attention_methods import (
    dense_attention as dense,
)
from sglang.test.kits.attention_unittest.runner_modes import (
    speculative_target_verify_runner as verify,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@unittest.skipIf(not torch.cuda.is_available(), "GPU required")
class TestEagleSWABackends(CustomTestCase):
    def _run_backend(self, backend, *, deterministic=False):
        original_init = dense.MockModelRunner.__init__
        original_input = verify._make_spec_verify_input

        def make_input(*args, **kwargs):
            out = original_input(*args, **kwargs)
            args[1].spec_algorithm = SpeculativeAlgorithm.EAGLE3
            return out

        for topk in (1, 2):

            def init(runner, *args, **kwargs):
                original_init(runner, *args, **kwargs)
                runner.model_config.dtype = runner.dtype
                runner.model_config.is_local_attention_model = False
                fields = dict(runner._server_args_override._fields)
                fields.update(
                    page_size=runner.page_size,
                    speculative_algorithm="EAGLE3",
                    speculative_eagle_topk=topk,
                    enable_deterministic_inference=deterministic,
                )
                override = get_context().override_server_args(**fields)
                override.install()
                self.addCleanup(override.restore)
                runner.spec_algorithm = SpeculativeAlgorithm.EAGLE3

            for prefixes, window in (
                ((9, 2), None),
                ((3, 4, 5), 4),
                ((9, 2), 4),
                ((1220, 325), 1023),
            ):
                case = dense.DenseAttentionCase(
                    name=f"{backend}_eagle_k{topk}_w{window}_{prefixes}",
                    backend=backend,
                    forward_mode=ForwardMode.TARGET_VERIFY,
                    num_heads=4,
                    num_kv_heads=2,
                    page_size=16,
                    prefix_lens=prefixes,
                    extend_lens=(3,) * len(prefixes),
                    sliding_window_size=window,
                )
                for run in (
                    verify.run_dense_spec_verify_case,
                    verify.run_dense_spec_verify_cuda_graph_case,
                ):
                    with (
                        self.subTest(case=case.name, run=run.__name__),
                        patch.object(dense.MockModelRunner, "__init__", init),
                        patch.object(verify, "_make_spec_verify_input", make_input),
                    ):
                        run(
                            self,
                            case,
                            topk=topk,
                            head_dim=64,
                            hidden_size=256,
                            max_context_len=2048,
                        )

    def test_triton(self):
        self._run_backend("triton")

    def test_triton_deterministic(self):
        self._run_backend("triton", deterministic=True)


if __name__ == "__main__":
    unittest.main()
