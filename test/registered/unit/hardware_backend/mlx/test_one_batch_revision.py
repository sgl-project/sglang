"""The standalone MLX benchmark must load the requested checkpoint revision."""

import importlib.util
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_mlx_ci
from sglang.test.test_utils import CustomTestCase

register_mlx_ci(est_time=2, suite="stage-a-unit-test-mlx")

_HAS_MLX = importlib.util.find_spec("mlx") is not None


@unittest.skipUnless(_HAS_MLX, "requires mlx")
class TestMlxBenchRevision(CustomTestCase):
    def test_checkpoint_revision_reaches_native_loader(self):
        from sglang.benchmark.one_batch import _MlxBenchRunner

        for revision in ("pinned-checkpoint-commit", None):
            with self.subTest(revision=revision):
                cfg = SimpleNamespace(
                    model_path="example/model",
                    trust_remote_code=False,
                    mem_fraction_static=0.45,
                    quantization="mlx_q4",
                    max_total_tokens=1024,
                    revision=revision,
                )
                with (
                    patch(
                        "sglang.benchmark.one_batch.resolving_view", return_value=cfg
                    ),
                    patch(
                        "sglang.srt.hardware_backend.mlx.model_runner.MlxModelRunner"
                    ) as loader,
                ):
                    benchmark = _MlxBenchRunner(object(), object())
                self.assertEqual(loader.call_args.kwargs["revision"], revision)
                self.assertEqual(loader.call_args.kwargs["model_path"], cfg.model_path)
                loader.return_value.init_cache_pools.assert_called_once_with(
                    req_to_token_pool=None
                )
                self.assertIs(benchmark.mlx_runner, loader.return_value)


if __name__ == "__main__":
    unittest.main()
