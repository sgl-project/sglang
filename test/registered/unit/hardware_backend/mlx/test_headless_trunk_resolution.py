"""Regression: headless-trunk resolution must unwrap VL-family model wrappers.

``MlxModelRunner._load_model`` resolves the logit-head-free trunk so that a
non-final chunked-prefill chunk can run the layers (updating KV / auxiliary
state) without materialising a ``[chunk, vocab]`` logits tensor whose next
token is discarded anyway.  The lookup used to be a single ``getattr(model,
"model", None)``: mlx-lm text models match, but VL-family wrappers (``qwen3_5``
and friends) hold the text stack one level deeper, as
``Model.language_model.model``.  On such a checkpoint the probe returned None
and *every* chunk -- including the discarded-output ones -- computed full
vocab logits: 4096 tokens x 248,320 vocab x 2 B ~= 1.9 GiB per chunk on
Qwen3.8-27B, the largest transient in the process, which is what pushed a
32 GiB machine into a Metal command-buffer OOM during prefill.

The startup log line "Model Model exposes no headless trunk (`.model`)"
(observed on Qwen3.8-27B, 2026-09-13) is the symptom this test prevents from
coming back.

MLX-gated because importing ``model_runner`` pulls in ``mlx.core``.
"""

from __future__ import annotations

import importlib.util
import inspect
import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_mlx_ci
from sglang.test.test_utils import CustomTestCase

register_mlx_ci(est_time=1, suite="stage-a-unit-test-mlx")

_HAS_MLX = importlib.util.find_spec("mlx") is not None
_SKIP_REASON = "requires mlx"

if _HAS_MLX:
    from sglang.srt.hardware_backend.mlx.model_runner import (
        MlxModelRunner,
        resolve_headless_trunk,
    )


def _callable_trunk(name: str):
    def trunk(input_ids, cache=None):
        return (name, input_ids, cache)

    return trunk


@unittest.skipUnless(_HAS_MLX, _SKIP_REASON)
class TestHeadlessTrunkResolution(CustomTestCase):
    def test_text_model_exposes_trunk_directly(self):
        """``Model.model`` (plain mlx-lm text models, e.g. qwen2/qwen3)."""
        trunk = _callable_trunk("text")
        model = SimpleNamespace(model=trunk, language_model=None)
        self.assertIs(resolve_headless_trunk(model), trunk)

    def test_vl_wrapper_nests_trunk_under_language_model(self):
        """``Model.language_model.model`` (qwen3_5 / VL-family wrappers)."""
        trunk = _callable_trunk("text")
        model = SimpleNamespace(language_model=SimpleNamespace(model=trunk))
        self.assertIs(resolve_headless_trunk(model), trunk)

    def test_vl_wrapper_without_nested_trunk_falls_back_to_outer(self):
        """A wrapper whose ``language_model`` is itself the trunk (or empty)."""
        trunk = _callable_trunk("text")
        model = SimpleNamespace(language_model=SimpleNamespace(), model=trunk)
        self.assertIs(resolve_headless_trunk(model), trunk)

    def test_no_trunk_returns_none(self):
        self.assertIsNone(resolve_headless_trunk(SimpleNamespace()))
        self.assertIsNone(
            resolve_headless_trunk(SimpleNamespace(language_model=SimpleNamespace()))
        )

    def test_non_callable_model_attribute_is_not_a_trunk(self):
        """``.model`` may be a plain object (config, submodule holder) -> no trunk."""
        model = SimpleNamespace(model=object())
        self.assertIsNone(resolve_headless_trunk(model))

    def test_load_model_still_uses_the_shared_resolver(self):
        """Guard the wiring: ``_load_model`` must not re-inline the old lookup."""
        src = inspect.getsource(MlxModelRunner._load_model)
        self.assertIn("resolve_headless_trunk(self.model)", src)
        self.assertNotIn('getattr(self.model, "model", None)', src)


if __name__ == "__main__":
    unittest.main()
