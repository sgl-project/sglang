"""Unit tests for the sgl-eval-backed evals in eval_accuracy_kit.

Hermetic: fakes the ``sgl_eval`` Python API via ``sys.modules`` (no real
install, no server). Covers the GPQA/AIME25 sgl-eval drivers, the GSM8K/MMLU
``*_backend`` toggle (run_eval default vs sgl_eval opt-in), GenConfig
construction, threshold assertion, and the skip-when-absent path.
"""

import sys
import types as pytypes
import unittest
from unittest.mock import MagicMock, patch

import requests

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.kits import eval_accuracy_kit as kit
from sglang.test.kits.eval_accuracy_kit import (
    AIME25Mixin,
    GPQAMixin,
    GSM8KMixin,
    MMLUMixin,
)
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _fake_get(url, *args, **kwargs):
    # flush_cache must succeed (GSM8K calls it unguarded); /server_info is
    # probed by _check_accept_length, which swallows RequestException.
    if str(url).endswith("/flush_cache"):
        return MagicMock()
    raise requests.RequestException()


def _fake_sgl_eval(score, cap):
    """Build fake sgl_eval submodules; record what the driver passes in `cap`."""
    spec = MagicMock(name="EvalSpec")
    spec.run.return_value = pytypes.SimpleNamespace(aggregate={"score": score})

    reg = pytypes.ModuleType("sgl_eval.registry")
    reg.get = (
        lambda name: (
            cap.__setitem__("eval_name", name),
            cap.__setitem__("spec", spec),
        )[0]
        or spec
    )

    smp = pytypes.ModuleType("sgl_eval.sampler")
    smp.ChatCompletionSampler = lambda **kw: (
        cap.__setitem__("sampler_kw", kw),
        MagicMock(),
    )[1]

    typ = pytypes.ModuleType("sgl_eval.types")
    typ.GenConfig = lambda **kw: (
        cap.__setitem__("gen_kw", kw),
        pytypes.SimpleNamespace(**kw),
    )[1]

    return {
        "sgl_eval": pytypes.ModuleType("sgl_eval"),
        "sgl_eval.registry": reg,
        "sgl_eval.sampler": smp,
        "sgl_eval.types": typ,
    }


class TestSglEvalApiDriver(CustomTestCase):
    def _run_sgl(self, host_cls, method, score, **attrs):
        cap = {}
        host = host_cls(method)
        host.base_url = "http://127.0.0.1:0"
        host.model = "test-model"
        for k, v in attrs.items():
            setattr(host, k, v)
        with patch.dict(sys.modules, _fake_sgl_eval(score, cap)), patch.object(
            kit.requests, "get", side_effect=_fake_get
        ):
            getattr(host, method)()
        return cap

    # ---- GPQA / AIME25 (sgl-eval only) ----
    def test_gpqa_builds_genconfig_and_runs_spec(self):
        class Host(GPQAMixin, CustomTestCase):
            pass

        cap = self._run_sgl(
            Host,
            "test_gpqa",
            0.9,
            gpqa_score_threshold=0.5,
            gpqa_thinking=True,
            gpqa_reasoning_effort="max",
            gpqa_max_tokens=200000,
            gpqa_temperature=1.0,
            gpqa_top_p=1.0,
            gpqa_n_repeats=4,
        )
        self.assertEqual(cap["eval_name"], "gpqa")
        self.assertEqual(cap["gen_kw"]["chat_template_kwargs"], {"thinking": True})
        self.assertEqual(cap["gen_kw"]["reasoning_effort"], "max")
        self.assertEqual(cap["gen_kw"]["max_tokens"], 200000)
        self.assertEqual(cap["gen_kw"].get("temperature"), 1.0)
        _, run_kwargs = cap["spec"].run.call_args
        self.assertEqual(run_kwargs["n_repeats"], 4)
        self.assertIsNone(run_kwargs["predictions_writer"])
        self.assertIsNone(run_kwargs["load_examples"])

    def test_thinking_disabled_sends_no_chat_template_kwargs(self):
        class Host(GPQAMixin, CustomTestCase):
            pass

        cap = self._run_sgl(
            Host, "test_gpqa", 0.9, gpqa_score_threshold=0.5, gpqa_thinking=False
        )
        self.assertIsNone(cap["gen_kw"]["chat_template_kwargs"])

    def test_below_threshold_fails(self):
        class Host(AIME25Mixin, CustomTestCase):
            pass

        with self.assertRaises(AssertionError):
            self._run_sgl(Host, "test_aime25", 0.1, aime25_score_threshold=0.5)

    def test_skips_when_sgl_eval_absent(self):
        class Host(GPQAMixin, CustomTestCase):
            pass

        host = Host("test_gpqa")
        host.base_url = "http://127.0.0.1:0"
        host.model = "m"
        host.gpqa_score_threshold = 0.5
        # None in sys.modules makes `import sgl_eval.registry` raise ImportError.
        absent = {
            k: None for k in ("sgl_eval.registry", "sgl_eval.sampler", "sgl_eval.types")
        }
        with patch.dict(sys.modules, absent):
            with self.assertRaises(unittest.SkipTest):
                host.test_gpqa()

    # ---- GSM8K / MMLU backend toggle ----
    def test_gsm8k_sgl_eval_backend(self):
        class Host(GSM8KMixin, CustomTestCase):
            pass

        cap = self._run_sgl(
            Host, "test_gsm8k", 0.95, gsm8k_accuracy_thres=0.5, gsm8k_backend="sgl_eval"
        )
        self.assertEqual(cap["eval_name"], "gsm8k")
        # non-thinking benchmark -> thinking off -> no chat_template_kwargs
        self.assertIsNone(cap["gen_kw"]["chat_template_kwargs"])

    def test_gsm8k_canonical_threshold_alias(self):
        # Canonical gsm8k_score_threshold is honored (alias for legacy
        # gsm8k_accuracy_thres); below-threshold still fails.
        class Host(GSM8KMixin, CustomTestCase):
            pass

        cap = self._run_sgl(
            Host,
            "test_gsm8k",
            0.95,
            gsm8k_score_threshold=0.5,
            gsm8k_backend="sgl_eval",
        )
        self.assertEqual(cap["eval_name"], "gsm8k")
        with self.assertRaises(AssertionError):
            self._run_sgl(
                Host,
                "test_gsm8k",
                0.30,
                gsm8k_score_threshold=0.5,
                gsm8k_backend="sgl_eval",
            )

    def test_mmlu_sgl_eval_backend(self):
        class Host(MMLUMixin, CustomTestCase):
            pass

        cap = self._run_sgl(
            Host, "test_mmlu", 0.9, mmlu_score_threshold=0.5, mmlu_backend="sgl_eval"
        )
        self.assertEqual(cap["eval_name"], "mmlu")
        self.assertIsNone(cap["gen_kw"]["chat_template_kwargs"])

    def test_gsm8k_default_backend_uses_run_eval(self):
        # Default backend must NOT touch sgl-eval; it goes through run_eval
        # (OpenAI completion API) -- unchanged for the existing consumers.
        class Host(GSM8KMixin, CustomTestCase):
            pass

        captured = {}

        def fake_run_eval(args):
            captured["args"] = args
            return {"score": 0.95}

        host = Host("test_gsm8k")
        host.base_url = "http://127.0.0.1:0"
        host.model = "m"
        host.gsm8k_accuracy_thres = 0.5
        with patch.object(kit, "run_eval", side_effect=fake_run_eval), patch.object(
            kit.requests, "get", side_effect=_fake_get
        ):
            host.test_gsm8k()
        self.assertEqual(captured["args"].eval_name, "gsm8k")
        self.assertEqual(captured["args"].api, "completion")


if __name__ == "__main__":
    unittest.main()
