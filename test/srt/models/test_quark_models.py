import dataclasses
import importlib
import importlib.metadata
import unittest
from types import SimpleNamespace
from typing import Optional

import huggingface_hub
import pytest
import torch
from packaging import version

from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval
from sglang.test.runners import DEFAULT_PROMPTS, SRTRunner
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


@dataclasses.dataclass
class ModelCase:
    model_path: str
    tp_size: int = 1
    context_length: Optional[int] = None
    mem_fraction_static: Optional[float] = None


ALL_MODELS_TP1 = [
    ModelCase("fxmarty/qwen_1.5-moe-a2.7b-mxfp4"),
    ModelCase(
        "fxmarty/Llama-4-Scout-17B-16E-Instruct-2-layers-mxfp4",
        tp_size=1,
        mem_fraction_static=0.7,
        context_length=100000,
    ),
]

ALL_MODELS_TP2 = [ModelCase("fxmarty/qwen_1.5-moe-a2.7b-mxfp4", tp_size=2)]

ALL_MODELS_TP8 = [
    ModelCase("fxmarty/deepseek_r1_3_layers_mxfp4", tp_size=8),
    ModelCase(
        "fxmarty/Llama-4-Scout-17B-16E-Instruct-2-layers-mxfp4",
        tp_size=8,
        mem_fraction_static=0.7,
        context_length=1000000,
    ),
]

QUARK_MXFP4_AVAILABLE = importlib.util.find_spec("quark") is not None and version.parse(
    importlib.metadata.version("amd-quark")
) >= version.parse("0.8.99")

try:
    huggingface_hub.list_repo_refs(
        "amd/Llama-3.3-70B-Instruct-WMXFP4-AMXFP4-KVFP8-Scale-UINT8-SQ"
    )
    HF_HUB_AMD_ORG_ACCESS = True
except huggingface_hub.errors.RepositoryNotFoundError:
    HF_HUB_AMD_ORG_ACCESS = False


@unittest.skipIf(not QUARK_MXFP4_AVAILABLE, reason="amd-quark>=0.9 is not available")
class TestQuarkMXFP4Loading(CustomTestCase):
    def _test_load_and_run(self, model_case: ModelCase):
        if torch.cuda.device_count() < model_case.tp_size:
            unittest.skip(
                f"This test requires >={model_case.tp_size} gpus, got only {torch.cuda.device_count()}"
            )

        prompts = [p for p in DEFAULT_PROMPTS if len(p) < 1000]
        max_new_tokens = 20

        with SRTRunner(
            model_case.model_path,
            tp_size=model_case.tp_size,
            model_type="generation",
            torch_dtype="auto",
            mem_fraction_static=model_case.mem_fraction_static,
            context_length=model_case.context_length,
        ) as srt_runner:
            srt_outputs = srt_runner.forward(prompts, max_new_tokens=max_new_tokens)

    # `parameterized` is not a dependency.
    def test_load_and_run_tp1(self):
        for model_case in ALL_MODELS_TP1:
            self._test_load_and_run(model_case)

    def test_load_and_run_tp2(self):
        for model_case in ALL_MODELS_TP2:
            self._test_load_and_run(model_case)

    def test_load_and_run_tp8(self):
        for model_case in ALL_MODELS_TP8:
            self._test_load_and_run(model_case)


@unittest.skipIf(not QUARK_MXFP4_AVAILABLE, reason="amd-quark>=0.9 is not available")
@unittest.skipIf(
    not HF_HUB_AMD_ORG_ACCESS,
    reason="Read access to huggingface.co/amd is required for this test.",
)
class TestR1MXFP4Accuracy(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        # Private model.
        cls.model = "amd/DeepSeek-R1-WMXFP4-AMXFP4-Scale-UINT8-MoE-Quant"

        cls.base_url = DEFAULT_URL_FOR_TEST

        other_args = [
            "--tp",
            "8",
            "--mem-fraction-static",
            "0.9",
            "--context-length",
            "38768",
            # TODO: Use again aiter attention backend when it is debugged.
            # Use to work on 20th May, but getting bad accuracy with
            # aiter attention backend from ~9th June onwards (with aiter==0.1.4).
            # Some changes to aiter_backend.py attention are probably responsible.
            "--attention-backend",
            "triton",
        ]

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=45 * 60,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        args = SimpleNamespace(
            num_shots=8,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval(args)
        print(f"{metrics=}")
        self.assertGreater(metrics["accuracy"], 0.96)


if __name__ == "__main__":
    unittest.main()
