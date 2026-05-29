"""End-to-end regression for `--enable-return-routed-experts` under
Frozen-KV MTP (the failing case fixed by the per-`TopKConfig`
`capture_routed_experts` opt-out).

The contract: with `temperature=0` greedy, weights and prompts fixed, the
routed_experts decoded from the MTP run must equal those from a
target-only baseline cell-by-cell. The MTP run exercises overlap +
CUDA graph; a second variant additionally enables piecewise CUDA graph
(the BYPASS path that reconstructs `TopKConfig` from scalar args).

Environment requirements:

  - `SGLANG_RUN_MTP_R3_REGRESSION=1` enables the test class.
  - `SGLANG_MTP_R3_TARGET_MODEL`: HuggingFace name or local path of the
    DeepSeek-V3-class target model (e.g. a DSV3 checkpoint).
  - `SGLANG_MTP_R3_DRAFT_MODEL_PATH` (optional): path passed via
    `--speculative-draft-model-path` when the MTP draft weights are
    separate; defaults to the same target path (DeepSeek-V3 MTP uses
    the target model's NextN layer directly).
  - >= 4 H100/H200/B200-class GPUs in tensor-parallel layout.

When `SGLANG_RUN_MTP_R3_REGRESSION=1` but model env vars are missing,
the test bails with a clear message rather than silently skipping.
"""

import asyncio
import json
import logging
import os
import unittest
from typing import List, Sequence

import aiohttp
import numpy as np
import torch

from sglang.benchmark.utils import download_and_cache_hf_file
from sglang.srt.state_capturer.routed_experts import (
    extract_routed_experts_from_meta_info,
)
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


_RUN_MTP_R3 = os.environ.get("SGLANG_RUN_MTP_R3_REGRESSION") == "1"

SHAREGPT_REPO_ID = "anon8231489123/ShareGPT_Vicuna_unfiltered"
SHAREGPT_FILENAME = "ShareGPT_V3_unfiltered_cleaned_split.json"
NUM_PROMPTS = 16
MAX_NEW_TOKENS = 64

logger = logging.getLogger(__name__)


async def _make_request(session, url, payload):
    async with session.post(url=url, json=payload) as response:
        return await response.json()


def _build_generate_payload(text: str) -> dict:
    return {
        "text": text,
        "sampling_params": {"temperature": 0, "max_new_tokens": MAX_NEW_TOKENS},
        "return_routed_experts": True,
    }


def _load_prompts() -> List[str]:
    dataset_path = download_and_cache_hf_file(SHAREGPT_REPO_ID, SHAREGPT_FILENAME)
    with open(dataset_path) as f:
        data = json.load(f)
    texts: List[str] = []
    for s in data:
        if "conversations" in s and s["conversations"]:
            try:
                text = s["conversations"][0]["value"]
                if isinstance(text, str) and 0 < len(text) <= 2000:
                    texts.append(text)
            except (KeyError, IndexError, TypeError):
                continue
        if len(texts) >= NUM_PROMPTS:
            break
    if not texts:
        raise ValueError(
            "Could not load any prompts from ShareGPT; verify dataset cache"
        )
    return texts[:NUM_PROMPTS]


async def _collect_routed_experts(url: str, texts: Sequence[str]) -> List[np.ndarray]:
    async with aiohttp.ClientSession() as session:
        tasks = [
            asyncio.create_task(
                _make_request(session, f"{url}/generate", _build_generate_payload(t))
            )
            for t in texts
        ]
        responses = await asyncio.gather(*tasks)
    out: List[np.ndarray] = []
    for resp in responses:
        if "error" in resp:
            raise RuntimeError(f"server error: {resp['error']}")
        out.append(extract_routed_experts_from_meta_info(resp))
    return out


def _assert_routed_experts_equal(
    baseline: Sequence[np.ndarray], reference: Sequence[np.ndarray]
) -> None:
    if len(baseline) != len(reference):
        raise AssertionError(
            f"prompt count mismatch: baseline={len(baseline)} "
            f"reference={len(reference)}"
        )
    for idx, (b, r) in enumerate(zip(baseline, reference)):
        if b.shape != r.shape:
            raise AssertionError(
                f"prompt {idx}: shape mismatch baseline={b.shape} "
                f"reference={r.shape}"
            )
        if not np.array_equal(b, r):
            diff = int((b != r).sum())
            raise AssertionError(
                f"prompt {idx}: routed_experts cell-wise mismatch "
                f"({diff} cells differ). Draft layers are polluting the "
                "target R3 buffer; verify the opt-out for the active draft "
                "architecture is in place."
            )


@unittest.skipUnless(
    _RUN_MTP_R3,
    "Frozen-KV MTP R3 regression requires GPU + MTP weights; set "
    "SGLANG_RUN_MTP_R3_REGRESSION=1 plus SGLANG_MTP_R3_TARGET_MODEL to enable.",
)
class TestReturnRoutedExpertsFrozenKVMTP(CustomTestCase):
    """Failing-case regression: Frozen-KV MTP + overlap + cuda-graph
    + --enable-return-routed-experts must produce target-equivalent
    routed-experts output.

    The class itself skips entirely when `SGLANG_RUN_MTP_R3_REGRESSION` is
    unset (e.g. unit-test runs on a CPU host). When set, all test methods
    run real launches against `SGLANG_MTP_R3_TARGET_MODEL` and assert
    cell-wise equality; failures are real regressions, not skips.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        model_path = os.environ.get("SGLANG_MTP_R3_TARGET_MODEL")
        if not model_path:
            raise unittest.SkipTest(
                "SGLANG_MTP_R3_TARGET_MODEL not set; cannot launch target / MTP "
                "servers. Set it to a DeepSeek-V3-class checkpoint path or "
                "HuggingFace name."
            )
        cls.model_path = model_path
        cls.draft_model_path = (
            os.environ.get("SGLANG_MTP_R3_DRAFT_MODEL_PATH") or model_path
        )
        cls.texts = _load_prompts()
        cls.target_only_args = [
            "--enable-return-routed-experts",
            "--enable-deterministic-inference",
            "--tp",
            "4",
        ]

    @classmethod
    def _mtp_args(cls, piecewise_cuda_graph: bool) -> List[str]:
        args = [
            "--enable-return-routed-experts",
            "--enable-deterministic-inference",
            "--tp",
            "4",
            "--speculative-algorithm",
            "EAGLE",
            "--speculative-draft-model-path",
            cls.draft_model_path,
            "--speculative-num-steps",
            "1",
            "--speculative-eagle-topk",
            "1",
            "--speculative-num-draft-tokens",
            "2",
        ]
        if piecewise_cuda_graph:
            args.append("--enable-piecewise-cuda-graph")
        return args

    @classmethod
    def _launch_and_collect(cls, other_args: List[str]) -> List[np.ndarray]:
        process = popen_launch_server(
            cls.model_path,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )
        try:
            return asyncio.run(
                _collect_routed_experts(DEFAULT_URL_FOR_TEST, cls.texts)
            )
        finally:
            kill_process_tree(process.pid)

    def test_overlap_cuda_graph_default(self) -> None:
        """Default overlap path: MTP draft + overlap + cuda-graph must
        match the target-only baseline. Without the opt-out, draft writes
        pollute the target's R3 buffer and these arrays diverge."""
        baseline = self._launch_and_collect(self.target_only_args)
        reference = self._launch_and_collect(
            self._mtp_args(piecewise_cuda_graph=False)
        )
        _assert_routed_experts_equal(baseline, reference)

    def test_overlap_cuda_graph_bypass_piecewise(self) -> None:
        """BYPASS piecewise-CUDA-graph variant: rebuilt TopKConfig must
        carry capture_routed_experts=False, otherwise the rebuilt config
        defaults back to True and pollution returns silently."""
        baseline = self._launch_and_collect(self.target_only_args)
        reference = self._launch_and_collect(
            self._mtp_args(piecewise_cuda_graph=True)
        )
        _assert_routed_experts_equal(baseline, reference)


if __name__ == "__main__":
    unittest.main()
