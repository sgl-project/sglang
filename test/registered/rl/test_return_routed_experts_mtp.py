"""End-to-end regression for `--enable-return-routed-experts` under the
draft worker paths the plan calls out: Frozen-KV MTP (the failing case)
and EAGLE under overlap + CUDA graph (the worker path that exercises the
per-`TopKConfig` opt-out at the most aggressive runtime).

Upstream limitation (recorded for AC-6):
  - `server_args.py:3727-3737` forces `--disable-overlap-schedule` when
    `--speculative-algorithm FROZEN_KV_MTP` is selected.
  - `spec_info.py:125-145` raises `ValueError` if overlap is requested
    together with FROZEN_KV_MTP.
  - `server_args._resolve_speculative_algorithm_alias` (server_args.py
    :325-355) only auto-promotes NEXTN/EAGLE to FROZEN_KV_MTP when the
    draft model architecture is `Gemma4AssistantForCausalLM`.
  - Consequence: the original plan's "Frozen-KV MTP + overlap + CUDA
    graph" contract cannot be exercised as a single configuration on
    today's runtime. It will be reachable once upstream lands
    overlap-capable Frozen-KV MTP.

This file therefore exercises the two reachable axes that, together,
prove the per-model opt-out holds where the failing case lives:

  - `FrozenKVMTPWorker` (via Gemma4 promotion) with CUDA graph, overlap
    disabled. Proves the failing-case worker path no longer pollutes the
    target's R3 buffer.
  - EAGLE draft with overlap + CUDA graph + (optionally) piecewise CUDA
    graph (BYPASS). Proves the per-model opt-out works under the
    overlap-capable runtime.

Both variants assert per-layer shape (`num_hidden_layers`,
`completion_tokens`, `num_experts_per_tok`) and cell-wise equality
against a target-only baseline at `temperature=0`/greedy.
"""

import asyncio
import json
import logging
import os
import unittest
from typing import List, Sequence

import aiohttp
import numpy as np

from sglang.benchmark.utils import download_and_cache_hf_file
from sglang.srt.state_capturer.routed_experts import (
    extract_routed_experts_from_meta_info,
)
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


# Register on the same multi-GPU runner the existing target-only R3
# regression uses. `est_time` accounts for two server launches plus the
# extra Frozen-KV MTP warmup.
register_cuda_ci(est_time=600, stage="extra-b", runner_config="4-gpu-h100")


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


async def _collect_meta_responses(url: str, texts: Sequence[str]) -> List[dict]:
    async with aiohttp.ClientSession() as session:
        tasks = [
            asyncio.create_task(
                _make_request(session, f"{url}/generate", _build_generate_payload(t))
            )
            for t in texts
        ]
        return await asyncio.gather(*tasks)


def _decode_routed_experts(
    responses: Sequence[dict],
    *,
    expected_num_layers: int,
    expected_topk: int,
) -> List[np.ndarray]:
    """Decode each response's base64-encoded routed_experts and reshape
    into `(completion_tokens, expected_num_layers, expected_topk)`. The
    completion_tokens dimension is inferred from each response (variable
    across prompts).
    """
    out: List[np.ndarray] = []
    for resp in responses:
        if "error" in resp:
            raise RuntimeError(f"server error: {resp['error']}")
        flat = extract_routed_experts_from_meta_info(resp)
        # flat is a 1-D int32 array; total cells must divide cleanly into
        # (num_layers, topk) per token.
        per_token_cells = expected_num_layers * expected_topk
        if flat.size % per_token_cells != 0:
            raise AssertionError(
                f"routed_experts size {flat.size} does not divide "
                f"num_layers*topk = {per_token_cells}; the server's "
                "per-layer shape contract may be broken"
            )
        completion_tokens = flat.size // per_token_cells
        out.append(flat.reshape(completion_tokens, expected_num_layers, expected_topk))
    return out


def _assert_routed_experts_equal(
    baseline: Sequence[np.ndarray],
    reference: Sequence[np.ndarray],
    *,
    expected_num_layers: int,
    expected_topk: int,
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
                f"reference={r.shape}; layer count or topk drifted"
            )
        if b.shape[1] != expected_num_layers or b.shape[2] != expected_topk:
            raise AssertionError(
                f"prompt {idx}: shape contract violated "
                f"(got {b.shape}, expected (*, {expected_num_layers}, "
                f"{expected_topk}))"
            )
        if not np.array_equal(b, r):
            diff = int((b != r).sum())
            raise AssertionError(
                f"prompt {idx}: routed_experts cell-wise mismatch "
                f"({diff} cells differ). Draft layers are polluting the "
                "target R3 buffer; verify the opt-out for the active draft "
                "architecture is in place."
            )


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


@unittest.skipUnless(
    _RUN_MTP_R3,
    "MTP R3 regression requires GPU + speculative-draft weights; set "
    "SGLANG_RUN_MTP_R3_REGRESSION=1 plus model env vars to enable.",
)
class TestReturnRoutedExpertsFrozenKVMTP(CustomTestCase):
    """Failing-case path: Frozen-KV MTP (via Gemma4 promotion).

    Required env:
      - `SGLANG_MTP_R3_TARGET_MODEL`: Gemma4-compatible target model path.
      - `SGLANG_MTP_R3_GEMMA4_DRAFT_PATH`: path to a model whose
        `config.architectures[0]` is `Gemma4AssistantForCausalLM` so
        `_resolve_speculative_algorithm_alias` promotes NEXTN -> FROZEN_KV_MTP.
      - `SGLANG_MTP_R3_NUM_LAYERS` and `SGLANG_MTP_R3_TOPK`: expected
        target-model layer count and per-token topk (used by the shape
        assertion). Required because the test cannot probe the model
        config without launching it; misconfiguration is caught early.

    Variants:
      - default: CUDA graph on, overlap OFF (only supported FROZEN_KV_MTP config today).
      - piecewise: CUDA graph on + `--enable-piecewise-cuda-graph` (BYPASS path).
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.target_path = _env("SGLANG_MTP_R3_TARGET_MODEL")
        cls.gemma4_draft = _env("SGLANG_MTP_R3_GEMMA4_DRAFT_PATH")
        cls.num_layers = int(_env("SGLANG_MTP_R3_NUM_LAYERS", "0"))
        cls.topk = int(_env("SGLANG_MTP_R3_TOPK", "0"))
        if not (cls.target_path and cls.gemma4_draft and cls.num_layers and cls.topk):
            raise unittest.SkipTest(
                "Frozen-KV MTP regression requires SGLANG_MTP_R3_TARGET_MODEL, "
                "SGLANG_MTP_R3_GEMMA4_DRAFT_PATH, SGLANG_MTP_R3_NUM_LAYERS, "
                "and SGLANG_MTP_R3_TOPK environment variables."
            )
        cls.texts = _load_prompts()
        # Target-only baseline runs without speculative decoding.
        cls.target_only_args = [
            "--enable-return-routed-experts",
            "--enable-deterministic-inference",
            "--tp",
            "4",
        ]

    @classmethod
    def _mtp_args(cls, piecewise_cuda_graph: bool) -> List[str]:
        # FROZEN_KV_MTP is auto-selected when the draft model's
        # architecture is Gemma4AssistantForCausalLM. server_args.py
        # automatically sets --disable-overlap-schedule for FROZEN_KV_MTP,
        # so this configuration intentionally does not request overlap.
        args = [
            "--enable-return-routed-experts",
            "--enable-deterministic-inference",
            "--tp",
            "4",
            "--speculative-algorithm",
            "NEXTN",  # promoted to FROZEN_KV_MTP when draft is Gemma4 assistant
            "--speculative-draft-model-path",
            cls.gemma4_draft,
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
            cls.target_path,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )
        try:
            responses = asyncio.run(
                _collect_meta_responses(DEFAULT_URL_FOR_TEST, cls.texts)
            )
        finally:
            kill_process_tree(process.pid)
        return _decode_routed_experts(
            responses,
            expected_num_layers=cls.num_layers,
            expected_topk=cls.topk,
        )

    def test_frozen_kv_mtp_default(self) -> None:
        """FROZEN_KV_MTP default: CUDA graph on, overlap OFF (the only
        configuration the current runtime supports for this worker)."""
        baseline = self._launch_and_collect(self.target_only_args)
        reference = self._launch_and_collect(
            self._mtp_args(piecewise_cuda_graph=False)
        )
        _assert_routed_experts_equal(
            baseline,
            reference,
            expected_num_layers=self.num_layers,
            expected_topk=self.topk,
        )

    def test_frozen_kv_mtp_piecewise_bypass(self) -> None:
        """FROZEN_KV_MTP + `--enable-piecewise-cuda-graph` exercises the
        BYPASS path that reconstructs `TopKConfig` from scalar args."""
        baseline = self._launch_and_collect(self.target_only_args)
        reference = self._launch_and_collect(
            self._mtp_args(piecewise_cuda_graph=True)
        )
        _assert_routed_experts_equal(
            baseline,
            reference,
            expected_num_layers=self.num_layers,
            expected_topk=self.topk,
        )


@unittest.skipUnless(
    _RUN_MTP_R3,
    "MTP R3 regression requires GPU + speculative-draft weights; set "
    "SGLANG_RUN_MTP_R3_REGRESSION=1 plus model env vars to enable.",
)
class TestReturnRoutedExpertsEagleOverlap(CustomTestCase):
    """Overlap-capable path: EAGLE draft + overlap + CUDA graph.

    The original plan's "overlap + CUDA graph" axis is currently
    reachable only through the EAGLE worker (not FROZEN_KV_MTP). Verifying
    that the per-`TopKConfig` opt-out holds under overlap proves the
    structural opt-out's runtime contract on the harder scheduling axis.

    Required env:
      - `SGLANG_MTP_R3_TARGET_MODEL`: any MoE target with EAGLE-compatible
        draft.
      - `SGLANG_MTP_R3_EAGLE_DRAFT_PATH`: EAGLE draft model path that is
        NOT Gemma4AssistantForCausalLM (avoid the FROZEN_KV_MTP promotion).
      - `SGLANG_MTP_R3_NUM_LAYERS`, `SGLANG_MTP_R3_TOPK`: as above.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.target_path = _env("SGLANG_MTP_R3_TARGET_MODEL")
        cls.eagle_draft = _env("SGLANG_MTP_R3_EAGLE_DRAFT_PATH")
        cls.num_layers = int(_env("SGLANG_MTP_R3_NUM_LAYERS", "0"))
        cls.topk = int(_env("SGLANG_MTP_R3_TOPK", "0"))
        if not (cls.target_path and cls.eagle_draft and cls.num_layers and cls.topk):
            raise unittest.SkipTest(
                "EAGLE overlap regression requires SGLANG_MTP_R3_TARGET_MODEL, "
                "SGLANG_MTP_R3_EAGLE_DRAFT_PATH, SGLANG_MTP_R3_NUM_LAYERS, "
                "and SGLANG_MTP_R3_TOPK environment variables."
            )
        cls.texts = _load_prompts()
        cls.target_only_args = [
            "--enable-return-routed-experts",
            "--enable-deterministic-inference",
            "--tp",
            "4",
        ]

    @classmethod
    def _eagle_args(cls, piecewise_cuda_graph: bool) -> List[str]:
        # Overlap is allowed for EAGLE; do not pass --disable-overlap-schedule.
        args = [
            "--enable-return-routed-experts",
            "--enable-deterministic-inference",
            "--tp",
            "4",
            "--speculative-algorithm",
            "EAGLE",
            "--speculative-draft-model-path",
            cls.eagle_draft,
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
            cls.target_path,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )
        try:
            responses = asyncio.run(
                _collect_meta_responses(DEFAULT_URL_FOR_TEST, cls.texts)
            )
        finally:
            kill_process_tree(process.pid)
        return _decode_routed_experts(
            responses,
            expected_num_layers=cls.num_layers,
            expected_topk=cls.topk,
        )

    def test_eagle_overlap_cuda_graph_default(self) -> None:
        baseline = self._launch_and_collect(self.target_only_args)
        reference = self._launch_and_collect(
            self._eagle_args(piecewise_cuda_graph=False)
        )
        _assert_routed_experts_equal(
            baseline,
            reference,
            expected_num_layers=self.num_layers,
            expected_topk=self.topk,
        )

    def test_eagle_overlap_cuda_graph_bypass_piecewise(self) -> None:
        baseline = self._launch_and_collect(self.target_only_args)
        reference = self._launch_and_collect(
            self._eagle_args(piecewise_cuda_graph=True)
        )
        _assert_routed_experts_equal(
            baseline,
            reference,
            expected_num_layers=self.num_layers,
            expected_topk=self.topk,
        )


if __name__ == "__main__":
    unittest.main()
