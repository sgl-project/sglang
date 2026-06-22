"""End-to-end regression: a MoE speculative draft must not pollute the
target's R3 routed-experts (`--enable-return-routed-experts`) capture buffer.

Both servers run the same Qwen3-30B-A3B target with
`--enable-return-routed-experts`; the reference additionally attaches the
Qwen3-MoE EAGLE3 draft. Draft workers opt out of capture via the
ModelRunner-level pass `disable_routed_experts_capture_for_draft`, which
flips `allow_routed_experts_capture=False` on every draft-side MoE `TopK`;
if one still wrote through the process-global capturer, the
per-output-token experts captured with the draft attached would diverge
from the target-only baseline.

The target + MoE-draft pair mirrors
`test/registered/spec/eagle/test_eagle_dp_attention.py`
(`Qwen/Qwen3-30B-A3B` + `Tengyunw/qwen3_30b_moe_eagle3`). The capture setup
and the per-(token, layer) expert-set comparison (with the same <10%
deterministic-inference jitter tolerance) mirror
`test/registered/rl/test_return_routed_experts.py`.
"""

import asyncio
import json
import logging
import unittest
from typing import List

import aiohttp
import torch
from torch.nn.utils.rnn import pad_sequence

from sglang.benchmark.utils import download_and_cache_hf_file
from sglang.srt.state_capturer.routed_experts import (
    extract_routed_experts_from_meta_info,
)
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_DRAFT_MODEL_EAGLE_DP_ATTN,
    DEFAULT_TARGET_MODEL_EAGLE_DP_ATTN,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

# Two server launches (target-only baseline + EAGLE3 draft reference) on the
# same multi-GPU runner the target-only R3 regression uses.
register_cuda_ci(est_time=600, stage="extra-b", runner_config="4-gpu-h100")

MODEL_PATH = DEFAULT_TARGET_MODEL_EAGLE_DP_ATTN  # Qwen/Qwen3-30B-A3B
DRAFT_PATH = DEFAULT_DRAFT_MODEL_EAGLE_DP_ATTN  # Tengyunw/qwen3_30b_moe_eagle3 (MoE)

# Qwen3-30B-A3B per-token routed-experts shape.
_NUM_LAYERS = 48
_TOPK = 8

SHAREGPT_REPO_ID = "anon8231489123/ShareGPT_Vicuna_unfiltered"
SHAREGPT_FILENAME = "ShareGPT_V3_unfiltered_cleaned_split.json"
NUM_PROMPTS = 32
MAX_NEW_TOKENS = 64
# Deterministic inference is not bit-identical across the spec-decode and
# autoregressive scheduling paths; the established target-only regression
# allows the same drift. Draft pollution would push mismatches far above this.
MISMATCH_TOLERANCE = 0.10

logger = logging.getLogger(__name__)


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
        raise ValueError("Could not load any prompts from ShareGPT; verify cache")
    return texts[:NUM_PROMPTS]


def _generate_payload(text: str) -> dict:
    return {
        "text": text,
        "sampling_params": {"temperature": 0, "max_new_tokens": MAX_NEW_TOKENS},
        "return_routed_experts": True,
    }


async def _post(session, url, payload):
    async with session.post(url=url, json=payload) as response:
        return await response.json()


def _check_expert_ids_valid(experts: List) -> None:
    tensor_list = [torch.tensor(seq) for seq in experts]
    padded = pad_sequence(tensor_list, batch_first=True, padding_value=0)
    if not ((padded >= 0) & (padded <= 127)).all():
        raise ValueError(
            f"expert indices out of range [0, 127]: max={padded.max()} "
            f"min={padded.min()}"
        )


def _count_mismatches(baseline, reference) -> int:
    total = 0
    for b_seq, r_seq in zip(baseline, reference):
        for b_tok, r_tok in zip(b_seq, r_seq):
            for b_topk, r_topk in zip(b_tok, r_tok):
                set_b, set_r = set(b_topk), set(r_topk)
                if set_b != set_r:
                    total += len(set_b - set_r)
                if len(b_topk) != len(set_b):
                    raise ValueError(f"duplicate expert ids in baseline: {b_topk}")
    return total


class TestReturnRoutedExpertsEagleMoEDraft(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        common = [
            "--trust-remote-code",
            "--enable-return-routed-experts",
            "--enable-deterministic-inference",
            "--tp-size",
            "4",
            "--attention-backend",
            "fa3",
            "--mem-fraction-static",
            "0.75",
            "--cuda-graph-max-bs",
            "64",
        ]
        cls.baseline_args = common
        cls.reference_args = common + [
            "--speculative-algorithm",
            "EAGLE3",
            "--speculative-num-steps",
            "6",
            "--speculative-eagle-topk",
            "10",
            "--speculative-num-draft-tokens",
            "32",
            "--speculative-draft-model-path",
            DRAFT_PATH,
        ]
        cls.texts = _load_prompts()
        cls.baseline = cls._collect(cls.baseline_args)
        cls.reference = cls._collect(cls.reference_args)

    @classmethod
    def _collect(cls, other_args) -> List:
        process = popen_launch_server(
            MODEL_PATH,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )
        try:
            responses = asyncio.run(cls._collect_async())
        finally:
            kill_process_tree(process.pid)
        out = []
        for resp in responses:
            if "error" in resp:
                raise RuntimeError(f"server error: {resp['error']}")
            out.append(
                extract_routed_experts_from_meta_info(resp).reshape(
                    -1, _NUM_LAYERS, _TOPK
                )
            )
        return out

    @classmethod
    async def _collect_async(cls):
        async with aiohttp.ClientSession() as session:
            tasks = [
                asyncio.create_task(
                    _post(
                        session,
                        f"{DEFAULT_URL_FOR_TEST}/generate",
                        _generate_payload(t),
                    )
                )
                for t in cls.texts
            ]
            return await asyncio.gather(*tasks)

    def test_draft_does_not_pollute_routed_experts(self):
        _check_expert_ids_valid(self.baseline)
        _check_expert_ids_valid(self.reference)

        total = sum(len(seq) for seq in self.baseline) * _NUM_LAYERS * _TOPK
        mismatches = _count_mismatches(self.baseline, self.reference)
        ratio = mismatches / total if total else 0.0
        logger.info(f"routed_experts mismatches: {mismatches}/{total} ({ratio:.4%})")
        self.assertLess(
            ratio,
            MISMATCH_TOLERANCE,
            f"MoE draft is polluting the target's R3 capture buffer: "
            f"{mismatches}/{total} ({ratio:.4%}) experts differ from the "
            "target-only baseline",
        )


if __name__ == "__main__":
    unittest.main()
