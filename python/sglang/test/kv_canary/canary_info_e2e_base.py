from __future__ import annotations

import unittest
from typing import Any, ClassVar

from sglang.srt.kv_canary.canary_info.checker import CanaryInfoMismatch
from sglang.srt.kv_canary.perturb.detokenizer_swap import (
    DetokenizerSwapRecord,
    DetokenizerSwapSide,
)
from sglang.test.kv_canary.canary_info_utils import (
    make_canary_info_payload,
    post_parallel_payloads,
    recheck_canary_infos,
)
from sglang.test.kv_canary.e2e_base import CanaryE2EBase
from sglang.test.kv_canary.violation_log_utils import wait_for_log_records

SWAP_AT_OUTPUT = 16


class CanaryInfoE2EBase(CanaryE2EBase):
    extra_env: ClassVar[dict[str, str]] = {"SGLANG_KV_CANARY_ENABLE_INFO": "1"}
    model_mode = "eagle"
    extra_server_args = ("--skip-tokenizer-init",)
    num_requests: ClassVar[int] = 8
    prompt_length: ClassVar[int] = 300
    max_new_tokens: ClassVar[int] = 128

    @classmethod
    def setUpClass(cls) -> None:
        if cls in (CanaryInfoE2EBase, DetokenizerSwapE2EBase):
            raise unittest.SkipTest("abstract base")
        super().setUpClass()

    def run_requests(self) -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
        payloads = [
            {
                "rid": f"canary-info-{i}",
                "stream": True,
                **make_canary_info_payload(
                    index=i,
                    prompt_length=self.prompt_length + i,
                    max_new_tokens=self.max_new_tokens,
                ),
            }
            for i in range(self.num_requests)
        ]
        results = post_parallel_payloads(
            url=self.base_url + "/generate", payloads=payloads, timeout=240.0
        )
        by_rid = {}
        for payload, result in zip(payloads, results, strict=True):
            self.assertEqual(result.get("status_code"), 200, result)
            body = result["json"]
            self.assertEqual(body["meta_info"]["id"], payload["rid"])
            by_rid[payload["rid"]] = (payload, body)
        return by_rid

    def recheck(
        self, payload: dict[str, Any], body: dict[str, Any]
    ) -> CanaryInfoMismatch | None:
        return recheck_canary_infos(
            rid=payload["rid"],
            input_ids=payload["input_ids"],
            sampling_params=payload["sampling_params"],
            output_ids=body["output_ids"],
            canary_infos=body["meta_info"].get("canary_infos"),
        )

    def collect_swaps(self) -> dict[str, DetokenizerSwapSide]:
        records, _ = wait_for_log_records(
            self._captured_log_text,
            DetokenizerSwapRecord.find_all,
            flush_wait_seconds=3.0,
            max_retries=10,
        )
        sides = {}
        for record in records:
            for side, foreign in ((record.a, record.b), (record.b, record.a)):
                self.assertNotIn(side.rid, sides)
                self.assertGreaterEqual(side.position, SWAP_AT_OUTPUT)
                sides[side.rid] = DetokenizerSwapSide(
                    rid=side.rid, position=side.position, token=foreign.token
                )
        self.assertGreaterEqual(len(sides), 2, records)
        return sides


class DetokenizerSwapE2EBase(CanaryInfoE2EBase):
    extra_server_args = ()
    extra_env: ClassVar[dict[str, str]] = {
        **CanaryInfoE2EBase.extra_env,
        "SGLANG_KV_CANARY_PERTURB_DETOKENIZER_SWAP_AT_OUTPUT": str(SWAP_AT_OUTPUT),
    }


def first_mismatch_positions(log_text: str) -> dict[str, int]:
    positions: dict[str, int] = {}
    for mismatch in CanaryInfoMismatch.find_all(log_text):
        positions.setdefault(mismatch.rid, mismatch.position)
    return positions
