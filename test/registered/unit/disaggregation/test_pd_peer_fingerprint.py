from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

import dataclasses
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.common.conn import CommonKVManager, PrefillServerInfo
from sglang.srt.disaggregation.peer_fingerprint import (
    build_peer_fingerprint,
    config_digest,
    peer_fingerprint_mismatches,
)
from sglang.test.test_utils import CustomTestCase

PAGE = 64
SNAPSHOT = "/hub/models--m/snapshots/b74ad8b5498dc74868186fedd7e08d7d4c76227e"
# Mistral Large 4 under PD: 72 latent entries, index-K on the 44 top-k computing layers.
INDEX_K_LAYERS = [l < 17 or (l - 17) % 4 in (2, 3) for l in range(72)]
DSPARK = {
    "algorithm": "DSPARK",
    "num_draft_tokens": 4,
    "num_steps": 3,
    "eagle_topk": 1,
    "dspark_block_size": 4,
    "dflash_block_size": None,
    "requires_prefill_draft_kv": True,
}


def fingerprint(
    *,
    speculation=None,
    latent_row_bytes=576,
    index_k_layers=INDEX_K_LAYERS,
    attn_tp_size=8,
):
    num_draft = 6 if speculation and speculation["algorithm"] == "DSPARK" else 0
    kv_args = SimpleNamespace(
        page_size=PAGE,
        kv_cache_dtype_str="fp8_e4m3",
        kv_layout=None,
        kv_item_lens=[latent_row_bytes * PAGE] * 72 + [2048 * PAGE] * num_draft,
        kv_layer_ids=list(range(72)) + [0, 1, 2] * 2 * (num_draft // 6),
        num_draft_entries=num_draft,
        state_types=[StateType.DSA],
        state_item_lens=[[132 * PAGE if keep else 0 for keep in index_k_layers]],
    )
    return build_peer_fingerprint(
        kv_args=kv_args,
        target_config={"architectures": ["PixtralForConditionalGeneration"], "n": 72},
        target_model_path=SNAPSHOT,
        target_revision=None,
        speculation=speculation,
        draft_model_path=SNAPSHOT if speculation else None,
        attn_tp_size=attn_tp_size,
        attn_cp_size=1,
        pp_size=1,
        dcp_size=1,
        draft_attn_tp_size=attn_tp_size if speculation else None,
        one_to_one_capable=True,
    )


class TestPeerFingerprint(CustomTestCase):
    def test_one_sided_dspark_is_rejected_in_both_directions(self):
        dspark, target_only = fingerprint(speculation=DSPARK), fingerprint()
        for decode, prefill in ((dspark, target_only), (target_only, dspark)):
            with self.subTest(decode=decode["speculation"]):
                problems = peer_fingerprint_mismatches(decode, prefill)
                self.assertTrue(any("speculation algorithm" in p for p in problems))

    def test_active_kv_layout_mismatch_is_rejected(self):
        problems = peer_fingerprint_mismatches(
            fingerprint(latent_row_bytes=576), fingerprint(latent_row_bytes=656)
        )
        self.assertTrue(any("byte stride differs at entry 0" in p for p in problems))

    def test_compatible_peers_pass(self):
        # Decode-only speculation without prefill-injected draft KV is an
        # upstream capability, and an elided (zero-length) buffer is skipped.
        eagle = {**DSPARK, "algorithm": "EAGLE", "requires_prefill_draft_kv": False}
        decode_hisparse_index_k = [True] * 72
        cases = {
            "matched DSpark": (
                fingerprint(speculation=DSPARK),
                fingerprint(speculation=DSPARK),
            ),
            "decode-only EAGLE": (fingerprint(speculation=eagle), fingerprint()),
            "elided index-K": (
                fingerprint(index_k_layers=decode_hisparse_index_k),
                fingerprint(),
            ),
            "target-only MLA across TP": (
                fingerprint(attn_tp_size=8),
                fingerprint(attn_tp_size=4),
            ),
        }
        for name, (decode, prefill) in cases.items():
            with self.subTest(name):
                self.assertEqual(peer_fingerprint_mismatches(decode, prefill), [])

    def test_draft_kv_across_tp_is_rejected(self):
        problems = peer_fingerprint_mismatches(
            fingerprint(speculation=DSPARK, attn_tp_size=8),
            fingerprint(speculation=DSPARK, attn_tp_size=4),
        )
        self.assertTrue(any("no cross-TP draft relayout" in p for p in problems))

    def test_decode_rejects_before_recording_the_prefill(self):
        manager = CommonKVManager.__new__(CommonKVManager)
        manager.prefill_info_table = {}
        manager.kv_args = SimpleNamespace(page_size=PAGE)
        manager.kv_cache_dtype_str = "fp8_e4m3"
        manager.kv_layout = None
        manager.peer_fingerprint = fingerprint(speculation=DSPARK)
        info = PrefillServerInfo(
            attn_tp_size=8,
            attn_cp_size=1,
            dp_size=1,
            pp_size=1,
            page_size=PAGE,
            kv_cache_dtype="fp8_e4m3",
            follow_bootstrap_room=True,
            peer_fingerprint=fingerprint(),
        )
        response = MagicMock(status_code=200)
        response.json.return_value = dataclasses.asdict(info)
        with patch(
            "sglang.srt.disaggregation.common.conn.requests.get",
            return_value=response,
        ):
            with self.assertRaisesRegex(RuntimeError, "PD peer mismatch"):
                manager.try_ensure_parallel_info("prefill:8998")
        self.assertEqual(manager.prefill_info_table, {})


class TestConfigDigest(CustomTestCase):
    def test_integer_keyed_maps_digest_without_local_keys(self):
        # Bug regression: the release config's int-keyed id2label maps crashed PD
        # startup when the local-key filter called key.startswith.
        def config(labels, local):
            return {
                "architectures": ["PixtralForConditionalGeneration"],
                "id2label": labels,
                "text_config": {
                    "id2label": labels,
                    "_name_or_path": local,
                    "transformers_version": local,
                },
            }

        digest = config_digest(config({0: "LABEL_0", 1: "LABEL_1"}, "/prefill"))
        self.assertEqual(
            config_digest(config({1: "LABEL_1", 0: "LABEL_0"}, "/decode")), digest
        )
        self.assertNotEqual(
            config_digest(config({0: "LABEL_0", 1: "OTHER"}, "/prefill")), digest
        )

    def test_string_keyed_digest_is_unchanged(self):
        # Pinned from the string-only implementation, so peers on either build agree.
        string_only = {
            "architectures": ["PixtralForConditionalGeneration"],
            "text_config": {
                "hidden_size": 12288,
                "_name_or_path": "/a",
                "transformers_version": "5.0",
            },
        }
        self.assertEqual(
            config_digest(string_only),
            "78ddb9cf97245bb430417688c2d7d82e568ce3902dae9bcdebc52b59a79ddba9",
        )


if __name__ == "__main__":
    unittest.main()
