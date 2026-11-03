from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

import asyncio
import copy
import dataclasses
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import parse_qsl, urlsplit

import pytest
from huggingface_hub import ResolvedRevision

from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.common.conn import (
    CommonKVBootstrapServer,
    CommonKVManager,
    PrefillServerInfo,
    _same_peer_identity,
)
from sglang.srt.disaggregation.peer_fingerprint import (
    build_peer_fingerprint,
    collect_peer_fingerprint,
    config_digest,
    peer_fingerprint_mismatches,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils.hf_transformers.config import HfModelConfigParser, get_config
from sglang.test.test_utils import CustomTestCase, published_topology

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
DFLASH = {
    **DSPARK,
    "algorithm": "DFLASH",
    "dspark_block_size": None,
    "dflash_block_size": 4,
}


def fingerprint(
    *,
    speculation=None,
    latent_row_bytes=576,
    index_k_layers=INDEX_K_LAYERS,
    attn_tp_size=8,
    **model_kwargs,
):
    num_draft = 6 if speculation else 0
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
    kwargs = dict(
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
    kwargs.update(model_kwargs)
    return build_peer_fingerprint(**kwargs)


class TestPeerFingerprint(CustomTestCase):
    def test_snapshot_identity_overrides_requested_revision(self):
        for role in ("target", "draft"):
            for revision in (None, "main"):
                with self.subTest(role=role, revision=revision):
                    kwargs = {
                        f"{role}_model_path": SNAPSHOT + "/checkpoint",
                        f"{role}_config": {"_commit_hash": "0" * 40},
                    }
                    local = fingerprint(speculation=DSPARK, **kwargs)
                    remote = fingerprint(
                        speculation=DSPARK,
                        **kwargs,
                        **{f"{role}_revision": revision},
                    )
                    self.assertEqual(peer_fingerprint_mismatches(local, remote), [])

    def test_loaded_checkpoint_identity_for_target_and_draft(self):
        sha = SNAPSHOT.rsplit("/", 1)[-1]
        cases = (
            ("main", sha, sha, sha, True),
            ("main", "main", sha, "0" * 40, False),
            ("main", "other", None, None, False),
            (None, sha, None, sha, True),
        )
        original_parse = HfModelConfigParser.parse
        for role in ("target", "draft"):
            for left, right, left_sha, right_sha, compatible in cases:
                with self.subTest(role=role, revisions=(left, right), sha=right_sha):
                    peers = []
                    for requested, commit in ((left, left_sha), (right, right_sha)):
                        with tempfile.TemporaryDirectory() as directory:
                            Path(directory, "config.json").write_text(
                                json.dumps({"model_type": "llama"})
                            )
                            resolved = (
                                ResolvedRevision(commit, initial=requested)
                                if commit
                                else requested
                            )
                            with (
                                patch(
                                    "sglang.srt.utils.hf_transformers.config.resolve_revision",
                                    return_value=resolved,
                                ) as resolve,
                                patch.object(
                                    HfModelConfigParser,
                                    "parse",
                                    autospec=True,
                                    side_effect=original_parse,
                                ) as parse,
                            ):
                                config = get_config(
                                    directory,
                                    trust_remote_code=False,
                                    revision=requested,
                                    cache_dir=directory,
                                    local_files_only=True,
                                    token=False,
                                    model_override_args={"_commit_hash": "stale"},
                                )
                            resolve.assert_called_once_with(
                                directory,
                                requested,
                                cache_dir=directory,
                                local_files_only=True,
                                token=False,
                            )
                            self.assertIs(parse.call_args.kwargs["revision"], resolved)
                            self.assertEqual(config._commit_hash, commit or requested)
                            peers.append(
                                fingerprint(
                                    speculation=DSPARK,
                                    **{
                                        f"{role}_model_path": "example/checkpoint",
                                        f"{role}_revision": requested,
                                        f"{role}_config": config,
                                    },
                                )
                            )
                    self.assertEqual(
                        peers[0][role]["config_digest"], peers[1][role]["config_digest"]
                    )
                    problems = peer_fingerprint_mismatches(*peers)
                    if compatible:
                        self.assertEqual(problems, [])
                    else:
                        self.assertTrue(any(f"{role} model" in p for p in problems))

    def test_pp_identity_ignores_rank_layout_but_not_checkpoint(self):
        first, other = fingerprint(), fingerprint()
        other["layout"]["kv_layer_ids"] = [72]
        self.assertTrue(_same_peer_identity(first, other))
        other["target"]["revision"] = "0" * 40
        self.assertFalse(_same_peer_identity(first, other))
        other["target"]["revision"] = None
        self.assertTrue(_same_peer_identity(first, other))
        self.assertTrue(_same_peer_identity(other, first))
        other["target"]["config_digest"] = "different"
        self.assertFalse(_same_peer_identity(first, other))
        self.assertTrue(peer_fingerprint_mismatches(first, other))

    def test_draft_in_same_repository_with_different_revision_is_not_bundled(self):
        kwargs = dict(
            speculation=DSPARK,
            target_model_path="example/checkpoint",
            target_config={"_commit_hash": "a" * 40},
            draft_model_path="example/checkpoint",
        )
        local = fingerprint(**kwargs, draft_config={"_commit_hash": "b" * 40})
        remote = fingerprint(**kwargs, draft_config={"_commit_hash": "c" * 40})
        self.assertTrue(
            any("draft model" in p for p in peer_fingerprint_mismatches(local, remote))
        )
        local = fingerprint(**kwargs, draft_config={"_commit_hash": "a" * 40})
        remote = fingerprint(**kwargs, draft_config={})
        self.assertEqual(peer_fingerprint_mismatches(local, remote), [])
        self.assertTrue(_same_peer_identity(local, remote))

    def test_prefill_injected_draft_kv_requires_a_matching_peer(self):
        for algorithm, settings in (
            (SpeculativeAlgorithm.DSPARK, DSPARK),
            (SpeculativeAlgorithm.DFLASH, DFLASH),
        ):
            speculation = {
                **settings,
                "requires_prefill_draft_kv": algorithm.injects_prefill_draft_kv(),
            }
            draft, target_only = fingerprint(speculation=speculation), fingerprint()
            for decode, prefill in (
                (draft, target_only),
                (target_only, draft),
                (draft, None),
            ):
                with self.subTest(
                    algorithm=algorithm,
                    decode=decode["speculation"],
                    legacy=prefill is None,
                ):
                    self.assertTrue(peer_fingerprint_mismatches(decode, prefill))

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
            "documented NEXTN across TP and step budgets": (
                fingerprint(speculation=eagle, attn_tp_size=1),
                fingerprint(
                    speculation={**eagle, "num_steps": 1, "num_draft_tokens": 2},
                    attn_tp_size=2,
                ),
            ),
        }
        for name, (decode, prefill) in cases.items():
            with self.subTest(name):
                self.assertEqual(peer_fingerprint_mismatches(decode, prefill), [])

    def test_generic_draft_kv_still_checks_checkpoint(self):
        eagle = {**DSPARK, "algorithm": "EAGLE", "requires_prefill_draft_kv": False}
        kwargs = dict(
            speculation=eagle, draft_model_path="draft", draft_config={"n": 1}
        )
        local = fingerprint(**kwargs, draft_revision="a" * 40, attn_tp_size=1)
        remote = fingerprint(**kwargs, draft_revision="b" * 40, attn_tp_size=2)
        self.assertTrue(
            any("draft model" in p for p in peer_fingerprint_mismatches(local, remote))
        )
        remote["draft"]["revision"] = None
        self.assertEqual(peer_fingerprint_mismatches(local, remote), [])
        remote["draft"]["config_digest"] = "different"
        self.assertTrue(peer_fingerprint_mismatches(local, remote))

    def test_dflash_layout_is_independent_of_pp_worker_presence(self):
        with published_topology(tp_size=4, pp_size=2, speculative_algorithm="DFLASH"):
            scheduler = SimpleNamespace(
                spec_algorithm=SpeculativeAlgorithm.DFLASH,
                model_config=SimpleNamespace(
                    hf_config={}, model_path=SNAPSHOT, revision=None
                ),
            )
            kv_args = SimpleNamespace(
                page_size=PAGE,
                kv_cache_dtype_str="fp8_e4m3",
                kv_layout=None,
                kv_item_lens=[128],
                kv_layer_ids=[0],
                num_draft_entries=0,
                state_types=[],
                state_item_lens=[],
            )
            peers = []
            for rank, worker in (
                (0, None),
                (
                    1,
                    SimpleNamespace(
                        draft_parallel_layout=SimpleNamespace(attn_tp_size=4)
                    ),
                ),
            ):
                scheduler.draft_worker = worker
                with get_parallel().override(
                    pp_rank=rank,
                    attn_tp_size=1,
                    attn_cp_size=4,
                    attn_dp_size=1,
                    attn_dp_enabled=False,
                    tp_group=SimpleNamespace(world_size=4),
                    enable_prefill_cp=True,
                ):
                    peers.append(
                        collect_peer_fingerprint(scheduler, kv_args, SimpleNamespace())
                    )
            self.assertEqual([p["layout"]["draft_attn_tp_size"] for p in peers], [4, 4])
            self.assertEqual(peer_fingerprint_mismatches(*peers), [])
            self.assertTrue(_same_peer_identity(*peers))

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


class TestFingerprintBootstrapCompatibility(CustomTestCase):
    def setUp(self):
        with patch.object(CommonKVBootstrapServer, "run"):
            self.server = CommonKVBootstrapServer("127.0.0.1", 0)
        self.server.attn_tp_size = self.server.attn_cp_size = 1
        self.server.dp_size = self.server.pp_size = 1
        self.server._registered_count = 1
        self.server.page_size = PAGE
        self.server.kv_cache_dtype = "fp8_e4m3"
        self.server.peer_fingerprint = fingerprint(attn_tp_size=1)

    def response(self, query):
        response = asyncio.run(
            self.server._handle_route_get(SimpleNamespace(query=query))
        )
        self.assertEqual(response.status, 200)
        return json.loads(response.text)

    def test_old_decode_does_not_receive_unknown_field(self):
        query = dict.fromkeys(
            ("prefill_dp_rank", "prefill_cp_rank", "target_tp_rank", "target_pp_rank"),
            "-1",
        )
        self.assertNotIn("peer_fingerprint", self.response(query))
        self.assertEqual(
            self.response({**query, "peer_fingerprint": "1"})["peer_fingerprint"],
            self.server.peer_fingerprint,
        )

    def test_pp_registration_checks_identity_before_recording_rank(self):
        first = fingerprint(
            speculation=DFLASH,
            attn_tp_size=1,
            draft_model_path="draft",
            draft_revision="b" * 40,
        )
        self.server.peer_fingerprint = first
        self.server.pp_size = 2
        row = dict(
            attn_tp_size=1,
            attn_tp_rank=0,
            attn_cp_size=1,
            attn_cp_rank=0,
            attn_dp_size=1,
            attn_dp_rank=0,
            pp_size=2,
            pp_rank=1,
            system_dp_size=1,
            system_dp_rank=0,
            rank_ip="10.0.0.1",
            rank_port=17000,
            page_size=PAGE,
            kv_cache_dtype="fp8_e4m3",
        )

        def register(peer):
            return asyncio.run(
                self.server._handle_route_put(
                    SimpleNamespace(
                        json=AsyncMock(return_value={**row, "peer_fingerprint": peer})
                    )
                )
            )

        for section, key, value in (
            ("target", "revision", "c" * 40),
            ("draft", "revision", "c" * 40),
            ("target", "config_digest", "different"),
            ("draft", "config_digest", "different"),
            ("speculation", "algorithm", "DSPARK"),
        ):
            other = copy.deepcopy(first)
            other[section][key] = value
            with self.subTest(section=section, key=key):
                self.assertEqual(register(other).status, 400)
                self.assertEqual(self.server._registered_count, 1)
                self.assertEqual(self.server.prefill_port_table, {})
        other = copy.deepcopy(first)
        other["target"]["revision"] = other["draft"]["revision"] = None
        other["layout"]["kv_layer_ids"] = [72]
        self.assertEqual(register(other).status, 200)
        self.assertTrue(self.server._is_ready())
        self.assertEqual(self.server.peer_fingerprint, first)
        self.assertEqual(register(None).status, 400)

    def test_new_decode_requests_fingerprint_and_handles_legacy_prefill(self):
        for legacy in (False, True):
            for speculation in (None, DSPARK, DFLASH):
                with self.subTest(legacy=legacy, speculation=speculation):
                    self.server.peer_fingerprint = (
                        None
                        if legacy
                        else fingerprint(speculation=speculation, attn_tp_size=1)
                    )
                    manager = CommonKVManager.__new__(CommonKVManager)
                    manager.prefill_info_table = {}
                    manager.kv_args = SimpleNamespace(page_size=PAGE, engine_rank=0)
                    manager.kv_cache_dtype_str = "fp8_e4m3"
                    manager.peer_fingerprint = fingerprint(
                        speculation=speculation, attn_tp_size=1
                    )
                    manager.attn_tp_size = manager.attn_cp_size = manager.pp_size = 1
                    manager.dcp_size = 1
                    manager.attn_cp_rank = manager.pp_rank = 0
                    manager.is_mla_backend = manager.is_hybrid_mla_backend = False

                    def get(url, **kwargs):
                        query = dict(parse_qsl(urlsplit(url).query))
                        self.assertEqual(query.get("peer_fingerprint"), "1")
                        response = MagicMock(status_code=200)
                        response.json.return_value = self.response(query)
                        return response

                    with patch(
                        "sglang.srt.disaggregation.common.conn.requests.get",
                        side_effect=get,
                    ):
                        if legacy and speculation:
                            with self.assertRaisesRegex(
                                RuntimeError, "no peer fingerprint"
                            ):
                                manager.try_ensure_parallel_info("prefill:8998")
                            self.assertEqual(manager.prefill_info_table, {})
                        else:
                            self.assertTrue(
                                manager.try_ensure_parallel_info("prefill:8998")
                            )
                            self.assertEqual(
                                manager.prefill_info_table[
                                    "prefill:8998"
                                ].peer_fingerprint,
                                self.server.peer_fingerprint,
                            )


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
        with self.subTest("decode-only EOS override"):
            self.assertEqual(
                config_digest({**string_only, "eos_token_id": [200001, 200008]}),
                config_digest(string_only),
            )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
