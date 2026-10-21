"""Catalog validation, scoring, native state transitions, and output transport."""

import argparse
import itertools
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import msgspec
import torch

from sglang.srt.beam_search.beam_group import BeamGroup
from sglang.srt.beam_search.catalog import (
    DeviceItemCatalog,
    ItemCatalog,
    get_catalog,
    load_catalogs,
)
from sglang.srt.beam_search.coordinator import BeamCoordinator
from sglang.srt.beam_search.output import (
    _build_beam_search_out_dict,
    pack_beam_search_output,
)
from sglang.srt.managers.io_struct import BeamSearchOutput
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


def records(roots=(0, 2, 5), depth=3):
    return [
        {"item_id": i, "token_ids": [root, *suffix]}
        for i, (root, suffix) in enumerate(
            itertools.product(roots, itertools.product((1, 4), repeat=depth - 1))
        )
    ]


def coordinator():
    return BeamCoordinator(
        model_config=SimpleNamespace(vocab_size=8, is_encoder_decoder=False),
        spec_algorithm=SimpleNamespace(is_none=lambda: True),
        dllm_enabled=False,
        max_req_len=8192,
        req_to_token_pool=SimpleNamespace(size=64, device="cpu"),
        token_to_kv_pool_allocator=None,
        tree_cache=SimpleNamespace(
            supports_swa=lambda: False, supports_mamba=lambda: False
        ),
        future_map=None,
    )


class TestGRCatalog(CustomTestCase):
    def tearDown(self):
        load_catalogs.cache_clear()

    def test_validation_and_item_zero(self):
        cat = ItemCatalog.from_records(records(), 8)
        self.assertEqual((cat.depth, cat.root_width), (3, 3))
        self.assertEqual(
            cat.resolve([0, 1, 1]), {"item_id": 0, "item_ids": [0], "is_complete": True}
        )
        self.assertEqual(
            cat.resolve([0, 1]), {"item_id": None, "item_ids": [], "is_complete": False}
        )
        for budget in (1, 2, 3):
            cat.validate_budget(budget)
        for budget in (0, 4, None, True, 1.5):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                cat.validate_budget(budget)

    def test_invalid_catalogs(self):
        good = {"item_id": "a", "token_ids": [0, 1]}
        cases = [
            [],
            [None],
            [{}],
            [{**good, "item_id": []}],
            [{**good, "item_id": True}],
            [{**good, "token_ids": []}],
            [{**good, "token_ids": [True]}],
            [{**good, "token_ids": [1.0]}],
            [{**good, "token_ids": [-1]}],
            [{**good, "token_ids": [8]}],
            [good, good],
            [good, {**good, "item_id": "b"}],
            [good, {"item_id": "b", "token_ids": [2]}],
        ]
        for rows in cases:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                ItemCatalog.from_records(rows, 8)

    def test_manifest_relative_paths_and_immutable_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            (p / "items.jsonl").write_text("\n".join(map(json.dumps, records())))
            manifest = p / "catalogs.json"
            manifest.write_text(json.dumps({"items": "items.jsonl"}))
            cat = get_catalog(str(manifest), "items", 8)
            (p / "items.jsonl").write_text("invalid now")
            self.assertIs(get_catalog(str(manifest), "items", 8), cat)
            with self.assertRaisesRegex(ValueError, "Unknown gr_catalog"):
                get_catalog(str(manifest), "missing", 8)
            with self.assertRaisesRegex(ValueError, "requires --gr-catalog-manifest"):
                get_catalog(None, "items", 8)
            load_catalogs.cache_clear()
            with self.assertRaisesRegex(ValueError, "Cannot load"):
                get_catalog(str(manifest), "items", 8)

    def test_cli_and_sampling_transport(self):
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        args = ServerArgs.from_cli_args(
            parser.parse_args(
                [
                    "--model-path",
                    "dummy",
                    "--enable-gr",
                    "--gr-catalog-manifest",
                    "/tmp/items.json",
                ]
            )
        )
        self.assertEqual(args.resolved_dict()["gr_catalog_manifest"], "/tmp/items.json")
        params = SamplingParams(
            beam_width=4, custom_params={"gr_catalog": "items"}, max_new_tokens=3
        )
        params.verify(8)
        encoded = msgspec.msgpack.encode(params)
        self.assertEqual(
            msgspec.msgpack.decode(encoded, type=SamplingParams).gr_catalog, "items"
        )
        for name, width in (
            ("", 4),
            ([], 4),
            ("items", 1),
            ("items", None),
            ("items", 2.5),
        ):
            with self.subTest(name=name, width=width), self.assertRaises(ValueError):
                SamplingParams(
                    beam_width=width, custom_params={"gr_catalog": name}
                ).verify(8)

    def test_admission_clamps_width_and_validates_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            (p / "items.jsonl").write_text(
                "\n".join(map(json.dumps, records(roots=(0,))))
            )
            manifest = p / "catalogs.json"
            manifest.write_text(json.dumps({"items": "items.jsonl"}))
            for enabled, budget, error_text in (
                (True, 3, None),
                (False, 3, "--enable-gr"),
                (True, 4, "max_new_tokens"),
            ):
                with self.subTest(enabled=enabled, budget=budget):
                    c = coordinator()
                    req = SimpleNamespace(
                        sampling_params=SamplingParams(
                            beam_width=4,
                            n=4,
                            max_new_tokens=budget,
                            custom_params={"gr_catalog": "items"},
                        ),
                        session_id=None,
                        lora_id=None,
                        origin_input_ids=[1, 2],
                        eos_token_ids=[0],
                        tokenizer=None,
                    )
                    recv = SimpleNamespace(
                        session_params=None,
                        return_logprob=False,
                        return_hidden_states=False,
                        return_sampling_mask=False,
                        return_routed_experts=False,
                    )
                    config = {
                        "get_serving": SimpleNamespace(
                            enable_gr=enabled, gr_catalog_manifest=str(manifest)
                        ),
                        "get_disagg": SimpleNamespace(disaggregation_mode="null"),
                        "get_schedule": SimpleNamespace(page_size=1),
                        "get_memory": SimpleNamespace(
                            enable_hisparse=False, enable_hierarchical_cache=False
                        ),
                        "get_parallel": SimpleNamespace(pp_size=1),
                        "is_dp_attention_enabled": False,
                    }
                    with ExitStack() as stack:
                        for name, value in config.items():
                            stack.enter_context(
                                patch(
                                    "sglang.srt.beam_search.coordinator." + name,
                                    return_value=value,
                                )
                            )
                        error = c.validate_and_init(req, recv)
                    if error_text:
                        self.assertIn(error_text, error)
                    else:
                        self.assertIsNone(error)
                        self.assertEqual(req.beam_group.beam_width, 1)
                        self.assertEqual(req.beam_group.catalog_nodes.tolist(), [0])
                        self.assertEqual(req.beam_group.num_return, 4)

    def test_device_mask_nodes_and_legal_normalization(self):
        catalog = DeviceItemCatalog(ItemCatalog.from_records(records(), 8), "cpu")
        nodes = catalog.initial_nodes()
        logits = torch.tensor([[1.0, 999.0, 2.0, 88.0, 55.0, 3.0, 99.0, 99.0]])
        values, tokens = catalog.topk(logits, nodes, 3, prefill=True)
        self.assertEqual(tokens.tolist(), [[5, 2, 0]])
        self.assertAlmostEqual(values.exp().sum().item(), 1.0, places=6)
        # EOS=0 is just data. Repeated and reordered parents produce independent nodes.
        next_nodes = catalog.advance(nodes, torch.tensor([0, 0, 0]), tokens[0])
        nodes = catalog.advance(
            next_nodes, torch.tensor([2, 0, 2]), torch.tensor([1, 4, 4])
        )
        self.assertNotEqual(nodes[0].item(), nodes[2].item())
        self.assertEqual(
            catalog.allowed(nodes).nonzero()[:, 1].tolist(), [1, 4, 1, 4, 1, 4]
        )
        self.assertFalse(catalog.allowed(torch.tensor([999])).any())

    def test_selection_history_and_scores_against_enumerated_oracle(self):
        torch.manual_seed(7)
        cat = DeviceItemCatalog(ItemCatalog.from_records(records(), 8), "cpu")
        width = 3
        group = BeamGroup(
            beam_width=width, num_candidates=width, max_new_tokens=3, length_penalty=0
        )
        group.catalog = cat
        group.catalog_id = "items"
        group.catalog_nodes = cat.initial_nodes()
        c = coordinator()
        paths, scores = [()], [0.0]
        for step in range(3):
            logits = torch.randn(len(paths), 8)
            candidates = []
            for parent, (path, cum) in enumerate(zip(paths, scores)):
                legal = sorted({p[step] for p in cat.catalog.items if p[:step] == path})
                vals = logits[parent, legal].double()
                normalized = vals - torch.logsumexp(vals, 0)
                candidates += [
                    (cum + float(v), parent, t) for t, v in zip(legal, normalized)
                ]
            selected = sorted(candidates, reverse=True)[:width]
            old_paths = paths
            paths = [old_paths[parent] + (t,) for _, parent, t in selected]
            scores = [v for v, _, _ in selected]
            vals, toks = cat.topk(logits, group.catalog_nodes, width, prefill=step == 0)
            c._select_group(
                group, vals, toks, step + 1, preserve_prefill_order=step == 0
            )
            # Commit is delayed until after selection, like the real scheduler.
            group.commit_pending(step + 1)
        result = group.finalize()
        self.assertEqual([tuple(r.tokens) for r in result], paths)
        for r, expected in zip(result, scores):
            self.assertAlmostEqual(r.cum_logprob, expected, places=6)

    def test_deferred_commit_and_final_overshoot(self):
        cat = DeviceItemCatalog(ItemCatalog.from_records(records(depth=2), 8), "cpu")
        group = BeamGroup(
            beam_width=3, num_candidates=3, max_new_tokens=2, length_penalty=0
        )
        group.catalog = cat
        group.catalog_nodes = cat.initial_nodes()
        c = coordinator()
        for tick in (1, 2, 3):
            logits = torch.zeros(group.catalog_nodes.numel(), 8)
            vals, toks = cat.topk(logits, group.catalog_nodes, 3, prefill=tick == 1)
            c._select_group(group, vals, toks, tick, preserve_prefill_order=tick == 1)
            if tick == 2:
                final_nodes = group.catalog_nodes.clone()
        # The overshoot never moves trie nodes onto terminal leaves.
        torch.testing.assert_close(final_nodes, group.catalog_nodes)
        self.assertFalse(group.commit_pending(1))
        self.assertEqual(group.num_committed, 1)
        self.assertTrue(group.commit_pending(2))
        self.assertEqual(group.num_committed, 2)
        self.assertEqual(group._pending_steps, [])
        self.assertTrue(
            all(cat.catalog.resolve(r.tokens)["is_complete"] for r in group.finalize())
        )

    def test_output_complete_and_partial_ipc(self):
        from sglang.srt.beam_search.beam_group import BeamResult

        cat = DeviceItemCatalog(ItemCatalog.from_records(records(), 8), "cpu")
        for tokens, complete in (([0, 1, 1], True), ([0], False)):
            group = SimpleNamespace(
                final_results=[BeamResult(tokens, -2.0, -2.0, None)],
                num_return=1,
                catalog=cat,
                catalog_id="items",
            )
            packed = pack_beam_search_output(SimpleNamespace(beam_group=group))
            packed = msgspec.msgpack.decode(
                msgspec.msgpack.encode(packed), type=BeamSearchOutput
            )
            out = _build_beam_search_out_dict(packed, {}, SimpleNamespace())
            meta = out["beam_results"][0]["meta_info"]
            self.assertEqual(meta["is_complete"], complete)
            self.assertEqual(meta["item_ids"], [0] if complete else [])
            self.assertEqual(meta["sequence_score"], -2.0)

    def test_nonfinite_selection_aborts_at_deferred_commit(self):
        group = BeamGroup(beam_width=2, num_candidates=2, max_new_tokens=1)
        group.catalog = DeviceItemCatalog(ItemCatalog.from_records(records(), 8), "cpu")
        group.catalog_nodes = group.catalog.initial_nodes()
        group.leader = SimpleNamespace(to_finish=None)
        vals, toks = group.catalog.topk(
            torch.full((1, 8), float("nan")), group.catalog_nodes, 2, prefill=True
        )
        coordinator()._select_group(group, vals, toks, 1, preserve_prefill_order=True)
        self.assertTrue(group.commit_pending(1))
        self.assertIn("non-finite", group.leader.to_finish.message)
        self.assertEqual(group.completed, [])


if __name__ == "__main__":
    unittest.main()
