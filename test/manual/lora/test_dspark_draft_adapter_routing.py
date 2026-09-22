"""CPU contracts and real request normalization, without CUDA-serving imports.

GenerateReqInput is loaded from its source AST because its module imports the
CUDA scheduler. The class/method bodies are unchanged; this is not a substitute
for the HTTP/IPC integration test on a GPU server.
"""

import ast
import asyncio
import dataclasses
import importlib.util
import sys
import types
import unittest
import uuid
from collections import Counter
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
ROUTING_NAME = "sglang.srt.speculative.dspark_components.dspark_lora_routing"
spec = importlib.util.spec_from_file_location(
    ROUTING_NAME,
    ROOT / "python/sglang/srt/speculative/dspark_components/dspark_lora_routing.py",
)
routing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(routing)


def load_request_class():
    path = ROOT / "python/sglang/srt/managers/io_struct.py"
    tree = ast.parse(path.read_text())
    definition = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "GenerateReqInput"
    )
    module = types.ModuleType("dspark_request_test")
    module.__dict__.update(
        dataclass=dataclasses.dataclass,
        field=dataclasses.field,
        uuid=uuid,
        Counter=Counter,
        get_return_hidden_states_mode=lambda v: v,
    )
    sys.modules[module.__name__] = module
    exec(
        compile(
            "from __future__ import annotations\n" + ast.unparse(definition),
            str(path),
            "exec",
        ),
        module.__dict__,
    )
    return module.GenerateReqInput


GenerateReqInput = load_request_class()


def config():
    return NS(
        speculative_dspark_lora_paths='{"rust":"/adapters/rust"}',
        speculative_algorithm="DSPARK",
        speculative_dspark_lora_path=None,
        speculative_draft_load_format=None,
        load_format="safetensors",
        tp_size=1,
        dp_size=1,
        pp_size=1,
        attn_cp_size=1,
        disable_overlap_schedule=True,
        disable_radix_cache=True,
        schedule_policy="fcfs",
        enable_priority_scheduling=False,
        enable_mixed_chunk=False,
        enable_dp_attention=False,
        enable_hierarchical_cache=False,
        enable_hisparse=False,
        enable_flexkv=False,
        enable_unified_cache_external_linker=False,
        enable_unified_memory=False,
        enable_torch_compile=False,
        enable_memory_saver=False,
        checkpoint_engine_wait_weights_before_ready=False,
        cpu_offload_gb=0,
        disaggregation_mode="null",
        device="cuda",
        cuda_graph_config=NS(
            prefill=NS(backend="disabled"), decode=NS(backend="disabled")
        ),
    )


class TestDraftAdapterRouting(unittest.TestCase):
    def test_registry_rejects_bad_and_duplicate_names(self):
        for raw in ("{}", "[]", '{"x":"a","x":"b"}', '{"../x":"a"}', '{"x":3}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                routing.parse_draft_adapters(raw)
        registry = routing.parse_draft_adapters('{"rust":"a","python":"b"}')
        self.assertEqual(dict(registry), {"rust": "a", "python": "b"})
        with self.assertRaises(TypeError):
            registry["rust"] = "other"

    def test_server_supported_and_unsafe_configurations(self):
        routing.validate_draft_adapter_server_config(config())
        for key, value in dict(
            tp_size=2,
            dp_size=2,
            pp_size=2,
            attn_cp_size=2,
            disable_overlap_schedule=False,
            disable_radix_cache=False,
            schedule_policy="lpm",
            enable_priority_scheduling=True,
            enable_mixed_chunk=True,
            enable_dp_attention=True,
            enable_hierarchical_cache=True,
            enable_torch_compile=True,
            enable_memory_saver=True,
            cpu_offload_gb=1,
            disaggregation_mode="decode",
            device="cpu",
            speculative_algorithm="EAGLE3",
            speculative_dspark_lora_path="path",
            load_format="dummy",
        ).items():
            c = config()
            setattr(c, key, value)
            with self.subTest(key=key), self.assertRaises(ValueError):
                routing.validate_draft_adapter_server_config(c)
        for phase in ("prefill", "decode"):
            c = config()
            getattr(c.cuda_graph_config, phase).backend = "full"
            with self.subTest(phase=phase), self.assertRaises(ValueError):
                routing.validate_draft_adapter_server_config(c)

    def normalize(self, request):
        with patch.dict(sys.modules, {ROUTING_NAME: routing}):
            request.normalize_batch_and_arguments()
        return request

    def test_real_single_and_parallel_request_normalization(self):
        single = self.normalize(GenerateReqInput(text="a", draft_adapter="rust"))
        self.assertEqual(single.draft_adapter, "rust")
        parallel = self.normalize(
            GenerateReqInput(text="a", draft_adapter="rust", sampling_params={"n": 3})
        )
        self.assertEqual([parallel[i].draft_adapter for i in range(3)], ["rust"] * 3)
        self.assertEqual([parallel[i].text for i in range(3)], ["a"] * 3)

    def test_real_batch_selector_expansion_and_target_namespace(self):
        request = self.normalize(
            GenerateReqInput(
                text=["a", "b"],
                draft_adapter=["rust", None],
                lora_path=["target1", "target2"],
                sampling_params={"n": 2},
            )
        )
        self.assertEqual(
            [request[i].draft_adapter for i in range(4)], ["rust", None, "rust", None]
        )
        self.assertEqual([request[i].text for i in range(4)], ["a", "b", "a", "b"])
        self.assertEqual(
            [request[i].lora_path for i in range(4)], ["target1", "target2"] * 2
        )
        broadcast = self.normalize(
            GenerateReqInput(text=["a", "b"], draft_adapter="rust")
        )
        self.assertEqual(
            [broadcast[i].draft_adapter for i in range(2)], ["rust", "rust"]
        )
        base = self.normalize(GenerateReqInput(text=["a", "b"]))
        self.assertEqual([base[i].draft_adapter for i in range(2)], [None, None])

    def test_bad_shapes_and_unknown_names_rejected_before_queueing(self):
        for text, value in [
            ("a", ["rust"]),
            (["a", "b"], ["rust"]),
            ("a", ""),
            ("a", 2),
            (["a"], [{}]),
        ]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.normalize(GenerateReqInput(text=text, draft_adapter=value))
        request = self.normalize(
            GenerateReqInput(text="a", draft_adapter="target-only-name")
        )
        with self.assertRaisesRegex(ValueError, "Unknown draft_adapter"):
            routing.validate_draft_adapter_request(request, {"rust": "/a"})
        with self.assertRaisesRegex(ValueError, "Unknown draft_adapter"):
            routing.validate_draft_adapter_request(request, {})

    def test_sessions_and_beams_rejected_even_for_base_in_bank_deployment(self):
        for kwargs in (
            {"session_id": "x"},
            {"session_params": {"id": "x"}},
            {"sampling_params": {"beam_width": 2}},
        ):
            request = self.normalize(GenerateReqInput(text="a", **kwargs))
            with self.assertRaises(ValueError):
                routing.validate_draft_adapter_request(request, {"rust": "/a"})
        routing.validate_draft_adapter_request(
            GenerateReqInput(text="a", session_id="x"), {}
        )

    def test_cohorts_preserve_fifo_and_do_not_starve_foreign_adapter(self):
        waiting = ["rust", "rust", "python", "rust", None, "python"]
        cohorts = []
        while waiting:
            gate = routing.DraftAdapterCohort([])
            batch = []
            for name in waiting:
                if not gate.admit(name):
                    break
                batch.append(name)
            cohorts.append(batch)
            waiting = waiting[len(batch) :]
        self.assertEqual(
            cohorts, [["rust", "rust"], ["python"], ["rust"], [None], ["python"]]
        )
        # Active Rust cannot admit new Rust arrivals past an older Python job.
        gate = routing.DraftAdapterCohort(["rust", "rust"])
        self.assertFalse(gate.admit("python"))

    def test_chunk_filter_retraction_and_base_are_not_wildcards(self):
        gate = routing.DraftAdapterCohort([None])
        self.assertFalse(gate.admit("rust"))
        gate = routing.DraftAdapterCohort(
            ["rust"]
        )  # remaining chunk or filtered survivor
        self.assertTrue(gate.admit("rust"))
        self.assertFalse(gate.admit(None))
        # Retracted Rust is selected again after the intervening Python cohort drains.
        gate = routing.DraftAdapterCohort([])
        self.assertTrue(gate.admit("rust"))
        with self.assertRaises(ValueError):
            routing.DraftAdapterCohort(["rust", "python"])
        with self.assertRaises(ValueError):
            routing.homogeneous_draft_adapter(["rust", None])


class TestDraftAdapterWorkerDispatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = (
            ROOT / "python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py"
        )
        tree = ast.parse(path.read_text())
        worker = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "DSparkWorkerV2"
        )
        method = next(
            n
            for n in worker.body
            if isinstance(n, ast.FunctionDef) and n.name == "forward_batch_generation"
        )
        namespace = {}
        exec(
            compile(
                "from __future__ import annotations\n" + ast.unparse(method),
                str(path),
                "exec",
            ),
            namespace,
        )
        cls.forward = staticmethod(namespace["forward_batch_generation"])

    def test_activation_precedes_both_target_prefill_projection_and_decode(self):
        for extend in (True, False):
            events = []
            worker = NS(
                _draft_adapter_bank=NS(
                    activate=lambda name: events.append(("activate", name))
                ),
                _verify_planner=NS(note_non_decode_step=lambda: None),
                _observers=NS(note_prefill_step=lambda: None),
                _forward_prefill=lambda *args: events.append(("prefill", None)),
                _forward_decode=lambda *args: events.append(("decode", None)),
            )
            batch = NS(
                reqs=[NS(draft_adapter="rust")],
                forward_mode=NS(is_extend=lambda: extend),
                is_extend_in_batch=False,
            )
            with patch.dict(sys.modules, {ROUTING_NAME: routing}):
                self.forward(worker, batch)
            self.assertEqual(
                events,
                [("activate", "rust"), ("prefill" if extend else "decode", None)],
            )

    def test_mixed_batch_fails_before_any_model_work(self):
        events = []
        worker = NS(_draft_adapter_bank=NS(activate=lambda name: events.append(name)))
        batch = NS(reqs=[NS(draft_adapter=None), NS(draft_adapter="rust")])
        with (
            patch.dict(sys.modules, {ROUTING_NAME: routing}),
            self.assertRaises(ValueError),
        ):
            self.forward(worker, batch)
        self.assertEqual(events, [])

    def test_feature_disabled_does_not_require_request_adapter_metadata(self):
        worker = NS(
            _draft_adapter_bank=None, _forward_decode=lambda *args: "ordinary-decode"
        )
        batch = NS(forward_mode=NS(is_extend=lambda: False), is_extend_in_batch=False)
        self.assertEqual(self.forward(worker, batch), "ordinary-decode")


class TestDraftAdapterEngineDispatch(unittest.TestCase):
    def test_sync_and_async_engine_preserve_independent_selectors(self):
        path = ROOT / "python/sglang/srt/entrypoints/engine.py"
        tree = ast.parse(path.read_text())
        engine = next(
            n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Engine"
        )
        methods = [
            n
            for n in engine.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            and n.name in ("generate", "async_generate")
        ]
        namespace = {"GenerateReqInput": GenerateReqInput}
        for method in methods:
            exec(
                compile(
                    "from __future__ import annotations\n" + ast.unparse(method),
                    str(path),
                    "exec",
                ),
                namespace,
            )

        async def receive(obj, request):
            yield (obj.draft_adapter, obj.lora_path)

        loop = asyncio.new_event_loop()

        def close_loop():
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()

        self.addCleanup(close_loop)
        worker = NS(
            loop=loop,
            _resolve_routed_dp_rank=lambda *args: None,
            tokenizer_manager=NS(generate_request=receive),
        )
        expected = (["rust", None], ["target-a", "target-b"])
        kwargs = dict(
            prompt=["a", "b"], draft_adapter=expected[0], lora_path=expected[1]
        )
        self.assertEqual(namespace["generate"](worker, **kwargs), expected)
        self.assertEqual(
            loop.run_until_complete(namespace["async_generate"](worker, **kwargs)),
            expected,
        )


if __name__ == "__main__":
    unittest.main()
