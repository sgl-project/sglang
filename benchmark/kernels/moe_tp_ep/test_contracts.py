"""CPU interface and experiment-comparability regressions, without NPU kernels."""

import ast
import copy
import json
import sys
import tempfile
import unittest
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from aiohttp import web
from bench_moe_tp_ep import (
    configure_tp_reduction,
    initialize_parallel,
    make_forward_batch,
)
from environment import hardware_identity, version_record
from summarize import summarize as summarize_module
from summarize_serving import summarize as summarize_serving

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EVIDENCE = HERE / "results" / "a3-2026-09-16"


def hardware_fixture(count=2):
    devices = [
        dict(logical_id=i, model="Ascend910_fixture", uuid=f"chip-{i}")
        for i in range(count)
    ]
    return dict(
        machine="aarch64",
        device_nodes=[f"davinci{i}" for i in range(count)],
        visible_devices={},
        cann=dict(version="9.1.0", sha256="cann-fixture"),
        driver=dict(version="25.5.0", sha256="driver-fixture"),
        devices=devices,
        rank_mapping=[
            dict(rank=i, local_rank=i, device=device)
            for i, device in enumerate(devices)
        ],
    )


def current_record(original):
    run = copy.deepcopy(original)
    run["schema_version"] = 3
    run["metadata"]["hardware"] = hardware_fixture()
    run["deepep_mode"] = (
        dict(requested=None, resolved=None)
        if run["backend"] == "none"
        else dict(
            requested="auto",
            resolved="normal" if run["phase"] == "prefill" else "low_latency",
        )
    )
    return run


def source_function(relative_path, name, namespace, body=None):
    """Use checkout signatures/bodies without importing device-only modules."""
    tree = ast.parse((ROOT / relative_path).read_text())
    node = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    node.decorator_list = []
    if body is not None:
        node.body = ast.parse(body).body
    module = ast.parse("from __future__ import annotations")
    module.body.append(node)
    exec(compile(ast.fix_missing_locations(module), relative_path, "exec"), namespace)
    return namespace[name]


class RuntimeContractTest(unittest.TestCase):
    def test_measurement_persists_requested_and_active_dispatcher_modes(self):
        import bench_moe_tp_ep as bench

        # Use the current runtime resolver, without importing device kernels.
        namespace = {}
        resolve = source_function(
            "python/sglang/srt/layers/moe/utils.py", "resolve", namespace
        )

        class DeepEPMode(Enum):
            NORMAL = "normal"
            LOW_LATENCY = "low_latency"
            AUTO = "auto"

        DeepEPMode.resolve = resolve
        namespace["DeepEPMode"] = DeepEPMode
        execute = source_function(
            "python/sglang/srt/batch_overlap/two_batch_overlap.py", "_execute", {}
        )
        wrapper_type = type("CPUDispatcherWrapper", (), {"_execute": execute})
        normal, low_latency = object(), object()
        records = []
        with tempfile.TemporaryDirectory() as temp:
            for phase in ("prefill", "decode"):
                for requested in ("auto", "normal", "low_latency"):
                    configured = DeepEPMode(requested)
                    dispatcher = SimpleNamespace(
                        deepep_mode=configured,
                        _normal_dispatcher=normal,
                        _low_latency_dispatcher=low_latency,
                        _get_impl=lambda: (
                            normal
                            if configured.resolve(phase == "prefill")
                            == DeepEPMode.NORMAL
                            else low_latency
                        ),
                    )
                    block = SimpleNamespace(
                        experts=SimpleNamespace(dispatcher=dispatcher),
                        benchmark_weight_sha256="weights",
                    )
                    expected = dict(
                        requested=requested,
                        resolved=configured.resolve(phase == "prefill").value,
                    )
                    self.assertEqual(
                        bench.deepep_mode_record(block, "deepep", requested), expected
                    )
                    wrapper = wrapper_type()
                    wrapper._inners = [dispatcher]
                    self.assertIs(wrapper._execute("_get_impl"), dispatcher._get_impl())
                    block.experts.dispatcher = wrapper
                    target = Path(temp) / f"{phase}_{requested}.json"
                    with (
                        patch.object(
                            sys,
                            "argv",
                            [
                                "bench",
                                "--model-path",
                                "fixture",
                                "--backend",
                                "deepep",
                                "--ep-size",
                                "2",
                                "--deepep-mode",
                                requested,
                                "--phase",
                                phase,
                                "--global-tokens",
                                "4",
                                "--iters",
                                "1",
                                "--out",
                                str(target),
                            ],
                        ),
                        patch.object(
                            bench,
                            "build_block",
                            return_value=(
                                block,
                                SimpleNamespace(
                                    num_experts=4, num_experts_per_tok=2, hidden_size=4
                                ),
                                SimpleNamespace(tp_size=2, ep_size=2),
                            ),
                        ),
                        patch.object(bench, "load_real_weights"),
                        patch.object(bench, "finalize_weights"),
                        patch.object(
                            bench,
                            "environment_metadata",
                            return_value={"hardware": hardware_fixture()},
                        ),
                        patch.object(
                            bench,
                            "rank_mapping",
                            return_value=hardware_fixture()["rank_mapping"],
                        ),
                        patch.object(
                            bench,
                            "make_input",
                            return_value=(torch.zeros(2, 4), [2, 2], "input"),
                        ),
                        patch.object(bench, "make_forward_batch"),
                        patch.object(
                            bench,
                            "time_forward",
                            return_value=([[1.0, 2.0]], [[[1.0, 2.0]], [[1.0, 2.0]]]),
                        ),
                        patch.object(bench.torch, "device", return_value="cpu"),
                        patch.object(bench.dist, "get_rank", return_value=0),
                        patch.object(bench.dist, "get_world_size", return_value=2),
                        patch.object(
                            bench.dist,
                            "all_gather_object",
                            side_effect=lambda outputs, obj: outputs.__setitem__(
                                slice(None), [obj, obj]
                            ),
                        ),
                        patch.object(bench.dist, "destroy_process_group"),
                        patch("builtins.print"),
                    ):
                        bench.main()
                    record = json.loads(target.read_text())
                    self.assertEqual(record["schema_version"], 3)
                    self.assertEqual(
                        record["deepep_mode"],
                        dict(
                            requested=requested,
                            resolved=configured.resolve(phase == "prefill").value,
                        ),
                    )
                    records.append(record)
        self.assertNotEqual(records[3]["deepep_mode"], records[4]["deepep_mode"])

    def test_tp_reduction_for_stage_based_and_historical_layers(self):
        tree = ast.parse((ROOT / "python/sglang/srt/models/qwen3_moe.py").read_text())
        decoder = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "Qwen3MoeDecoderLayer"
        )
        self.assertTrue(
            any(
                isinstance(node, ast.FunctionDef) and node.name == "stage_facts"
                for node in decoder.body
            )
        )
        for backend, layer, expected in (
            ("none", SimpleNamespace(stage_facts=True), True),
            ("deepep", SimpleNamespace(stage_facts=True), False),
            ("none", SimpleNamespace(), False),
        ):
            with self.subTest(backend=backend, layer=layer):
                block = SimpleNamespace(experts=SimpleNamespace(reduce_results=False))
                configure_tp_reduction(block, layer, backend)
                self.assertEqual(block.experts.reduce_results, expected)

    def test_parallel_publication_and_current_signature(self):
        calls = []
        initialize = source_function(
            "python/sglang/srt/distributed/parallel_state.py",
            "initialize_model_parallel",
            {"calls": calls},
            "calls.append(('groups', backend))",
        )
        self._check_parallel(initialize, calls, legacy=False)

    def test_legacy_parallel_interface(self):
        calls = []

        def initialize(tensor_model_parallel_size, expert_model_parallel_size):
            calls.append(
                ("groups", tensor_model_parallel_size, expert_model_parallel_size)
            )

        self._check_parallel(initialize, calls, legacy=True)

    def _check_parallel(self, initialize, calls, legacy):
        for backend, ep_size in (("none", 1), ("deepep", 2)):
            with self.subTest(backend=backend):
                calls.clear()
                args = SimpleNamespace(
                    tp_size=2, ep_size=ep_size, moe_a2a_backend=backend
                )

                def publish(server_args, *, role, ranks):
                    self.assertIs(server_args, args)
                    self.assertEqual(role, "scheduler")
                    self.assertEqual((ranks.world_rank, ranks.gpu_id), (1, 1))
                    calls.append(("publish",))

                def publish_legacy(server_args):
                    self.assertIs(server_args, args)
                    calls.append(("publish",))

                def moe_legacy(server_args):
                    self.assertIs(server_args, args)
                    calls.append(("moe",))

                def moe_current():
                    calls.append(("moe",))

                world = Mock(side_effect=lambda **kwargs: calls.append(("world",)))
                modules = {
                    "sglang.srt.distributed.parallel_state": SimpleNamespace(
                        init_distributed_environment=world,
                        initialize_model_parallel=initialize,
                    ),
                    "sglang.srt.layers.moe.utils": SimpleNamespace(
                        initialize_moe_config=moe_legacy if legacy else moe_current,
                    ),
                    "sglang.srt.runtime_context": SimpleNamespace(
                        publish=publish,
                        SpawnRanks=SimpleNamespace,
                    ),
                    "sglang.srt.server_args": SimpleNamespace(
                        set_global_server_args_for_scheduler=publish_legacy,
                    ),
                }
                with patch.dict(sys.modules, modules):
                    initialize_parallel(args, world=2, rank=1, local_rank=1)
                world.assert_called_once_with(
                    world_size=2,
                    rank=1,
                    local_rank=1,
                    backend="hccl",
                    moe_a2a_backend=backend,
                )
                self.assertEqual(
                    [x[0] for x in calls], ["publish", "moe", "world", "groups"]
                )
                self.assertEqual(
                    calls[-1], ("groups", 2, ep_size) if legacy else ("groups", "hccl")
                )

    def test_current_qwen_forward_with_local_valid_rows(self):
        # Execute the real Qwen3 TP and EP forward bodies; only router and expert
        # kernels are CPU stand-ins. This tests the batch consumer, not NPU math.
        namespace = {
            "torch": torch,
            "ExpertLocationDispatchInfo": SimpleNamespace(
                init_new=lambda **kwargs: None
            ),
            "get_moe_a2a_backend": lambda: SimpleNamespace(
                is_ascend_fuseep=lambda: False
            ),
        }
        methods = {
            name: source_function(
                "python/sglang/srt/models/qwen3_moe.py", name, namespace
            )
            for name in ("forward", "forward_normal", "forward_deepep")
        }
        block_type = type("CPUQwenForward", (), methods)
        for backend in ("none", "deepep"):
            namespace["is_deepep_class_backend"] = lambda: backend == "deepep"
            for is_extend in (True, False):
                for local_rows in (0, 1, 2, 3, 5):
                    with self.subTest(
                        backend=backend, extend=is_extend, rows=local_rows
                    ):
                        phase = Mock()
                        modes = SimpleNamespace(EXTEND="extend", DECODE="decode")
                        tensor = torch.tensor
                        with (
                            patch.dict(
                                sys.modules,
                                {
                                    "sglang.srt.layers.dp_attention": SimpleNamespace(
                                        set_is_extend_in_batch=phase
                                    ),
                                    "sglang.srt.model_executor.forward_batch_info": SimpleNamespace(
                                        ForwardMode=modes
                                    ),
                                },
                            ),
                            patch(
                                "bench_moe_tp_ep.torch.tensor",
                                side_effect=lambda *a, **kw: tensor(
                                    *a, **{**kw, "device": "cpu"}
                                ),
                            ),
                        ):
                            batch = make_forward_batch(local_rows, is_extend)
                        phase.assert_called_once_with(is_extend)
                        self.assertEqual(
                            batch.forward_mode, "extend" if is_extend else "decode"
                        )
                        self.assertEqual(
                            batch.moe_num_token_non_padded().item(), local_rows
                        )
                        self.assertIs(
                            batch.moe_num_token_non_padded(), batch.num_token_non_padded
                        )
                        topk = Mock(return_value="routed")
                        topk.empty_topk_output.return_value = "empty"
                        block = block_type()
                        block._use_mega_moe = False
                        block.layer_id = 0
                        block.gate = Mock(
                            return_value=(torch.ones(local_rows, 4), None)
                        )
                        block.topk = topk
                        block.experts = lambda hidden_states, topk_output: (
                            hidden_states + 1
                        )
                        hidden = torch.zeros(local_rows, 4)
                        torch.testing.assert_close(
                            block.forward(hidden, batch), hidden + 1
                        )
                        if local_rows == 0:
                            block.gate.assert_not_called()
                            topk.empty_topk_output.assert_called_once_with(
                                hidden.device
                            )
                        else:
                            # Current Qwen3 binds routing metadata on TopK and
                            # lets it resolve optional masks from forward context.
                            # This adapter supplies only valid, unpadded rows.
                            topk.assert_called_once()
                            self.assertEqual(topk.call_args.kwargs, {})
                            torch.testing.assert_close(topk.call_args.args[0], hidden)
                            torch.testing.assert_close(
                                topk.call_args.args[1], block.gate.return_value[0]
                            )


class SummaryContractTest(unittest.TestCase):
    def test_historical_summaries_are_unchanged(self):
        self.assertEqual(
            summarize_module(EVIDENCE / "formal", historical_provenance=EVIDENCE),
            json.loads((EVIDENCE / "module-summary.json").read_text()),
        )
        self.assertEqual(
            summarize_serving(EVIDENCE / "e2e", historical_provenance=EVIDENCE),
            json.loads((EVIDENCE / "serving-summary.json").read_text()),
        )

    def test_module_rejects_incomparable_runs(self):
        originals = {
            p.name: current_record(json.loads(p.read_text()))
            for p in (EVIDENCE / "formal").glob("*.json")
        }

        def four_devices(run):
            run["tp_size"] = run["ep_size"] = 4
            run["metadata"]["hardware"] = hardware_fixture(4)
            for point in run["results"]:
                point["samples_per_rank_ms"] *= 2
                n = point["global_tokens"]
                point["tokens_per_rank"] = [n // 4 + (r < n % 4) for r in range(4)]

        changes = {
            "packages": lambda r: r["metadata"]["packages"].update(
                {"deep-ep": "other-version"}
            ),
            "tp_size": four_devices,
            "source_diff": lambda r: r["metadata"].update(
                sglang_diff_sha256="other-diff"
            ),
            "scripts": lambda r: r["metadata"]["scripts_sha256"].update(
                {"workload.py": "other-helper"}
            ),
            "EP=": lambda r: r.update(ep_size=1),
            "mismatched hardware": lambda r: r["metadata"]["hardware"].update(
                machine="different-machine"
            ),
            "hardware": lambda r: r["metadata"]["hardware"]["cann"].update(
                version="other-cann"
            ),
            "missing hardware": lambda r: r["metadata"].pop("hardware"),
            "mapping": lambda r: r["metadata"]["hardware"]["rank_mapping"].reverse(),
            "deepep_mode": lambda r: (
                r["deepep_mode"].update(requested=r["deepep_mode"]["resolved"])
                if r["repeat_id"] == 1
                else None
            ),
        }
        for message, change in changes.items():
            with self.subTest(message=message), tempfile.TemporaryDirectory() as temp:
                for name, original in originals.items():
                    run = copy.deepcopy(original)
                    if name.startswith("deepep"):
                        change(run)
                    (Path(temp) / name).write_text(json.dumps(run))
                with self.assertRaisesRegex(ValueError, message):
                    summarize_module(Path(temp))

    def test_legacy_and_unknown_hardware_need_explicit_evidence(self):
        with self.assertRaisesRegex(ValueError, "historical provenance"):
            summarize_module(EVIDENCE / "formal")
        for key in ("cann", "driver", "devices"):
            hardware = hardware_fixture()
            hardware[key] = None
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "hardware"):
                hardware_identity(dict(hardware=hardware))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "version.info"
            path.write_text("Version=25.5.0\n")
            self.assertEqual(version_record([path])["version"], "25.5.0")

    def test_serving_requires_and_checks_server_identity(self):
        metadata = json.loads(next((EVIDENCE / "formal").glob("*.json")).read_text())[
            "metadata"
        ]
        metadata["hardware"] = hardware_fixture()
        originals = {}
        for backend in ("none", "deepep"):
            name = f"{backend}_r1_in1024.json"
            run = json.loads((EVIDENCE / "e2e" / name).read_text())
            command = json.loads(
                (
                    EVIDENCE / "server-commands" / f"{backend}_r1_command.json"
                ).read_text()
            )
            run["server"] = dict(command=command, metadata=copy.deepcopy(metadata))
            originals[name] = run

        def change_option(run, option, value):
            command = run["server"]["command"]
            command[command.index(option) + 1] = value

        changes = [
            (None, lambda r: None),
            ("missing per-run", lambda r: r.pop("server")),
            (
                "server_identity",
                lambda r: r["server"]["metadata"]["packages"].update(
                    {"deep-ep": "other-version"}
                ),
            ),
            (
                "server_identity",
                lambda r: r["server"]["metadata"].update(sglang_commit="other-source"),
            ),
            (
                "server_identity",
                lambda r: r["server"]["metadata"]["hardware"]["driver"].update(
                    version="other-driver"
                ),
            ),
            (
                "server_identity",
                lambda r: change_option(r, "--chunked-prefill-size", "4096"),
            ),
            (
                "server_identity",
                lambda r: (
                    change_option(r, "--tp-size", "4"),
                    change_option(r, "--ep-size", "4"),
                ),
            ),
        ]
        for message, change in changes:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as temp:
                for name, original in originals.items():
                    run = copy.deepcopy(original)
                    if name.startswith("deepep"):
                        change(run)
                    (Path(temp) / name).write_text(json.dumps(run))
                if message is None:
                    self.assertEqual(len(summarize_serving(Path(temp))), 1)
                else:
                    with self.assertRaisesRegex(ValueError, message):
                        summarize_serving(Path(temp))


class ServingCaptureTest(unittest.IsolatedAsyncioTestCase):
    def test_launch_records_identity_before_start(self):
        from launch_serving import main

        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "server.json"
            launch_args = ["--model-path", "fixture", "--tp-size", "2"]

            def launch(*args, **kwargs):
                record = json.loads(target.read_text())
                self.assertEqual(record["command"][3:], launch_args)
                self.assertEqual(sys.argv[1:], launch_args)
                self.assertEqual(record["metadata"]["sglang_commit"], "fixture-source")

            with (
                patch(
                    "sys.argv",
                    [
                        "launch_serving.py",
                        "--metadata",
                        str(target),
                        "--",
                        *launch_args,
                    ],
                ),
                patch(
                    "launch_serving.environment_metadata",
                    return_value={"sglang_commit": "fixture-source"},
                ),
                patch("launch_serving.runpy.run_module", side_effect=launch) as start,
            ):
                main()
                start.assert_called_once_with(
                    "sglang.launch_server", run_name="__main__", alter_sys=True
                )

    async def test_client_embeds_provenance_and_rejects_changes(self):
        from bench_serving import run

        with tempfile.TemporaryDirectory() as temp:
            provenance = Path(temp) / "server.json"
            saved = {"command": ["fixture"], "metadata": {"source": "fixture"}}
            provenance.write_text(json.dumps(saved))
            change = False

            async def generate(request):
                payload = await request.json()
                if change:
                    provenance.write_text("{}")
                count = payload["sampling_params"]["max_new_tokens"]
                event = json.dumps({"meta_info": {"completion_tokens": count}})
                return web.Response(
                    text=f"data: {event}\n\ndata: [DONE]\n\n",
                    content_type="text/event-stream",
                )

            app = web.Application()
            app.router.add_post("/generate", generate)
            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, "127.0.0.1", 0)
            await site.start()
            try:
                args = SimpleNamespace(
                    port=runner.addresses[0][1],
                    requests=2,
                    concurrency=1,
                    input_len=4,
                    output_len=2,
                    seed=17,
                    server_metadata=provenance,
                    out=str(Path(temp) / "result.json"),
                )
                with patch("builtins.print"):
                    await run(args)
                report = json.loads(Path(args.out).read_text())
                self.assertEqual(report["server"], saved)
                self.assertEqual(report["output_tokens"], 4)
                self.assertEqual(len(report["per_request"]), 2)
                change = True
                args.out = str(Path(temp) / "invalid.json")
                with self.assertRaisesRegex(ValueError, "metadata changed"):
                    await run(args)
                self.assertFalse(Path(args.out).exists())
            finally:
                await runner.cleanup()


if __name__ == "__main__":
    unittest.main()
