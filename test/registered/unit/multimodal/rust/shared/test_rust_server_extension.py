"""Model packages use one extension for server arguments and worker startup."""

import asyncio
import json
import os
import unittest
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch, sentinel

import msgspec
import zmq

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.arg_groups.overrides import declare_resolution  # noqa: E402
from sglang.srt.arg_groups.serving_hook import handle_ssl_validation  # noqa: E402
from sglang.srt.distributed.parallel_state_wrapper import ParallelState  # noqa: E402
from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.managers import load_snapshot  # noqa: E402
from sglang.srt.managers.io_struct import (  # noqa: E402
    AbortReq,
    ClearHiCacheReqInput,
    FlushCacheReqInput,
    FlushCacheReqOutput,
    GenerateReqInput,
    RustFrontendReadyReqInput,
    RustFrontendReadyReqOutput,
    sock_recv,
    sock_send,
)
from sglang.srt.managers.scheduler import Scheduler  # noqa: E402
from sglang.srt.managers.scheduler_components.output_sender import (  # noqa: E402
    SenderWrapper,
)
from sglang.srt.managers.scheduler_components.request_receiver import (  # noqa: E402
    SchedulerRequestReceiver,
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager  # noqa: E402
from sglang.srt.runtime_context import (  # noqa: E402
    get_context,
    publish,
    reset_context,
)
from sglang.srt.rust_server import server as server_module  # noqa: E402
from sglang.srt.rust_server.config import _build_server_args  # noqa: E402
from sglang.srt.rust_server.control import RustControlTransport  # noqa: E402
from sglang.srt.rust_server.readiness import publish_frontend_ready  # noqa: E402
from sglang.srt.rust_server.server import RustServer  # noqa: E402
from sglang.srt.rust_server.topology import (  # noqa: E402
    FrontendTopology,
    collect_worker_infos,
)
from sglang.srt.sampling.sampling_params import SamplingParams  # noqa: E402
from sglang.srt.server_args import ServerArgs  # noqa: E402
from sglang.srt.utils.network import get_zmq_socket_on_host  # noqa: E402
from sglang.srt.utils.weight_versions import WeightVersionSpan  # noqa: E402

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestRustServerExtension(CustomTestCase):
    def test_sampling_mask_request_fixture_matches_python_normalization(self):
        fixtures = json.loads(
            (
                Path(__file__).resolve().parents[6]
                / "rust/sglang-server/testdata/sampling_masks_python.json"
            ).read_text()
        )["requests"]
        for fixture in fixtures:
            with self.subTest(body=fixture["body"]):
                request = GenerateReqInput(**fixture["body"])
                request.normalize_batch_and_arguments()
                prompts = (
                    [request]
                    if request.is_single
                    else [request[i] for i in range(request.batch_size)]
                )
                self.assertEqual(
                    [
                        prompt.return_sampling_mask
                        for prompt in prompts
                        for _ in range(request.parallel_sample_num)
                    ],
                    fixture["masks"],
                )
        with self.assertRaisesRegex(ValueError, "Cannot use list return_sampling_mask"):
            GenerateReqInput(
                input_ids=[[1], [2]],
                return_sampling_mask=[True, False],
                sampling_params={"n": 2},
            ).normalize_batch_and_arguments()

    def test_input_embedding_fixture_uses_the_python_tokenizer_path(self):
        fixtures = json.loads(
            (
                Path(__file__).resolve().parents[6]
                / "rust/sglang-server/testdata/input_embeddings_python.json"
            ).read_text()
        )
        manager = TokenizerManager.__new__(TokenizerManager)
        manager.preferred_sampling_params = None
        manager.sampling_params_class = SamplingParams
        manager.tokenizer = None
        manager.mm_processor = None
        manager.model_config = SimpleNamespace(
            vocab_size=1000, hf_config=SimpleNamespace(architectures=[])
        )
        manager.rid_to_state = {}
        manager.context_len = 4096
        manager.num_reserved_tokens = 0
        manager.allow_auto_truncate = False
        manager.validate_total_tokens = True
        for fixture in fixtures:
            with self.subTest(body=fixture["body"]):
                request = GenerateReqInput(**fixture["body"])
                request.normalize_batch_and_arguments()
                prompts = (
                    [request]
                    if request.is_single
                    else [request[i] for i in range(request.batch_size)]
                )
                actual = []
                for prompt in prompts:
                    manager.rid_to_state[prompt.rid] = SimpleNamespace(
                        time_stats=Mock()
                    )
                    with get_context().override_server_args(disable_radix_cache=False):
                        with self.assertRaisesRegex(
                            ValueError, "--disable-radix-cache"
                        ):
                            asyncio.run(manager._tokenize_one_request(prompt))
                    with get_context().override_server_args(disable_radix_cache=True):
                        tokenized = asyncio.run(manager._tokenize_one_request(prompt))
                    actual.extend(
                        [tokenized.input_embeds] * request.parallel_sample_num
                    )
                self.assertEqual(actual, fixture["expected"])

    def test_generation_policy_fixture_uses_the_python_tokenizer_path(self):
        context = get_context().override_server_args()
        context.install()
        self.addCleanup(context.restore)
        fixtures = json.loads(
            (
                Path(__file__).resolve().parents[6]
                / "rust/sglang-server/testdata/generation_policy_python.json"
            ).read_text()
        )
        manager = SimpleNamespace(
            preferred_sampling_params=None,
            sampling_params_class=SamplingParams,
            tokenizer=None,
            model_config=SimpleNamespace(vocab_size=1000),
            rid_to_state={},
        )
        for fixture in fixtures:
            with self.subTest(body=fixture["body"]):
                request = GenerateReqInput(**fixture["body"])
                request.normalize_batch_and_arguments()
                prompts = (
                    [request]
                    if request.is_single
                    else [request[i] for i in range(request.batch_size)]
                )
                actual = []
                for prompt in prompts:
                    manager.rid_to_state[prompt.rid] = SimpleNamespace(
                        time_stats=Mock()
                    )
                    tokenized = TokenizerManager._create_tokenized_object(
                        manager, prompt, prompt.text, prompt.input_ids
                    )
                    actual.extend(
                        [
                            [
                                tokenized.custom_logit_processor,
                                tokenized.require_reasoning,
                                tokenized.sampling_params.custom_params,
                            ]
                        ]
                        * request.parallel_sample_num
                    )
                self.assertEqual(actual, fixture["expected"])

    def test_parent_readiness_requires_distinct_valid_private_acknowledgements(self):
        cases = [
            ("single", 1, [0], None),
            ("distributed", 4, [3, 3, 2, 1, 0], None),
            ("missing", 2, [0], TimeoutError),
            ("invalid", 2, [0], RuntimeError),
        ]
        for name, dp_size, ranks, error in cases:
            with self.subTest(case=name), TemporaryDirectory() as directory:
                args = SimpleNamespace(dp_size=1, ep_join_mode=None)
                declare_resolution(args, "resolved-test-topology", dp_size=dp_size)
                ports = SimpleNamespace(
                    scheduler_input_ipc_name=f"ipc://{directory}/requests",
                    tokenizer_ipc_name=f"ipc://{directory}/replies",
                )
                context = zmq.Context()
                requests = context.socket(zmq.PULL)
                replies = context.socket(zmq.PUSH)
                if dp_size > 1:
                    requests.bind(ports.scheduler_input_ipc_name)
                else:
                    requests.connect(ports.scheduler_input_ipc_name)
                replies.connect(ports.tokenizer_ipc_name)
                try:
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        ready = executor.submit(
                            publish_frontend_ready, args, ports, timeout_seconds=1
                        )
                        self.assertTrue(requests.poll(1000))
                        request = sock_recv(requests)
                        self.assertIsInstance(request, RustFrontendReadyReqInput)
                        for index, rank in enumerate(ranks):
                            scheduler = SimpleNamespace(
                                rust_server=SimpleNamespace(
                                    server=Mock(), push_control_output=Mock()
                                ),
                                ps=SimpleNamespace(dp_rank=rank),
                                ipc_channels=SimpleNamespace(
                                    send_to_tokenizer=SenderWrapper(replies)
                                ),
                            )
                            output = Scheduler.handle_rust_frontend_ready(
                                scheduler, request
                            )
                            self.assertIsInstance(output, RustFrontendReadyReqOutput)
                            scheduler.rust_server.server.mark_ready.assert_called_once_with()
                            if name == "invalid":
                                output.rid = "wrong"
                            Scheduler._send_control_output(scheduler, output, request)
                            scheduler.rust_server.push_control_output.assert_not_called()
                            if name == "distributed" and index == 1:
                                with self.assertRaises(FutureTimeout):
                                    ready.result(timeout=0.05)
                        if error is None:
                            ready.result(timeout=2)
                        else:
                            with self.assertRaisesRegex(
                                error,
                                r"Invalid Rust readiness|missing from DP ranks \[1\]",
                            ):
                                ready.result(timeout=2)
                finally:
                    requests.close(linger=0)
                    replies.close(linger=0)
                    context.term()

    def test_dp_ingress_uses_resolved_dispatch_policy_and_closes_on_failure(self):
        extension = SimpleNamespace(DpIngress=Mock())
        args = SimpleNamespace(
            dp_size=1,
            host="::",
            port=30010,
            load_balance_method="auto",
            enable_http2=False,
            http2_max_concurrent_streams=200,
            http2_initial_connection_window_size=1048576,
        )
        declare_resolution(
            args,
            "resolved-test-ingress",
            dp_size=2,
            host="127.0.0.1",
            port=30000,
            enable_http2=True,
            http2_max_concurrent_streams=17,
            http2_initial_connection_window_size=2097152,
        )
        workers = [
            {
                "rust_worker_infos": [
                    {"dp_rank": 1, "url": "http://127.0.0.1:30002"},
                    {"dp_rank": 0, "url": "http://127.0.0.1:30001"},
                ]
            }
        ]
        for method in (
            "round_robin",
            "follow_bootstrap_room",
            "total_requests",
            "total_tokens",
        ):
            with (
                self.subTest(method=method),
                patch.object(RustServer, "_load_extension", return_value=extension),
                patch.object(
                    server_module,
                    "get_parallel",
                    return_value=SimpleNamespace(load_balance_method=method),
                ),
                patch.object(
                    server_module,
                    "get_serving",
                    return_value=SimpleNamespace(preferred_sampling_params=None),
                ),
                patch(
                    "sglang.srt.rust_server.metrics.frontend_metrics_config",
                    return_value=None,
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "startup failed"):
                    with RustServer.dp_ingress(args, workers):
                        self.assertEqual(
                            extension.DpIngress.call_args.kwargs["load_balance_method"],
                            method,
                        )
                        self.assertEqual(
                            extension.DpIngress.call_args.kwargs["workers"],
                            [
                                (0, "http://127.0.0.1:30001"),
                                (1, "http://127.0.0.1:30002"),
                            ],
                        )
                        for field, expected in (
                            ("host", "127.0.0.1"),
                            ("port", 30000),
                            ("enable_http2", True),
                            ("http2_max_concurrent_streams", 17),
                            ("http2_initial_connection_window_size", 2097152),
                        ):
                            self.assertEqual(
                                extension.DpIngress.call_args.kwargs[field], expected
                            )
                        self.assertEqual(args.dp_size, 1)
                        self.assertEqual(args.port, 30010)
                        raise RuntimeError("startup failed")
                extension.DpIngress.return_value.close.assert_called_once_with()
                extension.DpIngress.reset_mock()

    def test_dp_controls_broadcast_once_and_return_ranked_replies_to_origin(self):
        context = zmq.Context()
        port, controller = get_zmq_socket_on_host(context, zmq.PULL)
        servers = [Mock(), Mock()]
        transports = [
            RustControlTransport(
                server, f"tcp://127.0.0.1:{port}", rank, cross_node=False
            )
            for rank, server in enumerate(servers)
        ]
        try:
            for origin, request in (
                (rank, request)
                for rank in range(2)
                for request in (
                    FlushCacheReqInput(rid=f"flush-{rank}", timeout_s=0.5),
                    ClearHiCacheReqInput(rid=f"storage-{rank}"),
                )
            ):
                transports[origin].broadcast(request)
                self.assertTrue(controller.poll(2000, zmq.POLLIN))
                received = sock_recv(controller)
                self.assertIsInstance(received, type(request))
                if isinstance(received, FlushCacheReqInput):
                    self.assertEqual(received.timeout_s, 0.5)
                self.assertEqual(received.http_worker_ipc, transports[origin].endpoint)
                payload = msgspec.msgpack.encode({"success": True})
                for transport in transports:
                    transport.send_result(received, payload)
                self.assertTrue(transports[origin].replies.poll(2000, zmq.POLLIN))
                transports[origin].drain_replies()
                self.assertEqual(
                    sorted(
                        call.args
                        for call in servers[
                            origin
                        ].push_control_result_part.call_args_list
                    ),
                    [(received.rid, rank, payload) for rank in range(2)],
                )
                self.assertFalse(controller.poll(0, zmq.POLLIN))
                servers[origin].reset_mock()

            rpc_port, rpc = get_zmq_socket_on_host(context, zmq.PULL)
            rpc_sender = context.socket(zmq.PUSH)
            rpc_sender.connect(f"tcp://127.0.0.1:{rpc_port}")
            try:
                transports[0].broadcast(FlushCacheReqInput(rid="collective"))
                sock_send(rpc_sender, FlushCacheReqInput(rid="rpc"))
                self.assertTrue(controller.poll(2000, zmq.POLLIN))
                self.assertTrue(rpc.poll(2000, zmq.POLLIN))
                ring = Mock()
                ring.drain.return_value = [sentinel.generation]
                receiver = SchedulerRequestReceiver(
                    recv_from_tokenizer=ring,
                    recv_from_controller=controller,
                    recv_from_rpc=rpc,
                    recv_skipper=None,
                    input_blocker=None,
                    mm_receiver=None,
                    ps=ParallelState.trivial(),
                    tp_group=None,
                    tp_cpu_group=None,
                    attn_tp_group=None,
                    attn_tp_cpu_group=None,
                    attn_cp_group=None,
                    attn_cp_cpu_group=None,
                    world_group=None,
                    server_args=None,
                    model_config=None,
                    max_recv_per_poll=1,
                    stream_output=Mock(),
                    get_last_batch=lambda: None,
                )
                with patch.object(envs.SGLANG_RUST_SERVER, "get", return_value=True):
                    received = receiver._pull_raw_reqs()
                self.assertIs(received[0], sentinel.generation)
                self.assertEqual(
                    [req.rid for req in received[1:]], ["collective", "rpc"]
                )
            finally:
                rpc_sender.close(linger=0)
                rpc.close(linger=0)
        finally:
            for transport in transports:
                transport.close()
            controller.close(linger=0)
            context.term()

    def test_topology_uses_replica_rank_and_collects_remote_leaders(self):
        for ps, rank, leader in [
            (ParallelState.trivial(), None, True),
            (ParallelState.trivial(dp_size=2, dp_rank=1), 1, True),
            (ParallelState.trivial(dp_size=4, dp_rank=3, attn_dp_rank=3), 3, True),
            (ParallelState.trivial(pp_rank=1), None, False),
            (ParallelState.trivial(attn_tp_rank=1), None, False),
            (ParallelState.trivial(attn_cp_rank=1), None, False),
        ]:
            topology = FrontendTopology.from_parallel_state(ps)
            self.assertEqual(topology.dp_rank, rank)
            self.assertEqual(topology.is_leader, leader)
        with self.assertRaises(ValueError):
            FrontendTopology.from_parallel_state(
                ParallelState.trivial(dp_size=2, dp_rank=None)
            )

        scheduler = SimpleNamespace(
            server_args=SimpleNamespace(host="127.0.0.1", nnodes=1),
            rust_server=RustServer(
                Mock(), 32768, topology=FrontendTopology(2, 1, True)
            ),
            world_group=SimpleNamespace(cpu_group=sentinel.group),
        )

        declare_resolution(
            scheduler.server_args, "resolved-test-hosts", host="::1", nnodes=2
        )

        def gather(workers, local, group):
            self.assertIs(group, sentinel.group)
            self.assertEqual(
                local["worker"], {"dp_rank": 1, "url": "http://[::1]:32768"}
            )
            self.assertEqual(local["metrics_source"]["source_id"], "first-node")
            second_source = {
                "source_id": "second-node",
                "url": "http://[::2]:32767/source/secret",
                "kind": "python",
            }
            workers[:] = [
                local,
                {"worker": None, "metrics_source": second_source},
                {
                    "worker": {"dp_rank": 0, "url": "http://[::2]:32769"},
                    "metrics_source": second_source,
                },
            ]

        with (
            patch.dict(
                os.environ,
                {
                    "SGLANG_RUST_METRICS_SOURCE": json.dumps(
                        {
                            "source_id": "first-node",
                            "url": "http://[::1]:32766/source/secret",
                            "kind": "python",
                        }
                    )
                },
            ),
            patch("torch.distributed.get_world_size", return_value=3),
            patch("torch.distributed.all_gather_object", side_effect=gather),
            patch(
                "sglang.srt.rust_server.topology.get_local_ip_auto", return_value="::1"
            ),
        ):
            workers = collect_worker_infos(scheduler, metrics_enabled=True)
            self.assertEqual([worker["dp_rank"] for worker in workers], [0, 1])
            for worker in workers:
                self.assertEqual(
                    {source["source_id"] for source in worker["metrics_sources"]},
                    {"first-node", "second-node"},
                )
                self.assertEqual(len(worker["metrics_sources"]), 2)

    def test_scheduler_abort_preserves_reason_and_weight_spans(self):
        publish(
            ServerArgs(model_path="dummy", weight_version="current"), role="scheduler"
        )
        self.addCleanup(reset_context)
        instance = RustServer(Mock(), http_port=0)
        for reason in (
            None,
            {"type": "abort", "message": "queue full", "status_code": 503},
        ):
            output = AbortReq(
                rid="request",
                finished_reason=reason,
                weight_versions=[WeightVersionSpan(version="old", start=0, end=2)],
            )
            self.assertTrue(instance.handle_scheduler_output(output))
            rid, payload = instance.server.push_abort_result.call_args.args
            self.assertEqual(rid, "request")
            decoded = msgspec.msgpack.decode(payload)
            self.assertEqual(
                decoded["finished_reason"],
                reason or {"type": "abort", "message": "Abort in waiting queue"},
            )
            self.assertEqual(decoded["weight_version"], "current")
            self.assertEqual(decoded["weight_versions"], [["old", 0, 2]])
        self.assertFalse(
            instance.handle_scheduler_output(FlushCacheReqOutput(success=True))
        )
        self.assertEqual(instance.server.push_abort_result.call_count, 2)

    def test_load_snapshot_handoff_matches_the_native_fixture(self):
        fixture_path = (
            Path(server_module.__file__).resolve().parents[4]
            / "rust/sglang-server/testdata/load_snapshot_python.json"
        )
        fixture = json.loads(fixture_path.read_text())
        fields = fixture["expected"].copy()
        for field, cls in (
            ("memory", load_snapshot.MemoryMetrics),
            ("speculative", load_snapshot.SpeculativeMetrics),
            ("lora", load_snapshot.LoRAMetrics),
            ("disaggregation", load_snapshot.DisaggregationMetrics),
            ("queues", load_snapshot.QueueMetrics),
        ):
            if fields.get(field) is not None:
                fields[field] = cls(**fields[field])
        snapshot = load_snapshot.LoadSnapshot(**fields)
        instance = RustServer(Mock(), http_port=0)
        instance.publish_load_snapshot(snapshot)
        payload = instance.server.publish_load_snapshot.call_args.args[0]
        self.assertEqual(payload.hex(), fixture["msgpack_hex"])
        self.assertEqual(msgspec.msgpack.decode(payload), fixture["expected"])

    def test_http2_requires_granian_only_for_the_python_frontend(self):
        server_args = ServerArgs(model_path="dummy", enable_http2=True)
        with patch.dict("sys.modules", {"granian": None}):
            with patch.object(envs.SGLANG_RUST_SERVER, "get", return_value=True):
                handle_ssl_validation(server_args)
            with patch.object(envs.SGLANG_RUST_SERVER, "get", return_value=False):
                with self.assertRaisesRegex(ValueError, "requires the 'granian'"):
                    handle_ssl_validation(server_args)

    def test_config_handoff_uses_resolved_namespaces_and_model_capabilities(self):
        server_args = ServerArgs(
            model_path="dummy",
            tokenizer_path="public-tokenizer",
            served_model_name="served-model",
            enable_metrics=True,
            enable_http2=True,
            http2_max_concurrent_streams=17,
            http2_initial_connection_window_size=2 * 1024 * 1024,
            enable_priority_scheduling=True,
            default_priority_value=7,
            enable_cache_report=True,
            return_input_ids=True,
            return_output_ids=True,
            return_hidden_states_mode="last",
            enable_custom_logit_processor=True,
            enable_strict_thinking=True,
            disable_radix_cache=True,
            extra_metric_labels={"cluster": "test"},
            tokenizer_metrics_allowed_custom_labels=["tenant"],
            tokenizer_metrics_custom_labels_header="x-tenant-metrics",
            bucket_inter_token_latency=[0.01, 0.1, float("inf")],
            prompt_tokens_buckets=["custom", "1", "32", "inf"],
        )
        publish(server_args, role="scheduler")
        self.addCleanup(reset_context)
        extension = SimpleNamespace(
            ServerArgs=SimpleNamespace,
            ModelConfig=SimpleNamespace,
            DefaultSamplingParams=SimpleNamespace,
            HiddenStatesMode=SimpleNamespace(Off="off", Last="last", Full="full"),
            DisaggregationMode=SimpleNamespace(
                Null="null", Prefill="prefill", Decode="decode"
            ),
        )
        scheduler = SimpleNamespace(
            server_args=server_args,
            ps=ParallelState.trivial(),
            tokenizer=SimpleNamespace(model_max_length=2**100, vocab_size=257),
            model_config=SimpleNamespace(
                context_len=4096,
                vocab_size=256,
                hidden_size=512,
                is_multimodal=True,
                is_generation=True,
                is_image_understandable_model=True,
                is_audio_understandable_model=False,
                hf_config=SimpleNamespace(
                    model_type="llama", architectures=["LlamaForCausalLM"]
                ),
                get_default_sampling_params=lambda: {"temperature": 0.5},
            ),
            rust_server_tokenizer_path=lambda: "/data/tokenizer/tokenizer.json",
            max_total_num_tokens=8192,
        )

        with (
            patch.dict(
                "os.environ",
                {
                    "SGLANG_RUST_METRICS_SOCKET": "/private/collector.sock",
                    "SGLANG_ENABLE_REQUEST_HEADER_OVERRIDES": "1",
                    "SGLANG_ENABLE_REQUEST_DECOMPRESSION": "1",
                },
            ),
            patch(
                "sglang.srt.rust_server.config.get_device_name",
                return_value="NVIDIA GB300",
            ),
        ):
            args = _build_server_args(scheduler, extension=extension)

        self.assertEqual(args.model_path, "dummy")
        self.assertEqual(args.served_model_name, "served-model")
        self.assertEqual(args.public_tokenizer_path, "public-tokenizer")
        self.assertEqual(args.tokenizer_path, "/data/tokenizer/tokenizer.json")
        self.assertEqual(args.tokenizer_model_max_length, 4096)
        self.assertEqual(args.tokenizer_vocab_size, 257)
        self.assertTrue(args.wait_for_parent_warmup)
        self.assertEqual(args.max_return_hidden_states, "last")
        self.assertTrue(args.enable_custom_logit_processor)
        self.assertTrue(args.enable_cache_report)
        self.assertTrue(args.return_input_ids)
        self.assertTrue(args.return_output_ids)
        self.assertTrue(args.enable_strict_thinking)
        self.assertTrue(args.disable_radix_cache)
        self.assertEqual(args.model_config.hidden_size, 512)
        self.assertEqual(args.disaggregation_mode, "null")
        self.assertTrue(args.enable_metrics)
        self.assertEqual(args.metrics_socket, "/private/collector.sock")
        metrics_config = json.loads(args.metrics_config)
        self.assertEqual(
            metrics_config["labels"],
            {
                "model_name": "served-model",
                "engine_type": "unified",
                "priority": "",
                "tenant": "",
                "cluster": "test",
            },
        )
        self.assertEqual(metrics_config["http_labels"], {"cluster": "test"})
        self.assertEqual(metrics_config["allowed_custom_labels"], ["tenant"])
        self.assertEqual(metrics_config["custom_labels_header"], "x-tenant-metrics")
        self.assertEqual(metrics_config["bucket_inter_token_latency"], [0.01, 0.1])
        self.assertEqual(metrics_config["prompt_tokens_buckets"], [1, 32])
        self.assertTrue(args.enable_http2)
        self.assertTrue(args.enable_request_header_overrides)
        self.assertTrue(args.enable_request_decompression)
        self.assertEqual(args.default_priority_value, 7)
        self.assertEqual(args.dp_size, 1)
        self.assertIsNone(args.dp_rank)
        self.assertEqual(args.http2_max_concurrent_streams, 17)
        self.assertEqual(args.http2_initial_connection_window_size, 2 * 1024 * 1024)
        self.assertEqual(args.accelerator, "NVIDIA GB300")
        self.assertEqual(args.num_accelerators, 1)
        self.assertTrue(args.model_config.is_generation)
        self.assertTrue(args.model_config.has_image_understanding)
        self.assertFalse(args.model_config.has_audio_understanding)
        self.assertEqual(args.model_config.model_type, "llama")
        self.assertEqual(args.model_config.architectures, ["LlamaForCausalLM"])
        self.assertEqual(args.model_config.default_sampling_params.temperature, 0.5)

    def test_launch_uses_the_model_extension_and_instance_worker_state(self):
        extension = ModuleType("model_server")
        extension.Server = Mock()
        extension.Server.return_value.http_port = 32768

        class ModelServer(RustServer):
            @classmethod
            def _load_extension(cls):
                return extension

            def _start_multimodal(self, scheduler):
                self.server.start_mm_workers(sentinel.spec, 8)

        scheduler = SimpleNamespace(
            ps=ParallelState.trivial(dp_size=2, dp_rank=1, attn_dp_rank=0),
            model_config=SimpleNamespace(is_multimodal=True),
        )
        with (
            patch.object(ModelServer, "_partition_cores", return_value=(None, None)),
            patch.object(
                server_module,
                "get_mm",
                return_value=SimpleNamespace(mm_processor_worker_num=8),
            ),
            patch.object(
                server_module,
                "get_serving",
                return_value=SimpleNamespace(host="::", port=30000),
            ),
            patch.object(
                server_module, "_build_server_args", return_value=sentinel.args
            ) as build_args,
        ):
            instance = ModelServer.launch(scheduler)

        build_args.assert_called_once_with(scheduler, extension=extension)
        extension.Server.assert_called_once_with(sentinel.args, cores=None, http_port=0)
        instance.server.start_mm_workers.assert_called_once_with(sentinel.spec, 8)
        self.assertIsInstance(instance, ModelServer)
        self.assertEqual(instance.http_port, 32768)
        self.assertEqual(instance.topology.dp_rank, 1)
        self.assertTrue(instance._multimodal_enabled)


if __name__ == "__main__":
    unittest.main()
