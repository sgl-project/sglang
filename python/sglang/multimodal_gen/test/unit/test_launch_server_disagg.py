# SPDX-License-Identifier: Apache-2.0
"""Unit tests for disaggregated launch-time calibration helpers."""

import os
import pickle
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.multimodal_gen.configs.pipeline_configs.qwen_image import (
    QwenImageEditPipelineConfig,
    QwenImagePipelineConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.wan import (
    Wan2_2_TI2V_5B_Config,
)
from sglang.multimodal_gen.configs.sample.qwenimage import (
    QwenImageSamplingParams,
)
from sglang.multimodal_gen.configs.sample.wan import (
    Wan2_2_TI2V_5B_SamplingParam,
)
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.distributed import parallel_state
from sglang.multimodal_gen.runtime.launch_server import (
    _build_disagg_calibration_reqs,
    _run_disagg_startup_calibration,
    _spawn_disagg_worker_group,
    _wait_for_disagg_role_registration,
    launch_disagg_role,
    launch_disagg_server,
    launch_pool_disagg_server,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.platforms.cpu import CpuPlatform
from sglang.multimodal_gen.runtime.server_args import ServerArgs


def _make_server_args(**overrides):
    kwargs = {
        "model_path": "/fake",
        "pipeline_config": QwenImagePipelineConfig(),
        "host": "127.0.0.1",
        "scheduler_port": 30020,
        "warmup": False,
        "log_level": "debug",
    }
    kwargs.update(overrides)
    return ServerArgs(**kwargs)


def _make_fake_role_args(**kwargs):
    return SimpleNamespace(
        **kwargs,
        resolved_role_device=lambda device=kwargs["disagg_role_device"]: device,
    )


class _FakeReadyReader:
    def __init__(self, events, rank, payload=None, error=None):
        self.events = events
        self.rank = rank
        self.payload = payload or {"status": "ready"}
        self.error = error
        self.closed = False

    def recv(self):
        self.events.append(("recv", self.rank))
        if self.error is not None:
            raise self.error
        return self.payload

    def close(self):
        if not self.closed:
            self.closed = True
            self.events.append(("close_reader", self.rank))


class _FakeReadyWriter:
    def __init__(self, events, rank):
        self.events = events
        self.rank = rank
        self.closed = False

    def close(self):
        if not self.closed:
            self.closed = True
            self.events.append(("close_writer", self.rank))


class _FakeProcess:
    def __init__(
        self,
        events,
        rank,
        worker_id,
        name,
        *,
        exitcode_after_join=None,
    ):
        self.events = events
        self.rank = rank
        self.worker_id = worker_id
        self.name = name
        self.exitcode = None
        self.exitcode_after_join = exitcode_after_join
        self._alive = False

    def start(self):
        self._alive = True
        platform_override = os.environ.get("SGLANG_DIFFUSION_PLATFORM_OVERRIDE")
        self.events.append(("start", self.rank, self.worker_id, self.name))
        self.events.append(("platform_override", self.rank, platform_override))

    def terminate(self):
        self._alive = False
        self.exitcode = -15
        self.events.append(("terminate", self.rank))

    def join(self, timeout=None):
        self._alive = False
        if self.exitcode_after_join is not None:
            self.exitcode = self.exitcode_after_join
        self.events.append(("join", self.rank, timeout))

    def is_alive(self):
        return self._alive


class _FakeSpawnContext:
    def __init__(self, events, reader_specs=None, process_specs=None):
        self.events = events
        self.reader_specs = reader_specs or {}
        self.process_specs = process_specs or {}
        self.pipe_rank = 0

    def Pipe(self, duplex=False):
        rank = self.pipe_rank
        self.pipe_rank += 1
        spec = self.reader_specs.get(rank, {})
        return (
            _FakeReadyReader(
                self.events,
                rank,
                payload=spec.get("payload"),
                error=spec.get("error"),
            ),
            _FakeReadyWriter(self.events, rank),
        )

    def Process(self, target, args, name, daemon):
        rank = args[1]
        spec = self.process_specs.get(rank, {})
        return _FakeProcess(
            self.events,
            rank,
            args[0],
            name,
            exitcode_after_join=spec.get("exitcode_after_join"),
        )


class TestDisaggStartupCalibrationHelpers(unittest.TestCase):
    def test_build_calibration_reqs_for_image_task(self):
        server_args = _make_server_args(
            warmup=True,
            warmup_resolutions=["640x480"],
            warmup_steps=3,
            model_id="Qwen-Image-Edit",
            pipeline_config=QwenImageEditPipelineConfig(),
        )

        with patch(
            "sglang.multimodal_gen.runtime.warmup_utils.asyncio.run",
            return_value="outputs/uploads/warmup_image.jpg",
        ), patch(
            "sglang.multimodal_gen.runtime.warmup_utils.save_image_to_path",
            new=MagicMock(return_value="ignored"),
        ):
            warmup_reqs = _build_disagg_calibration_reqs(server_args)

        self.assertEqual(len(warmup_reqs), 1)
        req = warmup_reqs[0]
        self.assertTrue(req.is_warmup)
        self.assertEqual(req.width, 640)
        self.assertEqual(req.height, 480)
        self.assertEqual(req.image_path, ["outputs/uploads/warmup_image.jpg"])
        self.assertIsInstance(req.sampling_params, QwenImageSamplingParams)
        self.assertEqual(req.negative_prompt, QwenImageSamplingParams().negative_prompt)
        self.assertNotEqual(req.negative_prompt, "")
        self.assertEqual(req.num_inference_steps, 3)

    def test_build_calibration_reqs_use_model_specific_sampling_params(self):
        server_args = _make_server_args(
            warmup=True,
            warmup_steps=2,
            model_id="Wan2.2-TI2V-5B-Diffusers",
            pipeline_config=Wan2_2_TI2V_5B_Config(),
        )

        with patch(
            "sglang.multimodal_gen.runtime.warmup_utils.asyncio.run",
            return_value="outputs/uploads/warmup_image.jpg",
        ), patch(
            "sglang.multimodal_gen.runtime.warmup_utils.save_image_to_path",
            new=MagicMock(return_value="ignored"),
        ):
            warmup_reqs = _build_disagg_calibration_reqs(server_args)

        self.assertEqual(len(warmup_reqs), 1)
        req = warmup_reqs[0]
        self.assertIsInstance(req.sampling_params, Wan2_2_TI2V_5B_SamplingParam)
        self.assertEqual(req.prompt, "warmup")
        self.assertEqual(req.image_path, ["outputs/uploads/warmup_image.jpg"])
        self.assertEqual(
            req.negative_prompt, Wan2_2_TI2V_5B_SamplingParam().negative_prompt
        )
        self.assertNotEqual(req.negative_prompt, "")
        self.assertTrue(req.is_warmup)
        self.assertEqual(req.num_inference_steps, 2)

    def test_run_startup_calibration_sends_all_requests(self):
        server_args = SimpleNamespace(
            warmup=True,
            disagg_timeout=12,
        )
        fake_socket = MagicMock()
        fake_context = MagicMock()
        fake_context.socket.return_value = fake_socket
        fake_socket.recv_multipart.side_effect = [
            [b"id", pickle.dumps(SimpleNamespace(error=None))],
            [b"id", pickle.dumps(SimpleNamespace(error=None))],
        ]

        with patch(
            "sglang.multimodal_gen.runtime.launch_server._build_disagg_calibration_reqs",
            return_value=[SimpleNamespace(name="r1"), SimpleNamespace(name="r2")],
        ), patch(
            "sglang.multimodal_gen.runtime.launch_server.zmq.Context",
            return_value=fake_context,
        ), patch(
            "sglang.multimodal_gen.runtime.launch_server.time.sleep"
        ):
            _run_disagg_startup_calibration("tcp://127.0.0.1:9999", server_args)

        self.assertEqual(fake_socket.send.call_count, 2)
        fake_socket.connect.assert_called_once_with("tcp://127.0.0.1:9999")
        fake_socket.close.assert_called_once_with(linger=0)
        fake_context.destroy.assert_called_once_with(linger=0)

    def test_run_startup_calibration_fails_fast_on_error_reply(self):
        server_args = SimpleNamespace(
            warmup=True,
            disagg_timeout=5,
        )
        fake_socket = MagicMock()
        fake_context = MagicMock()
        fake_context.socket.return_value = fake_socket
        fake_socket.recv_multipart.return_value = [
            b"id",
            pickle.dumps(SimpleNamespace(error="calibration failed")),
        ]

        with patch(
            "sglang.multimodal_gen.runtime.launch_server._build_disagg_calibration_reqs",
            return_value=[SimpleNamespace(name="r1")],
        ), patch(
            "sglang.multimodal_gen.runtime.launch_server.zmq.Context",
            return_value=fake_context,
        ), patch(
            "sglang.multimodal_gen.runtime.launch_server.time.sleep"
        ), self.assertRaisesRegex(
            RuntimeError, "Disagg startup calibration failed"
        ):
            _run_disagg_startup_calibration("tcp://127.0.0.1:9999", server_args)

    def test_wait_for_role_registration_returns_after_all_peers_register(self):
        diffusion_server = MagicMock()
        diffusion_server.get_stats.side_effect = [
            {"encoder_peers": 1, "denoiser_peers": 0, "decoder_peers": 0},
            {"encoder_peers": 1, "denoiser_peers": 1, "decoder_peers": 1},
        ]

        with patch("sglang.multimodal_gen.runtime.launch_server.time.sleep"):
            _wait_for_disagg_role_registration(
                diffusion_server,
                expected_encoders=1,
                expected_denoisers=1,
                expected_decoders=1,
                timeout_s=1.0,
            )

        self.assertEqual(diffusion_server.get_stats.call_count, 2)

    def test_wait_for_role_registration_times_out_with_peer_counts(self):
        diffusion_server = MagicMock()
        diffusion_server.get_stats.return_value = {
            "encoder_peers": 1,
            "denoiser_peers": 0,
            "decoder_peers": 0,
        }

        with patch(
            "sglang.multimodal_gen.runtime.launch_server.time.monotonic",
            side_effect=[0.0, 0.0, 0.2, 0.2],
        ), patch(
            "sglang.multimodal_gen.runtime.launch_server.time.sleep"
        ), self.assertRaisesRegex(
            RuntimeError,
            "encoder 1/1, denoiser 0/1, decoder 0/1",
        ):
            _wait_for_disagg_role_registration(
                diffusion_server,
                expected_encoders=1,
                expected_denoisers=1,
                expected_decoders=1,
                timeout_s=0.1,
            )

    def test_launch_disagg_server_waits_for_role_registration_before_warmup(self):
        server_args = _make_server_args(
            disagg_role=RoleType.SERVER,
            warmup=True,
            disagg_timeout=12,
            encoder_urls="tcp://127.0.0.1:33020",
            denoiser_urls="tcp://127.0.0.1:33021",
            decoder_urls="tcp://127.0.0.1:33022",
        )
        fake_server = MagicMock()
        events = []

        def record_wait(*args, **kwargs):
            del args, kwargs
            events.append("wait")

        def record_warmup(*args, **kwargs):
            del args, kwargs
            events.append("warmup")

        with patch(
            "sglang.multimodal_gen.runtime.launch_server.DiffusionServer",
            return_value=fake_server,
        ), patch(
            "sglang.multimodal_gen.runtime.launch_server._wait_for_disagg_role_registration",
            side_effect=record_wait,
        ) as wait_mock, patch(
            "sglang.multimodal_gen.runtime.launch_server._run_disagg_startup_calibration",
            side_effect=record_warmup,
        ) as warmup_mock, patch(
            "sglang.multimodal_gen.runtime.launch_server.launch_http_server_only"
        ):
            launch_disagg_server(server_args)

        fake_server.start.assert_called_once_with()
        wait_mock.assert_called_once_with(
            fake_server,
            expected_encoders=1,
            expected_denoisers=1,
            expected_decoders=1,
            timeout_s=12.0,
        )
        warmup_mock.assert_called_once()
        self.assertEqual(events, ["wait", "warmup"])

    def test_launch_disagg_server_waits_for_registration_without_warmup(self):
        server_args = _make_server_args(
            disagg_role=RoleType.SERVER,
            warmup=False,
            disagg_timeout=12,
            encoder_urls="tcp://127.0.0.1:33020",
            denoiser_urls="tcp://127.0.0.1:33021",
            decoder_urls="tcp://127.0.0.1:33022",
        )
        fake_server = MagicMock()

        with patch(
            "sglang.multimodal_gen.runtime.launch_server.DiffusionServer",
            return_value=fake_server,
        ), patch(
            "sglang.multimodal_gen.runtime.launch_server._wait_for_disagg_role_registration",
        ) as wait_mock, patch(
            "sglang.multimodal_gen.runtime.launch_server._run_disagg_startup_calibration",
        ) as warmup_mock, patch(
            "sglang.multimodal_gen.runtime.launch_server.launch_http_server_only"
        ):
            launch_disagg_server(server_args)

        fake_server.start.assert_called_once_with()
        wait_mock.assert_called_once_with(
            fake_server,
            expected_encoders=1,
            expected_denoisers=1,
            expected_decoders=1,
            timeout_s=12.0,
        )
        warmup_mock.assert_not_called()

    def test_launch_pool_disagg_server_waits_for_registration_before_warmup(self):
        server_args = _make_server_args(
            disagg_role_device="cuda",
            warmup=True,
            disagg_timeout=12,
        )
        fake_server = MagicMock()
        events = []

        def record_wait(*args, **kwargs):
            del args, kwargs
            events.append("wait")

        def record_warmup(*args, **kwargs):
            del args, kwargs
            events.append("warmup")

        with patch(
            "sglang.multimodal_gen.runtime.launch_server.DiffusionServer",
            return_value=fake_server,
        ), patch(
            "sglang.multimodal_gen.runtime.launch_server._wait_for_disagg_role_registration",
            side_effect=record_wait,
        ) as wait_mock, patch(
            "sglang.multimodal_gen.runtime.launch_server._run_disagg_startup_calibration",
            side_effect=record_warmup,
        ) as warmup_mock, patch(
            "sglang.multimodal_gen.runtime.launch_server.launch_http_server_only"
        ):
            launch_pool_disagg_server(
                server_args,
                encoder_gpus=[],
                denoiser_gpus=[],
                decoder_gpus=[],
                launch_http_server=False,
            )

        fake_server.start.assert_called_once_with()
        wait_mock.assert_called_once_with(
            fake_server,
            expected_encoders=0,
            expected_denoisers=0,
            expected_decoders=0,
            timeout_s=12.0,
        )
        warmup_mock.assert_called_once()
        self.assertEqual(events, ["wait", "warmup"])

    def test_launch_pool_disagg_server_waits_for_registration_without_warmup(self):
        server_args = _make_server_args(
            disagg_role_device="cuda",
            warmup=False,
            disagg_timeout=12,
        )
        fake_server = MagicMock()

        with patch(
            "sglang.multimodal_gen.runtime.launch_server.DiffusionServer",
            return_value=fake_server,
        ), patch(
            "sglang.multimodal_gen.runtime.launch_server._wait_for_disagg_role_registration",
        ) as wait_mock, patch(
            "sglang.multimodal_gen.runtime.launch_server._run_disagg_startup_calibration",
        ) as warmup_mock, patch(
            "sglang.multimodal_gen.runtime.launch_server.launch_http_server_only"
        ):
            launch_pool_disagg_server(
                server_args,
                encoder_gpus=[],
                denoiser_gpus=[],
                decoder_gpus=[],
                launch_http_server=False,
            )

        fake_server.start.assert_called_once_with()
        wait_mock.assert_called_once_with(
            fake_server,
            expected_encoders=0,
            expected_denoisers=0,
            expected_decoders=0,
            timeout_s=12.0,
        )
        warmup_mock.assert_not_called()


class TestDisaggWorkerLaunchOrdering(unittest.TestCase):
    def _assert_starts_and_closes_before_recv(self, events, expected_ranks):
        first_recv_idx = next(i for i, event in enumerate(events) if event[0] == "recv")
        start_ranks = [event[1] for event in events if event[0] == "start"]
        close_ranks = [event[1] for event in events if event[0] == "close_writer"]

        self.assertEqual(start_ranks, expected_ranks)
        self.assertEqual(close_ranks[: len(expected_ranks)], expected_ranks)

        start_indices = [idx for idx, event in enumerate(events) if event[0] == "start"]
        close_indices = [
            idx for idx, event in enumerate(events) if event[0] == "close_writer"
        ]
        for idx in start_indices + close_indices[: len(expected_ranks)]:
            self.assertLess(idx, first_recv_idx)

    def test_pool_launch_starts_all_ranks_before_waiting_for_ready(self):
        events = []
        pool_ctx = _FakeSpawnContext(events)
        server_args = _make_server_args(disagg_role_device="cuda")

        with patch(
            "sglang.multimodal_gen.runtime.launch_server.mp.get_context",
            return_value=pool_ctx,
        ), patch(
            "sglang.multimodal_gen.runtime.launch_server.ServerArgs.from_kwargs",
            side_effect=lambda **kwargs: _make_fake_role_args(**kwargs),
        ), patch(
            "sglang.multimodal_gen.runtime.launch_server.DiffusionServer"
        ) as diffusion_server_cls, patch(
            "sglang.multimodal_gen.runtime.launch_server._wait_for_disagg_role_registration"
        ), patch(
            "sglang.multimodal_gen.runtime.launch_server.is_port_available",
            return_value=True,
        ):
            diffusion_server_cls.return_value = MagicMock(start=MagicMock())
            launch_pool_disagg_server(
                server_args,
                encoder_gpus=[[4, 5, 6, 7]],
                denoiser_gpus=[],
                decoder_gpus=[],
                launch_http_server=False,
            )

        self._assert_starts_and_closes_before_recv(events, [0, 1, 2, 3])

    def test_standalone_role_launch_starts_all_ranks_before_waiting_for_ready(self):
        events = []
        pool_ctx = _FakeSpawnContext(events)
        server_args = _make_server_args(
            disagg_role=RoleType.ENCODER,
            disagg_server_addr="tcp://127.0.0.1:30020",
            scheduler_port=31020,
            num_gpus=4,
            base_gpu_id=4,
            disagg_role_device="cuda",
        )

        with patch(
            "sglang.multimodal_gen.runtime.launch_server.mp.get_context",
            return_value=pool_ctx,
        ), patch(
            "sglang.multimodal_gen.runtime.launch_server.ServerArgs.from_kwargs",
            side_effect=lambda **kwargs: _make_fake_role_args(**kwargs),
        ), patch(
            "sglang.multimodal_gen.runtime.launch_server.is_port_available",
            return_value=True,
        ):
            launch_disagg_role(server_args)

        self._assert_starts_and_closes_before_recv(events, [0, 1, 2, 3])

    def test_standalone_role_launch_uses_explicit_gpu_ids(self):
        events = []
        pool_ctx = _FakeSpawnContext(events)
        server_args = _make_server_args(
            disagg_role=RoleType.DENOISER,
            disagg_server_addr="tcp://127.0.0.1:30020",
            scheduler_port=31020,
            num_gpus=4,
            base_gpu_id=4,
            gpu_ids=[0, 1, 6, 7],
            disagg_role_device="cuda",
        )

        with patch(
            "sglang.multimodal_gen.runtime.launch_server.mp.get_context",
            return_value=pool_ctx,
        ), patch(
            "sglang.multimodal_gen.runtime.launch_server.ServerArgs.from_kwargs",
            side_effect=lambda **kwargs: _make_fake_role_args(**kwargs),
        ), patch(
            "sglang.multimodal_gen.runtime.launch_server.is_port_available",
            return_value=True,
        ):
            launch_disagg_role(server_args)

        worker_ids = [event[2] for event in events if event[0] == "start"]
        self.assertEqual(worker_ids, [0, 1, 6, 7])

    def test_worker_group_cleanup_terminates_started_processes_on_ready_failure(self):
        events = []
        pool_ctx = _FakeSpawnContext(events, reader_specs={1: {"error": EOFError()}})

        with self.assertRaisesRegex(
            RuntimeError,
            "Pool encoder\\[0\\] rank 1 exited before reporting ready",
        ):
            _spawn_disagg_worker_group(
                pool_ctx=pool_ctx,
                worker_ids=[4, 5, 6, 7],
                role_args=_make_fake_role_args(
                    disagg_role_device="cuda",
                    num_gpus=4,
                ),
                process_name_builder=lambda rank_idx: f"enc-r{rank_idx}",
                group_label="Pool encoder[0]",
            )

        terminate_ranks = [event[1] for event in events if event[0] == "terminate"]
        self.assertEqual(terminate_ranks, [0, 1, 2, 3])

    def test_worker_group_reports_exitcode_after_eof_ready_failure(self):
        events = []
        pool_ctx = _FakeSpawnContext(
            events,
            reader_specs={0: {"error": EOFError()}},
            process_specs={0: {"exitcode_after_join": 7}},
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "Pool encoder\\[0\\] rank 0 exited before reporting ready "
            "\\(exitcode=7\\)",
        ):
            _spawn_disagg_worker_group(
                pool_ctx=pool_ctx,
                worker_ids=[4],
                role_args=_make_fake_role_args(
                    disagg_role_device="cuda",
                    num_gpus=1,
                ),
                process_name_builder=lambda rank_idx: f"enc-r{rank_idx}",
                group_label="Pool encoder[0]",
            )

        self.assertIn(("join", 0, 0.1), events)

    def test_worker_group_sets_cpu_platform_override_before_spawn_start(self):
        events = []
        pool_ctx = _FakeSpawnContext(events)

        with patch.dict(os.environ, {}, clear=True):
            _spawn_disagg_worker_group(
                pool_ctx=pool_ctx,
                worker_ids=[0],
                role_args=_make_fake_role_args(
                    disagg_role_device="cpu",
                    num_gpus=1,
                ),
                process_name_builder=lambda rank_idx: f"enc-r{rank_idx}",
                group_label="Pool encoder[0]",
            )
            self.assertIsNone(os.environ.get("SGLANG_DIFFUSION_PLATFORM_OVERRIDE"))

        override_events = [event for event in events if event[0] == "platform_override"]
        self.assertEqual(override_events, [("platform_override", 0, "cpu")])

    def test_worker_group_restores_platform_override_between_roles(self):
        events = []
        pool_ctx = _FakeSpawnContext(events)

        with patch.dict(
            os.environ, {"SGLANG_DIFFUSION_PLATFORM_OVERRIDE": "parent"}, clear=False
        ):
            _spawn_disagg_worker_group(
                pool_ctx=pool_ctx,
                worker_ids=[0],
                role_args=_make_fake_role_args(
                    disagg_role_device="cpu",
                    num_gpus=1,
                ),
                process_name_builder=lambda rank_idx: f"enc-r{rank_idx}",
                group_label="Pool encoder[0]",
            )
            _spawn_disagg_worker_group(
                pool_ctx=pool_ctx,
                worker_ids=[1, 2],
                role_args=_make_fake_role_args(
                    disagg_role_device="cuda",
                    num_gpus=2,
                ),
                process_name_builder=lambda rank_idx: f"den-r{rank_idx}",
                group_label="Pool denoiser[0]",
            )
            self.assertEqual(
                os.environ.get("SGLANG_DIFFUSION_PLATFORM_OVERRIDE"), "parent"
            )

        override_events = [event for event in events if event[0] == "platform_override"]
        self.assertEqual(
            override_events,
            [
                ("platform_override", 0, "cpu"),
                ("platform_override", 0, "cuda"),
                ("platform_override", 1, "cuda"),
            ],
        )


# Consolidated from test_parallel_state.py.
def _fake_platform(
    *,
    cuda_alike: bool,
    device_name: str,
    mps: bool = False,
    musa: bool = False,
    npu: bool = False,
):
    return SimpleNamespace(
        is_cuda_alike=lambda: cuda_alike,
        is_mps=lambda: mps,
        is_musa=lambda: musa,
        is_npu=lambda: npu,
        device_name=device_name,
    )


class TestDistributedInitDeviceId(unittest.TestCase):
    def tearDown(self):
        parallel_state._WORLD = None

    def _run_init(self, *, platform, backend, device_id):
        fake_world = SimpleNamespace(world_size=1)
        with patch.object(
            parallel_state.torch.distributed,
            "is_initialized",
            return_value=False,
        ), patch.object(
            parallel_state.torch.distributed,
            "init_process_group",
        ) as init_process_group, patch.object(
            parallel_state.torch.distributed,
            "get_world_size",
            return_value=1,
        ), patch.object(
            parallel_state,
            "init_world_group",
            return_value=fake_world,
        ), patch(
            "sglang.multimodal_gen.runtime.platforms.current_platform",
            platform,
        ):
            parallel_state._WORLD = None
            parallel_state.init_distributed_environment(
                world_size=1,
                rank=0,
                distributed_init_method="tcp://127.0.0.1:12345",
                local_rank=0,
                backend=backend,
                device_id=device_id,
                timeout=12,
            )

        return init_process_group.call_args.kwargs

    def test_cpu_device_id_is_not_passed_to_process_group(self):
        kwargs = self._run_init(
            platform=_fake_platform(cuda_alike=False, device_name="CPU"),
            backend="gloo",
            device_id=torch.device("cpu"),
        )

        self.assertEqual(kwargs["backend"], "gloo")
        self.assertNotIn("device_id", kwargs)

    def test_nccl_backend_falls_back_to_gloo_without_device_id_on_cpu(self):
        kwargs = self._run_init(
            platform=_fake_platform(cuda_alike=False, device_name="CPU"),
            backend="nccl",
            device_id=torch.device("cpu"),
        )

        self.assertEqual(kwargs["backend"], "gloo")
        self.assertNotIn("device_id", kwargs)

    def test_indexed_cuda_device_id_is_passed_to_process_group(self):
        device_id = torch.device("cuda:1")
        kwargs = self._run_init(
            platform=_fake_platform(cuda_alike=True, device_name="cuda"),
            backend="nccl",
            device_id=device_id,
        )

        self.assertEqual(kwargs["backend"], "nccl")
        self.assertEqual(kwargs["device_id"], device_id)

    def test_musa_like_platform_keeps_omitting_device_id(self):
        kwargs = self._run_init(
            platform=_fake_platform(
                cuda_alike=True,
                device_name="musa",
                musa=True,
            ),
            backend="gloo",
            device_id=torch.device("cuda:0"),
        )

        self.assertEqual(kwargs["backend"], "gloo")
        self.assertNotIn("device_id", kwargs)


# Consolidated from test_cpu_platform.py.
class TestCpuPlatformAttentionBackend(unittest.TestCase):
    def test_cpu_defaults_to_torch_sdpa_backend(self):
        self.assertEqual(
            CpuPlatform.get_attn_backend_cls_str(None, 64, torch.float32),
            "sglang.multimodal_gen.runtime.layers.attention.backends.sdpa.SDPABackend",
        )

    def test_cpu_accepts_explicit_torch_sdpa_backend(self):
        self.assertEqual(
            CpuPlatform.get_attn_backend_cls_str(
                AttentionBackendEnum.TORCH_SDPA,
                64,
                torch.float32,
            ),
            "sglang.multimodal_gen.runtime.layers.attention.backends.sdpa.SDPABackend",
        )

    def test_cpu_rejects_flash_attention_backend(self):
        with self.assertRaisesRegex(ValueError, "FA is not supported on CPU"):
            CpuPlatform.get_attn_backend_cls_str(
                AttentionBackendEnum.FA,
                64,
                torch.float16,
            )


if __name__ == "__main__":
    unittest.main()
