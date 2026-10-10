# SPDX-License-Identifier: Apache-2.0
"""Cosmos-Dreams realtime (tick) session: adapter, action control, and tick helpers."""

import importlib.util
import os
import unittest
from types import SimpleNamespace

import torch

from sglang.multimodal_gen.configs.models.dits.cosmos_dreams import (
    parse_cosmos_dreams_manifest,
)
from sglang.multimodal_gen.configs.pipeline_configs.cosmos_dreams import (
    CosmosDreamsConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.cosmos_dreams_realtime import (
    CosmosDreamsRealtimeConfig,
)
from sglang.multimodal_gen.configs.sample.cosmos_dreams import (
    CosmosDreamsSamplingParams,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    RealtimeEvent,
    RealtimeVideoGenerationsRequest,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime import (
    realtime_adapter,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.adapters import (
    cosmos_dreams_realtime_adapter as adapter_module,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.adapters.cosmos_dreams_realtime_adapter import (
    ACTION_EVENT_KIND,
    CosmosDreamsActionControlState,
    CosmosDreamsRealtimeAdapter,
    idle_action_row,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.generate_session import (
    GenerateSession,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.registry import (
    get_realtime_model_adapter,
)
from sglang.multimodal_gen.runtime.pipelines.cosmos_dreams_realtime_pipeline import (
    CosmosDreamsRealtimePipeline,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos_dreams_realtime import (
    ACTION_ROWS_CONDITION,
    CosmosDreamsSessionState,
    block_actions,
    block_frames,
    tick_action_rows,
)


def _load_checkpoint_artifact() -> dict:
    path = os.path.join(os.path.dirname(__file__), "test_cosmos_dreams.py")
    spec = importlib.util.spec_from_file_location("test_cosmos_dreams_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CHECKPOINT_ARTIFACT


MANIFEST = parse_cosmos_dreams_manifest(_load_checkpoint_artifact())
CAMERA = MANIFEST.action_contract.embodiments["camera_pose"]
IDLE = idle_action_row(CAMERA)
FORWARD = [0.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]


class TestActionControl(unittest.TestCase):
    def test_idle_row_is_zero_translation_identity_rotation(self):
        self.assertEqual(IDLE, [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0])

    def _state(self) -> CosmosDreamsActionControlState:
        return CosmosDreamsActionControlState(
            idle_row=IDLE, raw_action_dim=CAMERA.raw_action_dim
        )

    def test_nothing_received_samples_idle_rows(self):
        rows = self._state().sample_rows(4)
        self.assertEqual(rows, [IDLE] * 4)

    def test_single_row_is_held_across_ticks(self):
        state = self._state()
        self.assertEqual(
            state.receive_event_payload(FORWARD, event_id=7), "kind=action, mode=hold"
        )
        self.assertEqual(state.sample_rows(3), [FORWARD] * 3)
        self.assertEqual(state.sample_rows(2), [FORWARD] * 2)
        self.assertEqual(state.latest_sampled_event_id, 7)

    def test_script_is_consumed_then_settles_on_idle(self):
        state = self._state()
        script = [FORWARD] * 5
        self.assertEqual(
            state.receive_event_payload(script, event_id=3),
            "kind=action, mode=script, rows=5",
        )
        self.assertEqual(state.sample_rows(4), [FORWARD] * 4)
        self.assertEqual(state.sample_rows(4), [FORWARD] + [IDLE] * 3)
        self.assertEqual(state.sample_rows(4), [IDLE] * 4)

    def test_state_mode_payload_and_bad_rows(self):
        state = self._state()
        summary = state.receive_event_payload(
            {"mode": "state", "transitions": [{"actions": FORWARD, "client_ts_ms": 5}]},
            event_id=9,
        )
        self.assertIn("mode=state", summary)
        self.assertEqual(state.sample_rows(2), [FORWARD] * 2)
        with self.assertRaises(ValueError):
            state.receive_event_payload([1.0, 2.0], event_id=1)
        with self.assertRaises(ValueError):
            state.receive_event_payload([[float("nan")] * 9], event_id=2)


class TestTickHelpers(unittest.TestCase):
    def test_block_frames_follow_the_trained_chunk(self):
        self.assertEqual(block_frames(None, MANIFEST), MANIFEST.chunk_size)
        self.assertEqual(
            block_frames(2 * MANIFEST.chunk_size, MANIFEST), 2 * MANIFEST.chunk_size
        )
        with self.assertRaises(ValueError):
            block_frames(MANIFEST.chunk_size + 1, MANIFEST)

    def test_tick_action_rows_normalize_and_pad(self):
        steps = MANIFEST.chunk_size * MANIFEST.action_tokens_per_frame
        rows = tick_action_rows(
            [FORWARD] * steps,
            manifest=MANIFEST,
            embodiment="camera_pose",
            frames=MANIFEST.chunk_size,
        )
        self.assertEqual(tuple(rows.shape), (steps, MANIFEST.max_action_dim))
        # pose_scale normalizer: translation x10, rotation unchanged, zero padding.
        self.assertAlmostEqual(rows[0, 2].item(), 1.0, places=5)
        self.assertAlmostEqual(rows[0, 3].item(), 1.0, places=5)
        self.assertEqual(rows[0, 9:].abs().sum().item(), 0.0)
        self.assertIsNone(
            tick_action_rows(
                None, manifest=MANIFEST, embodiment="camera_pose", frames=4
            )
        )
        with self.assertRaises(ValueError):
            tick_action_rows(
                [FORWARD] * (steps - 1),
                manifest=MANIFEST,
                embodiment="camera_pose",
                frames=4,
            )

    def test_block_actions_null_when_missing(self):
        action, null_frames = block_actions(
            None, frames=4, action_tokens_per_frame=4, model_action_dim=64
        )
        self.assertEqual(tuple(action.shape), (1, 16, 64))
        self.assertEqual(null_frames, (0, 1, 2, 3))
        rows = torch.ones(16, 64)
        action, null_frames = block_actions(
            rows, frames=4, action_tokens_per_frame=4, model_action_dim=64
        )
        self.assertEqual(tuple(action.shape), (1, 16, 64))
        self.assertEqual(null_frames, ())

    def test_session_state_dispose_clears_carry_over(self):
        state = CosmosDreamsSessionState()
        state.history = [(torch.zeros(1), torch.zeros(1))]
        state.next_frame = 9
        state.dispose()
        self.assertIsNone(state.history)
        self.assertEqual(state.next_frame, 0)
        self.assertIsNone(state.conditioning)


class TestRegistration(unittest.TestCase):
    def test_pipeline_declares_its_config_pair(self):
        self.assertEqual(
            CosmosDreamsRealtimePipeline.pipeline_name, "CosmosDreamsRealtimePipeline"
        )
        self.assertIs(
            CosmosDreamsRealtimePipeline.pipeline_config_cls, CosmosDreamsRealtimeConfig
        )
        self.assertIs(
            CosmosDreamsRealtimePipeline.sampling_params_cls, CosmosDreamsSamplingParams
        )
        # The explicit pipeline class refines the model-default config only for subclasses.
        self.assertTrue(issubclass(CosmosDreamsRealtimeConfig, CosmosDreamsConfig))

    def test_realtime_config_keeps_dit_resident_on_smaller_cards(self):
        deployment = CosmosDreamsRealtimeConfig.__new__(
            CosmosDreamsRealtimeConfig
        ).get_model_deployment_config()
        self.assertEqual(deployment.keep_resident_min_available_gb, 40)
        self.assertIn("dit", deployment.keep_resident_components)
        self.assertFalse(deployment.supports_cfg_parallel)

    def test_realtime_registry_resolves_adapter(self):
        server_args = SimpleNamespace(
            pipeline_config=CosmosDreamsRealtimeConfig.__new__(
                CosmosDreamsRealtimeConfig
            )
        )
        adapter = get_realtime_model_adapter(server_args)
        self.assertIsInstance(adapter, CosmosDreamsRealtimeAdapter)


class TestAdapter(unittest.TestCase):
    def _session(
        self, **request_fields
    ) -> tuple[CosmosDreamsRealtimeAdapter, GenerateSession]:
        adapter = CosmosDreamsRealtimeAdapter()
        session = GenerateSession()
        session.set_adapter(adapter)
        session.set_request(
            RealtimeVideoGenerationsRequest(
                type="init",
                prompt="a stone lion",
                fps=30,
                first_frame="/tmp/first_frame.jpg",
                **request_fields,
            )
        )
        adapter._state(session).configure(MANIFEST, embodiment="camera_pose")
        return adapter, session

    def test_prepare_next_request_block_zero_and_later(self):
        adapter, session = self._session()
        adapter.ingest_event(
            session,
            RealtimeEvent(
                type="event", kind=ACTION_EVENT_KIND, payload=FORWARD, event_id=11
            ),
        )
        seen: list[dict] = []

        def fake_build_sampling_params(request_id, **kwargs):
            seen.append(kwargs)
            return SimpleNamespace(
                request_id=request_id,
                prompt=kwargs["prompt"],
                condition_inputs=kwargs["condition_inputs"],
                realtime_chunk_size=kwargs["realtime_chunk_size"],
            )

        def fake_prepare_request(server_args, sampling_params):
            return SimpleNamespace(
                request_id=sampling_params.request_id,
                prompt=sampling_params.prompt,
                condition_inputs=dict(sampling_params.condition_inputs),
                realtime_chunk_size=sampling_params.realtime_chunk_size,
                session=None,
            )

        original = (
            adapter_module.build_sampling_params,
            realtime_adapter.prepare_request,
        )
        adapter_module.build_sampling_params = fake_build_sampling_params
        realtime_adapter.prepare_request = fake_prepare_request
        try:
            server_args = SimpleNamespace(pipeline_config=None)
            chunk0 = session.new_chunk()
            batch0 = adapter.prepare_next_request(session, server_args, chunk0)
            session.generate_chunk_completed()
            chunk1 = session.new_chunk()
            batch1 = adapter.prepare_next_request(session, server_args, chunk1)
        finally:
            adapter_module.build_sampling_params, realtime_adapter.prepare_request = (
                original
            )

        steps = MANIFEST.chunk_size * MANIFEST.action_tokens_per_frame
        self.assertEqual(seen[0]["image_path"], "/tmp/first_frame.jpg")
        self.assertIsNone(seen[1]["image_path"])
        self.assertEqual(
            seen[0]["num_frames"],
            1 + MANIFEST.chunk_size * MANIFEST.temporal_compression_factor,
        )
        self.assertEqual(seen[0]["fps"], 30)
        self.assertEqual(seen[0]["guidance_scale"], 1.0)
        self.assertEqual(seen[0]["domain_name"], "camera_pose")
        self.assertNotIn("size", seen[0])  # canvas follows the image aspect
        self.assertEqual(
            batch0.condition_inputs, {ACTION_ROWS_CONDITION: [FORWARD] * steps}
        )
        self.assertEqual(
            batch1.condition_inputs, {ACTION_ROWS_CONDITION: [FORWARD] * steps}
        )
        self.assertEqual(batch0.realtime_chunk_size, MANIFEST.chunk_size)
        self.assertEqual(batch0.block_idx, 0)
        self.assertEqual(batch1.block_idx, 1)
        self.assertEqual(batch1.realtime_event_id, 11)
        self.assertTrue(batch0.return_raw_frames)

    def test_explicit_size_and_unknown_event_kind(self):
        adapter, session = self._session(size="640x640")
        seen: list[dict] = []
        original = (
            adapter_module.build_sampling_params,
            realtime_adapter.prepare_request,
        )
        adapter_module.build_sampling_params = lambda request_id, **kwargs: (
            seen.append(kwargs)
            or SimpleNamespace(
                request_id=request_id,
                prompt=kwargs["prompt"],
                condition_inputs=kwargs["condition_inputs"],
                realtime_chunk_size=kwargs["realtime_chunk_size"],
            )
        )
        realtime_adapter.prepare_request = lambda server_args, sampling_params: (
            SimpleNamespace(
                request_id=sampling_params.request_id,
                prompt=sampling_params.prompt,
                condition_inputs=dict(sampling_params.condition_inputs),
                realtime_chunk_size=sampling_params.realtime_chunk_size,
                session=None,
            )
        )
        try:
            adapter.prepare_next_request(
                session, SimpleNamespace(pipeline_config=None), session.new_chunk()
            )
        finally:
            adapter_module.build_sampling_params, realtime_adapter.prepare_request = (
                original
            )
        self.assertEqual(seen[0]["size"], "640x640")
        with self.assertRaises(ValueError):
            adapter.ingest_event(
                session, RealtimeEvent(type="event", kind="prompt", payload="x")
            )


if __name__ == "__main__":
    unittest.main()
