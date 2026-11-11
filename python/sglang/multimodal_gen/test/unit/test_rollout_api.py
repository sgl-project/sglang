"""Unit tests for the rollout generate API (serialization, io_struct, rollout_api).

Request timing model (TestRequestStamps / TestStagesHeader):

    srv_recv .. forward .. build .. dump .. msgpack   (absolute wall-clock marks)
    -> all marks ride the x-sgld-timing response header
    stages_header: engine per-stage milliseconds -> x-sgld-stages header
"""

import json
import time
import types
import unittest
from unittest.mock import patch

import msgspec
import torch

from radix_transfer_plugins import StorageBackend
from sglang.multimodal_gen.runtime.entrypoints.post_training import rollout_api
from sglang.multimodal_gen.runtime.entrypoints.post_training.output_store import (
    RolloutOutputStore,
)
from sglang.multimodal_gen.runtime.entrypoints.post_training.request_timing import (
    MARKS,
    RequestStamps,
    stages_header,
)
from sglang.multimodal_gen.runtime.entrypoints.post_training.utils import (
    _maybe_deserialize,
    _maybe_serialize,
    bytes_to_tensor,
    tensor_to_bytes,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch
from sglang.multimodal_gen.runtime.post_training.rl_dataclasses import (
    RolloutDebugTensors,
    RolloutDenoisingEnv,
    RolloutDitTrajectory,
    RolloutTrajectoryData,
)


class TestTensorToBytesRoundtrip(unittest.TestCase):
    def _roundtrip(self, t: torch.Tensor):
        encoded = tensor_to_bytes(t)
        self.assertIsInstance(encoded, bytes)
        decoded = bytes_to_tensor(encoded)
        self.assertTrue(
            torch.equal(t, decoded), f"Mismatch for shape={t.shape} dtype={t.dtype}"
        )

    def test_float32_1d(self):
        self._roundtrip(torch.randn(16))

    def test_float32_nd(self):
        self._roundtrip(torch.randn(2, 4, 8, 8))

    def test_float16(self):
        self._roundtrip(torch.randn(3, 5).half())

    def test_int64(self):
        self._roundtrip(torch.arange(10))

    def test_bool(self):
        self._roundtrip(torch.tensor([True, False, True]))

    def test_scalar(self):
        self._roundtrip(torch.tensor(3.14))

    def test_empty(self):
        self._roundtrip(torch.empty(0))

    def test_cuda_tensor_moves_to_cpu(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        t = torch.randn(4, device="cuda")
        encoded = tensor_to_bytes(t)
        decoded = bytes_to_tensor(encoded)
        self.assertTrue(torch.equal(t.cpu(), decoded))

    def test_non_contiguous(self):
        t = torch.randn(4, 6)[:, ::2]
        self.assertFalse(t.is_contiguous())
        self._roundtrip(t.contiguous())
        decoded = bytes_to_tensor(tensor_to_bytes(t))
        self.assertTrue(torch.equal(t.contiguous(), decoded))

    def test_grad_tensor_detaches(self):
        t = torch.randn(3, requires_grad=True)
        encoded = tensor_to_bytes(t)
        decoded = bytes_to_tensor(encoded)
        self.assertFalse(decoded.requires_grad)
        self.assertTrue(torch.equal(t.detach(), decoded))


class TestMaybeSerialize(unittest.TestCase):
    def test_tensor(self):
        t = torch.randn(2, 3)
        result = _maybe_serialize(t)
        self.assertIsInstance(result, dict)
        self.assertTrue(result["__tensor__"])
        self.assertEqual(result["shape"], [2, 3])
        self.assertEqual(result["dtype"], "torch.float32")
        decoded = bytes_to_tensor(result["data"])
        self.assertTrue(torch.equal(t, decoded))

    def test_dict_with_tensors(self):
        d = {"a": torch.tensor([1.0]), "b": "hello", "c": 42}
        result = _maybe_serialize(d)
        self.assertIsInstance(result, dict)
        self.assertTrue(result["a"]["__tensor__"])
        self.assertEqual(result["b"], "hello")
        self.assertEqual(result["c"], 42)

    def test_list_with_tensors(self):
        lst = [torch.tensor(1.0), "text", torch.tensor(2.0)]
        result = _maybe_serialize(lst)
        self.assertIsInstance(result, list)
        self.assertTrue(result[0]["__tensor__"])
        self.assertEqual(result[1], "text")
        self.assertTrue(result[2]["__tensor__"])

    def test_nested_structure(self):
        nested = {
            "level1": {"level2": [torch.tensor(1.0), {"level3": torch.tensor(2.0)}]}
        }
        result = _maybe_serialize(nested)
        self.assertTrue(result["level1"]["level2"][0]["__tensor__"])
        self.assertTrue(result["level1"]["level2"][1]["level3"]["__tensor__"])

    def test_none_passthrough(self):
        self.assertIsNone(_maybe_serialize(None))

    def test_plain_values_passthrough(self):
        self.assertEqual(_maybe_serialize(42), 42)
        self.assertEqual(_maybe_serialize("hello"), "hello")
        self.assertAlmostEqual(_maybe_serialize(3.14), 3.14)

    def test_tuple_becomes_list(self):
        result = _maybe_serialize((torch.tensor(1.0), 2))
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)


from sglang.multimodal_gen.runtime.entrypoints.post_training.rollout_api import (
    _build_response,
    _serialize_rollout_trajectory,
)


class TestSerializeRolloutTrajectory(unittest.TestCase):
    def test_none_input(self):
        log_probs, debug, env, dit_traj = _serialize_rollout_trajectory(None)
        self.assertIsNone(log_probs)
        self.assertIsNone(debug)
        self.assertIsNone(env)
        self.assertIsNone(dit_traj)

    def test_log_probs_only(self):
        rtd = RolloutTrajectoryData(
            rollout_log_probs=torch.tensor([-1.0, -2.0]),
        )
        log_probs, debug, env, dit_traj = _serialize_rollout_trajectory(rtd)
        self.assertIsNotNone(log_probs)
        self.assertTrue(log_probs["__tensor__"])
        self.assertIsNone(debug)
        self.assertIsNone(env)
        self.assertIsNone(dit_traj)

    def test_log_probs_none_in_rtd(self):
        rtd = RolloutTrajectoryData(rollout_log_probs=None)
        log_probs, debug, env, dit_traj = _serialize_rollout_trajectory(rtd)
        self.assertIsNone(log_probs)
        self.assertIsNone(debug)
        self.assertIsNone(env)
        self.assertIsNone(dit_traj)

    def test_with_debug_tensors(self):
        dt = RolloutDebugTensors(
            rollout_variance_noises=torch.randn(2, 5, 4, 8, 8),
            rollout_prev_sample_means=torch.randn(2, 5, 4, 8, 8),
            rollout_noise_std_devs=torch.randn(2, 5, 1),
            rollout_model_outputs=torch.randn(2, 5, 4, 8, 8),
        )
        rtd = RolloutTrajectoryData(
            rollout_log_probs=torch.tensor([-0.5, -0.6]),
            rollout_debug_tensors=dt,
        )
        log_probs, debug, env, dit_traj = _serialize_rollout_trajectory(rtd)
        self.assertIsNotNone(log_probs)
        self.assertIsNotNone(debug)
        self.assertIsNone(env)
        self.assertIsNone(dit_traj)
        self.assertIn("rollout_variance_noises", debug)
        self.assertIn("rollout_prev_sample_means", debug)
        self.assertIn("rollout_noise_std_devs", debug)
        self.assertIn("rollout_model_outputs", debug)
        self.assertTrue(debug["rollout_variance_noises"]["__tensor__"])

    def test_debug_tensors_with_none_fields(self):
        dt = RolloutDebugTensors(
            rollout_variance_noises=None,
            rollout_prev_sample_means=torch.randn(1, 2, 4, 4, 4),
            rollout_noise_std_devs=None,
            rollout_model_outputs=None,
        )
        rtd = RolloutTrajectoryData(
            rollout_log_probs=torch.tensor([-0.3]),
            rollout_debug_tensors=dt,
        )
        log_probs, debug, env, dit_traj = _serialize_rollout_trajectory(rtd)
        self.assertIsNotNone(debug)
        self.assertIsNone(debug["rollout_variance_noises"])
        self.assertTrue(debug["rollout_prev_sample_means"]["__tensor__"])
        self.assertIsNone(env)
        self.assertIsNone(dit_traj)

    def test_with_denoising_env(self):
        rtd = RolloutTrajectoryData(
            denoising_env=RolloutDenoisingEnv(
                image_kwargs={"encoder_hidden_states_image": [torch.randn(1, 8)]},
                pos_cond_kwargs={"encoder_hidden_states": torch.randn(1, 8)},
                neg_cond_kwargs={"encoder_hidden_states": torch.randn(1, 8)},
                guidance=torch.tensor([3.5]),
            ),
            dit_trajectory=RolloutDitTrajectory(
                latents=torch.randn(1, 5, 4, 2, 2, 2),
                timesteps=torch.tensor([1.0, 0.75, 0.5, 0.25]),
                sigmas=torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0]),
            ),
        )
        _, _, env, dit_traj = _serialize_rollout_trajectory(
            rtd,
            serialized_dit_timesteps=_maybe_serialize(rtd.dit_trajectory.timesteps),
            serialized_dit_sigmas=_maybe_serialize(rtd.dit_trajectory.sigmas),
        )
        self.assertIsNotNone(env)
        self.assertIn("pos_cond_kwargs", env)
        self.assertNotIn("trajectory", env)
        self.assertIsNotNone(dit_traj)
        self.assertIn("latents", dit_traj)
        self.assertIn("timesteps", dit_traj)
        self.assertIn("sigmas", dit_traj)
        self.assertTrue(dit_traj["latents"]["__tensor__"])
        self.assertTrue(dit_traj["timesteps"]["__tensor__"])
        self.assertTrue(dit_traj["sigmas"]["__tensor__"])


class TestBuildResponse(unittest.TestCase):
    def _make_metrics(self, duration_s: float = 1.0):
        return types.SimpleNamespace(total_duration_s=duration_s)

    def test_minimal_output(self):
        batch = OutputBatch(
            output=torch.randn(1, 3, 1, 64, 64),
            rollout_trajectory_data=RolloutTrajectoryData(
                rollout_log_probs=torch.tensor([0.0]),
            ),
        )
        batch.metrics = self._make_metrics(2.5)
        resps = _build_response("r1", "prompt", 42, True, batch)
        self.assertEqual(len(resps), 1)
        resp = resps[0]
        self.assertEqual(resp.request_id, "r1")
        self.assertEqual(resp.prompt, "prompt")
        self.assertEqual(resp.seed, 42)
        self.assertIsNotNone(resp.generated_output)
        self.assertIsNotNone(resp.rollout_log_probs)
        lp = bytes_to_tensor(resp.rollout_log_probs["data"])
        self.assertEqual(lp.shape, ())
        self.assertAlmostEqual(resp.inference_time_s, 2.5)

    def test_full_response(self):
        batch = OutputBatch(
            output=torch.randn(1, 3, 1, 64, 64),
            rollout_trajectory_data=RolloutTrajectoryData(
                rollout_log_probs=torch.tensor([-0.5]),
            ),
            peak_memory_mb=8192.0,
        )
        batch.metrics = self._make_metrics(5.0)
        resps = _build_response("r2", "test", 99, True, batch)
        self.assertEqual(len(resps), 1)
        resp = resps[0]
        self.assertIsNotNone(resp.rollout_log_probs)
        self.assertIsNone(resp.rollout_debug_tensors)
        self.assertAlmostEqual(resp.peak_memory_mb, 8192.0)

    def test_no_metrics(self):
        batch = OutputBatch(
            output=torch.randn(1, 3, 1, 64, 64),
            rollout_trajectory_data=RolloutTrajectoryData(
                rollout_log_probs=torch.tensor([0.0]),
            ),
        )
        batch.metrics = None
        resp = _build_response("r3", "p", 1, True, batch)[0]
        self.assertIsNone(resp.inference_time_s)

    def test_zero_metrics(self):
        batch = OutputBatch(
            output=torch.randn(1, 3, 1, 64, 64),
            rollout_trajectory_data=RolloutTrajectoryData(
                rollout_log_probs=torch.tensor([0.0]),
            ),
        )
        batch.metrics = self._make_metrics(0.0)
        resp = _build_response("r4", "p", 1, True, batch)[0]
        self.assertIsNone(resp.inference_time_s)

    def test_zero_peak_memory(self):
        batch = OutputBatch(
            output=torch.randn(1, 3, 1, 64, 64),
            peak_memory_mb=0.0,
            rollout_trajectory_data=RolloutTrajectoryData(
                rollout_log_probs=torch.tensor([0.0]),
            ),
        )
        batch.metrics = None
        resp = _build_response("r6", "p", 1, True, batch)[0]
        self.assertIsNone(resp.peak_memory_mb)

    def test_batch_splits_log_probs_and_output(self):
        B, T = 2, 3
        batch = OutputBatch(
            output=torch.randn(B, 1, 8, 8),
            rollout_trajectory_data=RolloutTrajectoryData(
                rollout_log_probs=torch.randn(B, T),
            ),
        )
        batch.metrics = self._make_metrics(1.0)
        resps = _build_response("rb", "p", 0, True, batch)
        self.assertEqual(len(resps), B)
        lp0 = bytes_to_tensor(resps[0].rollout_log_probs["data"])
        lp1 = bytes_to_tensor(resps[1].rollout_log_probs["data"])
        self.assertEqual(lp0.shape, (T,))
        self.assertEqual(lp1.shape, (T,))
        g0 = bytes_to_tensor(resps[0].generated_output["data"])
        g1 = bytes_to_tensor(resps[1].generated_output["data"])
        self.assertEqual(g0.shape, (1, 8, 8))
        self.assertEqual(g1.shape, (1, 8, 8))
        self.assertFalse(torch.equal(g0, g1))

    def test_batch_dit_timesteps_on_each_row_one_serialize(self):
        B, T, D = 2, 4, 3
        batch = OutputBatch(
            output=torch.randn(B, 1, 8, 8),
            rollout_trajectory_data=RolloutTrajectoryData(
                rollout_log_probs=torch.randn(B, T),
                dit_trajectory=RolloutDitTrajectory(
                    latents=torch.randn(B, T + 1, D),
                    timesteps=torch.linspace(1.0, 0.0, T),
                    sigmas=torch.linspace(1.0, 0.0, T + 1),
                ),
            ),
        )
        batch.metrics = self._make_metrics(1.0)
        resps = _build_response("r", "p", 0, True, batch)
        self.assertEqual(len(resps), B)
        self.assertIsNotNone(resps[0].dit_trajectory)
        self.assertIsNotNone(resps[1].dit_trajectory)
        ts0 = bytes_to_tensor(resps[0].dit_trajectory["timesteps"]["data"])
        ts1 = bytes_to_tensor(resps[1].dit_trajectory["timesteps"]["data"])
        self.assertEqual(ts0.shape, (T,))
        self.assertTrue(torch.equal(ts0, ts1))
        sg0 = bytes_to_tensor(resps[0].dit_trajectory["sigmas"]["data"])
        sg1 = bytes_to_tensor(resps[1].dit_trajectory["sigmas"]["data"])
        self.assertEqual(sg0.shape, (T + 1,))
        self.assertTrue(torch.equal(sg0, sg1))
        self.assertEqual(
            _maybe_deserialize(resps[1].dit_trajectory["latents"]).shape, (T + 1, D)
        )

    def test_rollout_false_omits_trajectory(self):
        batch = OutputBatch(
            output=torch.randn(2, 1, 8, 8),
            rollout_trajectory_data=None,
        )
        batch.metrics = self._make_metrics(1.0)
        resps = _build_response("r0", "p", 0, False, batch)
        self.assertEqual(len(resps), 2)
        self.assertIsNone(resps[0].rollout_log_probs)
        self.assertIsNone(resps[1].rollout_log_probs)
        self.assertIsNotNone(resps[0].generated_output)


class TestBuildSamplingKwargs(unittest.TestCase):
    def _make_request(self, **overrides):
        from sglang.multimodal_gen.runtime.entrypoints.post_training.io_struct import (
            RolloutRequest,
        )

        base = dict(prompt="x", num_inference_steps=4, rollout=True)
        base.update(overrides)
        return RolloutRequest(**base)

    def test_step_index_filters_forwarded(self):
        from sglang.multimodal_gen.runtime.entrypoints.post_training.rollout_api import (
            _build_sampling_kwargs,
        )

        kwargs = _build_sampling_kwargs(
            self._make_request(
                rollout_sde_step_indices=[0, 2],
                rollout_return_step_indices=[1, 3],
                rollout_stop_step_index=2,
            )
        )
        self.assertEqual(kwargs["rollout_sde_step_indices"], [0, 2])
        self.assertEqual(kwargs["rollout_return_step_indices"], [1, 3])
        self.assertEqual(kwargs["rollout_stop_step_index"], 2)

    def test_step_index_filters_default_dropped_as_none(self):
        from sglang.multimodal_gen.runtime.entrypoints.post_training.rollout_api import (
            _build_sampling_kwargs,
        )

        kwargs = _build_sampling_kwargs(self._make_request())
        # None values are stripped; absence here is the correct default-path behavior.
        self.assertNotIn("rollout_sde_step_indices", kwargs)
        self.assertNotIn("rollout_return_step_indices", kwargs)

    def test_sampling_params_exposes_filters_via_req_getattr(self):
        from sglang.multimodal_gen.configs.sample.sampling_params import (
            SamplingParams,
        )
        from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req

        sp = SamplingParams(
            prompt="x",
            num_inference_steps=4,
            rollout=True,
            rollout_sde_step_indices=[0, 2],
            rollout_return_step_indices=[1, 3],
        )
        req = Req(sampling_params=sp)
        self.assertEqual(req.rollout_sde_step_indices, [0, 2])
        self.assertEqual(req.rollout_return_step_indices, [1, 3])


class TestRequestStamps(unittest.TestCase):
    """The client joins these marks with its own, so what matters is that the
    header decodes, keeps wall-clock order, and stays readable when a request
    returned early with only some marks taken."""

    def _decode(self, stamps: RequestStamps) -> dict:
        header = stamps.to_header()
        self.assertEqual(header, header.encode("ascii").decode())
        return json.loads(header)

    def test_marks_decode_in_wall_clock_order(self):
        before = time.time()
        stamps = RequestStamps("req-1")
        for name in MARKS:
            stamps.mark(name)
        decoded = self._decode(stamps)
        self.assertEqual(decoded.pop("request_id"), "req-1")
        self.assertEqual(set(decoded), set(MARKS))
        values = [decoded[name] for name in MARKS]
        self.assertEqual(values, sorted(values))
        self.assertGreaterEqual(values[0], round(before, 6) - 1e-6)

    def test_span_pairs_start_and_end_marks(self):
        stamps = RequestStamps()
        with stamps.span("forward"):
            pass
        with stamps.span("dump"):
            pass
        self.assertEqual(
            set(json.loads(stamps.to_header())),
            {"forward_start", "forward_end", "dump_end"},
        )

    def test_partial_marks_only_report_what_was_taken(self):
        stamps = RequestStamps("req-3")
        stamps.mark("srv_recv")
        self.assertEqual(set(self._decode(stamps)), {"srv_recv", "request_id"})

    def test_unknown_mark_is_rejected(self):
        with self.assertRaises(AssertionError):
            RequestStamps().mark("no_such_mark")


class TestStagesHeader(unittest.TestCase):
    """The engine's own stage breakdown is the only view inside the forward the
    client waits on, so it has to survive the trip and stay absent when unset."""

    def test_stages_are_reported_in_milliseconds(self):
        metrics = types.SimpleNamespace(
            stages={"decoding": 8123.4567, "text_encoding": 91.2}
        )
        self.assertEqual(
            json.loads(stages_header(metrics)),
            {"decoding": 8123.457, "text_encoding": 91.2},
        )

    def test_no_metrics_or_no_stages_yields_no_header(self):
        self.assertEqual(stages_header(None), "")
        self.assertEqual(stages_header(types.SimpleNamespace(stages={})), "")


class _MemoryBackend(StorageBackend):
    name = "memory"
    config_type = msgspec.Struct

    def __init__(self):
        self.objects = {}

    def put_many(self, objects):
        keys = []
        for parts in objects:
            key = f"mem/{len(self.objects)}"
            self.objects[key] = {name: bytes(data) for name, data in parts.items()}
            keys.append(key)
        return keys

    def get_many(self, keys, *, parts):
        return [
            {name: memoryview(bytearray(self.objects[key][name])) for name, _ in named}
            for key, named in zip(keys, parts)
        ]

    def register(self, memory):
        pass

    def unregister(self, memory):
        pass

    def get_ranges(self, reads, *, into):
        raise NotImplementedError

    def delete_many(self, keys, *, parts):
        for key in keys:
            del self.objects[key]

    def close(self):
        pass


def _store_batch(B=2, T=2, D=4, L=5, latent_steps=(0, 1, 2)):
    K = len(latent_steps)
    batch = OutputBatch(
        output=torch.randn(B, 1, 8, 8),
        rollout_trajectory_data=RolloutTrajectoryData(
            rollout_log_probs=torch.randn(B, T),
            rollout_debug_tensors=RolloutDebugTensors(
                rollout_model_outputs=torch.randn(B, T, D),
            ),
            denoising_env=RolloutDenoisingEnv(
                pos_cond_kwargs={
                    "encoder_hidden_states": [torch.randn(B, L, D)],
                    "pooled": torch.randn(L, D),
                    "img_shapes": [[1, 2, 3], [4, 5, 6]],
                    "txt_seq_lens": [7, 8],
                },
                guidance=torch.tensor([1.0, 2.0]),
            ),
            dit_trajectory=RolloutDitTrajectory(
                latents=torch.randn(B, K, D),
                timesteps=torch.linspace(1.0, 0.0, T),
                sigmas=torch.linspace(1.0, 0.0, T + 1),
                latent_step_indices=torch.tensor(latent_steps),
            ),
        ),
    )
    batch.metrics = types.SimpleNamespace(total_duration_s=1.0)
    return batch


def _assert_same_tensors(test, inline, via_store, stored):
    """Every marker in ``via_store`` resolves to the tensor ``inline`` carries there."""
    if isinstance(via_store, dict) and "__store__" in via_store:
        tensor = stored[via_store["__store__"]]
        test.assertTrue(torch.equal(_maybe_deserialize(inline), tensor))
    elif isinstance(via_store, list) and "__tensor__" in inline:
        rows = _maybe_deserialize(inline)
        test.assertEqual(len(rows), len(via_store))
        for row, marker in zip(rows, via_store):
            test.assertTrue(torch.equal(row, stored[marker["__store__"]]))
    elif isinstance(via_store, dict):
        test.assertEqual(set(inline), set(via_store))
        for key in via_store:
            _assert_same_tensors(test, inline[key], via_store[key], stored)
    elif isinstance(via_store, list):
        test.assertEqual(len(inline), len(via_store))
        for a, b in zip(inline, via_store):
            _assert_same_tensors(test, a, b, stored)
    else:
        test.assertEqual(inline, via_store)


class TestBuildResponseViaStore(unittest.TestCase):
    def test_batched_tensors_are_stored_one_field_per_sample_and_step(self):
        batch = _store_batch()
        rtd = batch.rollout_trajectory_data
        stored = {}
        resps = _build_response("r", "p", 0, True, batch, stored=stored)
        self.assertEqual(
            set(stored),
            {
                *(f"dit_trajectory.latents.{b}.{k}" for b in (0, 1) for k in (0, 1, 2)),
                *(
                    f"rollout_debug_tensors.rollout_model_outputs.{b}.{t}"
                    for b in (0, 1)
                    for t in (0, 1)
                ),
                "denoising_env.pos_cond_kwargs.encoder_hidden_states.0.0",
                "denoising_env.pos_cond_kwargs.encoder_hidden_states.0.1",
                "denoising_env.pos_cond_kwargs.pooled",
            },
        )
        self.assertTrue(
            torch.equal(
                stored["dit_trajectory.latents.1.2"], rtd.dit_trajectory.latents[1, 2]
            )
        )
        self.assertIs(
            stored["denoising_env.pos_cond_kwargs.pooled"],
            rtd.denoising_env.pos_cond_kwargs["pooled"],
        )
        for row, resp in enumerate(resps):
            self.assertEqual(
                resp.dit_trajectory["latents"],
                [{"__store__": f"dit_trajectory.latents.{row}.{k}"} for k in (0, 1, 2)],
            )
            self.assertEqual(
                resp.rollout_debug_tensors["rollout_model_outputs"],
                [
                    {
                        "__store__": f"rollout_debug_tensors.rollout_model_outputs.{row}.{t}"
                    }
                    for t in (0, 1)
                ],
            )
            cond = resp.denoising_env["pos_cond_kwargs"]
            self.assertEqual(
                cond["encoder_hidden_states"],
                [
                    {
                        "__store__": f"denoising_env.pos_cond_kwargs.encoder_hidden_states.0.{row}"
                    }
                ],
            )
            self.assertEqual(
                cond["pooled"], {"__store__": "denoising_env.pos_cond_kwargs.pooled"}
            )
            self.assertEqual(cond["img_shapes"], [[[1, 2, 3], [4, 5, 6]][row]])
            self.assertEqual(cond["txt_seq_lens"], [7, 8])
            self.assertIsNone(resp.rollout_debug_tensors["rollout_variance_noises"])
            self.assertIsNone(resp.output_store_ref)

    def test_windowed_latents_are_named_by_their_original_step(self):
        batch = _store_batch(latent_steps=(2, 3, 7))
        latents = batch.rollout_trajectory_data.dit_trajectory.latents
        stored = {}
        resps = _build_response("r", "p", 0, True, batch, stored=stored)
        for row, resp in enumerate(resps):
            self.assertEqual(
                resp.dit_trajectory["latents"],
                [{"__store__": f"dit_trajectory.latents.{row}.{s}"} for s in (2, 3, 7)],
            )
            for position, step in enumerate((2, 3, 7)):
                self.assertTrue(
                    torch.equal(
                        stored[f"dit_trajectory.latents.{row}.{step}"],
                        latents[row, position],
                    )
                )

    def test_small_tensors_stay_inline(self):
        batch = _store_batch(T=2)
        resps = _build_response("r", "p", 0, True, batch, stored={})
        for row, resp in enumerate(resps):
            self.assertEqual(
                bytes_to_tensor(resp.rollout_log_probs["data"]).shape, (2,)
            )
            self.assertEqual(
                bytes_to_tensor(resp.denoising_env["guidance"]["data"]).item(),
                row + 1.0,
            )
            self.assertEqual(
                bytes_to_tensor(resp.dit_trajectory["timesteps"]["data"]).shape, (2,)
            )
            self.assertEqual(
                bytes_to_tensor(
                    resp.dit_trajectory["latent_step_indices"]["data"]
                ).tolist(),
                [0, 1, 2],
            )
            self.assertIsNotNone(resp.generated_output)

    def test_markers_resolve_to_the_inline_tensors(self):
        batch = _store_batch()
        stored = {}
        inline = _build_response("r", "p", 0, True, batch)
        via_store = _build_response("r", "p", 0, True, batch, stored=stored)
        for a, b in zip(inline, via_store):
            for field in ("rollout_debug_tensors", "denoising_env", "dit_trajectory"):
                _assert_same_tensors(self, getattr(a, field), getattr(b, field), stored)

    def test_eval_response_stores_nothing(self):
        batch = OutputBatch(
            output=torch.randn(2, 1, 8, 8), rollout_trajectory_data=None
        )
        batch.metrics = None
        stored = {}
        _build_response("r", "p", 0, False, batch, stored=stored)
        self.assertEqual(stored, {})


class TestRolloutOutputStore(unittest.TestCase):
    def _store(self):
        backend = _MemoryBackend()
        store = RolloutOutputStore()
        args = types.SimpleNamespace(
            output_store_backend="memory", output_store_backend_extra_config=None
        )
        with patch("radix_transfer_plugins.create_backend", return_value=backend):
            store.initialize(args)
        return store, backend

    def test_disabled_without_a_backend(self):
        store = RolloutOutputStore()
        store.initialize(types.SimpleNamespace(output_store_backend="none"))
        self.assertFalse(store.enabled)

    def test_put_and_delete(self):
        store, backend = self._store()
        self.assertTrue(store.enabled)
        ref = store.put({"x": torch.ones(2, dtype=torch.bfloat16)})
        self.assertEqual(
            [(f.name, f.dtype, f.shape) for f in ref.fields], [("x", "bfloat16", (2,))]
        )
        self.assertIn(ref.key, backend.objects)
        store.delete(ref)
        self.assertEqual(backend.objects, {})
        store.delete(ref)
        store.close()
        self.assertFalse(store.enabled)

    def test_unsent_response_deletes_its_object(self):
        store, backend = self._store()
        ref = store.put({"x": torch.ones(2)})
        with patch.object(rollout_api, "rollout_output_store", store):
            parts = rollout_api._stream_parts([b"a", b"b"], ref)
            next(parts)
            parts.close()
        self.assertEqual(backend.objects, {})

    def test_sent_response_keeps_its_object(self):
        store, backend = self._store()
        ref = store.put({"x": torch.ones(2)})
        with patch.object(rollout_api, "rollout_output_store", store):
            self.assertEqual(
                list(rollout_api._stream_parts([b"a", b"b"], ref)), [b"a", b"b"]
            )
        self.assertIn(ref.key, backend.objects)


if __name__ == "__main__":
    unittest.main()
