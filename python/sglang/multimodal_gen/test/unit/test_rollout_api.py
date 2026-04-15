"""Unit tests for the rollout generate API (serialization, io_struct, rollout_api)."""

import types
import unittest

import torch

from sglang.multimodal_gen.runtime.entrypoints.post_training.utils import (
    _maybe_deserialize,
    _maybe_serialize,
    base64_to_tensor,
    tensor_to_base64,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch
from sglang.multimodal_gen.runtime.post_training.rl_dataclasses import (
    RolloutDebugTensors,
    RolloutDenoisingEnv,
    RolloutDitTrajectory,
    RolloutTrajectoryData,
)


class TestTensorToBase64Roundtrip(unittest.TestCase):

    def _roundtrip(self, t: torch.Tensor):
        encoded = tensor_to_base64(t)
        self.assertIsInstance(encoded, str)
        decoded = base64_to_tensor(encoded)
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
        encoded = tensor_to_base64(t)
        decoded = base64_to_tensor(encoded)
        self.assertTrue(torch.equal(t.cpu(), decoded))

    def test_non_contiguous(self):
        t = torch.randn(4, 6)[:, ::2]
        self.assertFalse(t.is_contiguous())
        self._roundtrip(t.contiguous())
        decoded = base64_to_tensor(tensor_to_base64(t))
        self.assertTrue(torch.equal(t.contiguous(), decoded))

    def test_grad_tensor_detaches(self):
        t = torch.randn(3, requires_grad=True)
        encoded = tensor_to_base64(t)
        decoded = base64_to_tensor(encoded)
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
        decoded = base64_to_tensor(result["data"])
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
            ),
        )
        _, _, env, dit_traj = _serialize_rollout_trajectory(
            rtd,
            serialized_dit_timesteps=_maybe_serialize(rtd.dit_trajectory.timesteps),
        )
        self.assertIsNotNone(env)
        self.assertIn("pos_cond_kwargs", env)
        self.assertNotIn("trajectory", env)
        self.assertIsNotNone(dit_traj)
        self.assertIn("latents", dit_traj)
        self.assertIn("timesteps", dit_traj)
        self.assertTrue(dit_traj["latents"]["__tensor__"])
        self.assertTrue(dit_traj["timesteps"]["__tensor__"])


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
        lp = base64_to_tensor(resp.rollout_log_probs["data"])
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
        lp0 = base64_to_tensor(resps[0].rollout_log_probs["data"])
        lp1 = base64_to_tensor(resps[1].rollout_log_probs["data"])
        self.assertEqual(lp0.shape, (T,))
        self.assertEqual(lp1.shape, (T,))
        g0 = base64_to_tensor(resps[0].generated_output["data"])
        g1 = base64_to_tensor(resps[1].generated_output["data"])
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
                ),
            ),
        )
        batch.metrics = self._make_metrics(1.0)
        resps = _build_response("r", "p", 0, True, batch)
        self.assertEqual(len(resps), B)
        self.assertIsNotNone(resps[0].dit_trajectory)
        self.assertIsNotNone(resps[1].dit_trajectory)
        ts0 = base64_to_tensor(resps[0].dit_trajectory["timesteps"]["data"])
        ts1 = base64_to_tensor(resps[1].dit_trajectory["timesteps"]["data"])
        self.assertEqual(ts0.shape, (T,))
        self.assertTrue(torch.equal(ts0, ts1))
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


if __name__ == "__main__":
    unittest.main()
