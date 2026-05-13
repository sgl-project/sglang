"""Integration tests for RayEngine and Ray HTTP server (requires GPU + Ray).

Tests the Ray actor scheduler backend:
  - Offline inference via Engine(use_ray=True) inside a Ray actor on a placement group
  - Data parallel (DP) and DP attention support
  - Error paths in RayEngine._launch_scheduler_processes()
  - HTTP server launched via --use-ray flag

Usage:
    # 1-GPU tests
    python -m pytest test/manual/test_ray_engine.py::TestRayEngineOfflineTP1 -v -s
    python -m pytest test/manual/test_ray_engine.py::TestRayEngineErrors -v -s
    python -m pytest test/manual/test_ray_engine.py::TestRayHTTPServerTP1 -v -s

    # 2-GPU tests
    python -m pytest test/manual/test_ray_engine.py::TestRayEngineOfflineTP2 -v -s
    python -m pytest test/manual/test_ray_engine.py::TestRayEngineOfflinePP2 -v -s
    python -m pytest test/manual/test_ray_engine.py::TestRayEngineOfflineDP2 -v -s
    python -m pytest test/manual/test_ray_engine.py::TestRayEngineOfflineDPAttention -v -s
"""

from __future__ import annotations

import os
import time
import unittest

import torch

from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST

# Allow overriding the model via env var for environments without gated access
_MODEL = os.environ.get("SGLANG_TEST_MODEL", DEFAULT_SMALL_MODEL_NAME_FOR_TEST)

# DP attention requires a model whose num_kv_heads divides evenly across the
# attention-TP dimension.  Qwen2.5-0.5B (kv_heads=2, attn_heads=14) hits a
# shape mismatch in the KV cache, so we use a larger model here.
_DP_ATTN_MODEL = os.environ.get("SGLANG_TEST_DP_ATTN_MODEL", "Qwen/Qwen3-8B")

try:
    import ray
    from ray.runtime_env import RuntimeEnv
    from ray.util.placement_group import placement_group
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

    # Prevent Ray from overriding CUDA_VISIBLE_DEVICES so that all GPUs
    # remain visible inside actors regardless of num_gpus allocation.
    _env_vars = {"RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1"}
    if os.environ.get("HF_TOKEN"):
        _env_vars["HF_TOKEN"] = os.environ["HF_TOKEN"]
    _RAY_RUNTIME_ENV = RuntimeEnv(env_vars=_env_vars)
    _has_ray = True
except ImportError:
    _has_ray = False
    _RAY_RUNTIME_ENV = None


_NUM_GPUS = torch.cuda.device_count()

_SAMPLING_PARAMS = {"max_new_tokens": 32, "temperature": 0.0}

_PROMPTS = [
    "The capital of France is",
    "Explain quantum computing in simple terms:",
    "Write a haiku about programming:",
    "What is 2 + 2?",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_engine_on_pg(
    tp_size, pp_size=1, dp_size=1, model=_MODEL, extra_kwargs=None
):
    """Create an EngineActor on a placement group and wait for it to be ready.

    Returns (engine_actor, placement_group).
    """

    @ray.remote
    class EngineActor:
        def __init__(self, **kwargs):
            from sglang.srt.ray.engine import RayEngine

            self.engine = RayEngine(**kwargs)

        def is_ready(self):
            return True

        def generate(self, prompt, sampling_params):
            return self.engine.generate(prompt=prompt, sampling_params=sampling_params)

        def shutdown(self):
            if self.engine:
                self.engine.shutdown()
                self.engine = None

    enable_dp_attention = (extra_kwargs or {}).get("enable_dp_attention", False)
    if enable_dp_attention:
        # DP attention folds DP into TP — total GPUs = tp_size * pp_size
        total_gpus = tp_size * pp_size
    else:
        total_gpus = dp_size * tp_size * pp_size
    pg = placement_group(
        [{"CPU": 1, "GPU": total_gpus}],
        strategy="STRICT_PACK",
    )
    ray.get(pg.ready())

    kwargs = dict(
        model_path=model,
        tp_size=tp_size,
        pp_size=pp_size,
        dp_size=dp_size,
    )
    if extra_kwargs:
        kwargs.update(extra_kwargs)

    actor = EngineActor.options(
        num_cpus=1,
        num_gpus=0,
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_bundle_index=0,
        ),
    ).remote(**kwargs)

    ray.get(actor.is_ready.remote(), timeout=600)
    return actor, pg


def _cleanup(actor, pg):
    """Shutdown engine actor and remove placement group."""
    try:
        ray.get(actor.shutdown.remote(), timeout=60)
    except Exception:
        pass
    try:
        ray.util.remove_placement_group(pg)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Tests: Offline TP=1
# ---------------------------------------------------------------------------


@unittest.skipUnless(_has_ray, "ray is not installed")
@unittest.skipUnless(_NUM_GPUS >= 1, "requires at least 1 GPU")
class TestRayEngineOfflineTP1(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not ray.is_initialized():
            ray.init(log_to_driver=True, runtime_env=_RAY_RUNTIME_ENV)
        cls.actor, cls.pg = _create_engine_on_pg(tp_size=1)

    @classmethod
    def tearDownClass(cls):
        _cleanup(cls.actor, cls.pg)
        ray.shutdown()

    def test_offline_generate(self):
        result = ray.get(
            self.actor.generate.remote("The capital of France is", _SAMPLING_PARAMS)
        )
        self.assertIn("text", result)
        self.assertGreater(len(result["text"]), 0)
        print(f"Generated: {result['text'][:200]}")

    def test_batch_generate(self):
        for prompt in _PROMPTS:
            result = ray.get(self.actor.generate.remote(prompt, _SAMPLING_PARAMS))
            self.assertIn("text", result)
            self.assertGreater(len(result["text"]), 0, f"Empty output for: {prompt}")

    def test_deterministic(self):
        prompt = "The meaning of life is"
        r1 = ray.get(self.actor.generate.remote(prompt, _SAMPLING_PARAMS))
        r2 = ray.get(self.actor.generate.remote(prompt, _SAMPLING_PARAMS))
        self.assertEqual(r1["text"], r2["text"])


# ---------------------------------------------------------------------------
# Tests: Offline TP=2
# ---------------------------------------------------------------------------


@unittest.skipUnless(_has_ray, "ray is not installed")
@unittest.skipUnless(_NUM_GPUS >= 2, "requires at least 2 GPUs")
class TestRayEngineOfflineTP2(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not ray.is_initialized():
            ray.init(log_to_driver=True, runtime_env=_RAY_RUNTIME_ENV)
        cls.actor, cls.pg = _create_engine_on_pg(tp_size=2)

    @classmethod
    def tearDownClass(cls):
        _cleanup(cls.actor, cls.pg)
        ray.shutdown()

    def test_offline_generate_tp2(self):
        result = ray.get(
            self.actor.generate.remote("The capital of France is", _SAMPLING_PARAMS)
        )
        self.assertIn("text", result)
        self.assertGreater(len(result["text"]), 0)
        print(f"Generated (TP=2): {result['text'][:200]}")

    def test_batch_generate_tp2(self):
        for prompt in _PROMPTS:
            result = ray.get(self.actor.generate.remote(prompt, _SAMPLING_PARAMS))
            self.assertIn("text", result)
            self.assertGreater(len(result["text"]), 0, f"Empty output for: {prompt}")


# ---------------------------------------------------------------------------
# Tests: Offline PP=2
# ---------------------------------------------------------------------------


@unittest.skipUnless(_has_ray, "ray is not installed")
@unittest.skipUnless(_NUM_GPUS >= 2, "requires at least 2 GPUs")
class TestRayEngineOfflinePP2(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not ray.is_initialized():
            ray.init(log_to_driver=True, runtime_env=_RAY_RUNTIME_ENV)
        cls.actor, cls.pg = _create_engine_on_pg(tp_size=1, pp_size=2)

    @classmethod
    def tearDownClass(cls):
        _cleanup(cls.actor, cls.pg)
        ray.shutdown()

    def test_offline_generate_pp2(self):
        result = ray.get(
            self.actor.generate.remote("The capital of France is", _SAMPLING_PARAMS)
        )
        self.assertIn("text", result)
        self.assertGreater(len(result["text"]), 0)
        print(f"Generated (PP=2): {result['text'][:200]}")

    def test_batch_generate_pp2(self):
        for prompt in _PROMPTS:
            result = ray.get(self.actor.generate.remote(prompt, _SAMPLING_PARAMS))
            self.assertIn("text", result)
            self.assertGreater(len(result["text"]), 0, f"Empty output for: {prompt}")


# ---------------------------------------------------------------------------
# Tests: Error paths
# ---------------------------------------------------------------------------


@unittest.skipUnless(_has_ray, "ray is not installed")
@unittest.skipUnless(_NUM_GPUS >= 2, "requires at least 2 GPUs")
class TestRayEngineOfflineDP2(unittest.TestCase):
    """Test Ray engine with dp_size=2, tp_size=1."""

    @classmethod
    def setUpClass(cls):
        if not ray.is_initialized():
            ray.init(log_to_driver=True, runtime_env=_RAY_RUNTIME_ENV)
        cls.actor, cls.pg = _create_engine_on_pg(tp_size=1, dp_size=2)

    @classmethod
    def tearDownClass(cls):
        _cleanup(cls.actor, cls.pg)
        ray.shutdown()

    def test_offline_generate_dp2(self):
        result = ray.get(
            self.actor.generate.remote("The capital of France is", _SAMPLING_PARAMS)
        )
        self.assertIn("text", result)
        self.assertGreater(len(result["text"]), 0)
        print(f"Generated (DP=2): {result['text'][:200]}")

    def test_batch_generate_dp2(self):
        for prompt in _PROMPTS:
            result = ray.get(self.actor.generate.remote(prompt, _SAMPLING_PARAMS))
            self.assertIn("text", result)
            self.assertGreater(len(result["text"]), 0, f"Empty output for: {prompt}")


# ---------------------------------------------------------------------------
# Tests: Offline DP Attention (dp=2, tp=2)
# ---------------------------------------------------------------------------


@unittest.skipUnless(_has_ray, "ray is not installed")
@unittest.skipUnless(_NUM_GPUS >= 2, "requires at least 2 GPUs")
class TestRayEngineOfflineDPAttention(unittest.TestCase):
    """Test Ray engine with dp_size=2, tp_size=2, enable_dp_attention=True."""

    @classmethod
    def setUpClass(cls):
        if not ray.is_initialized():
            ray.init(log_to_driver=True, runtime_env=_RAY_RUNTIME_ENV)
        cls.actor, cls.pg = _create_engine_on_pg(
            tp_size=2,
            dp_size=2,
            model=_DP_ATTN_MODEL,
            extra_kwargs={
                "enable_dp_attention": True,
                "disable_cuda_graph": True,
                "port": 31500,
            },
        )

    @classmethod
    def tearDownClass(cls):
        _cleanup(cls.actor, cls.pg)
        ray.shutdown()

    def test_offline_generate_dp_attention(self):
        result = ray.get(
            self.actor.generate.remote("The capital of France is", _SAMPLING_PARAMS)
        )
        self.assertIn("text", result)
        self.assertGreater(len(result["text"]), 0)
        print(f"Generated (DP-Attention): {result['text'][:200]}")

    def test_batch_generate_dp_attention(self):
        for prompt in _PROMPTS:
            result = ray.get(self.actor.generate.remote(prompt, _SAMPLING_PARAMS))
            self.assertIn("text", result)
            self.assertGreater(len(result["text"]), 0, f"Empty output for: {prompt}")


# ---------------------------------------------------------------------------
# Tests: Error paths
# ---------------------------------------------------------------------------


@unittest.skipUnless(_has_ray, "ray is not installed")
@unittest.skipUnless(_NUM_GPUS >= 1, "requires at least 1 GPU")
class TestRayEngineErrors(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not ray.is_initialized():
            ray.init(log_to_driver=True, runtime_env=_RAY_RUNTIME_ENV)

    @classmethod
    def tearDownClass(cls):
        ray.shutdown()

    def test_missing_placement_group_raises(self):
        """RayEngine without a placement group should raise RuntimeError."""

        @ray.remote(num_gpus=1)
        def _try_create_without_pg():
            from sglang.srt.ray.engine import RayEngine

            try:
                RayEngine(
                    model_path=_MODEL,
                    tp_size=1,
                    use_ray=True,
                )
                return None
            except RuntimeError as e:
                return str(e)

        error_msg = ray.get(_try_create_without_pg.remote(), timeout=120)
        self.assertIsNotNone(
            error_msg, "Expected RuntimeError but RayEngine created OK"
        )
        self.assertIn("placement group", error_msg.lower())


# ---------------------------------------------------------------------------
# Tests: HTTP server
# ---------------------------------------------------------------------------


@unittest.skipUnless(_has_ray, "ray is not installed")
@unittest.skipUnless(_NUM_GPUS >= 1, "requires at least 1 GPU")
class TestRayHTTPServerTP1(unittest.TestCase):
    """Test the Ray HTTP server path (launch_server.py --use-ray).

    Launches the server inside a Ray task on a placement group (mirrors
    examples/anyscale/driver_online.py) and sends HTTP requests to it.
    """

    @classmethod
    def setUpClass(cls):
        import requests as req_lib

        if not ray.is_initialized():
            ray.init(log_to_driver=True, runtime_env=_RAY_RUNTIME_ENV)

        cls.port = 30100
        cls.pg = placement_group(
            [{"CPU": 1, "GPU": 1}],
            strategy="STRICT_PACK",
        )
        ray.get(cls.pg.ready())

        pg_strategy = PlacementGroupSchedulingStrategy(
            placement_group=cls.pg,
            placement_group_bundle_index=0,
        )

        # Resolve the node IP where the server will run
        @ray.remote(num_cpus=0, num_gpus=0)
        def _get_ip():
            return ray.util.get_node_ip_address()

        cls.node_ip = ray.get(_get_ip.options(scheduling_strategy=pg_strategy).remote())
        cls.base_url = f"http://{cls.node_ip}:{cls.port}"

        # Launch server as a Ray task (blocks until server exits)
        @ray.remote
        def _launch(**kwargs):
            from sglang.srt.ray.http_server import launch_server
            from sglang.srt.server_args import ServerArgs

            launch_server(ServerArgs(**kwargs))

        cls.server_ref = _launch.options(
            num_cpus=1,
            num_gpus=0,
            scheduling_strategy=pg_strategy,
        ).remote(
            model_path=_MODEL,
            tp_size=1,
            port=cls.port,
            host="0.0.0.0",
            use_ray=True,
        )

        # Wait for health check
        t0 = time.time()
        timeout = 600
        healthy = False
        while time.time() - t0 < timeout:
            ready, _ = ray.wait([cls.server_ref], timeout=0)
            if ready:
                try:
                    ray.get(cls.server_ref)
                except Exception as e:
                    raise RuntimeError(f"Server task crashed: {e}") from e
                raise RuntimeError("Server task exited before becoming healthy")
            try:
                if req_lib.get(f"{cls.base_url}/health", timeout=5).status_code == 200:
                    healthy = True
                    break
            except req_lib.exceptions.RequestException:
                pass
            time.sleep(3)

        if not healthy:
            ray.cancel(cls.server_ref, force=True)
            raise RuntimeError(f"Server did not become healthy within {timeout}s")

    @classmethod
    def tearDownClass(cls):
        try:
            ray.cancel(cls.server_ref, force=True)
        except Exception:
            pass
        try:
            ray.util.remove_placement_group(cls.pg)
        except Exception:
            pass
        ray.shutdown()

    def test_health_endpoint(self):
        import requests

        resp = requests.get(f"{self.base_url}/health", timeout=10)
        self.assertEqual(resp.status_code, 200)

    def test_generate_endpoint(self):
        import requests

        resp = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": _SAMPLING_PARAMS,
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        self.assertIn("text", data)
        self.assertGreater(len(data["text"]), 0)
        print(f"HTTP response: {data['text'][:200]}")

    def test_generate_multiple(self):
        import requests

        for prompt in _PROMPTS:
            resp = requests.post(
                f"{self.base_url}/generate",
                json={
                    "text": prompt,
                    "sampling_params": _SAMPLING_PARAMS,
                },
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            self.assertIn("text", data)
            self.assertGreater(len(data["text"]), 0, f"Empty output for: {prompt}")


if __name__ == "__main__":
    unittest.main()
