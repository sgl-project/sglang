# SPDX-License-Identifier: Apache-2.0
"""Exercise the real HTTP frontend, spawned worker and action WebSockets on a compute node."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from importlib.util import find_spec
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

requires_motubrain = unittest.skipUnless(
    find_spec("motubrain2"), "requires the optional Motubrain2 inference package"
)


def _json(url, payload=None, timeout=90):
    req = urllib.request.Request(
        url,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


class NativeOutputValidityTest(unittest.TestCase):
    def test_dit_cache_options_reach_native_policy_loader(self):
        from sglang.multimodal_gen.configs.pipeline_configs.motubrain2 import (
            Motubrain2PipelineConfig,
        )
        from sglang.multimodal_gen.runtime.models.motubrain2.policy import (
            Motubrain2PolicyModel,
        )

        options = dict(
            dit_cache_enabled=True,
            dit_cache_threshold=0.9,
            dit_cache_threshold_2=0.8,
            dit_cache_skip_steps=3,
            dit_cache_skip_steps_2=1,
            dit_cache_damping=0.1,
        )
        config = Motubrain2PipelineConfig(wan="wan", text_encoder="text", **options)
        module = SimpleNamespace(DiTCacheConfig=SimpleNamespace(for_recipe=object()))
        load = Mock(side_effect=StopIteration)
        policy_module = SimpleNamespace(RawPolicy=SimpleNamespace(from_options=load))
        with tempfile.TemporaryDirectory() as bundle:
            for name in ("model.safetensors", "config.json", "profile.json"):
                (Path(bundle) / name).touch()
            with (
                patch.dict(
                    sys.modules,
                    {
                        "motubrain2.inference.dit_cache": module,
                        "motubrain2.inference.policy": policy_module,
                    },
                ),
                patch.object(config, "load_serving_profile"),
                self.assertRaises(StopIteration),
            ):
                Motubrain2PolicyModel.from_pretrained(bundle, config)
            args = load.call_args.args[0]
            self.assertEqual({k: getattr(args, k) for k in options}, options)

    def test_optional_dependency_is_needed_only_when_loading_a_policy(self):
        from sglang.multimodal_gen.runtime.pipelines.motubrain2 import (
            Motubrain2Pipeline,
        )

        config = Motubrain2Pipeline.pipeline_config_cls(wan="wan", text_encoder="text")
        native_import = __import__

        def without_native(name, *args, **kwargs):
            if name.startswith("motubrain2."):
                raise ModuleNotFoundError("missing native package", name="motubrain2")
            return native_import(name, *args, **kwargs)

        with (
            patch("builtins.__import__", side_effect=without_native),
            self.assertRaisesRegex(ImportError, "Install the optional Motubrain2"),
        ):
            config.load_serving_profile("unused")

    def test_http_sampling_accepts_native_outputs_without_a_model(self):
        from sglang.multimodal_gen.configs.pipeline_configs.motubrain2 import (
            Motubrain2PipelineConfig,
        )
        from sglang.multimodal_gen.runtime.entrypoints.action.motubrain2 import (
            prepare_motubrain2_payload,
        )
        from sglang.multimodal_gen.runtime.entrypoints.action.protocol import (
            build_action_sampling_params,
        )

        config = Motubrain2PipelineConfig(
            image_keys=("head",),
            action_dim=2,
            output_action_dim=2,
            action_horizon=3,
            default_num_inference_steps=4,
        )
        server_args = SimpleNamespace(
            model_path="unused",
            pipeline_class_name="Motubrain2Pipeline",
            pipeline_config=config,
            output_path=None,
            comfyui_mode=False,
        )

        def build(payload, **connection):
            request = dict(payload)
            prepare_motubrain2_payload(request, **connection)
            return build_action_sampling_params(request, server_args)

        observation = {
            "images": {"head": [[[0, 0, 0]]]},
            "state": [0, 1],
            "connection_state": {"spoofed": True},
            "_motubrain2_connection_state": {"spoofed": True},
        }
        nested = {
            "input": {"task": "pick up the block", "observation": observation},
            "parameters": {"return_native_outputs": True},
        }
        flat = {
            **observation,
            "prompt": "pick up the block",
            "return_native_outputs": True,
            "enable_cuda_graph": True,
            "runtime": {"cuda_graph": False, "prefix_cache": False},
        }
        override = {**flat, "parameters": {"return_native_outputs": False}}
        for name, payload, native, cache in (
            ("nested", nested, True, True),
            ("flat", flat, True, False),
            ("override", override, False, False),
        ):
            with self.subTest(name):
                params = build(payload)
                self.assertIs(params.return_native_outputs, native)
                self.assertIs(params.enable_prefix_cache, cache)
                self.assertFalse(params.enable_cuda_graph)
                self.assertEqual(params.num_inference_steps, 4)
                req = SimpleNamespace(extra={})
                params.apply_request_extra(req)
                worker = req.extra["motubrain2"]
                self.assertIs(worker["observation"]["return_native_outputs"], native)
                self.assertNotIn("connection_state", worker)
                self.assertNotIn("connection_state", worker["observation"])
                self.assertNotIn("_motubrain2_connection_state", worker["observation"])

        state = {}
        params = build(nested, session_id="server-session", connection_state=state)
        req = SimpleNamespace(extra={})
        params.apply_request_extra(req)
        worker = req.extra["motubrain2"]
        self.assertIs(worker["connection_state"], state)
        self.assertEqual(worker["observation"]["session_id"], "server-session")
        self.assertNotIn("_motubrain2_connection_state", params.observation)
        self.assertNotIn("_motubrain2_connection_state", worker["observation"])

        with self.assertRaisesRegex(ValueError, "does not implement CUDA graphs"):
            build({**flat, "runtime": {}})
        with self.assertRaisesRegex(ValueError, "must be a boolean"):
            build({**flat, "return_native_outputs": "true"})

    def test_stage_caches_only_identical_native_inputs(self):
        import numpy as np
        import torch

        from sglang.multimodal_gen.configs.pipeline_configs.motubrain2 import (
            Motubrain2PipelineConfig,
        )
        from sglang.multimodal_gen.runtime.models.motubrain2.policy import (
            Motubrain2PolicyModel,
        )
        from sglang.multimodal_gen.runtime.pipelines.motubrain2 import (
            Motubrain2ActionStage,
        )

        config = Motubrain2PipelineConfig(
            image_keys=("head",), state_dim=2, output_action_dim=2, action_horizon=2
        )
        args = SimpleNamespace(
            pipeline_config=config,
            disable_conditioning_cache=False,
            conditioning_cache_max_size_mb=1,
        )
        model = SimpleNamespace(
            observation_key=Motubrain2PolicyModel.observation_key,
            encode_observation=Mock(return_value={"video": torch.zeros(1)}),
            predict=Mock(return_value=(torch.ones(2, 2), None)),
        )
        with patch(
            "sglang.multimodal_gen.runtime.pipelines_core.stages.base.get_global_server_args",
            return_value=args,
        ):
            stage = Motubrain2ActionStage(model, config)
        observation = {
            "prompt": "test",
            "images": {"head": np.zeros((4, 4, 3), dtype=np.uint8)},
            "state": [0, 1],
        }
        options = {
            "enable_prefix_cache": True,
            "output_format": "list",
            "return_timing": True,
        }
        request = SimpleNamespace(
            request_id="request",
            num_inference_steps=4,
            is_warmup=False,
            generator=None,
            metrics=None,
            extra={"motubrain2": {"observation": observation, "options": options}},
        )

        def infer():
            return stage.forward(request, args).output[0]

        self.assertFalse(infer()["cache"]["hit"])
        self.assertTrue(infer()["cache"]["hit"])
        observation["images"]["head"][0, 0, 0] = 1
        self.assertFalse(infer()["cache"]["hit"])
        observation["state"] = [2, 1]
        self.assertFalse(infer()["cache"]["hit"])
        options["enable_prefix_cache"] = False
        self.assertEqual(infer()["cache"]["mode"], "disabled")
        self.assertEqual(model.encode_observation.call_count, 4)
        self.assertEqual(model.predict.call_count, 5)

    @requires_motubrain
    def test_legacy_native_cache_uses_the_stored_recipe_for_auto_mode(self):
        from motubrain2.inference.sampling import sampling_options

        from sglang.multimodal_gen.runtime.models.motubrain2.policy import (
            _check_native_dit_cache,
        )

        with tempfile.TemporaryDirectory() as bundle:
            for solver, steps, enabled, primary, secondary, needs_update in (
                ("euler", 30, None, None, None, True),  # Base defaults on
                ("cm", 4, None, None, None, False),  # Distill defaults off
                ("euler", 16, None, None, None, False),  # unrelated Euler recipe
                ("cm", 4, True, None, None, True),
                ("cm", 4, None, 0.9, None, True),
                ("cm", 4, None, None, 0.8, True),
                ("euler", 30, False, 0.9, 0.8, False),
            ):
                recipe = dict(solver=solver, steps=steps, video_shift=6, action_shift=1)
                (Path(bundle) / "sampling.json").write_text(json.dumps(recipe))
                self.assertEqual(sampling_options(bundle)["steps"], steps)
                for module in (None, SimpleNamespace(DiTCacheConfig=object)):
                    args = SimpleNamespace(
                        model=bundle,
                        steps=2,
                        dit_cache_enabled=enabled,
                        dit_cache_threshold=primary,
                        dit_cache_threshold_2=secondary,
                    )
                    with (
                        self.subTest(recipe=recipe, enabled=enabled, module=module),
                        patch.dict(
                            sys.modules, {"motubrain2.inference.dit_cache": module}
                        ),
                    ):
                        if needs_update:
                            with self.assertRaisesRegex(
                                ImportError, "DiTCacheConfig.for_recipe"
                            ):
                                _check_native_dit_cache(args)
                        else:
                            _check_native_dit_cache(args)
                            self.assertIsNone(args.dit_cache_threshold)
                            self.assertIsNone(args.dit_cache_threshold_2)

    @requires_motubrain
    def test_explicit_rtc_accepts_anchor_without_state_and_respects_native_delay(self):
        import torch

        from sglang.multimodal_gen.configs.pipeline_configs.motubrain2 import (
            Motubrain2PipelineConfig,
        )
        from sglang.multimodal_gen.configs.sample.motubrain2 import (
            validate_observation_options,
        )

        cfg = Motubrain2PipelineConfig(action_horizon=2, output_action_dim=2)
        base = {"anchor": [1, 2], "future": [[3, 4], [5, 6], [7, 8]]}
        for delay, count in ((0, 0), (1, 1), (100, 2)):
            rtc = validate_observation_options({"rtc": dict(base, delay=delay)}, cfg)
            anchor, locked = rtc.prefix(2, 2)
            self.assertTrue(torch.equal(anchor, torch.tensor([1, 2])))
            self.assertTrue(torch.equal(locked, torch.tensor(base["future"])[:count]))
        empty = validate_observation_options(
            {"state": [0, 0], "rtc": {"future": [], "delay": 0}}, cfg
        )
        self.assertEqual(tuple(empty.prefix(2, 2)[1].shape), (0, 2))
        for bad in (
            {"delay": False},
            {"delay": -1},
            {"delay": 0, "future": []},
            {"delay": 2, "anchor": [1]},
            {"delay": 2, "future": [[1, float("nan")]]},
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate_observation_options({"rtc": {**base, **bad}}, cfg)

    def test_session_rng_advances_only_after_a_successful_implicit_request(self):
        import torch

        from sglang.multimodal_gen.configs.pipeline_configs.motubrain2 import (
            Motubrain2PipelineConfig,
        )
        from sglang.multimodal_gen.runtime.models.motubrain2.policy import (
            Motubrain2PolicyModel,
        )

        class Policy(SimpleNamespace):
            def predict(self, observation, *, rtc=None):
                actions = torch.rand(2, 2, generator=self.generator)
                if observation.get("fail"):
                    raise ValueError("failure after drawing noise")
                return {"decoded_action": actions, "decoded_valid": actions.isfinite()}

        policy = Policy(
            device="cpu",
            model=torch.nn.Linear(1, 1),
            sampling={"steps": 1},
            processor=SimpleNamespace(
                action_steps=2,
                config=SimpleNamespace(action_dim=2),
                actions=SimpleNamespace(wire_dim=2),
                sparse_history=None,
            ),
        )
        config = Motubrain2PipelineConfig(text_cache_size=0, max_sessions=2)
        model = Motubrain2PolicyModel("tiny", config, policy)

        def request(
            session_id=None,
            *,
            seed=None,
            reset=False,
            fail=False,
            connection_state=None,
        ):
            from sglang.multimodal_gen.configs.sample.motubrain2 import (
                Motubrain2Observation,
            )

            observation = Motubrain2Observation(
                inputs={"fail": fail}, session_id=session_id, seed=seed, reset=reset
            )
            return model.predict(
                observation, {}, num_steps=1, connection_state=connection_state
            )[0]

        expected = torch.Generator().manual_seed(0)
        first = torch.rand(2, 2, generator=expected)
        second = torch.rand(2, 2, generator=expected)
        self.assertTrue(torch.equal(request("left"), first))
        self.assertTrue(torch.equal(request("right"), first))
        socket = {}
        self.assertTrue(torch.equal(request("left", connection_state=socket), first))
        self.assertTrue(
            torch.equal(request("left", seed=0, connection_state=socket), first)
        )
        with self.assertRaisesRegex(ValueError, "after drawing noise"):
            request("left", reset=True, fail=True, connection_state=socket)
        self.assertTrue(torch.equal(request("left", connection_state=socket), second))
        self.assertTrue(torch.equal(request("socket-only", connection_state={}), first))
        self.assertEqual(set(model.sessions), {"left", "right"})
        self.assertTrue(torch.equal(request("left", seed=0), first))
        for reset in (False, True):
            with self.assertRaisesRegex(ValueError, "after drawing noise"):
                request("left", reset=reset, fail=True)
        self.assertTrue(torch.equal(request("left"), second))
        self.assertTrue(torch.equal(request("right"), second))
        self.assertTrue(torch.equal(request("left", reset=True), first))
        self.assertTrue(torch.equal(request("other"), first))
        self.assertTrue(torch.equal(request("right"), first))  # evicted episode
        self.assertTrue(torch.equal(request(), first))
        self.assertTrue(torch.equal(request(), first))

    @requires_motubrain
    def test_partial_wire_chunk_keeps_its_mask_and_is_not_an_rtc_prefix(self):
        import torch
        from safetensors.torch import load

        from sglang.multimodal_gen.configs.pipeline_configs.motubrain2 import (
            Motubrain2PipelineConfig,
        )
        from sglang.multimodal_gen.runtime.models.motubrain2.policy import (
            Motubrain2PolicyModel,
        )

        processor = SimpleNamespace(
            action_steps=2,
            config=SimpleNamespace(action_dim=3),
            actions=SimpleNamespace(wire_dim=2),
            sparse_history=None,
        )
        mask = torch.ones(2, 2, dtype=torch.bool)
        sample = {
            "video": torch.ones(1, 4),
            "action": torch.ones(2, 3),
            "decoded_action": torch.ones(2, 2),
            "decoded_valid": mask,
        }
        observed_rtc = []

        def predict(observation, *, rtc=None):
            observed_rtc.append(rtc)
            return sample

        policy = SimpleNamespace(
            device="cpu",
            model=torch.nn.Linear(1, 1),
            sampling={"steps": 1},
            processor=processor,
            predict=predict,
        )
        config = Motubrain2PipelineConfig(rtc_enabled=True, text_cache_size=0)
        model = Motubrain2PolicyModel("tiny", config, policy)
        from sglang.multimodal_gen.configs.sample.motubrain2 import (
            Motubrain2Observation,
        )

        obs = Motubrain2Observation(inputs={}, session_id="left")

        def request(connection_state):
            return model.predict(
                obs, {}, num_steps=1, connection_state=connection_state
            )

        for state in (None, {}):  # HTTP session and WebSocket continuation
            obs.return_native_outputs = False
            self.assertIsNone(request(state)[1])
            mask[0, 1] = False
            with self.assertRaisesRegex(ValueError, "invalid controller"):
                request(state)
            obs.return_native_outputs = True
            _, native_outputs = request(state)
            result = load(base64.b64decode(native_outputs["data"]))
            self.assertTrue(torch.equal(result["decoded_valid"], mask))
            self.assertEqual(result["decoded_valid"].dtype, torch.bool)
            self.assertIsNotNone(observed_rtc[-1])
            mask.fill_(True)
            request(state)
            self.assertIsNone(observed_rtc[-1])


@requires_motubrain
@unittest.skipUnless(
    os.getenv("MOTUBRAIN2_RUN_SERVE_E2E") == "1",
    "set MOTUBRAIN2_RUN_SERVE_E2E=1 on a compute node with SGLang Diffusion",
)
class TinyServingTest(unittest.TestCase):
    def test_profile_defaults_and_rtc_across_transports(self):
        import numpy as np
        from safetensors import safe_open
        from safetensors.torch import load

        from sglang.multimodal_gen.runtime.entrypoints.action.protocol import (
            pack_msgpack,
            unpack_msgpack,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "motubrain2-serving"
            subprocess.run(
                [sys.executable, "-m", "motubrain2.tools.make_tiny_data", str(root)],
                check=True,
            )
            (root / "model").rename(root / "policy")
            profile = json.loads((root / "profile.json").read_text())
            profile["action"]["fields"][0]["width"] = 3
            (root / "profile.json").write_text(json.dumps(profile))
            (root / "policy" / "sampling.json").write_text(
                '{"solver":"euler","steps":2}'
            )
            (root / "model_index.json").write_text(
                '{"_class_name":"Motubrain2Pipeline"}'
            )
            config = {
                "wan": str(root / "wan"),
                "text_encoder": str(root / "text"),
                "rtc_enabled": True,
                "rtc_execute_prefix": 1,
                "rtc_delay": 2,
            }
            config_path = root / "serve.json"
            config_path.write_text(json.dumps(config))
            # Run the reference outside the server so its PyTorch attention
            # backend settings are independent of the SGLang worker.
            prediction = root / "reference.safetensors"
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "motubrain2.inference",
                    "--model",
                    str(root / "policy"),
                    "--profile",
                    str(root / "profile.json"),
                    "--wan",
                    str(root / "wan"),
                    "--text-encoder",
                    str(root / "text"),
                    "--input",
                    str(root / "observation.json"),
                    "--output",
                    str(prediction),
                    "--bf16",
                    "--seed",
                    "13",
                ],
                check=True,
            )
            with safe_open(prediction, framework="pt", device="cpu") as tensors:
                expected_outputs = {
                    key: tensors.get_tensor(key) for key in tensors.keys()
                }
                reference = expected_outputs["decoded_action"].float().numpy()
            with contextlib.ExitStack() as sockets:
                ports = []
                for _ in range(3):
                    sock = sockets.enter_context(socket.socket())
                    sock.bind(("127.0.0.1", 0))
                    ports.append(sock.getsockname()[1])
                port, master_port, scheduler_port = ports
            env = dict(os.environ)
            env.pop("SGLANG_EXTERNAL_MODEL_PACKAGE", None)
            server = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "sglang.cli.main",
                    "serve",
                    str(root),
                    "--model-type",
                    "diffusion",
                    "--pipeline-class-name",
                    "Motubrain2Pipeline",
                    "--pipeline-config-path",
                    str(config_path),
                    "--warmup-mode",
                    "off",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "--master-port",
                    str(master_port),
                    "--scheduler-port",
                    str(scheduler_port),
                    "--num-gpus",
                    "1",
                ],
                env=env,
                start_new_session=True,
            )
            try:
                base = f"http://127.0.0.1:{port}/v1/actions"
                # Cold imports of both spawned processes on a cluster-mounted
                # Python environment can take several minutes.
                deadline = time.monotonic() + 600
                while True:
                    if server.poll() is not None:
                        self.fail(
                            f"sglang serve exited with {server.returncode} during startup"
                        )
                    try:
                        metadata = _json(base + "/metadata", timeout=3)
                        break
                    except (urllib.error.URLError, TimeoutError):
                        if time.monotonic() >= deadline:
                            self.fail("timed out waiting for sglang serve")
                        time.sleep(1)
                self.assertEqual(metadata["model_impl"], "real")
                self.assertTrue(metadata["input"]["http_session_id_required"])
                self.assertFalse(metadata["capabilities"]["cuda_graph"])
                self.assertEqual(metadata["input"]["image_keys"], ["front"])
                self.assertEqual(metadata["output"]["action_horizon"], 4)
                self.assertEqual(metadata["output"]["action_dim"], 3)
                self.assertEqual(metadata["defaults"]["num_inference_steps"], 2)

                pixels = base64.b64encode(
                    (root / "observation.png").read_bytes()
                ).decode()
                observation = {
                    "prompt": "move the red block right",
                    "images": {"front": {"b64_json": pixels}},
                }

                def generate(
                    session, seed=None, *, native=False, reset=False, rtc=None
                ):
                    payload = {
                        "input": {
                            "observation": dict(
                                observation, session_id=session, reset=reset
                            )
                        },
                        "parameters": {} if seed is None else {"seed": seed},
                    }
                    if rtc is not None:
                        payload["input"]["observation"]["rtc"] = rtc
                    if native:
                        payload["parameters"]["return_native_outputs"] = True
                    result = _json(base + "/generations", payload)
                    if native:
                        response = result["extras"]["native_outputs"]
                        self.assertEqual(response["format"], "safetensors")
                        self.assertEqual(response["encoding"], "base64")
                        actual = load(base64.b64decode(response["data"], validate=True))
                        self.assertEqual(set(actual), set(expected_outputs))
                        for name, tensor in actual.items():
                            expected = expected_outputs[name]
                            self.assertEqual(tensor.dtype, expected.dtype)
                            np.testing.assert_allclose(
                                tensor.float(), expected.float(), atol=1e-6, rtol=0
                            )
                    else:
                        self.assertNotIn("native_outputs", result.get("extras", {}))
                    self.assertEqual(result["usage"]["denoise_steps"], 2)
                    action = result["data"][0]["action"]
                    self.assertEqual(action["shape"], [4, 3])
                    return np.asarray(action["values"])

                first = generate("left", 13, native=True)
                np.testing.assert_allclose(first, reference, atol=1e-6, rtol=0)
                other = generate("right", 29)
                self.assertFalse(np.allclose(first, other))
                np.testing.assert_allclose(
                    generate("left", 29)[:2], first[1:3], atol=1e-5
                )
                np.testing.assert_allclose(
                    generate("right", 13)[:2], other[1:3], atol=1e-5
                )
                implicit_first = generate("implicit")
                implicit_second = generate("implicit")
                np.testing.assert_array_equal(
                    generate("implicit", reset=True), implicit_first
                )
                anchored = {
                    "anchor": first[0].tolist(),
                    "future": first[1:].tolist(),
                    "delay": 2,
                }
                anchored_chunk = generate("right", 29, rtc=anchored)
                np.testing.assert_allclose(anchored_chunk[:2], first[1:3], atol=1e-5)
                zero_delay = {**anchored, "delay": 0}
                np.testing.assert_allclose(
                    generate("right", 13, rtc=zero_delay), first, atol=1e-5
                )
                invalid_requests = {
                    "missing RTC session": {"input": {"observation": observation}},
                    "wrong horizon": {"parameters": {"action_horizon": 32}},
                    "wrong width": {"parameters": {"action_dim": 14}},
                    "external action noise": {
                        "input": {
                            "observation": dict(
                                observation, session_id="left", noise=[[0.0]]
                            )
                        }
                    },
                    "unsupported CFG": {"parameters": {"guidance_scale": 3}},
                    "unsupported CUDA graph": {"runtime": {"cuda_graph": True}},
                    "masked camera": {
                        "input": {
                            "observation": dict(
                                observation,
                                session_id="left",
                                image_masks={"front": False},
                            )
                        }
                    },
                }
                for case, override in invalid_requests.items():
                    payload = {
                        "input": {"observation": dict(observation, session_id="left")},
                        **override,
                    }
                    with self.subTest(case=case):
                        with self.assertRaises(urllib.error.HTTPError) as error:
                            _json(base + "/generations", payload)
                        self.assertEqual(error.exception.code, 400)
                        error.exception.close()

                async def check_connections():
                    from websockets import connect

                    async def receive(ws):
                        reply = unpack_msgpack(await ws.recv())
                        self.assertNotIn("_motubrain2_connection_state", reply)
                        return reply

                    uri = f"ws://127.0.0.1:{port}/openpi/policy"
                    async with (
                        connect(uri) as left,
                        connect(f"ws://127.0.0.1:{port}/v1/actions/realtime") as right,
                    ):
                        for ws in (left, right):
                            self.assertEqual(
                                unpack_msgpack(await ws.recv())["model_impl"], "real"
                            )
                        await left.send(
                            pack_msgpack(dict(observation, seed=13, session_id="left"))
                        )
                        l1 = (await receive(left))["actions"]
                        await right.send(
                            pack_msgpack(dict(observation, seed=29, session_id="left"))
                        )
                        r1 = (await receive(right))["data"][0]["action"]["values"]
                        await left.send(
                            pack_msgpack(dict(observation, seed=29, session_id="left"))
                        )
                        l2 = (await receive(left))["actions"]
                        np.testing.assert_allclose(l1, first, atol=1e-5)
                        np.testing.assert_allclose(r1, other, atol=1e-5)
                        np.testing.assert_allclose(l2[:2], l1[1:3], atol=1e-5)
                        await left.send(
                            pack_msgpack(dict(observation, seed=29, rtc=anchored))
                        )
                        explicit_ws = (await receive(left))["actions"]
                        np.testing.assert_array_equal(explicit_ws, anchored_chunk)
                        for reset, expected in (
                            (True, implicit_first),
                            (False, implicit_second),
                        ):
                            await left.send(
                                pack_msgpack(dict(observation, reset=reset))
                            )
                            actual = (await receive(left))["actions"]
                            np.testing.assert_allclose(
                                actual, expected, rtol=0, atol=1e-6
                            )

                    async with connect(uri) as reconnected:
                        await receive(reconnected)  # metadata
                        await reconnected.send(
                            pack_msgpack(dict(observation, session_id="left"))
                        )
                        fresh = (await receive(reconnected))["actions"]
                        np.testing.assert_allclose(
                            fresh, implicit_first, rtol=0, atol=1e-6
                        )

                asyncio.run(asyncio.wait_for(check_connections(), timeout=120))
            finally:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(server.pid, signal.SIGTERM)
                try:
                    server.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(server.pid, signal.SIGKILL)
                    server.wait()


if __name__ == "__main__":
    unittest.main()
