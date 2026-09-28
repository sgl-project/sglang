"""Manual XPU distributed refit coverage using the Llama-3.2-1B model pair.

Run from the repository root, selecting a device tier that fits the visible cards:
	ZE_AFFINITY_MASK=2,3 python -m pytest \
		test/manual/rl/test_rl_xpu_basics.py -k tp1dp1 -vv

The registered harness supplies the full-model transfer, sampled weight checks,
tied-weight checks, and original three-second timing assertions. Its server flow
waits for generation before pausing, so these cases do not prove live refitting.
"""

import gc
import json
import shutil
import sys
import tempfile
import unittest
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForCausalLM

from sglang.srt.configs.model_config import get_num_indexer_layers
from sglang.srt.state_capturer.indexer_topk import extract_indexer_topk_from_meta_info
from sglang.srt.state_capturer.routed_experts import extract_routed_experts_from_meta_info
from sglang.srt.utils import get_device_count, is_xpu, kill_process_tree
from sglang.test.test_utils import (
	DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
	DEFAULT_SMALL_MODEL_NAME_FOR_TEST_BASE,
	DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
	CustomTestCase,
	find_available_port,
	popen_launch_server,
)

_TEST_ROOT = str(Path(__file__).resolve().parents[2])
if _TEST_ROOT not in sys.path:
	sys.path.insert(0, _TEST_ROOT)

from registered.rl.test_update_weights_from_disk_blackwell import (
	UpdateWeightsFromDiskBase,
)


@unittest.skipUnless(is_xpu(), "XPU-only distributed weight-update tests")
class TestRefitFromDistributedXPU(CustomTestCase):
	"""Explicit Engine/Server cases; each needs 1 + TP * DP visible devices."""

	@classmethod
	def setUpClass(cls):
		if get_device_count() < 2:
			raise unittest.SkipTest("At least two XPU devices are required")

		test_root = str(Path(__file__).resolve().parents[2])
		if test_root not in sys.path:
			sys.path.insert(0, test_root)
		from registered.rl import test_update_weights_from_distributed

		cls._harness = test_update_weights_from_distributed
		cls.model_path = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
		model = AutoModelForCausalLM.from_pretrained(
			cls.model_path, torch_dtype="bfloat16"
		)
		state_dict = model.state_dict()
		cls.shapes = {name: tensor.shape for name, tensor in state_dict.items()}
		del state_dict
		del model
		gc.collect()

	def _run(self, tp_size, dp_size, backend, load_format=None, pause_mode=None):
		required = 1 + tp_size * dp_size
		available = get_device_count()
		if available < required:
			self.skipTest(f"{required} XPU devices required, {available} visible")

		self._harness.test_update_weights_from_distributed(
			tp_size=tp_size,
			dp_size=dp_size,
			model_name=self.model_path,
			backend=backend,
			state_dict_key_to_shape=self.shapes,
			truncate_size=10,
			checking_parameters=[
				"model.embed_tokens.weight",
				"model.layers.0.input_layernorm.weight",
				"model.layers.1.self_attn.q_proj.weight",
				"model.layers.2.self_attn.k_proj.weight",
				"model.layers.3.self_attn.v_proj.weight",
				"model.layers.4.self_attn.o_proj.weight",
				"model.layers.5.mlp.gate_proj.weight",
				"model.layers.6.mlp.up_proj.weight",
				"model.layers.7.mlp.down_proj.weight",
				"model.layers.8.post_attention_layernorm.weight",
				"model.norm.weight",
				"lm_head.weight",
			],
			load_format=load_format,
			pause_generation_mode=pause_mode,
		)

	def test_tp1dp1_engine_default(self):
		self._run(1, 1, "Engine")

	def test_tp1dp1_engine_flattened_bucket(self):
		self._run(1, 1, "Engine", load_format="flattened_bucket")

	def test_tp1dp1_server_in_place_default(self):
		self._run(1, 1, "Server", pause_mode="in_place")

	def test_tp1dp1_server_in_place_flattened_bucket(self):
		self._run(1, 1, "Server", load_format="flattened_bucket", pause_mode="in_place")

	def test_tp1dp1_server_retract_default(self):
		self._run(1, 1, "Server", pause_mode="retract")

	def test_tp1dp1_server_retract_flattened_bucket(self):
		self._run(1, 1, "Server", load_format="flattened_bucket", pause_mode="retract")

	def test_tp1dp2_engine_default(self):
		self._run(1, 2, "Engine")

	def test_tp1dp2_engine_flattened_bucket(self):
		self._run(1, 2, "Engine", load_format="flattened_bucket")

	def test_tp1dp2_server_in_place_default(self):
		self._run(1, 2, "Server", pause_mode="in_place")

	def test_tp1dp2_server_in_place_flattened_bucket(self):
		self._run(1, 2, "Server", load_format="flattened_bucket", pause_mode="in_place")

	def test_tp1dp2_server_retract_default(self):
		self._run(1, 2, "Server", pause_mode="retract")

	def test_tp1dp2_server_retract_flattened_bucket(self):
		self._run(1, 2, "Server", load_format="flattened_bucket", pause_mode="retract")

	def test_tp2dp1_engine_default(self):
		self._run(2, 1, "Engine")

	def test_tp2dp1_engine_flattened_bucket(self):
		self._run(2, 1, "Engine", load_format="flattened_bucket")

	def test_tp2dp1_server_in_place_default(self):
		self._run(2, 1, "Server", pause_mode="in_place")

	def test_tp2dp1_server_in_place_flattened_bucket(self):
		self._run(2, 1, "Server", load_format="flattened_bucket", pause_mode="in_place")

	def test_tp2dp1_server_retract_default(self):
		self._run(2, 1, "Server", pause_mode="retract")

	def test_tp2dp1_server_retract_flattened_bucket(self):
		self._run(2, 1, "Server", load_format="flattened_bucket", pause_mode="retract")

	def test_tp2dp2_engine_default(self):
		self._run(2, 2, "Engine")

	def test_tp2dp2_engine_flattened_bucket(self):
		self._run(2, 2, "Engine", load_format="flattened_bucket")

	def test_tp2dp2_server_in_place_default(self):
		self._run(2, 2, "Server", pause_mode="in_place")

	def test_tp2dp2_server_in_place_flattened_bucket(self):
		self._run(2, 2, "Server", load_format="flattened_bucket", pause_mode="in_place")

	def test_tp2dp2_server_retract_default(self):
		self._run(2, 2, "Server", pause_mode="retract")

	def test_tp2dp2_server_retract_flattened_bucket(self):
		self._run(2, 2, "Server", load_format="flattened_bucket", pause_mode="retract")


@unittest.skipUnless(is_xpu(), "XPU-only disk weight-update tests")
class TestUpdateWeightsFromDisk(UpdateWeightsFromDiskBase, CustomTestCase):
	"""BF16 TP1/TP2 refits, memory handoff, and failed-update integrity.

	Run with ZE_AFFINITY_MASK=2,3 python -m pytest
	test/manual/rl/test_xpu_rl.py::TestUpdateWeightsFromDisk -vv.
	Requires the Llama-3.2-1B Instruct/base pair and XPU torch_memory_saver.
	Idle pause checks do not establish live partial-rollout correctness.
	Blackwell-only quantized formats are not covered.
	"""

	backend_test_suites = ({"tp_size": 1}, {"tp_size": 2})
	parameter_names = (
		"model.embed_tokens.weight",
		"model.layers.0.input_layernorm.weight",
		"model.layers.1.self_attn.q_proj.weight",
		"lm_head.weight",
	)

	@classmethod
	def setUpClass(cls):
		cls.model = snapshot_download(DEFAULT_SMALL_MODEL_NAME_FOR_TEST)
		cls.base_model = snapshot_download(DEFAULT_SMALL_MODEL_NAME_FOR_TEST_BASE)
		cls.references = {}
		for model_path in (cls.model, cls.base_model):
			model = AutoModelForCausalLM.from_pretrained(
				model_path, torch_dtype=torch.bfloat16, device_map="cpu"
			)
			cls.hidden_size = model.config.hidden_size
			parameters = dict(model.named_parameters(remove_duplicate=False))
			cls.references[model_path] = {
				name: parameters[name].detach().flatten()[:16].float().clone()
				for name in cls.parameter_names
			}
			del parameters, model
			gc.collect()
		super().setUpClass()

	def _launch_server(self, backend_test_suite):
		tp_size = backend_test_suite["tp_size"]
		if get_device_count() < tp_size:
			self.skipTest(f"{tp_size} visible XPU devices required")
		return popen_launch_server(
			self.model,
			self.base_url,
			timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
			other_args=[
				"--device", "xpu",
				# "--attention-backend", "intel_xpu",
				"--attention-backend", "triton",
				"--dtype", "bfloat16", "--tp-size", str(tp_size),
				"--enable-memory-saver", "--mem-fraction-static", "0.7",
				# Batch-invariant decode: without it the triton backend picks
				# kv splits from the batch, so a refit's cache flush changes the
				# reduction order and the 1e-4 logprob check sees 3e-2 drift.
				"--enable-deterministic-inference",
			],
		)

	@contextmanager
	def _server(self, tp_size=1):
		process = self._launch_server({"tp_size": tp_size})
		try:
			self.assertEqual(self._get_model_info(), self.model)
			yield
		finally:
			kill_process_tree(process.pid, wait_timeout=60)

	def _assert_weights(self, model_path):
		for name, expected in self.references[model_path].items():
			with self.subTest(parameter=name):
				values = self._post_json(
					"/get_weights_by_name", {"name": name, "truncate_size": 16}
				)
				actual = torch.tensor(values).flatten()[:16]
				torch.testing.assert_close(actual, expected, rtol=0, atol=0)

	def _refit(self, model_path, **options):
		result = self._post_json(
			"/update_weights_from_disk",
			{"model_path": model_path, "flush_cache": True, **options},
			timeout=self.update_timeout,
		)
		self.assertTrue(result.get("success"), result)
		self.assertEqual(self._get_model_info(), model_path)
		self._assert_weights(model_path)

	def _check_changed_checkpoint(self, tp_size):
		self.assertTrue(any(
			not torch.equal(self.references[self.model][name], self.references[self.base_model][name])
			for name in self.parameter_names
		), "The two checkpoint references must differ")
		with self._server(tp_size):
			self._assert_weights(self.model)
			baseline = self._get_decode_logprob_signature()
			for step in range(2):
				with self.subTest(step=step):
					self._wait_until_idle()
					self._refit(self.base_model, load_format="safetensors")
					changed = self._get_decode_logprob_signature()
					self.assertNotEqual(changed, baseline)
					self._wait_until_idle()
					self._refit(self.model)
					self._assert_decode_logprob_unchanged(
						baseline, self._get_decode_logprob_signature()
					)

	def test_changed_checkpoint_tp1(self):
		self._check_changed_checkpoint(1)

	def test_changed_checkpoint_tp2(self):
		self._check_changed_checkpoint(2)

	def test_pause_refit_continue(self):
		with self._server():
			baseline = self._get_decode_logprob_signature()
			for mode in ("abort", "in_place", "retract"):
				with self.subTest(mode=mode):
					self._wait_until_idle()
					self._post_json("/pause_generation", {"mode": mode})
					try:
						self._refit(self.base_model)
					finally:
						self._post_json("/continue_generation", {})
					self._assert_non_empty_decode()
					self._wait_until_idle()
					self._refit(self.model)
					self._assert_decode_logprob_unchanged(
						baseline, self._get_decode_logprob_signature()
					)

	def _check_failed_update(self, malformed):
		with tempfile.TemporaryDirectory(prefix="sglang-xpu-invalid-") as directory:
			shutil.copyfile(Path(self.model) / "config.json", Path(directory) / "config.json")
			if malformed:
				save_file({
					"model.layers.0.input_layernorm.weight": torch.full(
						(self.hidden_size,), 2.0, dtype=torch.bfloat16,
					),
					"model.layers.0.self_attn.q_proj.weight": torch.zeros(1, 1),
				}, str(Path(directory) / "model.safetensors"))
			with self._server():
				baseline = self._get_decode_logprob_signature()
				self._wait_until_idle()
				snapshot = self._post_json("/weights_checker", {"action": "snapshot"})
				self.assertTrue(snapshot.get("success"), snapshot)
				response = requests.post(
					f"{self.base_url}/update_weights_from_disk",
					json={"model_path": directory, "flush_cache": True},
					timeout=self.update_timeout,
				)
				self.assertIn(response.status_code, (200, 400), response.text)
				result = response.json()
				self.assertIs(result.get("success"), False, result)
				self.assertTrue(result.get("message"), result)
				self.assertEqual(self._get_model_info(), self.model)
				comparison = self._post_json("/weights_checker", {"action": "compare"})
				self.assertTrue(comparison.get("success"), comparison)
				self._assert_weights(self.model)
				self._assert_decode_logprob_unchanged(baseline, self._get_decode_logprob_signature())
				self._wait_until_idle()
				self._refit(self.model)

	def test_missing_weights_preserves_model(self):
		self._check_failed_update(malformed=False)

	def test_malformed_weights_rolls_back(self):
		self._check_failed_update(malformed=True)


class _CaptureXPUCase(CustomTestCase):
	@classmethod
	def setUpClass(cls):
		if not is_xpu() or get_device_count() < cls.required_devices:
			raise unittest.SkipTest(f"{cls.required_devices} visible XPU devices required")
		cls.config = AutoConfig.from_pretrained(cls.model, trust_remote_code=True)

	@contextmanager
	def _server(self, extra_args=()):
		self.base_url = f"http://127.0.0.1:{find_available_port(21000)}"
		process = popen_launch_server(
			self.model,
			self.base_url,
			timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
			other_args=[
				"--device", "xpu", "--dtype", "bfloat16",
				"--tp-size", str(self.required_devices),
				"--context-length", "4096", "--max-total-tokens", "8192",
				"--chunked-prefill-size", "4096", "--max-running-requests", "4",
				"--mem-fraction-static", "0.8", "--random-seed", "0",
				*self.server_args, *extra_args,
			],
		)
		try:
			yield
		finally:
			kill_process_tree(process.pid, wait_timeout=60)

	def _post(self, endpoint, payload):
		response = requests.post(self.base_url + endpoint, json=payload, timeout=180)
		response.raise_for_status()
		if endpoint == "/flush_cache":
			return response.text
		body = response.json()
		if isinstance(body, dict):
			self.assertNotIn("error", body, body)
		return body

	def _generate(self, text, capture=True, **extra):
		body = self._post("/generate", {
			"text": text,
			"sampling_params": {"temperature": 0, "max_new_tokens": 8, "ignore_eos": True},
			self.request_flag: capture,
			**extra,
		})
		self.assertEqual(body["meta_info"]["completion_tokens"], 8)
		if not capture:
			self.assertNotIn(self.metadata_key, body["meta_info"])
		return body

	def _rows(self, body, start_len=0):
		meta = body["meta_info"]
		return max(0, meta["prompt_tokens"] + meta["completion_tokens"] - 1 - start_len)


class TestReturnRoutedExpertsXPU(_CaptureXPUCase):
	"""Pretrained BF16 Qwen3 MoE capture on four XPUs; no DeepEP coverage."""

	model = "Qwen/Qwen3-30B-A3B"
	required_devices = 4
	request_flag = "return_routed_experts"
	metadata_key = "routed_experts"
	server_args = (
		"--attention-backend", "triton", "--enable-return-routed-experts",
		"--enable-deterministic-inference",
	)
	prompt = "Explain why the sky appears blue during the day in one sentence."

	def _experts(self, body, start_len=0):
		flat = extract_routed_experts_from_meta_info(body)
		shape = (self._rows(body, start_len), self.config.num_hidden_layers,
			self.config.num_experts_per_tok)
		self.assertEqual(flat.dtype, np.dtype("int32"))
		self.assertEqual(flat.size, int(np.prod(shape)))
		experts = flat.reshape(shape)
		self.assertGreater(experts.size, 0)
		self.assertTrue(((experts >= 0) & (experts < self.config.num_experts)).all())
		ordered = np.sort(experts, axis=-1)
		self.assertTrue((np.diff(ordered, axis=-1) > 0).all(), "Duplicate expert IDs")
		return experts

	def test_generate_and_request_flags(self):
		with self._server():
			captured = self._generate(self.prompt)
			self._experts(captured)
			plain = self._generate(self.prompt, capture=False)
			self.assertEqual(captured["output_ids"], plain["output_ids"])
			self.assertEqual(captured["text"], plain["text"])

	def test_start_len_and_cache_hit(self):
		with self._server():
			self._post("/flush_cache", {})
			full = self._generate(self.prompt)
			self.assertEqual(full["meta_info"]["cached_tokens"], 0)
			experts = self._experts(full)
			prompt_tokens = full["meta_info"]["prompt_tokens"]
			for start_len in (0, max(1, prompt_tokens // 2), prompt_tokens):
				with self.subTest(start_len=start_len):
					cropped = self._generate(self.prompt, routed_experts_start_len=start_len)
					self.assertEqual(full["output_ids"], cropped["output_ids"])
					np.testing.assert_array_equal(experts[start_len:], self._experts(cropped, start_len))
					if start_len < prompt_tokens - 1:
						self.assertGreater(cropped["meta_info"]["cached_tokens"], start_len)
			response = requests.post(self.base_url + "/generate", json={
				"text": self.prompt, "return_routed_experts": True,
				"routed_experts_start_len": prompt_tokens + 1,
				"sampling_params": {"max_new_tokens": 8},
			}, timeout=180)
			self.assertIn(response.status_code, (200, 400))
			self.assertIn("is higher than the number of input tokens", response.text)

	def test_concurrent_mixed_requests(self):
		texts = [self.prompt, "Compute 7 plus 12."]
		with self._server():
			isolated = [self._generate(text) for text in texts]
			self._post("/flush_cache", {})
			with ThreadPoolExecutor(max_workers=4) as executor:
				jobs = [(index, capture, executor.submit(self._generate, text, capture))
					for index, text in enumerate(texts) for capture in (False, True)]
				for index, capture, future in jobs:
					body = future.result()
					self.assertEqual(body["output_ids"], isolated[index]["output_ids"])
					if capture:
						np.testing.assert_array_equal(self._experts(body), self._experts(isolated[index]))

	def test_openai_endpoints(self):
		with self._server():
			for endpoint, prompt_payload in (
				("/v1/completions", {"prompt": self.prompt}),
				("/v1/chat/completions", {"messages": [{"role": "user", "content": self.prompt}]}),
			):
				with self.subTest(endpoint=endpoint):
					body = self._post(endpoint, {
						"model": self.model, **prompt_payload, "temperature": 0,
						"max_tokens": 8, "return_routed_experts": True,
					})
					self.assertGreater(body["usage"]["completion_tokens"], 0)
					self._experts({"meta_info": {
						**body["usage"], "routed_experts": body["sglext"]["routed_experts"],
					}})


class TestReturnIndexerTopkXPU(_CaptureXPUCase):
	"""Two-card DSA capture checks using a random fixture, not model-quality tests."""

	model = "yujiepan/glm-moe-dsa-tiny-random"
	required_devices = 2
	request_flag = "return_indexer_topk"
	metadata_key = "indexer_topk"
	server_args = (
		"--trust-remote-code", "--dp-size", "2", "--enable-dp-attention",
		"--enable-return-indexer-topk", "--page-size", "64",
	)
	prompt = "What is the capital of France?"
	long_prompt = "word " * 2600

	def _indexer_args(self, shared):
		layers = get_num_indexer_layers(self.config)
		self.assertGreaterEqual(layers, 2)
		return ("--json-model-override-args", json.dumps({
			"indexer_types": ["full"] + ["shared" if shared else "full"] * (layers - 1),
			"num_nextn_predict_layers": 0,
		}))

	def _topk(self, body):
		flat = extract_indexer_topk_from_meta_info(body)
		shape = (self._rows(body), get_num_indexer_layers(self.config), self.config.index_topk)
		self.assertEqual(flat.dtype, np.dtype("int32"))
		self.assertEqual(flat.size, int(np.prod(shape)))
		topk = flat.reshape(shape)
		self.assertGreater(topk.size, 0)
		positions = np.arange(shape[0])[:, None, None]
		self.assertTrue(((topk >= -1) & (topk <= positions)).all(), "Noncausal or invalid index")
		expected = np.minimum(np.arange(shape[0]) + 1, shape[2])
		np.testing.assert_array_equal((topk >= 0).sum(axis=-1), np.broadcast_to(expected[:, None], shape[:2]))
		ordered = np.sort(topk, axis=-1)
		self.assertFalse(((ordered[..., 1:] == ordered[..., :-1]) & (ordered[..., 1:] >= 0)).any())
		return topk

	def test_shared_layers_and_mixed_requests(self):
		with self._server(self._indexer_args(shared=True)):
			with ThreadPoolExecutor(max_workers=3) as executor:
				jobs = [executor.submit(self._generate, text, capture)
					for text, capture in ((self.long_prompt, True), (self.prompt, True), (self.prompt, False))]
				long_result, short_result, plain = [job.result() for job in jobs]
			self.assertGreater(long_result["meta_info"]["prompt_tokens"], self.config.index_topk)
			self.assertEqual(short_result["output_ids"], plain["output_ids"])
			for body in (long_result, short_result):
				topk = self._topk(body)
				for layer in range(1, topk.shape[1]):
					np.testing.assert_array_equal(topk[:, layer], topk[:, layer - 1])

	def test_fresh_layers_and_saturation(self):
		with self._server(self._indexer_args(shared=False)):
			body = self._generate(self.long_prompt)
			self.assertGreater(body["meta_info"]["prompt_tokens"], self.config.index_topk)
			topk = self._topk(body)
			selected = np.sort(topk[self.config.index_topk:, :2], axis=-1)
			self.assertFalse(np.array_equal(selected[:, 0], selected[:, 1]),
				"Independent fixture layers selected identical position sets")
			self.assertEqual(int((topk[-1, 0] >= 0).sum()), self.config.index_topk)

	def test_cache_hit_and_repeated_requests(self):
		cache_prompt = "word " * 128
		with self._server(self._indexer_args(shared=True)):
			for iteration in range(2):
				with self.subTest(iteration=iteration):
					self._post("/flush_cache", {})
					cold = self._generate(cache_prompt, routed_dp_rank=0)
					warm = self._generate(cache_prompt, routed_dp_rank=0)
					self.assertGreater(cold["meta_info"]["prompt_tokens"], 64)
					self.assertEqual(cold["meta_info"]["cached_tokens"], 0)
					self.assertGreater(warm["meta_info"]["cached_tokens"], 0)
					self.assertEqual(cold["output_ids"], warm["output_ids"])
					np.testing.assert_array_equal(
						np.sort(self._topk(cold), axis=-1),
						np.sort(self._topk(warm), axis=-1),
					)


if __name__ == "__main__":
	unittest.main()
