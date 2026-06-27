"""Unified EAGLE/EAGLE3 speculative-decoding server fixture.

A single popen-server base whose launch is fully described by class attributes,
so concrete test classes only flip knobs (overlap on/off, model, topk, page size,
backend, env overrides). Pair it with the kits in
``sglang.test.kits.spec_server_kits`` to assemble test classes.

The primary axis is ``disable_overlap``:
  - ``False`` -> overlap scheduler
  - ``True``  -> synchronous (non-overlap) scheduling
"""

import contextlib
import random
import time

import requests

from sglang.srt.environ import envs
from sglang.srt.utils.common import kill_process_tree
from sglang.srt.utils.hf_transformers_utils import get_tokenizer
from sglang.test.test_utils import (
    DEFAULT_DRAFT_MODEL_EAGLE,
    DEFAULT_DRAFT_MODEL_EAGLE3,
    DEFAULT_TARGET_MODEL_EAGLE,
    DEFAULT_TARGET_MODEL_EAGLE3,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

# Chat-style prompts shared by send_request / send_requests_abort.
PROMPTS = [
    "[INST] <<SYS>>\\nYou are a helpful assistant.\\n<</SYS>>\\nToday is a sunny day and I like[/INST]"
    '[INST] <<SYS>>\\nYou are a helpful assistant.\\n<</SYS>>\\nWhat are the mental triggers in Jeff Walker\'s Product Launch Formula and "Launch" book?[/INST]',
    "[INST] <<SYS>>\\nYou are a helpful assistant.\\n<</SYS>>\\nSummarize Russell Brunson's Perfect Webinar Script...[/INST]",
    "[INST] <<SYS>>\\nYou are a helpful assistant.\\n<</SYS>>\\nwho are you?[/INST]",
    "[INST] <<SYS>>\\nYou are a helpful assistant.\\n<</SYS>>\\nwhere are you from?[/INST]",
]


class SpecEagleServerBase(CustomTestCase):
    """Launch a single EAGLE/EAGLE3 spec server from class-attribute knobs."""

    # -- model --
    model = DEFAULT_TARGET_MODEL_EAGLE3
    draft_model = DEFAULT_DRAFT_MODEL_EAGLE3
    spec_algo = "EAGLE3"

    # -- speculative config --
    spec_steps = 5
    spec_topk = 1
    spec_tokens = 6

    # -- runtime config --
    page_size = 1
    attention_backend = "flashinfer"
    # Primary axis: False -> overlap scheduler; True -> synchronous (non-overlap).
    disable_overlap = False
    mem_fraction_static = 0.85
    max_running_requests = 8
    chunked_prefill_size = 128
    # bf16 rather than fp16: fp16 activations can overflow (-> Inf -> NaN) on
    # degenerate draft branches in verify and trip the CI NaN asserts.
    dtype = "bfloat16"
    cuda_graph_max_bs = None
    trust_remote_code = True
    # Launch with --enable-return-hidden-states so SpecHiddenStatesKit can probe
    # per-request hidden states; per-request gated, so other requests don't pay.
    enable_return_hidden_states = False

    # -- extras --
    # env_overrides: (env_var_obj, value) pairs applied only around launch.
    # Declare ONLY this class's own; _merged_env_overrides() unions them down the
    # MRO (base first), so never restate a base's. Derived wins on a repeated env.
    env_overrides = ()
    extra_args = ()

    @classmethod
    def _launch_args(cls):
        args = [
            "--speculative-algorithm",
            cls.spec_algo,
            "--speculative-draft-model-path",
            cls.draft_model,
            "--speculative-num-steps",
            str(cls.spec_steps),
            "--speculative-eagle-topk",
            str(cls.spec_topk),
            "--speculative-num-draft-tokens",
            str(cls.spec_tokens),
            "--page-size",
            str(cls.page_size),
            "--attention-backend",
            cls.attention_backend,
            "--mem-fraction-static",
            str(cls.mem_fraction_static),
            "--max-running-requests",
            str(cls.max_running_requests),
            "--chunked-prefill-size",
            str(cls.chunked_prefill_size),
            "--dtype",
            cls.dtype,
        ]
        if cls.disable_overlap:
            args.append("--disable-overlap-schedule")
        if cls.trust_remote_code:
            args.append("--trust-remote-code")
        if cls.enable_return_hidden_states:
            args.append("--enable-return-hidden-states")
        if cls.cuda_graph_max_bs is not None:
            args += ["--cuda-graph-max-bs", str(cls.cuda_graph_max_bs)]
        args += [str(a) for a in cls.extra_args]
        return args

    @classmethod
    def _merged_env_overrides(cls):
        # Base first so a derived class wins for a repeated env var.
        merged = []
        for klass in reversed(cls.__mro__):
            merged.extend(klass.__dict__.get("env_overrides", ()))
        return merged

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        # Alias so kit methods can use either name.
        cls.target_model = cls.model
        cls._tokenizer = None
        with contextlib.ExitStack() as stack:
            stack.enter_context(
                envs.SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN.override(True)
            )
            for env_var, value in cls._merged_env_overrides():
                stack.enter_context(env_var.override(value))
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=cls._launch_args(),
            )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid, wait_timeout=60)

    @property
    def tokenizer(self):
        if type(self)._tokenizer is None:
            type(self)._tokenizer = get_tokenizer(self.model)
        return type(self)._tokenizer

    # -- helpers used by kits --

    def send_request(self):
        time.sleep(random.uniform(0, 2))
        for prompt in PROMPTS:
            url = self.base_url + "/generate"
            data = {
                "text": prompt,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 1024,
                },
            }
            response = requests.post(url, json=data)
            assert response.status_code == 200

    def send_requests_abort(self):
        for prompt in PROMPTS:
            try:
                time.sleep(random.uniform(0, 2))
                url = self.base_url + "/generate"
                data = {
                    "model": "base",
                    "text": prompt,
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 1024,
                    },
                }
                # set timeout = 1s, mock disconnected
                requests.post(url, json=data, timeout=1)
            except Exception as e:
                print(e)
                pass

    def run_decode(self, sampling_params):
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": "Human: Write a travel blog post to Hawaii.\n\nAssistant:",
                "sampling_params": {
                    "max_new_tokens": 48,
                    "n": 1,
                    "temperature": 0.7,
                    **sampling_params,
                },
                "return_logprob": True,
                "top_logprobs_num": 5,
                "return_text_in_logprobs": True,
                "logprob_start_len": 0,
            },
        )
        self.assertEqual(response.status_code, 200)


class Eagle3Base(SpecEagleServerBase):
    """EAGLE3 (Llama-3.1) config preset, topk=1 / page_size=1 by default."""

    model = DEFAULT_TARGET_MODEL_EAGLE3
    draft_model = DEFAULT_DRAFT_MODEL_EAGLE3
    spec_algo = "EAGLE3"
    spec_steps = 5
    spec_topk = 1
    spec_tokens = 6
    attention_backend = "flashinfer"
    chunked_prefill_size = 1024
    # EAGLE3 topk=1 accepts modestly; tune against CI if needed.
    acc_length_thres = 1.6
    batch_accept_len_thres = 1.3
    gsm8k_score_thres = 0.7
    gsm8k_accept_len_thres = 1.3


class EagleLlama2Base(SpecEagleServerBase):
    """EAGLE (Llama-2) config preset. topk=8 tree -> synchronous path; gsm8k is low."""

    model = DEFAULT_TARGET_MODEL_EAGLE
    draft_model = DEFAULT_DRAFT_MODEL_EAGLE
    spec_algo = "EAGLE"
    spec_steps = 5
    spec_topk = 8
    spec_tokens = 64
    attention_backend = "flashinfer"
    chunked_prefill_size = 128
    mem_fraction_static = 0.7
    gsm8k_score_thres = 0.20
    acc_length_thres = 3.0
    batch_accept_len_thres = 1.8
    # topk>1 tree verify runs on the synchronous (non-overlap) path.
    disable_overlap = True
