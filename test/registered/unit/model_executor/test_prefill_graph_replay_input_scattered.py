"""A prefill cuda graph replay must not run input-scattered.

With --enable-attn-tp-input-scattered a model's forward opens
``AttnTpContext.maybe_input_scattered`` and every extend batch is scattered
(the vocab embedding then skips its all-reduce). A breakable (or full) prefill
graph is captured through ``layer_model.forward`` outside that scope, so its
body was recorded in the ordinary, all-reduced layout. At replay,
``ModelRunner._forward_raw`` calls ``PrefillCudaGraphRunner.execute``, whose
eager outer forward opens the scope: without the replay flag the embedding
would feed unreduced vocab-shard partial sums into a body that expects full
rows. That happens for models whose replay embeds eagerly, i.e. the archs with
a registered ``input_embeds`` graph slot (``is_multimodal`` such as
GLM-5.3-Flash); a text-only model embeds inside the captured body instead.

The test drives the real ``ModelRunner._forward_raw`` graph/eager branch, the
real ``PrefillCudaGraphRunner.execute`` and the real ``AttnTpContext`` with
fakes for the ModelRunner instance, the runner's collaborators, the batch and
the published config. The model config is a subset of
the public zai-org/GLM-5.3-Flash config.json (no weights).
"""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.configs.model_config import (
    is_deepseek_dsa,
    is_multimodal_model,
)
from sglang.srt.layers.layer_boundary.adapters import attention as attention_mod
from sglang.srt.layers.layer_boundary.adapters.attention import AttnTpContext
from sglang.srt.model_executor import cuda_graph_config as cgc
from sglang.srt.model_executor import model_runner as model_runner_mod
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.model_executor.runner import prefill_cuda_graph_runner as pcgr_mod
from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
    PrefillCudaGraphRunner,
)
from sglang.srt.runtime_context import get_forward
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

TP_SIZE = 8

# Subset of zai-org/GLM-5.3-Flash config.json (public; revision
# 46a050bec13002ad29ee795c16e61cf4fa3c4a9c). Only the fields the eligibility
# check reads; no weights are involved.
GLM53_FLASH_CONFIG = {
    "architectures": ["Glm5NextForConditionalGeneration"],
    "text_config": {
        "model_type": "glm5_next_text",
        "q_lora_rank": 1536,
        "kv_lora_rank": 512,
        "mhc": True,
        "index_topk": 2048,
        "vocab_size": 154880,
    },
}


def _published_graph_config(prefill_backend):
    """What ``get_exec().graph.cuda_graph_config`` answers for the checks."""
    cfg = SimpleNamespace(
        prefill=SimpleNamespace(backend=prefill_backend),
        decode=SimpleNamespace(backend=cgc.Backend.FULL),
    )
    return SimpleNamespace(graph=SimpleNamespace(cuda_graph_config=cfg))


def _extend_batch(num_tokens=64):
    return SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        input_ids=torch.arange(num_tokens),
        can_run_tbo=False,
        token_indices_to_pool=None,
        global_num_tokens_cpu=None,
    )


class FakePrefillGraphRunner(PrefillCudaGraphRunner):
    """The real ``PrefillCudaGraphRunner.execute`` on the breakable backend,
    with its collaborators faked (no ``__init__``, no CUDA).

    capture(): the body is traced by ``layer_model.forward`` directly, outside
    the model's ``maybe_input_scattered`` scope (``_run_forward``).
    _execute_body_capture(): the replay runs the eager outer
    ``*ForCausalLM.forward`` (``model_runner.model.forward``), which opens the
    scope; the embedding reads ``input_scattered`` to decide whether to
    all-reduce (vocab_parallel_embedding.py), then the captured body replays.
    """

    def __init__(self, ctx):  # noqa: super().__init__ needs a model runner
        self.ctx = ctx
        self.captured_scattered = None
        self.embedding_all_reduces = None
        self.replayed_scattered = None
        self.replays = 0
        self.backend = SimpleNamespace(replay_session=nullcontext)
        self.enable_cp_bcg_capture = False
        self.raw_num_tokens = 0
        self.model_runner = None
        self.device_module = None

    def capture(self, batch):
        self.captured_scattered = get_forward().attn_input_scattered

    def can_run_graph(self, forward_batch):
        return True

    def model_context(self):
        return nullcontext()

    def _validate_capture_hidden_mode(self, forward_batch):
        pass

    def load_batch(self, forward_batch, **kwargs):
        return SimpleNamespace(input_ids=forward_batch.input_ids), None

    def _shape_key(self, static_num_tokens, forward_batch):
        return SimpleNamespace(variant_label=None)

    def _wait_for_layer_loads(self):
        pass

    def _uses_eager_prefill_tail(self):
        return True

    def _finalize_execute_output(self, output):
        return output

    def _execute_body_capture(self, forward_batch, *args, **kwargs):
        self.replays += 1
        with self.ctx.maybe_input_scattered(forward_batch):
            self.embedding_all_reduces = not self.ctx.input_scattered
            self.replayed_scattered = self.ctx.input_scattered
        return "logits"


class TestPrefillGraphReplayInputScattered(CustomTestCase):
    def setUp(self):
        cfg = GLM53_FLASH_CONFIG
        patches = [
            patch.object(attention_mod, "_is_cuda", True),
            patch.object(attention_mod, "input_scattered_configured", lambda: True),
            patch.object(
                attention_mod,
                "get_parallel",
                lambda: SimpleNamespace(
                    enable_attn_tp_input_scattered=True, tp_size=TP_SIZE
                ),
            ),
            patch.object(
                attention_mod,
                "get_spec",
                lambda: SimpleNamespace(speculative_algorithm=None),
            ),
            patch.object(
                cgc,
                "get_exec",
                lambda: _published_graph_config(cgc.Backend.BREAKABLE),
            ),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        # Model construction (glm5_next.py): init_context(q_lora_rank, use_dsa, mhc).
        text = cfg["text_config"]
        self.ctx = AttnTpContext()
        self.ctx.init_context(text["q_lora_rank"], is_deepseek_dsa(text), text["mhc"])

    def _model_runner_self(self, graph_runner):
        fake = MagicMock(name="ModelRunner")
        fake.device = "cuda"
        fake.decode_cuda_graph_runner = None
        fake.uocr_greedy_graph = None
        fake.hisparse_coordinator = None
        fake.device_timer = None
        fake.prefill_cuda_graph_runner = graph_runner
        fake.pp_group = SimpleNamespace(is_last_rank=False)
        fake.eager_runner.execute = lambda fb, **kw: self._eager(fb)
        fake._extend_forward_kwargs = lambda fb, pp: {}
        return fake

    def _eager(self, forward_batch):
        # Eager extend: the whole forward is scattered when the flag allows.
        with self.ctx.maybe_input_scattered(forward_batch):
            self.eager_scattered = self.ctx.input_scattered
        return "eager-logits"

    def _forward_raw(self, graph_runner, forward_batch):
        """Run the real ModelRunner._forward_raw graph/eager dispatch."""
        mr = model_runner_mod
        with (
            patch.object(
                mr,
                "get_model",
                lambda: SimpleNamespace(override=lambda **k: nullcontext()),
            ),
            patch.object(mr, "has_forward_context", lambda: False),
            patch.object(mr, "forward_context", lambda *a, **k: nullcontext()),
            patch.object(mr, "ForwardContext", lambda **k: None),
            patch.object(mr, "get_global_dwdp_manager", lambda: None),
            patch.object(
                mr, "_prefill_cuda_graph_allows_context_parallel", lambda *a: True
            ),
            patch.object(
                pcgr_mod, "maybe_publish_prefill_shared_read_done", lambda *a: None
            ),
        ):
            return mr.ModelRunner._forward_raw(
                self._model_runner_self(graph_runner), forward_batch, None
            )

    # --- controls: pass with and without the fix -----------------------------

    def test_public_config_is_eligible(self):
        cfg = GLM53_FLASH_CONFIG
        # The replay embeds eagerly only for archs with an input_embeds slot.
        self.assertTrue(is_multimodal_model(cfg["architectures"]))
        self.assertIsNotNone(cfg["text_config"]["q_lora_rank"])
        self.assertTrue(self.ctx.allow_input_scattered)  # mHC admits DSA

    def test_eager_extend_is_scattered(self):
        """Without a graph the SP path stays on: the fix must not touch it."""
        runner = FakePrefillGraphRunner(self.ctx)
        runner.can_run_graph = lambda fb: False
        self.eager_scattered = None
        out = self._forward_raw(runner, _extend_batch())
        self.assertEqual(out.logits_output, "eager-logits")
        self.assertFalse(out.can_run_graph)
        self.assertTrue(self.eager_scattered)
        self.assertTrue(self.ctx.use_input_scattered(_extend_batch()))

    def test_flag_off_never_scatters(self):
        self.ctx.allow_input_scattered = False
        runner = FakePrefillGraphRunner(self.ctx)
        runner.capture(_extend_batch())
        out = self._forward_raw(runner, _extend_batch())
        self.assertTrue(out.can_run_graph)
        self.assertFalse(runner.replayed_scattered)
        self.assertTrue(runner.embedding_all_reduces)

    def test_replay_scope_does_not_leak(self):
        runner = FakePrefillGraphRunner(self.ctx)
        self._forward_raw(runner, _extend_batch())
        self.assertTrue(self.ctx.use_input_scattered(_extend_batch()))
        self.assertFalse(get_forward().attn_input_scattered)

    # --- the bug --------------------------------------------------------------

    def test_breakable_replay_runs_in_the_captured_layout(self):
        """Without the replay flag the replay's outer forward is input-scattered although
        the graph body was captured unscattered, so the embedding skips its
        all-reduce and the body consumes partial sums."""
        runner = FakePrefillGraphRunner(self.ctx)
        batch = _extend_batch()
        runner.capture(batch)
        self.assertFalse(runner.captured_scattered)  # capture is outside the scope

        out = self._forward_raw(runner, batch)

        self.assertTrue(out.can_run_graph)
        self.assertEqual(runner.replays, 1)
        self.assertEqual(
            runner.replayed_scattered,
            runner.captured_scattered,
            "replay must run in the layout the graph was captured in",
        )
        self.assertTrue(
            runner.embedding_all_reduces,
            "embedding must all-reduce: the captured body expects full rows",
        )

    def test_predicate_under_replay_scope(self):
        """The predicate itself, asked from inside the graph/eager branch."""
        seen = {}
        runner = FakePrefillGraphRunner(self.ctx)

        def body(forward_batch, *args, **kwargs):
            seen["use_input_scattered"] = self.ctx.use_input_scattered(forward_batch)
            return "logits"

        runner._execute_body_capture = body
        self._forward_raw(runner, _extend_batch())
        self.assertFalse(seen["use_input_scattered"])


if __name__ == "__main__":
    unittest.main()
