"""LoRA serving on the Beam fixture against the eager oracle on merged weights.

Each adapter adapts one serving path of the fixture, with different ranks, and
all of them run in one batch with a base request. Every request must match the
eager reference on its own merged weights, so an adapter the engine drops,
misplaces or leaks into another request fails. Layouts cover TP1, TP2, TP2 with
EP2 and PP2, both dense LoRA backends, decode CUDA graphs, adapters loaded after
startup under --lora-target-modules all, and per-expert and shared-outer expert
factors on the legacy Triton MoE runner.
"""

import tempfile
import unittest
from pathlib import Path

import msgspec
import torch

import sglang as sgl
from sglang.srt.utils import is_flashinfer_available
from sglang.srt.utils.common import (
    is_sm90_supported,
    is_sm100_supported,
    is_sm120_supported,
)
from sglang.test.beam_fixture import (
    BEAM_FIXTURE_MAX_KL,
    assert_logprobs_close,
    eager_beam_logits,
    make_beam_fixture,
)
from sglang.test.beam_lora_fixture import (
    LORA_TARGET_GROUPS,
    make_beam_lora_adapter,
    merge_beam_lora,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=240, stage="base-b", runner_config="2-gpu-large")

# name -> (target groups, rank). Ranks differ so the batch mixes them.
ADAPTERS = {
    "attention": (("attention",), 8),
    "attention_gate": (("attention_gate",), 4),
    "dense_mlp": (("dense_mlp",), 16),
    "experts": (("experts",), 8),
    "embeddings": (("embed_tokens", "lm_head"), 4),
    "all": (tuple(LORA_TARGET_GROUPS), 8),
}


class Layout(msgspec.Struct, frozen=True, kw_only=True):
    parallel: dict
    lora_backend: str
    cuda_graph: bool
    expert_layout: str
    # Start with --lora-target-modules all and load every adapter afterwards.
    dynamic_load: bool = False


# Each dimension takes every value in some layout; layouts needing more GPUs
# than present are skipped.
LAYOUTS = {
    "tp1_csgmv": Layout(
        parallel=dict(tp_size=1),
        lora_backend="csgmv",
        cuda_graph=False,
        expert_layout="per_expert",
    ),
    "tp1_triton_graphs_shared_outer": Layout(
        parallel=dict(tp_size=1),
        lora_backend="triton",
        cuda_graph=True,
        expert_layout="shared_outer",
    ),
    "tp2_triton_graphs_dynamic": Layout(
        parallel=dict(tp_size=2),
        lora_backend="triton",
        cuda_graph=True,
        expert_layout="per_expert",
        dynamic_load=True,
    ),
    "tp2_ep2_csgmv": Layout(
        parallel=dict(tp_size=2, ep_size=2),
        lora_backend="csgmv",
        cuda_graph=False,
        expert_layout="per_expert",
    ),
    "pp2_csgmv": Layout(
        parallel=dict(pp_size=2),
        lora_backend="csgmv",
        cuda_graph=False,
        expert_layout="per_expert",
    ),
}
GENERATED_TOKENS = 4
# The LoRA path rounds its rank-r intermediate to BF16 on top of the base kernels'
# rounding. Worst per-token difference measured across these layouts: 0.29 on
# H200 and 0.22 on GB300, where the same merged weights served without LoRA
# already reach 0.15. The per-distribution KL bound is unchanged.
LORA_LOGPROB_ATOL = 0.4


def prompt(length, shift):
    return [1 + (i * 17 + shift) % 127 for i in range(length)]


class TestBeamLoRA(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not is_flashinfer_available() or not (
            is_sm90_supported() or is_sm100_supported() or is_sm120_supported()
        ):
            raise unittest.SkipTest(
                "Beam TRT-LLM MHA test requires FlashInfer and SM90, SM100, or SM120"
            )
        cls.root = tempfile.TemporaryDirectory(prefix="beam-lora-")
        cls.config, state = make_beam_fixture(cls.root.name)
        cls.state = {name: value.cuda() for name, value in state.items()}
        cls.adapters = {}
        cls.merged = {None: cls.state}
        for layout in ("per_expert", "shared_outer"):
            for seed, (name, (groups, rank)) in enumerate(ADAPTERS.items()):
                adapter = make_beam_lora_adapter(
                    Path(cls.root.name) / layout / name,
                    cls.config,
                    cls.state,
                    groups,
                    rank=rank,
                    alpha=2 * rank,
                    seed=seed,
                    expert_layout=layout,
                )
                cls.adapters[layout, name] = adapter.path
                cls.merged[layout, name] = merge_beam_lora(cls.state, adapter)

    @classmethod
    def tearDownClass(cls):
        for attribute in ("merged", "state"):
            if hasattr(cls, attribute):
                delattr(cls, attribute)
        if hasattr(cls, "root"):
            cls.root.cleanup()

    def launch(self, layout):
        adapter_paths = {
            name: self.adapters[layout.expert_layout, name] for name in ADAPTERS
        }
        if layout.dynamic_load:
            lora_kwargs = dict(
                lora_target_modules=["all"],
                max_lora_rank=max(rank for _, rank in ADAPTERS.values()),
            )
        else:
            lora_kwargs = dict(lora_paths=adapter_paths)
        engine = sgl.Engine(
            model_path=self.root.name,
            skip_tokenizer_init=True,
            dtype="bfloat16",
            attention_backend="trtllm_mha",
            page_size=64,
            moe_runner_backend="triton",
            disable_cuda_graph=not layout.cuda_graph,
            cuda_graph_max_bs_decode=8,
            disable_radix_cache=True,
            chunked_prefill_size=-1,
            max_total_tokens=4096,
            mem_fraction_static=0.2,
            enable_lora=True,
            lora_backend=layout.lora_backend,
            max_loras_per_batch=len(ADAPTERS) + 1,
            **lora_kwargs,
            **layout.parallel,
        )
        if layout.dynamic_load:
            for name, path in adapter_paths.items():
                result = engine.load_lora_adapter(lora_name=name, lora_path=path)
                self.assertTrue(result.success, result.error_message)
        return engine

    def check_batch(self, engine, requests, expert_layout, tag):
        """Generate every (prompt, adapter) request in one batch; compare each
        with the eager reference on its adapter's merged weights."""
        outputs = engine.generate(
            input_ids=[token_ids for token_ids, _ in requests],
            sampling_params={
                "temperature": 0,
                "max_new_tokens": GENERATED_TOKENS,
                "ignore_eos": True,
            },
            return_logprob=True,
            token_ids_logprob=list(range(self.config.vocab_size)),
            lora_path=[name for _, name in requests],
        )
        for (token_ids, name), output in zip(requests, outputs):
            with self.subTest(tag=tag, adapter=name, prompt_length=len(token_ids)):
                generated = output["output_ids"]
                self.assertEqual(len(generated), GENERATED_TOKENS)
                actual = torch.tensor(
                    [
                        [entry[0] for entry in row]
                        for row in output["meta_info"]["output_token_ids_logprobs"]
                    ],
                    device="cuda",
                )
                weights = self.merged[None if name is None else (expert_layout, name)]
                expected = eager_beam_logits(
                    self.config, weights, token_ids + generated[:-1]
                )[len(token_ids) - 1 :].log_softmax(-1)
                assert_logprobs_close(
                    self,
                    actual,
                    expected,
                    tag=f"{tag} adapter={name}",
                    atol=LORA_LOGPROB_ATOL,
                )
                if name is not None:
                    # The adapter reached the output: the base reference is far off.
                    base = eager_beam_logits(
                        self.config, self.state, token_ids + generated[:-1]
                    )[len(token_ids) - 1 :].log_softmax(-1)
                    kl_vs_base = (base.exp() * (base - actual)).sum(-1).max().item()
                    self.assertGreater(kl_vs_base, 10 * BEAM_FIXTURE_MAX_KL)

    def test_adapters_match_the_merged_reference(self):
        for tag, layout in LAYOUTS.items():
            gpus = layout.parallel.get("tp_size", 1) * layout.parallel.get("pp_size", 1)
            if gpus > torch.cuda.device_count():
                print(f"{tag}: skipped, needs {gpus} GPUs", flush=True)
                continue
            with self.subTest(layout=tag):
                engine = self.launch(layout)
                try:
                    mixed = [
                        (prompt(6, shift), name)
                        for shift, name in enumerate([*ADAPTERS, None])
                    ]
                    self.check_batch(engine, mixed, layout.expert_layout, tag)
                    # 513 tokens straddle the 512-token sliding window.
                    long_prompts = [(prompt(513, 0), "all"), (prompt(513, 1), None)]
                    self.check_batch(engine, long_prompts, layout.expert_layout, tag)
                finally:
                    engine.shutdown()


if __name__ == "__main__":
    unittest.main()
