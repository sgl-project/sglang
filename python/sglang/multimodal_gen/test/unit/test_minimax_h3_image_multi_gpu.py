# SPDX-License-Identifier: Apache-2.0
"""The image loop under Ulysses / TP parallelism over real gloo groups.

Every layout spawns its own processes (also the one-process reference, so the
thread count and runtime initialization are identical), initializes the real
SGLang parallel groups on CPU and runs the full ``minimax_h3_image_denoise_loop``
through a tiny real MiniMaxH3DiTModel for a t2i and an edit request. Ulysses2
and TP2 shard the row-local GEMMs / attention heads, so they are compared with
a tolerance.

Ring needs FlashAttention and is not runnable on CPU.
"""

import dataclasses
import os
import socket
import zlib

import pytest
import torch
import torch.multiprocessing as mp

from sglang.multimodal_gen.configs.models.dits.minimax_h3 import (
    MiniMaxH3DiTArchConfig,
    MiniMaxH3DiTConfig,
)

TEXT_DIM = 24
TINY_ARCH = dict(
    num_layers=2,
    token_refiner_num_layers=1,
    hidden_size=32,
    num_attention_heads=4,
    attention_head_dim=16,
    ffn_hidden_size=64,
    text_dim=TEXT_DIM,
    timestep_input_dim=16,
    time_embed_hidden_size=32,
    time_embed_dim=16,
    adaln_out_features=18 * 32,
    final_adaln_out_features=2 * 32,
    rope_inv_freq_len=2,
    # Native [q_all, k_all, v_all] rows, so one fused full-size tensor loads
    # through the stock MergedColumnParallelLinear loader on every TP rank.
    qkv_checkpoint_grouped=False,
)
# t = 0, 0.4, 0.999: the edit references are restored on two steps and
# unmasked on the third.
SIGMAS = [1.0, 0.6, 0.001, 0.0]
CASES = {
    # (text_len, latent_h, latent_w, reference blocks)
    "t2i": (11, 8, 8, []),
    "edit": (11, 8, 8, [{"kind": "image", "latent_h": 4, "latent_w": 6}]),
}
LAYOUTS = {
    "ulysses2": dict(tp=1, ulysses=2, cfg=1),
    "tp2": dict(tp=2, ulysses=1, cfg=1),
}
# TP2 all-reduces bf16 partial products, so its latents differ from the
# single-process ones by up to ~1e-4 on O(3) values (one bf16 ulp there is
# ~1.6e-2). Ulysses only changes the row split and stays well inside it.
SHARDED_TOLERANCE = dict(rtol=0.0, atol=5e-4)


def _seeded(name: str) -> torch.Generator:
    return torch.Generator().manual_seed(zlib.crc32(name.encode()))


def _randn(name: str, *shape: int, scale: float = 1.0) -> torch.Tensor:
    return torch.randn(*shape, generator=_seeded(name), dtype=torch.float32) * scale


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _build_model(tp_size: int):
    """A tiny real DiT whose full weights are crc32(name)-seeded on every rank;
    TP ranks load their shard of the same full tensor through the layer's own
    weight loader."""
    from sglang.multimodal_gen.runtime.layers.linear import (
        ColumnParallelLinear,
        RowParallelLinear,
    )
    from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import MiniMaxH3DiTModel

    arch = dataclasses.replace(MiniMaxH3DiTArchConfig(), **TINY_ARCH)
    model = MiniMaxH3DiTModel(MiniMaxH3DiTConfig(arch_config=arch), hf_config={}).eval()
    with torch.no_grad():
        for module_name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                name = f"{module_name}.{param_name}" if module_name else param_name
                shape = list(param.shape)
                if isinstance(module, ColumnParallelLinear):
                    shape[0] *= tp_size
                elif isinstance(module, RowParallelLinear) and param_name == "weight":
                    shape[1] *= tp_size
                values = _randn(name, *shape, scale=0.05)
                if name.endswith("norm.weight") or name.endswith("norm_q.weight"):
                    values = values + 1.0
                values = values.to(param.dtype)
                loader = getattr(param, "weight_loader", None)
                if loader is None:
                    param.copy_(values)
                else:
                    loader(param, values)
        length = model.rope.inv_freq.numel()
        model.rope.inv_freq.copy_(
            1.0 / (10000 ** (torch.arange(0, 2 * length, 2).float() / (2 * length)))
        )
    return model


def _branch_inputs(case: str, name: str, text_len: int):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.denoising import (
        _image_branch_inputs,
    )

    _, latent_h, latent_w, blocks = CASES[case]
    embedding = {
        "text_len": text_len,
        "hidden_states": _randn(f"{case}.{name}.text", text_len, TEXT_DIM),
        "text_token_tags": torch.zeros(text_len, dtype=torch.long),
    }
    return _image_branch_inputs(embedding, blocks, latent_h, latent_w)


def _case_inputs(case: str):
    text_len, latent_h, latent_w, blocks = CASES[case]
    n_cond = sum(block["latent_h"] * block["latent_w"] // 4 for block in blocks)
    n_rows = n_cond + latent_h * latent_w // 4
    initial_rows = _randn(f"{case}.initial", n_rows, 96)
    anchors = _randn(f"{case}.anchors", n_cond, 96)
    return _branch_inputs(case, "positive", text_len), initial_rows, anchors


def _run_cases(model) -> dict[str, torch.Tensor]:
    from sglang.multimodal_gen.runtime.layers.attention.selector import (
        global_force_attn_backend_context_manager,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.denoise_loop import (
        minimax_h3_image_denoise_loop,
    )
    from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum

    results = {}
    with global_force_attn_backend_context_manager(AttentionBackendEnum.TORCH_SDPA):
        for case in CASES:
            positive, initial_rows, anchors = _case_inputs(case)
            results[case] = minimax_h3_image_denoise_loop(
                model=model,
                positive=positive,
                initial_video_rows=initial_rows,
                keyframe_cond_rows=anchors,
                sigmas=SIGMAS,
                device=torch.device("cpu"),
            )
    return results


def _worker(rank, world_size, port, layout, result_dir):
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE=str(world_size),
    )
    torch.set_num_threads(2)
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        get_classifier_free_guidance_world_size,
        get_tp_world_size,
        get_ulysses_ctx,
        maybe_init_distributed_environment_and_model_parallel,
    )

    maybe_init_distributed_environment_and_model_parallel(
        tp_size=layout["tp"],
        sp_size=layout["ulysses"],
        cfg_degree=layout["cfg"],
        ulysses_degree=layout["ulysses"],
    )
    try:
        results = _run_cases(_build_model(layout["tp"]))
        # Prove the loop ran under the requested groups, not a degenerate init.
        results["ctx"] = dict(
            tp=get_tp_world_size(),
            ulysses=get_ulysses_ctx()[0],
            cfg=get_classifier_free_guidance_world_size(),
        )
        torch.save(results, f"{result_dir}/rank{rank}.pt")
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


def _run_layout(result_dir, layout) -> list[dict[str, torch.Tensor]]:
    world_size = layout["tp"] * layout["ulysses"] * layout["cfg"]
    os.makedirs(result_dir)
    mp.spawn(
        _worker,
        args=(world_size, _free_port(), layout, str(result_dir)),
        nprocs=world_size,
        join=True,
    )
    return [torch.load(f"{result_dir}/rank{rank}.pt") for rank in range(world_size)]


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    (result,) = _run_layout(
        tmp_path_factory.mktemp("single") / "run", dict(tp=1, ulysses=1, cfg=1)
    )
    return result


def _latent_keys(result) -> list[str]:
    return [key for key in result if key != "ctx"]


def _assert_all_ranks_agree(results):
    # The GPU worker emits rank 0's output; every rank must hold the same latents.
    for rank_result in results[1:]:
        assert rank_result["ctx"] == results[0]["ctx"]
        assert _latent_keys(rank_result) == _latent_keys(results[0])
        for case in _latent_keys(results[0]):
            assert torch.equal(rank_result[case], results[0][case]), case


@pytest.mark.skipif(
    torch.cuda.is_available(),
    reason="CPU-only weight-free test; it mixes CPU and CUDA tensors on GPU hosts",
)
@pytest.mark.parametrize("layout", ["ulysses2", "tp2"])
def test_sharded_layouts_match_the_single_process_latents(tmp_path, reference, layout):
    results = _run_layout(tmp_path / layout, LAYOUTS[layout])
    assert results[0]["ctx"] == LAYOUTS[layout]
    _assert_all_ranks_agree(results)
    assert _latent_keys(results[0]) == list(CASES)
    for case in CASES:
        got, want = results[0][case], reference[case]
        assert got.shape == want.shape and torch.isfinite(got).all(), case
        torch.testing.assert_close(got, want, **SHARDED_TOLERANCE, msg=case)
