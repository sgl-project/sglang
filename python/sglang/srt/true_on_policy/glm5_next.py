"""Admission of GLM-5.3-Flash (``glm5_next``) under TOP: what its mHC, MoE, checkpoint and MTP programs require.

The KDA layers are admitted by their backend (``kda_chunked.validate_kda_contract``); the DSA layers by their op's
validator (``glm5_next_dsa``), called here with the rest of the model when it is built. The trainer runs the same
programs.
"""

# GLM-5.3-Flash's hyper-connection settings; TOP's mhc op runs these (DeepSeek-V4's).
_MHC = {"hc_mult": 4, "hc_sinkhorn_iters": 20, "hc_eps": 1e-6}
# The clamp TOP's BF16 experts and clamped dense MLP apply (moe.top_clamp_swiglu_input, silu_and_mul_clamp).
SWIGLU_LIMIT = 10.0


def validate_top_glm5_next(config, quant_config) -> None:
    """``config`` is the text config. Raises unless the model's programs are the ones TOP runs; no-op when TOP is
    off."""
    from sglang.srt.true_on_policy import is_true_on_policy_enabled

    if not is_true_on_policy_enabled():
        return
    if quant_config is not None:
        raise NotImplementedError(
            "[top] GLM-5.3-Flash TOP admits the BF16 checkpoint only (no quantized program yet)"
        )
    _validate_mhc(config)
    _validate_moe(config)
    _validate_dsa(config)
    _validate_mtp(config)


def _validate_dsa(config) -> None:
    import torch

    from sglang.srt.runtime_context import get_server_args
    from sglang.srt.true_on_policy.glm5_next_dsa import (
        validate_device,
        validate_glm5_next_dsa,
    )

    validate_glm5_next_dsa(config, get_server_args())
    validate_device(torch.cuda.current_device())


def _validate_mtp(config) -> None:
    from sglang.srt.arg_groups.overrides import resolved_view
    from sglang.srt.runtime_context import get_server_args
    from sglang.srt.true_on_policy.mtp import validate_mtp

    # The resolved view: the draft path defaults to the target checkpoint there (speculative_hook). (5, 1, 6) is the
    # cookbook low-latency recipe.
    validate_mtp(
        config,
        resolved_view(get_server_args()),
        family="GLM-5.3-Flash",
        extra_recipes=((5, 1, 6),),
    )


def _validate_mhc(config) -> None:
    from sglang.srt.environ import envs
    from sglang.srt.layers.deep_gemm_wrapper.configurer import ENABLE_JIT_DEEPGEMM

    if not config.mhc or any(getattr(config, k) != v for k, v in _MHC.items()):
        raise NotImplementedError(f"[top] GLM-5.3-Flash mHC requires {_MHC}")
    if not (
        envs.SGLANG_OPT_USE_TILELANG_MHC_PRE.get()
        and envs.SGLANG_OPT_USE_TILELANG_MHC_POST.get()
        and envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.get()
        and ENABLE_JIT_DEEPGEMM
    ):
        raise NotImplementedError(
            "[top] GLM-5.3-Flash mHC requires TileLang pre/post and the DeepGEMM prenorm (the trainer calls the "
            "same mhc_pre/mhc_post); without them glm5_next runs its PyTorch mHC"
        )
    if envs.SGLANG_OPT_USE_FLASHINFER_MHC.get():
        raise NotImplementedError("[top] GLM-5.3-Flash does not admit FlashInfer mHC")


def _validate_moe(config) -> None:
    from sglang.srt.true_on_policy.moe_ep import validate_ep_runtime

    if (
        config.scoring_func != "sigmoid"
        or config.topk_method != "noaux_tc"
        or config.n_group != 1
        or config.topk_group != 1
        or not config.norm_topk_prob
        or config.hidden_act != "silu"
    ):
        raise NotImplementedError(
            "[top] GLM-5.3-Flash requires ungrouped sigmoid/bias routing with normalized weights and SwiGLU"
        )
    if config.swiglu_limit != SWIGLU_LIMIT:
        raise NotImplementedError(
            f"[top] GLM-5.3-Flash clamps SwiGLU at {SWIGLU_LIMIT}; got {config.swiglu_limit}"
        )
    validate_ep_runtime()
