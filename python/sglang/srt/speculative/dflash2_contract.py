"""Checkpoint contracts shared by the dense DFlash2 serving adapters."""


def validate_dflash2(config):
    cfg = config.dflash_config
    block = cfg.get("block_size")
    taps, group = cfg.get("conv_kernel_size", 0), cfg.get("conv_group_size", 0)
    rank, top_k = cfg.get("selector_rank", 0), cfg.get("selector_top_k", 0)
    if (
        any(type(value) is not int for value in (block, taps, group, rank, top_k))
        or block != config.block_size
        or not 1 <= taps <= block
        or group < 1
        or config.hidden_size % group
        or rank < 1
        or not 1 <= top_k <= config.vocab_size
    ):
        raise ValueError("Invalid DFlash2 block, convolution or selector geometry")
    if cfg.get("conv_base_kernel_identity_folded") is not True:
        raise ValueError("DFlash2 export must include the folded convolution identity")
    if getattr(config, "num_experts", 0) or cfg.get("use_hc", False):
        raise ValueError("This DFlash2 adapter supports dense non-HC checkpoints only")


def checked_weights(model, weights):
    """Check HF keys before the fused/TP loader can silently ignore a tensor."""
    parameters = dict(model.named_parameters())
    expected = set()
    for name in parameters:
        if ".qkv_proj." in name:
            expected.update(
                name.replace("qkv_proj", part)
                for part in ("q_proj", "k_proj", "v_proj")
            )
        elif ".gate_up_proj." in name:
            expected.update(
                name.replace("gate_up_proj", part) for part in ("gate_proj", "up_proj")
            )
        else:
            expected.add(name)
    seen = set()
    for name, tensor in weights:
        if name not in expected or name in seen:
            raise ValueError(f"Unexpected or duplicate DFlash2 weight: {name}")
        if (
            name.startswith(("candidate_selector.", "mtp_prefix_conditioner."))
            or "_conv." in name
        ):
            import torch

            if (
                parameters[name].shape != tensor.shape
                or not torch.isfinite(tensor).all()
            ):
                raise ValueError(f"Invalid DFlash2 weight: {name}")
        seen.add(name)
        yield name, tensor
    if seen != expected:
        raise ValueError(f"Missing DFlash2 weights: {sorted(expected - seen)}")
