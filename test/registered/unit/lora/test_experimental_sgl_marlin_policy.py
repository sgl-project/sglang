"""Guards the LoRA + non-trivial expert-placement startup rejection; reds when a
placement term leaves the or-chain, or when a `--lora-paths`-only launch stops
counting as LoRA, silently landing adapter deltas on remapped experts."""

import types

import pytest

from sglang.srt.lora.marlin_lora_temp.policy import (
    validate_experimental_sgl_marlin_server_args,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


# Both spellings of "this server serves adapters". The validator must apply the
# same guards to each; only the second one exercises the tri-state.
LORA_LAUNCH_FORMS = (
    {"enable_lora": True, "lora_paths": []},
    {"enable_lora": None, "lora_paths": ["adapter=/tmp/adapter"]},
)


def _validate_server(**overrides):
    """Runs the real validator on a minimal stand-in; the resolved view is fixed
    at `ep_size=4` / `moe_a2a_backend="none"` so the placement chain is reached."""
    server_args = dict(
        enable_lora=True,
        lora_paths=[],
        lora_use_virtual_experts=True,
        lora_backend="triton",
        init_expert_location="trivial",
        ep_num_redundant_experts=0,
        enable_eplb=False,
        elastic_ep_backend=None,
        enable_elastic_expert_backup=False,
        elastic_ep_rejoin=False,
    )
    server_args.update(overrides)
    return validate_experimental_sgl_marlin_server_args(
        types.SimpleNamespace(**server_args),
        types.SimpleNamespace(ep_size=4, moe_a2a_backend="none"),
    )


@pytest.mark.parametrize(
    "placement",
    [
        {"init_expert_location": "random"},
        {"ep_num_redundant_experts": 1},
        {"enable_eplb": True},
        {"elastic_ep_backend": "mooncake"},
        {"enable_elastic_expert_backup": True},
        {"elastic_ep_rejoin": True},
    ],
    ids=[
        "init_expert_location",
        "ep_num_redundant_experts",
        "enable_eplb",
        "elastic_ep_backend",
        "enable_elastic_expert_backup",
        "elastic_ep_rejoin",
    ],
)
def test_lora_ep_placement_validation(placement):
    """Guards the or-chain: reds when a placement term stops rejecting, or -- if
    only the `lora_paths` form reds -- when the implicit-enable tri-state
    collapsed and adapter-path launches bypass the LoRA guards."""
    for launch in LORA_LAUNCH_FORMS:
        with pytest.raises(ValueError, match="trivial expert placement"):
            _validate_server(**launch, **placement)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
