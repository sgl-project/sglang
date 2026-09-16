"""A draft entry class answers the loader's questions exactly when its target does.

The loader asks the *entry class* it instantiates for the shared-experts-fusion
decision (`install_shared_experts_fusion_decision`). A draft is its own entry
class -- `...NextN`, `...MTP`, `...DSpark`, `...Eagle3` -- so when the target
family carries auto-disable conditions and the draft's class does not expose
them, the draft resolves a *different* decision than the target it drafts for,
and its weights are laid out for the other layout. That shipped once: the DSV4
DSpark draft skipped its bundled shared-expert tensors until #33312 gave it the
gate.

For drafts with one target identifiable by name, check hook presence parity.
Identity is deliberately not required -- a draft that delegates with adapted
arguments (the Qwen3.5 MTP unwraps `text_config` and substitutes the MTP
quantization config) is exactly right. Bailing shares one NextN entry across
several target versions, so check its decisions with each target's config.
"""

import re
import unittest
from itertools import product
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.layers.moe.utils import (
    MoeA2ABackend,
    draft_model_build_scope,
    install_shared_experts_fusion_decision,
)
from sglang.srt.models.registry import ModelRegistry
from sglang.srt.runtime_context import get_context, get_flags
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=14, suite="base-a-test-cpu")

# The hooks the loader resolves on the entry class where a draft/target
# disagreement is a defect rather than a difference. Weight-name maps
# (`packed_modules_mapping`, `hf_to_sglang_mapper`) are deliberately absent: a
# draft's checkpoint has its own names, so those differ by design.
PARITY_HOOKS = ("shared_experts_fusion_disable_reason",)

# `...Eagle`/`...Eagle3` drafts are standalone checkpoints with their own
# architecture; the rest mirror a stage of their target.
DRAFT_SUFFIX = re.compile(r"(NextN|MTP|DSpark|DFlash|Standalone)$")


def _target_of(arch: str, archs: dict):
    """The arch a draft entry class drafts for, if it is named after it."""
    if arch == "BailingMoeForCausalLMNextN":
        # V1/V2/V2.5/V3 share this entry. Their decisions are checked below;
        # removing NextN would incorrectly pair every config with V1.
        return None
    match = DRAFT_SUFFIX.search(arch)
    if not match:
        return None
    base = arch[: match.start()]
    for candidate in (base, f"{base}ForCausalLM"):
        if candidate in archs and candidate != arch:
            return candidate
    return None


class TestDraftEntryHookParity(CustomTestCase):
    def test_bailing_nextn_matches_each_target_configuration(self):
        from sglang.srt.models import bailing_moe_v3

        draft_arch = "BailingMoeForCausalLMNextN"
        draft_cls = ModelRegistry.resolve_model_cls(draft_arch)[0]
        # The four targets rewritten to this entry by ModelConfig.
        targets = (
            ("BailingMoeForCausalLM", "bailing_moe", False),
            ("BailingMoeV2ForCausalLM", "bailing_moe", False),
            ("BailingMoeV2_5ForCausalLM", "bailing_hybrid", False),
            ("BailingMoeV3ForCausalLM", "bailing_hybrid", True),
        )
        quant_configs = (None, SimpleNamespace(get_name=lambda: "w4afp8"))
        # Exercise both outcomes on CPU CI using supported-device metadata;
        # the production gates and decision installer run without substitutes.
        with (
            patch.object(bailing_moe_v3, "_is_cuda", True),
            patch.object(
                bailing_moe_v3.torch.cuda, "get_device_capability", return_value=(9, 0)
            ),
        ):
            for target, quant_config, user_disabled in product(
                targets, quant_configs, (False, True)
            ):
                target_arch, model_type, use_kda = target
                target_cls = ModelRegistry.resolve_model_cls(target_arch)[0]
                config = dict(
                    model_type=model_type,
                    use_kda=use_kda,
                    num_shared_experts=1,
                    moe_intermediate_size=1024,
                )
                target_config = SimpleNamespace(architectures=[target_arch], **config)
                draft_config = SimpleNamespace(architectures=[draft_arch], **config)
                expected_disabled = user_disabled or (
                    use_kda and quant_config is not None
                )
                moe = get_flags().moe
                with (
                    self.subTest(
                        target=target_arch,
                        quantized=quant_config is not None,
                        user_disabled=user_disabled,
                    ),
                    get_context().override_server_args(
                        disable_shared_experts_fusion=user_disabled
                    ),
                    patch.multiple(
                        moe,
                        a2a_backend=MoeA2ABackend.NONE,
                        disable_shared_experts_fusion=None,
                        speculative_disable_shared_experts_fusion=None,
                        in_speculative_scope=False,
                    ),
                ):
                    install_shared_experts_fusion_decision(
                        target_cls, target_config, quant_config
                    )
                    self.assertEqual(
                        moe.disable_shared_experts_fusion, expected_disabled
                    )
                    with draft_model_build_scope():
                        install_shared_experts_fusion_decision(
                            draft_cls, draft_config, quant_config
                        )
                        self.assertEqual(
                            moe.disable_shared_experts_fusion, expected_disabled
                        )
                        self.assertEqual(
                            moe.speculative_disable_shared_experts_fusion,
                            expected_disabled,
                        )
                    self.assertEqual(
                        moe.disable_shared_experts_fusion, expected_disabled
                    )

    def test_a_draft_resolves_a_gate_exactly_when_its_target_does(self):
        archs = {}
        for arch in sorted(ModelRegistry.get_supported_archs()):
            try:
                archs[arch] = ModelRegistry.resolve_model_cls(arch)[0]
            except Exception:
                continue  # an arch this environment cannot import

        checked, offenders = 0, []
        for arch, cls in archs.items():
            target = _target_of(arch, archs)
            if target is None:
                continue
            for hook in PARITY_HOOKS:
                checked += 1
                draft_has = hasattr(cls, hook)
                target_has = hasattr(archs[target], hook)
                if draft_has == target_has:
                    continue
                which = "the draft" if target_has else "the target"
                offenders.append(
                    f"{arch} vs {target}: {hook} missing on {which} "
                    f"({cls.__module__} / {archs[target].__module__})"
                )

        self.assertGreater(checked, 10, "the draft/target pairing found nothing")
        self.assertEqual(
            [],
            offenders,
            "a draft entry class and the target it drafts for must resolve the "
            "same loader hooks; otherwise the loader installs one decision for "
            "the target and another for the draft, and the draft's weights are "
            "laid out for the wrong one:\n  " + "\n  ".join(offenders),
        )


if __name__ == "__main__":
    unittest.main()
