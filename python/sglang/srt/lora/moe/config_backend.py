"""Production config boundary for the BF16 MoE LoRA engine.

The selector is intentionally pure, while :class:`MoeLoraConfigBackend`
owns all stateful execution objects.  Every runner that the selector may
return is constructed and factor-validated when the resident LoRA buffers are
bound.  A forward, including CUDA-graph capture, therefore performs only a
pure config lookup followed by a dictionary lookup; it can never initialize a
provider, stream, event, or runner lazily.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from sglang.srt.lora.moe.config import (
    ConfigChoice,
    ConfigInput,
    Phase,
    architecture_for_capability,
    choices_for,
    select_config,
)
from sglang.srt.lora.moe.execution_plan import (
    ActivationFamily,
)
from sglang.srt.lora.moe.moe_lora_runner import (
    MoeLoraBatch,
    MoeLoraRunner,
)
from sglang.srt.lora.moe.workspace import MoeLoraWorkspace

if TYPE_CHECKING:
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardCombineInput,
        StandardDispatchOutput,
    )


logger = logging.getLogger(__name__)


class MoeLoraConfigBackend:
    """One MoE layer's prevalidated config choices and shared workspace.

    The layer's activation and GPU architecture are static.  Factor layout is
    learned when the LoRA memory pool binds its resident buffers; that is the
    last point at which this object may construct runners or providers.
    """

    _config_logged = False

    def __init__(
        self,
        base_layer: FusedMoE,
        *,
        capability: tuple[int, int],
        activation: ActivationFamily,
        hidden_size: int,
        num_local_experts: int,
        workspace: MoeLoraWorkspace | None = None,
    ) -> None:
        self._base_layer = base_layer
        self.capability_major, self.capability_minor = capability
        self.architecture = architecture_for_capability(*capability)
        self.activation = activation
        self.hidden_size = int(hidden_size)
        self.num_local_experts = int(num_local_experts)
        self.workspace = workspace if workspace is not None else MoeLoraWorkspace()
        self._is_shared_outer: bool | None = None
        self._runners: dict[str, MoeLoraRunner] = {}
        self._choices: tuple[ConfigChoice, ...] = ()

    @classmethod
    def from_layer(
        cls,
        base_layer: FusedMoE,
        *,
        workspace: MoeLoraWorkspace | None = None,
    ) -> MoeLoraConfigBackend:
        """Read layer-static config inputs without constructing a provider."""
        weight_device = base_layer.w2_weight.device
        if weight_device.type != "cuda":
            raise NotImplementedError("MoE LoRA config requires a CUDA layer")
        capability = torch.cuda.get_device_capability(weight_device)
        config = base_layer.moe_runner_config
        if config.activation == "silu" and config.is_gated:
            activation = ActivationFamily.SWIGLU
        elif config.activation == "relu2" and not config.is_gated:
            activation = ActivationFamily.RELU2
        else:
            raise NotImplementedError(
                "MoE LoRA config supports gated SiLU or non-gated ReLU2"
            )
        return cls(
            base_layer,
            capability=capability,
            activation=activation,
            hidden_size=int(base_layer.w2_weight.shape[1]),
            num_local_experts=int(base_layer.num_local_experts),
            workspace=workspace,
        )

    @property
    def is_bound(self) -> bool:
        return self._is_shared_outer is not None

    @property
    def choices(self) -> tuple[ConfigChoice, ...]:
        """All choices available after factor binding."""
        return self._choices

    def bind_factors(
        self,
        *,
        gate_up_lora_a: torch.Tensor,
        gate_up_lora_b: torch.Tensor,
        down_lora_a: torch.Tensor,
        down_lora_b: torch.Tensor,
        is_shared_outer: bool,
    ) -> None:
        """Construct and validate the complete runner set before capture.

        Initial binding is transactional: an unsupported choice leaves this
        backend unbound rather than exposing a partially populated config.
        Rebinding a compatible resident buffer contract revalidates the
        existing runners and never recreates providers.
        """
        selected_choices = choices_for(
            self.architecture,
            is_shared_outer,
            self.activation,
            hidden_size=self.hidden_size,
            num_local_experts=self.num_local_experts,
        )
        factor_kwargs = {
            "gate_up_lora_a": gate_up_lora_a,
            "gate_up_lora_b": gate_up_lora_b,
            "down_lora_a": down_lora_a,
            "down_lora_b": down_lora_b,
            "is_shared_outer": is_shared_outer,
        }
        if self.is_bound:
            if is_shared_outer != self._is_shared_outer:
                raise ValueError(
                    "resident MoE-LoRA factor layout changed after config binding"
                )
            if tuple(choice.key for choice in selected_choices) != tuple(
                choice.key for choice in self._choices
            ):
                raise RuntimeError("the bound MoE-LoRA config choice set changed")
            for runner in self._runners.values():
                runner.validate_factors(**factor_kwargs)
            return

        runners: dict[str, MoeLoraRunner] = {}
        for choice in selected_choices:
            if choice.key in runners:
                raise RuntimeError(f"duplicate MoE-LoRA config key {choice.key!r}")
            runner = MoeLoraRunner.from_layer(
                self._base_layer,
                provider_name=choice.provider,
                execution_plan=choice.plan,
                launch_config=choice.launch_config,
                workspace=self.workspace,
            )
            runner.validate_factors(**factor_kwargs)
            runners[choice.key] = runner

        if not runners:
            raise RuntimeError("the MoE-LoRA config produced no executable choices")
        self._runners = runners
        self._choices = selected_choices
        self._is_shared_outer = is_shared_outer
        # Every MoE layer of a model resolves the same menu; log it once.
        if not MoeLoraConfigBackend._config_logged:
            MoeLoraConfigBackend._config_logged = True
            logger.info(
                "MoE LoRA config bound (%s, hidden=%d, local_experts=%d): %s",
                self.architecture.value,
                self.hidden_size,
                self.num_local_experts,
                ", ".join(f"{c.key}@{c.provider}" for c in selected_choices),
            )

    def select(
        self,
        batch: MoeLoraBatch,
        *,
        num_tokens: int,
    ) -> ConfigChoice:
        """Select an already-created runner for every batch.

        Config uses the resident physical rank in eager and graph mode.  The
        current kernels contract over the padded resident factor tensors, so
        the logical adapter rank does not reduce their GEMM K dimension.
        Selecting a logical-rank-tuned launch here would therefore describe a
        different workload than the one actually executed.

        Base-only eager batches intentionally keep the same runner topology as
        active and graph batches.  Besides keeping one ownership model, this
        avoids calling a resident base MoE runner that may have been created
        with in-place output before the LoRA wrapper attached; such a call can
        mutate the shared-expert input tensor needed after routed experts.
        """
        self._check_bound_batch(batch)
        resolved_mode = Phase.PREFILL if batch.is_prefill else Phase.DECODE
        active_rank = batch.physical_rank
        choice = select_config(
            ConfigInput(
                capability_major=self.capability_major,
                capability_minor=self.capability_minor,
                is_shared_outer=batch.is_shared_outer,
                activation=self.activation,
                mode=resolved_mode,
                num_tokens=int(num_tokens),
                active_rank=int(active_rank),
                hidden_size=self.hidden_size,
                num_local_experts=self.num_local_experts,
                has_active_lora=bool(batch.has_active_lora),
                use_cuda_graph=bool(batch.use_cuda_graph),
            )
        )
        if choice.key not in self._runners:
            raise RuntimeError(
                f"config selected unbound MoE-LoRA runner {choice.key!r}"
            )
        return choice

    def run_selected(
        self,
        choice: ConfigChoice,
        dispatch_output: StandardDispatchOutput,
        batch: MoeLoraBatch,
        *,
        output_dtype: torch.dtype | None = None,
    ) -> StandardCombineInput:
        """Execute a choice returned by :meth:`select` after dispatch."""
        self._check_bound_batch(batch)
        runner = self._runners.get(choice.key)
        if runner is None:
            raise ValueError(f"choice {choice.key!r} is not bound to this layer")
        bound_choice = next(
            (candidate for candidate in self._choices if candidate.key == choice.key),
            None,
        )
        if bound_choice != choice:
            raise ValueError(
                f"choice {choice.key!r} does not match this layer's bound config"
            )
        return runner.run(
            dispatch_output,
            batch,
            output_dtype=output_dtype,
        )

    def _check_bound_batch(self, batch: MoeLoraBatch) -> None:
        if not self.is_bound:
            raise RuntimeError("MoE LoRA factors must be bound before config selection")
        if batch.is_shared_outer != self._is_shared_outer:
            raise ValueError(
                "batch MoE-LoRA factor layout does not match the bound config"
            )
