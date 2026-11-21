# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Opt-in FlashInfer HC sum backend for gated hyper-connections.

FlashInfer (``flashinfer.hyperconnection``) owns the HC math, the native weight
layouts, kernel dispatch, runtime scratch, and the Apply context with its row
slicing. SGLang owns the native parameters and their checkpoint loading, and
wires the Apply context and the prepared square sums (an HCSumState) through
the layer boundary.

A module's native parameters are the folded Down+Inject ``packed_down_weight``,
the interleaved Gemma RMS ``packed_norm_weight`` and the ordinary Up. A model's
load_weights stages the raw checkpoint Down, Inject and norm (the legacy
parameter names) on the CPU and, once it has read the checkpoint, packs them
straight into the native parameters; native tensors load as they are and are
never folded. That ends the initial load, before any state_dict export. Loaders
that write native storage directly (dummy, sharded, presharded, IPC) end it at
binding, which the model's precompile_kernels_after_loading hook runs before
CUDA graph capture. The runtime then holds the packed weights and a derived Up
layout, so every later weight write is rejected before it mutates anything.
All ranks must use the same backend support and FlashInfer installation. Native
presharded/IPC caches must be created with the same setting as the serving model.
"""

import logging
from functools import partial
from typing import Callable, Iterable, Iterator, Optional, Tuple

import torch
from torch import nn

from sglang.kernels.ops.gemm.hc_mix import _deterministic_inference
from sglang.srt.environ import envs
from sglang.srt.layers.hyperconnection import GatedResidual, HyperConnectionConfig
from sglang.srt.layers.layer_boundary.residual.gated import GatedResidualState
from sglang.srt.runtime_context import get_exec, get_parallel

logger = logging.getLogger(__name__)

_SUPPORTED_CAPABILITIES = ((10, 0), (10, 3))

_FLASHINFER_HC_API = (
    "hc_sum_support_reason",
    "hc_sum_weight_shapes",
    "pack_hc_sum_weights_into",
    "bind_hc_sum_weights",
    "HCSumRuntime",
)

# The legacy GatedResidual parameter names, which checkpoints use.
_RAW_CHECKPOINT_NAMES = (
    ("input_mix_weight_down.weight", "down"),
    ("block_inject_weight.weight", "inject"),
    ("hc_norm.weight", "norm"),
)
_NATIVE_KINDS = frozenset(("down", "norm"))
_NATIVE_PARAMETER_NAMES = ("packed_down_weight", "packed_norm_weight")

_MIXED_FORMATS = "raw checkpoint and native packed HC weights cannot be mixed"


def _flashinfer_hc():
    """flashinfer.hyperconnection, which must hold the whole HC sum API."""
    message = (
        "SGLANG_ENABLE_FLASHINFER_HC_SUM is set on a configuration that FlashInfer "
        "HC sum would serve, but flashinfer.hyperconnection does not provide its "
        f"API {list(_FLASHINFER_HC_API)}. Install a FlashInfer build containing "
        "HC sum support and its CUDA JIT sources, or unset "
        "SGLANG_ENABLE_FLASHINFER_HC_SUM."
    )
    try:
        import flashinfer.hyperconnection as api
    except ImportError as exc:
        raise ImportError(message) from exc
    missing = [name for name in _FLASHINFER_HC_API if not hasattr(api, name)]
    if missing:
        raise ImportError(f"{message} Missing: {missing}.")
    return api


def create_gated_residual(
    config: HyperConnectionConfig,
    *,
    use_mix: bool = True,
    use_combine: bool = True,
    role: Optional[str] = None,
) -> nn.Module:
    """A gated hyper-connection module, decided once at construction: a
    FlashInferGatedResidual when SGLANG_ENABLE_FLASHINFER_HC_SUM opts in and
    FlashInfer HC sum supports ``config``, else the legacy GatedResidual."""
    if envs.SGLANG_ENABLE_FLASHINFER_HC_SUM.get():
        reason = _unsupported_reason(
            config, use_mix=use_mix, use_combine=use_combine, role=role
        )
        if reason is None:
            return FlashInferGatedResidual(config)
        logger.warning_once(
            f"FlashInfer HC sum does not serve a hyper-connection ({reason}); "
            "it uses GatedResidual."
        )
    return GatedResidual(config, use_mix, use_combine, role)


def _unsupported_reason(
    config: HyperConnectionConfig,
    *,
    use_mix: bool,
    use_combine: bool,
    role: Optional[str],
) -> Optional[str]:
    """Why FlashInfer HC sum cannot serve ``config``; None when it can. What
    is decided here before asking FlashInfer needs no FlashInfer, so a
    configuration that passes it requires FlashInfer's API."""
    if not (use_mix and use_combine):
        return "it does not both mix and combine"
    if config.hc_count != 4 or (config.mtp_hc and role is not None and "mtp" in role):
        return "it does not have 4 streams"
    if not config.hc_per_branch_norm:
        return "its norm is not per branch"
    if config.params_dtype not in (torch.bfloat16, torch.float16):
        return f"its dtype is {config.params_dtype}"
    if (
        not torch.cuda.is_available()
        or torch.cuda.get_device_capability() not in _SUPPORTED_CAPABILITIES
    ):
        return "the GPU is not SM100 or SM103"
    compile_reason = _torch_compile_reason()
    if compile_reason is not None:
        return compile_reason
    if _deterministic_inference():
        return "--enable-deterministic-inference requires batch-invariant operations"
    return _flashinfer_hc().hc_sum_support_reason(
        hidden_size=config.hidden_size,
        lowrank=config.hc_lowrank,
        dtype=config.params_dtype,
        # Not the parameters' device: an IPC weight-cache client builds them on meta.
        device=torch.device("cuda", torch.cuda.current_device()),
        rms_eps=config.rms_norm_eps,
    )


def _torch_compile_reason() -> Optional[str]:
    # torch.compile would trace FlashInfer's Python runtime and Apply contexts.
    try:
        enable_torch_compile = get_exec().graph.enable_torch_compile
    except ValueError:  # Not published, as in unit tests.
        return None
    if enable_torch_compile:
        return "--enable-torch-compile is set"
    return None


class FlashInferGatedResidual(nn.Module):
    """A gated hyper-connection that both mixes and combines, served by
    FlashInfer HC sum. ``mix`` returns the mixed input and FlashInfer's
    HCApplyContext, whose ``apply`` writes the block output back and returns
    the next residual's HCSumState."""

    _derived_weight_cache_error = (
        "Online weight updates are not supported with FlashInfer HC sum "
        "(SGLANG_ENABLE_FLASHINFER_HC_SUM): its runtime and captured CUDA "
        "graphs hold the packed HC weights and a derived Up layout. Restart "
        "without it to use online weight updates."
    )

    def __init__(self, config: HyperConnectionConfig):
        super().__init__()
        self.config = config
        self.hc_count = config.hc_count
        self.hidden_size = config.hidden_size
        self.params_dtype = config.params_dtype
        down_shape, norm_shape = _flashinfer_hc().hc_sum_weight_shapes(
            config.hidden_size, config.hc_lowrank
        )
        # FlashInfer's native layouts: Down and Inject rows folded with the
        # norm gamma [4, ceil8(L + 4), H], and the Gemma RMS weights [H, 4].
        self.packed_down_weight = nn.Parameter(
            torch.empty(down_shape, dtype=config.params_dtype), requires_grad=False
        )
        self.packed_norm_weight = nn.Parameter(
            torch.empty(norm_shape, dtype=config.params_dtype), requires_grad=False
        )
        self.input_mix_weight_up = nn.Linear(
            config.hc_lowrank,
            config.hc_count * config.hidden_size,
            bias=False,
            dtype=config.params_dtype,
        )
        self._install_native_loaders()
        width = config.hc_count * config.hidden_size
        self._raw_weight_shapes = {
            "down": (config.hc_lowrank, width),
            "inject": (config.hc_count, width),
            "norm": (width,),
        }
        self._deterministic = _deterministic_inference()
        # CPU copies of raw checkpoint tensors until load_weights packs them.
        self._raw_weights = {}
        self._native_weights = set()
        self._up_loaded = False
        self._initial_load_done = False
        self._runtime = None
        self._bind_views()
        self.register_load_state_dict_pre_hook(_before_state_dict_load)

    def _install_native_loaders(self) -> None:
        self.packed_down_weight.weight_loader = partial(
            self._load_native_weight, "down"
        )
        self.packed_norm_weight.weight_loader = partial(
            self._load_native_weight, "norm"
        )
        # Up has the same layout in raw and native checkpoints, but it must
        # still be present before completing an ordinary checkpoint load.
        self.input_mix_weight_up.weight.weight_loader = partial(
            self._load_native_weight, "up"
        )

    def _bind_views(self) -> None:
        # Plain tensors on the parameters' storage, which the runtime's plan
        # holds; the offloader keeps parameters such attributes alias resident.
        self._packed_down = self.packed_down_weight.detach()
        self._packed_norm = self.packed_norm_weight.detach()
        self._up = self.input_mix_weight_up.weight.detach()

    def _storage_key(self) -> Tuple[Tuple[torch.device, torch.dtype, int], ...]:
        return tuple(
            (tensor.device, tensor.dtype, tensor.data_ptr())
            for tensor in (
                self.packed_down_weight,
                self.packed_norm_weight,
                self.input_mix_weight_up.weight,
            )
        )

    def _apply(self, fn, recurse=True):
        storage = self._storage_key()
        module = super()._apply(fn, recurse=recurse)
        # A conversion that cannot set .data in place replaces the Parameters.
        self._install_native_loaders()
        self._bind_views()
        self.params_dtype = self.packed_down_weight.dtype
        # Staged checkpoint tensors stay on CPU through device moves, but
        # must follow a dtype conversion before packing into native storage.
        self._raw_weights = {
            name: weight.to(dtype=self.params_dtype)
            for name, weight in self._raw_weights.items()
        }
        if self._storage_key() != storage:
            # The plan holds the old storage; bind the moved weights anew.
            self._runtime = None
        return module

    def _reject_after_initial_load(self) -> None:
        if self._initial_load_done:
            raise ValueError(self._derived_weight_cache_error)

    @torch.no_grad()
    def _stage_raw_weight(self, kind: str, loaded_weight: torch.Tensor) -> None:
        self._reject_after_initial_load()
        if self._native_weights:
            raise ValueError(_MIXED_FORMATS)
        shape = self._raw_weight_shapes[kind]
        if tuple(loaded_weight.shape) != shape:
            raise ValueError(
                f"raw HC {kind} weight has shape {tuple(loaded_weight.shape)}, "
                f"expected {shape}"
            )
        # A copy: loaders may reuse the buffer behind ``loaded_weight``.
        staged = torch.empty(shape, dtype=self.packed_down_weight.dtype, device="cpu")
        staged.copy_(loaded_weight)
        self._raw_weights[kind] = staged

    @torch.no_grad()
    def _load_native_weight(
        self, kind: str, param: nn.Parameter, loaded_weight: torch.Tensor
    ) -> None:
        self._reject_after_initial_load()
        if kind != "up" and self._raw_weights:
            raise ValueError(_MIXED_FORMATS)
        if loaded_weight.shape != param.shape:
            raise ValueError(
                f"native HC {kind} weight has shape {tuple(loaded_weight.shape)}, "
                f"expected {tuple(param.shape)}"
            )
        param.data.copy_(loaded_weight)
        if kind == "up":
            self._up_loaded = True
        else:
            self._native_weights.add(kind)

    def _finish_initial_load(self) -> None:
        if not self._up_loaded:
            raise ValueError("the initial HC checkpoint lacks the Up weight")
        if self._raw_weights:
            missing = sorted(self._raw_weight_shapes.keys() - self._raw_weights.keys())
            if missing:
                raise ValueError(
                    f"the raw HC checkpoint lacks {missing}: Down, Inject and the "
                    "norm are packed together"
                )
            self._pack_raw_weights()
        elif not self._native_weights:
            raise ValueError("the checkpoint has none of the HC weights")
        elif self._native_weights != _NATIVE_KINDS:
            missing = sorted(_NATIVE_KINDS - self._native_weights)
            raise ValueError(f"the native HC checkpoint lacks {missing}")
        else:
            # Native external exporters may leave padding unspecified. This
            # path owns the destination; IPC binding itself remains read-only.
            self.zero_packed_down_padding()
        self._initial_load_done = True

    def _pack_raw_weights(self) -> None:
        # Released first: a failed pack must not leave tensors a retry folds.
        raw, self._raw_weights = self._raw_weights, {}
        self._bind_views()
        _flashinfer_hc().pack_hc_sum_weights_into(
            raw["down"],
            raw["inject"],
            raw["norm"].view(self.hc_count, self.hidden_size),
            out_down=self._packed_down,
            out_norm=self._packed_norm,
        )

    @torch.no_grad()
    def zero_packed_down_padding(self) -> None:
        # Binding requires the rows after the L Down and hc_count Inject rows
        # to be zero; dummy initialization fills them.
        self.packed_down_weight[:, self.config.hc_lowrank + self.hc_count :].zero_()

    def bind(self) -> None:
        """Bind the native weights into a FlashInfer runtime. Ends the initial
        load; must run outside CUDA graph capture."""
        if self._raw_weights:
            raise ValueError(
                "raw HC checkpoint weights were staged but not packed; the "
                "model's load_weights packs them"
            )
        self._initial_load_done = True
        if self._runtime is not None:
            return
        api = _flashinfer_hc()
        self._bind_views()
        plan = api.bind_hc_sum_weights(
            self._packed_down,
            self._up,
            self._packed_norm,
            lowrank=self.config.hc_lowrank,
            rms_eps=self.config.rms_norm_eps,
        )
        self._runtime = api.HCSumRuntime(plan, deterministic=self._deterministic)

    def mix(self, residual: torch.Tensor, prepared=None):
        """The mixed input and the HCApplyContext of ``residual``. ``prepared``
        is the HCSumState of the apply that wrote ``residual``, which the caller
        guarantees nothing has moved or changed rows of since; None makes
        FlashInfer compute the sums."""
        if self._runtime is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "FlashInfer HC sum weights must be bound (the model's "
                    "precompile_kernels_after_loading) before CUDA graph capture"
                )
            self.bind()
        return self._runtime.mix(residual, prepared)


def _before_state_dict_load(module, state_dict, prefix, *_):
    module._reject_after_initial_load()
    raw = [
        prefix + raw_name
        for raw_name, _ in _RAW_CHECKPOINT_NAMES
        if prefix + raw_name in state_dict
    ]
    if raw:
        raise ValueError(
            f"raw HC checkpoint tensors {raw} load through the model's "
            "load_weights, which packs them; load_state_dict takes native ones"
        )


class FlashInferGatedResidualState(GatedResidualState):
    """GatedResidualState whose read carries FlashInfer's HCApplyContext to
    the write-back. The context validates the residual it is applied to and
    slices its own rows. ``attn_mix`` and ``ffn_mix`` return what
    FlashInferGatedResidual.mix does; ``attn_combine`` and ``ffn_combine``
    take the block output and the context, and return the next residual."""

    def __init__(self, **fields):
        super().__init__(**fields)
        self.context = None

    def _read(self, mix, residual, out_norm):
        if out_norm is not None:
            raise NotImplementedError(
                "a gated hyper-connection read with a separate norm; the read "
                "normalizes the streams itself"
            )
        hidden_states, self.context = mix(residual)
        return hidden_states, self.context.residual

    def apply_attn_combine(self, hidden_states, residual):
        return self.attn_combine(hidden_states, self.context.with_residual(residual))

    def apply_ffn_combine(self, hidden_states, residual):
        return self.ffn_combine(hidden_states, self.context.with_residual(residual))

    def clear_coefficients(self):
        self.context = None

    def slice_residual_attn_tp(self, residual):
        parallel = get_parallel()
        self.context = self.context.with_residual(residual).slice_rows(
            parallel.attn_tp_size, parallel.attn_tp_rank
        )
        return self.context.residual


def flashinfer_hc_modules(
    model: nn.Module,
) -> Iterator[Tuple[str, FlashInferGatedResidual]]:
    return (
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, FlashInferGatedResidual)
    )


def _before_model_state_dict_load(model, state_dict, prefix, *_):
    for name, module in flashinfer_hc_modules(model):
        _before_state_dict_load(module, state_dict, f"{prefix}{name}.")


def install_flashinfer_hc_load_guard(model: nn.Module) -> None:
    """Make load_state_dict on ``model`` check its FlashInfer HC modules
    before it writes into any submodule."""
    if next(flashinfer_hc_modules(model), None) is not None:
        model.register_load_state_dict_pre_hook(_before_model_state_dict_load)


def bind_flashinfer_hc(model: nn.Module) -> None:
    """Bind every FlashInfer HC module of ``model``. For the model's
    precompile_kernels_after_loading, which every loader reaches before CUDA
    graph capture."""
    for name, module in flashinfer_hc_modules(model):
        try:
            module.bind()
        except ValueError as exc:
            raise ValueError(f"{name}: {exc}") from exc


def zero_flashinfer_hc_padding(model: nn.Module) -> None:
    """For the model's post_load_weights, which the loaders that fill native
    storage without load_weights run; the IPC weight cache runs none."""
    for _, module in flashinfer_hc_modules(model):
        module.zero_packed_down_padding()


class FlashInferHCCheckpoint:
    """One load_weights call's raw HC checkpoint tensors, routed by their
    legacy parameter names to the FlashInfer HC modules that stage them, and
    packed once the call has read the checkpoint. Constructing it rejects a
    load after the initial one, before anything writes a weight."""

    def __init__(self, model: nn.Module):
        self._modules = dict(flashinfer_hc_modules(model))
        self._legacy_modules = {
            name
            for name, module in model.named_modules()
            if isinstance(module, GatedResidual)
        }
        for module in self._modules.values():
            module._reject_after_initial_load()

    def stage(self, name: str, loaded_weight: torch.Tensor) -> bool:
        """Stage ``loaded_weight`` when ``name``, a parameter name of the
        model, is a raw tensor of one of its FlashInfer HC modules."""
        prefix, _, leaf = name.rpartition(".")
        if leaf in _NATIVE_PARAMETER_NAMES and prefix in self._legacy_modules:
            raise ValueError(
                f"{name}: native FlashInfer HC weights cannot load into the legacy "
                "backend. Load the original checkpoint or enable a supported "
                "FlashInfer HC sum configuration."
            )
        if not self._modules:
            return False
        for raw_name, kind in _RAW_CHECKPOINT_NAMES:
            if name.endswith("." + raw_name):
                module = self._modules.get(name[: -len(raw_name) - 1])
                if module is None:
                    return False
                module._stage_raw_weight(kind, loaded_weight)
                return True
        return False

    def route(
        self,
        weights: Iterable[Tuple[str, torch.Tensor]],
        *,
        param_name: Callable[[str], Optional[str]],
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        """``weights`` without the raw tensors staged here. ``param_name``
        maps a checkpoint name to the model's parameter name, or to None for
        a tensor the model does not load."""
        for name, loaded_weight in weights:
            mapped = param_name(name)
            if mapped is not None and self.stage(mapped, loaded_weight):
                continue
            yield name, loaded_weight

    def finish(self) -> set:
        """Pack each module's staged raw tensors, after requiring every module
        to have loaded all of its weights in one format. Returns the native
        parameter names written."""
        loaded = set()
        for name, module in self._modules.items():
            try:
                module._finish_initial_load()
            except ValueError as exc:
                raise ValueError(f"{name}: {exc}") from exc
            loaded.update(f"{name}.{param}" for param in _NATIVE_PARAMETER_NAMES)
        return loaded
