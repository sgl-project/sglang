"""Single-storage GPU parameters for the packed HyperConnection path.

Checkpoint Down/Inject/RMS tensors are mapped into the native parameter layout.
Native state dictionaries contain already-folded Down+Inject and interleaved RMS;
they must not be folded again. No original checkpoint tensors are retained.
"""

from functools import cache
from typing import Optional

import torch
from torch import nn

from sglang.kernels.ops.gemm.hc_mix import _deterministic_inference
from sglang.srt.layers.hyperconnection import GatedResidual, HyperConnectionBase


@cache
def _sum_state_type():
    from sglang.kernels.ops.elementwise.hc_sum_state import HCSumState

    return HCSumState


class _HCSumContext:
    """Mix-owned buffers carried through the active attention/MoE boundary."""

    def __init__(self, operator, state, buffers):
        self.operator = operator
        self.state = state
        self.buffers = buffers

    def slice_rows(self, rank, size):
        def split(tensor):
            return tensor.tensor_split(size)[rank]

        state = self.state._replace(
            residual=split(self.state.residual), sum_sq=split(self.state.sum_sq)
        )
        output = self.buffers.output._replace(
            residual=split(self.buffers.output.residual),
            sum_sq=split(self.buffers.output.sum_sq),
        )
        buffers = self.buffers._replace(
            mixed=split(self.buffers.mixed),
            alpha=split(self.buffers.alpha),
            output=output,
            partial_sums=(
                split(self.buffers.partial_sums)
                if self.buffers.partial_sums is not None
                else None
            ),
        )
        return _HCSumContext(self.operator, state, buffers)

    def combine(self, block_output):
        from sglang.kernels.ops.elementwise.hc_apply_sum import hc_apply_sum

        # Mix owns the output storage. Apply validates the new core output,
        # including the 16-byte-aligned statistics of an odd-row TP slice.
        hc_apply_sum(
            block_output,
            self.state.residual,
            self.buffers.alpha,
            out=self.buffers.output.residual,
            sum_sq=self.buffers.output.sum_sq,
            partial_sums=self.buffers.partial_sums,
        )
        return self.buffers.output


class _PackedParameter(nn.Module):
    def __init__(self, shape, dtype):
        super().__init__()
        # Honor the enclosing device context, including meta initialization.
        self.weight = nn.Parameter(torch.empty(shape, dtype=dtype), requires_grad=False)


class PackedGatedResidual(GatedResidual):
    """Native Down+Inject/RMS storage, shared by the small- and large-T kernels.

    Checkpoint weights are folded once after the initial load. Online updates
    are unsupported; native checkpoints contain the already-folded parameters.
    """

    _derived_weight_cache_error = (
        "Online weight updates are not supported with packed HyperConnection "
        "weights in this version. Reload the model to change its weights."
    )
    # Also reject sessions that bypass loaders and write native storage directly.
    _reject_weight_update_sessions = True

    def __init__(self, config, role: Optional[str] = None):
        HyperConnectionBase.__init__(self, config, True, True, role)
        c, h, lowrank = self.hc_count, self.hidden_size, config.hc_lowrank
        if lowrank <= 0 or lowrank % 64:
            raise ValueError("packed HC requires a positive hc_lowrank aligned to 64")
        npad = ((lowrank + c + 7) // 8) * 8
        self._packed_sum_weights = True
        self._deterministic_hc = _deterministic_inference()
        self.input_mix_weight_down = _PackedParameter((c, npad, h), config.params_dtype)
        self.hc_norm = _PackedParameter((h, c), config.params_dtype)
        self.input_mix_weight_up = nn.Linear(
            lowrank, c * h, bias=False, dtype=config.params_dtype
        )
        self._sum_operators = None
        self._sum_weights_dirty = True
        self._hc_loaded_weights = set()
        self._hc_load_format = None
        self._hc_weights_ready = False
        self._install_weight_loaders()
        self.register_load_state_dict_pre_hook(self._before_native_state_load)
        self.register_load_state_dict_post_hook(self._after_native_state_load)

    def can_use_sum_state(self, residual):
        if (
            torch.is_grad_enabled()
            or self._sum_operators is None
            or self._sum_weights_dirty
        ):
            return False
        large = self._sum_operators[1]
        w = large.weights
        return (
            residual.ndim == 2
            and residual.shape[1] == 4 * self.hidden_size
            and residual.shape[0] <= large.max_rows
            and residual.device == w.matrix.device
            and residual.dtype == w.matrix.dtype
            and residual.is_contiguous()
            and residual.data_ptr() % 32 == 0
        )

    def mix(self, hyper_input):
        is_state = isinstance(hyper_input, _sum_state_type())
        residual = hyper_input.residual if is_state else hyper_input
        if not self.can_use_sum_state(residual):
            raise ValueError(
                "Packed HC requires prepared weights and supported inference input"
            )
        operator = self._sum_operators[int(residual.shape[0] > 24)]
        state = hyper_input if is_state else operator.bootstrap(residual)
        mixed, buffers = operator._mix_allocated(state)
        return mixed, _HCSumContext(operator, state, buffers)

    def mix_prepared(self, residual, prepared_input=None):
        if isinstance(prepared_input, _sum_state_type()):
            previous = prepared_input.residual
            same_rows = previous is residual or (
                previous.shape == residual.shape
                and previous.stride() == residual.stride()
                and previous.dtype == residual.dtype
                and previous.device == residual.device
                and previous.data_ptr() == residual.data_ptr()
            )
            if same_rows:
                # An odd row slice offsets [T,4] statistics by 16 bytes.
                if prepared_input.sum_sq.data_ptr() % 32:
                    prepared_input = prepared_input._replace(
                        sum_sq=prepared_input.sum_sq.clone()
                    )
                return self.mix(prepared_input)
        return self.mix(residual)

    def combine(self, block_output, residuals):
        return residuals.combine(block_output)

    def combine_and_prepare(self, block_output, residuals, next_norm=None):
        state = self.combine(block_output, residuals)
        return state.residual, state

    @property
    def _hc_packed_down(self):
        return self.input_mix_weight_down.weight

    @property
    def _hc_packed_norm(self):
        # Never materialize a hidden copy of an imported/assigned parameter.
        return self.hc_norm.weight.view(-1)

    def _install_weight_loaders(self):
        for parameter in self.parameters():
            parameter.weight_loader = self._load_sum_source_weight

    @torch.no_grad()
    def _load_sum_source_weight(self, parameter, loaded_weight, *, copy_value=True):
        if self._hc_weights_ready:
            raise ValueError(self._derived_weight_cache_error)
        c, h, lowrank = self.hc_count, self.hidden_size, self.config.hc_lowrank
        if parameter is self.input_mix_weight_up.weight:
            if loaded_weight.shape != parameter.shape:
                raise ValueError("HC Up weight shape mismatch")
            key, source_format, value, destination = (
                "up",
                None,
                loaded_weight,
                parameter,
            )
        elif parameter is self.input_mix_weight_down.weight:
            if loaded_weight.shape == parameter.shape:
                key, source_format = "packed", "native"
                value, destination = loaded_weight, parameter
            elif loaded_weight.shape in ((lowrank, c * h), (c, c * h)):
                is_inject = loaded_weight.shape[0] == c
                key, source_format = ("inject" if is_inject else "down"), "checkpoint"
                start = lowrank if is_inject else 0
                rows = c if is_inject else lowrank
                destination = parameter[:, start : start + rows]
                value = loaded_weight.reshape(rows, c, h).permute(1, 0, 2)
            else:
                raise ValueError("HC Down/Inject checkpoint or packed shape mismatch")
        elif parameter is self.hc_norm.weight:
            key = "norm"
            if loaded_weight.shape == (c * h,):
                source_format = "checkpoint"
                value = loaded_weight.view(c, h).T
            elif loaded_weight.shape == parameter.shape:
                source_format, value = "native", loaded_weight
            else:
                raise ValueError("HC RMS checkpoint or packed shape mismatch")
            destination = parameter
        else:
            raise ValueError("Unknown packed HC parameter")
        if source_format is not None:
            if self._hc_load_format not in (None, source_format):
                raise ValueError(
                    "Cannot mix raw checkpoint and native packed HC weights"
                )
            self._hc_load_format = source_format
        self._sum_weights_dirty = True
        if copy_value:
            destination.copy_(value)
        self._hc_loaded_weights.add(key)

    def offload_resident_parameters(self):
        # Both shells retain views of these tensors (and a padded Up layout).
        # Parameter-only functional_call offload cannot replace those views.
        return self.parameters()

    def _validate_initial_weights(self):
        loaded = self._hc_loaded_weights
        if self._hc_load_format == "checkpoint":
            if not {"down", "inject", "norm", "up"} <= loaded:
                raise ValueError(
                    "Initial packed HC loading requires Down, Inject, RMS and Up"
                )
        elif self._hc_load_format == "native":
            if not {"packed", "norm", "up"} <= loaded:
                raise ValueError(
                    "Initial native HC loading requires packed Down, RMS and Up"
                )
        else:
            raise ValueError("Packed HC weights have not been loaded")

    @torch.no_grad()
    def _fold_checkpoint_weights(self):
        packed = self.input_mix_weight_down.weight
        norm = self.hc_norm.weight
        # Temporary FP32 storage is bounded by one branch, not another model.
        for branch in range(self.hc_count):
            gamma = 1.0 + norm[:, branch].float()
            live = packed[branch, : self.config.hc_lowrank + self.hc_count]
            live.copy_((live.float() * gamma).to(packed.dtype))
        packed[:, self.config.hc_lowrank + self.hc_count :].zero_()
        # Binding can fail independently (e.g. compilation); a retry must not
        # fold a second time after this conversion has already succeeded.
        self._hc_loaded_weights.difference_update(("down", "inject"))
        self._hc_loaded_weights.add("packed")
        self._hc_load_format = "native"

    @torch.no_grad()
    def prepare_sum_state_weights(self, *, force=False):
        if self._hc_weights_ready:
            return True
        if force and not self._hc_loaded_weights:
            # Dummy/sharded loaders initialize the registered native tensors
            # directly. They need local views, not another checkpoint fold.
            self._hc_load_format = "native"
            self._hc_loaded_weights.update(("packed", "norm", "up"))
        self._validate_initial_weights()
        if self._hc_load_format == "checkpoint":
            self._fold_checkpoint_weights()
        self._bind_native_weights()
        self._hc_weights_ready = True
        self._hc_loaded_weights.clear()
        self._hc_load_format = None
        self._sum_weights_dirty = False
        return True

    @torch.no_grad()
    def _bind_native_weights(self):
        from sglang.kernels.ops.elementwise.hc_down_batched_sum import (
            BatchedDownWeights,
        )
        from sglang.kernels.ops.elementwise.hc_sum_state import SmallTSumHCShell
        from sglang.kernels.ops.elementwise.hc_sum_state_large import LargeTSumHCShell

        packed = self._hc_packed_down
        if not self.hc_norm.weight.is_contiguous():
            raise ValueError("Native HC RMS parameter must be contiguous")
        norm = self._hc_packed_norm
        up = self.input_mix_weight_up.weight
        npad = ((self.config.hc_lowrank + self.hc_count + 7) // 8) * 8
        if packed.shape != (self.hc_count, npad, self.hidden_size):
            raise ValueError("Invalid native HC Down+Inject layout")
        if self.hc_norm.weight.shape != (self.hidden_size, self.hc_count):
            raise ValueError("Invalid native HC RMS layout")
        if not all(t.is_contiguous() for t in (packed, norm, up)):
            raise ValueError("Native HC parameters must be contiguous")
        if any(
            t.device != packed.device or t.dtype != packed.dtype for t in (norm, up)
        ):
            raise ValueError("Native HC parameters must share device and dtype")
        if packed.dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("Native HC parameters require BF16 or FP16")
        if not packed.is_cuda:
            # Uniform CPU/meta placement is valid before onload; mixed
            # CPU/CUDA parameters are not a successfully prepared runtime state.
            self._sum_operators = None
            return
        weights = BatchedDownWeights(
            packed.transpose(1, 2),
            self.hidden_size,
            self.config.hc_lowrank,
            self.hc_count,
        )
        large = LargeTSumHCShell(
            self,
            down_weights=weights,
            norm_weight_permuted=norm,
            deterministic=self._deterministic_hc,
        )
        small = SmallTSumHCShell(
            self,
            down_weights=weights,
            norm_weight_permuted=norm,
            deterministic=self._deterministic_hc,
        )
        self._sum_operators = (small, large)
        self._install_weight_loaders()

    @torch.no_grad()
    def restore_sum_state_from_ipc(self):
        if self._hc_load_format == "checkpoint":
            raise ValueError("Preprocessed HC loading cannot contain raw weights")
        self.prepare_sum_state_weights(force=True)

    def _apply(self, fn, recurse=True):
        self._sum_operators = None
        result = nn.Module._apply(self, fn, recurse=recurse)
        self._install_weight_loaders()
        if self._hc_weights_ready and not self._hc_loaded_weights:
            self._bind_native_weights()
        return result

    def _before_native_state_load(
        self,
        module,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        if self._hc_weights_ready:
            raise ValueError(self._derived_weight_cache_error)
        assign = local_metadata.get("assign_to_params_buffers", False)
        names = {
            "input_mix_weight_down.weight": self.input_mix_weight_down.weight,
            "block_inject_weight.weight": self.input_mix_weight_down.weight,
            "hc_norm.weight": self.hc_norm.weight,
            "input_mix_weight_up.weight": self.input_mix_weight_up.weight,
        }
        selected = {
            name: state_dict[prefix + name]
            for name in names
            if prefix + name in state_dict
        }
        down = selected.get("input_mix_weight_down.weight")
        norm = selected.get("hc_norm.weight")
        raw = (
            "block_inject_weight.weight" in selected
            or (down is not None and down.ndim == 2)
            or (norm is not None and norm.ndim == 1)
        )
        if raw and assign:
            raise ValueError(
                "Raw HC checkpoint loading requires assign=False; "
                "assign=True supports native packed state"
            )
        if raw:
            if self._hc_packed_down.is_meta:
                raise ValueError(
                    "Materialize HC before raw checkpoint loading; "
                    "meta assign supports native packed state"
                )
            for name, value in selected.items():
                self._load_sum_source_weight(
                    names[name], value, copy_value=name != "input_mix_weight_up.weight"
                )
            self._validate_initial_weights()
            self._fold_checkpoint_weights()
            state_dict.pop(prefix + "block_inject_weight.weight", None)
            if down is not None or "block_inject_weight.weight" in selected:
                state_dict[prefix + "input_mix_weight_down.weight"] = (
                    self._hc_packed_down.detach()
                )
            if "hc_norm.weight" in selected:
                state_dict[prefix + "hc_norm.weight"] = self.hc_norm.weight.detach()
        else:
            for name, value in selected.items():
                # Ordinary Module loading performs the one native copy/assign
                # after this pre-hook; here only validate and mark dependencies.
                self._load_sum_source_weight(names[name], value, copy_value=False)

    def _after_native_state_load(self, module, incompatible_keys):
        if not self._hc_weights_ready and not self._hc_loaded_weights:
            return
        # Native assign may replace Parameter objects during the initial load.
        self.prepare_sum_state_weights()
        self._install_weight_loaders()


def check_packed_hc_initial_load(model, *_):
    """Reject reloads before a model loader or state-dict traversal writes anything."""
    for module in model.modules():
        if isinstance(module, PackedGatedResidual) and module._hc_weights_ready:
            raise ValueError(module._derived_weight_cache_error)


def prepare_packed_hc_weights(model, *, force=True, weights_preprocessed=False):
    """Bind initial HC weights; preprocessed imports must never be folded again."""
    for name, module in model.named_modules():
        if isinstance(module, PackedGatedResidual):
            try:
                if weights_preprocessed:
                    module.restore_sum_state_from_ipc()
                else:
                    module.prepare_sum_state_weights(force=force)
            except ValueError as exc:
                raise ValueError(f"{name or '<root>'}: {exc}") from exc


def create_gated_residual(config, use_mix=True, use_combine=True, role=None):
    c, h, lowrank = config.hc_count, config.hidden_size, config.hc_lowrank
    if (
        use_mix
        and use_combine
        and config.hc_per_branch_norm
        and c == 4
        and not (config.mtp_hc and role is not None and "mtp" in role)
        and 0 < h <= 16384
        and h % 512 == 0
        and lowrank > 0
        # Small-T Down keeps Inject in a separate, CTA-aligned alpha-only tile.
        and lowrank % 64 == 0
        and lowrank + c <= ((lowrank + 127) // 128) * 128
        and config.params_dtype in (torch.bfloat16, torch.float16)
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] == 10
    ):
        return PackedGatedResidual(config, role)
    return GatedResidual(config, use_mix, use_combine, role)
