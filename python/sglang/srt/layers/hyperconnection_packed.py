"""Single-storage GPU parameters for the packed HyperConnection path.

Checkpoint Down/Inject/RMS tensors are mapped into the native parameter layout.
Native state dictionaries contain already-folded Down+Inject and interleaved RMS;
they must not be folded again. No original checkpoint tensors are retained.
"""

import logging
from contextlib import contextmanager
from functools import cache
from typing import Optional

import msgspec
import torch
from torch import nn

from sglang.kernels.ops.gemm.hc_mix import _deterministic_inference
from sglang.srt.layers.hyperconnection import GatedResidual, HyperConnectionBase

logger = logging.getLogger(__name__)


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
        self.sliced = False

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
        context = _HCSumContext(self.operator, state, buffers)
        context.sliced = True
        return context

    def combine(self, block_output):
        if not self.sliced:
            return self.operator.combine(self.state, self.buffers, block_output)
        from sglang.kernels.ops.elementwise.hc_apply_sum import hc_apply_sum

        # Projections already ran on the full rows; only Apply follows this slice.
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

    Updates are serialized with inference. Raw checkpoint updates are finalized
    once at a complete-load boundary, so RMS may arrive after either projection.
    Changing RMS requires both raw projections; native updates use packed tensors.
    """

    def __init__(self, config, role: Optional[str] = None):
        HyperConnectionBase.__init__(self, config, True, True, role)
        c, h, lowrank = self.hc_count, self.hidden_size, config.hc_lowrank
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
        self._hc_needs_full_reload = False
        self._hc_update_open = False
        self._hc_refresh_up = False
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
        buffers = operator.allocate_buffers(residual.shape[0])
        mixed, _ = operator.mix(state, buffers)
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
            parameter.hc_weight_loader = self._load_sum_source_weight
        self.input_mix_weight_down.weight.hc_merged_down_inject = True
        self.input_mix_weight_down.weight.weight_loader_aliases = (
            "input_mix_weight_down.weight",
            "block_inject_weight.weight",
        )

    @torch.no_grad()
    def _load_sum_source_weight(self, parameter, loaded_weight, *, copy_value=True):
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
        try:
            if copy_value:
                destination.copy_(value)
        except Exception:
            update_open = self._hc_update_open
            self.abort_sum_state_update()
            self._hc_update_open = update_open
            raise
        self._hc_loaded_weights.add(key)

    def begin_sum_state_update(self, *, reuse=False):
        if self._hc_update_open:
            if reuse:
                return False
            raise ValueError("A packed HC weight update is already open")
        if self._hc_loaded_weights:
            logger.warning(
                "Discarding an incomplete HC update; reload complete HC weights"
            )
            self.abort_sum_state_update()
        self._hc_update_open = True
        return True

    def abort_sum_state_update(self):
        # In-place writes are not rolled back. A subsequent session must supply
        # a complete raw or native group before this module can run again.
        self._hc_update_open = False
        self._hc_weights_ready = False
        self._hc_needs_full_reload = True
        self._sum_weights_dirty = True
        self._hc_loaded_weights.clear()
        self._hc_load_format = None
        self._hc_refresh_up = False

    def offload_resident_parameters(self):
        # Both shells retain views of these tensors (and a padded Up layout).
        # Parameter-only functional_call offload cannot replace those views.
        return self.parameters()

    def finish_sum_state_update(self, *, force=False):
        self.prepare_sum_state_weights(force=force, finalize=True)
        self._hc_update_open = False

    def validate_sum_state_weights(self):
        loaded = self._hc_loaded_weights
        complete_required = self._hc_needs_full_reload or not self._hc_weights_ready
        if self._hc_load_format == "checkpoint":
            pair = loaded & {"down", "inject"}
            if len(pair) == 1 or ("norm" in loaded and len(pair) != 2):
                raise ValueError(
                    "Packed HC requires Down and Inject together; "
                    "an RMS update also requires both original projections"
                )
            if complete_required and not {"down", "inject", "norm", "up"} <= loaded:
                raise ValueError(
                    "Initial packed HC loading requires Down, Inject, RMS and Up"
                )
        elif self._hc_load_format == "native":
            if "norm" in loaded and "packed" not in loaded:
                raise ValueError(
                    "A native HC RMS update also requires folded Down+Inject"
                )
            if complete_required and not {"packed", "norm", "up"} <= loaded:
                raise ValueError(
                    "Initial native HC loading requires packed Down, RMS and Up"
                )
        elif complete_required:
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
    def prepare_sum_state_weights(self, *, force=False, finalize=False):
        if force:
            self._hc_refresh_up = True
            self._sum_weights_dirty = True
        if self._hc_update_open and not finalize:
            return False
        if self._hc_refresh_up and not self._hc_loaded_weights:
            if self._hc_needs_full_reload:
                raise ValueError(
                    "A failed HC update requires a complete raw or native reload; "
                    "force preparation cannot validate unobserved writes"
                )
            # Sharded/native loaders and P2P/RDMA write the registered native
            # tensors directly. Force rebinds caches; it never folds them again.
            self._hc_weights_ready = True
            self._sum_weights_dirty = True
        if not self._sum_weights_dirty:
            return True
        self.validate_sum_state_weights()
        if self._hc_load_format == "checkpoint" and "down" in self._hc_loaded_weights:
            self._fold_checkpoint_weights()
        self._hc_weights_ready = True
        self._bind_native_weights(
            refresh_up=self._hc_refresh_up or "up" in self._hc_loaded_weights
        )
        self._hc_loaded_weights.clear()
        self._hc_load_format = None
        self._sum_weights_dirty = False
        self._hc_refresh_up = False
        self._hc_needs_full_reload = False
        if finalize:
            self._hc_update_open = False
        return True

    @torch.no_grad()
    def _bind_native_weights(self, *, refresh_up):
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
        if self._sum_operators is None:
            small = SmallTSumHCShell(
                self,
                down_weights=weights,
                norm_weight_permuted=norm,
                deterministic=self._deterministic_hc,
            )
        else:
            from sglang.kernels.ops.gemm.hc_mix_splitk import permute_pad_up_weight

            small = self._sum_operators[0]
            previous = small.weights
            if refresh_up:
                previous.up.copy_(permute_pad_up_weight(up, self.hc_count))
            # Scratch is client-local and is never exported over IPC. Refresh
            # its initial state at serialized load boundaries, preserving Graph pointers.
            previous.scratch.zero_()
            small.weights = msgspec.structs.replace(
                previous, down_inject=weights.matrix, norm_permuted=norm
            )
        self._sum_operators = (small, large)
        self._install_weight_loaders()

    @torch.no_grad()
    def restore_sum_state_from_ipc(self):
        if self._hc_needs_full_reload:
            raise ValueError("A failed HC update requires a complete reload")
        self._hc_weights_ready = True
        self._bind_native_weights(refresh_up=True)
        self._sum_weights_dirty = False

    def _apply(self, fn, recurse=True):
        self._sum_operators = None
        result = nn.Module._apply(self, fn, recurse=recurse)
        self._install_weight_loaders()
        if self._hc_weights_ready and not self._hc_loaded_weights:
            self._bind_native_weights(refresh_up=True)
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
        if not self._hc_update_open and "hc_norm.weight" in selected and down is None:
            raise ValueError(
                "An RMS update requires Down and Inject (or their native packed tensor)"
            )
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
        if assign and selected:
            # Replacing storage requires new shells and Graph recapture.
            self._sum_operators = None
            self._sum_weights_dirty = True
        if raw:
            if self._hc_packed_down.is_meta:
                raise ValueError(
                    "Materialize HC before raw checkpoint loading; "
                    "meta assign supports native packed state"
                )
            if (
                not self._hc_update_open
                and not {
                    "input_mix_weight_down.weight",
                    "block_inject_weight.weight",
                }
                <= selected.keys()
            ):
                raise ValueError("Packed HC requires Down and Inject together")
            for name, value in selected.items():
                self._load_sum_source_weight(
                    names[name], value, copy_value=name != "input_mix_weight_up.weight"
                )
            # A state_dict call can be one bucket of an explicit update session.
            # Preserve raw values until end: a later bucket may supply new RMS.
            if not self._hc_update_open:
                self.validate_sum_state_weights()
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
        # assign=True may replace Parameter objects after the pre-hook. Rebind
        # views and loader callbacks as well as refreshing Up's derived layout.
        self.prepare_sum_state_weights(finalize=not self._hc_update_open)
        self._install_weight_loaders()


class PackedHCModelMixin:
    """Model-owned loading hooks for native HC parameters and their local views.

    The generic loaders/updater dispatch model hooks, without inspecting HC
    layers. Preprocessed native imports only rebind views; they must not rerun
    checkpoint folding or other model/quantization postprocessing.
    """

    def _packed_hc_modules(self):
        return (
            (name, module)
            for name, module in self.named_modules()
            if isinstance(module, PackedGatedResidual)
        )

    def post_load_weights(self, *, force=True):
        for _, module in self._packed_hc_modules():
            module.prepare_sum_state_weights(force=force)

    def post_load_weights_from_direct(self, *, force=False):
        # Direct/custom loaders only changed parameter storage. Do not rerun
        # unrelated checkpoint transformations on already-processed weights.
        PackedHCModelMixin.post_load_weights(self, force=force)

    def post_load_weights_from_preprocessed(self):
        for _, module in self._packed_hc_modules():
            module.restore_sum_state_from_ipc()

    def begin_weight_update(self):
        for _, module in self._packed_hc_modules():
            module.begin_sum_state_update()

    def validate_weight_update(self):
        for name, module in self._packed_hc_modules():
            try:
                module.validate_sum_state_weights()
            except ValueError as exc:
                raise ValueError(f"{name or '<root>'}: {exc}") from exc

    def end_weight_update(self, *, force=False):
        self.validate_weight_update()
        for _, module in self._packed_hc_modules():
            module.finish_sum_state_update(force=force)

    @contextmanager
    def weight_update_context(self):
        # Checkpoint-engine can deliver multiple buckets inside or outside an
        # already-open update session. Only finish/abort sessions opened here.
        owned = []
        try:
            for name, module in self._packed_hc_modules():
                if module.begin_sum_state_update(reuse=True):
                    owned.append((name, module))
            yield
            for name, module in owned:
                try:
                    module.validate_sum_state_weights()
                except ValueError as exc:
                    raise ValueError(f"{name or '<root>'}: {exc}") from exc
            for _, module in owned:
                module.finish_sum_state_update()
        except Exception:
            for _, module in owned:
                module.abort_sum_state_update()
            raise


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
        and lowrank % 8 == 0
        and lowrank + c <= ((lowrank + 127) // 128) * 128
        and config.params_dtype in (torch.bfloat16, torch.float16)
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] == 10
    ):
        return PackedGatedResidual(config, role)
    return GatedResidual(config, use_mix, use_combine, role)
