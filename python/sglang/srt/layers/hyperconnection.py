from functools import cache, lru_cache
from typing import Optional

import msgspec
import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.kernels.ops.gemm.hc_mix import (
    _deterministic_inference,
    fused_hc_mix,
    fused_hc_mix_supported,
)


@lru_cache(None)
def _hc_gate_stream(device: int) -> torch.cuda.Stream:
    return torch.cuda.Stream(device=device)


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


class _HCCheckpointWeight(nn.Module):
    """CPU checkpoint storage; only the prepared HC layout is uploaded to CUDA."""

    def __init__(self, shape, dtype):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(shape, device="cpu", dtype=dtype), requires_grad=False
        )

    def _apply(self, fn, recurse=True):
        # Honor pre-load dtype conversion without moving checkpoint payloads
        # to CUDA. The parent rejects migration once prepared weights exist.
        dtype = fn(self.weight.new_empty(0)).dtype
        self.weight.data = self.weight.data.to(dtype=dtype)
        return self

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        key = prefix + "weight"
        if key in state_dict:
            try:
                self.weight.weight_loader(self.weight, state_dict[key])
            except (ValueError, RuntimeError) as exc:
                error_msgs.append(f"{key}: {exc}")
        elif strict:
            missing_keys.append(key)
        if strict:
            unexpected_keys.extend(
                name for name in state_dict if name.startswith(prefix) and name != key
            )


class HyperConnectionConfig(msgspec.Struct, frozen=True):
    hc_count: int = 4
    hidden_size: int = 64
    params_dtype: torch.dtype = torch.bfloat16
    mtp_hc: bool = False
    hc_lowrank: int = 16
    rms_norm_eps: float = 1e-6
    hc_per_branch_norm: bool = False


class GroupedGemmaRMSNorm(nn.Module):
    def __init__(
        self, hidden_size: int, eps: float = 1e-6, group_size: Optional[int] = None
    ):
        super().__init__()
        if group_size is not None and hidden_size % group_size != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by group_size ({group_size})"
            )
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps
        self.group_size = group_size
        self.weight.weight_loader = self._weight_loader
        # The JIT kernel requires group_size to be a multiple of 512; this is
        # init-static, so resolve it once here (device/dtype stay per-call).
        effective_group_size = group_size if group_size is not None else hidden_size
        self._jit_group_size = (
            effective_group_size if effective_group_size % 512 == 0 else None
        )

    def _weight_loader(self, param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        assert param.size() == loaded_weight.size()
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            self._jit_group_size is not None
            and x.is_cuda
            and x.dtype in (torch.bfloat16, torch.float16)
        ):
            from sglang.kernels.ops.layernorm.grouped_gemma_rmsnorm import (
                grouped_gemma_rmsnorm,
            )

            return grouped_gemma_rmsnorm(
                x, self.weight, self._jit_group_size, self.variance_epsilon
            )
        input_dtype = x.dtype
        x_float = x.float()
        if self.group_size is None:
            variance = x_float.pow(2).mean(dim=-1, keepdim=True)
            x_norm = x_float * torch.rsqrt(variance + self.variance_epsilon)
        else:
            x_grouped = x_float.reshape(
                *x_float.shape[:-1],
                x_float.shape[-1] // self.group_size,
                self.group_size,
            )
            variance = x_grouped.pow(2).mean(dim=-1, keepdim=True)
            x_norm = (
                x_grouped * torch.rsqrt(variance + self.variance_epsilon)
            ).flatten(-2)
        return (x_norm * (1.0 + self.weight.float())).to(input_dtype)


class HyperConnectionBase(nn.Module):
    def __init__(
        self,
        config: HyperConnectionConfig,
        use_mix: bool = True,
        use_combine: bool = True,
        role: Optional[str] = None,
    ):
        super().__init__()

        self.config = config
        self.hc_count = config.hc_count
        if config.mtp_hc and role is not None and "mtp" in role:
            self.hc_count = self.hc_count + 1
        self.hidden_size = config.hidden_size
        self.params_dtype = config.params_dtype

    def mix(self, hyper_input: torch.Tensor):
        assert hyper_input.shape[-1] == self.hc_count * self.hidden_size
        mixed_input = hyper_input.view(
            *hyper_input.shape[:-1], self.hc_count, self.hidden_size
        ).mean(dim=-2)
        return mixed_input, hyper_input

    def combine(
        self, block_output: torch.Tensor, residual: torch.Tensor
    ) -> torch.Tensor:
        assert residual.shape[-1] == self.hc_count * self.hidden_size
        assert block_output.shape[-1] == self.hidden_size
        residual_reshaped = residual.view(
            *residual.shape[:-1], self.hc_count, self.hidden_size
        )
        combined_output = residual_reshaped + block_output.unsqueeze(-2)
        combined_output = combined_output.view(
            *residual.shape[:-1], self.hc_count * self.hidden_size
        )
        return combined_output


class GatedResidual(HyperConnectionBase):
    def __init__(
        self,
        config: HyperConnectionConfig,
        use_mix: bool = True,
        use_combine: bool = True,
        role: Optional[str] = None,
        *,
        packed_weights: bool = False,
    ):
        super().__init__(config, use_mix, use_combine, role)

        # Packed full HC keeps Down/Inject/RMS checkpoint tensors on CPU.
        # Other instances retain their original parameter storage and Tensor ABI.
        lowrank = config.hc_lowrank
        self._packed_sum_weights = (
            packed_weights
            and use_mix
            and use_combine
            and config.hc_per_branch_norm
            and self.hc_count == 4
            and 0 < self.hidden_size <= 16384
            and self.hidden_size % 512 == 0
            and lowrank > 0
            and lowrank % 8 == 0
            and lowrank + 4 <= ((lowrank + 127) // 128) * 128
            and config.params_dtype in (torch.bfloat16, torch.float16)
            and torch.cuda.is_available()
            and torch.cuda.get_device_capability()[0] == 10
        )
        # Resolve once at construction, before preparing weights or Graph capture.
        # This fixes atomic ordering; small/large-T numerical paths stay distinct.
        self._deterministic_hc = _deterministic_inference()
        self._hc_loaded_weights = set()
        self.register_buffer("_hc_packed_down", None, persistent=False)
        self.register_buffer("_hc_packed_norm", None, persistent=False)

        norm_dim = (
            self.config.hidden_size * self.hc_count
            if self.config.hc_per_branch_norm
            else self.config.hidden_size
        )
        norm_group_size = (
            self.config.hidden_size if self.config.hc_per_branch_norm else None
        )
        if self._packed_sum_weights:
            self.hc_norm = _HCCheckpointWeight((norm_dim,), config.params_dtype)
            self.hc_norm.weight.data.zero_()
        else:
            self.hc_norm = GroupedGemmaRMSNorm(
                norm_dim, eps=self.config.rms_norm_eps, group_size=norm_group_size
            )

        if use_mix:
            self.input_mix_weight_down = (
                _HCCheckpointWeight(
                    (self.config.hc_lowrank, self.hidden_size * self.hc_count),
                    config.params_dtype,
                )
                if self._packed_sum_weights
                else nn.Linear(
                    self.hidden_size * self.hc_count,
                    self.config.hc_lowrank,
                    bias=False,
                    dtype=config.params_dtype,
                )
            )
            self.input_mix_weight_up = nn.Linear(
                self.config.hc_lowrank,
                self.hc_count * self.hidden_size,
                bias=False,
                dtype=config.params_dtype,
            )
            lowrank = self.config.hc_lowrank
            self._jit_mix_ok = (
                torch.cuda.is_available()
                # The CuTe split-K pair is tcgen05 (sm_100 family) only.
                and torch.cuda.get_device_capability()[0] == 10
                and (self.hc_count * self.hidden_size) % 2048 == 0
                and self.hidden_size % 8 == 0
                and lowrank > 0
                and lowrank % 8 == 0
            )
            self._mix_up_weight_padded = None

        self._gate_stream = None
        if use_combine:
            self.block_inject_weight = (
                _HCCheckpointWeight(
                    (self.hc_count, self.hidden_size * self.hc_count),
                    config.params_dtype,
                )
                if self._packed_sum_weights
                else nn.Linear(
                    self.hidden_size * self.hc_count,
                    self.hc_count,
                    bias=False,
                    dtype=config.params_dtype,
                )
            )
            # hc_combine rejects other shapes; device and dtype are checked per call.
            self._jit_combine_ok = (
                self.hidden_size % 8 == 0
                and (self.hc_count * self.hidden_size) % 2048 == 0
            )
            vecs = self.hc_count * self.hidden_size // 8
            self._split_combine_ok = (
                self._jit_combine_ok
                and vecs % (8 * 160) == 0
                and (self.hidden_size // 8) % (vecs // 8) == 0
            )
            if (
                not self._packed_sum_weights
                and self.hc_count == 4
                and self.hidden_size == 2560
                and config.hc_per_branch_norm
                and config.params_dtype == torch.bfloat16
                and torch.cuda.is_available()
                and torch.cuda.get_device_capability()[0] == 10
            ):
                # Create before graph capture; all layers on a device share it.
                self._gate_stream = _hc_gate_stream(torch.cuda.current_device())

        def _mix_compute(
            hyper_input_normed: torch.Tensor,
            input_mix_weight_down: torch.Tensor,
            input_mix_weight_up: torch.Tensor,
            hc: int,
            hs: int,
        ) -> torch.Tensor:
            input_mix_weight = F.silu(
                F.linear(hyper_input_normed, input_mix_weight_down) / hc
            )
            input_mix_weight = F.linear(input_mix_weight, input_mix_weight_up)
            input_mix_weight = torch.sigmoid(input_mix_weight)
            input_mix_weight = input_mix_weight.unflatten(-1, (hc, hs))
            output = (
                input_mix_weight * hyper_input_normed.unflatten(-1, (hc, hs))
            ).mean(dim=-2)
            return output

        def _combine_compute(
            block_output: torch.Tensor,
            residual: torch.Tensor,
            normed_residual: torch.Tensor,
            block_inject_weight: torch.Tensor,
            hc: int,
            hs: int,
        ) -> torch.Tensor:
            R = residual.unflatten(-1, (hc, hs))
            block_inject_weight_out = 2 * torch.sigmoid(
                F.linear(normed_residual, block_inject_weight) / hc
            )
            injection = block_output.unsqueeze(-2) * block_inject_weight_out.unsqueeze(
                -1
            )
            return (R + injection).flatten(-2)

        self._mix_compute = torch.compile(_mix_compute)
        self._combine_compute = torch.compile(_combine_compute)
        self._sum_operators = None
        self._sum_weights_dirty = True
        if use_mix and use_combine:
            for parameter in (
                self.hc_norm.weight,
                self.input_mix_weight_down.weight,
                self.input_mix_weight_up.weight,
                self.block_inject_weight.weight,
            ):
                parameter.weight_loader = self._load_sum_source_weight
                # Direct loading must not run arbitrary sharded weight_loaders.
                # HC explicitly opts into this replicated-parameter callback.
                parameter.hc_weight_loader = self._load_sum_source_weight
        self.register_load_state_dict_pre_hook(self._before_sum_state_dict_load)
        self.register_load_state_dict_post_hook(self._after_sum_state_dict_load)

    def _load_sum_source_weight(self, parameter, loaded_weight):
        # HC parameters are replicated; refresh derived layouts at load completion,
        # not during inference.
        if parameter.shape != loaded_weight.shape:
            raise ValueError("HC source weight shape mismatch")
        if self._packed_sum_weights:
            names = (
                ("down", self.input_mix_weight_down.weight),
                ("inject", self.block_inject_weight.weight),
                ("norm", self.hc_norm.weight),
                ("up", self.input_mix_weight_up.weight),
            )
            name = next(name for name, target in names if parameter is target)
            # Down/Inject/RMS stay on CPU for exact checkpoint round-trips.
            # Only Up keeps its original CUDA parameter layout.
            parameter.data.copy_(loaded_weight)
            self._hc_loaded_weights.add(name)
            self._sum_weights_dirty = True
            return
        parameter.data.copy_(loaded_weight)
        self._sum_weights_dirty = True

    def _before_sum_state_dict_load(
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
        if not self._packed_sum_weights:
            return
        names = {
            "down": "input_mix_weight_down.weight",
            "inject": "block_inject_weight.weight",
            "norm": "hc_norm.weight",
            "up": "input_mix_weight_up.weight",
        }
        loaded = {name for name, key in names.items() if prefix + key in state_dict}
        if len(loaded & {"down", "inject"}) == 1:
            raise ValueError(
                "Packed HC requires Down and Inject to be updated together"
            )
        # Ordinary Up child loads bypass the per-parameter loader hook.
        self._hc_loaded_weights.update(loaded)
        if loaded:
            self._sum_weights_dirty = True

    def _after_sum_state_dict_load(self, module, incompatible_keys):
        self.prepare_sum_state_weights()
        self.validate_sum_state_weights()

    def _apply(self, fn, recurse=True):
        if self._packed_sum_weights and self._hc_packed_down is not None:
            raise RuntimeError(
                "Place/cast packed HC before loading weights; recreate and reload "
                "the checkpoint to change its device/dtype after preparation"
            )
        # A device/dtype conversion invalidates derived storage and requires
        # graph recapture, just like conversion of the original parameters.
        result = super()._apply(fn, recurse=recurse)
        self._sum_operators = None
        self._sum_weights_dirty = True
        return result

    @torch.no_grad()
    def _prepare_packed_sum_state_weights(self):
        if len(self._hc_loaded_weights & {"down", "inject"}) == 1:
            # A weight-update session may deliver the pair in separate buckets.
            # Keep the previous operators unusable until the other half arrives.
            self._sum_weights_dirty = True
            return False
        if self._sum_operators is not None and not self._sum_weights_dirty:
            return True
        up, norm = self.input_mix_weight_up.weight, self.hc_norm.weight
        if (
            not up.is_cuda
            or norm.device.type != "cpu"
            or norm.dtype != up.dtype
            or up.dtype not in (torch.bfloat16, torch.float16)
        ):
            raise ValueError("Packed HC requires CUDA Up and CPU RMS in BF16/FP16")
        c, h, lowrank = self.hc_count, self.hidden_size, self.config.hc_lowrank
        npad = ((lowrank + c + 7) // 8) * 8
        # Fold in FP32 on CPU, cast once, then upload the shared packed layout.
        gamma = 1.0 + norm.detach().to(device="cpu", dtype=torch.float32)
        folded = torch.zeros((npad, c * h), device="cpu", dtype=up.dtype)
        folded[:lowrank] = (self.input_mix_weight_down.weight.float() * gamma).to(
            up.dtype
        )
        folded[lowrank : lowrank + c] = (
            self.block_inject_weight.weight.float() * gamma
        ).to(up.dtype)
        host_packed = folded.view(npad, c, h).permute(1, 0, 2).contiguous()
        packed = self._hc_packed_down
        if packed is None:
            packed = host_packed.to(up.device)
        else:
            if (
                packed.shape != host_packed.shape
                or packed.dtype != up.dtype
                or packed.device != up.device
            ):
                raise ValueError("Packed HC storage changed; recreate and reload")
            packed.copy_(host_packed)
        # Upload RMS in the shared [H,C] layout without an intermediate CUDA copy.
        host_norm = norm.detach().view(c, h).T.contiguous().flatten()
        packed_norm = self._hc_packed_norm
        if packed_norm is None:
            packed_norm = host_norm.to(up.device)
        else:
            if (
                packed_norm.shape != host_norm.shape
                or packed_norm.dtype != up.dtype
                or packed_norm.device != up.device
            ):
                raise ValueError("Packed RMS storage changed; recreate and reload")
            packed_norm.copy_(host_norm)
        return self._bind_packed_sum_state_weights(packed, packed_norm)

    @torch.no_grad()
    def restore_sum_state_from_ipc(self):
        """Rebuild client-local operators without writing shared IPC weights."""
        if self._packed_sum_weights:
            self._bind_packed_sum_state_weights(
                self._hc_packed_down, self._hc_packed_norm
            )

    def _bind_packed_sum_state_weights(self, packed, packed_norm):
        from sglang.kernels.ops.elementwise.hc_down_batched_sum import (
            BatchedDownWeights,
        )
        from sglang.kernels.ops.elementwise.hc_sum_state import SmallTSumHCShell
        from sglang.kernels.ops.elementwise.hc_sum_state_large import LargeTSumHCShell

        c, h, lowrank = self.hc_count, self.hidden_size, self.config.hc_lowrank
        npad = ((lowrank + c + 7) // 8) * 8
        if (
            packed is None
            or packed_norm is None
            or packed.shape != (c, npad, h)
            or not packed.is_contiguous()
        ):
            raise ValueError("Packed HC requires loaded Down/Inject and RMS buffers")
        weights = BatchedDownWeights(packed.transpose(1, 2), h, lowrank, c)
        large = LargeTSumHCShell(
            self,
            down_weights=weights,
            norm_weight_permuted=packed_norm,
            deterministic=self._deterministic_hc,
        )
        if self._sum_operators is None:
            small = SmallTSumHCShell(
                self,
                down_weights=weights,
                norm_weight_permuted=packed_norm,
                deterministic=self._deterministic_hc,
            )
        else:
            from sglang.kernels.ops.elementwise.hc_mix import permute_pad_up_weight

            small = self._sum_operators[0]
            up = permute_pad_up_weight(self.input_mix_weight_up.weight, c)
            previous = small.weights
            if (
                previous.up.shape != up.shape
                or previous.up.dtype != up.dtype
                or previous.up.device != up.device
            ):
                raise ValueError("Prepared HC Up storage changed; recreate and reload")
            # Captured graphs retain Up and scratch addresses across in-place reloads.
            previous.up.copy_(up)
            small.weights = msgspec.structs.replace(
                previous, down_inject=weights.matrix, norm_permuted=packed_norm
            )
        self._hc_packed_down = packed
        self._hc_packed_norm = packed_norm
        self._sum_operators = (small, large)
        # IPC mapping replaces Parameter objects, so restore all loader hooks.
        for parameter in (
            self.input_mix_weight_down.weight,
            self.block_inject_weight.weight,
            self.input_mix_weight_up.weight,
            self.hc_norm.weight,
        ):
            parameter.weight_loader = self._load_sum_source_weight
            parameter.hc_weight_loader = self._load_sum_source_weight
        self._hc_loaded_weights.clear()
        self._sum_weights_dirty = False
        return True

    def validate_sum_state_weights(self):
        """Reject unfinished packed HC updates at a complete-load boundary."""
        if not self._packed_sum_weights:
            return
        loaded_pair = self._hc_loaded_weights & {"down", "inject"}
        if len(loaded_pair) == 1:
            missing = "Inject" if "down" in loaded_pair else "Down"
            raise ValueError(
                "Packed HC requires Down and Inject to be updated together; "
                f"missing {missing} at the end of the weight update"
            )
        if self._sum_operators is None or self._sum_weights_dirty:
            raise ValueError("Packed HC weights are not prepared after loading")

    @torch.no_grad()
    def prepare_sum_state_weights(self, *, force: bool = False):
        """Prepare packed HC after loading; return False for non-packed instances.

        Incomplete Down/Inject pairs stay dirty; validate at load completion.
        Unchanged Up/RMS need not be resent; ``force`` handles bypassed loader hooks.
        In-place reload preserves captured buffers; serialize it with inference.
        Replacing parameter storage still requires graph recapture.
        """
        if self._packed_sum_weights:
            if force:
                self._sum_weights_dirty = True
            return self._prepare_packed_sum_state_weights()
        self._sum_operators = None
        return False

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

    def bootstrap_sum_state(self, residual):
        if not self.can_use_sum_state(residual):
            raise ValueError("HC sum-state weights/input are not prepared or supported")
        return self._sum_operators[int(residual.shape[0] > 24)].bootstrap(residual)

    def mix(
        self,
        hyper_input: torch.Tensor,
        normalized_input: Optional[torch.Tensor] = None,
        *,
        use_sum_state: bool = False,
    ):
        state_type = _sum_state_type()
        is_state = isinstance(hyper_input, state_type)
        residual = hyper_input.residual if is_state else hyper_input
        use_prepared = (
            self._packed_sum_weights or use_sum_state or is_state
        ) and self.can_use_sum_state(residual)
        if self._packed_sum_weights and not use_prepared:
            raise ValueError(
                "Packed HC requires prepared weights and supported inference input"
            )
        if use_prepared:
            operator = self._sum_operators[int(residual.shape[0] > 24)]
            state = hyper_input if is_state else operator.bootstrap(residual)
            buffers = operator.allocate_buffers(residual.shape[0])
            mixed, _ = operator.mix(state, buffers)
            return mixed, _HCSumContext(operator, state, buffers)
        if is_state:
            hyper_input = residual
        assert hyper_input.shape[-1] == self.hc_count * self.hidden_size
        if hyper_input.shape[0] == 0:
            mixed_input = hyper_input.new_empty(
                (*hyper_input.shape[:-1], self.hidden_size), dtype=self.params_dtype
            )
            return mixed_input, (hyper_input, hyper_input, None)

        if normalized_input is not None:
            assert self.config.hc_per_branch_norm
            assert normalized_input.shape == hyper_input.shape
            hyper_input_normed = normalized_input
        elif self.config.hc_per_branch_norm:
            hyper_input_normed = self.hc_norm(hyper_input)
        else:
            hyper_input_normed = self.hc_norm(
                hyper_input.unflatten(-1, (self.hc_count, self.hidden_size))
            ).flatten(-2)
        if (
            self._jit_mix_ok
            and hyper_input_normed.is_cuda
            and hyper_input_normed.dtype in (torch.bfloat16, torch.float16)
            and hyper_input_normed.shape[0] <= 24
        ):
            from sglang.kernels.ops.gemm.hc_mix_splitk import (
                hc_mix,
                permute_pad_up_weight,
            )

            if self._mix_up_weight_padded is None:
                self._mix_up_weight_padded = permute_pad_up_weight(
                    self.input_mix_weight_up.weight, self.hc_count
                )
            mixed_input = hc_mix(
                hyper_input_normed,
                self.input_mix_weight_down.weight.data,
                self._mix_up_weight_padded,
                self.hc_count,
                self.hidden_size,
            ).to(self.params_dtype)
        elif fused_hc_mix_supported(
            hyper_input_normed,
            self.input_mix_weight_down.weight,
            self.input_mix_weight_up.weight,
        ):
            mixed_input = fused_hc_mix(
                hyper_input_normed,
                self.input_mix_weight_down.weight,
                self.input_mix_weight_up.weight,
                self.hc_count,
                self.hidden_size,
            ).to(self.params_dtype)
        else:
            mixed_input = self._mix_compute(
                hyper_input_normed,
                self.input_mix_weight_down.weight,
                self.input_mix_weight_up.weight,
                self.hc_count,
                self.hidden_size,
            ).to(self.params_dtype)
        partials = None
        stream = self._gate_stream
        if (
            stream is not None
            and hyper_input.shape[0] == 1
            and hyper_input.dtype == torch.bfloat16
            and not torch.compiler.is_compiling()
        ):
            from sglang.kernels.ops.elementwise.hc_combine_decode import hc_combine_gate

            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                partials = hc_combine_gate(
                    hyper_input_normed, self.block_inject_weight.weight
                )
            hyper_input_normed.record_stream(stream)
        return mixed_input, (hyper_input, hyper_input_normed, partials)

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
            return self.mix(residual, use_sum_state=True)
        return self.mix(residual, normalized_input=prepared_input)

    def combine_and_prepare(self, block_output, residuals, next_norm=None):
        if isinstance(residuals, _HCSumContext):
            state = residuals.combine(block_output)
            return state.residual, state
        if not isinstance(next_norm, GroupedGemmaRMSNorm):
            next_norm = None
        return self.combine_and_normalize(block_output, residuals, next_norm)

    def combine_and_normalize(
        self,
        block_output: torch.Tensor,
        residuals: tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
        next_norm: Optional[GroupedGemmaRMSNorm],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Return the residual and its next HC normalization explicitly.

        The BF16 combine result is retained before calculating the next norm.
        Callers must only pass the norm when no PLE or other update intervenes.
        """
        hyper_input, _, partials = residuals
        if (
            partials is not None
            and next_norm is not None
            and next_norm.group_size == self.hidden_size
            and next_norm.weight.dtype == torch.bfloat16
            and block_output.dtype == torch.bfloat16
        ):
            from sglang.kernels.ops.elementwise.hc_combine_decode import (
                hc_combine_apply_norm,
            )

            current = torch.cuda.current_stream()
            current.wait_stream(self._gate_stream)
            partials.record_stream(current)
            return hc_combine_apply_norm(
                block_output,
                hyper_input,
                partials,
                next_norm.weight,
                next_norm.variance_epsilon,
            )
        return self.combine(block_output, residuals), None

    def combine(
        self,
        block_output: torch.Tensor,
        residuals,
    ):
        if isinstance(residuals, _HCSumContext):
            return residuals.combine(block_output)
        hyper_input, hyper_input_normed, partials = residuals
        if partials is not None:
            from sglang.kernels.ops.elementwise.hc_combine_decode import (
                hc_combine_apply,
            )

            current = torch.cuda.current_stream()
            current.wait_stream(self._gate_stream)
            partials.record_stream(current)
            return hc_combine_apply(block_output, hyper_input, partials)
        assert hyper_input.shape[-1] == self.hc_count * self.hidden_size
        assert block_output.shape[-1] == self.hidden_size
        if block_output.shape[0] == 0:
            return hyper_input.to(self.params_dtype)

        if (
            self._jit_combine_ok
            and block_output.is_cuda
            and block_output.dtype in (torch.bfloat16, torch.float16)
            and hyper_input.dtype == block_output.dtype
            and hyper_input_normed.dtype == block_output.dtype
            and self.block_inject_weight.weight.dtype == block_output.dtype
        ):
            if self._split_combine_ok and block_output.shape[0] <= 32:
                from sglang.kernels.ops.elementwise.hc_combine import (
                    hc_combine_split,
                )

                return hc_combine_split(
                    block_output,
                    hyper_input,
                    hyper_input_normed,
                    self.block_inject_weight.weight.data,
                    self.hc_count,
                    self.hidden_size,
                )
            from sglang.kernels.ops.elementwise.hc_combine import hc_combine

            return hc_combine(
                block_output,
                hyper_input,
                hyper_input_normed,
                self.block_inject_weight.weight,
                self.hc_count,
                self.hidden_size,
            )

        updated_residuals = self._combine_compute(
            block_output,
            hyper_input,
            hyper_input_normed,
            self.block_inject_weight.weight,
            self.hc_count,
            self.hidden_size,
        ).to(self.params_dtype)
        return updated_residuals


HYPERCONNECTION_CLASS_DICT = {
    "hyperconnection_average": HyperConnectionBase,
    "gated_residual_simple": GatedResidual,
}
