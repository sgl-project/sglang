from typing import Optional

import msgspec
import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.srt.layers.hc_mix_triton import fused_hc_mix, fused_hc_mix_supported


class _HCSumContext:
    """Per-call state across the attention/MoE core; never stored on the layer."""

    def __init__(self, operator, state, buffers):
        self.operator = operator
        self.state = state
        self.buffers = buffers


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

        # Only explicitly selected, structurally supported full HC instances
        # omit the original GPU parameters. Final mixers and other paths keep
        # their existing storage, loader and Tensor ABI.
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
                    device=torch.cuda.current_device(),
                    dtype=config.params_dtype,
                )
            )
            self.input_mix_weight_up = nn.Linear(
                self.config.hc_lowrank,
                self.hc_count * self.hidden_size,
                bias=False,
                device=torch.cuda.current_device(),
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
                    device=torch.cuda.current_device(),
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
        self._hc_source_packed = None
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
        # These are replicated, non-sharded HC parameters. Keep checkpoint
        # names/storage authoritative; derived layouts are refreshed at the end
        # of the model's load_weights transaction, never on the inference path.
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
        self._hc_source_packed = None
        self._sum_weights_dirty = True
        return result

    @torch.no_grad()
    def _prepare_packed_sum_state_weights(self):
        if len(self._hc_loaded_weights & {"down", "inject"}) == 1:
            raise ValueError(
                "Packed HC requires Down and Inject to be updated together"
            )
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
        # Fold on CPU in FP32, cast once and reorder before the single upload.
        # No full original, FP32-folded or second prepared matrix lives on GPU.
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
        # Upload RMS directly in the shared [H,C] layout. Do not create an
        # original [C,H] CUDA copy merely to transpose it during preparation.
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
        operators = (
            SmallTSumHCShell(
                self, down_weights=weights, norm_weight_permuted=packed_norm
            ),
            LargeTSumHCShell(
                self, down_weights=weights, norm_weight_permuted=packed_norm
            ),
        )
        self._hc_packed_down = packed
        self._hc_packed_norm = packed_norm
        self._sum_operators = operators
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

    @torch.no_grad()
    def prepare_sum_state_weights(self, *, force: bool = False):
        """Prepare both paths after a complete weight-load transaction.

        Packed instances retain CPU checkpoint values and one shared CUDA
        Down/Inject storage plus one shared [H,C] RMS storage. Down and Inject
        must be updated together; unchanged Up/RMS need not be resent.
        ``force`` handles loaders that copy through
        state_dict instead of weight_loader. Reloading requires graph recapture.
        Legacy instances keep their existing source/derived-storage behavior.
        Large-T Up shares its original parameter storage in both cases.
        Like the existing HC dispatcher, a layer executes on the worker's
        serialized compute stream; per-call outputs are not cached on it.
        """
        if self._packed_sum_weights:
            if force:
                self._sum_weights_dirty = True
            return self._prepare_packed_sum_state_weights()
        from sglang.kernels.ops.elementwise.hc_mix import pad_lowrank

        if not (
            self.config.hc_per_branch_norm
            and self.hc_count == 4
            and 0 < self.hidden_size <= 16384
            and self.hidden_size % 512 == 0
            and self.config.hc_lowrank > 0
            and self.config.hc_lowrank % 8 == 0
            and self.config.hc_lowrank + 4 <= pad_lowrank(self.config.hc_lowrank)
            and hasattr(self, "input_mix_weight_down")
            and hasattr(self, "block_inject_weight")
            and self.hc_norm.weight.is_cuda
            and self.hc_norm.weight.dtype in (torch.bfloat16, torch.float16)
            and torch.cuda.get_device_capability(self.hc_norm.weight.device)[0] == 10
        ):
            self._sum_operators = None
            return False
        from sglang.kernels.ops.elementwise.hc_sum_state import SmallTSumHCShell
        from sglang.kernels.ops.elementwise.hc_sum_state_large import LargeTSumHCShell

        # One physical checkpoint storage; the old parameter names remain
        # loader/state_dict/fallback-compatible views. Folded small/large GEMM
        # layouts below are derived from this storage and the RMS weight.
        down = self.input_mix_weight_down.weight
        inject = self.block_inject_weight.weight
        shape = (self.config.hc_lowrank + 4, 4 * self.hidden_size)
        if self._hc_source_packed is None:
            self._hc_source_packed = down.new_empty(shape)
        packed = self._hc_source_packed
        if (
            packed.shape != shape
            or packed.device != down.device
            or packed.dtype != down.dtype
        ):
            raise RuntimeError("HC checkpoint storage changed; recreate and recapture")
        packed[: self.config.hc_lowrank].copy_(down)
        packed[self.config.hc_lowrank :].copy_(inject)
        down.data = packed[: self.config.hc_lowrank]
        inject.data = packed[self.config.hc_lowrank :]

        # load_state_dict(assign=True) may replace Parameter objects.
        for parameter in (
            self.hc_norm.weight,
            self.input_mix_weight_down.weight,
            self.input_mix_weight_up.weight,
            self.block_inject_weight.weight,
        ):
            parameter.weight_loader = self._load_sum_source_weight
            parameter.hc_weight_loader = self._load_sum_source_weight
        small = SmallTSumHCShell(self)
        new = (
            small,
            LargeTSumHCShell(self, norm_weight_permuted=small.weights.norm_permuted),
        )
        old = self._sum_operators
        if old is None:
            self._sum_operators = new
        else:
            pairs = [
                (old[0].weights.down_inject, new[0].weights.down_inject),
                (old[0].weights.up, new[0].weights.up),
                (old[0].weights.norm_permuted, new[0].weights.norm_permuted),
                (old[1].weights.matrix, new[1].weights.matrix),
            ]
            if any(
                a.shape != b.shape or a.dtype != b.dtype or a.device != b.device
                for a, b in pairs
            ):
                raise RuntimeError(
                    "HC storage metadata changed; move/recreate the module and recapture graphs"
                )
            for destination, source in pairs:
                destination.copy_(source)
            # Up shares source storage; both paths share the refreshed RMS.
            old[1].up_weight = new[1].up_weight
            old[1].norm_weight = old[0].weights.norm_permuted
        self._sum_weights_dirty = False
        return True

    def can_use_sum_state(self, residual):
        if (
            torch.is_grad_enabled()
            or self._sum_operators is None
            or self._sum_weights_dirty
        ):
            return False
        from sglang.kernels.ops.elementwise.hc_down_batched_sum import _max_hc_rows

        w = self._sum_operators[1].weights
        return (
            residual.ndim == 2
            and residual.shape[1] == 4 * self.hidden_size
            and residual.shape[0] <= _max_hc_rows(w)
            and residual.device == w.matrix.device
            and residual.dtype == w.matrix.dtype
            and residual.is_contiguous()
            and residual.data_ptr() % 32 == 0
        )

    def bootstrap_sum_state(self, residual):
        if not self.can_use_sum_state(residual):
            raise ValueError("HC sum-state weights/input are not prepared or supported")
        return self._sum_operators[int(residual.shape[0] > 24)].bootstrap(residual)

    def mix(self, hyper_input: torch.Tensor, *, use_sum_state: bool = False):
        from sglang.kernels.ops.elementwise.hc_sum_state import HCSumState

        if self._packed_sum_weights:
            residual = (
                hyper_input.residual
                if isinstance(hyper_input, HCSumState)
                else hyper_input
            )
            if not self.can_use_sum_state(residual):
                raise ValueError(
                    "Packed HC requires prepared weights and supported inference input"
                )
            use_sum_state = True

        # Tensor callers keep the original two-tensor ABI unless the model
        # explicitly opts in. Passing a changed residual tensor bootstraps new
        # statistics; an unchanged state reuses the previous Apply's sums.
        if (
            use_sum_state
            and not isinstance(hyper_input, HCSumState)
            and self.can_use_sum_state(hyper_input)
        ):
            hyper_input = self.bootstrap_sum_state(hyper_input)

        if isinstance(hyper_input, HCSumState):
            if self.can_use_sum_state(hyper_input.residual):
                operator = self._sum_operators[int(hyper_input.residual.shape[0] > 24)]
                buffers = operator.allocate_buffers(hyper_input.residual.shape[0])
                mixed, _ = operator.mix(hyper_input, buffers)
                return mixed, _HCSumContext(operator, hyper_input, buffers)
            # A non-fused/final consumer observes the ordinary residual tensor;
            # it must not interpret pre-rounding sums as exact stored RMS.
            hyper_input = hyper_input.residual
        assert hyper_input.shape[-1] == self.hc_count * self.hidden_size
        if hyper_input.shape[0] == 0:
            mixed_input = hyper_input.new_empty(
                (*hyper_input.shape[:-1], self.hidden_size), dtype=self.params_dtype
            )
            return mixed_input, (hyper_input, hyper_input)

        if self.config.hc_per_branch_norm:
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
            from sglang.kernels.ops.elementwise.hc_mix import (
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
        return mixed_input, (hyper_input, hyper_input_normed)

    def combine(self, block_output: torch.Tensor, residuals) -> torch.Tensor:
        if isinstance(residuals, _HCSumContext):
            return residuals.operator.combine(
                residuals.state, residuals.buffers, block_output
            )
        hyper_input, hyper_input_normed = residuals
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
