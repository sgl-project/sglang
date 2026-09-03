# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/moe_wna16.py
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
import torch

from sglang.srt.distributed.parallel_state import get_tp_group
from sglang.srt.layers.moe import MoeRunner, MoeRunnerBackend, MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo
from sglang.srt.layers.quantization.awq import AWQConfig
from sglang.srt.layers.quantization.base_config import (
    FusedMoEMethodBase,
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.gptq import GPTQConfig, GPTQMarlinConfig
from sglang.srt.layers.quantization.unquant import (
    UnquantizedFusedMoEMethod,
    UnquantizedLinearMethod,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import get_device_capability, set_weight_attrs

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )


def get_weight_perm(num_bits: int):
    perm_list: List[int] = []
    for i in range(32):
        perm1: List[int] = []
        col = i // 4
        for block in [0, 1]:
            for row in [
                2 * (i % 4),
                2 * (i % 4) + 1,
                2 * (i % 4 + 4),
                2 * (i % 4 + 4) + 1,
            ]:
                perm1.append(16 * row + col + 8 * block)
        for j in range(4):
            perm_list.extend([p + 256 * j for p in perm1])

    perm = np.array(perm_list)

    if num_bits == 4:
        interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7])
    elif num_bits == 8:
        interleave = np.array([0, 2, 1, 3])
    else:
        raise Exception("num_bits must be 4 or 8, got {}".format(num_bits))

    perm = perm.reshape((-1, len(interleave)))[:, interleave].ravel()
    perm = torch.from_numpy(perm)
    return perm


class MoeWNA16Config(QuantizationConfig):
    """Config class for MOE WNA16 (W8A16/W4A16/W2A16) quantization."""

    def __init__(
        self,
        linear_quant_method: str,
        weight_bits: int,
        group_size: int,
        has_zp: bool,
        lm_head_quantized: bool,
        modules_to_not_convert: Optional[List[str]],
        full_config: Dict[str, Any],
    ) -> None:
        super().__init__()
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.has_zp = has_zp
        self.bit8_pack_factor = 8 // self.weight_bits
        self.lm_head_quantized = lm_head_quantized
        self.linear_quant_method = linear_quant_method
        self.full_config = full_config
        self.use_marlin = False
        # Avoid circular import

        if self.linear_quant_method == "gptq":
            self.use_marlin = GPTQMarlinConfig.is_gptq_marlin_compatible(full_config)
        elif self.linear_quant_method == "awq":
            capability_tuple = get_device_capability()
            device_capability = (
                -1
                if capability_tuple is None
                else capability_tuple[0] * 10 + capability_tuple[1]
            )
            awq_min_capability = AWQConfig.get_min_capability()
            if device_capability < awq_min_capability:
                raise ValueError(
                    "The quantization method moe_wna16 + awq is not supported "
                    "for the current GPU. "
                    f"Minimum capability: {awq_min_capability}. "
                    f"Current capability: {device_capability}."
                )
        else:
            raise ValueError("moe_wna16 only support gptq and awq.")

        if modules_to_not_convert is None:
            self.modules_to_not_convert = []
        else:
            self.modules_to_not_convert = modules_to_not_convert

    @classmethod
    def get_name(cls) -> str:
        return "moe_wna16"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return ["quantize_config.json"]

    def get_scaled_act_names(self) -> List[str]:
        raise NotImplementedError

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> MoeWNA16Config:
        quant_method = cls.get_from_keys(config, ["quant_method"])
        weight_bits = cls.get_from_keys(config, ["bits"])
        group_size = cls.get_from_keys(config, ["group_size"])
        lm_head_quantized = cls.get_from_keys_or(config, ["lm_head"], default=False)
        if quant_method == "gptq":
            has_zp = not cls.get_from_keys(config, ["sym"])
            modules_to_not_convert = []
        elif quant_method == "awq":
            has_zp = cls.get_from_keys(config, ["zero_point"])
            modules_to_not_convert = cls.get_from_keys_or(
                config, ["modules_to_not_convert"], None
            )
        else:
            raise ValueError("moe_wna16 only support gptq and awq.")

        return cls(
            quant_method,
            weight_bits,
            group_size,
            has_zp,
            lm_head_quantized,
            modules_to_not_convert,
            config,
        )

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant) -> Optional[str]:
        if user_quant == "moe_wna16" and cls.is_moe_wna16_compatible(hf_quant_cfg):
            return cls.get_name()
        return None

    @classmethod
    def is_moe_wna16_compatible(cls, quant_config: Dict[str, Any]):
        # Extract data from quant config.
        quant_method = quant_config.get("quant_method", "").lower()
        num_bits = quant_config.get("bits")
        desc_act = quant_config.get("desc_act")

        capability_tuple = get_device_capability()
        device_capability = (
            -1
            if all(capability is None for capability in capability_tuple)
            else capability_tuple[0] * 10 + capability_tuple[1]
        )
        # Avoid circular import
        awq_min_capability = AWQConfig.get_min_capability()

        # 2-bit added: the Triton kernel now also unpacks 4 values per byte
        # (use_int2_w2a16). Needed because a 176B MoE only fits on 24 GB VRAM
        # + 30 GB RAM at ~2 bpw - 4-bit would be 62 GiB.
        gptq_compatible = (
            quant_method == "gptq" and not desc_act and num_bits in [2, 4, 8]
        )
        awq_compatible = (
            quant_method == "awq"
            and num_bits == 4
            and device_capability >= awq_min_capability
        )

        return gptq_compatible or awq_compatible

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        # avoid circular import
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

        if is_layer_skipped_quant(prefix, self.modules_to_not_convert):
            if isinstance(layer, FusedMoE):
                return UnquantizedFusedMoEMethod()
            return UnquantizedLinearMethod()
        elif isinstance(layer, LinearBase):

            if self.linear_quant_method == "gptq":
                if self.use_marlin:
                    return GPTQMarlinConfig.from_config(
                        self.full_config
                    ).get_quant_method(layer, prefix)
                else:
                    return GPTQConfig.from_config(self.full_config).get_quant_method(
                        layer, prefix
                    )
            elif self.linear_quant_method == "awq":
                return AWQConfig.from_config(self.full_config).get_quant_method(
                    layer, prefix
                )
            else:
                raise ValueError("moe_wna16 only support gptq and awq.")
        elif isinstance(layer, FusedMoE):
            return MoeWNA16Method(self)
        return None


def is_layer_skipped_quant(prefix: str, modules_to_not_convert: List[str]):
    return any(module_name in prefix for module_name in modules_to_not_convert)


class MoeWNA16Method(FusedMoEMethodBase):
    """Linear method for MOE WNA16 (W8A16/W4A16/W2A16) quantization.

    Args:
        quant_config: The MOE WNA16 (W8A16/W4A16) quantization config.
    """

    def __init__(self, quant_config: MoeWNA16Config):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

        layer.quant_config = self.quant_config
        bit8_pack_factor = self.quant_config.bit8_pack_factor
        group_size = self.quant_config.group_size
        group_size_div_factor = 1

        # make intermediate_size and hidden_size diviable by group_size
        # we reduce the group size to ensure that
        # and we would repeat the loaded_weight later
        while intermediate_size_per_partition % group_size or hidden_size % group_size:
            group_size = group_size // 2
            group_size_div_factor *= 2
            assert group_size >= 32
        layer.group_size = group_size
        layer.group_size_div_factor = group_size_div_factor

        strategy = FusedMoeWeightScaleSupported.GROUP.value
        extra_weight_attrs.update({"quant_method": strategy, "is_transposed": False})

        assert "weight_loader" in extra_weight_attrs
        weight_loader = extra_weight_attrs["weight_loader"]
        wrapped_weight_loader = MoeWNA16Method.get_weight_loader(layer, weight_loader)
        extra_weight_attrs["weight_loader"] = wrapped_weight_loader

        # Fused gate_up_proj (column parallel)
        w13_qweight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // bit8_pack_factor,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_qweight", w13_qweight)
        set_weight_attrs(w13_qweight, extra_weight_attrs)

        # down_proj (row parallel)
        w2_qweight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // bit8_pack_factor,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_qweight", w2_qweight)
        set_weight_attrs(w2_qweight, extra_weight_attrs)

        w13_scales = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // group_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_scales", w13_scales)
        set_weight_attrs(w13_scales, extra_weight_attrs)

        w2_scales = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // group_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_scales", w2_scales)
        set_weight_attrs(w2_scales, extra_weight_attrs)

        if self.quant_config.has_zp:
            w13_qzeros = torch.nn.Parameter(
                torch.zeros(
                    num_experts,
                    2 * intermediate_size_per_partition // bit8_pack_factor,
                    hidden_size // group_size,
                    dtype=torch.uint8,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_qzeros", w13_qzeros)
            set_weight_attrs(w13_qzeros, extra_weight_attrs)

            w2_qzeros = torch.nn.Parameter(
                torch.zeros(
                    num_experts,
                    hidden_size // bit8_pack_factor,
                    intermediate_size_per_partition // group_size,
                    dtype=torch.uint8,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w2_qzeros", w2_qzeros)
            set_weight_attrs(w2_qzeros, extra_weight_attrs)

        if self.quant_config.linear_quant_method == "gptq":
            # some param are unused, but we need to init them in order to
            # load weights
            invalid_param_keys = ["w13_g_idx", "w2_g_idx"]
            if not self.quant_config.has_zp:
                invalid_param_keys += ["w13_qzeros", "w2_qzeros"]
            for key in invalid_param_keys:
                param = torch.nn.Parameter(
                    torch.empty((0,), dtype=torch.int32), requires_grad=False
                )
                layer.register_parameter(key, param)
                set_weight_attrs(param, extra_weight_attrs)

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config
        self.runner = MoeRunner(MoeRunnerBackend.TRITON, moe_runner_config)

    def _process_weights_after_loading_disabled(self, layer) -> None:
        """Set up expert streaming right after this layer finished loading."""
        self._maybe_expert_streamer(layer)

    def get_triton_quant_info(self, layer: torch.nn.Module) -> TritonMoeQuantInfo:
        weight_bits = self.quant_config.weight_bits
        has_zp = self.quant_config.has_zp
        return TritonMoeQuantInfo(
            w13_weight=layer.w13_qweight,
            w2_weight=layer.w2_qweight,
            use_int4_w4a16=weight_bits == 4,
            use_int8_w8a16=weight_bits == 8,
            use_int2_w2a16=weight_bits == 2,
            w13_scale=layer.w13_scales,
            w2_scale=layer.w2_scales,
            w13_zp=layer.w13_qzeros if has_zp else None,
            w2_zp=layer.w2_qzeros if has_zp else None,
            block_shape=[0, layer.group_size],
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """2-bit: re-lay each expert tensor as N-contiguous int32 words / [E, K/128, N] scales.

        Same bytes, re-arranged once. The tiled kernel's B loads become coalesced 128-B
        lines and the decode GEMV can read experts in place. Offloaded tensors live in
        pinned host memory at this point; re-pin after the copy.
        """
        if self.quant_config.weight_bits != 2:
            return
        if os.environ.get("SGLANG_MOE_NCONTIG", "1") != "1":
            return
        from sglang.srt.layers.moe.expert_gemv import to_word_ncontig

        for name in ("w13_qweight", "w2_qweight", "w13_scales", "w2_scales"):
            p = getattr(layer, name, None)
            if p is None or p.data.dim() != 3:
                continue
            t = p.data
            was_pinned = (not t.is_cuda) and t.is_pinned()
            if name.endswith("qweight"):
                t2 = to_word_ncontig(t)  # [E, K/16, N] int32
            else:
                t2 = t.transpose(1, 2).contiguous()  # [E, K/128, N]
            if was_pinned:
                t2 = t2.pin_memory()
            p.data = t2
            del t
        layer._b_n_contig = True
        self._collect_for_placement(layer)

    _PLACE_LAYERS = []  # layers seen so far (loader order)
    _PLACE_FREQ = None

    def _collect_for_placement(self, layer):
        path = os.environ.get("SGLANG_MOE_PLACEMENT")
        if not path:
            return
        cls = type(self)
        if cls._PLACE_FREQ is None:
            cls._PLACE_FREQ = torch.load(path)["mass"]
        n_layers = int(cls._PLACE_FREQ.shape[0])
        lid = int(getattr(layer, "layer_id", -1))
        if 0 <= lid < n_layers:
            cls._PLACE_LAYERS.append(layer)
        if len(cls._PLACE_LAYERS) == n_layers:
            if os.environ.get("SGLANG_MOE_ELASTIC") == "1":
                from sglang.srt.layers.moe.expert_elastic import ExpertElastic

                inst = ExpertElastic(
                    cls._PLACE_FREQ,
                    int(os.environ.get("SGLANG_MOE_PLACEMENT_S", "184")),
                )
                inst.place_all(cls._PLACE_LAYERS)
                cls._ELASTIC = inst
            else:
                self._run_placement()
            cls._PLACE_LAYERS = []

    def _run_placement(self):
        """Interleaved, memory-neutral hot/cold split over all layers."""
        cls = type(self)
        freq = cls._PLACE_FREQ
        S_ = int(os.environ.get("SGLANG_MOE_PLACEMENT_S", "184"))
        names = ("w13_qweight", "w2_qweight", "w13_scales", "w2_scales")
        layers = cls._PLACE_LAYERS
        host = [l for l in layers if not l.w13_qweight.data.is_cuda]
        gpu = [l for l in layers if l.w13_qweight.data.is_cuda]
        logger.info(
            "expert placement: %d host layers, %d GPU layers, S=%d",
            len(host),
            len(gpu),
            S_,
        )
        pools = {n: [] for n in names}  # free host slots: (tensor, row)
        extra_keep = []

        def split_ids(l):
            order = torch.argsort(freq[int(l.layer_id)], descending=True)
            return order[:S_].sort().values, order[S_:].sort().values

        def place_host(l):
            hot_ids, cold_ids = split_ids(l)
            E = int(l.w13_qweight.data.shape[0])
            addr, proto, keep = {}, {}, []
            for name in names:
                p = getattr(l, name, None)
                if p is None:
                    continue
                t = p.data
                rb = t[0].numel() * t.element_size()
                tab = torch.empty(E, dtype=torch.int64)
                if t.is_cuda:  # split layer: this kind is on the GPU
                    tab, h, kp = place_gpu_kind(t, hot_ids, cold_ids, name, rb, E)
                    keep += kp
                else:
                    h = t.index_select(0, hot_ids).to("cuda").contiguous()
                    tab[hot_ids] = h.data_ptr() + torch.arange(S_) * rb
                    tab[cold_ids] = t.data_ptr() + cold_ids * rb
                    pools[name].extend((t, int(r)) for r in hot_ids.tolist())
                    keep.append(t)
                p.data = h
                addr[name] = tab.cuda()
                proto[name] = h
                del t
            l._placed = {"addr": addr, "proto": proto, "E": E, "S": S_, "keep": keep}
            l._gemv_tabs = None

        def place_gpu_kind(t, hot_ids, cold_ids, name, rb, E):
            tab = torch.empty(E, dtype=torch.int64)
            keep = []
            need = int(cold_ids.numel())
            pool = pools[name]
            if len(pool) < need:
                extra = torch.empty(
                    (need - len(pool),) + tuple(t.shape[1:]), dtype=t.dtype
                ).pin_memory()
                pool.extend((extra, i) for i in range(need - len(pool)))
                extra_keep.append(extra)
                keep.append(extra)
            slots = [pool.pop() for _ in range(need)]
            cold_cpu = t.index_select(0, cold_ids.to(t.device)).to("cpu")
            for i, (dst, r) in enumerate(slots):
                dst[r].copy_(cold_cpu[i])
                tab[int(cold_ids[i])] = dst.data_ptr() + r * rb
                keep.append(dst)
            del cold_cpu
            h = t.index_select(0, hot_ids.to(t.device)).contiguous()
            tab[hot_ids] = h.data_ptr() + torch.arange(S_) * rb
            return tab, h, keep

        def place_gpu(l):
            hot_ids, cold_ids = split_ids(l)
            E = int(l.w13_qweight.data.shape[0])
            addr, proto, keep = {}, {}, []
            for name in names:
                p = getattr(l, name, None)
                if p is None:
                    continue
                t = p.data
                rb = t[0].numel() * t.element_size()
                tab, h, kp = place_gpu_kind(t, hot_ids, cold_ids, name, rb, E)
                keep += kp
                p.data = h
                addr[name] = tab.cuda()
                proto[name] = h
                del t
            torch.cuda.empty_cache()
            l._placed = {"addr": addr, "proto": proto, "E": E, "S": S_, "keep": keep}
            l._gemv_tabs = None

        # interleave so donated slots exist before they are consumed: ~1.8 host layers per GPU layer
        hi = gi = 0
        ratio = max(1.0, (E_cold := (512 - S_)) / max(S_, 1))
        credit = 0.0
        while hi < len(host) or gi < len(gpu):
            if hi < len(host) and (credit < ratio or gi >= len(gpu)):
                place_host(host[hi])
                hi += 1
                credit += 1.0
            else:
                place_gpu(gpu[gi])
                gi += 1
                credit -= ratio
        free_left = {n: len(v) for n, v in pools.items()}
        logger.info(
            "expert placement done: free host slots left %s, pinned fallbacks %d",
            free_left,
            len(extra_keep),
        )
        cls._PLACE_EXTRA = extra_keep

    def _gemv_tables(self, layer):
        tabs = getattr(layer, "_gemv_tabs", None)
        if tabs is None:
            from sglang.srt.layers.moe.expert_gemv import make_tables

            placed = getattr(layer, "_placed", None)
            if placed is None:
                w13, s13 = layer.w13_qweight.data, layer.w13_scales.data
                w2, s2 = layer.w2_qweight.data, layer.w2_scales.data
                tabs = (
                    make_tables(w13, s13),
                    make_tables(w2, s2),
                    w13.shape[2],
                    w13.shape[1] * 16,
                    w2.shape[2],
                    w2.shape[1] * 16,
                    s13.dtype == torch.bfloat16,
                )
            else:
                a, pr = placed["addr"], placed["proto"]
                h13, h2 = pr["w13_qweight"], pr["w2_qweight"]
                tabs = (
                    (
                        a["w13_qweight"],
                        a["w13_scales"],
                        h13.stride(1),
                        pr["w13_scales"].stride(1),
                    ),
                    (
                        a["w2_qweight"],
                        a["w2_scales"],
                        h2.stride(1),
                        pr["w2_scales"].stride(1),
                    ),
                    h13.shape[2],
                    h13.shape[1] * 16,
                    h2.shape[2],
                    h2.shape[1] * 16,
                    pr["w13_scales"].dtype == torch.bfloat16,
                )
            layer._gemv_tabs = tabs
        return tabs

    def _apply_gemv(self, layer, dispatch_output):
        """Batch-1 path: T independent 2-bit GEMVs reading experts in place."""
        from sglang.srt.layers.moe.expert_gemv import moe_gemv_int2_tab
        from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

        x = dispatch_output.hidden_states
        topk = dispatch_output.topk_output
        ids = topk.topk_ids.reshape(-1)
        w = topk.topk_weights.reshape(-1).to(torch.float32)
        M_, top_k = topk.topk_ids.shape
        (wt13, st13, sw13, ss13), (wt2, st2, sw2, ss2), N13, K13, N2, K2, sbf = (
            self._gemv_tables(layer)
        )
        inter = N13 // 2
        c13 = moe_gemv_int2_tab(
            x,
            wt13,
            st13,
            ids,
            w,
            N13,
            K13,
            sw13,
            ss13,
            top_k=top_k,
            mul_routed_weight=False,
            scale_bf16=sbf,
        )
        h = torch.nn.functional.silu(c13[:, :inter]) * c13[:, inter:]
        c2 = moe_gemv_int2_tab(
            h,
            wt2,
            st2,
            ids,
            w,
            N2,
            K2,
            sw2,
            ss2,
            top_k=1,
            mul_routed_weight=True,
            scale_bf16=sbf,
        )
        out = c2.view(M_, top_k, N2).sum(dim=1)
        rsf = self.moe_runner_config.routed_scaling_factor
        if rsf is not None and rsf != 1.0:
            out = out * rsf
        return StandardCombineInput(hidden_states=out.to(x.dtype))

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        assert (
            self.moe_runner_config.activation == "silu"
        ), "Only SiLU activation is supported."
        _el = getattr(type(self), "_ELASTIC", None)
        if _el is not None:
            _el.poll(
                int(dispatch_output.hidden_states.shape[0]),
                int(getattr(layer, "layer_id", 0)),
            )

        if (
            getattr(layer, "_b_n_contig", False)
            and self.quant_config.weight_bits == 2
            and dispatch_output.hidden_states.shape[0] <= 16
            and os.environ.get("SGLANG_MOE_GEMV", "1") == "1"
        ):
            return self._apply_gemv(layer, dispatch_output)

        streamer = self._maybe_expert_streamer(layer)
        if streamer is None:
            quant_info = self.get_triton_quant_info(layer)
            return self.runner.run(dispatch_output, quant_info)

        # MoE-aware offloading: gather only the selected experts into a
        # compact buffer and renumber the ids onto it. The Triton kernel
        # stays unchanged.
        topk = dispatch_output.topk_output
        new_ids, tensors = streamer.gather(topk.topk_ids)
        dispatch_output = dispatch_output._replace(
            topk_output=topk._replace(topk_ids=new_ids)
        )
        weight_bits = self.quant_config.weight_bits
        has_zp = self.quant_config.has_zp
        quant_info = TritonMoeQuantInfo(
            w13_weight=tensors["w13_qweight"],
            w2_weight=tensors["w2_qweight"],
            use_int4_w4a16=weight_bits == 4,
            use_int8_w8a16=weight_bits == 8,
            use_int2_w2a16=weight_bits == 2,
            w13_scale=tensors["w13_scales"],
            w2_scale=tensors["w2_scales"],
            w13_zp=tensors.get("w13_qzeros") if has_zp else None,
            w2_zp=tensors.get("w2_qzeros") if has_zp else None,
            block_shape=[0, layer.group_size],
        )
        return self.runner.run(dispatch_output, quant_info)

    def _maybe_expert_streamer(self, layer):
        """Create the streamer once, on demand."""
        st = getattr(layer, "_expert_streamer", "unset")
        if st != "unset":
            return st
        from sglang.srt.layers.moe.expert_stream import ExpertStreamer, enabled

        if not enabled() or not hasattr(layer, "w13_qweight"):
            layer._expert_streamer = None
            return None
        # All four expert tensors resident on the device: the plain path is
        # byte-for-byte what the streamer would produce, minus the copy. The
        # gate is per LAYER, never per tensor - one moe_align result numbers
        # both GEMMs, so a split layer must keep the streamer.
        _names = ("w13_qweight", "w2_qweight", "w13_scales", "w2_scales")
        _ts = [getattr(layer, n).data for n in _names if hasattr(layer, n)]
        if (
            _ts
            and all(t.is_cuda for t in _ts)
            and getattr(layer, "_placed", None) is None
        ):
            layer._expert_streamer = None
            return None
        layer._expert_streamer = ExpertStreamer(layer)
        return layer._expert_streamer

    @staticmethod
    def get_weight_loader(layer, weight_loader):

        def convert_awq_tensor(tensor, tensor_type):
            # convert awq qweight/qzeros to a standard format (assume int4)
            # qweight: (k, n // pack_factor_bit32) -> (n, k // pack_factor_bit8)
            # qzeros: (k // group_size, n // pack_factor_bit32) ->
            #         (n // pack_factor_bit8, k // group_size)
            # pack_factor_bit32 = 32 // weight_bits
            # pack_factor_bit8 = 8 // weight_bits

            # 0. suppose origin shape (a, b), dtype int32
            # 1. convert to uint8, shape (a, b) -> (a, 4 * b)
            size0 = tensor.size(0)
            tensor = tensor.view(torch.uint8)

            # 2. unpack to uint4 (only when weight_bits == 4)
            #    shape (a, 4 * b) -> (a, 4 * b, 2)
            shifter = torch.tensor([0, 4], dtype=torch.uint8, device=tensor.device)
            tensor = (tensor[:, :, None] >> shifter) & 0xF

            # 3. change order, see
            # https://github.com/casper-hansen/AutoAWQ/blob/v0.2.8/awq/utils/quant_utils.py
            # shape -> (a, 4 * b * pack_factor_bit8)
            reverse_awq_pack_order = [0, 4, 1, 5, 2, 6, 3, 7]
            tensor = tensor.view(-1, 8)[:, reverse_awq_pack_order]
            tensor = tensor.view(size0, -1)

            # 4. transpose, shape -> (4 * b * pack_factor_bit8, a)
            tensor = tensor.T.contiguous()

            # 5. repack (only when weight_bits == 4)
            # qweight shape -> (4 * b * pack_factor_bit8, a // pack_factor_bit8)
            # qzeros shape -> (4 * b, a)

            if tensor_type == "qweight":
                tensor = tensor[:, 1::2] * 16 + tensor[:, ::2]
            elif tensor_type == "qzeros":
                tensor = tensor[1::2, :] * 16 + tensor[::2, :]
            return tensor

        def convert_gptq_int4_qzeros(tensor):
            tensor = tensor.view(torch.uint8)
            shifter = torch.tensor([0, 4], dtype=torch.uint8, device=tensor.device)
            tensor = (tensor[:, :, None] >> shifter) & 0xF
            tensor = tensor + 1
            tensor = tensor[:, :, 0] + tensor[:, :, 1] * 16
            return tensor

        def moe_wna16_weight_loader(
            param: torch.nn.Parameter,
            loaded_weight: torch.Tensor,
            weight_name: str,
            shard_id: str,
            expert_id: int,
        ):
            if "g_idx" in weight_name:
                return
            if not layer.quant_config.has_zp and "qzeros" in weight_name:
                return

            device = get_tp_group().device
            tp_rank = get_parallel().tp_rank
            loaded_weight = loaded_weight.to(device)
            shard_size = layer.intermediate_size_per_partition

            # convert gptq and awq weight to a standard format
            if layer.quant_config.linear_quant_method == "awq":
                assert layer.quant_config.weight_bits == 4
                if "weight" in weight_name:
                    loaded_weight = convert_awq_tensor(loaded_weight, "qweight")
                elif "zeros" in weight_name:
                    loaded_weight = convert_awq_tensor(loaded_weight, "qzeros")
                else:
                    loaded_weight = loaded_weight.T
            elif layer.quant_config.linear_quant_method == "gptq":
                assert layer.quant_config.weight_bits in [2, 4, 8]
                if "weight" in weight_name:
                    # Identical for 2/4/8 bits: viewing the int32 storage as
                    # uint8 lays the values out along K (16/8/4 per int32), and
                    # the kernel derives byte and shift from the bit width.
                    # Verified against real AutoRound 2-bit tensors.
                    loaded_weight = loaded_weight.T.contiguous().view(torch.uint8)
                elif "zeros" in weight_name:
                    if layer.quant_config.weight_bits == 2:
                        raise NotImplementedError(
                            "asymmetric 2-bit GPTQ (qzeros) is not implemented "
                            "here; only sym=True is supported"
                        )
                    # add 1 to gptq qzeros to align with awq
                    loaded_weight = loaded_weight.view(torch.uint8)
                    if layer.quant_config.weight_bits == 4:
                        loaded_weight = convert_gptq_int4_qzeros(loaded_weight).T
                    else:
                        loaded_weight = loaded_weight.T + 1
                else:
                    loaded_weight = loaded_weight.T

            # repeat the qzeros/scales to fit new group size
            if (
                layer.group_size_div_factor > 1
                and "qzeros" in weight_name
                or "scales" in weight_name
            ):
                loaded_weight = loaded_weight.repeat_interleave(
                    layer.group_size_div_factor, 1
                )

            if "w13_qzeros" in weight_name:
                tensor = loaded_weight.view(
                    layer.moe_tp_size, -1, loaded_weight.size(1)
                )[tp_rank]
                if shard_id == "w1":
                    param.data[expert_id, : shard_size // 2] = tensor
                else:
                    param.data[expert_id, shard_size // 2 :] = tensor
            elif "w2_qzeros" in weight_name:
                param.data[expert_id] = loaded_weight.view(
                    loaded_weight.size(0), layer.moe_tp_size, -1
                )[:, tp_rank]
            else:
                weight_loader(param, loaded_weight, weight_name, shard_id, expert_id)

        return moe_wna16_weight_loader
