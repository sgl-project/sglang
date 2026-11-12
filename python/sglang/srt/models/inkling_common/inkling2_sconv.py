import torch
from torch import nn

from sglang.kernels.ops.inkling2.linear import linear
from sglang.kernels.ops.inkling2.sconv import (
    routed_sconv_epilogue,
    short_convolution,
)
from sglang.srt.layers.utils.common import set_weight_attrs
from sglang.srt.models.inkling_common.kernels.sconv import update_sconv_cache
from sglang.srt.models.inkling_common.sconv import (
    SconvType,
)
from sglang.srt.models.inkling_common.sconv import (
    ShortConvolution as InklingShortConvolution,
)
from sglang.srt.runtime_context import get_parallel


class ShortConvolution(InklingShortConvolution):
    def __init__(
        self,
        hidden_size: int,
        kernel_size: int,
        sconv_type: SconvType,
        *,
        layer_id: int,
        tp_rank: int | None = None,
        tp_size: int | None = None,
        is_dynamic: bool = False,
        dynamic_input_size: int | None = None,
        n_heads: int = 1,
    ):
        super().__init__(
            hidden_size, kernel_size, sconv_type, layer_id=layer_id, tp_rank=tp_rank
        )
        self.tp_size = get_parallel().attn_tp_size if tp_size is None else tp_size
        self.is_dynamic = is_dynamic
        self.n_heads = n_heads
        if is_dynamic:
            assert dynamic_input_size is not None and hidden_size % n_heads == 0
            self.weight = None
            self.weight_proj = nn.Parameter(
                torch.empty(n_heads * kernel_size, dynamic_input_size),
                requires_grad=False,
            )
            self.static_w = nn.Parameter(
                torch.empty(kernel_size, hidden_size), requires_grad=False
            )
            set_weight_attrs(
                self.weight_proj, {"weight_loader": self.weight_proj_loader}
            )
            set_weight_attrs(self.static_w, {"weight_loader": self.static_w_loader})

    def weight_proj_loader(self, param, loaded_weight):
        if loaded_weight.shape != param.shape:
            assert loaded_weight.shape[0] % self.tp_size == 0
            loaded_weight = loaded_weight.narrow(
                0, self.tp_rank * param.shape[0], param.shape[0]
            )
        assert loaded_weight.shape == param.shape
        param.data.copy_(loaded_weight)

    def static_w_loader(self, param, loaded_weight):
        if loaded_weight.shape != param.shape:
            loaded_weight = loaded_weight.narrow(
                1, self.tp_rank * param.shape[1], param.shape[1]
            )
        assert loaded_weight.shape == param.shape
        param.data.copy_(loaded_weight)

    def forward(self, hidden_states, positions, forward_batch, *, dynamic_input=None):
        mode = forward_batch.forward_mode
        if not (mode.is_decode() or mode.is_extend()):
            raise NotImplementedError(
                "Short convolution supports prefill, decode, and target verification"
            )
        meta = self._conv_state(forward_batch)
        cache = self._sconv_cache()
        if self.is_dynamic:
            assert dynamic_input is not None
            weight = self.static_w
            dynamic_weights = linear(
                dynamic_input.to(hidden_states.dtype),
                self.weight_proj.to(hidden_states.dtype),
            ).view(hidden_states.shape[0], self.n_heads, self.kernel_size[0])
        else:
            assert dynamic_input is None
            weight, dynamic_weights = self._weight_2d(), None
        decode, verify = mode.is_decode(), mode.is_target_verify()
        extend = not decode and not verify
        if extend:
            self._prepare_extend_sconv_cache(
                forward_batch, cache, hidden_states, meta.track_conv_indices
            )
        output = short_convolution(
            hidden_states,
            weight,
            cache,
            meta.precomputed,
            dynamic_weights=dynamic_weights,
            decode=decode,
            track_mask=forward_batch.mamba_track_mask if decode else None,
            track_indices=meta.track_cache_indices if decode else None,
            window=self._intermediate_window() if verify else None,
        )
        if extend:
            update_sconv_cache(
                x=hidden_states,
                sconv_cache=cache,
                cache_indices=meta.cache_indices,
                has_initial_state=meta.has_initial_state,
                query_start_loc=meta.query_start_loc,
            )
        return output

    def routed_decode_epilogue(self, routed, attention, partial, forward_batch):
        """Decode: combine routed expert rows, convolve, and return the residual
        update (partial + attention + conv, or attention + conv without partial)."""
        assert not self.is_dynamic
        meta = self._conv_state(forward_batch)
        verify = forward_batch.forward_mode.is_target_verify()
        return routed_sconv_epilogue(
            routed.rows,
            routed.src2dst,
            routed.top_k,
            self._weight_2d(),
            self._sconv_cache(),
            meta.precomputed,
            attention,
            partial,
            track_mask=None if verify else forward_batch.mamba_track_mask,
            track_indices=None if verify else meta.track_cache_indices,
            window=self._intermediate_window() if verify else None,
            round_bf16=routed.rows.dtype == torch.float32,
        )
