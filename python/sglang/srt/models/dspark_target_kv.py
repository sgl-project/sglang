"""Explicit KV-input DSpark checkpoint architecture; legacy exports stay separate."""

from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn.functional as F
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.models.dflash import DFlashAttention, DFlashDecoderLayer
from sglang.srt.models.dspark import DSparkDraftModel, gather_and_crop_vocab
from sglang.srt.speculative.dspark_components.dspark_target_kv_contract import (
    SharedHeadTransform,
    read_target_kv_draft_contract,
)
from sglang.srt.speculative.dspark_components.dspark_target_kv_encoder import (
    TargetKVContextEncoder,
)
from sglang.srt.speculative.ragged_verify import (
    RaggedVerifyMode,
    read_ragged_verify_mode,
)
from sglang.srt.training_capture.protocol import ContractError


class TargetKVRMSNorm(RMSNorm):
    """Preserve the training backbone's narrow residual-add boundary."""

    def __init__(self, hidden_size, eps):
        super().__init__(hidden_size, eps=eps, cast_x_before_out_mul=True)

    def forward(self, x, residual=None):
        if residual is None:
            return super().forward(x)
        residual = x + residual
        return super().forward(residual), residual


class TargetKVAttention(DFlashAttention):
    round_rope_intermediates = True

    def __init__(self, config, layer_id, quant_config=None):
        super().__init__(config, layer_id, quant_config)
        self.attn.use_target_kv_attention = True
        rotary = self.rotary_emb
        if (
            not hasattr(rotary, "cos_sin_cache")
            or rotary.rotary_dim != self.head_dim
            or not rotary.is_neox_style
        ):
            raise ContractError("KV draft requires full split-half table RoPE")

    def _rotate(self, positions, value):
        table = self.rotary_emb.cos_sin_cache[positions].to(value.dtype)
        cosine, sine = table.chunk(2, dim=-1)
        cosine, sine = cosine[:, None], sine[:, None]
        first, second = value.reshape(len(positions), -1, self.head_dim).chunk(2, -1)
        return torch.cat(
            (first * cosine - second * sine, second * cosine + first * sine), -1
        ).reshape_as(value)

    def apply_qk_rope(self, positions, q, k):
        return self._rotate(positions, q), self._rotate(positions, k)

    def apply_k_rope(self, positions, k):
        return self._rotate(positions, k)


class TargetKVSiluAndMul(SiluAndMul):
    def forward(self, x):
        # The training MLP rounds SiLU before multiplying the up projection.
        return self.forward_native(x)


class TargetKVDecoderLayer(DFlashDecoderLayer):
    attention_cls = TargetKVAttention

    def __init__(self, config, layer_id, quant_config=None):
        super().__init__(config, layer_id, quant_config)
        eps = float(getattr(config, "rms_norm_eps", 1e-6))
        self.input_layernorm = TargetKVRMSNorm(config.hidden_size, eps)
        self.post_attention_layernorm = TargetKVRMSNorm(config.hidden_size, eps)
        self.mlp.act_fn = TargetKVSiluAndMul()


class DSparkTargetKVDraftModel(DSparkDraftModel):
    decoder_layer_cls = TargetKVDecoderLayer

    def __init__(self, config, quant_config=None, prefix=""):
        contract = read_target_kv_draft_contract(config)
        if contract is None:
            raise ContractError(
                "KV draft architecture requires explicit input_mode=target_kv"
            )
        if quant_config is not None:
            raise ContractError(
                "KV draft v1 requires unquantized encoder/backbone weights"
            )
        if read_ragged_verify_mode() is not RaggedVerifyMode.STATIC:
            raise ContractError(
                "confidence-disabled KV draft requires static verify scheduling"
            )
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)
        self.norm = TargetKVRMSNorm(
            config.hidden_size, float(getattr(config, "rms_norm_eps", 1e-6))
        )
        self.target_kv_contract = contract
        self.shared_head_transform = SharedHeadTransform.decode(
            contract.teacher.output_transform
        )
        # SpecForge's Qwen3 norms round normalized activations before weighting.
        for module in self.modules():
            if isinstance(module, RMSNorm):
                module.cast_x_before_out_mul = True
        # The old hidden projection is absent from this architecture/state_dict.
        del self.fc, self.hidden_norm
        self.kv_encoder = TargetKVContextEncoder(contract)
        if self.gamma != contract.sequence.prediction_count:
            raise ContractError("draft block_size and KV prediction_count disagree")

    def project_target_hidden(self, target_hidden):
        raise ContractError(
            "KV-input DSpark cannot consume selected target hidden states"
        )

    def compute_base_logits(self, hidden):
        if self.lm_head is None:
            raise ContractError("KV draft requires the bound shared target head")
        weight = self.lm_head.weight
        logits = gather_and_crop_vocab(
            F.linear(hidden.to(weight.dtype), weight), self.lm_head
        )
        return self.shared_head_transform.apply(logits), None

    def write_target_kv(self, *, target_kv, pool, positions, cache_loc):
        self.write_context_kv(
            ctx_hidden=self.kv_encoder(target_kv, positions),
            pool=pool,
            positions=positions,
            cache_loc=cache_loc,
        )

    def load_weights(self, weights):
        """Validate the complete unquantized export before touching parameters."""
        parameters = dict(self.named_parameters())
        coverage = defaultdict(set)
        normalized = []
        shard_mapping = (
            (".q_proj.", ".qkv_proj.", "q"),
            (".k_proj.", ".qkv_proj.", "k"),
            (".v_proj.", ".qkv_proj.", "v"),
            (".gate_proj.", ".gate_up_proj.", "gate"),
            (".up_proj.", ".gate_up_proj.", "up"),
        )
        for original_name, weight in weights:
            name = original_name.removeprefix("model.")
            parameter_name, shard = name, "full"
            # A Markov gate_proj is a full parameter, not a backbone MLP shard.
            if name not in parameters:
                for source, target, part in shard_mapping:
                    if source in name:
                        parameter_name, shard = name.replace(source, target), part
                        break
            if parameter_name not in parameters:
                raise ContractError(f"unexpected KV draft weight: {original_name}")
            self._validate_checkpoint_tensor(
                original_name, parameter_name, parameters[parameter_name], shard, weight
            )
            if shard in coverage[parameter_name] or (
                coverage[parameter_name]
                and (shard == "full" or "full" in coverage[parameter_name])
            ):
                raise ContractError(f"duplicate KV draft weight: {original_name}")
            coverage[parameter_name].add(shard)
            normalized.append((name, weight))
        for name in parameters:
            expected = (
                {"q", "k", "v"}
                if ".qkv_proj." in name
                else {"gate", "up"}
                if ".gate_up_proj." in name
                else {"full"}
            )
            if coverage[name] not in ({"full"}, expected):
                raise ContractError(f"missing KV draft weight or shard: {name}")
        super().load_weights(normalized)
        self._fused_kv_write_cache = None
        self._stacked_ctx_kv_cache = False

    @torch.no_grad()
    def _validate_checkpoint_tensor(
        self, original_name, name, parameter, shard, weight
    ):
        if (
            not isinstance(weight, torch.Tensor)
            or weight.layout != torch.strided
            or weight.is_meta
            or weight.dtype not in (torch.float32, torch.float16, torch.bfloat16)
        ):
            raise ContractError(
                f"KV draft weight must be a dense FP32/FP16/BF16 tensor: {original_name}"
            )
        expected = list(parameter.shape)
        module = self.get_submodule(name.rsplit(".", 1)[0])
        if isinstance(module, QKVParallelLinear):
            # Checkpoint heads are logical heads, before TP replication/sharding.
            sizes = [
                module.total_num_heads * module.head_size,
                module.total_num_kv_heads * module.head_size,
                module.total_num_kv_heads * module.v_head_size,
            ]
            expected[0] = (
                sum(sizes)
                if shard == "full"
                else sizes[{"q": 0, "k": 1, "v": 2}[shard]]
            )
        elif isinstance(module, MergedColumnParallelLinear):
            expected[0] = (
                sum(module.output_sizes)
                if shard == "full"
                else module.output_sizes[{"gate": 0, "up": 1}[shard]]
            )
        elif isinstance(module, ColumnParallelLinear):
            expected[0] = module.output_size
        elif isinstance(module, RowParallelLinear) and parameter.ndim == 2:
            expected[1] = module.input_size
        elif shard != "full":
            index = {"q": 0, "k": 1, "v": 2, "gate": 0, "up": 1}[shard]
            expected[0] = module.output_sizes[index]
        if tuple(weight.shape) != tuple(expected):
            raise ContractError(
                f"KV draft weight shape mismatch for {original_name}: "
                f"expected {tuple(expected)}, got {tuple(weight.shape)}"
            )
        # Extrema detect NaN/Inf and overflow in the destination dtype without
        # allocating a converted copy or a boolean mask of the whole weight.
        extrema = torch.stack(torch.aminmax(weight)).to(dtype=parameter.dtype)
        if not torch.isfinite(extrema).all().item():
            raise ContractError(
                f"KV draft weight is nonfinite in {parameter.dtype}: {original_name}"
            )


EntryClass = DSparkTargetKVDraftModel
