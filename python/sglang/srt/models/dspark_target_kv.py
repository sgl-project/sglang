"""Explicit KV-input DSpark checkpoint architecture; legacy exports stay separate."""

from __future__ import annotations

from collections import defaultdict

import torch.nn.functional as F
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


class DSparkTargetKVDraftModel(DSparkDraftModel):
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
        self.target_kv_contract = contract
        self.shared_head_transform = SharedHeadTransform.decode(
            contract.teacher.output_transform
        )
        # The inherited attention/backbone remains identical. The old hidden
        # projection is not part of the new architecture or its state_dict.
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
        """Reject missing/duplicate/foreign tensors before touching parameters."""
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
            for source, target, part in shard_mapping:
                if source in name:
                    parameter_name, shard = name.replace(source, target), part
                    break
            if parameter_name not in parameters:
                raise ContractError(f"unexpected KV draft weight: {original_name}")
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


EntryClass = DSparkTargetKVDraftModel
