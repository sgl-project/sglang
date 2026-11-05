"""Translate the supported deployment profile to native source execution."""

from pathlib import Path

from ...contracts import CostProvider, ExecutionTrace
from .qwen_profile import QwenCostMapping
from .source import verify_source


class SGLangAdapter:
    def __init__(self, source_root, mode):
        if mode != "afd":
            raise ValueError(f"AFD_ADAPTER_MODE_UNSUPPORTED: {mode}")
        if source_root is None:
            raise ValueError("AFD_SOURCE_ROOT_REQUIRED")
        self.source_root = Path(source_root)
        self.mode = mode

    def cost_mapping(self, config):
        if config["mode"] != self.mode:
            raise ValueError("AFD_ADAPTER_CONFIG_MISMATCH")
        return QwenCostMapping(config)

    def template_config(self, *, batch_size, effective_kv, ffn_ranks):
        return dict(
            model="qwen3_235b",
            mode=self.mode,
            ffn_ranks=ffn_ranks,
            batch_size=batch_size,
            effective_kv=effective_kv,
            layers=94,
        )

    def verify(self):
        return verify_source(self.source_root, mode=self.mode)

    def execute(self, provider: CostProvider) -> ExecutionTrace:
        config = provider.config
        if config["mode"] != self.mode:
            raise ValueError("AFD_ADAPTER_CONFIG_MISMATCH")
        mapping = self.cost_mapping(config)
        sources = self.verify()
        from .afd import run

        result = run(
            provider,
            source_root=self.source_root,
            F=config["ffn_ranks"],
            layers=config["layers"],
            stage_rows=mapping.layout.stage_rows,
            full_decode=True,
        )
        devices = mapping.layout.attention_ranks + mapping.layout.ffn_ranks
        return ExecutionTrace(result["events"], sources, devices)
