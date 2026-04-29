from __future__ import annotations

from sglang.srt.relaykv import RelayKVConfig, make_shadow_plan
from sglang.srt.relaykv.memory import (
    estimate_kv_memory_for_plan,
    estimate_kv_memory_from_metadata,
    validate_shadow_log_schema,
)


class _FakeModelConfig:
    num_hidden_layers = 28
    num_key_value_heads = 2
    num_attention_heads = 16
    hidden_size = 2048
    head_dim = 128


def _assert_close(actual: float, expected: float) -> None:
    if abs(actual - expected) > 1e-3:
        raise AssertionError(f"expected {expected}, got {actual}")


def main() -> None:
    plan = make_shadow_plan(
        2535,
        RelayKVConfig(
            enabled=True,
            mode="shadow",
            resident_budget_tokens=1024,
            recent_window=768,
            anchor_pages=4,
        ),
    )
    estimate = estimate_kv_memory_from_metadata(
        seq_len=plan.seq_len,
        planned_resident_tokens=plan.planned_resident_tokens,
        planned_cold_tokens=plan.planned_cold_tokens,
        num_layers=28,
        num_key_value_heads=2,
        head_dim=128,
        kv_dtype_bytes=2,
    )

    if estimate.num_layers != 28:
        raise AssertionError(estimate)
    if estimate.kv_bytes_per_token != 28672:
        raise AssertionError(estimate)
    _assert_close(estimate.planned_resident_kv_mib, 28.0)
    _assert_close(estimate.planned_cold_kv_mib, 41.316)

    estimate_from_model = estimate_kv_memory_for_plan(
        plan,
        model_config=_FakeModelConfig(),
        kv_dtype="torch.bfloat16",
    )
    payload = plan.to_log_dict()
    payload.update(estimate_from_model.to_log_dict())
    validate_shadow_log_schema(payload)

    print("relaykv_memory_smoke: ok")
    print(payload)


if __name__ == "__main__":
    main()
