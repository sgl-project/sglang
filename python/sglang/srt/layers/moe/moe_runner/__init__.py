from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig


def __getattr__(name: str):
    if name == "MoeRunner":
        from sglang.srt.layers.moe.moe_runner.runner import MoeRunner

        return MoeRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["MoeRunnerConfig", "MoeRunner"]
