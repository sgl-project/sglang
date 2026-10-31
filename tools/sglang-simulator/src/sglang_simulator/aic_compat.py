"""AIC Python SDK / compiled-core boundary; never substitute missing costs."""

import importlib
import importlib.util
import math
import os
from importlib.metadata import PackageNotFoundError, version


class AICSDK:
    """Select once, then use enums, database and operations from the same SDK.

    Set SGLANG_SIMULATOR_AIC_API to legacy or core to pin the API family.
    Auto prefers core when installed. A broken installation is an error, not
    permission to fall back to a different database or analytical estimate.
    """

    def __init__(self, api=None):
        api = api or os.environ.get("SGLANG_SIMULATOR_AIC_API", "auto")
        if api not in ("auto", "legacy", "core"):
            raise ValueError("AIC API must be auto, legacy or core")
        if api == "auto":
            api = (
                "core" if importlib.util.find_spec("aiconfigurator_core") else "legacy"
            )
        self.api = api
        self.package = "aiconfigurator_core" if api == "core" else "aiconfigurator"
        self.common = self.module("common")
        try:
            self.version = version(self.package.replace("_", "-"))
        except PackageNotFoundError:
            self.version = "source checkout (pin its Git revision separately)"

    def module(self, name):
        return importlib.import_module(f"{self.package}.sdk.{name}")

    def session(self, model, database, backend):
        if self.api == "legacy":
            return self.module("inference_session").InferenceSession(
                model=model, database=database, backend=backend
            )
        return _CoreSession(model, database, backend)

    def query(self, database, kind, *, systems_path=None, **shape):
        """Return milliseconds and unmodified SDK provenance for one operation.

        This is a cost preparation API, not a live request scheduler. It does
        not certify kernel, page layout or module-boundary matching. Callers
        must audit those separately before constructing an AFD cost record.
        """
        args = dict(shape)
        quant_type = {
            "gemm": "GEMMQuantMode",
            "generation_attention": "KVCacheQuantMode",
            "moe": "MoEQuantMode",
        }
        if kind not in quant_type:
            raise NotImplementedError(f"Unsupported AIC operation: {kind}")
        quant = getattr(getattr(self.common, quant_type[kind]), args.pop("quant"))
        if kind == "moe":
            args.setdefault("is_context", False)
        lane_order = args.pop("lane_order", None)
        if self.api == "legacy":
            if lane_order is not None:
                raise NotImplementedError(
                    "Legacy attention query cannot select a kernel lane"
                )
            quant_key = (
                "kvcache_quant_mode" if kind == "generation_attention" else "quant_mode"
            )
            value = getattr(database, f"query_{kind}")(**args, **{quant_key: quant})
            ms, source = float(value), str(value.source)
        else:
            core = importlib.import_module(self.package)
            ops = importlib.import_module(f"{self.package}._aiconfigurator_core")
            runtime = dict(is_context=False, batch_size=1, s=1)
            if kind == "gemm":
                op = ops.GEMM(kind, 1, args.pop("n"), args.pop("k"), quant)
                runtime["x"] = args.pop("m")
            elif kind == "generation_attention":
                options = dict(head_size=args.pop("head_size", 128))
                if lane_order is not None:
                    options["lane_order"] = lane_order
                op = ops.GenerationAttention(
                    kind, 1, args.pop("n"), args.pop("n_kv"), quant, **options
                )
                runtime.update(batch_size=args.pop("b"), s=args.pop("s"))
            else:
                runtime["is_context"] = args.pop("is_context", False)
                op = ops.MoE(
                    kind,
                    1,
                    args.pop("hidden_size"),
                    args.pop("inter_size"),
                    args.pop("topk"),
                    args.pop("num_experts"),
                    args.pop("moe_tp_size"),
                    args.pop("moe_ep_size"),
                    quant,
                    args.pop("workload_distribution"),
                    1,
                    is_context=runtime["is_context"],
                    moe_backend=args.pop("moe_backend", None),
                )
                runtime["x"] = args.pop("num_tokens")
            if args:
                raise NotImplementedError(
                    f"Unsupported core query options: {sorted(args)}"
                )
            engine = self.module("engine").EngineHandle.for_database(
                database, systems_path=systems_path
            )
            values = engine.evaluate_ops_json(core.ops_json_from_ops([op]), **runtime)
            if len(values) != 1:
                raise ValueError("AIC returned an unexpected operation count")
            _, ms, _, source = values[0]
        if not math.isfinite(ms) or ms <= 0:
            raise ValueError(f"AIC returned invalid latency: {ms}")
        return dict(
            latency_ms=ms,
            sdk_source=source,
            api=self.api,
            sdk_version=self.version,
            data_version=database.version,
            operation=kind,
            query=shape,
        )


class _CoreSession:
    """The core SDK moved static execution from InferenceSession to backend."""

    def __init__(self, model, database, backend):
        self._model, self._database, self._backend = model, database, backend

    def run_static(self, runtime_config, mode):
        return self._backend.run_static(
            self._model, self._database, runtime_config, mode
        )
