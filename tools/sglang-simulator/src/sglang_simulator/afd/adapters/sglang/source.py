"""Check the interfaces used by the adapter in a trusted native checkout."""

import ast
from pathlib import Path


def verify_source(source_root, *, mode=None, routing="f-side"):
    if routing not in ("f-side", "direct"):
        raise ValueError(f"AFD_ROUTING_UNSUPPORTED: {routing}")
    if mode not in (None, "afd"):
        raise ValueError(f"AFD_ADAPTER_MODE_UNSUPPORTED: {mode}")
    required = {
        "batch_overlap/operations.py": [
            "execute_operations",
            "execute_overlapped_operations",
            "YieldOperation",
            "ExecutionOperation",
        ],
        "afd/layer_executor.py": ["execute_layer_operations"],
        "afd/pipeline.py": ["AFDAttentionPipeline._run_layers"],
    }
    if routing == "direct":
        required["afd/pipeline.py"].append("AFDAttentionPipeline.__init__")
        required["afd/expert_exchange.py"] = [
            "combine_partials",
            "AFDDirectExchange.__init__",
            "AFDDirectExchange.prepare_step",
            "AFDDirectExchange.dispatch",
            "AFDDirectExchange._exchange",
            "AFDDirectExchange.run_ffn_step",
            "AFDDirectExchange.run_layer",
        ]
        required.update(
            {
                "afd/config.py": ["AFDConfig"],
                "afd/contracts.py": ["AFDError", "AFDRole"],
                "afd/connector.py": ["AFDConnector"],
                "afd/ffn_backend.py": [
                    "direct_step_reason",
                    "select_direct_step",
                    "select_m2n_step",
                ],
                "afd/model_adapters/base.py": [
                    "AFDStage",
                    "AFDDecoderAdapter.__init__",
                    "AFDDecoderAdapter.split_step",
                    "AFDDecoderAdapter.local_compute",
                    "AFDDecoderAdapter.finish_layer",
                    "AFDDecoderAdapter.join_step",
                ],
                "afd/model_adapters/qwen3_moe.py": ["Qwen3AFDAdapter"],
                "model_executor/forward_batch_view.py": ["slice_batch_field"],
                "batch_overlap/two_batch_overlap.py": [
                    "TboForwardBatchPreparer._get_num_token_non_padded_cpu",
                    "TboForwardBatchPreparer._split_num_token_non_padded",
                ],
            }
        )
    else:
        required["afd/pipeline.py"].append("AFDFFNPipeline._run_layers")
        required["afd/model_adapters/base.py"] = [
            "AFDDecoderAdapter.local_compute",
            "AFDDecoderAdapter._group_ffn_output",
            "AFDDecoderAdapter._compute_ffn_output",
        ]
    root = Path(source_root).resolve()
    files = []
    for relative, symbols in required.items():
        name = "python/sglang/srt/" + relative
        path = root / name
        if not path.is_file() or not path.resolve().is_relative_to(root):
            raise ValueError(f"AFD_SOURCE_MISSING: {name}")
        try:
            tree = ast.parse(path.read_text(), filename=str(path))
        except SyntaxError as exc:
            raise ValueError(f"AFD_SOURCE_API_MISMATCH: {name}: {exc.msg}") from exc
        for symbol in symbols:
            node = tree
            for part in symbol.split("."):
                node = next(
                    (
                        n
                        for n in getattr(node, "body", [])
                        if isinstance(n, (ast.ClassDef, ast.FunctionDef))
                        and n.name == part
                    ),
                    None,
                )
                if node is None:
                    raise ValueError(f"AFD_SOURCE_API_MISMATCH: {name}: {symbol}")
        files.append(name)
    return files
