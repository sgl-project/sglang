"""
Parser factory for loading different parser implementations.
"""

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Optional

# 获取当前文件所在目录
_CURRENT_DIR = Path(__file__).parent
_LEGACY_PARSERS_DIR = _CURRENT_DIR / "_legacy_parsers"


def create_parser(parser_source: str = "default"):
    """
    Create a parser instance based on the source specification.

    Args:
        parser_source: Parser source specification. Options:
            - "default": Use the default parser (sglang_qwen3_coder_detector.Qwen3CoderDetector)
            - "legacy_v3": Use sglang_qwen3_coder_new_detector_v3.Qwen3CoderDetector
            - "legacy_vllm": Use sglang_qwen3_coder_detector_from_vllm_1231_nonstream.Qwen3CoderDetector
            - Or a module path like "module.path:ClassName"

    Returns:
        Parser instance (BaseFormatDetector or compatible)
    """
    if parser_source == "default":
        from sglang_qwen3_coder_detector import Qwen3CoderDetector

        return Qwen3CoderDetector()

    elif parser_source == "legacy_v3":
        # Import from legacy parsers
        legacy_v3_path = _LEGACY_PARSERS_DIR / "sglang_qwen3_coder_new_detector_v3.py"
        if not legacy_v3_path.exists():
            raise ImportError(f"Legacy parser v3 not found at {legacy_v3_path}")

        spec = importlib.util.spec_from_file_location("sglang_qwen3_coder_new_detector_v3", legacy_v3_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        return module.Qwen3CoderDetector()

    elif parser_source == "legacy_vllm":
        # Import from legacy parsers
        legacy_vllm_path = _LEGACY_PARSERS_DIR / "sglang_qwen3_coder_detector_from_vllm_1231_nonstream.py"
        if not legacy_vllm_path.exists():
            raise ImportError(f"Legacy parser vllm not found at {legacy_vllm_path}")

        spec = importlib.util.spec_from_file_location("sglang_qwen3_coder_detector_from_vllm_1231_nonstream", legacy_vllm_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        return module.Qwen3CoderDetector()

    elif ":" in parser_source:
        # Custom module path format: "module.path:ClassName"
        module_path, class_name = parser_source.rsplit(":", 1)

        # Try to import as a file path first
        if Path(module_path).exists():
            spec = importlib.util.spec_from_file_location(Path(module_path).stem, module_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
        else:
            # Try to import as a regular module
            module = importlib.import_module(module_path)

        cls = getattr(module, class_name)
        return cls()

    else:
        raise ValueError(f"Unknown parser_source: {parser_source}. " f"Supported options: 'default', 'legacy_v3', 'legacy_vllm', or 'module.path:ClassName'")
