"""
Patch to add shape logging to SGLang's TP worker.

This patch adds optional shape logging to tp_worker.py

Apply with:
    cd /sgl-workspace/sglang
    python examples/profiler/apply_shape_logging_patch.py
    
Then run:
    python -m sglang.launch_server \\
        --model-path Qwen/Qwen2.5-14B-Instruct \\
        --tp 8 \\
        --enable-shape-profiling \\
        --shape-profile-rank 0 \\
        --shape-profile-output gpu0_shapes.jsonl
"""

import os
import sys


PATCH_CONTENT = '''
# Add after imports (around line 52)
# Shape profiling support
import os as _os
_ENABLE_SHAPE_PROFILING = _os.environ.get("SGLANG_PROFILE_SHAPES", "0") == "1"
_SHAPE_PROFILE_RANK = int(_os.environ.get("SGLANG_PROFILE_SHAPES_RANK", "0"))
_SHAPE_PROFILE_FILE = _os.environ.get("SGLANG_PROFILE_SHAPES_FILE", "shapes.jsonl")

if _ENABLE_SHAPE_PROFILING:
    import sys as _sys
    _profiler_path = _os.path.join(_os.path.dirname(__file__), "../../../examples/profiler")
    if _os.path.exists(_profiler_path):
        _sys.path.insert(0, _profiler_path)
        try:
            from torch_shape_logger_rank import CompactRankAwareShapeLogger
            print(f"[Shape Profiling] Logger imported, will profile rank {_SHAPE_PROFILE_RANK}")
        except ImportError as e:
            print(f"[Shape Profiling] Failed to import logger: {e}")
            _ENABLE_SHAPE_PROFILING = False

# Add to TpModelWorker.__init__ (after line 280: self.device = self.model_runner.device)
        # Initialize shape logger if enabled
        self._shape_logger = None
        self._shape_logger_active = False
        if _ENABLE_SHAPE_PROFILING and self.tp_rank == _SHAPE_PROFILE_RANK:
            try:
                self._shape_logger = CompactRankAwareShapeLogger(
                    output_file=_SHAPE_PROFILE_FILE,
                    verbose=False,
                    only_rank=_SHAPE_PROFILE_RANK,
                )
                print(f"[Rank {self.tp_rank}] Shape logger initialized: {_SHAPE_PROFILE_FILE}")
            except Exception as e:
                print(f"[Rank {self.tp_rank}] Failed to create shape logger: {e}")

# Add to forward_batch_generation (at start of function, after line 360)
        # Activate shape logger if not already active
        if self._shape_logger and not self._shape_logger_active:
            try:
                self._shape_logger.__enter__()
                self._shape_logger_active = True
                print(f"[Rank {self.tp_rank}] Shape logging activated")
            except Exception as e:
                print(f"[Rank {self.tp_rank}] Failed to activate shape logger: {e}")
'''


def main():
    print("=" * 80)
    print("SGLang Shape Logging Patch Instructions")
    print("=" * 80)
    print()
    print("This patch adds shape logging to tp_worker.py")
    print()
    print("=" * 80)
    print("Manual Steps:")
    print("=" * 80)
    print()
    print("1. Backup the original file:")
    print("   cp python/sglang/srt/managers/tp_worker.py \\")
    print("      python/sglang/srt/managers/tp_worker.py.bak")
    print()
    print("2. Edit python/sglang/srt/managers/tp_worker.py:")
    print()
    print("   a) After the imports (around line 52), add:")
    print('''
   # Shape profiling support
   import os as _os
   _ENABLE_SHAPE_PROFILING = _os.environ.get("SGLANG_PROFILE_SHAPES", "0") == "1"
   _SHAPE_PROFILE_RANK = int(_os.environ.get("SGLANG_PROFILE_SHAPES_RANK", "0"))
   _SHAPE_PROFILE_FILE = _os.environ.get("SGLANG_PROFILE_SHAPES_FILE", "shapes.jsonl")

   if _ENABLE_SHAPE_PROFILING:
       import sys as _sys
       _profiler_path = _os.path.join(_os.path.dirname(__file__), "../../../examples/profiler")
       if _os.path.exists(_profiler_path):
           _sys.path.insert(0, _profiler_path)
           try:
               from torch_shape_logger_rank import CompactRankAwareShapeLogger
               print(f"[Shape Profiling] Enabled for rank {_SHAPE_PROFILE_RANK}")
           except ImportError as e:
               print(f"[Shape Profiling] Failed to import: {e}")
               _ENABLE_SHAPE_PROFILING = False
''')
    print()
    print("   b) In TpModelWorker.__init__, after 'self.device = self.model_runner.device', add:")
    print('''
        # Initialize shape logger
        self._shape_logger = None
        self._shape_logger_active = False
        if _ENABLE_SHAPE_PROFILING and self.tp_rank == _SHAPE_PROFILE_RANK:
            self._shape_logger = CompactRankAwareShapeLogger(
                output_file=_SHAPE_PROFILE_FILE,
                verbose=False,
                only_rank=_SHAPE_PROFILE_RANK,
            )
            print(f"[Rank {self.tp_rank}] Shape logger ready: {_SHAPE_PROFILE_FILE}")
''')
    print()
    print("   c) In forward_batch_generation, at the start, add:")
    print('''
        # Activate shape logger on first forward
        if self._shape_logger and not self._shape_logger_active:
            self._shape_logger.__enter__()
            self._shape_logger_active = True
            print(f"[Rank {self.tp_rank}] Shape logging started")
''')
    print()
    print("=" * 80)
    print("3. Then run with environment variables:")
    print("=" * 80)
    print()
    print("   export SGLANG_PROFILE_SHAPES=1")
    print("   export SGLANG_PROFILE_SHAPES_RANK=0")
    print("   export SGLANG_PROFILE_SHAPES_FILE=gpu0_shapes.jsonl")
    print()
    print("   python -m sglang.launch_server \\")
    print("       --model-path Qwen/Qwen2.5-14B-Instruct \\")
    print("       --tp 8 \\")
    print("       --port 30000")
    print()
    print("=" * 80)
    print()
    
    # Offer to show the file location
    tp_worker_path = "/sgl-workspace/sglang/python/sglang/srt/managers/tp_worker.py"
    print(f"File to edit: {tp_worker_path}")
    print()


if __name__ == "__main__":
    main()
