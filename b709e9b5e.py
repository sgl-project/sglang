"""Auto-generated reproduce script. Audit each call, then run.

commit:  b709e9b5ef7d38913653048f27960e65617da04b
subject: cp-bcg(support-api,mechanical_provable): Restore the package-level support predicate

Each call is a faithful relocation primitive. Running this reproduces the commit
in a throwaway worktree and diffs it byte-for-byte; PASS means the commit is
exactly these relocations.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mechanical_refactor_reproduction_utils import Repro

r = Repro(base='b709e9b5ef7d38913653048f27960e65617da04b~1', target='b709e9b5ef7d38913653048f27960e65617da04b')
r.repath_import('python/sglang/srt/arg_groups/cuda_graph_hook.py', old_module='sglang.srt.layers.cp.bcg.bcg', new_module='sglang.srt.layers.cp.bcg', name='supports_prefill_cp_bcg')
r.repath_import('python/sglang/srt/arg_groups/model_hook.py', old_module='sglang.srt.layers.cp.bcg.bcg', new_module='sglang.srt.layers.cp.bcg', name='supports_prefill_cp_bcg')
r.move_symbol('supports_prefill_cp_bcg', src='python/sglang/srt/layers/cp/bcg/bcg.py', dst='python/sglang/srt/layers/cp/bcg/__init__.py', into_class=None, from_class=None, dedent=0, drop_self_annotation=False, before=None, after=None, leave_delegate=None, delegate_name=None)
r.remove_imported_name('python/sglang/srt/layers/cp/bcg/bcg.py', module='sglang.srt.arg_groups.overrides', name='attention_backends_of', asname=None)
r.remove_imported_name('python/sglang/srt/layers/cp/bcg/bcg.py', module='sglang.srt.layers.cp.base', name='CPAttentionBackendKind', asname=None)
r.remove_imported_name('python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py', module='sglang.srt.layers.cp.bcg.bcg', name='supports_prefill_cp_bcg', asname=None)
r.add_import('python/sglang/srt/layers/cp/bcg/__init__.py', 'from __future__ import annotations')
r.add_import('python/sglang/srt/layers/cp/bcg/__init__.py', 'from typing import TYPE_CHECKING')
r.add_import('python/sglang/srt/layers/cp/bcg/__init__.py', 'from sglang.srt.arg_groups.overrides import (\n    attention_backends_of,\n    resolved_view,\n    resolving_view,\n)')
r.add_import('python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py', 'from sglang.srt.layers.cp.bcg import supports_prefill_cp_bcg')
r.add_typechecking_import('python/sglang/srt/layers/cp/bcg/__init__.py', 'from sglang.srt.server_args import ServerArgs')
# Preserve the target's blank-line separation after the TYPE_CHECKING block.
def preserve_spacing(root):
    path = root / "python/sglang/srt/layers/cp/bcg/__init__.py"
    text = path.read_text()
    text = text.replace("from sglang.srt.server_args import ServerArgs\n\nPREFILL", "from sglang.srt.server_args import ServerArgs\n\n\nPREFILL")
    path.write_text(text)
r.ops.append(preserve_spacing)
residual = r.run()
sys.exit(1 if residual else 0)
