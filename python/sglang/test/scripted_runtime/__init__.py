from sglang.srt.environ import envs

# Spawned server and scheduler processes import the scripted-runtime package
# before constructing their cache. Keep inspection in test processes while
# preserving the selected backend and its compatibility fallback.
if envs.SGLANG_TEST_SCRIPTED_RUNTIME.get():
    from sglang.test.scripted_runtime.tree_core_inspection import (
        install_tree_core_inspectors,
    )

    install_tree_core_inspectors()
