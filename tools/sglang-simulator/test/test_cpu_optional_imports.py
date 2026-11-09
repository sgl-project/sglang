"""Fresh-process CPU boundary controls without accelerator dependencies.

These controls exercise import and delegation contracts. Real SGLang/torch
quantization validation is covered separately by test_cpu_bootstrap.py.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

SIMULATOR = Path(__file__).resolve().parents[1]


def run_isolated(code, source, *arguments):
    env = os.environ.copy()
    env.pop("SGLANG_SIMULATOR_BOOTSTRAP", None)
    env.pop("SGLANG_USE_CPU_ENGINE", None)
    result = subprocess.run(
        [sys.executable, "-S", "-c", code, str(source), *arguments],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "bootstrap,cpu", [(False, False), (True, False), (False, True), (True, True)]
)
def test_fresh_process_cpu_flags_and_repeat_activation(bootstrap, cpu):
    run_isolated(
        """
import os, runpy, sys
from types import ModuleType, SimpleNamespace

def available():
    return True

def capability(*args):
    return (9, 0)

torch = ModuleType('torch')
torch.cuda = SimpleNamespace(is_available=available, get_device_capability=capability)
torch.version = SimpleNamespace(hip='original-rocm')
sys.modules['torch'] = torch
modules = {}
for name in ('sglang', 'sglang.srt', 'sglang.srt.utils', 'sglang.srt.utils.common',
             'sglang_simulator', 'sglang_simulator.simulation',
             'sglang_simulator.simulation.sglang',
             'sglang_simulator.simulation.sglang.sgl_kernel_hook',
             'sglang_simulator.simulation.sglang.dspark_args'):
    module = ModuleType(name)
    module.__path__ = []
    modules[name] = sys.modules[name] = module
    if '.' in name:
        parent, leaf = name.rsplit('.', 1)
        setattr(modules[parent], leaf, module)
utils = modules['sglang.srt.utils']
common = modules['sglang.srt.utils.common']
utils.get_device_capability = common.get_device_capability = capability
calls = []
modules['sglang_simulator.simulation.sglang.sgl_kernel_hook'].install_load_utils_stub = lambda: calls.append('loader')
# The native adapter has its own activation controls. Keep this generic CPU
# control independent of whether that optional adapter is present in the tree.
modules['sglang_simulator.simulation.sglang.dspark_args'].install_cpu_dspark_args = lambda: None
if sys.argv[2] == '1':
    os.environ['SGLANG_SIMULATOR_BOOTSTRAP'] = '1'
if sys.argv[3] == '1':
    os.environ['SGLANG_USE_CPU_ENGINE'] = '1'
namespace = runpy.run_path(sys.argv[1])
namespace['apply_cpu_simulation_compat']()
enabled = sys.argv[2:] == ['1', '1']
if enabled:
    assert torch.cuda.is_available() is False
    assert torch.cuda.get_device_capability() == (10, 0)
    assert torch.version.hip is None
    assert utils.get_device_capability() == common.get_device_capability() == (10, 0)
    assert calls == ['loader', 'loader']
else:
    assert torch.cuda.is_available is available
    assert torch.cuda.get_device_capability is capability
    assert torch.version.hip == 'original-rocm'
    assert utils.get_device_capability is common.get_device_capability is capability
    assert calls == []
assert 'sglang.srt.layers.quantization' not in sys.modules
""",
        SIMULATOR / "src/usercustomize.py",
        str(int(bootstrap)),
        str(int(cpu)),
    )
