# SPDX-License-Identifier: Apache-2.0
"""Device-specific MoE bundles; legacy flat bundles belong to SM103 only."""

import json
import os
from pathlib import Path


def q38_arch_name(capability):
    if tuple(capability) not in ((10, 3), (10, 7)):
        raise ValueError(f'Q38 MoE AOT requires SM103 or SM107, got {capability}')
    return f'sm_{capability[0]}{capability[1]}a'


def bundle_manifest(variable, capability):
    arch = q38_arch_name(capability)
    root = Path(os.environ[variable])
    manifest_path = root / 'manifest.json'
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        targets = manifest.get('architectures')
        if targets is not None:
            target = targets.get(arch, targets.get(arch.replace('_', '')))
            if target is None:
                raise ValueError(f'{variable}: bundle has no {arch} target')
            relative = target if isinstance(target, str) else target.get('path')
            if not relative:
                raise ValueError(f'{variable}: {arch} target needs a relative path')
            root = (root / relative).resolve()
            bundle_root = Path(os.environ[variable]).resolve()
            if root != bundle_root and bundle_root not in root.parents:
                raise ValueError(f'{variable}: {arch} path escapes bundle root: {relative}')
            manifest_path = root / 'manifest.json'
    if (root / arch / 'manifest.json').is_file():
        root = root / arch
        manifest_path = root / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    # Existing qualified SM103 bundles predate the arch field. They remain valid
    # on SM103, but must never be loaded on SM107.
    actual = manifest.get('arch', 'sm_103a')
    if actual != arch:
        raise ValueError(f'{variable}: bundle arch {actual} does not match device {arch}')
    return root, manifest
