# SPDX-License-Identifier: Apache-2.0
"""One immutable component bundle per owner generation, not one cache per model.

ModuleDict only adds namespaces. The existing common traversal/mapping/transport
retains storage aliases and exact tensor ties across the whole bundle.
"""

import torch


def build_bundle(components, *, meta=False):
    bundle = torch.nn.ModuleDict()
    for component in components:
        if not component.name or component.name in bundle:
            raise ValueError("Cache bundle requires distinct, nonempty component names")
        bundle[component.name] = (
            component.build_meta() if meta else component.load_ordinary()[0]
        )
    if not bundle:
        raise ValueError("Cache bundle must contain at least one component")
    # Do not recursively change audited submodule modes or invoke custom train().
    bundle.training = False
    return bundle


def validate_manifest_components(manifest, names):
    roots = {name for name, _ in manifest.training if name and "." not in name}
    if not names or len(set(names)) != len(names) or roots != set(names):
        raise ValueError(
            "Weight-cache manifest component set differs from prepared bundle"
        )


def retain_importer(bundle, importer):
    # Components can outlive the pipeline/wrapper. Install every reference before
    # mapping/finalization so partial failures cannot detach the producer guard.
    bundle.__dict__["_weight_cache_importer"] = importer
    for model in bundle.values():
        model.__dict__["_weight_cache_importer"] = importer


def component_storage_bytes(manifest, names):
    """Reachable bytes per component; aliases can make their sum exceed the union."""
    validate_manifest_components(manifest, names)
    sizes = {storage.group: storage.nbytes for storage in manifest.storages}
    return {
        name: sum(
            sizes[group]
            for group in {
                tensor.storage_group
                for tensor in manifest.tensors
                if tensor.name.startswith(name + ".")
            }
        )
        for name in names
    }
