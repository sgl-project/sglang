import logging

try:
    from kernels.layer.func import FuncRepository as _FR
    from kernels.layer.layer import LayerRepository as _LR

    _lr_orig = _LR.__init__

    def _lr_patched(self, repo_id, *, layer_name, revision=None, version=None, **kw):
        if revision is None and version is None:
            revision = "main"
        _lr_orig(
            self,
            repo_id,
            layer_name=layer_name,
            revision=revision,
            version=version,
            **kw,
        )

    _LR.__init__ = _lr_patched

    _fr_orig = _FR.__init__

    def _fr_patched(self, repo_id, *, func_name, revision=None, version=None, **kw):
        if revision is None and version is None:
            revision = "main"
        _fr_orig(
            self, repo_id, func_name=func_name, revision=revision, version=version, **kw
        )

    _FR.__init__ = _fr_patched
except ImportError:
    pass
except Exception as e:
    logging.getLogger(__name__).warning(f"patch_kernels failed: {e}")
