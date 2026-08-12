# Temporary FlashInfer MNNVL CuTe DSL provider

This package mirrors only the `flashinfer.comm` surface used by the SGLang
integration. It is removable once the serving image contains a FlashInfer
release with the MNNVL CuTe DSL all-reduce-fusion backend.

`comm/mnnvl_cutedsl_ar.py` is copied from the in-development FlashInfer branch;
only imports of existing FlashInfer infrastructure are redirected to the
installed package. `comm/mnnvl_cutedsl/` is copied unchanged so later kernel
refreshes remain mechanical directory syncs. The current snapshot comes from
FlashInfer commit `b23d193d92d77227ecdf575eb651e8a69e78c720` plus its local
in-progress kernel changes.

Model code must import `sglang.srt.layers.flashinfer_provider`, never this
package directly.
