# SPDX-License-Identifier: Apache-2.0
"""Compatibility imports and NVIDIA settings for megakernel providers."""

from sglang.kernels.plugin_kernels.megakernel.spec import (
    EAGLE_DRAFT_CAPTURE_CHECK as EAGLE_DRAFT_CAPTURE_CHECK,
)
from sglang.kernels.plugin_kernels.megakernel.spec import ENTRIES as ENTRIES
from sglang.kernels.plugin_kernels.megakernel.spec import (
    GRAPH_VARIANT_RUNNERS as GRAPH_VARIANT_RUNNERS,
)
from sglang.kernels.plugin_kernels.megakernel.spec import (
    MEGAKERNEL_API_VERSION as MEGAKERNEL_API_VERSION,
)
from sglang.kernels.plugin_kernels.megakernel.spec import BuildContext as BuildContext
from sglang.kernels.plugin_kernels.megakernel.spec import (
    BuiltMegakernel as BuiltMegakernel,
)
from sglang.kernels.plugin_kernels.megakernel.spec import Feature as Feature
from sglang.kernels.plugin_kernels.megakernel.spec import (
    GrammarMaskView as GrammarMaskView,
)
from sglang.kernels.plugin_kernels.megakernel.spec import HooksFactory as HooksFactory
from sglang.kernels.plugin_kernels.megakernel.spec import LocalKeysFn as LocalKeysFn
from sglang.kernels.plugin_kernels.megakernel.spec import (
    MegakernelSpec as MegakernelSpec,
)
from sglang.kernels.plugin_kernels.megakernel.spec import Numerics as Numerics
from sglang.kernels.plugin_kernels.megakernel.spec import NumericsKind as NumericsKind
from sglang.kernels.plugin_kernels.megakernel.spec import PostVerifyIO as PostVerifyIO
from sglang.kernels.plugin_kernels.megakernel.spec import (
    PostVerifyOutput as PostVerifyOutput,
)
from sglang.kernels.plugin_kernels.megakernel.spec import (
    PostVerifyStrategy as PostVerifyStrategy,
)
from sglang.kernels.plugin_kernels.megakernel.spec import (
    RecurrentCommitIO as RecurrentCommitIO,
)
from sglang.kernels.plugin_kernels.megakernel.spec import Requirements as Requirements
from sglang.kernels.plugin_kernels.megakernel.spec import (
    ShardedLogitsView as ShardedLogitsView,
)
from sglang.kernels.plugin_kernels.megakernel.spec import (
    SpeculativeRuntimeHooks as SpeculativeRuntimeHooks,
)
from sglang.kernels.plugin_kernels.megakernel.spec import Step as Step
from sglang.kernels.plugin_kernels.megakernel.spec import (
    SupportsDecodeTensorEntry as SupportsDecodeTensorEntry,
)
from sglang.kernels.plugin_kernels.megakernel.spec import (
    SupportsDFlashRuntime as SupportsDFlashRuntime,
)
from sglang.kernels.plugin_kernels.megakernel.spec import (
    SupportsPrefillBody as SupportsPrefillBody,
)
from sglang.kernels.plugin_kernels.megakernel.spec import (
    SupportsRecurrentCommit as SupportsRecurrentCommit,
)
from sglang.kernels.plugin_kernels.megakernel.spec import (
    SupportsStateSlotReset as SupportsStateSlotReset,
)
from sglang.kernels.plugin_kernels.megakernel.spec import (
    VerifyLogitsView as VerifyLogitsView,
)
from sglang.kernels.plugin_kernels.megakernel.spec import (
    VerifyPenaltiesView as VerifyPenaltiesView,
)
from sglang.kernels.plugin_kernels.megakernel.spec import VerifyRunner as VerifyRunner
from sglang.kernels.plugin_kernels.megakernel.spec import (
    VerifySamplingView as VerifySamplingView,
)
from sglang.kernels.plugin_kernels.megakernel.spec import (
    VerifyTreeView as VerifyTreeView,
)
from sglang.kernels.plugin_kernels.megakernel.spec import (
    VocabShardView as VocabShardView,
)
from sglang.kernels.plugin_kernels.megakernel.spec import (
    WatermarkStateView as WatermarkStateView,
)
from sglang.kernels.plugin_kernels.megakernel.spec import find_spec as find_spec
from sglang.kernels.plugin_kernels.megakernel.spec import register as register
from sglang.kernels.plugin_kernels.megakernel.spec import (
    registered_architectures as registered_architectures,
)
from sglang.kernels.plugin_kernels.megakernel.spec import (
    spec_registered as spec_registered,
)

# the sglang.srt.plugins entry point that registers the specs
ENTRY_POINT = "nvidia_megakernel"
FLAG = "--enable-nvidia-megakernel"
# The variables of Requirements.env that the flag sets to the spec's value when they
# are unset, at argument resolution: SGLang reads each only when a model is built, in
# the model processes started after it. A value the user set is kept and checked.
FLAG_SETS_ENV = frozenset({"SGLANG_ENABLE_EMBED_REPLICATION"})
