"""Control messages that invalidate a composite inference weight snapshot.

Count both dispatch and reply (including failed/partial updates). This also
detects changes without a new model path or caller-provided weight_version.
The tokenizer manager's existing per-inference reader lock still owns weight
synchronization; a composite endpoint must NOT nest another reader around it.
"""

from sglang.srt.managers.io_struct import (
    BeginWeightUpdateReqInput,
    BeginWeightUpdateReqOutput,
    EndWeightUpdateReqInput,
    EndWeightUpdateReqOutput,
    ReleaseMemoryOccupationReqInput,
    ReleaseMemoryOccupationReqOutput,
    ResumeMemoryOccupationReqInput,
    ResumeMemoryOccupationReqOutput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightFromDiskReqOutput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromDistributedReqOutput,
    UpdateWeightsFromIPCReqInput,
    UpdateWeightsFromIPCReqOutput,
    UpdateWeightsFromTensorReqInput,
    UpdateWeightsFromTensorReqOutput,
    UpdateWeightVersionReqInput,
    UpdateWeightVersionReqOutput,
)

WEIGHT_UPDATE_MESSAGES = (
    BeginWeightUpdateReqInput,
    BeginWeightUpdateReqOutput,
    EndWeightUpdateReqInput,
    EndWeightUpdateReqOutput,
    ReleaseMemoryOccupationReqInput,
    ReleaseMemoryOccupationReqOutput,
    ResumeMemoryOccupationReqInput,
    ResumeMemoryOccupationReqOutput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightFromDiskReqOutput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromDistributedReqOutput,
    UpdateWeightsFromIPCReqInput,
    UpdateWeightsFromIPCReqOutput,
    UpdateWeightsFromTensorReqInput,
    UpdateWeightsFromTensorReqOutput,
    UpdateWeightVersionReqInput,
    UpdateWeightVersionReqOutput,
)
