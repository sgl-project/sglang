#ifndef ACLNN_FUSED_LIGHTNING_INDEXER_MANAGE_H_
#define ACLNN_FUSED_LIGHTNING_INDEXER_MANAGE_H_
#include "aclnn/acl_meta.h"
#include "aclnn/aclnn_base.h"
#ifdef __cplusplus
extern "C" {
#endif
__attribute__((visibility("default")))
aclnnStatus aclnnFusedLightningIndexerManageGetWorkspaceSize(
    const aclTensor *indexWeights, const aclTensor *queryDequantScale,
    const aclTensor *query, const aclTensor *indexKeyDequantScale,
    const aclTensor *indexKeyCache, const aclTensor *indexBlockTable,
    const aclTensor *actualSeqLengthsQuery, const aclTensor *actualSeqLengthsKey,
    const aclTensor *offloadSeqLengthsKey, const aclTensor *numCacheTokens,
    const aclTensor *requestState, const aclTensor *reqPoolEntries,
    const aclTensor *cacheSlotsPool, const aclTensor *topkSrcIds,
    const aclTensor *topkDstSlots, const aclTensor *topkMissCounts,
    const aclTensor *missSrcIds, const aclTensor *missDstSlots,
    const aclTensor *missCounts, const aclTensor *cacheSlotsPoolOut,
    uint64_t *workspaceSize, aclOpExecutor **executor);
__attribute__((visibility("default")))
aclnnStatus aclnnFusedLightningIndexerManage(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    const aclrtStream stream);
#ifdef __cplusplus
}
#endif
#endif
