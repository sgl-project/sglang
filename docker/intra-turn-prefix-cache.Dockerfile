# syntax=docker/dockerfile:1

ARG BASE_IMAGE=lmsysorg/sglang:deepseek-v4-hopper
FROM ${BASE_IMAGE}

COPY python/sglang/srt/managers/schedule_policy.py \
    /sgl-workspace/sglang/python/sglang/srt/managers/schedule_policy.py
COPY python/sglang/srt/managers/schedule_batch.py \
    /sgl-workspace/sglang/python/sglang/srt/managers/schedule_batch.py
COPY python/sglang/srt/managers/intra_turn_prefix_cache.py \
    /sgl-workspace/sglang/python/sglang/srt/managers/intra_turn_prefix_cache.py
COPY python/sglang/srt/managers/scheduler.py \
    /sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py
COPY python/sglang/srt/environ.py \
    /sgl-workspace/sglang/python/sglang/srt/environ.py

ENV SGLANG_ENABLE_CACHE_AGNOSTIC_IN_BATCH_PREFIX_CACHING=true \
    SGLANG_ENABLE_TWO_STAGE_INTRA_TURN_PREFIX_CACHE=true \
    IN_BATCH_PREFIX_CACHING_CACHE_AGNOSTIC_MAX_QUEUE_SCAN=128 \
    SGLANG_INTRA_TURN_PREFIX_CACHE_MIN_SHARED_TOKENS=32
