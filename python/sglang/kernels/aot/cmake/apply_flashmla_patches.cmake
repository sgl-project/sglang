# Combined final delta of issue #6's six FlashMLA patches on 3620b63:
# https://github.com/radixark/sglang-k31/issues/6#issuecomment-6147994885
# Replace with an immutable upstream pin once available.
get_filename_component(source_parent "${FLASHMLA_SOURCE_DIR}" DIRECTORY)
execute_process(
    # FetchContent may be inside SGLang's worktree: prevent git from discovering
    # that parent and silently skipping paths outside its current subdirectory.
    COMMAND "${CMAKE_COMMAND}" -E env
        "GIT_CEILING_DIRECTORIES=${source_parent}"
        "${GIT_EXECUTABLE}" apply "${FLASHMLA_PATCH_DIR}/kimi-native-heads.patch"
    WORKING_DIRECTORY "${FLASHMLA_SOURCE_DIR}"
    RESULT_VARIABLE result
    ERROR_VARIABLE detail
)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "Cannot apply FlashMLA integration patch: ${detail}")
endif()
if(NOT EXISTS "${FLASHMLA_SOURCE_DIR}/csrc/api/sparse_decode_sched.h")
    message(FATAL_ERROR "FlashMLA patch was skipped: schedule header is missing")
endif()
