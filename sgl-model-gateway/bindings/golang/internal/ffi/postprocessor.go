// Package ffi provides Go bindings for SGLang's Rust FFI (Foreign Function Interface).
package ffi

/*
#cgo LDFLAGS: -lsgl_model_gateway_go -ldl
#include <stdlib.h>
#include <stdint.h>

// Error codes (must match client.go)
typedef enum {
    SGL_ERROR_SUCCESS = 0,
    SGL_ERROR_INVALID_ARGUMENT = 1,
    SGL_ERROR_TOKENIZATION_ERROR = 2,
    SGL_ERROR_PARSING_ERROR = 3,
    SGL_ERROR_MEMORY_ERROR = 4,
    SGL_ERROR_UNKNOWN = 99
} SglErrorCode;

// Opaque handle (must match grpc_converter.go)
typedef void* GrpcResponseConverterHandle;

// Postprocessor functions
SglErrorCode sgl_postprocess_stream_chunk(
    GrpcResponseConverterHandle* converter_handle,
    const char* proto_chunk_json,
    char** openai_json_out,
    int* is_done_out,
    char** error_out
);

SglErrorCode sgl_postprocess_stream_chunks_batch(
    GrpcResponseConverterHandle* converter_handle,
    const char* proto_chunks_json_array,
    int max_chunks,
    char** openai_chunks_json_array_out,
    int* chunks_count_out,
    char** error_out
);

// Memory management
void sgl_free_string(char* s);
*/
import "C"

import (
	"fmt"
	"unsafe"
)

// GrpcResponseConverterHandle wraps the Rust gRPC response converter FFI handle
type GrpcResponseConverterHandle struct {
	handle *C.GrpcResponseConverterHandle
}

// PostprocessStreamChunk postprocesses a gRPC stream chunk to OpenAI format
//
// This function:
// 1. Parses the proto chunk from JSON
// 2. Converts it to OpenAI format using the converter handle
// 3. Returns the OpenAI format JSON
//
// Returns the OpenAI format JSON, is_done flag, and any error.
func PostprocessStreamChunk(converterHandle *GrpcResponseConverterHandle, protoChunkJSON string) (openaiJSON string, isDone bool, err error) {
	if converterHandle == nil || converterHandle.handle == nil {
		return "", false, fmt.Errorf("invalid converter handle")
	}

	protoChunkJSONC := C.CString(protoChunkJSON)
	defer C.free(unsafe.Pointer(protoChunkJSONC))

	var openaiJSONOut *C.char
	var isDoneOut C.int
	var errorOut *C.char

	errorCode := C.sgl_postprocess_stream_chunk(
		converterHandle.handle,
		protoChunkJSONC,
		&openaiJSONOut,
		&isDoneOut,
		&errorOut,
	)

	if errorCode != C.SGL_ERROR_SUCCESS {
		errorMsg := ""
		if errorOut != nil {
			errorMsg = C.GoString(errorOut)
			C.sgl_free_string(errorOut)
		}
		return "", false, fmt.Errorf("postprocessing failed: %s", errorMsg)
	}

	openaiJSON = C.GoString(openaiJSONOut)
	isDone = isDoneOut != 0

	// Free the C string allocated by Rust
	if openaiJSONOut != nil {
		C.sgl_free_string(openaiJSONOut)
	}

	return openaiJSON, isDone, nil
}

// PostprocessStreamChunksBatch postprocesses multiple gRPC stream chunks in batch
//
// This function processes multiple chunks in a single FFI call, significantly reducing
// FFI overhead in streaming scenarios.
//
// Arguments:
// - converterHandle: Converter handle
// - protoChunksJSONArray: JSON array string of proto chunks
// - maxChunks: Maximum number of chunks to process (for safety, typically 10-20)
//
// Returns:
// - openaiChunksJSONArray: JSON array of OpenAI format chunks
// - chunksCount: Number of processed chunks
// - error: Any error that occurred
func PostprocessStreamChunksBatch(converterHandle *GrpcResponseConverterHandle, protoChunksJSONArray string, maxChunks int) (openaiChunksJSONArray string, chunksCount int, err error) {
	if converterHandle == nil || converterHandle.handle == nil {
		return "", 0, fmt.Errorf("invalid converter handle")
	}

	protoChunksJSONArrayC := C.CString(protoChunksJSONArray)
	defer C.free(unsafe.Pointer(protoChunksJSONArrayC))

	var openaiChunksJSONArrayOut *C.char
	var chunksCountOut C.int
	var errorOut *C.char

	errorCode := C.sgl_postprocess_stream_chunks_batch(
		converterHandle.handle,
		protoChunksJSONArrayC,
		C.int(maxChunks),
		&openaiChunksJSONArrayOut,
		&chunksCountOut,
		&errorOut,
	)

	if errorCode != C.SGL_ERROR_SUCCESS {
		errorMsg := ""
		if errorOut != nil {
			errorMsg = C.GoString(errorOut)
			C.sgl_free_string(errorOut)
		}
		return "", 0, fmt.Errorf("batch postprocessing failed: %s", errorMsg)
	}

	openaiChunksJSONArray = C.GoString(openaiChunksJSONArrayOut)
	chunksCount = int(chunksCountOut)

	// Free the C string allocated by Rust
	if openaiChunksJSONArrayOut != nil {
		C.sgl_free_string(openaiChunksJSONArrayOut)
	}

	return openaiChunksJSONArray, chunksCount, nil
}
