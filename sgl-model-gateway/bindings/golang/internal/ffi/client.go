// Package ffi provides Go bindings for SGLang's Rust FFI (Foreign Function Interface).
//
// This package wraps the Rust FFI layer of SGLang, providing low-level access to:
// - Client creation and connection management
// - Chat completion streaming
// - Stream reading and response conversion
// - Memory management for C strings
//
// Internal use only: This package is intended for internal use by the sglang package.
// End users should use the public sglang package instead.
package ffi

/*
#cgo LDFLAGS: -lsgl_model_gateway_go -ldl
#include <stdlib.h>
#include <stdint.h>

// Error codes
typedef enum {
    SGL_ERROR_SUCCESS = 0,
    SGL_ERROR_INVALID_ARGUMENT = 1,
    SGL_ERROR_TOKENIZATION_ERROR = 2,
    SGL_ERROR_PARSING_ERROR = 3,
    SGL_ERROR_MEMORY_ERROR = 4,
    SGL_ERROR_UNKNOWN = 99
} SglErrorCode;

// Opaque handles
typedef void* SglangClientHandle;
typedef void* SglangStreamHandle;

// Client SDK functions
SglangClientHandle* sgl_client_create(const char* endpoint, const char* tokenizer_path, char** error_out);
void sgl_client_free(SglangClientHandle* handle);
SglErrorCode sgl_client_chat_completion_stream(SglangClientHandle* client_handle, const char* request_json, SglangStreamHandle** stream_handle_out, char** error_out);
SglErrorCode sgl_stream_read_next(SglangStreamHandle* stream_handle, char** response_json_out, int* is_done_out, char** error_out);
void sgl_stream_free(SglangStreamHandle* handle);
void sgl_free_string(char* s);
*/
import "C"

import (
	"fmt"
	"unsafe"
)

// ErrorCode represents FFI error codes returned by Rust functions.
//
// These codes indicate the result of FFI operations. Use Error() to get a human-readable
// error message.
type ErrorCode int

const (
	// ErrorSuccess indicates the operation completed successfully
	ErrorSuccess ErrorCode = 0
	// ErrorInvalidArgument indicates invalid arguments were passed to the FFI function
	ErrorInvalidArgument ErrorCode = 1
	// ErrorTokenizationError indicates an error during tokenization
	ErrorTokenizationError ErrorCode = 2
	// ErrorParsingError indicates an error parsing the response or request
	ErrorParsingError ErrorCode = 3
	// ErrorMemoryError indicates a memory allocation error
	ErrorMemoryError ErrorCode = 4
	// ErrorUnknown indicates an unclassified error
	ErrorUnknown ErrorCode = 99
)

// Error implements the error interface for ErrorCode.
func (e ErrorCode) Error() string {
	switch e {
	case ErrorSuccess:
		return "success"
	case ErrorInvalidArgument:
		return "invalid argument"
	case ErrorTokenizationError:
		return "tokenization error"
	case ErrorParsingError:
		return "parsing error"
	case ErrorMemoryError:
		return "memory error"
	case ErrorUnknown:
		return "unknown error"
	default:
		return fmt.Sprintf("unknown error code: %d", e)
	}
}

// SglangClientHandle wraps the Rust client SDK FFI handle.
//
// This struct maintains a connection to the SGLang gRPC server and is used
// to create streams and manage the underlying Rust client resources.
type SglangClientHandle struct {
	handle *C.SglangClientHandle
}

// NewClient creates a new SGLang client handle via FFI.
//
// This function initializes the Rust client with the given endpoint and tokenizer path.
//
// Parameters:
// - endpoint: gRPC endpoint URL (e.g., "grpc://localhost:20000")
// - tokenizerPath: Path to tokenizer directory
//
// Returns:
// - *SglangClientHandle: A new client handle
// - error: An error if client creation failed
func NewClient(endpoint, tokenizerPath string) (*SglangClientHandle, error) {
	cEndpoint := C.CString(endpoint)
	defer C.free(unsafe.Pointer(cEndpoint))

	cTokenizerPath := C.CString(tokenizerPath)
	defer C.free(unsafe.Pointer(cTokenizerPath))

	var errorPtr *C.char
	handle := C.sgl_client_create(cEndpoint, cTokenizerPath, &errorPtr)

	if handle == nil {
		errorMsg := ""
		if errorPtr != nil {
			errorMsg = C.GoString(errorPtr)
			C.sgl_free_string(errorPtr)
		}
		if errorMsg == "" {
			errorMsg = "failed to create client"
		}
		return nil, fmt.Errorf("%s", errorMsg)
	}

	return &SglangClientHandle{handle: handle}, nil
}

// Free releases the client handle
func (h *SglangClientHandle) Free() {
	if h.handle != nil {
		C.sgl_client_free(h.handle)
		h.handle = nil
	}
}

// ChatCompletionStream creates a streaming chat completion request
func (h *SglangClientHandle) ChatCompletionStream(requestJSON string) (*SglangStreamHandle, error) {
	if h.handle == nil {
		return nil, fmt.Errorf("client handle is nil")
	}

	cRequestJSON := C.CString(requestJSON)
	defer C.free(unsafe.Pointer(cRequestJSON))

	var streamHandle *C.SglangStreamHandle
	var errorPtr *C.char

	result := C.sgl_client_chat_completion_stream(
		h.handle,
		cRequestJSON,
		&streamHandle,
		&errorPtr,
	)

	if ErrorCode(result) != ErrorSuccess {
		errorMsg := ""
		if errorPtr != nil {
			errorMsg = C.GoString(errorPtr)
			C.sgl_free_string(errorPtr)
		}
		if errorMsg == "" {
			errorMsg = fmt.Sprintf("error code %d", result)
		}
		return nil, fmt.Errorf("%s", errorMsg)
	}

	if streamHandle == nil {
		return nil, fmt.Errorf("stream handle is nil")
	}

	return &SglangStreamHandle{handle: streamHandle}, nil
}

// SglangStreamHandle wraps the Rust stream FFI handle
type SglangStreamHandle struct {
	handle *C.SglangStreamHandle
}

// ReadNext reads the next chunk from the stream
// Returns: (responseJSON, isDone, error)
func (h *SglangStreamHandle) ReadNext() (string, bool, error) {
	if h.handle == nil {
		return "", true, fmt.Errorf("stream handle is nil")
	}

	var responseJSON *C.char
	var isDone C.int
	var errorPtr *C.char

	result := C.sgl_stream_read_next(
		h.handle,
		&responseJSON,
		&isDone,
		&errorPtr,
	)

	if ErrorCode(result) != ErrorSuccess {
		errorMsg := ""
		if errorPtr != nil {
			errorMsg = C.GoString(errorPtr)
			C.sgl_free_string(errorPtr)
		}
		if errorMsg == "" {
			errorMsg = fmt.Sprintf("error code %d", result)
		}
		return "", isDone == 1, fmt.Errorf("%s", errorMsg)
	}

	responseStr := ""
	if responseJSON != nil {
		responseStr = C.GoString(responseJSON)
		C.sgl_free_string(responseJSON)
	}

	return responseStr, isDone == 1, nil
}

// Free releases the stream handle
func (h *SglangStreamHandle) Free() {
	if h.handle != nil {
		C.sgl_stream_free(h.handle)
		h.handle = nil
	}
}
