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

// Opaque handles
typedef void* TokenizerHandle;
typedef void* GrpcResponseConverterHandle;

// Converter functions
GrpcResponseConverterHandle* sgl_grpc_response_converter_create(
    TokenizerHandle* tokenizer_handle,
    const char* model,
    const char* request_id,
    const char* tools_json,
    const char* tool_choice_json,
    const char* stop,
    const char* stop_token_ids,
    int skip_special_tokens,
    int initial_prompt_tokens,
    char** error_out
);

void sgl_grpc_response_converter_free(GrpcResponseConverterHandle* handle);

// Tokenizer functions
TokenizerHandle* sgl_tokenizer_create_from_file(const char* tokenizer_path, char** error_out);
void sgl_tokenizer_free(TokenizerHandle* handle);

// Memory management
void sgl_free_string(char* s);
*/
import "C"

import (
	"fmt"
	"unsafe"
)

// CreateGrpcResponseConverter creates a gRPC response converter handle
// This function creates a new tokenizer handle each time (for backward compatibility)
// For better performance, use CreateGrpcResponseConverterWithTokenizer with a cached tokenizer
func CreateGrpcResponseConverter(
	tokenizerPath string,
	model string,
	requestID string,
	toolsJSON string,
	toolChoiceJSON string,
	stopJSON string,
	stopTokenIDs []uint32,
	skipSpecialTokens bool,
	initialPromptTokens int32,
) (*GrpcResponseConverterHandle, error) {
	// Create tokenizer handle
	tokenizerHandle, err := createTokenizerHandle(tokenizerPath)
	if err != nil {
		return nil, fmt.Errorf("failed to create tokenizer handle: %w", err)
	}
	defer C.sgl_tokenizer_free(tokenizerHandle)

	return createGrpcResponseConverterWithTokenizerHandle(
		tokenizerHandle,
		model,
		requestID,
		toolsJSON,
		toolChoiceJSON,
		stopJSON,
		stopTokenIDs,
		skipSpecialTokens,
		initialPromptTokens,
	)
}

// CreateGrpcResponseConverterWithTokenizer creates a gRPC response converter handle using a cached tokenizer
// This is more efficient as it reuses the tokenizer instead of creating a new one each time
func CreateGrpcResponseConverterWithTokenizer(
	tokenizerHandle *TokenizerHandle,
	model string,
	requestID string,
	toolsJSON string,
	toolChoiceJSON string,
	stopJSON string,
	stopTokenIDs []uint32,
	skipSpecialTokens bool,
	initialPromptTokens int32,
) (*GrpcResponseConverterHandle, error) {
	if tokenizerHandle == nil || tokenizerHandle.handle == nil {
		return nil, fmt.Errorf("invalid tokenizer handle")
	}

	return createGrpcResponseConverterWithTokenizerHandle(
		tokenizerHandle.handle,
		model,
		requestID,
		toolsJSON,
		toolChoiceJSON,
		stopJSON,
		stopTokenIDs,
		skipSpecialTokens,
		initialPromptTokens,
	)
}

// createGrpcResponseConverterWithTokenizerHandle is the internal implementation
func createGrpcResponseConverterWithTokenizerHandle(
	tokenizerHandle *C.TokenizerHandle,
	model string,
	requestID string,
	toolsJSON string,
	toolChoiceJSON string,
	stopJSON string,
	stopTokenIDs []uint32,
	skipSpecialTokens bool,
	initialPromptTokens int32,
) (*GrpcResponseConverterHandle, error) {

	// Convert strings to C strings
	modelC := C.CString(model)
	defer C.free(unsafe.Pointer(modelC))

	requestIDC := C.CString(requestID)
	defer C.free(unsafe.Pointer(requestIDC))

	var toolsJSONC *C.char
	if toolsJSON != "" {
		toolsJSONC = C.CString(toolsJSON)
		defer C.free(unsafe.Pointer(toolsJSONC))
	}

	var toolChoiceJSONC *C.char
	if toolChoiceJSON != "" {
		toolChoiceJSONC = C.CString(toolChoiceJSON)
		defer C.free(unsafe.Pointer(toolChoiceJSONC))
	}

	var stopJSONC *C.char
	if stopJSON != "" {
		stopJSONC = C.CString(stopJSON)
		defer C.free(unsafe.Pointer(stopJSONC))
	}

	// Convert stop_token_ids to JSON string
	stopTokenIDsJSON := ""
	if len(stopTokenIDs) > 0 {
		stopTokenIDsJSON = fmt.Sprintf("[%d", stopTokenIDs[0])
		for i := 1; i < len(stopTokenIDs); i++ {
			stopTokenIDsJSON += fmt.Sprintf(",%d", stopTokenIDs[i])
		}
		stopTokenIDsJSON += "]"
	}

	var stopTokenIDsJSONC *C.char
	if stopTokenIDsJSON != "" {
		stopTokenIDsJSONC = C.CString(stopTokenIDsJSON)
		defer C.free(unsafe.Pointer(stopTokenIDsJSONC))
	}

	var errorOut *C.char
	skipSpecialTokensC := C.int(0)
	if skipSpecialTokens {
		skipSpecialTokensC = C.int(1)
	}

	initialPromptTokensC := C.int(initialPromptTokens)

	converterHandle := C.sgl_grpc_response_converter_create(
		tokenizerHandle,
		modelC,
		requestIDC,
		toolsJSONC,
		toolChoiceJSONC,
		stopJSONC,
		stopTokenIDsJSONC,
		skipSpecialTokensC,
		initialPromptTokensC,
		&errorOut,
	)

	if converterHandle == nil {
		errorMsg := ""
		if errorOut != nil {
			errorMsg = C.GoString(errorOut)
			C.sgl_free_string(errorOut)
		}
		if errorMsg == "" {
			errorMsg = "failed to create converter handle"
		}
		return nil, fmt.Errorf("%s", errorMsg)
	}

	return &GrpcResponseConverterHandle{
		handle: converterHandle,
	}, nil
}

// FreeGrpcResponseConverter frees a gRPC response converter handle
func FreeGrpcResponseConverter(handle *GrpcResponseConverterHandle) {
	if handle != nil && handle.handle != nil {
		C.sgl_grpc_response_converter_free(handle.handle)
		handle.handle = nil
	}
}

// TokenizerHandle wraps the Rust tokenizer FFI handle
type TokenizerHandle struct {
	handle *C.TokenizerHandle
}

// CreateTokenizerHandle creates a tokenizer handle (exported for caching)
func CreateTokenizerHandle(tokenizerPath string) (*TokenizerHandle, error) {
	tokenizerPathC := C.CString(tokenizerPath)
	defer C.free(unsafe.Pointer(tokenizerPathC))

	var errorOut *C.char
	tokenizerHandle := C.sgl_tokenizer_create_from_file(tokenizerPathC, &errorOut)

	if tokenizerHandle == nil {
		errorMsg := ""
		if errorOut != nil {
			errorMsg = C.GoString(errorOut)
			C.sgl_free_string(errorOut)
		}
		if errorMsg == "" {
			errorMsg = "failed to create tokenizer handle"
		}
		return nil, fmt.Errorf("%s", errorMsg)
	}

	return &TokenizerHandle{
		handle: tokenizerHandle,
	}, nil
}

// FreeTokenizerHandle frees a tokenizer handle
func FreeTokenizerHandle(handle *TokenizerHandle) {
	if handle != nil && handle.handle != nil {
		C.sgl_tokenizer_free(handle.handle)
		handle.handle = nil
	}
}

// createTokenizerHandle creates a tokenizer handle (helper function, internal use)
func createTokenizerHandle(tokenizerPath string) (*C.TokenizerHandle, error) {
	tokenizerPathC := C.CString(tokenizerPath)
	defer C.free(unsafe.Pointer(tokenizerPathC))

	var errorOut *C.char
	tokenizerHandle := C.sgl_tokenizer_create_from_file(tokenizerPathC, &errorOut)

	if tokenizerHandle == nil {
		errorMsg := ""
		if errorOut != nil {
			errorMsg = C.GoString(errorOut)
			C.sgl_free_string(errorOut)
		}
		if errorMsg == "" {
			errorMsg = "failed to create tokenizer handle"
		}
		return nil, fmt.Errorf("%s", errorMsg)
	}

	return tokenizerHandle, nil
}
