package handlers

import (
	"encoding/json"

	"github.com/valyala/fasthttp"
	"go.uber.org/zap"
)

// ModelsHandler handles model list requests
type ModelsHandler struct {
	logger        *zap.Logger
	tokenizerPath string
}

// NewModelsHandler creates a new models handler
func NewModelsHandler(logger *zap.Logger, tokenizerPath string) *ModelsHandler {
	return &ModelsHandler{
		logger:        logger,
		tokenizerPath: tokenizerPath,
	}
}

// List handles GET /v1/models
func (h *ModelsHandler) List(ctx *fasthttp.RequestCtx) {
	// Return a default model for OpenAI compatibility
	ctx.SetStatusCode(200)
	ctx.SetContentType("application/json")

	response := map[string]interface{}{
		"object": "list",
		"data": []map[string]interface{}{
			{
				"id":       "default",
				"object":   "model",
				"created": 1677610602,
				"owned_by": "sglang",
			},
		},
	}

	jsonData, _ := json.Marshal(response)
	ctx.Write(jsonData)
}

// GetModelInfo handles GET /get_model_info
// Returns model information compatible with SGLang RuntimeEndpoint
func (h *ModelsHandler) GetModelInfo(ctx *fasthttp.RequestCtx) {
	ctx.SetStatusCode(200)
	ctx.SetContentType("application/json")

	// Return model info compatible with SGLang RuntimeEndpoint expectations
	response := map[string]interface{}{
		"model_path": h.tokenizerPath, // Use tokenizer path as model path
		"tokenizer_path": h.tokenizerPath,
		"is_generation": true,
		"preferred_sampling_params": "",
		"weight_version": "",
		"has_image_understanding": false,
		"has_audio_understanding": false,
		"model_type": "",
		"architectures": nil,
	}

	jsonData, _ := json.Marshal(response)
	ctx.Write(jsonData)
}
