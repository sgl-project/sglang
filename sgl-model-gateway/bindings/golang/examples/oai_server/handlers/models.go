package handlers

import (
	"encoding/json"

	"github.com/valyala/fasthttp"
	"go.uber.org/zap"
)

// ModelsHandler handles model list requests
type ModelsHandler struct {
	logger *zap.Logger
}

// NewModelsHandler creates a new models handler
func NewModelsHandler(logger *zap.Logger) *ModelsHandler {
	return &ModelsHandler{
		logger: logger,
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




