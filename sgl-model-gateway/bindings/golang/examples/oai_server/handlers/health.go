package handlers

import (
	"encoding/json"

	"github.com/valyala/fasthttp"
	"go.uber.org/zap"
)

// HealthHandler handles health check requests
type HealthHandler struct {
	logger *zap.Logger
}

// NewHealthHandler creates a new health handler
func NewHealthHandler(logger *zap.Logger) *HealthHandler {
	return &HealthHandler{
		logger: logger,
	}
}

// Check handles GET /health
func (h *HealthHandler) Check(ctx *fasthttp.RequestCtx) {
	ctx.SetStatusCode(200)
	ctx.SetContentType("application/json")

	response := map[string]string{
		"status": "ok",
	}

	jsonData, _ := json.Marshal(response)
	ctx.Write(jsonData)
}
