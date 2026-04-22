package utils

import (
	"encoding/json"

	"github.com/valyala/fasthttp"
)

// RespondError sends an error response in OpenAI format
func RespondError(ctx *fasthttp.RequestCtx, statusCode int, message, errorType string) {
	ctx.SetStatusCode(statusCode)
	ctx.SetContentType("application/json")

	response := map[string]interface{}{
		"error": map[string]interface{}{
			"message": message,
			"type":    errorType,
			"code":    statusCode,
		},
	}

	jsonData, _ := json.Marshal(response)
	ctx.Write(jsonData)
}

// BuildResponseBase builds the base response structure for OpenAI-compatible responses
func BuildResponseBase(id string, created int64, model string) map[string]interface{} {
	return map[string]interface{}{
		"id":      id,
		"object":  "chat.completion",
		"created": created,
		"model":   model,
	}
}
