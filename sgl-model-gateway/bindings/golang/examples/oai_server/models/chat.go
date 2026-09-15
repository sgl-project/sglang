package models

// ChatRequest represents an OpenAI-compatible chat completion request
type ChatRequest struct {
	Model                string                   `json:"model" binding:"required"`
	Messages             []map[string]string      `json:"messages" binding:"required"`
	Stream               bool                     `json:"stream,omitempty"`
	Temperature          *float64                 `json:"temperature,omitempty"`
	TopP                 *float64                 `json:"top_p,omitempty"`
	MaxTokens            *int                     `json:"max_tokens,omitempty"`              // OpenAI API standard field
	MaxCompletionTokens *int                     `json:"max_completion_tokens,omitempty"`    // SGLang-specific field (used by bench_serving.py)
	Tools                []map[string]interface{} `json:"tools,omitempty"`
	ToolChoice           interface{}              `json:"tool_choice,omitempty"`
}
