// Package sglang provides a Go SDK for SGLang gRPC API.
//
// SGLang is a fast language model serving framework. This package provides a Go client
// library for interacting with SGLang's gRPC API, following the style of OpenAI's Go SDK.
//
// Basic usage:
//
//	client, err := sglang.NewClient(sglang.ClientConfig{
//		Endpoint:      "grpc://localhost:20000",
//		TokenizerPath: "/path/to/tokenizer",
//	})
//	if err != nil {
//		log.Fatal(err)
//	}
//	defer client.Close()
//
//	resp, err := client.CreateChatCompletion(ctx, sglang.ChatCompletionRequest{
//		Model: "default",
//		Messages: []sglang.ChatMessage{
//			{Role: "user", Content: "Hello"},
//		},
//	})
//
// For streaming responses, use CreateChatCompletionStream instead.
package sglang

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
	"sync"

	"github.com/sglang/sglang-go-grpc-sdk/internal/ffi"
)

// Client is the main client for interacting with SGLang gRPC API.
// It manages the connection to the SGLang server and handles both streaming
// and non-streaming chat completions.
//
// Thread-safe: All public methods are safe for concurrent use.
type Client struct {
	endpoint      string
	tokenizerPath string
	clientHandle  *ffi.SglangClientHandle
	mu            sync.RWMutex
}

// ClientConfig holds configuration for creating a new client.
type ClientConfig struct {
	// Endpoint is the gRPC endpoint URL (e.g., "grpc://localhost:20000").
	// Required field. Must include the scheme (grpc://) and port number.
	Endpoint string

	// TokenizerPath is the path to the tokenizer directory containing
	// tokenizer configuration files (e.g., tokenizer.json, vocab.json).
	// Required field.
	TokenizerPath string
}

// NewClient creates a new SGLang client with the given configuration.
//
// The client maintains a long-lived connection to the SGLang server and should
// be reused for multiple requests. Call Close() to release resources.
//
// Returns an error if:
// - Endpoint is empty
// - TokenizerPath is empty
// - Connection to the server fails
func NewClient(config ClientConfig) (*Client, error) {
	if config.Endpoint == "" {
		return nil, errors.New("endpoint is required")
	}
	if config.TokenizerPath == "" {
		return nil, errors.New("tokenizer path is required")
	}

	clientHandle, err := ffi.NewClient(config.Endpoint, config.TokenizerPath)
	if err != nil {
		return nil, fmt.Errorf("failed to create client: %w", err)
	}

	return &Client{
		endpoint:      config.Endpoint,
		tokenizerPath: config.TokenizerPath,
		clientHandle:  clientHandle,
	}, nil
}

// Close closes the client and releases all resources.
//
// After Close() is called, the client cannot be used for further requests.
// Calling Close() multiple times is safe and idempotent.
func (c *Client) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.clientHandle != nil {
		c.clientHandle.Free()
		c.clientHandle = nil
	}
	return nil
}

// ChatCompletionRequest represents a request for chat completion.
// It follows the OpenAI API style for familiar usage.
type ChatCompletionRequest struct {
	// Model specifies the model to use for completion (e.g., "default")
	Model string `json:"model"`
	// Messages is the list of messages in the conversation
	Messages            []ChatMessage   `json:"messages"`
	Temperature         *float32        `json:"temperature,omitempty"`
	TopP                *float32        `json:"top_p,omitempty"`
	TopK                *int            `json:"top_k,omitempty"`
	MaxCompletionTokens *int            `json:"max_completion_tokens,omitempty"`
	Stream              bool            `json:"stream"`
	Tools               []Tool          `json:"tools,omitempty"`
	ToolChoice          interface{}     `json:"tool_choice,omitempty"`
	Stop                interface{}     `json:"stop,omitempty"`
	StopTokenIDs        []int           `json:"stop_token_ids,omitempty"`
	SkipSpecialTokens   bool            `json:"skip_special_tokens,omitempty"`
	FrequencyPenalty    *float32        `json:"frequency_penalty,omitempty"`
	PresencePenalty     *float32        `json:"presence_penalty,omitempty"`
	ResponseFormat      *ResponseFormat `json:"response_format,omitempty"`
	Seed                *int            `json:"seed,omitempty"`
	Logprobs            bool            `json:"logprobs,omitempty"`
	TopLogprobs         *int            `json:"top_logprobs,omitempty"`
	User                string          `json:"user,omitempty"`
}

// ChatMessage represents a single message in a chat conversation
type ChatMessage struct {
	Role    string      `json:"role"`
	Content interface{} `json:"content"`
	Name    string      `json:"name,omitempty"`
}

// Tool represents a tool/function that can be called
type Tool struct {
	Type     string   `json:"type"`
	Function Function `json:"function"`
}

// Function represents a function definition
type Function struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description,omitempty"`
	Parameters  map[string]interface{} `json:"parameters"`
}

// ResponseFormat represents the response format
type ResponseFormat struct {
	Type string `json:"type"`
}

// ChatCompletionResponse represents a non-streaming chat completion response
type ChatCompletionResponse struct {
	ID                string   `json:"id"`
	Object            string   `json:"object"`
	Created           int64    `json:"created"`
	Model             string   `json:"model"`
	SystemFingerprint string   `json:"system_fingerprint,omitempty"`
	Choices           []Choice `json:"choices"`
	Usage             Usage    `json:"usage"`
}

// Choice represents a choice in the completion response
type Choice struct {
	Index        int     `json:"index"`
	Message      Message `json:"message"`
	FinishReason string  `json:"finish_reason"`
}

// Message represents a message in the response
type Message struct {
	Role      string     `json:"role"`
	Content   string     `json:"content"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
}

// ToolCall represents a tool call in the response
type ToolCall struct {
	ID       string       `json:"id"`
	Type     string       `json:"type"`
	Function FunctionCall `json:"function"`
}

// FunctionCall represents a function call
type FunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// Usage represents token usage information
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// ChatCompletionStreamResponse represents a streaming chat completion response
type ChatCompletionStreamResponse struct {
	ID                string         `json:"id"`
	Object            string         `json:"object"`
	Created           int64          `json:"created"`
	Model             string         `json:"model"`
	SystemFingerprint string         `json:"system_fingerprint,omitempty"`
	Choices           []StreamChoice `json:"choices"`
	Usage             *Usage         `json:"usage,omitempty"`
}

// StreamChoice represents a choice in a streaming response
type StreamChoice struct {
	Index        int          `json:"index"`
	Delta        MessageDelta `json:"delta"`
	FinishReason string       `json:"finish_reason,omitempty"`
}

// MessageDelta represents incremental message updates
type MessageDelta struct {
	Role      string     `json:"role,omitempty"`
	Content   string     `json:"content,omitempty"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
}

// CreateChatCompletion creates a non-streaming chat completion with context support.
//
// Context Support:
// The ctx parameter is fully supported for cancellation and timeouts:
// - If ctx is cancelled, the request will be interrupted on the next stream.Recv() call
// - If ctx times out, the request will return context.DeadlineExceeded
//
// Example with timeout:
//
//	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
//	defer cancel()
//	resp, err := client.CreateChatCompletion(ctx, req)
//
// Note: Internally, this creates a stream and collects all chunks,
// so context monitoring happens at the chunk level.
func (c *Client) CreateChatCompletion(ctx context.Context, req ChatCompletionRequest) (*ChatCompletionResponse, error) {
	// For non-streaming, we'll collect all chunks and return the final response
	req.Stream = true // We still use streaming internally, but collect all chunks

	// Prepare request: if Tools is empty, set to nil for proper JSON serialization
	if len(req.Tools) == 0 {
		req.Tools = nil
	}

	stream, err := c.CreateChatCompletionStream(ctx, req)
	if err != nil {
		return nil, err
	}
	defer stream.Close()

	var fullContent strings.Builder
	var fullToolCalls []ToolCall
	var finishReason string
	var usage Usage
	var responseID string
	var created int64
	var model string
	var systemFingerprint string

	for {
		chunk, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}

		if chunk.ID != "" {
			responseID = chunk.ID
		}
		if chunk.Created > 0 {
			created = chunk.Created
		}
		if chunk.Model != "" {
			model = chunk.Model
		}
		if chunk.SystemFingerprint != "" {
			systemFingerprint = chunk.SystemFingerprint
		}

		for _, choice := range chunk.Choices {
			if choice.Delta.Content != "" {
				fullContent.WriteString(choice.Delta.Content)
			}
			if len(choice.Delta.ToolCalls) > 0 {
				fullToolCalls = append(fullToolCalls, choice.Delta.ToolCalls...)
			}
			// Always update finish_reason if present (even if empty string, but should not be empty)
			// The last chunk (Complete message) should have finish_reason set
			if choice.FinishReason != "" {
				finishReason = choice.FinishReason
			}
		}

		// Extract usage from chunk if available (usually in the last chunk)
		// Always update usage if present, as the last chunk should have the final usage
		if chunk.Usage != nil {
			usage = *chunk.Usage
		}
	}

	// Build final response
	message := Message{
		Role:    "assistant",
		Content: fullContent.String(),
	}
	if len(fullToolCalls) > 0 {
		message.ToolCalls = fullToolCalls
	}

	// Ensure finish_reason is set (defensive check)
	// If finish_reason is still empty, default to "stop"
	if finishReason == "" {
		finishReason = "stop"
	}

	return &ChatCompletionResponse{
		ID:                responseID,
		Object:            "chat.completion",
		Created:           created,
		Model:             model,
		SystemFingerprint: systemFingerprint,
		Choices: []Choice{
			{
				Index:        0,
				Message:      message,
				FinishReason: finishReason,
			},
		},
		Usage: usage,
	}, nil
}

// ChatCompletionStream represents a streaming chat completion
type ChatCompletionStream struct {
	stream *ffi.SglangStreamHandle
	mu     sync.Mutex
	done   bool               // Track if stream has been marked as done
	ctx    context.Context    // Context for cancellation support
	cancel context.CancelFunc // Cancel function to stop monitoring goroutine
	closed chan struct{}      // Signal when stream is closed
}

// Recv receives the next chunk from the stream.
//
// Supports context cancellation: if the context passed to CreateChatCompletionStream
// is cancelled, Recv will return context.Canceled error on the next call.
func (s *ChatCompletionStream) Recv() (*ChatCompletionStreamResponse, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Check if context was cancelled
	select {
	case <-s.ctx.Done():
		return nil, s.ctx.Err() // Returns context.Canceled or context.DeadlineExceeded
	default:
	}

	if s.stream == nil {
		return nil, io.EOF
	}

	// If stream was already marked as done, immediately return EOF
	// This prevents calling ReadNext() again after isDone=1
	if s.done {
		return nil, io.EOF
	}

	// Loop to handle empty responses (Ok(None) from Rust)
	// Keep reading until we get actual data or stream ends
	for {
		responseJSON, isDone, err := s.stream.ReadNext()
		if err != nil {
			return nil, err
		}

		// Mark stream as done if ReadNext indicates completion
		if isDone {
			s.done = true
		}

		// If we have a response, parse and return it
		if responseJSON != "" {
			var response ChatCompletionStreamResponse
			if err := json.Unmarshal([]byte(responseJSON), &response); err != nil {
				return nil, fmt.Errorf("failed to parse response: %w", err)
			}
			return &response, nil
		}

		// If stream is done but no response, return EOF
		if isDone {
			return nil, io.EOF
		}

		// Empty response and stream not done - loop to read next chunk
		// This handles Ok(None) cases where Rust returns no data but stream continues
	}
}

// Close closes the stream and cancels any pending operations.
func (s *ChatCompletionStream) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Cancel the context to signal the monitoring goroutine to stop
	if s.cancel != nil {
		s.cancel()
	}

	// Signal that stream is closed
	select {
	case <-s.closed:
		// Already closed
	default:
		close(s.closed)
	}

	// Free the stream to mark it as completed
	// This prevents AbortOnDropStream from sending abort when dropped
	if s.stream != nil {
		s.stream.Free()
		s.stream = nil
	}
	return nil
}

// CreateChatCompletionStream creates a streaming chat completion with context cancellation support.
//
// Context Support:
// The ctx parameter is now fully supported for cancellation and timeouts:
// - If ctx is cancelled, stream.Recv() will return context.Canceled on the next call
// - If ctx times out (WithTimeout), stream.Recv() will return context.DeadlineExceeded
// - Calling stream.Close() also cancels the context
//
// Example with timeout:
//
//	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
//	defer cancel()
//	stream, err := client.CreateChatCompletionStream(ctx, req)
//	// Stream will auto-close if 30 seconds elapse
//
// Example with cancellation:
//
//	ctx, cancel := context.WithCancel(context.Background())
//	stream, err := client.CreateChatCompletionStream(ctx, req)
//	go func() {
//	    time.Sleep(5*time.Second)
//	    cancel()  // Cancel after 5 seconds
//	}()
func (c *Client) CreateChatCompletionStream(ctx context.Context, req ChatCompletionRequest) (*ChatCompletionStream, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if c.clientHandle == nil {
		return nil, errors.New("client is closed")
	}

	// Marshal request to JSON, then ensure tools field is always present.
	// Due to omitempty tag, empty Tools slice will be omitted from JSON.
	// We need to ensure tools field is always present as [] when empty (not omitted),
	// matching the behavior of complete_sdk example.
	reqJSON, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Unmarshal into map and ensure tools field is present
	var reqMap map[string]interface{}
	if err := json.Unmarshal(reqJSON, &reqMap); err != nil {
		return nil, fmt.Errorf("failed to unmarshal request to map: %w", err)
	}

	// Add empty tools array if not present
	if _, exists := reqMap["tools"]; !exists {
		reqMap["tools"] = []interface{}{}
	}

	// Marshal back to JSON
	reqJSON, err = json.Marshal(reqMap)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request map to JSON: %w", err)
	}

	// Create stream
	streamHandle, err := c.clientHandle.ChatCompletionStream(string(reqJSON))
	if err != nil {
		return nil, fmt.Errorf("failed to create stream: %w", err)
	}

	// Create a child context from the provided context for cancellation support
	streamCtx, cancel := context.WithCancel(ctx)

	stream := &ChatCompletionStream{
		stream: streamHandle,
		ctx:    streamCtx,
		cancel: cancel,
		closed: make(chan struct{}),
	}

	return stream, nil
}
