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
	"time"

	grpcclient "github.com/sglang/sglang-go-grpc-sdk/internal/grpc"
)

// Client is the main client for interacting with SGLang gRPC API.
// It manages the connection to the SGLang server and handles both streaming
// and non-streaming chat completions.
//
// Thread-safe: All public methods are safe for concurrent use.
type Client struct {
	endpoint      string
	tokenizerPath string
	grpcClient    *grpcclient.GrpcClient // gRPC-based client
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

	// ChannelBufferSizes configures buffer sizes for internal channels.
	// If nil, default values will be used (optimized for high concurrency).
	ChannelBufferSizes *ChannelBufferSizes

	// Timeouts configures timeout values for various operations.
	// If nil, default values will be used.
	Timeouts *Timeouts
}

// ChannelBufferSizes configures buffer sizes for internal channels.
// These affect concurrency and memory usage. Larger buffers allow more
// concurrent operations but use more memory.
type ChannelBufferSizes = grpcclient.ChannelBufferSizes

// Timeouts configures timeout values for various operations.
type Timeouts = grpcclient.Timeouts

// defaultChannelBufferSizes returns default channel buffer sizes optimized for high concurrency (10k+).
// These values are designed to handle thousands of concurrent requests without blocking.
func defaultChannelBufferSizes() ChannelBufferSizes {
	return ChannelBufferSizes{
		ResultJSONChan: 10000, // Increased for high concurrency: each request may produce 200-500 chunks
		ErrChan:        100,   // Errors are rare, 100 is sufficient
		RecvChan:       2000,  // Increased for high concurrency: more gRPC responses to buffer
	}
}

// defaultTimeouts returns default timeout values.
func defaultTimeouts() Timeouts {
	return Timeouts{
		KeepaliveTime:    300 * time.Second, // Increased to reduce ping frequency and avoid "too many pings" errors
		KeepaliveTimeout: 20 * time.Second,
		CloseTimeout:     5 * time.Second,
	}
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

	bufferSizes := defaultChannelBufferSizes()
	if config.ChannelBufferSizes != nil {
		if config.ChannelBufferSizes.ResultJSONChan > 0 {
			bufferSizes.ResultJSONChan = config.ChannelBufferSizes.ResultJSONChan
		}
		if config.ChannelBufferSizes.ErrChan > 0 {
			bufferSizes.ErrChan = config.ChannelBufferSizes.ErrChan
		}
		if config.ChannelBufferSizes.RecvChan > 0 {
			bufferSizes.RecvChan = config.ChannelBufferSizes.RecvChan
		}
	}

	timeouts := defaultTimeouts()
	if config.Timeouts != nil {
		if config.Timeouts.KeepaliveTime > 0 {
			timeouts.KeepaliveTime = config.Timeouts.KeepaliveTime
		}
		if config.Timeouts.KeepaliveTimeout > 0 {
			timeouts.KeepaliveTimeout = config.Timeouts.KeepaliveTimeout
		}
		if config.Timeouts.CloseTimeout > 0 {
			timeouts.CloseTimeout = config.Timeouts.CloseTimeout
		}
	}

	grpcClient, err := grpcclient.NewGrpcClient(config.Endpoint, config.TokenizerPath, bufferSizes, timeouts)
	if err != nil {
		return nil, fmt.Errorf("failed to create gRPC client: %w", err)
	}

	return &Client{
		endpoint:      config.Endpoint,
		tokenizerPath: config.TokenizerPath,
		grpcClient:    grpcClient,
	}, nil
}

// Close closes the client and releases all resources.
//
// After Close() is called, the client cannot be used for further requests.
// Calling Close() multiple times is safe and idempotent.
func (c *Client) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.grpcClient != nil {
		if err := c.grpcClient.Close(); err != nil {
			return err
		}
		c.grpcClient = nil
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
// - If ctx is cancelled, the request will be interrupted on the next stream.RecvJSON() call
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
		chunkJSON, err := stream.RecvJSON()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}

		var chunk ChatCompletionStreamResponse
		if err := json.Unmarshal([]byte(chunkJSON), &chunk); err != nil {
			return nil, fmt.Errorf("failed to parse chunk: %w", err)
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
			if choice.FinishReason != "" {
				finishReason = choice.FinishReason
			}
		}

		if chunk.Usage != nil {
			usage = *chunk.Usage
		}
	}

	message := Message{
		Role:    "assistant",
		Content: fullContent.String(),
	}
	if len(fullToolCalls) > 0 {
		message.ToolCalls = fullToolCalls
	}

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
	grpcStream *grpcclient.GrpcChatCompletionStream
	ctx        context.Context
	cancel     context.CancelFunc
}

func (s *ChatCompletionStream) RecvJSON() (string, error) {
	return s.grpcStream.RecvJSON()
}

// Close closes the stream and cancels any pending operations.
func (s *ChatCompletionStream) Close() error {
	if s.cancel != nil {
		s.cancel()
	}
	if s.grpcStream != nil {
		return s.grpcStream.Close()
	}
	return nil
}

// CreateChatCompletionStream creates a streaming chat completion with context cancellation support.
//
// Context Support:
// The ctx parameter is now fully supported for cancellation and timeouts:
// - If ctx is cancelled, stream.RecvJSON() will return context.Canceled on the next call
// - If ctx times out (WithTimeout), stream.RecvJSON() will return context.DeadlineExceeded
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
	reqJSON, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	var reqMap map[string]interface{}
	if err := json.Unmarshal(reqJSON, &reqMap); err != nil {
		return nil, fmt.Errorf("failed to unmarshal request to map: %w", err)
	}

	if _, exists := reqMap["tools"]; !exists {
		reqMap["tools"] = []interface{}{}
	}

	reqJSON, err = json.Marshal(reqMap)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request map to JSON: %w", err)
	}

	if c.grpcClient == nil {
		return nil, errors.New("gRPC client is closed")
	}

	grpcStream, err := c.grpcClient.CreateChatCompletionStream(ctx, string(reqJSON))
	if err != nil {
		return nil, fmt.Errorf("failed to create gRPC stream: %w", err)
	}

	streamCtx, cancel := context.WithCancel(ctx)
	return &ChatCompletionStream{
		grpcStream: grpcStream,
		ctx:        streamCtx,
		cancel:     cancel,
	}, nil
}
