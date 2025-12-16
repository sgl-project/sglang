package sglang

import (
	"context"
	"testing"
)

// TestClientConfig tests ClientConfig validation
func TestClientConfig(t *testing.T) {
	tests := []struct {
		name    string
		config  ClientConfig
		wantErr bool
	}{
		{
			name: "valid config",
			config: ClientConfig{
				Endpoint:      "grpc://localhost:20000",
				TokenizerPath: "/path/to/tokenizer",
			},
			wantErr: false,
		},
		{
			name: "missing endpoint",
			config: ClientConfig{
				Endpoint:      "",
				TokenizerPath: "/path/to/tokenizer",
			},
			wantErr: true,
		},
		{
			name: "missing tokenizer path",
			config: ClientConfig{
				Endpoint:      "grpc://localhost:20000",
				TokenizerPath: "",
			},
			wantErr: true,
		},
		{
			name: "both missing",
			config: ClientConfig{
				Endpoint:      "",
				TokenizerPath: "",
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewClient(tt.config)
			if (err != nil) != tt.wantErr {
				t.Errorf("NewClient() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

// TestChatMessageTypes tests ChatMessage struct and its variants
func TestChatMessageTypes(t *testing.T) {
	msg := ChatMessage{
		Role:    "user",
		Content: "Hello",
	}

	if msg.Role != "user" {
		t.Errorf("Expected role 'user', got '%s'", msg.Role)
	}
	if msg.Content != "Hello" {
		t.Errorf("Expected content 'Hello', got '%s'", msg.Content)
	}
}

// TestChatCompletionRequestValidation tests ChatCompletionRequest validation
func TestChatCompletionRequestValidation(t *testing.T) {
	// Test valid request
	req := ChatCompletionRequest{
		Model: "default",
		Messages: []ChatMessage{
			{Role: "user", Content: "test"},
		},
		Stream: false,
	}

	if req.Model == "" {
		t.Error("Expected model to be set")
	}

	if len(req.Messages) == 0 {
		t.Error("Expected messages to be non-empty")
	}

	if req.Messages[0].Role != "user" {
		t.Errorf("Expected first message role 'user', got '%s'", req.Messages[0].Role)
	}
}

// TestClientClose tests that Close can be called multiple times safely
func TestClientClose(t *testing.T) {
	// Create a mock client (note: in real tests, you might want to skip this
	// if it requires actual server connection)
	config := ClientConfig{
		Endpoint:      "grpc://localhost:20000",
		TokenizerPath: "/path/to/tokenizer",
	}

	// Skip if connection fails (expected in unit test environment)
	client, err := NewClient(config)
	if err != nil {
		t.Skip("Skipping client close test: server not available")
	}

	// First close should succeed
	if err := client.Close(); err != nil {
		t.Errorf("First Close() failed: %v", err)
	}

	// Second close should also succeed (idempotent)
	if err := client.Close(); err != nil {
		t.Errorf("Second Close() failed: %v", err)
	}
}

// TestChatCompletionResponseTypes tests response type structures
func TestChatCompletionResponseTypes(t *testing.T) {
	resp := ChatCompletionResponse{
		ID:      "test-id",
		Model:   "default",
		Created: 1234567890,
		Choices: []Choice{
			{
				Message: Message{
					Role:    "assistant",
					Content: "Hello",
				},
				FinishReason: "stop",
			},
		},
		Usage: Usage{
			PromptTokens:     10,
			CompletionTokens: 20,
			TotalTokens:      30,
		},
	}

	if resp.ID != "test-id" {
		t.Errorf("Expected ID 'test-id', got '%s'", resp.ID)
	}

	if len(resp.Choices) != 1 {
		t.Errorf("Expected 1 choice, got %d", len(resp.Choices))
	}

	if resp.Choices[0].Message.Content != "Hello" {
		t.Errorf("Expected content 'Hello', got '%s'", resp.Choices[0].Message.Content)
	}

	if resp.Usage.TotalTokens != 30 {
		t.Errorf("Expected total tokens 30, got %d", resp.Usage.TotalTokens)
	}
}

// TestStreamingResponseTypes tests streaming response structures
func TestStreamingResponseTypes(t *testing.T) {
	chunk := ChatCompletionStreamResponse{
		ID:      "stream-id",
		Created: 1234567890,
		Choices: []StreamChoice{
			{
				Index: 0,
				Delta: MessageDelta{
					Content: "Hello",
				},
				FinishReason: "",
			},
		},
	}

	if chunk.ID != "stream-id" {
		t.Errorf("Expected ID 'stream-id', got '%s'", chunk.ID)
	}

	if len(chunk.Choices) == 0 {
		t.Error("Expected at least one choice")
	}

	if chunk.Choices[0].Delta.Content != "Hello" {
		t.Errorf("Expected delta content 'Hello', got '%s'", chunk.Choices[0].Delta.Content)
	}
}

// TestToolCallStructure tests Tool and ToolCall structures
func TestToolCallStructure(t *testing.T) {
	tool := Tool{
		Type: "function",
		Function: Function{
			Name:        "get_weather",
			Description: "Get the weather",
			Parameters: map[string]interface{}{
				"location": "string",
			},
		},
	}

	if tool.Type != "function" {
		t.Errorf("Expected tool type 'function', got '%s'", tool.Type)
	}

	if tool.Function.Name != "get_weather" {
		t.Errorf("Expected function name 'get_weather', got '%s'", tool.Function.Name)
	}

	toolCall := ToolCall{
		ID:   "call-123",
		Type: "function",
		Function: FunctionCall{
			Name:      "get_weather",
			Arguments: `{"location": "San Francisco"}`,
		},
	}

	if toolCall.ID != "call-123" {
		t.Errorf("Expected tool call ID 'call-123', got '%s'", toolCall.ID)
	}
}

// TestConcurrentClientOperations tests thread safety
// This is a basic test that just verifies concurrent calls don't panic
func TestConcurrentClientOperations(t *testing.T) {
	config := ClientConfig{
		Endpoint:      "grpc://localhost:20000",
		TokenizerPath: "/path/to/tokenizer",
	}

	client, err := NewClient(config)
	if err != nil {
		t.Skip("Skipping concurrent operations test: server not available")
	}
	defer client.Close()

	// Try concurrent Close calls (should not panic or race)
	done := make(chan bool, 2)

	go func() {
		client.Close()
		done <- true
	}()

	go func() {
		client.Close()
		done <- true
	}()

	<-done
	<-done
}

// BenchmarkChatCompletionRequest benchmarks request creation
func BenchmarkChatCompletionRequest(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = ChatCompletionRequest{
			Model: "default",
			Messages: []ChatMessage{
				{Role: "user", Content: "test message"},
			},
			Stream:              false,
			Temperature:         floatPtr(0.7),
			MaxCompletionTokens: intPtr(100),
		}
	}
}

// Helper functions for benchmarks
func floatPtr(f float32) *float32 {
	return &f
}

func intPtr(i int) *int {
	return &i
}

// TestContextCancellation tests that cancelled context is handled gracefully.
//
// NOTE: Currently, the FFI layer is blocking and doesn't actively monitor context cancellation.
// This test verifies that the client at least returns an error rather than panicking or
// hanging indefinitely when a pre-cancelled context is passed.
//
// Future: When FFI supports context cancellation (via signals or async operations),
// this test should be updated to assert that the error is context.Canceled or wrapped
// context cancellation error.
func TestContextCancellation(t *testing.T) {
	config := ClientConfig{
		Endpoint:      "grpc://localhost:20000",
		TokenizerPath: "/path/to/tokenizer",
	}

	client, err := NewClient(config)
	if err != nil {
		t.Skip("Skipping context cancellation test: server not available")
	}
	defer client.Close()

	// Create a pre-cancelled context
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	req := ChatCompletionRequest{
		Model: "default",
		Messages: []ChatMessage{
			{Role: "user", Content: "test"},
		},
	}

	// Attempt request with cancelled context
	// Since FFI is blocking, we expect either:
	// 1. An error from the server/network
	// 2. The call to complete normally (FFI doesn't check context)
	// What we DON'T expect is a panic or indefinite hang
	_, err = client.CreateChatCompletion(ctx, req)
	if err != nil {
		t.Logf("Request with cancelled context returned error: %v", err)
	} else {
		t.Logf("Request with cancelled context completed (FFI may not support context cancellation)")
	}
}
