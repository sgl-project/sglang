//go:build integration
// +build integration

// integration_test.go contains integration tests that require a running SGLang server
//
// To run these tests:
// 1. Start an SGLang server: python -m sglang.launch_server --model-path meta-llama/Llama-2-7b-hf
// 2. Run: go test -tags=integration -run TestIntegration

package sglang

import (
	"context"
	"io"
	"os"
	"testing"
	"time"
)

// getTestConfig returns test configuration from environment or defaults
func getTestConfig(t *testing.T) ClientConfig {
	endpoint := os.Getenv("SGL_GRPC_ENDPOINT")
	if endpoint == "" {
		endpoint = "grpc://localhost:20000"
	}

	tokenizerPath := os.Getenv("SGL_TOKENIZER_PATH")
	if tokenizerPath == "" {
		t.Skip("SGL_TOKENIZER_PATH not set")
	}

	return ClientConfig{
		Endpoint:      endpoint,
		TokenizerPath: tokenizerPath,
	}
}

// TestIntegrationNonStreamingCompletion tests non-streaming chat completion
func TestIntegrationNonStreamingCompletion(t *testing.T) {
	config := getTestConfig(t)

	client, err := NewClient(config)
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}
	defer client.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	req := ChatCompletionRequest{
		Model: "default",
		Messages: []ChatMessage{
			{Role: "user", Content: "Say 'Hello, World!' only"},
		},
		Stream:              false,
		Temperature:         float32Ptr(0.0),
		MaxCompletionTokens: intPtr(50),
	}

	resp, err := client.CreateChatCompletion(ctx, req)
	if err != nil {
		t.Fatalf("CreateChatCompletion failed: %v", err)
	}

	if resp.ID == "" {
		t.Error("Response ID is empty")
	}

	if len(resp.Choices) == 0 {
		t.Error("Response has no choices")
	}

	if resp.Choices[0].Message.Content == "" {
		t.Error("Response content is empty")
	}

	if resp.Usage == nil || resp.Usage.TotalTokens == 0 {
		t.Error("Usage information is missing or invalid")
	}

	t.Logf("Response: %s", resp.Choices[0].Message.Content)
	t.Logf("Usage: %+v", resp.Usage)
}

// TestIntegrationStreamingCompletion tests streaming chat completion
func TestIntegrationStreamingCompletion(t *testing.T) {
	config := getTestConfig(t)

	client, err := NewClient(config)
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}
	defer client.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	req := ChatCompletionRequest{
		Model: "default",
		Messages: []ChatMessage{
			{Role: "user", Content: "Count from 1 to 5"},
		},
		Stream:              true,
		Temperature:         float32Ptr(0.0),
		MaxCompletionTokens: intPtr(100),
	}

	stream, err := client.CreateChatCompletionStream(ctx, req)
	if err != nil {
		t.Fatalf("CreateChatCompletionStream failed: %v", err)
	}
	defer stream.Close()

	chunkCount := 0
	totalContent := ""

	for {
		chunk, err := stream.Recv()
		if err == io.EOF {
			// io.EOF is expected at end of stream
			break
		}
		if err != nil {
			t.Fatalf("Stream error: %v", err)
		}

		chunkCount++

		for _, choice := range chunk.Choices {
			if choice.Delta.Content != "" {
				totalContent += choice.Delta.Content
			}
		}
	}

	if chunkCount == 0 {
		t.Error("Received no chunks from stream")
	}

	if totalContent == "" {
		t.Error("Received no content from stream")
	}

	t.Logf("Received %d chunks with content: %s", chunkCount, totalContent)
}

// TestIntegrationConcurrentRequests tests multiple concurrent requests
func TestIntegrationConcurrentRequests(t *testing.T) {
	config := getTestConfig(t)

	client, err := NewClient(config)
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}
	defer client.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	numRequests := 3
	done := make(chan error, numRequests)

	for i := 0; i < numRequests; i++ {
		go func(idx int) {
			req := ChatCompletionRequest{
				Model: "default",
				Messages: []ChatMessage{
					{Role: "user", Content: "Say 'test'"},
				},
				Stream:              false,
				MaxCompletionTokens: intPtr(50),
			}

			_, err := client.CreateChatCompletion(ctx, req)
			done <- err
		}(i)
	}

	// Collect results
	for i := 0; i < numRequests; i++ {
		if err := <-done; err != nil {
			t.Errorf("Request %d failed: %v", i, err)
		}
	}

	t.Logf("All %d concurrent requests completed successfully", numRequests)
}

// TestIntegrationContextCancellation tests that context cancellation is handled
func TestIntegrationContextCancellation(t *testing.T) {
	config := getTestConfig(t)

	client, err := NewClient(config)
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}
	defer client.Close()

	// Create a context that cancels immediately
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	req := ChatCompletionRequest{
		Model: "default",
		Messages: []ChatMessage{
			{Role: "user", Content: "test"},
		},
		Stream: false,
	}

	// Should handle cancelled context gracefully
	_, err = client.CreateChatCompletion(ctx, req)
	if err == nil {
		t.Error("Expected error from cancelled context")
	}

	t.Logf("Cancelled context handled: %v", err)
}

// Helper functions
func float32Ptr(f float32) *float32 {
	return &f
}

func intPtr(i int) *int {
	return &i
}
