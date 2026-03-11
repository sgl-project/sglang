// Streaming example demonstrating real-time streaming with SGLang Go SDK
package main

import (
	"context"
	"fmt"
	"io"
	"log"
	"os"
	"strings"
	"time"

	"github.com/sglang/sglang-go-grpc-sdk"
)

func main() {
	// Get configuration from environment or command line
	endpoint := os.Getenv("SGL_GRPC_ENDPOINT")
	if endpoint == "" {
		endpoint = "grpc://localhost:20000"
	}

	tokenizerPath := os.Getenv("SGL_TOKENIZER_PATH")
	if tokenizerPath == "" {
		tokenizerPath = "./examples/tokenizer"
	}

	// Create client
	client, err := sglang.NewClient(sglang.ClientConfig{
		Endpoint:      endpoint,
		TokenizerPath: tokenizerPath,
	})
	if err != nil {
		log.Fatalf("Failed to create client: %v", err)
	}
	defer client.Close()

	// Create streaming chat completion request
	req := sglang.ChatCompletionRequest{
		Model: "default",
		Messages: []sglang.ChatMessage{
			{
				Role:    "system",
				Content: "You are a helpful assistant.",
			},
			{
				Role:    "user",
				Content: "写一首春天的诗歌",
			},
		},
		Stream:              true,
		Temperature:         float32Ptr(0.7),
		MaxCompletionTokens: intPtr(500),
		SkipSpecialTokens:   true,
		Tools:               nil, // Use nil instead of empty slice to avoid template errors
	}

	// Create streaming completion
	ctx := context.Background()
	stream, err := client.CreateChatCompletionStream(ctx, req)
	if err != nil {
		log.Fatalf("Failed to create stream: %v", err)
	}
	defer stream.Close()

	fmt.Println("=== Streaming Response ===")
	fmt.Println()

	var fullContent strings.Builder
	chunkCount := 0
	startTime := time.Now()
	var firstTokenTime time.Time
	firstTokenReceived := false

	for {
		chunk, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			log.Fatalf("Stream error: %v", err)
		}

		chunkCount++

		// Extract content from delta
		for _, choice := range chunk.Choices {
			if choice.Delta.Content != "" {
				fmt.Print(choice.Delta.Content)
				fullContent.WriteString(choice.Delta.Content)

				// Track first token time (TTFT)
				if !firstTokenReceived {
					firstTokenTime = time.Now()
					firstTokenReceived = true
					ttft := firstTokenTime.Sub(startTime)
					fmt.Printf("\n[TTFT: %v]\n", ttft)
				}
			}

			if choice.FinishReason != "" {
				fmt.Printf("\n\n[Finished: %s]\n", choice.FinishReason)
			}
		}
	}

	// Calculate metrics
	if firstTokenReceived {
		elapsed := time.Since(startTime)
		tokensPerSecond := float64(fullContent.Len()) / elapsed.Seconds()
		fmt.Printf("\n=== Metrics ===\n")
		fmt.Printf("Total chunks: %d\n", chunkCount)
		fmt.Printf("Total content length: %d characters\n", fullContent.Len())
		fmt.Printf("Time elapsed: %v\n", elapsed)
		fmt.Printf("Tokens per second: %.2f\n", tokensPerSecond)
	}
}

func float32Ptr(f float32) *float32 {
	return &f
}

func intPtr(i int) *int {
	return &i
}
