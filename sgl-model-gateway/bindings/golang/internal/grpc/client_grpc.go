// Package grpc provides gRPC client implementation for SGLang
package grpc

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/keepalive"
	"google.golang.org/protobuf/types/known/timestamppb"

	"github.com/sglang/sglang-go-grpc-sdk/internal/ffi"
	"github.com/sglang/sglang-go-grpc-sdk/internal/proto"
)

type grpcClientStream interface {
	Recv() (*proto.GenerateResponse, error)
	CloseSend() error
}

// recvResult holds the result of a Recv() call
type recvResult struct {
	resp *proto.GenerateResponse
	err  error
}

type GrpcClient struct {
	conn            *grpc.ClientConn
	client          proto.SglangSchedulerClient
	tokenizerPath   string
	tokenizerHandle *ffi.TokenizerHandle
	bufferSizes     ChannelBufferSizes
	timeouts        Timeouts
	requestCounter  uint64 // Atomic counter to ensure unique request IDs
}

type ChannelBufferSizes struct {
	ResultJSONChan int
	ErrChan        int
	RecvChan       int
}

type Timeouts struct {
	KeepaliveTime    time.Duration
	KeepaliveTimeout time.Duration
	CloseTimeout     time.Duration
}

func NewGrpcClient(endpoint, tokenizerPath string, bufferSizes ChannelBufferSizes, timeouts Timeouts) (*GrpcClient, error) {
	endpoint = strings.TrimPrefix(endpoint, "grpc://")
	if !strings.Contains(endpoint, ":") {
		return nil, fmt.Errorf("invalid endpoint format: %s (expected grpc://host:port)", endpoint)
	}

	keepaliveParams := keepalive.ClientParameters{
		Time:                timeouts.KeepaliveTime,
		Timeout:             timeouts.KeepaliveTimeout,
		PermitWithoutStream: false,
	}

	opts := []grpc.DialOption{
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithKeepaliveParams(keepaliveParams),
	}

	conn, err := grpc.NewClient(endpoint, opts...)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to gRPC server: %w", err)
	}

	client := proto.NewSglangSchedulerClient(conn)

	tokenizerHandle, err := ffi.CreateTokenizerHandle(tokenizerPath)
	if err != nil {
		conn.Close()
		return nil, fmt.Errorf("failed to create tokenizer handle: %w", err)
	}

	return &GrpcClient{
		conn:            conn,
		client:          client,
		tokenizerPath:   tokenizerPath,
		tokenizerHandle: tokenizerHandle,
		bufferSizes:     bufferSizes,
		timeouts:        timeouts,
	}, nil
}

func (c *GrpcClient) Close() error {
	if c.tokenizerHandle != nil {
		ffi.FreeTokenizerHandle(c.tokenizerHandle)
		c.tokenizerHandle = nil
	}

	if c.conn != nil {
		return c.conn.Close()
	}
	return nil
}

func (c *GrpcClient) CreateChatCompletionStream(ctx context.Context, reqJSON string) (*GrpcChatCompletionStream, error) {
	if c.tokenizerHandle == nil {
		return nil, fmt.Errorf("tokenizer handle is nil (should be created at startup)")
	}

	preprocessed, err := ffi.PreprocessChatRequestWithTokenizer(reqJSON, c.tokenizerHandle)
	if err != nil {
		return nil, fmt.Errorf("preprocessing failed: %w", err)
	}
	defer func() {
		if preprocessed != nil {
			preprocessed.Free()
		}
	}()

	// Parse request JSON to get parameters
	var reqMap map[string]interface{}
	if err := json.Unmarshal([]byte(reqJSON), &reqMap); err != nil {
		return nil, fmt.Errorf("failed to parse request JSON: %w", err)
	}

	model, _ := reqMap["model"].(string)
	if model == "" {
		model = "default"
	}

	// Build GenerateRequest
	// Generate unique request ID using timestamp + atomic counter to avoid collisions
	// This matches Rust version's UUID-based approach for uniqueness
	counter := atomic.AddUint64(&c.requestCounter, 1)
	requestID := fmt.Sprintf("chatcmpl-%d-%d", time.Now().UnixNano(), counter)
	generateReq := &proto.GenerateRequest{
		RequestId: requestID,
		Tokenized: &proto.TokenizedInput{
			OriginalText: preprocessed.PromptText,
			InputIds:     preprocessed.TokenIDs,
		},
		Stream: true,
	}

	// Set sampling parameters
	samplingParams := &proto.SamplingParams{
		Temperature:       1.0,
		TopP:              1.0,
		TopK:              -1,
		SkipSpecialTokens: true,
	}

	if temp, ok := reqMap["temperature"].(float64); ok {
		samplingParams.Temperature = float32(temp)
	}
	if topP, ok := reqMap["top_p"].(float64); ok {
		samplingParams.TopP = float32(topP)
	}
	if topK, ok := reqMap["top_k"].(float64); ok {
		samplingParams.TopK = int32(topK)
	}
	var maxTokensInt *int32
	if maxCompletionTokens, ok := reqMap["max_completion_tokens"].(float64); ok {
		tokens := int32(maxCompletionTokens)
		maxTokensInt = &tokens
	} else if maxTokens, ok := reqMap["max_tokens"].(float64); ok {
		tokens := int32(maxTokens)
		maxTokensInt = &tokens
	}
	if maxTokensInt != nil {
		samplingParams.MaxNewTokens = maxTokensInt
	}

	// Parse tool constraints if available
	if preprocessed.ToolConstraintsJSON != "" {
		var toolConstraints map[string]interface{}
		if err := json.Unmarshal([]byte(preprocessed.ToolConstraintsJSON), &toolConstraints); err == nil {
			if regex, ok := toolConstraints["regex"].(string); ok {
				samplingParams.Constraint = &proto.SamplingParams_Regex{Regex: regex}
			} else if jsonSchema, ok := toolConstraints["json_schema"].(string); ok {
				samplingParams.Constraint = &proto.SamplingParams_JsonSchema{JsonSchema: jsonSchema}
			}
		}
	}

	generateReq.SamplingParams = samplingParams
	generateReq.Timestamp = timestamppb.Now()

	stream, err := c.client.Generate(ctx, generateReq)
	if err != nil {
		return nil, fmt.Errorf("failed to create gRPC stream: %w", err)
	}
	toolsJSON := ""
	if tools, ok := reqMap["tools"].([]interface{}); ok && len(tools) > 0 {
		toolsBytes, _ := json.Marshal(tools)
		toolsJSON = string(toolsBytes)
	}

	toolChoiceJSON := ""
	if toolChoice, ok := reqMap["tool_choice"]; ok {
		toolChoiceBytes, _ := json.Marshal(toolChoice)
		toolChoiceJSON = string(toolChoiceBytes)
	}

	stopJSON := ""
	if stop, ok := reqMap["stop"]; ok {
		stopBytes, _ := json.Marshal(stop)
		stopJSON = string(stopBytes)
	}

	stopTokenIDs := []uint32{}
	if stopTokenIDsVal, ok := reqMap["stop_token_ids"].([]interface{}); ok {
		for _, id := range stopTokenIDsVal {
			if idFloat, ok := id.(float64); ok {
				stopTokenIDs = append(stopTokenIDs, uint32(idFloat))
			}
		}
	}

	skipSpecialTokens := true
	if skipSpecialTokensVal, ok := reqMap["skip_special_tokens"].(bool); ok {
		skipSpecialTokens = skipSpecialTokensVal
	}

	if c.tokenizerHandle == nil {
		stream.CloseSend()
		return nil, fmt.Errorf("tokenizer handle is nil (should be created at startup)")
	}

	converterHandle, err := ffi.CreateGrpcResponseConverterWithTokenizer(
		c.tokenizerHandle,
		model,
		generateReq.RequestId,
		toolsJSON,
		toolChoiceJSON,
		stopJSON,
		stopTokenIDs,
		skipSpecialTokens,
		preprocessed.PromptTokens, // Pass initial prompt tokens from preprocessing
	)
	if err != nil {
		stream.CloseSend()
		return nil, fmt.Errorf("failed to create converter handle: %w", err)
	}

	batchSize := 1
	batchPostprocessor := ffi.NewBatchPostprocessor(converterHandle, batchSize, 0)

	streamCtx, cancel := context.WithCancel(ctx)
	grpcStream := &GrpcChatCompletionStream{
		stream:             stream,
		converterHandle:    converterHandle,
		batchPostprocessor: batchPostprocessor,
		batchSize:          batchSize,
		ctx:                streamCtx,
		cancel:             cancel,
		resultJSONChan:     make(chan string, c.bufferSizes.ResultJSONChan),
		errChan:            make(chan error, c.bufferSizes.ErrChan),
		readLoopDone:       make(chan struct{}),
		requestID:          generateReq.RequestId,
		model:              model,
		processWg:          sync.WaitGroup{},
		closeTimeout:       c.timeouts.CloseTimeout,
		bufferSizes:        c.bufferSizes,
	}

	go grpcStream.readLoop()

	return grpcStream, nil
}

// GrpcChatCompletionStream represents a streaming chat completion via gRPC
type GrpcChatCompletionStream struct {
	stream             grpcClientStream
	converterHandle    *ffi.GrpcResponseConverterHandle
	batchPostprocessor *ffi.BatchPostprocessor
	batchSize          int
	ctx                context.Context
	cancel             context.CancelFunc
	closed             int32
	resultJSONChan     chan string
	errChan            chan error
	readLoopDone       chan struct{}
	requestID          string
	model              string
	processWg          sync.WaitGroup
	closeTimeout       time.Duration
	bufferSizes        ChannelBufferSizes
	clientDisconnected int32 // Atomic flag: 1 if client disconnected, 0 otherwise
}

func (s *GrpcChatCompletionStream) readLoop() {
	defer func() {
		atomic.StoreInt32(&s.closed, 1)
		s.processWg.Wait()
		close(s.resultJSONChan)
		close(s.errChan)
		close(s.readLoopDone)
		// Cancel context after channels are closed to ensure errors are read first
		if s.cancel != nil {
			s.cancel()
		}
	}()

	recvChan := make(chan recvResult, s.bufferSizes.RecvChan)
	const firstRecvTimeout = 60 * time.Second

	go func() {
		defer close(recvChan)
		recvCount := 0
		for {
			select {
			case <-s.ctx.Done():
				// Skip CloseSend() if client disconnected
				if atomic.LoadInt32(&s.clientDisconnected) == 0 {
					_ = s.stream.CloseSend()
				}
				return
			default:
			}

			recvCount++
			var protoResp *proto.GenerateResponse
			var err error

			// First Recv() with timeout
			if recvCount == 1 {
				recvDone := make(chan recvResult, 1)
				go func() {
					resp, recvErr := s.stream.Recv()
					recvDone <- recvResult{resp: resp, err: recvErr}
				}()

				select {
				case result := <-recvDone:
					protoResp = result.resp
					err = result.err
				case <-time.After(firstRecvTimeout):
					timeoutErr := fmt.Errorf("stream.Recv() timeout after %v: backend may not be responding (request_id=%s)", firstRecvTimeout, s.requestID)
					select {
					case recvChan <- recvResult{resp: nil, err: timeoutErr}:
					case <-s.ctx.Done():
					}
					return
				case <-s.ctx.Done():
					return
				}
			} else {
				// Normal Recv()
				protoResp, err = s.stream.Recv()
			}

			if err != nil {
				select {
				case recvChan <- recvResult{resp: nil, err: err}:
				case <-s.ctx.Done():
					return
				}
				return
			}

			select {
			case <-s.ctx.Done():
				// Skip CloseSend() if client disconnected
				if atomic.LoadInt32(&s.clientDisconnected) == 0 {
					_ = s.stream.CloseSend()
				}
				return
			case recvChan <- recvResult{resp: protoResp, err: nil}:
			}
		}
	}()

	for {
		select {
		case <-s.ctx.Done():
			// Skip CloseSend() if client disconnected
			if atomic.LoadInt32(&s.clientDisconnected) == 0 {
				_ = s.stream.CloseSend()
			}
			return
		case result, ok := <-recvChan:
			if !ok {
				return
			}
			if result.err != nil {
				if result.err == io.EOF {
					results, flushErr := s.flushBatch()
					if flushErr != nil {
						select {
						case s.errChan <- fmt.Errorf("failed to flush batch: %w", flushErr):
						case <-s.ctx.Done():
						}
						return
					}
					for _, resultJSON := range results {
						select {
						case s.resultJSONChan <- resultJSON:
						case <-s.ctx.Done():
							return
						}
					}
					return
				}
				select {
				case s.errChan <- result.err:
				case <-s.ctx.Done():
				}
				return
			}

			if result.resp != nil {
				s.processWg.Add(1)
				go func(resp *proto.GenerateResponse) {
					defer s.processWg.Done()
					s.processAndSendResponse(resp)
				}(result.resp)
			}
		}
	}
}

func (s *GrpcChatCompletionStream) processAndSendResponse(protoResp *proto.GenerateResponse) {
	select {
	case <-s.ctx.Done():
		return
	default:
	}

	if protoResp == nil {
		return
	}

	protoJSON, err := protoToJSON(protoResp)
	if err != nil {
		select {
		case s.errChan <- fmt.Errorf("failed to convert proto to JSON: %w", err):
		case <-s.ctx.Done():
		}
		return
	}

	if s.batchPostprocessor == nil {
		select {
		case s.errChan <- fmt.Errorf("batch postprocessor is nil"):
		case <-s.ctx.Done():
		}
		return
	}

	results, _, err := s.batchPostprocessor.AddChunk(protoJSON)
	if err != nil {
		select {
		case s.errChan <- fmt.Errorf("batch postprocessing failed: %w", err):
		case <-s.ctx.Done():
		}
		return
	}

	for _, resultJSON := range results {
		select {
		case s.resultJSONChan <- resultJSON:
		case <-s.ctx.Done():
			return
		}
	}
}

func (s *GrpcChatCompletionStream) RecvJSON() (string, error) {
	// Use a loop instead of recursion to avoid stack overflow if there are many empty strings
	for {
		// Check errChan first to prioritize actual errors over context cancellation
		select {
		case err, ok := <-s.errChan:
			if !ok {
				return "", io.EOF
			}
			return "", err
		default:
		}

		select {
		case resultJSON, ok := <-s.resultJSONChan:
			if !ok {
				return "", io.EOF
			}
			// Skip empty strings and continue loop instead of recursing
			if resultJSON != "" {
				return resultJSON, nil
			}
			// Empty string, continue loop to get next result
			continue
		case err, ok := <-s.errChan:
			if !ok {
				return "", io.EOF
			}
			return "", err
		case <-s.ctx.Done():
			return "", s.ctx.Err()
		}
	}
}

// SetClientDisconnected marks that the client has disconnected.
// When Close() is called, it will not call CloseSend() to avoid aborting the request on server side.
func (s *GrpcChatCompletionStream) SetClientDisconnected() {
	atomic.StoreInt32(&s.clientDisconnected, 1)
}

func (s *GrpcChatCompletionStream) Close() error {
	if !atomic.CompareAndSwapInt32(&s.closed, 0, 1) {
		return nil
	}

	if s.cancel != nil {
		s.cancel()
	}

	clientDisconnected := atomic.LoadInt32(&s.clientDisconnected) == 1

	select {
	case <-s.readLoopDone:
		// readLoop completed
	default:
		if !clientDisconnected {
			// Call CloseSend() if client didn't disconnect
			_ = s.stream.CloseSend()
		}
		select {
		case <-s.readLoopDone:
		case <-time.After(s.closeTimeout):
		}
	}

	_, _ = s.flushBatch()

	if s.converterHandle != nil {
		ffi.FreeGrpcResponseConverter(s.converterHandle)
	}

	return nil
}

func (s *GrpcChatCompletionStream) flushBatch() ([]string, error) {
	if s.batchPostprocessor != nil {
		results, err := s.batchPostprocessor.Flush()
		if err != nil {
			return nil, fmt.Errorf("batch flush failed: %w", err)
		}
		return results, nil
	}
	return nil, nil
}

func protoToJSON(resp *proto.GenerateResponse) (string, error) {
	var sb strings.Builder
	sb.Grow(500)

	sb.WriteString(`{"request_id":`)
	if resp.RequestId == "" {
		sb.WriteString(`""`)
	} else {
		requestIDJSON, err := json.Marshal(resp.RequestId)
		if err != nil {
			return "", err
		}
		sb.Write(requestIDJSON)
	}

	switch r := resp.Response.(type) {
	case *proto.GenerateResponse_Chunk:
		sb.WriteString(`,"chunk":{`)
		sb.WriteString(`"token_ids":`)
		tokenIDsJSON, err := json.Marshal(r.Chunk.TokenIds)
		if err != nil {
			return "", err
		}
		sb.Write(tokenIDsJSON)
		sb.WriteString(`,"prompt_tokens":`)
		sb.WriteString(strconv.FormatInt(int64(r.Chunk.PromptTokens), 10))
		sb.WriteString(`,"completion_tokens":`)
		sb.WriteString(strconv.FormatInt(int64(r.Chunk.CompletionTokens), 10))
		sb.WriteString(`,"cached_tokens":`)
		sb.WriteString(strconv.FormatInt(int64(r.Chunk.CachedTokens), 10))
		sb.WriteString(`,"index":`)
		sb.WriteString(strconv.FormatInt(int64(r.Chunk.Index), 10))
		sb.WriteString(`}`)
	case *proto.GenerateResponse_Complete:
		sb.WriteString(`,"complete":{`)
		sb.WriteString(`"output_ids":`)
		outputIDsJSON, err := json.Marshal(r.Complete.OutputIds)
		if err != nil {
			return "", err
		}
		sb.Write(outputIDsJSON)
		sb.WriteString(`,"finish_reason":`)
		finishReasonJSON, err := json.Marshal(r.Complete.FinishReason)
		if err != nil {
			return "", err
		}
		sb.Write(finishReasonJSON)
		sb.WriteString(`,"prompt_tokens":`)
		sb.WriteString(strconv.FormatInt(int64(r.Complete.PromptTokens), 10))
		sb.WriteString(`,"completion_tokens":`)
		sb.WriteString(strconv.FormatInt(int64(r.Complete.CompletionTokens), 10))
		sb.WriteString(`,"cached_tokens":`)
		sb.WriteString(strconv.FormatInt(int64(r.Complete.CachedTokens), 10))
		sb.WriteString(`}`)
	case *proto.GenerateResponse_Error:
		sb.WriteString(`,"error":{`)
		sb.WriteString(`"message":`)
		messageJSON, err := json.Marshal(r.Error.Message)
		if err != nil {
			return "", err
		}
		sb.Write(messageJSON)
		sb.WriteString(`,"http_status_code":`)
		httpStatusCodeJSON, err := json.Marshal(r.Error.HttpStatusCode)
		if err != nil {
			return "", err
		}
		sb.Write(httpStatusCodeJSON)
		if r.Error.Details != "" {
			sb.WriteString(`,"details":`)
			detailsJSON, err := json.Marshal(r.Error.Details)
			if err != nil {
				return "", err
			}
			sb.Write(detailsJSON)
		}
		sb.WriteString(`}`)
	}

	sb.WriteString(`}`)
	return sb.String(), nil
}

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
