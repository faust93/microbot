package onnx

import (
	"context"
	"fmt"
	"math"
	"sync"
	"time"
)

type ModelConfig struct {
	Name                string
	Path                string
	TokenizerPath       string
	BatchSize           int
	NormalizeEmbeddings bool
	MaxTokens           int
	PoolingStrategy     string // cls, mean, max
	ONNXRuntime         string // onnxruntime.so path
}

// ONNXEmbeddingEngine implements embedding generation using ONNX Runtime
type ONNXEmbeddingEngine struct {
	modelName string
	modelPath string
	config    *ModelConfig
	dimension int
	tokenizer *Tokenizer
	session   ONNXSession // Interface to allow mocking
	stats     *InferenceStats
	mutex     sync.RWMutex
}

// ONNXSession interface allows for testing and mocking
type ONNXSession interface {
	Run(inputs []ONNXValue) ([]ONNXValue, error)
	GetInputCount() int
	GetOutputCount() int
	GetInputName(index int) string
	GetOutputName(index int) string
	Destroy()
}

// ONNXValue interface for ONNX tensors
type ONNXValue interface {
	GetData() interface{}
	GetShape() []int64
	Destroy()
}

// InferenceStats tracks performance metrics
type InferenceStats struct {
	TotalInferences int64
	TotalTokens     int64
	AverageLatency  time.Duration
	P95Latency      time.Duration
	ErrorRate       float64
	ThroughputTPS   float64
	RecentLatencies []time.Duration
	TotalErrors     int64
	mutex           sync.RWMutex
}

// NewONNXEmbeddingEngine creates a new ONNX-based embedding engine
func NewONNXEmbeddingEngine(config *ModelConfig) (*ONNXEmbeddingEngine, error) {
	if config.ONNXRuntime == "" {
		config.ONNXRuntime = "/usr/lib/libonnxruntime.so" // Default path
	}
	if config.BatchSize <= 0 {
		config.BatchSize = 32
	}
	if config.MaxTokens <= 0 {
		config.MaxTokens = 512
	}
	if config.PoolingStrategy == "" {
		config.PoolingStrategy = "cls"
	}
	engine := &ONNXEmbeddingEngine{
		modelName: config.Name,
		modelPath: config.Path,
		config:    config,
		stats:     NewInferenceStats(),
	}

	// Initialize tokenizer
	tokenizer, err := NewTokenizerWithConfig(config.TokenizerPath)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize tokenizer: %v", err)
	}
	engine.tokenizer = tokenizer

	// Load ONNX session
	session, err := engine.createSession()
	if err != nil {
		return nil, err // Error already wrapped in createSession
	}
	engine.session = session

	return engine, nil
}

// createSession creates an ONNX Runtime session
func (e *ONNXEmbeddingEngine) createSession() (ONNXSession, error) {
	// Try to create a real ONNX Runtime session
	session, err := NewRealONNXSession(e.modelPath, e.config.ONNXRuntime)
	if err != nil {
		return nil, fmt.Errorf("failed to create session: %v", err)
	}

	return session, nil
}

// Embed generates embeddings for the given content
func (e *ONNXEmbeddingEngine) Embed(ctx context.Context, content []string) ([][]float32, error) {

	// Validate input
	if len(content) == 0 {
		return nil, fmt.Errorf("no content provided")
	}

	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	// Check if session is initialized
	if e.session == nil {
		return nil, fmt.Errorf("session not initialized")
	}

	start := time.Now()
	defer func() {
		e.stats.RecordInference(len(content), time.Since(start))
	}()

	// Tokenize inputs
	tokens, err := e.tokenizer.TokenizeBatch(content, e.config.MaxTokens)
	if err != nil {
		e.stats.RecordError()
		return nil, fmt.Errorf("tokenization failed %v", err)
	}

	// Create input tensors
	inputs, err := e.createInputTensors(tokens)
	if err != nil {
		e.stats.RecordError()
		return nil, fmt.Errorf("failed to create input tensors %v", err)
	}
	defer func() {
		for _, input := range inputs {
			input.Destroy()
		}
	}()

	// Run inference
	outputs, err := e.session.Run(inputs)
	if err != nil {
		e.stats.RecordError()
		return nil, fmt.Errorf("inference failed %v", err)
	}
	defer func() {
		for _, output := range outputs {
			output.Destroy()
		}
	}()

	// Extract embeddings
	embeddings, err := e.extractEmbeddings(outputs[0])
	if err != nil {
		e.stats.RecordError()
		return nil, fmt.Errorf("failed to extract embeddings %v", err)
	}

	// Normalize if configured
	if e.config.NormalizeEmbeddings {
		embeddings = normalizeEmbeddings(embeddings)
	}

	return embeddings, nil
}

// EmbedBatch processes content in batches for optimal performance
func (e *ONNXEmbeddingEngine) EmbedBatch(ctx context.Context, content []string, batchSize int) ([][]float32, error) {
	if batchSize <= 0 {
		batchSize = e.config.BatchSize
		if batchSize <= 0 {
			batchSize = 32 // Default batch size
		}
	}

	var allEmbeddings [][]float32

	for i := 0; i < len(content); i += batchSize {

		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		end := i + batchSize
		if end > len(content) {
			end = len(content)
		}

		batch := content[i:end]
		embeddings, err := e.Embed(ctx, batch)
		if err != nil {
			return nil, fmt.Errorf("batch processing failed at index %d: %w", i, err)
		}

		allEmbeddings = append(allEmbeddings, embeddings...)
	}

	return allEmbeddings, nil
}

// Close releases model resources
func (e *ONNXEmbeddingEngine) Close() error {
	e.mutex.Lock()
	defer e.mutex.Unlock()

	if e.session != nil {
		e.session.Destroy()
		e.session = nil
	}

	if e.tokenizer != nil {
		e.tokenizer.Close()
		e.tokenizer = nil
	}

	return nil
}

// createInputTensors creates ONNX tensors from tokenized input, including attention masks
func (e *ONNXEmbeddingEngine) createInputTensors(tokens [][]int64) ([]ONNXValue, error) {
	if len(tokens) == 0 {
		return nil, fmt.Errorf("no tokens provided")
	}

	var inputs []ONNXValue
	cleanup := func() {
		for _, input := range inputs {
			input.Destroy()
		}
	}
	// Use the real tensor creation if we have a real session
	if realSession, ok := e.session.(*RealONNXSession); ok {
		// Create input_ids tensor
		if realSession.GetInputCount() > 0 {
			inputName := realSession.GetInputName(0)
			inputTensor, err := CreateInputTensorFromTokens(tokens, inputName)
			if err != nil {
				cleanup()
				return nil, fmt.Errorf("failed to create input_ids tensor: %w", err)
			}
			inputs = append(inputs, inputTensor)
		}

		// Create attention_mask tensor if the model expects it
		if realSession.GetInputCount() > 1 {
			// Generate attention masks (1 for real tokens, 0 for padding)
			masks := e.generateAttentionMasks(tokens)
			maskTensor, err := CreateAttentionMaskTensor(masks)
			if err != nil {
				cleanup()
				return nil, fmt.Errorf("failed to create attention_mask tensor: %w", err)
			}
			inputs = append(inputs, maskTensor)
		}

		// Create token_type_ids tensor if the model expects it (3rd input)
		if realSession.GetInputCount() > 2 {
			batchSize := len(tokens)
			seqLen := len(tokens[0])
			tokenTypeTensor, err := CreateTokenTypeIdsTensor(batchSize, seqLen)
			if err != nil {
				cleanup()
				return nil, fmt.Errorf("failed to create token_type_ids tensor: %w", err)
			}
			inputs = append(inputs, tokenTypeTensor)
		}

		return inputs, nil
	}

	return nil, fmt.Errorf("input sensors error")
}

// generateAttentionMasks creates attention masks for the tokenized input
func (e *ONNXEmbeddingEngine) generateAttentionMasks(tokens [][]int64) [][]int64 {
	masks := make([][]int64, len(tokens))

	for i, seq := range tokens {
		mask := make([]int64, len(seq))
		for j, token := range seq {
			if token != 0 { // Assuming 0 is the padding token
				mask[j] = 1
			} else {
				mask[j] = 0
			}
		}
		masks[i] = mask
	}

	return masks
}

// extractEmbeddings extracts embeddings from ONNX output tensor
func (e *ONNXEmbeddingEngine) extractEmbeddings(output ONNXValue) ([][]float32, error) {
	// Use real extraction if we have a real tensor
	if realTensor, ok := output.(*RealONNXTensor); ok {
		// Use the configured pooling strategy, defaulting to CLS token
		poolingStrategy := e.config.PoolingStrategy
		if poolingStrategy == "" {
			poolingStrategy = "cls" // Default to CLS token pooling
		}

		embeddings, err := ExtractEmbeddingsFromTensor(realTensor, poolingStrategy)
		if err != nil {
			return nil, err
		}

		// Update model info with discovered dimension
		if len(embeddings) > 0 && e.dimension == 0 {
			e.dimension = len(embeddings[0])
		}

		return embeddings, nil
	}

	return nil, fmt.Errorf("output tensor error")
}

// normalizeEmbeddings normalizes embeddings to unit length
func normalizeEmbeddings(embeddings [][]float32) [][]float32 {
	for i, embedding := range embeddings {
		var norm float32
		for _, val := range embedding {
			norm += val * val
		}
		norm = float32(math.Sqrt(float64(norm)))

		if norm > 0 {
			for j := range embedding {
				embeddings[i][j] = embedding[j] / norm
			}
		}
	}
	return embeddings
}

// GetStats returns inference statistics
func (e *ONNXEmbeddingEngine) GetStats() *InferenceStats {
	e.mutex.RLock()
	defer e.mutex.RUnlock()
	statsCopy := *e.stats
	return &statsCopy
}

// NewInferenceStats creates a new inference statistics tracker
func NewInferenceStats() *InferenceStats {
	return &InferenceStats{
		RecentLatencies: make([]time.Duration, 0, 100),
	}
}

// RecordInference records an inference operation
func (s *InferenceStats) RecordInference(tokenCount int, latency time.Duration) {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	s.TotalInferences++
	s.TotalTokens += int64(tokenCount)

	// Update recent latencies
	s.RecentLatencies = append(s.RecentLatencies, latency)
	if len(s.RecentLatencies) > 100 {
		s.RecentLatencies = s.RecentLatencies[1:]
	}

	// Update statistics
	s.updateStats()
}

// RecordError records an inference error
func (s *InferenceStats) RecordError() {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	s.TotalErrors++
	s.updateErrorRate()
}

func (s *InferenceStats) updateStats() {
	if len(s.RecentLatencies) == 0 {
		return
	}

	// Calculate average and P95 in one pass
	var total time.Duration
	var p95Index int

	if len(s.RecentLatencies) >= 20 {
		// For larger windows, estimate P95 without full sort
		p95Index = len(s.RecentLatencies) * 95 / 100
	}

	for i, lat := range s.RecentLatencies {
		total += lat
		if i == p95Index && p95Index > 0 {
			s.P95Latency = lat
		}
	}

	s.AverageLatency = total / time.Duration(len(s.RecentLatencies))

	// Calculate throughput
	if s.AverageLatency > 0 {
		s.ThroughputTPS = float64(time.Second) / float64(s.AverageLatency)
	}
}

func (s *InferenceStats) updateErrorRate() {
	if s.TotalInferences > 0 {
		s.ErrorRate = float64(s.TotalErrors) / float64(s.TotalInferences)
	}
}
