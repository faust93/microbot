package memory

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/local/picobot/internal/agent/memory/onnx"
)

// ONNXEmbedder uses an ONNX model to generate embeddings for memory items.
type ONNXEmbedder struct {
	ctx           context.Context
	engine        *onnx.ONNXEmbeddingEngine
	chunkMaxWords int
}

// NewONNXEmbedder constructs an ONNXEmbedder by loading the model from the given path.
func NewONNXEmbedder(config *onnx.ModelConfig) (*ONNXEmbedder, error) {
	var onnxemb ONNXEmbedder
	engine, err := onnx.NewONNXEmbeddingEngine(config)
	if err != nil {
		return nil, err
	}

	onnxemb.engine = engine
	onnxemb.chunkMaxWords = 200 // Adjust as needed based on model/tokenizer limits
	onnxemb.ctx = context.Background()

	return &onnxemb, nil
}

func (e *ONNXEmbedder) Close() error {
	return e.engine.Close()
}

// Embed implements the Embedder interface. It generates embeddings for the given memory items.
func (e *ONNXEmbedder) Embed(text string) ([]float32, error) {
	chunks := splitIntoChunks(text, e.chunkMaxWords) // Adjust chunk size as needed

	vec, err := e.engine.EmbedBatch(e.ctx, chunks, 0)
	if err != nil {
		return nil, fmt.Errorf("failed to generate embedding for text: %w", err)
	}
	if len(vec) == 0 || len(vec[0]) == 0 {
		return nil, errors.New("no embeddings found in the response")
	}

	averaged := averageEmbeddings(vec)
	return averaged, nil
}

// averageEmbeddings takes multiple embeddings and returns their mean.
// All embeddings must have the same dimension.
func averageEmbeddings(embeddings [][]float32) []float32 {
	if len(embeddings) == 0 {
		return nil
	}

	if len(embeddings) == 1 {
		// Single embedding; just return a copy
		result := make([]float32, len(embeddings[0]))
		copy(result, embeddings[0])
		return result
	}

	dim := len(embeddings[0])
	result := make([]float32, dim)

	// Sum all embeddings
	for _, emb := range embeddings {
		if len(emb) != dim {
			fmt.Printf("warning: embedding dimension mismatch: expected %d, got %d\n", dim, len(emb))
			continue
		}
		for i, val := range emb {
			result[i] += val
		}
	}

	// Divide by count to get the mean
	n := float32(len(embeddings))
	for i := range result {
		result[i] /= n
	}

	return result
}

func splitIntoChunks(text string, maxWords int) []string {
	words := strings.Fields(text)
	chunks := []string{}
	for i := 0; i < len(words); i += maxWords {
		end := i + maxWords
		if end > len(words) {
			end = len(words)
		}
		chunk := strings.Join(words[i:end], " ")
		chunks = append(chunks, chunk)
	}
	return chunks
}
