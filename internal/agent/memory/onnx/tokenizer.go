package onnx

import (
	"fmt"
	"regexp"
	"strings"
	"unicode"

	"github.com/sugarme/tokenizer"
	"github.com/sugarme/tokenizer/pretrained"
)

// Tokenizer handles text tokenization for embedding models
type Tokenizer struct {
	modelName string
	maxLength int
	tokenizer *tokenizer.Tokenizer
}

func NewTokenizerWithConfig(vocabPath string) (*Tokenizer, error) {
	if vocabPath == "" {
		vocabPath = "tokenizer.json" // Default path
	}

	tokenizer := &Tokenizer{
		modelName: "unknown_model",
		maxLength: 512,
	}

	tk, err := pretrained.FromFile(vocabPath)
	if err != nil {
		return nil, fmt.Errorf("%w", err)
	}

	tokenizer.tokenizer = tk

	return tokenizer, nil
}

// TokenizeBatch tokenizes a batch of texts
func (t *Tokenizer) TokenizeBatch(texts []string, maxLength int) ([][]int64, error) {
	if maxLength <= 0 {
		maxLength = t.maxLength
	}

	tokens := make([][]int64, len(texts))
	for i, text := range texts {
		tokenized, err := t.Tokenize(text, maxLength)
		if err != nil {
			return nil, fmt.Errorf("failed to tokenize text %d: %w", i, err)
		}
		tokens[i] = tokenized
	}

	return tokens, nil
}

// Tokenize converts text to token IDs
func (t *Tokenizer) Tokenize(text string, maxLength int) ([]int64, error) {
	if maxLength <= 0 {
		maxLength = t.maxLength
	}
	if maxLength <= 0 {
		return nil, fmt.Errorf("maxLength must be > 0")
	}

	// Basic preprocessing
	text = t.preprocess(text)

	// Convert words to token IDs
	var tokens []int64

	en, err := t.tokenizer.EncodeSingle(text, true)
	if err != nil {
		return nil, err
	}

	// Convert each int to int64 and append
	for _, token := range en.Ids {
		tokens = append(tokens, int64(token))
	}

	for len(tokens) < maxLength {
		tokens = append(tokens, 0)
	}

	return tokens, nil
}

// preprocess performs basic text preprocessing
func (t *Tokenizer) preprocess(text string) string {
	// Convert to lowercase
	text = strings.ToLower(text)

	// Remove extra whitespace
	text = strings.TrimSpace(text)
	text = regexp.MustCompile(`\s+`).ReplaceAllString(text, " ")

	// Remove special characters |, -, *, `
	// text = regexp.MustCompile(`[|\-*/”:–“_)()]+`).ReplaceAllString(text, "")

	return text
}

// basicTokenize performs basic word tokenization
func (t *Tokenizer) basicTokenize(text string) []string {
	// Split on whitespace and punctuation
	words := make([]string, 0)
	current := make([]rune, 0)

	for _, r := range text {
		if unicode.IsSpace(r) || unicode.IsPunct(r) {
			if len(current) > 0 {
				words = append(words, string(current))
				current = current[:0]
			}
			if unicode.IsPunct(r) {
				words = append(words, string(r))
			}
		} else {
			current = append(current, r)
		}
	}

	if len(current) > 0 {
		words = append(words, string(current))
	}

	return words
}

// Close releases tokenizer resources
func (t *Tokenizer) Close() error {
	return nil
}

// GetModelName returns the tokenizer's model name
func (t *Tokenizer) GetModelName() string {
	return t.modelName
}

// SetMaxLength sets the maximum sequence length
func (t *Tokenizer) SetMaxLength(maxLength int) {
	t.maxLength = maxLength
}
