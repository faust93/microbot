package providers

import (
	"context"
	"fmt"
)

// StubProvider is a simple provider useful for local testing. It echoes back the last user message.
type StubProvider struct{}

func NewStubProvider() *StubProvider { return &StubProvider{} }

func (p *StubProvider) Chat(ctx context.Context, messages []Message, tools []ToolDefinition, model string, temperature float64, maxTokens int) (LLMResponse, error) {
	// Find last user message
	last := ""
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role == "user" {
			last = messages[i].Content
			break
		}
	}
	if last == "" {
		return LLMResponse{Content: "(stub) Hello from StubProvider"}, nil
	}
	return LLMResponse{Content: fmt.Sprintf("(stub) Echo: %s", last)}, nil
}

func (p *StubProvider) GetDefaultModel() string { return "stub-model" }
