package providers

import "context"

// Message represents a chat message to/from the LLM.
type Message struct {
	Role       string     `json:"role"` // "system" | "user" | "assistant" | "tool"
	Content    string     `json:"content"`
	ToolCallID string     `json:"tool_call_id,omitempty"` // set when Role == "tool"
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`   // set on assistant msgs with tool calls
}

// ToolDefinition is a lightweight description of a tool available to the model.
type ToolDefinition struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Parameters  map[string]interface{} `json:"parameters,omitempty"`
}

// ToolCall represents a request from the LLM to invoke a tool.
type ToolCall struct {
	ID        string                 `json:"id"`
	Name      string                 `json:"name"`
	Arguments map[string]interface{} `json:"arguments"`
}

// LLMResponse is a normalized response from a provider.
type LLMResponse struct {
	Content      string     `json:"content"`
	HasToolCalls bool       `json:"hasToolCalls"`
	ToolCalls    []ToolCall `json:"toolCalls,omitempty"`
}

// LLMProvider is the interface used by the agent loop to call LLMs.
type LLMProvider interface {
	// Chat sends messages to the model and returns a normalized response.
	Chat(ctx context.Context, messages []Message, tools []ToolDefinition, model string, temperature float64, maxTokens int) (LLMResponse, error)

	// GetDefaultModel returns the provider's default model string.
	GetDefaultModel() string
}
