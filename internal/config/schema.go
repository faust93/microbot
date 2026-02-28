package config

// Config holds picobot configuration (minimal for v0).
type Config struct {
	Agents    AgentsConfig    `json:"agents"`
	Channels  ChannelsConfig  `json:"channels"`
	Providers ProvidersConfig `json:"providers"`
	Memory    MemoryConfig    `json:"memory"`
	Tools     ToolsConfig     `json:"tools"`
}

type AgentsConfig struct {
	Defaults AgentDefaults `json:"defaults"`
}

type AgentDefaults struct {
	Workspace          string  `json:"workspace"`
	Model              string  `json:"model"`
	MaxTokens          int     `json:"maxTokens"`
	Temperature        float64 `json:"temperature"`
	MaxToolIterations  int     `json:"maxToolIterations"`
	HeartbeatIntervalS int     `json:"heartbeatIntervalS"`
}

type ChannelsConfig struct {
	Telegram TelegramConfig `json:"telegram"`
	Ntfy     NtfyConfig     `json:"ntfy"`
}

type TelegramConfig struct {
	Enabled   bool     `json:"enabled"`
	Token     string   `json:"token"`
	AllowFrom []string `json:"allowFrom"`
}

type NtfyConfig struct {
	Enabled bool   `json:"enabled"`
	Token   string `json:"token"`
	Server  string `json:"server"`
	Topic   string `json:"topic"`
}

type ProvidersConfig struct {
	OpenAI *ProviderConfig `json:"openai,omitempty"`
}

type ProviderConfig struct {
	APIKey  string `json:"apiKey"`
	APIBase string `json:"apiBase"`
	Timeout int    `json:"timeout"` // to prevent provider timeouts for long-running tool calls
}

type ToolsConfig struct {
	MCP *MCPConfig `json:"mcp,omitempty"`
}

type MCPConfig struct {
	Enabled bool                       `json:"enabled"`
	Servers map[string]MCPServerConfig `json:"servers"`
}

type MCPServerConfig struct {
	Transport string            `json:"transport"`
	Command   string            `json:"command,omitempty"`
	Args      []string          `json:"args,omitempty"`
	URL       string            `json:"url,omitempty"`
	Headers   map[string]string `json:"headers,omitempty"`
}

type MemoryConfig struct {
	Enabled           bool    `json:"enabled"`
	EmbedType         string  `json:"embedType"`                   // e.g., "onnx"
	DbPath            string  `json:"dbPath,omitempty"`            // path to SQLite db file (if using SQLite-backed memory)
	ONNXModelPath     string  `json:"onnxModelPath,omitempty"`     // path to ONNX model file (if EmbedType is "onnx")
	ONNXTokenizerPath string  `json:"onnxTokenizerPath,omitempty"` // path to tokenizer file (if needed by the ONNX model)
	Threshold         float32 `json:"threshold,omitempty"`         // number of similar items to retrieve in QueryHistory
	TopK              int     `json:"topK,omitempty"`              // max number of items to return in QueryHistory
}
