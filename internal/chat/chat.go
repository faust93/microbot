package chat

import "time"

// Inbound represents an incoming message to the agent.
type Inbound struct {
	Channel   string
	SenderID  string
	ChatID    string
	Content   string
	Timestamp time.Time
	Media     []string
	Metadata  map[string]interface{}
}

// Outbound represents a message produced by the agent.
type Outbound struct {
	Channel  string
	ChatID   string
	Content  string
	ReplyTo  string
	Media    []string
	Metadata map[string]interface{}
}

// Hub provides simple buffered channels for inbound/outbound messages.
type Hub struct {
	In  chan Inbound
	Out chan Outbound

	TelegramOut chan Outbound
	NtfyOut     chan Outbound
}

// NewHub constructs a new Hub with the given buffer size.
func NewHub(buffer int) *Hub {
	return &Hub{
		In:          make(chan Inbound, buffer),
		Out:         make(chan Outbound, buffer),
		TelegramOut: make(chan Outbound, buffer),
		NtfyOut:     make(chan Outbound, buffer),
	}
}

// Close closes the channels.
func (h *Hub) Close() {
	close(h.In)
	close(h.Out)
	close(h.TelegramOut)
	close(h.NtfyOut)
}
