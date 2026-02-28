package session

import (
	"encoding/json"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// MaxHistorySize is the maximum number of messages kept in a session.
// Older messages are trimmed and saved to persistent memory. This keeps the in-memory session small and focused on recent context.
// Important information should be persisted via write_memory, not session history.
const MaxHistorySize = 50

type Message struct {
	Role      string `json:"role"` // "user", "assistant", "tool"
	Content   string `json:"content"`
	Timestamp string `json:"timestamp"` // RFC3339 format
}

// Session holds a short chat history.
type Session struct {
	Key     string
	History []*Message
}

// SessionManager stores sessions in memory and persists to disk under workspace.
type SessionManager struct {
	mu        sync.RWMutex
	sessions  map[string]*Session
	workspace string
}

func NewSessionManager(workspace string) *SessionManager {
	return &SessionManager{sessions: make(map[string]*Session), workspace: workspace}
}

func (sm *SessionManager) GetOrCreate(key string) *Session {
	sm.mu.Lock()
	defer sm.mu.Unlock()
	if s, ok := sm.sessions[key]; ok {
		return s
	}
	s := &Session{Key: key, History: make([]*Message, 0)}
	sm.sessions[key] = s
	return s
}

func (sm *SessionManager) Save(s *Session) error {
	sm.mu.Lock()
	defer sm.mu.Unlock()
	// Trim history to the most recent messages
	s.Trim()
	path := filepath.Join(sm.workspace, "sessions")
	os.MkdirAll(path, 0755)
	fpath := filepath.Join(path, s.Key+".json")
	b, err := json.MarshalIndent(s, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(fpath, b, 0644)
}

func (sm *SessionManager) LoadAll() error {
	sm.mu.Lock()
	defer sm.mu.Unlock()
	path := filepath.Join(sm.workspace, "sessions")
	_ = os.MkdirAll(path, 0755)
	entries, err := os.ReadDir(path)
	if err != nil {
		return err
	}
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		b, err := os.ReadFile(filepath.Join(path, e.Name()))
		if err != nil {
			continue
		}
		var s Session
		if err := json.Unmarshal(b, &s); err != nil {
			continue
		}
		sm.sessions[s.Key] = &s
	}
	return nil
}

func (sm *SessionManager) TrimAll() []*Message {
	sm.mu.Lock()
	defer sm.mu.Unlock()
	var trimmedMsg []*Message
	for _, s := range sm.sessions {
		trimmed := s.Trim()
		if len(trimmed) > 0 {
			trimmedMsg = append(trimmedMsg, trimmed...)
		}
	}
	return trimmedMsg
}

func (s *Session) AddMessage(role, content string) {
	msg := &Message{
		Role:      role,
		Content:   content,
		Timestamp: time.Now().Format(time.RFC3339),
	}
	s.History = append(s.History, msg)
}

// GetHistory returns the session history.
func (s *Session) GetHistory() []*Message {
	return s.History
}

// trim keeps only the last MaxHistorySize messages, discarding the oldest.
func (s *Session) Trim() []*Message {
	if len(s.History) > MaxHistorySize {
		trimmed := s.History[:len(s.History)-MaxHistorySize]
		s.History = s.History[len(s.History)-MaxHistorySize:]
		return trimmed
	}
	return nil
}
