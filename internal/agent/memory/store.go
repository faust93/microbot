package memory

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// MemoryItem is a stored memory entry.
// Kind is "short" or "long". Timestamp is in UTC.
type MemoryItem struct {
	Kind       string
	Role       string
	Text       string
	Similarity float64
	Timestamp  string
}

// MemoryStore is a minimal in-memory memory system with simple query capabilities.
// - Long-term: append-only list (persisted in a real implementation)
// - Short-term: append-only list with a configurable limit (recent items kept)
// This is intentionally simple for v0 and unit-testable.
type MemoryStore struct {
	workspace string // workspace root (used for disk-backed memory)
	memoryDir string // workspace/memory/
	limit     int    // max short-term items to keep
	mu        sync.RWMutex
}

// NewMemoryStore creates an in-memory store with short-term limit (e.g., 100).
// Kept for tests and simple use-cases.
func NewMemoryStore(limit int) *MemoryStore {
	return NewMemoryStoreWithWorkspace(".", limit)
}

// NewMemoryStoreWithWorkspace creates a MemoryStore backed by files under workspace/memory/.
func NewMemoryStoreWithWorkspace(workspace string, limit int) *MemoryStore {
	if limit <= 0 {
		limit = 100
	}
	ms := &MemoryStore{
		workspace: workspace,
		memoryDir: workspace + "/memory",
		limit:     limit,
	}
	// ensure memory directory exists
	_ = os.MkdirAll(ms.memoryDir, 0o755)
	return ms
}

// ReadLongTerm reads the long-term MEMORY.md file under workspace/memory/MEMORY.md
func (s *MemoryStore) ReadLongTerm() (string, error) {
	path := filepath.Join(s.memoryDir, "MEMORY.md")
	b, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return "", nil
		}
		return "", err
	}
	return string(b), nil
}

// WriteLongTerm writes content to MEMORY.md (overwrites).
func (s *MemoryStore) WriteLongTerm(content string) error {
	if err := os.MkdirAll(s.memoryDir, 0o755); err != nil {
		return err
	}
	path := filepath.Join(s.memoryDir, "MEMORY.md")
	return os.WriteFile(path, []byte(content), 0o644)
}

// ReadToday reads today's memory note file (YYYY-MM-DD.md)
func (s *MemoryStore) ReadToday() (string, error) {
	name := time.Now().UTC().Format("2006-01-02") + ".md"
	path := filepath.Join(s.memoryDir, name)
	b, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return "", nil
		}
		return "", err
	}
	return string(b), nil
}

// AppendToday appends a line (with timestamp) to today's memory note file.
func (s *MemoryStore) AppendToday(text string) error {
	if err := os.MkdirAll(s.memoryDir, 0o755); err != nil {
		return err
	}
	name := time.Now().UTC().Format("2006-01-02") + ".md"
	path := filepath.Join(s.memoryDir, name)
	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = fmt.Fprintf(f, "[%s] %s\n", time.Now().UTC().Format(time.RFC3339), text)
	return err
}

// GetRecentMemories reads last N days' files and joins them with separators.
func (s *MemoryStore) GetRecentMemories(days int) (string, error) {
	if days <= 0 {
		days = 1
	}
	parts := make([]string, 0, days)
	for i := 0; i < days; i++ {
		d := time.Now().UTC().AddDate(0, 0, -i)
		name := d.Format("2006-01-02") + ".md"
		path := filepath.Join(s.memoryDir, name)
		b, err := os.ReadFile(path)
		if err != nil {
			if os.IsNotExist(err) {
				continue
			}
			return "", err
		}
		parts = append(parts, string(b))
	}
	return strings.Join(parts, "\n---\n"), nil
}

// GetMemoryContext returns combined long-term memory + today's notes for the system prompt.
func (s *MemoryStore) GetMemoryContext() (string, error) {
	lt, err := s.ReadLongTerm()
	if err != nil {
		return "", err
	}
	td, err := s.ReadToday()
	if err != nil {
		return "", err
	}
	if lt == "" && td == "" {
		return "", nil
	}
	if lt == "" {
		return td, nil
	}
	if td == "" {
		return lt, nil
	}
	return lt + "\n\n---\n\n" + td, nil
}
