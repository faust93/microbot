package agent

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/local/picobot/internal/agent/memory"
	"github.com/local/picobot/internal/agent/skills"
	"github.com/local/picobot/internal/providers"
	"github.com/local/picobot/internal/session"
)

// ContextBuilder builds messages for the LLM from session history and current message.
type ContextBuilder struct {
	workspace    string
	ranker       memory.Ranker
	topK         int
	skillsLoader *skills.Loader
}

func NewContextBuilder(workspace string, r memory.Ranker, topK int) *ContextBuilder {
	return &ContextBuilder{
		workspace:    workspace,
		ranker:       r,
		topK:         topK,
		skillsLoader: skills.NewLoader(workspace),
	}
}

func (cb *ContextBuilder) BuildMessages(history []*session.Message, currentMessage string, channel, chatID string, memoryContext string, memories []memory.MemoryItem) []providers.Message {
	msgs := make([]providers.Message, 0, len(history)+8)
	// system prompt
	system := "You are Picobot, a helpful assistant.\n\n"

	time_now := time.Now().Format("2006-01-02 15:04 (Monday)")
	tmpl := `## Current Time
%s

## Workspace
Your workspace is at: %s
- Memory files: %s/memory/MEMORY.md
- Daily notes: %s/memory/YYYY-MM-DD.md
- Custom skills: %s/skills/{skill-name}/SKILL.md

IMPORTANT: For normal conversation, just respond with text - do not call the message tool!
Only use the 'message' tool when you need to send a message to a specific chat channel.

## Current Session
Channel: %s
Chat ID: %s
`
	system = system + fmt.Sprintf(tmpl, time_now, cb.workspace, cb.workspace, cb.workspace, cb.workspace, channel, chatID) + "\n\n"

	// Load workspace bootstrap files (SOUL.md, AGENTS.md, USER.md, TOOLS.md)
	// These define the agent's personality, instructions, and available tools documentation.
	bootstrapFiles := []string{"SOUL.md", "AGENTS.md", "USER.md", "TOOLS.md"}
	for _, name := range bootstrapFiles {
		p := filepath.Join(cb.workspace, name)
		data, err := os.ReadFile(p)
		if err != nil {
			continue // file may not exist yet, skip silently
		}
		content := strings.TrimSpace(string(data))
		if content != "" {
			system = system + fmt.Sprintf("## %s\n\n%s", name, content)
		}
	}

	// instruction for memory tool usage
	system = system + "Always be helpful, accurate, and concise. If you decide something should be remembered, call the tool 'write_memory' with JSON arguments: {\"target\": \"today\"|\"long\", \"content\": \"...\", \"append\": true|false}. Use a tool call rather than plain chat text when writing memory.\n\n"

	// Load and include skills context
	loadedSkills, err := cb.skillsLoader.LoadAll()
	if err != nil {
		log.Printf("error loading skills: %v", err)
	}
	if len(loadedSkills) > 0 {
		var sb strings.Builder
		sb.WriteString("# Skills\n\n")
		sb.WriteString("The following skills extend your capabilities. To use a skill, read it using read_skill tool.\n\n")
		sb.WriteString("<skills>\n")
		for _, skill := range loadedSkills {
			sb.WriteString(" <skill>\n")
			sb.WriteString(fmt.Sprintf("  <name>%s</name>\n", skill.Name))
			sb.WriteString(fmt.Sprintf("  <description>%s</description>\n", skill.Description))
			sb.WriteString(" </skill>\n")
		}
		sb.WriteString("</skills>\n\n")
		system = system + sb.String()
	}

	// include file-based memory context (long-term + today's notes) if present
	if memoryContext != "" {
		system = system + "Memory:\n" + memoryContext
	}

	// select top-K memories using ranker if available
	selected := memories
	if cb.ranker != nil && len(memories) > 0 {
		selected = cb.ranker.Rank(currentMessage, memories, cb.topK)
	}
	if len(selected) > 0 {
		var sb strings.Builder
		sb.WriteString("Relevant memories:\n")
		for _, m := range selected {
			sb.WriteString(fmt.Sprintf("- %s (%s)\n", m.Text, m.Kind))
		}
		system = system + sb.String()
	}

	msgs = append(msgs, providers.Message{Role: "system", Content: system})

	// replay history
	for _, h := range history {
		role := h.Role
		content := h.Content
		msgs = append(msgs, providers.Message{Role: role, Content: content})
	}

	// current
	msgs = append(msgs, providers.Message{Role: "user", Content: currentMessage})
	return msgs
}
