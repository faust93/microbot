package agent

import (
	"context"
	"log"
	"os"
	"regexp"
	"strings"
	"time"

	"github.com/local/picobot/internal/agent/memory"
	"github.com/local/picobot/internal/agent/tools"
	"github.com/local/picobot/internal/chat"
	"github.com/local/picobot/internal/config"
	"github.com/local/picobot/internal/cron"
	"github.com/local/picobot/internal/providers"
	"github.com/local/picobot/internal/session"
)

var rememberRE = regexp.MustCompile(`(?i)^remember(?:\s+to)?\s+(.+)$`)

// AgentLoop is the core processing loop; it holds an LLM provider, tools, sessions and context builder.
type AgentLoop struct {
	hub           *chat.Hub
	provider      providers.LLMProvider
	tools         *tools.Registry
	sessions      *session.SessionManager
	context       *ContextBuilder
	memory        *memory.MemoryStore
	memoryPersist *memory.MemoryPersist
	model         string
	maxIterations int
	temperature   float64
	maxTokens     int
	running       bool
}

// NewAgentLoop creates a new AgentLoop with the given provider.
func NewAgentLoop(b *chat.Hub, provider providers.LLMProvider, model string, maxIterations int, Temperature float64, MaxTokens int, workspace string, scheduler *cron.Scheduler, toolsConfig *config.ToolsConfig, memoryConfig *config.MemoryConfig) *AgentLoop {
	if model == "" {
		model = provider.GetDefaultModel()
	}
	if workspace == "" {
		workspace = "."
	}
	reg := tools.NewRegistry()
	// register default tools
	reg.Register(tools.NewMessageTool(b))

	// Open an os.Root anchored at the workspace for kernel-enforced sandboxing.
	root, err := os.OpenRoot(workspace)
	if err != nil {
		log.Fatalf("failed to open workspace root %q: %v", workspace, err)
	}

	fsTool, err := tools.NewFilesystemTool(workspace)
	if err != nil {
		log.Fatalf("failed to create filesystem tool: %v", err)
	}
	reg.Register(fsTool)

	reg.Register(tools.NewExecTool(60))
	reg.Register(tools.NewWebTool())
	reg.Register(tools.NewSpawnTool())
	if scheduler != nil {
		reg.Register(tools.NewCronTool(scheduler))
	}

	sm := session.NewSessionManager(workspace)
	_ = sm.LoadAll() // best effort load existing sessions on startup

	memPersist := (*memory.MemoryPersist)(nil)
	if memoryConfig != nil && memoryConfig.Enabled {

		memPersist = memory.NewPersistMemory(memoryConfig)
		if memPersist == nil {
			log.Fatalf("failed to initialize memory system with config: %+v", memoryConfig)
		}

		log.Printf("Persistent memory store initialized with %s embedder", memoryConfig.EmbedType)
	}

	ctx := NewContextBuilder(workspace, memPersist)
	mem := memory.NewMemoryStoreWithWorkspace(workspace, 100)
	// register memory tool (needs store instance)
	reg.Register(tools.NewWriteMemoryTool(mem))

	// register skill management tools (share the same os.Root)
	skillMgr := tools.NewSkillManager(root)
	reg.Register(tools.NewCreateSkillTool(skillMgr))
	reg.Register(tools.NewListSkillsTool(skillMgr))
	reg.Register(tools.NewReadSkillTool(skillMgr))
	reg.Register(tools.NewDeleteSkillTool(skillMgr))

	tools.RegisterMCPFromConfig(reg, toolsConfig)

	return &AgentLoop{hub: b, provider: provider, tools: reg, sessions: sm, context: ctx, memory: mem, memoryPersist: memPersist, model: model, maxIterations: maxIterations, temperature: Temperature, maxTokens: MaxTokens}
}

// Run starts processing inbound messages. This is a blocking call until context is canceled.
func (a *AgentLoop) Run(ctx context.Context) {
	a.running = true
	log.Println("Agent loop started")

	for a.running {
		select {
		case <-ctx.Done():
			log.Println("Agent loop received shutdown signal")
			a.running = false
			return
		case msg, ok := <-a.hub.In:
			if !ok {
				log.Println("Inbound channel closed, stopping agent loop")
				a.running = false
				return
			}

			log.Printf("Processing message from %s:%s\n", msg.Channel, msg.SenderID)

			// Quick heuristic: if user asks the agent to remember something explicitly,
			// store it in today's note and reply immediately without calling the LLM.
			trimmed := strings.TrimSpace(msg.Content)
			rememberRe := rememberRE
			if matches := rememberRe.FindStringSubmatch(trimmed); len(matches) == 2 {
				note := matches[1]
				if err := a.memory.AppendToday(note); err != nil {
					log.Printf("error appending to memory: %v", err)
				}
				out := chat.Outbound{Channel: msg.Channel, ChatID: msg.ChatID, Content: "OK, I've remembered that."}
				select {
				case a.hub.Out <- out:
				default:
					log.Println("Outbound channel full, dropping message")
				}
				// save to session as well
				session := a.sessions.GetOrCreate(msg.Channel + ":" + msg.ChatID)
				session.AddMessage("user", msg.Content)
				session.AddMessage("assistant", "OK, I've remembered that.")
				a.sessions.Save(session)
				continue
			}

			// Set tool context (so message tool knows channel+chat)
			if mt := a.tools.Get("message"); mt != nil {
				if mtool, ok := mt.(interface{ SetContext(string, string) }); ok {
					mtool.SetContext(msg.Channel, msg.ChatID)
				}
			}
			if ct := a.tools.Get("cron"); ct != nil {
				if ctool, ok := ct.(interface{ SetContext(string, string) }); ok {
					ctool.SetContext(msg.Channel, msg.ChatID)
				}
			}

			// Build messages from session, long-term memory, and recent memory
			session := a.sessions.GetOrCreate(msg.Channel + ":" + msg.ChatID)
			// get file-backed memory context (long-term + today)
			memCtx, _ := a.memory.GetMemoryContext()
			// query persistent memory for relevant items
			memories := []memory.MemoryItem{}
			if a.memoryPersist != nil {
				memx, err := a.memoryPersist.QueryHistory(msg.Channel+msg.ChatID, msg.Content, 0)
				if err != nil {
					log.Printf("Failed to query persistent memory: %v", err)
				} else {
					//log.Printf("Memory query returned %d items:\n", len(memx))
					for i, m := range memx {
						log.Printf("Result[%d] Similarity: %.4f Role: %s Content: %q\n\n", i, m.Similarity, m.Role, m.Text)
						memories = append(memories, memory.MemoryItem{
							Role:       m.Role,
							Text:       m.Text,
							Timestamp:  m.Timestamp,
							Similarity: m.Similarity,
							Kind:       "Persistent",
						})
					}
				}
			}

			messages := a.context.BuildMessages(session.GetHistory(), msg.Content, msg.Channel, msg.ChatID, memCtx, memories)

			iteration := 0
			finalContent := ""
			lastToolResult := ""
			toolDefs := a.tools.Definitions()
			for iteration < a.maxIterations {
				iteration++
				resp, err := a.provider.Chat(ctx, messages, toolDefs, a.model, a.temperature, a.maxTokens)
				if err != nil {
					log.Printf("provider error: %v", err)
					finalContent = "Sorry, I encountered an error while processing your request."
					break
				}

				if resp.HasToolCalls {
					// append assistant message with tool_calls attached
					messages = append(messages, providers.Message{Role: "assistant", Content: resp.Content, ToolCalls: resp.ToolCalls})
					// Execute each tool call and return results with "tool" role
					for _, tc := range resp.ToolCalls {
						res, err := a.tools.Execute(ctx, tc.Name, tc.Arguments)
						if err != nil {
							res = "(tool error) " + err.Error()
						}
						lastToolResult = res
						messages = append(messages, providers.Message{Role: "tool", Content: res, ToolCallID: tc.ID})
					}
					// loop again
					continue
				} else {
					finalContent = resp.Content
					break
				}
			}

			if finalContent == "" && lastToolResult != "" {
				finalContent = lastToolResult
			} else if finalContent == "" {
				finalContent = "I've completed processing but have no response to give."
			}

			// Save session
			session.AddMessage("user", msg.Content)
			session.AddMessage("assistant", finalContent)

			// save trimmed history to persistent memory before saving session, to avoid blowing up session file size and LLM context window.
			// This means trimmed messages won't be in session history but will be in memory history if memory is enabled.
			if a.memoryPersist != nil {
				msgs := a.sessions.TrimAll()
				for _, m := range msgs {
					if m.Role != "user" {
						//log.Printf("Storing trimmed history to memory: Role: %s Content: %q\n", m.Role, m.Content)
						err := a.memoryPersist.StoreHistory(msg.Channel+msg.ChatID, m.Role, m.Content, m.Timestamp)
						if err != nil {
							log.Printf("Failed to store trimmed history: %v", err)
						}
					}
				}
			} else {
				_ = a.sessions.TrimAll()
			}
			a.sessions.Save(session)

			out := chat.Outbound{Channel: msg.Channel, ChatID: msg.ChatID, Content: finalContent}
			select {
			case a.hub.Out <- out:
			default:
				log.Println("Outbound channel full, dropping message")
			}
		default:
			// idle tick
			time.Sleep(100 * time.Millisecond)
		}
	}
}

// ProcessDirect sends a message directly to the provider and returns the response.
// It supports tool calling - if the model requests tools, they will be executed.
func (a *AgentLoop) ProcessDirect(content string, timeout time.Duration) (string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	// Set tool context so message/cron tools know the originating channel,
	// matching what Run() does for hub-based messages.
	if mt := a.tools.Get("message"); mt != nil {
		if mtool, ok := mt.(interface{ SetContext(string, string) }); ok {
			mtool.SetContext("cli", "direct")
		}
	}
	if ct := a.tools.Get("cron"); ct != nil {
		if ctool, ok := ct.(interface{ SetContext(string, string) }); ok {
			ctool.SetContext("cli", "direct")
		}
	}

	// Build full context (bootstrap files, skills, memory) just like the main loop
	memCtx, _ := a.memory.GetMemoryContext()
	memories := []memory.MemoryItem{} //a.memory.Recent(5)
	messages := a.context.BuildMessages(nil, content, "cli", "direct", memCtx, memories)

	// Support tool calling iterations (similar to main loop)
	var lastToolResult string
	for iteration := 0; iteration < a.maxIterations; iteration++ {
		resp, err := a.provider.Chat(ctx, messages, a.tools.Definitions(), a.model, a.temperature, a.maxTokens)
		if err != nil {
			return "", err
		}

		if !resp.HasToolCalls {
			// No tool calls, return the response (fall back to last tool result if empty)
			if resp.Content != "" {
				return resp.Content, nil
			}
			if lastToolResult != "" {
				return lastToolResult, nil
			}
			return resp.Content, nil
		}

		// Execute tool calls
		messages = append(messages, providers.Message{Role: "assistant", Content: resp.Content, ToolCalls: resp.ToolCalls})
		for _, tc := range resp.ToolCalls {
			result, err := a.tools.Execute(ctx, tc.Name, tc.Arguments)
			if err != nil {
				result = "(tool error) " + err.Error()
			}
			lastToolResult = result
			messages = append(messages, providers.Message{Role: "tool", Content: result, ToolCallID: tc.ID})
		}
	}

	return "Max iterations reached without final response", nil
}
