package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"time"

	cfgpkg "github.com/local/picobot/internal/config"
	mcpclient "github.com/mark3labs/mcp-go/client"
	transport "github.com/mark3labs/mcp-go/client/transport"
	"github.com/mark3labs/mcp-go/mcp"
)

// RegisterMCPFromConfig inspects the config and registers MCP-based remote tools
// into the provided registry. Each server's tools are registered with names
// prefixed by "mcp.<server>.<tool>".
func RegisterMCPFromConfig(reg *Registry, cfg cfgpkg.Config) {
	if cfg.Tools.MCP == nil || !cfg.Tools.MCP.Enabled {
		return
	}

	for srvName, srv := range cfg.Tools.MCP.Servers {
		// build transport
		var tr transport.Interface
		switch strings.ToLower(srv.Transport) {
		case "stdio":
			// expand ~ in command
			cmd := srv.Command
			if strings.HasPrefix(cmd, "~/") {
				if h, err := os.UserHomeDir(); err == nil {
					cmd = filepath.Join(h, cmd[2:])
				}
			}
			tr = transport.NewStdio(cmd, nil, srv.Args...)
			log.Printf("mcp: starting stdio transport for %s: %s %v\n", srvName, cmd, srv.Args)
		case "http":
			// convert headers
			hdr := make(map[string]string)
			for k, v := range srv.Headers {
				hdr[k] = v
			}
			// create streamable HTTP transport (SDK transport factory)
			t, err := transport.NewStreamableHTTP(srv.URL, transport.WithHTTPHeaders(hdr))
			if err != nil {
				log.Printf("mcp: failed to create http transport for %s: %v", srvName, err)
				continue
			}
			tr = t
		default:
			log.Printf("mcp: unknown transport %q for server %s", srv.Transport, srvName)
			continue
		}

		// create client
		cli := mcpclient.NewClient(tr)
		ctx := context.Background()
		if err := cli.Start(ctx); err != nil {
			log.Printf("mcp: failed to start client for %s: %v", srvName, err)
			continue
		}

		// Initialize the MCP session in a goroutine to avoid blocking stdio read/write loops
		initDone := make(chan error, 1)
		go func() {
			initRequest := mcp.InitializeRequest{
				Params: mcp.InitializeParams{
					ProtocolVersion: mcp.LATEST_PROTOCOL_VERSION,
					Capabilities:    mcp.ClientCapabilities{},
					ClientInfo: mcp.Implementation{
						Name:    "picobot",
						Version: "1.0.0",
					},
				},
			}
			initResult, err := cli.Initialize(ctx, initRequest)
			if err != nil {
				log.Printf("Failed to initialize: %v", err)
			} else {
				log.Printf(
					"Initialized with server: %s %s\n\n",
					initResult.ServerInfo.Name,
					initResult.ServerInfo.Version,
				)
			}
			initDone <- err
		}()

		// Wait for initialize to complete (with timeout)
		initCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
		select {
		case err := <-initDone:
			cancel()
			if err != nil {
				log.Printf("mcp: initialize failed for %s: %v", srvName, err)
				// continue - client may still work for simple calls
			}
		case <-initCtx.Done():
			cancel()
			log.Printf("mcp: initialize timeout for %s", srvName)
			continue
		}

		// list tools exposed by server
		toolsRes, err := cli.ListTools(ctx, mcp.ListToolsRequest{})
		if err != nil {
			log.Printf("mcp: failed to list tools for %s: %v", srvName, err)
			continue
		}

		for _, t := range toolsRes.Tools {
			// try to convert the tool input schema into a generic map for provider tooling
			var params map[string]interface{}
			if b, err := json.Marshal(t.InputSchema); err == nil {
				_ = json.Unmarshal(b, &params)
			}

			// register each remote tool using its original name so model tool-calls match
			regName := t.Name
			rt := &mcpRemoteTool{client: cli, server: srvName, toolName: t.Name, description: t.Description, parameters: params}
			reg.Register(rt.withName(regName))
		}
	}
}

type mcpRemoteTool struct {
	client      *mcpclient.Client
	server      string
	toolName    string
	name        string
	description string
	parameters  map[string]interface{}
}

func (m *mcpRemoteTool) withName(n string) Tool {
	m.name = n
	return m
}

func (m *mcpRemoteTool) Name() string                       { return m.name }
func (m *mcpRemoteTool) Description() string                { return m.description }
func (m *mcpRemoteTool) Parameters() map[string]interface{} { return m.parameters }

func (m *mcpRemoteTool) Execute(ctx context.Context, args map[string]interface{}) (string, error) {
	// construct call request
	req := mcp.CallToolRequest{}
	// populate params minimally
	req.Params.Name = m.toolName
	req.Params.Arguments = args

	res, err := m.client.CallTool(ctx, req)
	if err != nil {
		return "", err
	}
	// return formatted result
	return fmt.Sprintf("%v", res), nil
}
