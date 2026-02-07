# Development Guide

This document covers how to set up a local development environment, build, test, and publish Picobot.

## Prerequisites

- [Go](https://go.dev/dl/) 1.25+ installed
- [Docker](https://www.docker.com/) installed (for container builds)
- A [Docker Hub](https://hub.docker.com/) account (for publishing)

## Project Structure

```
cmd/picobot/          CLI entry point (main.go)
embeds/               Embedded assets (sample skills bundled into binary)
  skills/             Sample skills extracted on onboard
internal/
  agent/              Agent loop, context, tools, skills
  bus/                Event bus (Inbound / Outbound channels)
  channels/           Telegram integration
  config/             Config schema, loader, onboarding
  cron/               Cron scheduler
  heartbeat/          Periodic task checker
  memory/             Memory read/write/rank
  providers/          OpenRouter, Ollama, Stub
  session/            Session manager
docker/               Dockerfile, compose, entrypoint
```

## Local Development

### Clone and install dependencies

```sh
git clone https://github.com/user/picobot.git
cd picobot
go mod download
```

### Build the binary

```sh
go build -o picobot ./cmd/picobot
```

The binary will be created in the current directory.

### Run locally

```sh
# First-time setup — creates ~/.picobot config and workspace
./picobot onboard

# Single-shot query
./picobot agent -m "Hello!"

# Start gateway (long-running mode with Telegram)
./picobot gateway
```

### Run tests

```sh
# Run all tests
go test ./...

# Run tests for a specific package
go test ./internal/cron/
go test ./internal/agent/

# Run tests with verbose output
go test -v ./...
```

## Versioning

The version string is defined in `cmd/picobot/main.go`:

```go
const version = "x.x.x"
```

Update this value before building a new release.

## Cross-Compilation

Build for different architectures without any runtime dependencies:

```sh
# Linux AMD64 (most VPS / servers)
GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go build -ldflags="-s -w" -o picobot_linux_amd64 ./cmd/picobot

# Linux ARM64 (Raspberry Pi, ARM servers)
GOOS=linux GOARCH=arm64 CGO_ENABLED=0 go build -ldflags="-s -w" -o picobot_linux_arm64 ./cmd/picobot

# macOS ARM64 (Apple Silicon)
GOOS=darwin GOARCH=arm64 CGO_ENABLED=0 go build -ldflags="-s -w" -o picobot_mac_arm64 ./cmd/picobot
```

The `-ldflags="-s -w"` flags strip debug symbols, keeping the binary at ~11MB.

## Docker

### Build the image

The project uses a multi-stage Docker build (Alpine-based, ~33MB final image):

```sh
docker build -f docker/Dockerfile -t louisho5/picobot:latest .
```

> **Note:** Run this from the project root, not from inside `docker/`.

### Test the image locally

```sh
docker run --rm -it \
  -e OPENROUTER_API_KEY="your-key" \
  -e PICOBOT_MODEL="google/gemini-2.5-flash" \
  -e TELEGRAM_BOT_TOKEN="your-token" \
  -v ./picobot-data:/home/picobot/.picobot \
  louisho5/picobot:latest
```

Check logs:

```sh
docker logs -f picobot
```

### Push to Docker Hub

1. **Log in** (one-time):

```sh
docker login
```

2. **Build and push** (single command):

```sh
go build ./... && \
docker build -f docker/Dockerfile -t louisho5/picobot:latest . && \
docker push louisho5/picobot:latest
```

3. **Verify** the image is live at [hub.docker.com/r/louisho5/picobot](https://hub.docker.com/r/louisho5/picobot).

### Tagging a specific version

If you want to publish a versioned tag alongside `latest`:

```sh
docker tag louisho5/picobot:latest louisho5/picobot:v0.1.0
docker push louisho5/picobot:v0.1.0
```

### Full release workflow

```sh
# 1. Update version in cmd/picobot/main.go
# 2. Run tests
go test ./...

# 3. Build Go binary (validates compilation)
go build ./...

# 4. Build Docker image
docker build -f docker/Dockerfile -t louisho5/picobot:latest .

# 5. Push to Docker Hub
docker push louisho5/picobot:latest

# 6. (Optional) Tag and push a versioned release
docker tag louisho5/picobot:latest louisho5/picobot:v0.1.0
docker push louisho5/picobot:v0.1.0
```

## Docker Compose (Development)

For local testing with Docker Compose:

```sh
cd docker
cp .env.example .env
# Edit .env with your API keys
docker compose up -d
```

View logs:

```sh
docker compose logs -f
```

Stop:

```sh
docker compose down
```

## Environment Variables

These environment variables configure the Docker container:

| Variable | Description | Required |
|---|---|---|
| `OPENROUTER_API_KEY` | OpenRouter API key | Yes |
| `PICOBOT_MODEL` | LLM model to use (e.g. `google/gemini-2.5-flash`) | No |
| `TELEGRAM_BOT_TOKEN` | Telegram Bot API token | Yes (for gateway) |
| `TELEGRAM_ALLOW_FROM` | Comma-separated Telegram user IDs to allow | No |

## Adding a New Tool

1. Create a new file in `internal/agent/tools/` (e.g. `mytool.go`)
2. Implement the `Tool` interface from `internal/agent/tools/base.go`
3. Register the tool in `internal/agent/tools/registry.go`
4. Run tests: `go test ./internal/agent/...`

## Adding a New Provider

1. Create a new file in `internal/providers/` (e.g. `myprovider.go`)
2. Implement the `Provider` interface from `internal/providers/base.go`
3. Wire it up in the provider factory in `internal/providers/`
4. Add config schema fields in `internal/config/schema.go`

## Troubleshooting

**Binary won't compile:**
```sh
go mod tidy
go build ./...
```

**Docker build fails on `go mod download`:**
Make sure `go.sum` is committed and up to date:
```sh
go mod tidy
git add go.mod go.sum
```

**Image push rejected:**
Verify you're logged in and the repository name matches:
```sh
docker login
docker images | grep picobot
```
