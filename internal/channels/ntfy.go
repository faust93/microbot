package channels

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/local/picobot/internal/chat"
)

type NtfyChannel struct {
	url    string
	token  string
	topic  string
	client *http.Client
}

func StartNtfy(ctx context.Context, hub *chat.Hub, server, token string, topic string) error {
	if server == "" {
		server = "https://ntfy.sh"
	}
	if token == "" {
		return fmt.Errorf("ntfy token not provided")
	}

	nc := NewNtfyChannel(server, token, topic)

	// Start a goroutine to listen for outbound messages and send them via ntfy
	go func() {
		for {
			select {
			case <-ctx.Done():
				log.Println("ntfy: stopping outbound sender")
				return
			case msg := <-hub.NtfyOut:
				title := "Picobot"
				if err := nc.Send(title, msg.ChatID, msg.Content); err != nil {
					log.Printf("ntfy: failed to send message: %v", err)
				}
			}
		}
	}()

	log.Printf("ntfy channel started with topic '%s'", topic)
	return nil
}

func NewNtfyChannel(server string, token string, topic string) *NtfyChannel {
	return &NtfyChannel{
		url:    server,
		token:  token,
		topic:  topic,
		client: &http.Client{Timeout: 10 * time.Second},
	}
}

func (nc *NtfyChannel) Send(title, chatID, message string) error {
	topic := nc.topic
	if chatID != "default" {
		topic = chatID
	}
	endpoint := fmt.Sprintf("%s/%s", nc.url, topic)

	body := strings.NewReader(message)
	req, err := http.NewRequest("POST", endpoint, body)
	if err != nil {
		return err
	}

	req.Header.Set("Authorization", "Bearer "+nc.token)
	req.Header.Set("Title", title)

	// Use the custom client to do the request
	resp, err := nc.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("ntfy returned status %d", resp.StatusCode)
	}

	return nil
}
