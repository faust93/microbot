package channels

import (
	"context"
	"log"

	"github.com/local/picobot/internal/chat"
)

func StartProxy(ctx context.Context, hub *chat.Hub) error {
	log.Println("Starting proxy channel")

	go func() {
		for {
			select {
			case <-ctx.Done():
				log.Println("proxy: stopping outbound sender")
				return
			case msg := <-hub.Out:
				switch msg.Channel {
				case "telegram":
					select {
					case hub.TelegramOut <- msg:
						log.Printf("proxy: forwarded message to telegram channel for chatID %s", msg.ChatID)
					default:
						log.Printf("telegram channel full, dropping message for %s", msg.ChatID)
					}
				case "ntfy":
					select {
					case hub.NtfyOut <- msg:
						log.Printf("proxy: forwarded message to ntfy channel for chatID %s", msg.ChatID)
					default:
						log.Printf("ntfy channel full, dropping message for %s", msg.ChatID)
					}
				default:
					log.Printf("unknown channel type: %s", msg.Channel)
				}
			}
		}
	}()

	return nil
}
