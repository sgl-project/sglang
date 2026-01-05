import { useSessionStore } from '../../stores/useSessionStore';
import { MessageBubble } from './MessageBubble';

export function MessageList() {
  const messages = useSessionStore((state) => state.messages);

  if (messages.length === 0) {
    return (
      <div className="empty-state">
        <div className="empty-icon">∿</div>
        <div className="empty-title">Start a conversation</div>
        <div className="empty-subtitle">
          Type a message below. Hover assistant tokens to see attention patterns.
        </div>
      </div>
    );
  }

  return (
    <div className="message-list">
      {messages.map((msg) => (
        <MessageBubble key={msg.id} message={msg} />
      ))}
    </div>
  );
}
