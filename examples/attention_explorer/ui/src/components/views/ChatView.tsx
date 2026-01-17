import { MessageList } from '../chat/MessageList';
import { InputBar } from '../chat/InputBar';
import { useAttentionStream } from '../../hooks/useAttentionStream';

export function ChatView() {
  const { sendMessage, abort, isStreaming, error } = useAttentionStream();

  return (
    <div className="card chat-view">
      <div className="card-header">
        <div className="card-title">
          <span>Conversation</span>
          <span className="subtitle">Hover/tap an assistant token to reveal its attention anchors.</span>
        </div>
        <div className="badges">
          <span className="badge">production</span>
        </div>
      </div>
      <div className="card-content">
        <MessageList />
        {error && <div className="error-message">{error.message}</div>}
      </div>
      <div className="card-footer">
        <InputBar onSend={sendMessage} onAbort={abort} isStreaming={isStreaming} />
      </div>
    </div>
  );
}
