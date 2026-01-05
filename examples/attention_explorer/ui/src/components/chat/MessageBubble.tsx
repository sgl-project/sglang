import { Message } from '../../api/types';
import { TokenLine } from './TokenLine';

interface MessageBubbleProps {
  message: Message;
}

export function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === 'user';
  const isAssistant = message.role === 'assistant';

  return (
    <div className={`bubble ${message.role}`}>
      <div className="bubble-meta">
        <span className={`badge ${isUser || isAssistant ? 'strong' : ''}`}>
          {message.role.charAt(0).toUpperCase() + message.role.slice(1)}
        </span>
        {isAssistant && <span className="badge">hover/tap tokens</span>}
        {message.manifold_zone && (
          <span className="badge strong" style={{ marginLeft: 'auto' }}>
            {message.manifold_zone.replace('_', ' ')}
          </span>
        )}
      </div>
      <div className="bubble-content">{message.content}</div>
      {message.tokens && message.tokens.length > 0 && (
        <TokenLine
          tokens={message.tokens}
          attention={message.attention}
          type={isAssistant ? 'output' : 'input'}
        />
      )}
    </div>
  );
}
