import { useMemo } from 'react';
import { Message } from '../../api/types';
import { TokenLine } from './TokenLine';
import { SegmentTimeline } from './SegmentTimeline';
import { SegmentedTokenLine } from './SegmentedTokenLine';
import { VirtualTokenList, VIRTUALIZATION_THRESHOLD } from './VirtualTokenList';

interface MessageBubbleProps {
  message: Message;
}

// Check if content has think tags
function hasThinkSection(content: string): boolean {
  return /<think>[\s\S]*?<\/think>/.test(content);
}

// Strip think tags for display (show clean output)
function stripThinkTags(content: string): string {
  return content
    .replace(/<think>[\s\S]*?<\/think>/g, '')
    .trim();
}

export function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === 'user';
  const isAssistant = message.role === 'assistant';

  // Parse think sections for assistant messages
  const hasThink = useMemo(
    () => isAssistant && hasThinkSection(message.content),
    [isAssistant, message.content]
  );

  const displayContent = useMemo(
    () => (hasThink ? stripThinkTags(message.content) : message.content),
    [hasThink, message.content]
  );

  return (
    <div className={`bubble ${message.role} ${hasThink ? 'has-think' : ''}`}>
      <div className="bubble-meta">
        <span className={`badge ${isUser || isAssistant ? 'strong' : ''}`}>
          {message.role.charAt(0).toUpperCase() + message.role.slice(1)}
        </span>
        {isAssistant && <span className="badge">hover/tap tokens</span>}
        {hasThink && (
          <span className="badge think-badge">
            💭 Reasoning
          </span>
        )}
        {message.manifold_zone && (
          <span className="badge strong" style={{ marginLeft: 'auto' }}>
            {message.manifold_zone.replace('_', ' ')}
          </span>
        )}
      </div>

      {/* Segment timeline for messages with think sections */}
      {hasThink && message.tokens && message.tokens.length > 0 && (
        <SegmentTimeline
          content={message.content}
          tokens={message.tokens}
          compact
        />
      )}

      {/* Display content (stripped of think tags for clean reading) */}
      <div className="bubble-content">
        {displayContent || <span className="muted">(reasoning only)</span>}
      </div>

      {/* Token visualization */}
      {message.tokens && message.tokens.length > 0 && (
        // Use virtualization for large token counts (without think sections)
        message.tokens.length > VIRTUALIZATION_THRESHOLD && !hasThink ? (
          <VirtualTokenList
            tokens={message.tokens}
            attention={message.attention}
            type={isAssistant ? 'output' : 'input'}
          />
        ) : hasThink ? (
          <SegmentedTokenLine
            content={message.content}
            tokens={message.tokens}
            attention={message.attention}
            type={isAssistant ? 'output' : 'input'}
            defaultCollapsed={true}
          />
        ) : (
          <TokenLine
            tokens={message.tokens}
            attention={message.attention}
            type={isAssistant ? 'output' : 'input'}
          />
        )
      )}
    </div>
  );
}
