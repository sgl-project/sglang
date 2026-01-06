import { useState, useCallback, KeyboardEvent, useEffect, useRef } from 'react';

interface InputBarProps {
  onSend: (message: string) => void;
  onAbort: () => void;
  isStreaming: boolean;
}

export function InputBar({ onSend, onAbort, isStreaming }: InputBarProps) {
  const [input, setInput] = useState('');
  const [charCount, setCharCount] = useState(0);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    setCharCount(input.length);
  }, [input]);

  // Auto-focus when not streaming
  useEffect(() => {
    if (!isStreaming && textareaRef.current) {
      textareaRef.current.focus();
    }
  }, [isStreaming]);

  const handleSend = useCallback(() => {
    const trimmed = input.trim();
    if (trimmed && !isStreaming) {
      onSend(trimmed);
      setInput('');
    }
  }, [input, isStreaming, onSend]);

  const handleKeyDown = useCallback(
    (e: KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    },
    [handleSend]
  );

  return (
    <div className="input-bar" style={{ position: 'relative' }}>
      <div style={{ flex: 1, position: 'relative' }}>
        <textarea
          ref={textareaRef}
          className="input-textarea"
          placeholder={isStreaming ? 'Waiting for response...' : 'Type a message... (Enter to send)'}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={isStreaming}
          rows={2}
          style={{
            opacity: isStreaming ? 0.6 : 1,
            transition: 'opacity 0.2s ease'
          }}
        />
        {charCount > 0 && !isStreaming && (
          <span style={{
            position: 'absolute',
            bottom: '8px',
            right: '8px',
            fontSize: '10px',
            color: 'var(--muted)',
            opacity: 0.7
          }}>
            {charCount} chars
          </span>
        )}
      </div>
      <div className="input-actions" style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
        {isStreaming ? (
          <>
            <div className="streaming-indicator">
              Generating...
            </div>
            <button
              className="btn btn-danger"
              onClick={onAbort}
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '4px'
              }}
            >
              <span style={{ fontSize: '14px' }}>⏹</span>
              Stop
            </button>
          </>
        ) : (
          <button
            className="btn btn-primary"
            onClick={handleSend}
            disabled={!input.trim()}
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '4px',
              minWidth: '80px'
            }}
          >
            <span style={{ fontSize: '14px' }}>↵</span>
            Send
          </button>
        )}
      </div>
    </div>
  );
}
