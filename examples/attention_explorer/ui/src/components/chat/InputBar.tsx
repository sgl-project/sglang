import { useState, useCallback, KeyboardEvent } from 'react';

interface InputBarProps {
  onSend: (message: string) => void;
  onAbort: () => void;
  isStreaming: boolean;
}

export function InputBar({ onSend, onAbort, isStreaming }: InputBarProps) {
  const [input, setInput] = useState('');

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
    <div className="input-bar">
      <textarea
        className="input-textarea"
        placeholder="Type a message... (Enter to send, Shift+Enter for newline)"
        value={input}
        onChange={(e) => setInput(e.target.value)}
        onKeyDown={handleKeyDown}
        disabled={isStreaming}
        rows={2}
      />
      <div className="input-actions">
        {isStreaming ? (
          <button className="btn btn-danger" onClick={onAbort}>
            Stop
          </button>
        ) : (
          <button className="btn btn-primary" onClick={handleSend} disabled={!input.trim()}>
            Send
          </button>
        )}
      </div>
    </div>
  );
}
