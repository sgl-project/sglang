import { useState, useCallback, useRef, useEffect } from 'react';
import { useSessionStore } from '../stores/useSessionStore';
import { useUIStore } from '../stores/useUIStore';
import { AttentionStreamClient, TokenAttentionAligner } from '../api/client';
import { extractFingerprint, isFingerprintMode } from '../api/types';

export interface UseAttentionStreamOptions {
  baseUrl?: string;
  model?: string;
}

export function useAttentionStream(options: UseAttentionStreamOptions = {}) {
  const { baseUrl = 'http://localhost:30000', model = 'Qwen/Qwen3-Next-80B-A3B-Thinking-FP8' } = options;

  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  const program = useUIStore((state) => state.program);
  const setConnected = useUIStore((state) => state.setConnected);
  const addMessage = useSessionStore((state) => state.addMessage);
  const appendToken = useSessionStore((state) => state.appendToken);
  const setFingerprint = useSessionStore((state) => state.setFingerprint);

  const clientRef = useRef<AttentionStreamClient | null>(null);
  const alignerRef = useRef<TokenAttentionAligner>(new TokenAttentionAligner());

  useEffect(() => {
    const client = new AttentionStreamClient({
      baseUrl,
      model,
      program,
      onToken: (token, index) => {
        alignerRef.current.addToken(token);
        appendToken(token, alignerRef.current.getAttention(index));
      },
      onAttention: (entry) => {
        alignerRef.current.addAttention(entry);

        if (isFingerprintMode(entry)) {
          const fp = extractFingerprint(entry);
          if (fp) setFingerprint(fp);
        }
      },
      onFinish: () => {
        setIsStreaming(false);
        const stats = alignerRef.current.getAlignmentStats();
        console.log('Alignment stats:', stats);
      },
      onError: (err) => {
        setError(err);
        setIsStreaming(false);
      },
    });

    clientRef.current = client;
    setConnected(true);

    return () => {
      client.abort();
      setConnected(false);
    };
  }, [baseUrl, model, program, appendToken, setFingerprint, setConnected]);

  useEffect(() => {
    clientRef.current?.setProgram(program);
  }, [program]);

  const sendMessage = useCallback(
    async (content: string) => {
      if (!clientRef.current) return;

      setIsStreaming(true);
      setError(null);

      alignerRef.current = new TokenAttentionAligner();

      addMessage({
        id: `user-${Date.now()}`,
        role: 'user',
        content,
        timestamp: Date.now(),
      });

      const messages = useSessionStore.getState().messages.map((m) => ({
        role: m.role,
        content: m.content,
      }));

      const prefillChars = messages.reduce((sum, m) => sum + m.content.length, 0);
      alignerRef.current.setPrefillLength(Math.ceil(prefillChars / 4));

      try {
        await clientRef.current.stream(messages);
      } catch (err) {
        setError(err as Error);
        setIsStreaming(false);
      }
    },
    [addMessage]
  );

  const abort = useCallback(() => {
    clientRef.current?.abort();
    setIsStreaming(false);
  }, []);

  return {
    sendMessage,
    abort,
    isStreaming,
    error,
  };
}
