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
  const setModelName = useUIStore((state) => state.setModelName);
  const addMessage = useSessionStore((state) => state.addMessage);
  const appendToken = useSessionStore((state) => state.appendToken);
  const setFingerprint = useSessionStore((state) => state.setFingerprint);

  const clientRef = useRef<AttentionStreamClient | null>(null);
  const alignerRef = useRef<TokenAttentionAligner>(new TokenAttentionAligner());

  // Check server health and fetch model info
  useEffect(() => {
    let cancelled = false;

    const checkConnection = async () => {
      try {
        // Check health endpoint
        const healthRes = await fetch(`${baseUrl}/health`, { method: 'GET' });
        if (!healthRes.ok) throw new Error('Server not healthy');

        // Get model info
        const modelsRes = await fetch(`${baseUrl}/v1/models`, { method: 'GET' });
        if (modelsRes.ok) {
          const data = await modelsRes.json();
          if (data.data?.[0]?.id && !cancelled) {
            setModelName(data.data[0].id);
          }
        }

        if (!cancelled) {
          setConnected(true);
        }
      } catch (err) {
        console.warn('Server connection check failed:', err);
        if (!cancelled) {
          setConnected(false);
          setModelName('Not connected');
        }
      }
    };

    checkConnection();
    // Re-check every 10 seconds
    const interval = setInterval(checkConnection, 10000);

    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [baseUrl, setConnected, setModelName]);

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

    return () => {
      client.abort();
    };
  }, [baseUrl, model, program, appendToken, setFingerprint]);

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
