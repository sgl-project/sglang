import { useState, useCallback, useRef, useEffect } from 'react';
import { useSessionStore } from '../stores/useSessionStore';
import { useUIStore } from '../stores/useUIStore';
import { AttentionStreamClient } from '../api/client';
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
  const appendAttention = useSessionStore((state) => state.appendAttention);
  const setFingerprint = useSessionStore((state) => state.setFingerprint);

  const clientRef = useRef<AttentionStreamClient | null>(null);

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
      onToken: (token) => {
        appendToken(token);
      },
      onAttention: (entry) => {
        // Store attention in the session (will be matched to tokens by step)
        appendAttention(entry);

        if (isFingerprintMode(entry)) {
          const fp = extractFingerprint(entry);
          if (fp) setFingerprint(fp);
        }
      },
      onFinish: () => {
        setIsStreaming(false);
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
  }, [baseUrl, model, program, appendToken, appendAttention, setFingerprint]);

  useEffect(() => {
    clientRef.current?.setProgram(program);
  }, [program]);

  const sendMessage = useCallback(
    async (content: string) => {
      if (!clientRef.current) return;

      setIsStreaming(true);
      setError(null);

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
