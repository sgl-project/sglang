import { create } from 'zustand';
import {
  TraceSession,
  Message,
  TokenEntry,
  Segment,
  DecodeStep,
  AttentionEntry,
  MoERoutingEntry,
  createTraceSession,
  detectSegments,
  computeSessionMetrics,
  extractFingerprint,
  getManifoldZone,
} from '../api/types';

interface TraceState {
  // Current active trace session
  currentTrace: TraceSession | null;

  // Saved traces for history/comparison
  savedTraces: TraceSession[];

  // Actions
  initTrace: (model: string) => void;
  clearTrace: () => void;

  // Message operations
  addUserMessage: (content: string) => void;
  startAssistantMessage: () => void;
  appendToken: (token: string, attention?: AttentionEntry, moe?: MoERoutingEntry) => void;
  finishAssistantMessage: () => void;

  // Direct data operations
  addDecodeStep: (step: DecodeStep) => void;
  updateMetrics: () => void;

  // Trace management
  saveCurrentTrace: () => void;
  loadTrace: (traceId: string) => void;
  deleteTrace: (traceId: string) => void;

  // Export/Import
  exportTraceAsJSONL: () => void;
  importTraceFromJSONL: (jsonlContent: string) => void;

  // Getters (computed from current trace)
  getTokenAt: (index: number) => TokenEntry | null;
  getStepAt: (index: number) => DecodeStep | null;
  getSegmentForToken: (index: number) => Segment | null;
  getTokensInSegment: (segmentId: string) => TokenEntry[];
}

export const useTraceStore = create<TraceState>((set, get) => ({
  currentTrace: null,
  savedTraces: [],

  initTrace: (model) => {
    set({ currentTrace: createTraceSession(model) });
  },

  clearTrace: () => {
    set({ currentTrace: null });
  },

  addUserMessage: (content) => {
    set((state) => {
      if (!state.currentTrace) return state;

      const messageId = `user-${Date.now()}`;
      const message: Message = {
        id: messageId,
        role: 'user',
        content,
        timestamp: Date.now(),
      };

      // Create user segment (no tokens since we don't tokenize user input client-side)
      const segment: Segment = {
        id: `${messageId}-main`,
        type: 'user',
        startTokenIndex: state.currentTrace.tokens.length,
        endTokenIndex: state.currentTrace.tokens.length, // Empty for now
        messageId,
      };

      return {
        currentTrace: {
          ...state.currentTrace,
          messages: [...state.currentTrace.messages, message],
          segments: [...state.currentTrace.segments, segment],
          updatedAt: Date.now(),
        },
      };
    });
  },

  startAssistantMessage: () => {
    set((state) => {
      if (!state.currentTrace) return state;

      const messageId = `assistant-${Date.now()}`;
      const message: Message = {
        id: messageId,
        role: 'assistant',
        content: '',
        tokens: [],
        attention: [],
        timestamp: Date.now(),
      };

      return {
        currentTrace: {
          ...state.currentTrace,
          messages: [...state.currentTrace.messages, message],
          isStreaming: true,
          streamingTokenIndex: state.currentTrace.tokens.length,
          updatedAt: Date.now(),
        },
      };
    });
  },

  appendToken: (token, attention, moe) => {
    set((state) => {
      if (!state.currentTrace || !state.currentTrace.isStreaming) return state;

      const trace = state.currentTrace;
      const messages = [...trace.messages];
      const lastMsg = messages[messages.length - 1];

      if (!lastMsg || lastMsg.role !== 'assistant') return state;

      // Update message content and tokens
      const newTokens = [...(lastMsg.tokens || []), token];
      const newContent = lastMsg.content + token;
      const newAttention = [...(lastMsg.attention || [])];

      // Add attention at the correct index
      const tokenIndex = trace.tokens.length;
      if (attention) {
        newAttention[newTokens.length - 1] = attention;
      }

      messages[messages.length - 1] = {
        ...lastMsg,
        content: newContent,
        tokens: newTokens,
        attention: newAttention,
      };

      // Create token entry
      const tokenEntry: TokenEntry = {
        index: tokenIndex,
        text: token,
        segmentId: `${lastMsg.id}-main`, // Will be updated on finalize
        role: 'generated',
      };

      // Create decode step if we have attention data
      const newSteps = [...trace.steps];
      if (attention || moe) {
        const fp = attention ? extractFingerprint(attention) : undefined;
        // Use getManifoldZone to prefer server-side zone (security fix)
        const zone = attention ? getManifoldZone(attention) : undefined;
        const step: DecodeStep = {
          tokenIndex,
          attention,
          moe,
          fingerprint: fp || undefined,
          manifoldZone: zone !== 'unknown' ? zone : undefined,
        };
        newSteps.push(step);
      }

      return {
        currentTrace: {
          ...trace,
          messages,
          tokens: [...trace.tokens, tokenEntry],
          steps: newSteps,
          updatedAt: Date.now(),
        },
      };
    });
  },

  finishAssistantMessage: () => {
    set((state) => {
      if (!state.currentTrace) return state;

      const trace = state.currentTrace;
      const messages = [...trace.messages];
      const lastMsg = messages[messages.length - 1];

      if (!lastMsg || lastMsg.role !== 'assistant') return state;

      // Detect segments for think/output boundaries
      const startIndex = trace.streamingTokenIndex;
      const messageTokens = lastMsg.tokens || [];
      const newSegments = detectSegments(
        lastMsg.content,
        lastMsg.id,
        startIndex,
        messageTokens
      );

      // Update token entries with correct segment IDs
      const tokens = [...trace.tokens];
      for (let i = startIndex; i < tokens.length; i++) {
        // Find which segment this token belongs to
        for (const seg of newSegments) {
          if (i >= seg.startTokenIndex && i < seg.endTokenIndex) {
            tokens[i] = { ...tokens[i], segmentId: seg.id };
            break;
          }
        }
      }

      // Compute session metrics
      const metrics = computeSessionMetrics(trace.steps);

      return {
        currentTrace: {
          ...trace,
          messages,
          tokens,
          segments: [...trace.segments, ...newSegments],
          metrics,
          isStreaming: false,
          streamingTokenIndex: -1,
          updatedAt: Date.now(),
        },
      };
    });
  },

  addDecodeStep: (step) => {
    set((state) => {
      if (!state.currentTrace) return state;

      return {
        currentTrace: {
          ...state.currentTrace,
          steps: [...state.currentTrace.steps, step],
          updatedAt: Date.now(),
        },
      };
    });
  },

  updateMetrics: () => {
    set((state) => {
      if (!state.currentTrace) return state;

      const metrics = computeSessionMetrics(state.currentTrace.steps);

      return {
        currentTrace: {
          ...state.currentTrace,
          metrics,
          updatedAt: Date.now(),
        },
      };
    });
  },

  saveCurrentTrace: () => {
    set((state) => {
      if (!state.currentTrace) return state;

      // Don't save if already in saved traces
      const exists = state.savedTraces.some((t) => t.id === state.currentTrace?.id);
      if (exists) return state;

      return {
        savedTraces: [...state.savedTraces, state.currentTrace],
      };
    });
  },

  loadTrace: (traceId) => {
    set((state) => {
      const trace = state.savedTraces.find((t) => t.id === traceId);
      if (!trace) return state;

      return { currentTrace: { ...trace } };
    });
  },

  deleteTrace: (traceId) => {
    set((state) => ({
      savedTraces: state.savedTraces.filter((t) => t.id !== traceId),
    }));
  },

  exportTraceAsJSONL: () => {
    const trace = get().currentTrace;
    if (!trace) return;

    // Build JSONL content with one JSON object per line
    // Using record_type to avoid conflicts with existing 'type' fields in data
    const lines: string[] = [];

    // Header with metadata
    lines.push(JSON.stringify({
      record_type: 'header',
      version: '1.0',
      trace_id: trace.id,
      model: trace.model,
      created_at: trace.createdAt,
      updated_at: trace.updatedAt,
    }));

    // Messages
    for (const msg of trace.messages) {
      lines.push(JSON.stringify({
        record_type: 'message',
        ...msg,
      }));
    }

    // Tokens
    for (const token of trace.tokens) {
      lines.push(JSON.stringify({
        record_type: 'token',
        ...token,
      }));
    }

    // Decode steps (attention + MoE data)
    for (const step of trace.steps) {
      lines.push(JSON.stringify({
        record_type: 'step',
        ...step,
      }));
    }

    // Segments
    for (const segment of trace.segments) {
      lines.push(JSON.stringify({
        record_type: 'segment',
        ...segment,
      }));
    }

    // Metrics
    if (trace.metrics) {
      lines.push(JSON.stringify({
        record_type: 'metrics',
        ...trace.metrics,
      }));
    }

    // Download as file
    const content = lines.join('\n');
    const blob = new Blob([content], { type: 'application/x-ndjson' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `trace-${trace.id}-${new Date().toISOString().slice(0, 10)}.jsonl`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  },

  importTraceFromJSONL: (jsonlContent) => {
    try {
      const lines = jsonlContent.trim().split('\n').filter(l => l.trim());

      let header: any = null;
      const messages: Message[] = [];
      const tokens: TokenEntry[] = [];
      const steps: DecodeStep[] = [];
      const segments: Segment[] = [];
      let metrics: any = null;

      for (const line of lines) {
        const obj = JSON.parse(line);
        const { record_type, ...data } = obj;

        switch (record_type) {
          case 'header':
            header = data;
            break;
          case 'message':
            messages.push(data as Message);
            break;
          case 'token':
            tokens.push(data as TokenEntry);
            break;
          case 'step':
            steps.push(data as DecodeStep);
            break;
          case 'segment':
            segments.push(data as Segment);
            break;
          case 'metrics':
            metrics = data;
            break;
        }
      }

      if (!header) {
        throw new Error('Invalid JSONL: missing header');
      }

      // Reconstruct trace
      const trace: TraceSession = {
        id: header.trace_id || `imported-${Date.now()}`,
        model: header.model || 'Unknown',
        createdAt: header.created_at || Date.now(),
        updatedAt: header.updated_at || Date.now(),
        messages,
        tokens,
        segments,
        steps,
        metrics,
        isStreaming: false,
        streamingTokenIndex: -1,
      };

      set({ currentTrace: trace });
    } catch (error) {
      console.error('Failed to import trace:', error);
      throw error;
    }
  },

  // Getters
  getTokenAt: (index) => {
    const trace = get().currentTrace;
    return trace?.tokens[index] ?? null;
  },

  getStepAt: (index) => {
    const trace = get().currentTrace;
    return trace?.steps.find((s) => s.tokenIndex === index) ?? null;
  },

  getSegmentForToken: (index) => {
    const trace = get().currentTrace;
    if (!trace) return null;

    return (
      trace.segments.find(
        (s) => index >= s.startTokenIndex && index < s.endTokenIndex
      ) ?? null
    );
  },

  getTokensInSegment: (segmentId) => {
    const trace = get().currentTrace;
    if (!trace) return [];

    return trace.tokens.filter((t) => t.segmentId === segmentId);
  },
}));
