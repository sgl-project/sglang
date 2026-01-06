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
  classifyManifold,
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
        const step: DecodeStep = {
          tokenIndex,
          attention,
          moe,
          fingerprint: fp || undefined,
          manifoldZone: fp ? classifyManifold(fp) : undefined,
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
