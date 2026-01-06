import { create } from 'zustand';
import { Message, AttentionEntry, Fingerprint, MoERoutingEntry } from '../api/types';
import { useTraceStore } from './useTraceStore';

interface SessionState {
  messages: Message[];
  currentTokens: string[];
  currentAttention: Map<number, AttentionEntry>;  // Map step -> attention entry
  currentMoE: MoERoutingEntry[];
  fingerprint: Fingerprint | null;
  inputTokens: string[];
  isStreaming: boolean;

  addMessage: (msg: Message) => void;
  appendToken: (token: string) => void;
  appendAttention: (entry: AttentionEntry) => void;
  appendMoE: (entry: MoERoutingEntry) => void;
  setFingerprint: (fp: Fingerprint | null) => void;
  setInputTokens: (tokens: string[]) => void;
  startStreaming: () => void;
  finishStreaming: () => void;
  clear: () => void;
}

export const useSessionStore = create<SessionState>((set, get) => ({
  messages: [],
  currentTokens: [],
  currentAttention: new Map(),
  currentMoE: [],
  fingerprint: null,
  inputTokens: [],
  isStreaming: false,

  addMessage: (msg) => {
    // Sync user messages to TraceStore
    if (msg.role === 'user') {
      const traceStore = useTraceStore.getState();
      if (traceStore.currentTrace) {
        traceStore.addUserMessage(msg.content);
      }
    }

    set((state) => ({
      messages: [...state.messages, msg],
      currentTokens: msg.role === 'user' ? [] : state.currentTokens,
      currentAttention: msg.role === 'user' ? new Map() : state.currentAttention,
      currentMoE: msg.role === 'user' ? [] : state.currentMoE,
    }));
  },

  appendToken: (token) => {
    const state = get();
    const newTokens = [...state.currentTokens, token];
    const tokenIndex = newTokens.length - 1;

    // Convert map to array for storage in message, matching by token index
    const attentionArray: AttentionEntry[] = [];
    for (let i = 0; i < newTokens.length; i++) {
      const entry = state.currentAttention.get(i);
      if (entry) {
        attentionArray[i] = entry;
      }
    }

    const messages = [...state.messages];
    const lastMsg = messages[messages.length - 1];

    if (lastMsg?.role === 'assistant') {
      messages[messages.length - 1] = {
        ...lastMsg,
        content: lastMsg.content + token,
        tokens: newTokens,
        attention: attentionArray,
      };
    } else {
      messages.push({
        id: `assistant-${Date.now()}`,
        role: 'assistant',
        content: token,
        tokens: [token],
        attention: attentionArray,
        timestamp: Date.now(),
      });
    }

    // Sync to TraceStore
    const traceStore = useTraceStore.getState();
    if (traceStore.currentTrace) {
      const attention = state.currentAttention.get(tokenIndex);
      traceStore.appendToken(token, attention);
    }

    set({ messages, currentTokens: newTokens });
  },

  appendAttention: (entry) =>
    set((state) => {
      // Get step from entry (attention entries have step property)
      // Server sends step starting at 1, so token index = step - 1
      const step = 'step' in entry ? (entry as any).step :
                   'decode_step' in entry ? (entry as any).decode_step :
                   state.currentAttention.size + 1;  // Fallback: next index + 1

      // Guard against invalid step values (step should be >= 1)
      const tokenIndex = Math.max(0, step - 1);

      const newAttention = new Map(state.currentAttention);
      newAttention.set(tokenIndex, entry);

      // Update the message with new attention data
      const messages = [...state.messages];
      const lastMsg = messages[messages.length - 1];

      if (lastMsg?.role === 'assistant') {
        const attentionArray: AttentionEntry[] = [...(lastMsg.attention || [])];
        attentionArray[tokenIndex] = entry;
        messages[messages.length - 1] = {
          ...lastMsg,
          attention: attentionArray,
        };
      }

      return { messages, currentAttention: newAttention };
    }),

  appendMoE: (entry) =>
    set((state) => ({
      currentMoE: [...state.currentMoE, entry],
    })),

  setFingerprint: (fp) => set({ fingerprint: fp }),

  setInputTokens: (tokens) => set({ inputTokens: tokens }),

  startStreaming: () => {
    // Start assistant message in TraceStore
    const traceStore = useTraceStore.getState();
    if (traceStore.currentTrace) {
      traceStore.startAssistantMessage();
    }
    set({ isStreaming: true });
  },

  finishStreaming: () => {
    // Finish assistant message in TraceStore
    const traceStore = useTraceStore.getState();
    if (traceStore.currentTrace) {
      traceStore.finishAssistantMessage();
    }
    set({ isStreaming: false });
  },

  clear: () => {
    // Also clear TraceStore
    const traceStore = useTraceStore.getState();
    traceStore.clearTrace();

    set({
      messages: [],
      currentTokens: [],
      currentAttention: new Map(),
      currentMoE: [],
      fingerprint: null,
      inputTokens: [],
      isStreaming: false,
    });
  },
}));
