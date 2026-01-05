import { create } from 'zustand';
import { Message, AttentionEntry, Fingerprint, MoERoutingEntry } from '../api/types';

interface SessionState {
  messages: Message[];
  currentTokens: string[];
  currentAttention: AttentionEntry[];
  currentMoE: MoERoutingEntry[];
  fingerprint: Fingerprint | null;
  inputTokens: string[];

  addMessage: (msg: Message) => void;
  appendToken: (token: string, attention: AttentionEntry | null) => void;
  appendMoE: (entry: MoERoutingEntry) => void;
  setFingerprint: (fp: Fingerprint | null) => void;
  setInputTokens: (tokens: string[]) => void;
  clear: () => void;
}

export const useSessionStore = create<SessionState>((set, get) => ({
  messages: [],
  currentTokens: [],
  currentAttention: [],
  currentMoE: [],
  fingerprint: null,
  inputTokens: [],

  addMessage: (msg) =>
    set((state) => ({
      messages: [...state.messages, msg],
      currentTokens: msg.role === 'user' ? [] : state.currentTokens,
      currentAttention: msg.role === 'user' ? [] : state.currentAttention,
      currentMoE: msg.role === 'user' ? [] : state.currentMoE,
    })),

  appendToken: (token, attention) =>
    set((state) => {
      const newTokens = [...state.currentTokens, token];
      const newAttention = attention
        ? [...state.currentAttention, attention]
        : state.currentAttention;

      const messages = [...state.messages];
      const lastMsg = messages[messages.length - 1];

      if (lastMsg?.role === 'assistant') {
        messages[messages.length - 1] = {
          ...lastMsg,
          content: lastMsg.content + token,
          tokens: newTokens,
          attention: newAttention,
        };
      } else {
        messages.push({
          id: `assistant-${Date.now()}`,
          role: 'assistant',
          content: token,
          tokens: [token],
          attention: attention ? [attention] : [],
          timestamp: Date.now(),
        });
      }

      return { messages, currentTokens: newTokens, currentAttention: newAttention };
    }),

  appendMoE: (entry) =>
    set((state) => ({
      currentMoE: [...state.currentMoE, entry],
    })),

  setFingerprint: (fp) => set({ fingerprint: fp }),

  setInputTokens: (tokens) => set({ inputTokens: tokens }),

  clear: () =>
    set({
      messages: [],
      currentTokens: [],
      currentAttention: [],
      currentMoE: [],
      fingerprint: null,
      inputTokens: [],
    }),
}));
