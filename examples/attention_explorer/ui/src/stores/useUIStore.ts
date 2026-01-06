import { create } from 'zustand';
import { Program, View } from '../api/types';
import { TokenAttentionDetail } from '../api/client';

interface UIState {
  view: View;
  program: Program;
  selectedTokenIndex: number | null;
  hoveredTokenIndex: number | null;
  selectedLayerId: number;
  segment: 'think' | 'output';
  isConnected: boolean;
  modelName: string;
  baseUrl: string;

  // Token detail state
  tokenDetail: TokenAttentionDetail | null;
  tokenDetailLoading: boolean;
  tokenDetailError: string | null;

  setView: (v: View) => void;
  setProgram: (p: Program) => void;
  selectToken: (idx: number | null) => void;
  hoverToken: (idx: number | null) => void;
  setLayer: (id: number) => void;
  setSegment: (s: 'think' | 'output') => void;
  setConnected: (connected: boolean) => void;
  setModelName: (name: string) => void;
  setBaseUrl: (url: string) => void;

  // Token detail actions
  setTokenDetail: (detail: TokenAttentionDetail | null) => void;
  setTokenDetailLoading: (loading: boolean) => void;
  setTokenDetailError: (error: string | null) => void;
}

export const useUIStore = create<UIState>((set) => ({
  view: 'chat',
  program: 'discovery',
  selectedTokenIndex: null,
  hoveredTokenIndex: null,
  selectedLayerId: -1,
  segment: 'output',
  isConnected: false,
  modelName: 'Not connected',
  baseUrl: 'http://localhost:30000',

  tokenDetail: null,
  tokenDetailLoading: false,
  tokenDetailError: null,

  setView: (view) => set({ view }),
  setProgram: (program) => set({ program }),
  selectToken: (selectedTokenIndex) => set({ selectedTokenIndex, tokenDetail: null, tokenDetailError: null }),
  hoverToken: (hoveredTokenIndex) => set({ hoveredTokenIndex }),
  setLayer: (selectedLayerId) => set({ selectedLayerId }),
  setSegment: (segment) => set({ segment }),
  setConnected: (isConnected) => set({ isConnected }),
  setModelName: (modelName) => set({ modelName }),
  setBaseUrl: (baseUrl) => set({ baseUrl }),

  setTokenDetail: (tokenDetail) => set({ tokenDetail, tokenDetailLoading: false }),
  setTokenDetailLoading: (tokenDetailLoading) => set({ tokenDetailLoading }),
  setTokenDetailError: (tokenDetailError) => set({ tokenDetailError, tokenDetailLoading: false }),
}));
