import { create } from 'zustand';
import { Program, View } from '../api/types';

interface UIState {
  view: View;
  program: Program;
  selectedTokenIndex: number | null;
  hoveredTokenIndex: number | null;
  selectedLayerId: number;
  segment: 'think' | 'output';
  isConnected: boolean;
  modelName: string;

  setView: (v: View) => void;
  setProgram: (p: Program) => void;
  selectToken: (idx: number | null) => void;
  hoverToken: (idx: number | null) => void;
  setLayer: (id: number) => void;
  setSegment: (s: 'think' | 'output') => void;
  setConnected: (connected: boolean) => void;
  setModelName: (name: string) => void;
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

  setView: (view) => set({ view }),
  setProgram: (program) => set({ program }),
  selectToken: (selectedTokenIndex) => set({ selectedTokenIndex }),
  hoverToken: (hoveredTokenIndex) => set({ hoveredTokenIndex }),
  setLayer: (selectedLayerId) => set({ selectedLayerId }),
  setSegment: (segment) => set({ segment }),
  setConnected: (isConnected) => set({ isConnected }),
  setModelName: (modelName) => set({ modelName }),
}));
