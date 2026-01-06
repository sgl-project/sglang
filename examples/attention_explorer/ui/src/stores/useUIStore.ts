import { create } from 'zustand';
import { Program, View } from '../api/types';
import { TokenAttentionDetail } from '../api/client';

export type DrawerState = 'closed' | 'hovering' | 'pinned';
export type DrawerTab = 'links' | 'signal' | 'moe';
export type MetricScope = 'all' | 'think' | 'output' | 'boundary';

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

  // Token Lens Drawer state
  drawerState: DrawerState;
  drawerTokenIndex: number | null;
  drawerTab: DrawerTab;
  hoverTimeoutId: ReturnType<typeof setTimeout> | null;

  // Metric scope for InsightPanel
  metricScope: MetricScope;

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

  // Drawer actions
  openDrawerHover: (tokenIndex: number) => void;
  closeDrawerHover: () => void;
  pinDrawer: (tokenIndex: number) => void;
  unpinDrawer: () => void;
  setDrawerTab: (tab: DrawerTab) => void;
  clearHoverTimeout: () => void;

  // Metric scope actions
  setMetricScope: (scope: MetricScope) => void;
}

const HOVER_CLOSE_DELAY = 300; // ms before drawer closes after mouse leaves
const HOVER_OPEN_DELAY = 150; // ms before drawer opens on hover (prevents flash during quick clicks)

export const useUIStore = create<UIState>((set, get) => ({
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

  // Drawer state
  drawerState: 'closed',
  drawerTokenIndex: null,
  drawerTab: 'links',
  hoverTimeoutId: null,

  // Metric scope
  metricScope: 'all',

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

  // Drawer actions
  openDrawerHover: (tokenIndex) => {
    const { drawerState, hoverTimeoutId } = get();
    // Clear any pending close timeout
    if (hoverTimeoutId) {
      clearTimeout(hoverTimeoutId);
    }
    // Always set hovered state for visual feedback
    set({ hoveredTokenIndex: tokenIndex });
    // Don't open drawer if pinned (but hovered state is still shown)
    if (drawerState === 'pinned') {
      return;
    }
    // Delay drawer opening to prevent interference with quick clicks
    const openTimeoutId = setTimeout(() => {
      const current = get();
      // Only open if still hovering this token and not pinned
      if (current.hoveredTokenIndex === tokenIndex && current.drawerState !== 'pinned') {
        set({
          drawerState: 'hovering',
          drawerTokenIndex: tokenIndex,
          hoverTimeoutId: null,
        });
      }
    }, HOVER_OPEN_DELAY);
    set({ hoverTimeoutId: openTimeoutId });
  },

  closeDrawerHover: () => {
    const { drawerState, hoverTimeoutId } = get();
    // Don't close if pinned
    if (drawerState === 'pinned') {
      return;
    }
    // Clear any existing timeout
    if (hoverTimeoutId) {
      clearTimeout(hoverTimeoutId);
    }
    // Set a delayed close
    const timeoutId = setTimeout(() => {
      const current = get();
      // Only close if still in hovering state
      if (current.drawerState === 'hovering') {
        set({
          drawerState: 'closed',
          drawerTokenIndex: null,
          hoveredTokenIndex: null,
          hoverTimeoutId: null,
        });
      }
    }, HOVER_CLOSE_DELAY);
    set({ hoverTimeoutId: timeoutId });
  },

  pinDrawer: (tokenIndex) => {
    const { hoverTimeoutId, drawerTokenIndex, drawerState } = get();
    // Clear any pending close timeout
    if (hoverTimeoutId) {
      clearTimeout(hoverTimeoutId);
    }
    // If clicking the same pinned token, unpin
    if (drawerState === 'pinned' && drawerTokenIndex === tokenIndex) {
      set({
        drawerState: 'closed',
        drawerTokenIndex: null,
        selectedTokenIndex: null,
        hoverTimeoutId: null,
      });
      return;
    }
    // Pin to this token
    set({
      drawerState: 'pinned',
      drawerTokenIndex: tokenIndex,
      selectedTokenIndex: tokenIndex,
      hoverTimeoutId: null,
    });
  },

  unpinDrawer: () => {
    set({
      drawerState: 'closed',
      drawerTokenIndex: null,
      selectedTokenIndex: null,
      hoverTimeoutId: null,
    });
  },

  setDrawerTab: (drawerTab) => set({ drawerTab }),

  clearHoverTimeout: () => {
    const { hoverTimeoutId } = get();
    if (hoverTimeoutId) {
      clearTimeout(hoverTimeoutId);
      set({ hoverTimeoutId: null });
    }
  },

  setMetricScope: (metricScope) => set({ metricScope }),
}));
