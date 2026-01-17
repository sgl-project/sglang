import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import {
  QuantizationComparison,
  QuantComparePromptResult,
  getQuantCompareStatus,
  getQuantQualityTier,
} from '../api/types';

interface QuantCompareState {
  // Current comparison data
  comparison: QuantizationComparison | null;

  // History of comparisons
  history: QuantizationComparison[];

  // Loading state
  isLoading: boolean;
  error: string | null;

  // Actions
  loadFromJSON: (jsonString: string) => void;
  loadFromFile: (file: File) => Promise<void>;
  clearComparison: () => void;
  addToHistory: (comparison: QuantizationComparison) => void;
  loadFromHistory: (index: number) => void;
  clearHistory: () => void;
}

// Sample demo data for when no results are loaded
const DEMO_COMPARISON: QuantizationComparison = {
  model: 'Qwen/Qwen3-1.7B',
  quantization: 'SINQ INT4',
  overall_mean_jaccard: 0.6548,
  overall_pass_rate: 0,
  results: [
    {
      prompt: 'What are the three primary colors?',
      mean_jaccard: 0.6383,
      min_jaccard: 0.25,
      max_jaccard: 1.0,
      status: 'WARN',
    },
    {
      prompt: 'Explain gravity in one sentence.',
      mean_jaccard: 0.6727,
      min_jaccard: 0.1765,
      max_jaccard: 1.0,
      status: 'WARN',
    },
    {
      prompt: 'What is the capital of France?',
      mean_jaccard: 0.557,
      min_jaccard: 0.25,
      max_jaccard: 1.0,
      status: 'WARN',
    },
    {
      prompt: 'Name three planets in our solar system.',
      mean_jaccard: 0.6354,
      min_jaccard: 0.3333,
      max_jaccard: 1.0,
      status: 'WARN',
    },
    {
      prompt: 'What does DNA stand for?',
      mean_jaccard: 0.7704,
      min_jaccard: 0.4286,
      max_jaccard: 1.0,
      status: 'WARN',
    },
  ],
  timestamp: Date.now(),
  quality_tier: 'ACCEPTABLE',
};

export const useQuantCompareStore = create<QuantCompareState>()(
  persist(
    (set, get) => ({
      comparison: DEMO_COMPARISON,
      history: [],
      isLoading: false,
      error: null,

      loadFromJSON: (jsonString: string) => {
        try {
          set({ isLoading: true, error: null });

          const data = JSON.parse(jsonString);

          // Normalize the data structure
          const comparison: QuantizationComparison = {
            model: data.model || 'Unknown Model',
            quantization: data.quantization || 'Unknown',
            overall_mean_jaccard: data.overall_mean_jaccard ?? 0,
            overall_pass_rate:
              data.overall_pass_rate ??
              (data.results
                ? data.results.filter((r: QuantComparePromptResult) => r.status === 'PASS').length /
                  (data.results.length || 1)
                : 0),
            results: (data.results || []).map((r: Partial<QuantComparePromptResult>) => ({
              prompt: r.prompt || '',
              bf16Response: r.bf16Response,
              int4Response: r.int4Response,
              mean_jaccard: r.mean_jaccard ?? 0,
              min_jaccard: r.min_jaccard ?? 0,
              max_jaccard: r.max_jaccard ?? 0,
              std_jaccard: r.std_jaccard,
              tokens_compared: r.tokens_compared,
              divergent_count: r.divergent_count,
              divergent_tokens: r.divergent_tokens,
              per_token_jaccard: r.per_token_jaccard,
              status: r.status || getQuantCompareStatus(r.mean_jaccard ?? 0),
            })),
            timestamp: data.timestamp || Date.now(),
            quality_tier: data.quality_tier || getQuantQualityTier(data.overall_mean_jaccard ?? 0),
          };

          // Add to history
          const history = [...get().history];
          history.unshift(comparison);
          if (history.length > 10) history.pop();

          set({ comparison, history, isLoading: false });
        } catch (e) {
          set({
            error: `Failed to parse JSON: ${e instanceof Error ? e.message : 'Unknown error'}`,
            isLoading: false,
          });
        }
      },

      loadFromFile: async (file: File) => {
        try {
          set({ isLoading: true, error: null });
          const text = await file.text();
          get().loadFromJSON(text);
        } catch (e) {
          set({
            error: `Failed to read file: ${e instanceof Error ? e.message : 'Unknown error'}`,
            isLoading: false,
          });
        }
      },

      clearComparison: () => {
        set({ comparison: null, error: null });
      },

      addToHistory: (comparison: QuantizationComparison) => {
        const history = [...get().history];
        history.unshift(comparison);
        if (history.length > 10) history.pop();
        set({ history });
      },

      loadFromHistory: (index: number) => {
        const history = get().history;
        if (index >= 0 && index < history.length) {
          set({ comparison: history[index], error: null });
        }
      },

      clearHistory: () => {
        set({ history: [] });
      },
    }),
    {
      name: 'quant-compare-storage',
      partialize: (state) => ({ history: state.history }),
    }
  )
);
