/**
 * De-Rotation Visualizer
 *
 * Interactive visual component showing:
 * - Raw attention (with position)
 * - De-rotated attention (pure semantics)
 * - Visual comparison with intuitive explanations
 */

import React, { useState, useMemo } from 'react';
import { EducationalTooltip, ZoneBadge, GLOSSARY } from './AttentionGlossary';

// =============================================================================
// TYPES
// =============================================================================

interface TokenAttention {
  token: string;
  position: number;
  rawScore: number;
  semanticScore: number;
  positionalBias: number;
}

interface DerotationData {
  queryToken: string;
  queryPosition: number;
  attentions: TokenAttention[];
  rotationalVariance: number;
  semanticEntropy: number;
  manifoldZone: string;
  pattern: string;
}

// =============================================================================
// VISUAL EXPLANATIONS
// =============================================================================

const PATTERN_EXPLANATIONS = {
  high_semantic_low_positional: {
    title: "Semantic Connection",
    icon: "🧠",
    color: "#4CAF50",
    simple: "The model is connecting these words based on their MEANING, not just because they're nearby.",
    visual: (
      <div className="flex items-center gap-4 p-3 bg-green-900/30 rounded-lg">
        <div className="text-4xl">📖</div>
        <div className="flex-1">
          <div className="text-sm text-gray-300">Like reading a story and connecting:</div>
          <div className="text-green-400 font-mono mt-1">
            "The hero...[500 words]...saved the day"
          </div>
          <div className="text-xs text-gray-400 mt-1">
            ↑ You connect "hero" to "saved" because of MEANING, not position
          </div>
        </div>
      </div>
    ),
    interpretation: "This is genuine understanding. The model recognizes conceptual relationships across distance."
  },
  high_positional_low_semantic: {
    title: "Positional Pattern",
    icon: "📍",
    color: "#FF5722",
    simple: "The model is mostly paying attention because these words are NEARBY, not because of meaning.",
    visual: (
      <div className="flex items-center gap-4 p-3 bg-orange-900/30 rounded-lg">
        <div className="text-4xl">✏️</div>
        <div className="flex-1">
          <div className="text-sm text-gray-300">Like checking grammar:</div>
          <div className="text-orange-400 font-mono mt-1">
            "The dogs are running"
          </div>
          <div className="text-xs text-gray-400 mt-1">
            ↑ "are" attends to "dogs" because they're adjacent (subject-verb)
          </div>
        </div>
      </div>
    ),
    interpretation: "Normal for syntax/grammar. Question it if you expected reasoning."
  },
  high_both: {
    title: "Strong Aligned Signal",
    icon: "✅",
    color: "#2196F3",
    simple: "These words are BOTH nearby AND semantically related. Very confident signal!",
    visual: (
      <div className="flex items-center gap-4 p-3 bg-blue-900/30 rounded-lg">
        <div className="text-4xl">🎯</div>
        <div className="flex-1">
          <div className="text-sm text-gray-300">Like a perfect match:</div>
          <div className="text-blue-400 font-mono mt-1">
            "The red apple"
          </div>
          <div className="text-xs text-gray-400 mt-1">
            ↑ "red" and "apple" are adjacent AND semantically linked
          </div>
        </div>
      </div>
    ),
    interpretation: "High confidence. Both position and meaning support this attention."
  },
  sink_dominated: {
    title: "Uncertainty (Sink Pattern)",
    icon: "❓",
    color: "#9E9E9E",
    simple: "The model is dumping attention to the beginning - it's UNCERTAIN about what to focus on.",
    visual: (
      <div className="flex items-center gap-4 p-3 bg-gray-700/50 rounded-lg">
        <div className="text-4xl">🕳️</div>
        <div className="flex-1">
          <div className="text-sm text-gray-300">Like when you're confused:</div>
          <div className="text-gray-400 font-mono mt-1">
            "[BOS] system... hmm, not sure what to focus on"
          </div>
          <div className="text-xs text-gray-400 mt-1">
            ↑ Attention flows to "safe" beginning tokens when uncertain
          </div>
        </div>
      </div>
    ),
    interpretation: "The model may be confused. Check if the input is ambiguous."
  },
  low_both: {
    title: "Weak Connection",
    icon: "➖",
    color: "#607D8B",
    simple: "Neither position nor meaning creates a strong connection here.",
    visual: (
      <div className="flex items-center gap-4 p-3 bg-gray-700/30 rounded-lg">
        <div className="text-4xl">🌫️</div>
        <div className="flex-1">
          <div className="text-sm text-gray-300">Like background noise:</div>
          <div className="text-gray-400 font-mono mt-1">
            Unrelated words getting small attention shares
          </div>
          <div className="text-xs text-gray-400 mt-1">
            ↑ This is the "background" - not the main signal
          </div>
        </div>
      </div>
    ),
    interpretation: "Normal background attention. Focus on the tokens with stronger signals."
  }
};

// =============================================================================
// SUB-COMPONENTS
// =============================================================================

/**
 * Visual comparison bar showing raw vs semantic attention
 */
const AttentionComparisonBar: React.FC<{
  token: string;
  position: number;
  rawScore: number;
  semanticScore: number;
  positionalBias: number;
  queryPosition: number;
  isExpanded: boolean;
  onToggle: () => void;
}> = ({
  token,
  position,
  rawScore,
  semanticScore,
  positionalBias,
  queryPosition,
  isExpanded,
  onToggle
}) => {
  const distance = Math.abs(queryPosition - position);

  // Determine which is dominant
  const isDrivenBySemantic = semanticScore > positionalBias;
  const dominantColor = isDrivenBySemantic ? '#4CAF50' : '#FF5722';

  return (
    <div
      className="bg-gray-800 rounded-lg p-3 cursor-pointer hover:bg-gray-750 transition-all"
      onClick={onToggle}
    >
      {/* Header row */}
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className="font-mono text-white bg-gray-700 px-2 py-1 rounded">
            {token}
          </span>
          <span className="text-xs text-gray-400">
            pos {position} ({distance === 0 ? 'same' : `${distance} away`})
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span
            className="text-lg font-bold"
            style={{ color: dominantColor }}
          >
            {(rawScore * 100).toFixed(1)}%
          </span>
          <span className="text-xs px-2 py-1 rounded" style={{
            backgroundColor: isDrivenBySemantic ? '#4CAF5030' : '#FF572230',
            color: dominantColor
          }}>
            {isDrivenBySemantic ? '🧠 Semantic' : '📍 Positional'}
          </span>
        </div>
      </div>

      {/* Visual comparison bars */}
      <div className="space-y-1">
        {/* Raw attention */}
        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-400 w-20">Raw</span>
          <div className="flex-1 h-4 bg-gray-700 rounded-full overflow-hidden">
            <div
              className="h-full rounded-full transition-all"
              style={{
                width: `${rawScore * 100}%`,
                background: `linear-gradient(90deg, #4CAF50 ${semanticScore / rawScore * 100}%, #FF5722 100%)`
              }}
            />
          </div>
        </div>

        {/* Semantic component */}
        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-400 w-20">Semantic</span>
          <div className="flex-1 h-4 bg-gray-700 rounded-full overflow-hidden">
            <div
              className="h-full bg-green-500 rounded-full transition-all"
              style={{ width: `${semanticScore * 100}%` }}
            />
          </div>
        </div>

        {/* Positional bias */}
        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-400 w-20">Positional</span>
          <div className="flex-1 h-4 bg-gray-700 rounded-full overflow-hidden">
            <div
              className="h-full bg-orange-500 rounded-full transition-all"
              style={{ width: `${positionalBias * 100}%` }}
            />
          </div>
        </div>
      </div>

      {/* Expanded explanation */}
      {isExpanded && (
        <div className="mt-3 p-3 bg-gray-700 rounded-lg text-sm">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <div className="text-gray-400 text-xs mb-1">What this means:</div>
              {isDrivenBySemantic ? (
                <p className="text-green-300">
                  This attention is primarily driven by <strong>meaning</strong>.
                  The model sees "{token}" as semantically relevant regardless of its position.
                </p>
              ) : (
                <p className="text-orange-300">
                  This attention is primarily driven by <strong>position</strong>.
                  The model attends to "{token}" because it's {distance < 20 ? 'nearby' : 'at a specific distance'}.
                </p>
              )}
            </div>
            <div>
              <div className="text-gray-400 text-xs mb-1">Numbers breakdown:</div>
              <div className="space-y-1 font-mono text-xs">
                <div>Raw: {(rawScore * 100).toFixed(1)}% of attention</div>
                <div className="text-green-400">Semantic: {(semanticScore * 100).toFixed(1)}%</div>
                <div className="text-orange-400">Positional bias: {(positionalBias * 100).toFixed(1)}%</div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

/**
 * Compass visualization showing semantic vs positional direction
 */
const SemanticCompass: React.FC<{
  semanticScore: number;
  positionalScore: number;
}> = ({ semanticScore, positionalScore }) => {
  // Calculate angle: 0° = pure semantic, 90° = pure positional
  const total = semanticScore + positionalScore;
  const angle = total > 0 ? Math.atan2(positionalScore, semanticScore) * (180 / Math.PI) : 45;

  return (
    <div className="relative w-32 h-32">
      {/* Background circle */}
      <svg viewBox="0 0 100 100" className="w-full h-full">
        {/* Quadrants */}
        <path d="M50,50 L50,0 A50,50 0 0,1 100,50 Z" fill="#4CAF5020" />
        <path d="M50,50 L100,50 A50,50 0 0,1 50,100 Z" fill="#FF572220" />
        <path d="M50,50 L50,100 A50,50 0 0,1 0,50 Z" fill="#FF572220" />
        <path d="M50,50 L0,50 A50,50 0 0,1 50,0 Z" fill="#4CAF5020" />

        {/* Axis lines */}
        <line x1="50" y1="0" x2="50" y2="100" stroke="#333" strokeWidth="1" />
        <line x1="0" y1="50" x2="100" y2="50" stroke="#333" strokeWidth="1" />

        {/* Labels */}
        <text x="50" y="12" textAnchor="middle" fill="#4CAF50" fontSize="8">Semantic</text>
        <text x="88" y="54" textAnchor="middle" fill="#FF5722" fontSize="8">Positional</text>

        {/* Needle */}
        <line
          x1="50"
          y1="50"
          x2={50 + 40 * Math.cos(angle * Math.PI / 180)}
          y2={50 + 40 * Math.sin(angle * Math.PI / 180)}
          stroke="#FFD700"
          strokeWidth="3"
          strokeLinecap="round"
        />

        {/* Center dot */}
        <circle cx="50" cy="50" r="4" fill="#FFD700" />
      </svg>

      {/* Legend */}
      <div className="absolute -bottom-6 left-0 right-0 text-center text-xs text-gray-400">
        {angle < 30 ? '🧠 Semantic-driven' :
         angle > 60 ? '📍 Position-driven' :
         '⚖️ Balanced'}
      </div>
    </div>
  );
};

// =============================================================================
// MAIN COMPONENT
// =============================================================================

export const DerotationVisualizer: React.FC<{
  data: DerotationData;
}> = ({ data }) => {
  const [expandedToken, setExpandedToken] = useState<number | null>(null);
  const [viewMode, setViewMode] = useState<'comparison' | 'raw' | 'semantic'>('comparison');

  // Calculate aggregate metrics
  const avgSemantic = useMemo(() =>
    data.attentions.reduce((sum, a) => sum + a.semanticScore, 0) / data.attentions.length,
    [data.attentions]
  );

  const avgPositional = useMemo(() =>
    data.attentions.reduce((sum, a) => sum + a.positionalBias, 0) / data.attentions.length,
    [data.attentions]
  );

  const patternExplanation = PATTERN_EXPLANATIONS[data.pattern as keyof typeof PATTERN_EXPLANATIONS]
    || PATTERN_EXPLANATIONS.low_both;

  return (
    <div className="bg-gray-900 rounded-xl p-6 space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h2 className="text-xl font-bold text-white flex items-center gap-2">
            <EducationalTooltip term="de_rotation">
              <span>De-Rotation Analysis</span>
            </EducationalTooltip>
            <span className="text-2xl">{patternExplanation.icon}</span>
          </h2>
          <p className="text-gray-400 mt-1">
            Separating <span className="text-green-400">meaning</span> from{' '}
            <span className="text-orange-400">position</span> in attention
          </p>
        </div>
        <ZoneBadge zone={data.manifoldZone} />
      </div>

      {/* Query context */}
      <div className="bg-gray-800 rounded-lg p-4">
        <div className="text-sm text-gray-400 mb-2">Analyzing attention from:</div>
        <div className="flex items-center gap-3">
          <span className="text-2xl font-mono text-white bg-blue-600 px-3 py-1 rounded">
            {data.queryToken}
          </span>
          <span className="text-gray-400">at position {data.queryPosition}</span>
        </div>
      </div>

      {/* Pattern explanation with visual */}
      <div className="rounded-lg p-4" style={{ backgroundColor: patternExplanation.color + '15' }}>
        <div className="flex items-center gap-3 mb-3">
          <span className="text-3xl">{patternExplanation.icon}</span>
          <div>
            <h3 className="font-bold text-lg" style={{ color: patternExplanation.color }}>
              {patternExplanation.title}
            </h3>
            <p className="text-gray-300 text-sm">{patternExplanation.simple}</p>
          </div>
        </div>
        {patternExplanation.visual}
        <p className="text-gray-400 text-sm mt-3 italic">
          {patternExplanation.interpretation}
        </p>
      </div>

      {/* Compass and metrics */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Semantic compass */}
        <div className="bg-gray-800 rounded-lg p-4 flex flex-col items-center">
          <div className="text-sm text-gray-400 mb-2">Attention Direction</div>
          <SemanticCompass
            semanticScore={avgSemantic}
            positionalScore={avgPositional}
          />
        </div>

        {/* Key metrics */}
        <div className="bg-gray-800 rounded-lg p-4 space-y-3">
          <div className="text-sm text-gray-400">Key Metrics</div>

          <EducationalTooltip term="rotational_variance">
            <div className="flex justify-between items-center">
              <span className="text-gray-300">🔄 Rotational Variance</span>
              <span className={`font-mono ${data.rotationalVariance > 0.3 ? 'text-orange-400' : 'text-green-400'}`}>
                {data.rotationalVariance.toFixed(2)}
              </span>
            </div>
          </EducationalTooltip>

          <EducationalTooltip term="attention_entropy">
            <div className="flex justify-between items-center">
              <span className="text-gray-300">🎯 Semantic Entropy</span>
              <span className="font-mono text-purple-400">
                {data.semanticEntropy.toFixed(2)}
              </span>
            </div>
          </EducationalTooltip>

          <div className="flex justify-between items-center">
            <span className="text-gray-300">🧠 Avg Semantic</span>
            <span className="font-mono text-green-400">
              {(avgSemantic * 100).toFixed(1)}%
            </span>
          </div>

          <div className="flex justify-between items-center">
            <span className="text-gray-300">📍 Avg Positional</span>
            <span className="font-mono text-orange-400">
              {(avgPositional * 100).toFixed(1)}%
            </span>
          </div>
        </div>

        {/* Interpretation guide */}
        <div className="bg-gray-800 rounded-lg p-4">
          <div className="text-sm text-gray-400 mb-2">How to Read This</div>
          <div className="space-y-2 text-sm">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-green-500 rounded" />
              <span className="text-gray-300">Green = Semantic (meaning-based)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-orange-500 rounded" />
              <span className="text-gray-300">Orange = Positional (proximity-based)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-yellow-500 rounded" />
              <span className="text-gray-300">Yellow = Combined raw attention</span>
            </div>
            <p className="text-gray-500 text-xs mt-2">
              Click any token to see detailed breakdown
            </p>
          </div>
        </div>
      </div>

      {/* View mode toggle */}
      <div className="flex gap-2">
        {(['comparison', 'raw', 'semantic'] as const).map(mode => (
          <button
            key={mode}
            onClick={() => setViewMode(mode)}
            className={`px-4 py-2 rounded-lg text-sm transition-all ${
              viewMode === mode
                ? 'bg-blue-500 text-white'
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
          >
            {mode === 'comparison' ? '⚖️ Compare' :
             mode === 'raw' ? '📊 Raw' :
             '🧠 Semantic'}
          </button>
        ))}
      </div>

      {/* Token list */}
      <div className="space-y-2">
        <div className="text-sm text-gray-400">
          Token-by-token breakdown (click to expand):
        </div>
        {data.attentions
          .sort((a, b) => b.rawScore - a.rawScore)
          .map((attention, idx) => (
            <AttentionComparisonBar
              key={idx}
              token={attention.token}
              position={attention.position}
              rawScore={attention.rawScore}
              semanticScore={attention.semanticScore}
              positionalBias={attention.positionalBias}
              queryPosition={data.queryPosition}
              isExpanded={expandedToken === idx}
              onToggle={() => setExpandedToken(expandedToken === idx ? null : idx)}
            />
          ))}
      </div>

      {/* Legend / Help */}
      <div className="text-center text-sm text-gray-500 border-t border-gray-700 pt-4">
        <EducationalTooltip term="rope">
          <span className="cursor-help underline decoration-dotted">What is RoPE?</span>
        </EducationalTooltip>
        {' • '}
        <EducationalTooltip term="de_rotation">
          <span className="cursor-help underline decoration-dotted">Why de-rotate?</span>
        </EducationalTooltip>
        {' • '}
        <EducationalTooltip term="semantic_attention">
          <span className="cursor-help underline decoration-dotted">Semantic attention</span>
        </EducationalTooltip>
      </div>
    </div>
  );
};

export default DerotationVisualizer;
