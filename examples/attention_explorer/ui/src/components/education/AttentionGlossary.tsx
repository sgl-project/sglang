/**
 * Educational Glossary and Tooltips for Attention Explorer
 *
 * Provides visual, intuitive explanations for all attention concepts.
 * Every technical term has:
 * - Simple one-liner
 * - Visual analogy with icons
 * - Interactive example
 * - "Why it matters" for practical understanding
 */

import React, { useState } from 'react';

// =============================================================================
// GLOSSARY DATA - Visual-First Explanations
// =============================================================================

export interface GlossaryEntry {
  term: string;
  simple: string;           // One-line explanation
  visual: string;           // Visual analogy
  icon: string;             // Emoji/icon for quick recognition
  color: string;            // Associated color
  example: {
    scenario: string;
    interpretation: string;
  };
  whyItMatters: string;
  relatedTerms: string[];
}

export const GLOSSARY: Record<string, GlossaryEntry> = {
  // -------------------------------------------------------------------------
  // CORE CONCEPTS
  // -------------------------------------------------------------------------
  attention: {
    term: "Attention",
    simple: "How the model decides which previous words to 'look at' when predicting the next word",
    visual: "Imagine a spotlight sweeping over a page - attention is which words the spotlight highlights",
    icon: "🔦",
    color: "#FFD700",
    example: {
      scenario: "When writing 'The cat sat on the ___'",
      interpretation: "The model's attention spotlight shines brightest on 'cat' (what sat?) and 'on' (where?) to predict 'mat'"
    },
    whyItMatters: "Attention reveals the model's reasoning process - which connections it's making",
    relatedTerms: ["attention_score", "attention_entropy", "manifold_zone"]
  },

  attention_score: {
    term: "Attention Score",
    simple: "A number (0-100%) showing how much the model is 'looking at' a specific word",
    visual: "Like a brightness dial - 0% is dark (ignored), 100% is blindingly bright (fully focused)",
    icon: "💡",
    color: "#FFA500",
    example: {
      scenario: "Score of 0.45 (45%) to word 'cat'",
      interpretation: "Nearly half the model's attention budget is spent on 'cat' - it's very important for the current prediction"
    },
    whyItMatters: "High scores reveal which words the model considers most important",
    relatedTerms: ["attention", "topk_attention", "attention_mass"]
  },

  rope: {
    term: "RoPE (Rotary Position Embedding)",
    simple: "How the model knows WHERE words are in the sequence - their position",
    visual: "Like a clock: each word gets a unique 'time' (angle). Word 1 is at 1 o'clock, word 10 at 10 o'clock",
    icon: "🧭",
    color: "#9C27B0",
    example: {
      scenario: "Words at positions 5 and 6 vs positions 5 and 500",
      interpretation: "Adjacent words (5,6) have similar 'clock angles' - the model knows they're neighbors. Words 5 and 500 have very different angles - the model knows they're far apart"
    },
    whyItMatters: "RoPE is why models can understand 'the red car' differently than 'the car red' - position matters!",
    relatedTerms: ["de_rotation", "positional_attention", "semantic_attention"]
  },

  de_rotation: {
    term: "De-Rotation (Pure Semantics)",
    simple: "Mathematically removing position to see attention based only on MEANING",
    visual: "Like removing tinted sunglasses - raw attention has a 'position tint', de-rotation removes it to see true colors",
    icon: "👓",
    color: "#00BCD4",
    example: {
      scenario: "Word 'conclusion' attends to 'hypothesis' (500 words earlier)",
      interpretation: "Raw attention: moderate (they're far). De-rotated: VERY HIGH - they're semantically connected even across distance"
    },
    whyItMatters: "Reveals if the model understands concepts or just uses proximity shortcuts",
    relatedTerms: ["rope", "semantic_attention", "rotational_variance"]
  },

  // -------------------------------------------------------------------------
  // ATTENTION TYPES
  // -------------------------------------------------------------------------
  semantic_attention: {
    term: "Semantic Attention",
    simple: "Attention based on MEANING - words that are conceptually related",
    visual: "Like a mind map - connecting related ideas regardless of where they appear on the page",
    icon: "🧠",
    color: "#4CAF50",
    example: {
      scenario: "'Doctor' attending to 'patient' and 'hospital'",
      interpretation: "These words are semantically linked - the model understands their conceptual relationship"
    },
    whyItMatters: "High semantic attention = genuine understanding. The model gets the concepts.",
    relatedTerms: ["de_rotation", "positional_attention", "manifold_zone"]
  },

  positional_attention: {
    term: "Positional Attention",
    simple: "Attention based on PROXIMITY - nearby words get more attention",
    visual: "Like talking at a party - you naturally pay more attention to people standing close to you",
    icon: "📍",
    color: "#FF5722",
    example: {
      scenario: "'running' attending to 'is' right before it",
      interpretation: "Grammar is local - 'is running' works because these words are adjacent. Position matters for syntax."
    },
    whyItMatters: "Normal for grammar, but suspicious for reasoning. Position shortcuts can hide lack of understanding.",
    relatedTerms: ["rope", "semantic_attention", "local_mass"]
  },

  // -------------------------------------------------------------------------
  // MANIFOLD ZONES
  // -------------------------------------------------------------------------
  syntax_floor: {
    term: "Syntax Floor",
    simple: "Model is processing grammar and local word relationships",
    visual: "Like proofreading - checking if words fit together grammatically",
    icon: "📝",
    color: "#4CAF50",
    example: {
      scenario: "Attention from 'dogs' to 'The' and 'are'",
      interpretation: "Subject-verb agreement: 'The dogs ARE' not 'The dogs IS'. Model is doing grammar."
    },
    whyItMatters: "Syntax floor = foundational language processing. Fast and reliable.",
    relatedTerms: ["local_mass", "positional_attention", "attention_entropy"]
  },

  semantic_bridge: {
    term: "Semantic Bridge",
    simple: "Model is connecting meaning across medium distances",
    visual: "Like building a bridge between two riverbanks - linking related concepts",
    icon: "🌉",
    color: "#2196F3",
    example: {
      scenario: "Attention from 'therefore' to 'because' (50 words earlier)",
      interpretation: "The model is linking cause and effect across a paragraph - understanding argument structure"
    },
    whyItMatters: "Semantic bridges show comprehension - the model is following the thread of meaning.",
    relatedTerms: ["semantic_attention", "mid_mass", "de_rotation"]
  },

  structure_ripple: {
    term: "Structure Ripple",
    simple: "Model is reasoning across the entire context - big picture thinking",
    visual: "Like ripples in a pond spreading outward - attention reaches far across the document",
    icon: "🌊",
    color: "#FF9800",
    example: {
      scenario: "Attention from answer token to question keywords (2000 words earlier)",
      interpretation: "The model is actively retrieving and using information from far back - true long-range reasoning"
    },
    whyItMatters: "Structure ripples indicate complex reasoning, retrieval, or document-level understanding.",
    relatedTerms: ["long_mass", "semantic_attention", "attention_entropy"]
  },

  // -------------------------------------------------------------------------
  // METRICS
  // -------------------------------------------------------------------------
  attention_entropy: {
    term: "Attention Entropy",
    simple: "How spread out (diffuse) vs focused (concentrated) the attention is",
    visual: "Spotlight vs floodlight: Low entropy = laser-focused spotlight. High entropy = dim floodlight everywhere.",
    icon: "🎯",
    color: "#E91E63",
    example: {
      scenario: "Entropy 0.5 vs Entropy 4.0",
      interpretation: "0.5: Model is CERTAIN, focusing on 1-2 tokens. 4.0: Model is UNCERTAIN, spreading attention across many tokens."
    },
    whyItMatters: "Low entropy = confident decision. High entropy = weighing many options (often during reasoning).",
    relatedTerms: ["attention_score", "attention", "manifold_zone"]
  },

  rotational_variance: {
    term: "Rotational Variance",
    simple: "How much POSITION affects the attention pattern",
    visual: "Like measuring how much your opinion changes based on WHERE you meet someone",
    icon: "🔄",
    color: "#673AB7",
    example: {
      scenario: "Variance 0.1 vs 0.8",
      interpretation: "0.1: Position barely matters - this is SEMANTIC attention. 0.8: Position dominates - this is POSITIONAL attention."
    },
    whyItMatters: "Distinguishes genuine understanding from positional shortcuts.",
    relatedTerms: ["de_rotation", "semantic_attention", "positional_attention"]
  },

  local_mass: {
    term: "Local Mass",
    simple: "How much attention goes to NEARBY words (within ~16 tokens)",
    visual: "Like the gravity of nearby planets - how much do neighbors pull?",
    icon: "🏠",
    color: "#8BC34A",
    example: {
      scenario: "Local mass 0.7 (70%)",
      interpretation: "70% of attention stays local. Model is focused on immediate context - likely doing syntax/grammar."
    },
    whyItMatters: "High local mass = syntax processing. Low local mass = looking further for context.",
    relatedTerms: ["syntax_floor", "positional_attention", "mid_mass"]
  },

  long_mass: {
    term: "Long-Range Mass",
    simple: "How much attention reaches FAR back (256+ tokens ago)",
    visual: "Like remembering something from the beginning of a long conversation",
    icon: "🔭",
    color: "#FF5722",
    example: {
      scenario: "Long mass 0.4 (40%)",
      interpretation: "40% of attention reaches way back - the model is actively using distant context for this prediction."
    },
    whyItMatters: "High long mass = retrieval or long-range reasoning. The model remembers and uses earlier info.",
    relatedTerms: ["structure_ripple", "semantic_attention", "attention_entropy"]
  },

  // -------------------------------------------------------------------------
  // SPECIAL PATTERNS
  // -------------------------------------------------------------------------
  attention_sink: {
    term: "Attention Sink",
    simple: "First few tokens that absorb 'leftover' attention when the model is uncertain",
    visual: "Like a drain in a sink - excess water (attention) flows there when it has nowhere else to go",
    icon: "🕳️",
    color: "#607D8B",
    example: {
      scenario: "Token <BOS> receives 35% attention",
      interpretation: "The model is uncertain! It's dumping attention to the beginning because it doesn't have a clear target."
    },
    whyItMatters: "High sink attention = uncertainty. The model may be confused or the input may be ambiguous.",
    relatedTerms: ["sinq_anchor", "attention_entropy", "attention"]
  },

  sinq_anchor: {
    term: "Sinq Anchor",
    simple: "Using the attention sink as a REFERENCE POINT instead of just filtering it",
    visual: "Like using True North on a compass - the sink becomes (0,0) and we measure direction from there",
    icon: "⚓",
    color: "#795548",
    example: {
      scenario: "Sink attention vector: [0.8, 0.2], Target attention: [0.3, 0.9]",
      interpretation: "Relative to the sink, this attention is pointing in a different 'semantic direction' - useful for clustering patterns."
    },
    whyItMatters: "Provides a stable reference point for comparing attention across different positions.",
    relatedTerms: ["attention_sink", "de_rotation", "manifold_zone"]
  },

  topk_attention: {
    term: "Top-K Attention",
    simple: "Only showing the K highest attention scores (e.g., top 10)",
    visual: "Like a 'Top 10' chart - we focus on the most important connections, ignoring tiny scores",
    icon: "🏆",
    color: "#FFC107",
    example: {
      scenario: "Top-10 attention with scores [0.25, 0.18, 0.12, ...]",
      interpretation: "These 10 tokens capture the most important attention. The remaining hundreds of tokens share the leftover ~30%."
    },
    whyItMatters: "Makes attention interpretable - you can't visualize attention to thousands of tokens!",
    relatedTerms: ["attention_score", "attention", "attention_mass"]
  },

  fingerprint: {
    term: "Attention Fingerprint",
    simple: "A compact 'signature' of an attention pattern (20 numbers)",
    visual: "Like a thumbprint - unique identifier that captures the essence of the attention shape",
    icon: "🔑",
    color: "#009688",
    example: {
      scenario: "Fingerprint: [0.3, 0.2, 0.5, 2.1, ...]",
      interpretation: "This fingerprint encodes: 30% local, 20% mid-range, 50% long-range attention, with entropy 2.1..."
    },
    whyItMatters: "Fingerprints enable clustering, routing, and comparing patterns across millions of tokens.",
    relatedTerms: ["manifold_zone", "local_mass", "attention_entropy"]
  }
};

// =============================================================================
// VISUAL COMPONENTS
// =============================================================================

interface TooltipProps {
  term: string;
  children: React.ReactNode;
  showVisual?: boolean;
}

/**
 * Educational tooltip that appears on hover
 * Shows: icon, simple explanation, visual analogy
 */
export const EducationalTooltip: React.FC<TooltipProps> = ({
  term,
  children,
  showVisual = true
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const entry = GLOSSARY[term];

  if (!entry) {
    return <>{children}</>;
  }

  return (
    <div
      className="relative inline-block"
      onMouseEnter={() => setIsOpen(true)}
      onMouseLeave={() => setIsOpen(false)}
    >
      <span
        className="cursor-help border-b border-dotted border-gray-400"
        style={{ borderColor: entry.color }}
      >
        {children}
      </span>

      {isOpen && (
        <div
          className="absolute z-50 w-80 p-4 bg-gray-900 rounded-lg shadow-xl border"
          style={{ borderColor: entry.color, left: '50%', transform: 'translateX(-50%)', top: '100%', marginTop: '8px' }}
        >
          {/* Header */}
          <div className="flex items-center gap-2 mb-2">
            <span className="text-2xl">{entry.icon}</span>
            <span className="font-bold text-white">{entry.term}</span>
          </div>

          {/* Simple explanation */}
          <p className="text-gray-300 text-sm mb-3">{entry.simple}</p>

          {/* Visual analogy */}
          {showVisual && (
            <div className="bg-gray-800 rounded p-2 mb-3">
              <div className="text-xs text-gray-400 mb-1">Visual Analogy:</div>
              <p className="text-sm text-blue-300 italic">{entry.visual}</p>
            </div>
          )}

          {/* Example */}
          <div className="text-xs">
            <div className="text-gray-400">Example:</div>
            <div className="text-green-300">"{entry.example.scenario}"</div>
            <div className="text-gray-300 mt-1">→ {entry.example.interpretation}</div>
          </div>

          {/* Arrow pointer */}
          <div
            className="absolute w-3 h-3 bg-gray-900 transform rotate-45"
            style={{
              borderTop: `1px solid ${entry.color}`,
              borderLeft: `1px solid ${entry.color}`,
              top: '-7px',
              left: '50%',
              marginLeft: '-6px'
            }}
          />
        </div>
      )}
    </div>
  );
};

/**
 * Glossary card for full-page glossary view
 */
export const GlossaryCard: React.FC<{ termKey: string }> = ({ termKey }) => {
  const entry = GLOSSARY[termKey];
  if (!entry) return null;

  return (
    <div
      className="bg-gray-800 rounded-lg p-4 border-l-4"
      style={{ borderLeftColor: entry.color }}
    >
      <div className="flex items-center gap-3 mb-3">
        <span className="text-3xl">{entry.icon}</span>
        <div>
          <h3 className="font-bold text-white text-lg">{entry.term}</h3>
          <p className="text-gray-400 text-sm">{entry.simple}</p>
        </div>
      </div>

      <div className="grid gap-3">
        {/* Visual Analogy */}
        <div className="bg-gray-700 rounded p-3">
          <div className="text-xs text-gray-400 uppercase mb-1">Think of it like...</div>
          <p className="text-blue-300">{entry.visual}</p>
        </div>

        {/* Example */}
        <div className="bg-gray-700 rounded p-3">
          <div className="text-xs text-gray-400 uppercase mb-1">Example</div>
          <p className="text-green-300 font-mono text-sm mb-2">{entry.example.scenario}</p>
          <p className="text-gray-300 text-sm">→ {entry.example.interpretation}</p>
        </div>

        {/* Why it matters */}
        <div className="bg-gray-700 rounded p-3">
          <div className="text-xs text-gray-400 uppercase mb-1">Why it matters</div>
          <p className="text-yellow-300">{entry.whyItMatters}</p>
        </div>

        {/* Related terms */}
        <div className="flex flex-wrap gap-2">
          {entry.relatedTerms.map(related => {
            const relatedEntry = GLOSSARY[related];
            return relatedEntry ? (
              <span
                key={related}
                className="text-xs px-2 py-1 rounded-full"
                style={{ backgroundColor: relatedEntry.color + '30', color: relatedEntry.color }}
              >
                {relatedEntry.icon} {relatedEntry.term}
              </span>
            ) : null;
          })}
        </div>
      </div>
    </div>
  );
};

/**
 * Compact metric display with educational hover
 */
export const MetricWithExplanation: React.FC<{
  term: string;
  value: number;
  format?: 'percent' | 'decimal' | 'entropy';
  size?: 'sm' | 'md' | 'lg';
}> = ({ term, value, format = 'decimal', size = 'md' }) => {
  const entry = GLOSSARY[term];
  if (!entry) return <span>{value}</span>;

  const formatValue = () => {
    switch (format) {
      case 'percent': return `${(value * 100).toFixed(0)}%`;
      case 'entropy': return value.toFixed(2);
      default: return value.toFixed(3);
    }
  };

  const sizeClasses = {
    sm: 'text-sm',
    md: 'text-base',
    lg: 'text-lg font-bold'
  };

  return (
    <EducationalTooltip term={term}>
      <span
        className={`${sizeClasses[size]} font-mono`}
        style={{ color: entry.color }}
      >
        {entry.icon} {formatValue()}
      </span>
    </EducationalTooltip>
  );
};

/**
 * Zone badge with visual indicator
 */
export const ZoneBadge: React.FC<{
  zone: string;
  showExplanation?: boolean;
}> = ({ zone, showExplanation = true }) => {
  const zoneKey = zone.toLowerCase().replace(/ /g, '_');
  const entry = GLOSSARY[zoneKey];

  if (!entry) {
    return <span className="px-2 py-1 bg-gray-700 rounded text-sm">{zone}</span>;
  }

  const badge = (
    <span
      className="inline-flex items-center gap-1 px-3 py-1 rounded-full text-sm font-medium"
      style={{ backgroundColor: entry.color + '30', color: entry.color }}
    >
      {entry.icon} {entry.term}
    </span>
  );

  return showExplanation ? (
    <EducationalTooltip term={zoneKey}>{badge}</EducationalTooltip>
  ) : badge;
};

/**
 * Visual attention interpretation panel
 */
export const AttentionInterpretation: React.FC<{
  pattern: string;
  semanticScore: number;
  positionalScore: number;
  variance: number;
  zone: string;
}> = ({ pattern, semanticScore, positionalScore, variance, zone }) => {
  // Determine the visual representation
  const getPatternVisual = () => {
    switch (pattern) {
      case 'high_semantic_low_positional':
        return {
          icon: '🧠',
          label: 'Semantic Understanding',
          color: '#4CAF50',
          description: 'The model is connecting concepts based on MEANING',
          confidence: 'High confidence in semantic relationship'
        };
      case 'high_positional_low_semantic':
        return {
          icon: '📍',
          label: 'Positional Pattern',
          color: '#FF5722',
          description: 'Attention driven by token PROXIMITY',
          confidence: 'Normal for syntax, verify reasoning'
        };
      case 'high_both':
        return {
          icon: '✅',
          label: 'Strong Signal',
          color: '#2196F3',
          description: 'Both position AND meaning align',
          confidence: 'Very high confidence'
        };
      case 'sink_dominated':
        return {
          icon: '❓',
          label: 'Uncertain',
          color: '#9E9E9E',
          description: 'Attention going to sink tokens',
          confidence: 'Model may be confused'
        };
      default:
        return {
          icon: '➖',
          label: 'Neutral',
          color: '#607D8B',
          description: 'No strong pattern detected',
          confidence: 'Background attention'
        };
    }
  };

  const visual = getPatternVisual();

  return (
    <div className="bg-gray-800 rounded-lg p-4 space-y-4">
      {/* Header */}
      <div className="flex items-center gap-3">
        <span className="text-4xl">{visual.icon}</span>
        <div>
          <h3 className="font-bold text-lg" style={{ color: visual.color }}>
            {visual.label}
          </h3>
          <p className="text-gray-400 text-sm">{visual.description}</p>
        </div>
      </div>

      {/* Visual bar comparison */}
      <div className="space-y-2">
        <div>
          <div className="flex justify-between text-xs text-gray-400 mb-1">
            <span>Semantic</span>
            <span>{(semanticScore * 100).toFixed(0)}%</span>
          </div>
          <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
            <div
              className="h-full rounded-full transition-all"
              style={{
                width: `${semanticScore * 100}%`,
                backgroundColor: '#4CAF50'
              }}
            />
          </div>
        </div>

        <div>
          <div className="flex justify-between text-xs text-gray-400 mb-1">
            <span>Positional</span>
            <span>{(positionalScore * 100).toFixed(0)}%</span>
          </div>
          <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
            <div
              className="h-full rounded-full transition-all"
              style={{
                width: `${positionalScore * 100}%`,
                backgroundColor: '#FF5722'
              }}
            />
          </div>
        </div>
      </div>

      {/* Zone */}
      <div className="flex items-center justify-between">
        <span className="text-gray-400 text-sm">Zone:</span>
        <ZoneBadge zone={zone} />
      </div>

      {/* Confidence */}
      <div className="text-sm text-center py-2 px-3 bg-gray-700 rounded" style={{ color: visual.color }}>
        {visual.confidence}
      </div>
    </div>
  );
};

// =============================================================================
// FULL GLOSSARY PAGE
// =============================================================================

export const GlossaryPage: React.FC = () => {
  const [filter, setFilter] = useState<string>('all');

  const categories = {
    all: 'All Terms',
    core: 'Core Concepts',
    zones: 'Manifold Zones',
    metrics: 'Metrics',
    special: 'Special Patterns'
  };

  const categoryMap: Record<string, string[]> = {
    core: ['attention', 'attention_score', 'rope', 'de_rotation', 'semantic_attention', 'positional_attention'],
    zones: ['syntax_floor', 'semantic_bridge', 'structure_ripple'],
    metrics: ['attention_entropy', 'rotational_variance', 'local_mass', 'long_mass'],
    special: ['attention_sink', 'sinq_anchor', 'topk_attention', 'fingerprint']
  };

  const getFilteredTerms = () => {
    if (filter === 'all') return Object.keys(GLOSSARY);
    return categoryMap[filter] || [];
  };

  return (
    <div className="p-6 max-w-4xl mx-auto">
      <h1 className="text-3xl font-bold text-white mb-2">Attention Explorer Glossary</h1>
      <p className="text-gray-400 mb-6">
        Visual explanations for understanding transformer attention patterns
      </p>

      {/* Category filters */}
      <div className="flex flex-wrap gap-2 mb-6">
        {Object.entries(categories).map(([key, label]) => (
          <button
            key={key}
            onClick={() => setFilter(key)}
            className={`px-4 py-2 rounded-full text-sm transition-all ${
              filter === key
                ? 'bg-blue-500 text-white'
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
          >
            {label}
          </button>
        ))}
      </div>

      {/* Glossary grid */}
      <div className="grid gap-4 md:grid-cols-2">
        {getFilteredTerms().map(termKey => (
          <GlossaryCard key={termKey} termKey={termKey} />
        ))}
      </div>
    </div>
  );
};

export default EducationalTooltip;
