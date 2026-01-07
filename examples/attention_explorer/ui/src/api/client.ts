// SGLang OpenAI-compatible client with attention capture

import {
  AttentionCaptureParams,
  ChatCompletionChunkWithAttention,
  ChatCompletionWithAttention,
  AttentionEntry,
  MoERoutingEntry,
  Program,
} from './types';

// ============================================================================
// CONFIGURATION
// ============================================================================

export interface ClientConfig {
  baseUrl: string;
  model: string;
  program: Program;
  onToken?: (token: string, index: number) => void;
  onAttention?: (entry: AttentionEntry, index: number) => void;
  onMoE?: (entry: MoERoutingEntry, index: number) => void;
  onFinish?: (fullResponse: ChatCompletionWithAttention) => void;
  onError?: (error: Error) => void;
}

const PROGRAM_CONFIGS: Record<Program, AttentionCaptureParams> = {
  prod: {
    return_attention_tokens: true,
    top_k_attention: 10,
  },
  debug: {
    return_attention_tokens: true,
    top_k_attention: 10,
    attention_capture_layer_ids: [7, 15, 23, 31],
    return_moe_routing: true,
    moe_routing_top_k: 2,
  },
  discovery: {
    return_attention_tokens: true,
    top_k_attention: 10,
    attention_sketch_mode: true,
    return_moe_routing: true,
  },
};

// ============================================================================
// STREAMING CLIENT
// ============================================================================

export class AttentionStreamClient {
  private config: ClientConfig;
  private abortController: AbortController | null = null;

  constructor(config: ClientConfig) {
    this.config = config;
  }

  setProgram(program: Program) {
    this.config.program = program;
  }

  abort() {
    this.abortController?.abort();
    this.abortController = null;
  }

  async stream(messages: Array<{ role: string; content: string }>): Promise<void> {
    this.abort();
    this.abortController = new AbortController();

    const captureParams = PROGRAM_CONFIGS[this.config.program];

    const response = await fetch(`${this.config.baseUrl}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: this.config.model,
        messages,
        stream: true,
        ...captureParams,
      }),
      signal: this.abortController.signal,
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`API error: ${response.status} - ${error}`);
    }

    if (!response.body) {
      throw new Error('No response body');
    }

    await this.processStream(response.body);
  }

  private async processStream(body: ReadableStream<Uint8Array>): Promise<void> {
    const reader = body.getReader();
    const decoder = new TextDecoder();

    const buffer = new StreamBuffer();
    let tokenIndex = 0;
    let sseBuffer = '';  // Buffer for incomplete SSE data

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const text = decoder.decode(value, { stream: true });
        sseBuffer += text;

        // Parse complete SSE events from buffer
        const { events, remaining } = parseSSEEventsWithBuffer(sseBuffer);
        sseBuffer = remaining;

        for (const event of events) {
          if (event === '[DONE]') {
            localStorage.setItem('__done_received__', 'true');
            this.config.onFinish?.(buffer.toResponse());
            return;
          }

          try {
            const chunk: ChatCompletionChunkWithAttention = JSON.parse(event);
            const delta = chunk.choices[0]?.delta;
            const choice = chunk.choices[0] as any;

            if (delta?.content) {
              const tokens = tokenizeIncremental(delta.content, buffer.pendingText);
              buffer.pendingText = tokens.pending;

              for (const token of tokens.complete) {
                buffer.tokens.push(token);
                this.config.onToken?.(token, tokenIndex);
                tokenIndex++;
              }
            }

            // Handle attention_token at chunk level (streaming format)
            if (chunk.attention_token) {
              buffer.attention.push(chunk.attention_token);
              this.config.onAttention?.(chunk.attention_token, buffer.attention.length - 1);
            }

            // Handle attention_tokens inside choices (some server versions)
            if (choice?.attention_tokens) {
              for (const attn of choice.attention_tokens) {
                buffer.attention.push(attn);
                this.config.onAttention?.(attn, buffer.attention.length - 1);
              }
            }

            // Handle attention_tokens in delta (newer server versions)
            const deltaAny = delta as any;
            if (deltaAny?.attention_tokens) {
              for (const attn of deltaAny.attention_tokens) {
                buffer.attention.push(attn);
                this.config.onAttention?.(attn, buffer.attention.length - 1);
              }
            }

            if (chunk.moe_routing_step) {
              buffer.moe.push(chunk.moe_routing_step);
              this.config.onMoE?.(chunk.moe_routing_step, buffer.moe.length - 1);
            }

            if (chunk.choices[0]?.finish_reason) {
              buffer.finishReason = chunk.choices[0].finish_reason;
            }
          } catch (e) {
            // Only warn if it looks like a real parse error (not empty/partial data)
            if (event.length > 10) {
              console.warn('[Client] Failed to parse SSE event (len=' + event.length + '):', event.slice(0, 200) + '...', e);
            }
          }
        }
      }
    } finally {
      reader.releaseLock();
    }
  }
}

// ============================================================================
// STREAM BUFFER
// ============================================================================

class StreamBuffer {
  tokens: string[] = [];
  attention: AttentionEntry[] = [];
  moe: MoERoutingEntry[] = [];
  pendingText: string = '';
  finishReason: string | null = null;

  toResponse(): ChatCompletionWithAttention {
    return {
      id: `chat-${Date.now()}`,
      object: 'chat.completion',
      created: Math.floor(Date.now() / 1000),
      model: 'default',
      choices: [
        {
          index: 0,
          message: {
            role: 'assistant',
            content: this.tokens.join(''),
          },
          finish_reason: this.finishReason || 'stop',
        },
      ],
      attention_tokens: this.attention,
      moe_routing: this.moe,
    };
  }
}

// ============================================================================
// SSE PARSING
// ============================================================================

interface SSEParseResult {
  events: string[];
  remaining: string;
}

/**
 * Parse SSE events from a buffer, handling incomplete data that spans chunks.
 * SSE format: "data: {...}\n\n" - events are separated by double newlines.
 */
function parseSSEEventsWithBuffer(buffer: string): SSEParseResult {
  const events: string[] = [];

  // Split by double newline (SSE event separator)
  const parts = buffer.split('\n\n');

  // Last part might be incomplete - keep it in buffer
  const remaining = parts.pop() || '';

  for (const part of parts) {
    const lines = part.split('\n');
    for (const line of lines) {
      if (line.startsWith('data: ')) {
        const data = line.slice(6).trim();
        if (data) {
          events.push(data);
        }
      }
    }
  }

  return { events, remaining };
}

// ============================================================================
// TOKEN ALIGNMENT
// ============================================================================

interface TokenizeResult {
  complete: string[];
  pending: string;
}

function tokenizeIncremental(newText: string, pending: string): TokenizeResult {
  const combined = pending + newText;
  const tokens: string[] = [];
  let current = '';

  for (let i = 0; i < combined.length; i++) {
    const char = combined[i];

    if (char === ' ' || char === '\n' || char === '\t') {
      if (current) {
        tokens.push(current);
        current = '';
      }
      tokens.push(char);
    } else {
      current += char;
    }
  }

  const endsWithBoundary = /[\s]$/.test(newText);

  if (endsWithBoundary || current === '') {
    if (current) tokens.push(current);
    return { complete: tokens, pending: '' };
  } else {
    return { complete: tokens, pending: current };
  }
}

// ============================================================================
// TOKEN-ATTENTION ALIGNER
// ============================================================================

export class TokenAttentionAligner {
  private tokens: string[] = [];
  private attention: Map<number, AttentionEntry> = new Map();
  private prefillLength: number = 0;

  setPrefillLength(length: number) {
    this.prefillLength = length;
  }

  addToken(token: string): number {
    const index = this.tokens.length;
    this.tokens.push(token);
    return index;
  }

  addAttention(entry: AttentionEntry): void {
    const step =
      'step' in entry ? entry.step : 'decode_step' in entry ? entry.decode_step : this.attention.size;

    const tokenIndex = this.prefillLength + step;
    this.attention.set(tokenIndex, entry);
  }

  getAttention(tokenIndex: number): AttentionEntry | null {
    return this.attention.get(tokenIndex) || null;
  }

  getAligned(): Array<{ token: string; index: number; attention: AttentionEntry | null }> {
    return this.tokens.map((token, index) => ({
      token,
      index,
      attention: this.attention.get(index) || null,
    }));
  }

  getAlignmentStats(): { tokens: number; attention: number; aligned: number; missing: number } {
    const decodeTokens = this.tokens.length - this.prefillLength;
    const aligned = Array.from(this.attention.keys()).filter((i) => i >= this.prefillLength).length;

    return {
      tokens: this.tokens.length,
      attention: this.attention.size,
      aligned,
      missing: Math.max(0, decodeTokens - aligned),
    };
  }
}

// ============================================================================
// NON-STREAMING CLIENT
// ============================================================================

export async function fetchWithAttention(
  baseUrl: string,
  messages: Array<{ role: string; content: string }>,
  params: AttentionCaptureParams = {}
): Promise<ChatCompletionWithAttention> {
  const response = await fetch(`${baseUrl}/v1/chat/completions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: 'default',
      messages,
      stream: false,
      ...params,
    }),
  });

  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }

  return response.json();
}

// ============================================================================
// TOKEN ATTENTION DETAIL FETCHER
// ============================================================================

export interface TokenAttentionDetail {
  tokenIndex: number;
  tokenText: string;
  topK: Array<{
    position: number;
    score: number;
    tokenText?: string;
  }>;
  entropy: number;
  localMass: number;
  midMass: number;
  longMass: number;
  layerId: number;
  mode: string;
}

/**
 * Fetches detailed attention data for a specific token by making a focused request
 * This is used when the user clicks on a token to see what it attended to
 */
export async function fetchTokenAttentionDetail(
  baseUrl: string,
  model: string,
  contextMessages: Array<{ role: string; content: string }>,
  targetTokenIndex: number,
  inputTokens: string[]
): Promise<TokenAttentionDetail | null> {
  try {
    // Make a request with raw attention mode to get detailed top-k
    const response = await fetch(`${baseUrl}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model,
        messages: contextMessages,
        max_tokens: 1, // Just need attention data, not generation
        stream: false,
        return_attention_tokens: true,
        top_k_attention: 10,
        // Force raw mode to get actual token positions, even if server is in fingerprint mode
        attention_fingerprint_mode: false,
        // Request specific layers for detail view
        attention_capture_layer_ids: [7, 15, 23, 31],
      }),
    });

    if (!response.ok) {
      console.warn('Failed to fetch token attention detail:', response.status);
      return null;
    }

    const data = await response.json();

    // Extract attention from response
    const attentionTokens = data.choices?.[0]?.attention_tokens || data.attention_tokens || [];

    if (attentionTokens.length === 0) {
      return null;
    }

    // Get the attention entry (use first available)
    const entry = attentionTokens[0];

    // Build result based on mode
    if (entry.mode === 'raw') {
      const layerId = entry.layer_id || 31;
      const layer = entry.layers?.[layerId] || entry;

      const topK = (layer.token_positions || []).map((pos: number, i: number) => ({
        position: pos,
        score: layer.attention_scores?.[i] || 0,
        tokenText: inputTokens[pos] || `[${pos}]`,
      })).slice(0, 10);

      return {
        tokenIndex: targetTokenIndex,
        tokenText: `Token #${targetTokenIndex}`,
        topK,
        entropy: 0,
        localMass: 0,
        midMass: 0,
        longMass: 0,
        layerId,
        mode: 'raw',
      };
    }

    if (entry.mode === 'fingerprint') {
      const fp = entry.fingerprint || [];
      return {
        tokenIndex: targetTokenIndex,
        tokenText: `Token #${targetTokenIndex}`,
        topK: [],
        entropy: fp[19] ?? fp[3] ?? 0,
        localMass: fp[16] ?? fp[0] ?? 0,
        midMass: fp[17] ?? fp[1] ?? 0,
        longMass: fp[18] ?? fp[2] ?? 0,
        layerId: -1,
        mode: 'fingerprint',
      };
    }

    if (entry.mode === 'sketch') {
      const sketch = entry.sketch || Object.values(entry.layer_sketches || {})[0];
      if (sketch) {
        const topK = (sketch.top_hubs || []).map((pos: number, i: number) => ({
          position: pos,
          score: sketch.hub_scores?.[i] || 0,
          tokenText: inputTokens[pos] || `[${pos}]`,
        })).slice(0, 10);

        return {
          tokenIndex: targetTokenIndex,
          tokenText: `Token #${targetTokenIndex}`,
          topK,
          entropy: sketch.entropy || 0,
          localMass: sketch.local_mass || 0,
          midMass: sketch.mid_mass || 0,
          longMass: sketch.long_mass || 0,
          layerId: entry.layer_id || -1,
          mode: 'sketch',
        };
      }
    }

    return null;
  } catch (error) {
    console.error('Error fetching token attention detail:', error);
    return null;
  }
}
