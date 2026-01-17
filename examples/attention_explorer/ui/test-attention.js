// Simple script to test attention parsing
// Run with: node test-attention.js

const fetch = globalThis.fetch;

async function testAttention() {
  const response = await fetch('http://localhost:8000/v1/chat/completions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: 'Qwen/Qwen3-Next-80B-A3B-Thinking-FP8',
      messages: [{ role: 'user', content: 'Hi' }],
      max_tokens: 3,
      stream: true,
      return_attention_tokens: true,
    }),
  });

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let chunkCount = 0;
  let attentionFound = false;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n\n');
    buffer = lines.pop() || '';

    for (const line of lines) {
      if (line.startsWith('data: ')) {
        const data = line.slice(6).trim();
        if (data === '[DONE]') {
          console.log('Got [DONE]');
          continue;
        }
        chunkCount++;
        try {
          const chunk = JSON.parse(data);
          const delta = chunk.choices?.[0]?.delta;
          if (delta?.attention_tokens) {
            console.log(`Chunk ${chunkCount}: HAS attention_tokens! Count:`, delta.attention_tokens.length);
            attentionFound = true;
          } else if (delta?.content) {
            console.log(`Chunk ${chunkCount}: content="${delta.content}"`);
          } else if (chunk.choices?.[0]?.finish_reason) {
            console.log(`Chunk ${chunkCount}: finish_reason="${chunk.choices[0].finish_reason}"`);
          } else {
            console.log(`Chunk ${chunkCount}: other`);
          }
        } catch (e) {
          console.log(`Chunk ${chunkCount}: PARSE ERROR`, e.message);
        }
      }
    }
  }

  console.log('\n=== Summary ===');
  console.log('Total chunks:', chunkCount);
  console.log('Attention found:', attentionFound);
}

testAttention().catch(console.error);
