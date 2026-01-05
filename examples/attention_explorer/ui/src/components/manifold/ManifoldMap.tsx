import { useRef, useEffect } from 'react';
import { useSessionStore } from '../../stores/useSessionStore';

// Mock cluster data - in production, load from artifacts
const MOCK_CLUSTERS = [
  { cx: 0.35, cy: 0.55, n: 160, name: 'JSON/Schema', color: 'rgba(85,214,166,0.8)' },
  { cx: 0.62, cy: 0.42, n: 140, name: 'Codegen', color: 'rgba(122,162,255,0.8)' },
  { cx: 0.45, cy: 0.30, n: 120, name: 'Reasoning', color: 'rgba(255,204,102,0.8)' },
  { cx: 0.75, cy: 0.68, n: 60, name: 'Creative', color: 'rgba(255,107,107,0.65)' },
];

export function ManifoldMap() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fingerprint = useSessionStore((state) => state.fingerprint);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Handle device pixel ratio
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const w = rect.width;
    const h = rect.height;

    // Clear
    ctx.clearRect(0, 0, w, h);

    // Draw mock cluster points
    const jitter = (s: number) => (Math.random() - 0.5) * s;

    MOCK_CLUSTERS.forEach((cluster) => {
      ctx.fillStyle = cluster.color;
      for (let i = 0; i < cluster.n; i++) {
        const x = (cluster.cx + jitter(0.12)) * w;
        const y = (cluster.cy + jitter(0.10)) * h;
        ctx.beginPath();
        ctx.arc(x, y, 2, 0, Math.PI * 2);
        ctx.fill();
      }
    });

    // Draw current session point ("you are here")
    if (fingerprint) {
      // Project fingerprint to approximate UMAP coords
      const px = (0.5 + fingerprint.local_mass * 0.3 - fingerprint.long_mass * 0.2) * w;
      const py = (0.5 - fingerprint.entropy * 0.3 + fingerprint.mid_mass * 0.2) * h;

      ctx.strokeStyle = 'rgba(255,255,255,0.85)';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(px, py, 6, 0, Math.PI * 2);
      ctx.stroke();

      ctx.fillStyle = 'rgba(255,255,255,0.85)';
      ctx.font = '12px system-ui, sans-serif';
      ctx.fillText('you', px + 10, py + 4);
    }

    // Draw cluster labels
    ctx.fillStyle = 'rgba(255,255,255,0.5)';
    ctx.font = '11px system-ui, sans-serif';
    MOCK_CLUSTERS.forEach((cluster) => {
      ctx.fillText(cluster.name, cluster.cx * w - 20, cluster.cy * h - 15);
    });
  }, [fingerprint]);

  return (
    <canvas
      ref={canvasRef}
      className="manifold-canvas"
      style={{ width: '100%', height: '280px' }}
    />
  );
}
