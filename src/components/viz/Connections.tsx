import { useEffect, useMemo, useRef, useState } from 'react';
import type { Tensor } from '../../lib/tensor';
import { formatValue } from '../../lib/tensor';

const W = 440;
const H = 460;
const MAX_IN = 40;
const MAX_OUT = 28;
const X_IN = 70;
const X_OUT = W - 70;

/** Picks up to `k` indices with the largest norm, returned in index order. */
function topByNorm(norms: Float32Array, k: number): number[] {
  const idx = Array.from(norms.keys());
  if (idx.length <= k) return idx;
  return idx.sort((a, b) => norms[b] - norms[a]).slice(0, k).sort((a, b) => a - b);
}

type Hover = { side: 'in' | 'out'; pos: number } | null;

/**
 * Draws a Linear layer's weight matrix [out, in] as a bipartite neuron graph.
 * Edge opacity is |w|, color is sign.
 */
export default function Connections({ tensor }: { tensor: Tensor }) {
  const [outN, inN] = tensor.shape;
  const values = tensor.values();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [hover, setHover] = useState<Hover>(null);

  const { ins, outs, maxAbs } = useMemo(() => {
    const rowNorm = new Float32Array(outN);
    const colNorm = new Float32Array(inN);
    for (let o = 0; o < outN; o++) {
      for (let i = 0; i < inN; i++) {
        const v = values[o * inN + i] ** 2;
        rowNorm[o] += v;
        colNorm[i] += v;
      }
    }
    const ins = topByNorm(colNorm, MAX_IN);
    const outs = topByNorm(rowNorm, MAX_OUT);
    let maxAbs = 0;
    for (const o of outs) for (const i of ins) maxAbs = Math.max(maxAbs, Math.abs(values[o * inN + i]));
    return { ins, outs, maxAbs: maxAbs || 1 };
  }, [values, outN, inN]);

  const yFor = (n: number, k: number) => (n === 1 ? H / 2 : 24 + (k * (H - 48)) / (n - 1));
  const radius = (n: number) => Math.max(2.5, Math.min(7, (H - 48) / n / 2.6));

  useEffect(() => {
    const canvas = canvasRef.current!;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    const ctx = canvas.getContext('2d')!;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, W, H);

    const active = (k: number, side: 'in' | 'out') => !hover || (hover.side === side && hover.pos === k);
    // Draw weak edges first so strong ones end up on top.
    const edges: [number, number, number][] = [];
    outs.forEach((o, ko) =>
      ins.forEach((i, ki) => {
        if (hover && !active(ki, 'in') && !active(ko, 'out')) return;
        edges.push([ki, ko, values[o * inN + i]]);
      }),
    );
    edges.sort((a, b) => Math.abs(a[2]) - Math.abs(b[2]));
    for (const [ki, ko, v] of edges) {
      const s = Math.abs(v) / maxAbs;
      const alpha = hover ? 0.15 + 0.85 * s : Math.pow(s, 1.6) * 0.9;
      if (alpha < 0.01) continue;
      ctx.strokeStyle = v >= 0 ? `rgba(255,138,56,${alpha})` : `rgba(72,140,255,${alpha})`;
      ctx.lineWidth = hover ? 0.6 + 2 * s : 0.4 + 1.6 * s;
      ctx.beginPath();
      ctx.moveTo(X_IN, yFor(ins.length, ki));
      ctx.lineTo(X_OUT, yFor(outs.length, ko));
      ctx.stroke();
    }

    const drawNodes = (list: number[], x: number, side: 'in' | 'out') => {
      const r = radius(list.length);
      list.forEach((idx, k) => {
        const y = yFor(list.length, k);
        const on = hover?.side === side && hover.pos === k;
        ctx.fillStyle = on ? '#ffffff' : '#2a2f3a';
        ctx.strokeStyle = on ? '#ffffff' : '#8a94a6';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.arc(x, y, r, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
        if (list.length <= 40 || on) {
          ctx.fillStyle = on ? '#ffffff' : '#8a94a6';
          ctx.font = '10px ui-monospace, SFMono-Regular, Menlo, monospace';
          ctx.textAlign = side === 'in' ? 'right' : 'left';
          ctx.textBaseline = 'middle';
          ctx.fillText(String(idx), side === 'in' ? x - r - 5 : x + r + 5, y);
        }
      });
    };
    drawNodes(ins, X_IN, 'in');
    drawNodes(outs, X_OUT, 'out');
  }, [ins, outs, values, inN, maxAbs, hover]);

  const onMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const mx = ((e.clientX - rect.left) / rect.width) * W;
    const my = ((e.clientY - rect.top) / rect.height) * H;
    const side = mx < W / 2 ? 'in' : 'out';
    const list = side === 'in' ? ins : outs;
    if (Math.abs(mx - (side === 'in' ? X_IN : X_OUT)) > 50) return setHover(null);
    let best = 0;
    list.forEach((_, k) => {
      if (Math.abs(yFor(list.length, k) - my) < Math.abs(yFor(list.length, best) - my)) best = k;
    });
    setHover({ side, pos: best });
  };

  let readout = 'Hover a neuron to isolate its connections';
  if (hover) {
    const idx = hover.side === 'in' ? ins[hover.pos] : outs[hover.pos];
    let sum = 0;
    let n = 0;
    if (hover.side === 'in') for (let o = 0; o < outN; o++, n++) sum += Math.abs(values[o * inN + idx]);
    else for (let i = 0; i < inN; i++, n++) sum += Math.abs(values[idx * inN + i]);
    readout = `${hover.side === 'in' ? 'input' : 'output'} neuron ${idx} · mean |w| over all ${n} connections = ${formatValue(sum / n)}`;
  }

  return (
    <div className="viz">
      <p className="muted small">
        Inputs ({ins.length < inN ? `top ${ins.length} of ${inN} by L2 norm` : inN}) → outputs (
        {outs.length < outN ? `top ${outs.length} of ${outN}` : outN}). Line opacity ∝ |w|;{' '}
        <span className="pos-text">positive</span> / <span className="neg-text">negative</span>.
      </p>
      <canvas
        ref={canvasRef}
        className="connections"
        style={{ aspectRatio: `${W} / ${H}` }}
        onMouseMove={onMove}
        onMouseLeave={() => setHover(null)}
      />
      <div className="viz-footer">
        <span className="mono hover-readout">{readout}</span>
      </div>
    </div>
  );
}
