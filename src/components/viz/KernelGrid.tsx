import { useEffect, useMemo, useRef, useState } from 'react';
import type { Tensor } from '../../lib/tensor';
import { formatValue } from '../../lib/tensor';
import { diverging } from '../../lib/colors';
import { ColorBar, useScale } from './common';

const MAX_FILTERS = 64;
const MAX_CHANNELS = 32;
const WIDTH = 440;
const GAP = 2;

type Mode = 'channels' | 'rgb';

interface Layout {
  tiles: { o: number; c: number; x: number; y: number }[];
  scale: number;
  width: number;
  height: number;
}

/** Visualizes conv weights [out, in, kh, kw] (or [out, in, k]) as a grid of kernel tiles. */
export default function KernelGrid({ tensor }: { tensor: Tensor }) {
  const [O, I] = tensor.shape;
  const kh = tensor.ndim === 4 ? tensor.shape[2] : 1;
  const kw = tensor.shape[tensor.ndim - 1];
  const canRgb = I === 3 && tensor.ndim === 4;
  const [mode, setMode] = useState<Mode>(canRgb ? 'rgb' : 'channels');
  const [perFilter, setPerFilter] = useState(false);
  const [limit, scaleControl] = useScale(tensor);
  const [hover, setHover] = useState<string | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const values = tensor.values();
  const ksize = kh * kw;

  const nFilters = Math.min(O, MAX_FILTERS);
  const nChannels = Math.min(I, MAX_CHANNELS);
  // One tile per filter when there's a single input channel or we're compositing RGB.
  const wrapped = mode === 'rgb' || I === 1;

  const layout = useMemo<Layout>(() => {
    const cols = wrapped ? Math.min(nFilters, 16) : nChannels;
    const rows = wrapped ? Math.ceil(nFilters / cols) : nFilters;
    const scale = Math.max(1, Math.min(24, Math.floor((WIDTH - GAP * cols) / (cols * kw))));
    const tw = kw * scale + GAP;
    const th = kh * scale + GAP;
    const tiles: Layout['tiles'] = [];
    for (let o = 0; o < nFilters; o++) {
      if (wrapped) tiles.push({ o, c: 0, x: (o % cols) * tw, y: Math.floor(o / cols) * th });
      else for (let c = 0; c < nChannels; c++) tiles.push({ o, c, x: c * tw, y: o * th });
    }
    return { tiles, scale, width: cols * tw, height: rows * th };
  }, [wrapped, nFilters, nChannels, kh, kw]);

  // Max |w| per filter, for per-filter normalization.
  const filterMax = useMemo(() => {
    const per = I * ksize;
    const out = new Float32Array(O);
    for (let o = 0; o < O; o++) {
      let m = 0;
      for (let i = o * per; i < (o + 1) * per; i++) m = Math.max(m, Math.abs(values[i]));
      out[o] = m || 1;
    }
    return out;
  }, [values, O, I, ksize]);

  useEffect(() => {
    const canvas = canvasRef.current!;
    const { width, height, scale, tiles } = layout;
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d')!;
    const img = ctx.createImageData(width, height);
    const px = new Uint8ClampedArray(4);
    for (const { o, c, x, y } of tiles) {
      const norm = perFilter ? filterMax[o] : limit;
      for (let ky = 0; ky < kh; ky++) {
        for (let kx = 0; kx < kw; kx++) {
          if (mode === 'rgb') {
            // Classic first-layer view: map each filter's 3 input channels to R, G, B.
            for (let ch = 0; ch < 3; ch++) {
              const v = values[((o * I + ch) * kh + ky) * kw + kx];
              px[ch] = 128 + (v / norm) * 127;
            }
            px[3] = 255;
          } else {
            diverging(values[((o * I + c) * kh + ky) * kw + kx] / norm, px, 0);
          }
          for (let sy = 0; sy < scale; sy++) {
            let off = ((y + ky * scale + sy) * width + x + kx * scale) * 4;
            for (let sx = 0; sx < scale; sx++, off += 4) img.data.set(px, off);
          }
        }
      }
    }
    ctx.putImageData(img, 0, 0);
  }, [layout, values, limit, perFilter, filterMax, mode, I, kh, kw]);

  const onMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const mx = ((e.clientX - rect.left) / rect.width) * layout.width;
    const my = ((e.clientY - rect.top) / rect.height) * layout.height;
    const s = layout.scale;
    const tile = layout.tiles.find(
      (t) => mx >= t.x && mx < t.x + kw * s && my >= t.y && my < t.y + kh * s,
    );
    if (!tile) return setHover(null);
    const kx = Math.floor((mx - tile.x) / s);
    const ky = Math.floor((my - tile.y) / s);
    if (mode === 'rgb') {
      const rgb = [0, 1, 2].map((ch) => formatValue(values[((tile.o * I + ch) * kh + ky) * kw + kx]));
      setHover(`filter ${tile.o} @(${ky},${kx})  R ${rgb[0]}  G ${rgb[1]}  B ${rgb[2]}`);
    } else {
      const v = values[((tile.o * I + tile.c) * kh + ky) * kw + kx];
      setHover(`filter ${tile.o}, in-channel ${tile.c} @(${ky},${kx}) = ${formatValue(v)}`);
    }
  };

  return (
    <div className="viz">
      <div className="viz-toolbar">
        {canRgb && (
          <div className="segmented">
            <button className={mode === 'rgb' ? 'active' : ''} onClick={() => setMode('rgb')}>
              RGB
            </button>
            <button className={mode === 'channels' ? 'active' : ''} onClick={() => setMode('channels')}>
              per channel
            </button>
          </div>
        )}
        <label className="check">
          <input type="checkbox" checked={perFilter} onChange={(e) => setPerFilter(e.target.checked)} />
          normalize per filter
        </label>
        <span className="spacer" />
        {!perFilter && scaleControl}
      </div>
      <p className="muted small">
        {wrapped
          ? `${nFilters}${O > nFilters ? ` of ${O}` : ''} filters, ${kh}×${kw} each`
          : `Rows: output filters (${nFilters}${O > nFilters ? ` of ${O}` : ''}) · Columns: input channels (${nChannels}${I > nChannels ? ` of ${I}` : ''}) · ${kh}×${kw} kernels`}
      </p>
      <div className="kernel-wrap">
        <canvas
          ref={canvasRef}
          className="pixelated"
          style={{ width: layout.width, maxWidth: '100%' }}
          onMouseMove={onMove}
          onMouseLeave={() => setHover(null)}
        />
      </div>
      <div className="viz-footer">
        <span className="mono hover-readout">{hover ?? 'Hover a kernel pixel'}</span>
        {mode !== 'rgb' && !perFilter && <ColorBar limit={limit} />}
      </div>
    </div>
  );
}
