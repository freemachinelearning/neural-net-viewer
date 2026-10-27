import { useEffect, useMemo, useRef, useState } from 'react';
import type { Tensor } from '../../lib/tensor';
import { as2D, formatValue } from '../../lib/tensor';
import { diverging } from '../../lib/colors';
import { ColorBar, useScale } from './common';

const MAX_CELLS = 512;

interface Grid {
  rows: number;
  cols: number;
  /** Displayed (possibly downsampled) dimensions. */
  dr: number;
  dc: number;
  cells: Float32Array;
}

/**
 * Downsamples a [rows, cols] matrix to at most MAX_CELLS per side, keeping the
 * largest-magnitude value in each block so that strong weights stay visible.
 */
function reduce(values: Float32Array, rows: number, cols: number): Grid {
  const dr = Math.min(rows, MAX_CELLS);
  const dc = Math.min(cols, MAX_CELLS);
  if (dr === rows && dc === cols) return { rows, cols, dr, dc, cells: values };
  const cells = new Float32Array(dr * dc);
  const mag = new Float32Array(dr * dc).fill(-1);
  for (let r = 0; r < rows; r++) {
    const rr = Math.floor((r * dr) / rows) * dc;
    const base = r * cols;
    for (let c = 0; c < cols; c++) {
      const k = rr + Math.floor((c * dc) / cols);
      const v = values[base + c];
      const a = Math.abs(v);
      if (a > mag[k] || Number.isNaN(v)) {
        mag[k] = a;
        cells[k] = v;
      }
    }
  }
  return { rows, cols, dr, dc, cells };
}

export default function Heatmap({ tensor }: { tensor: Tensor }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [limit, scaleControl] = useScale(tensor);
  const [hover, setHover] = useState<{ r0: number; r1: number; c0: number; c1: number; v: number } | null>(null);

  const grid = useMemo(() => {
    let [rows, cols] = as2D(tensor.shape);
    // Wrap long vectors into a block so they stay legible.
    if (rows === 1 && cols > 64) {
      rows = Math.ceil(cols / 64);
      cols = 64;
      const padded = new Float32Array(rows * cols).fill(NaN);
      padded.set(tensor.values());
      return { ...reduce(padded, rows, cols), wrapped: true };
    }
    return { ...reduce(tensor.values(), rows, cols), wrapped: false };
  }, [tensor]);

  useEffect(() => {
    const canvas = canvasRef.current!;
    canvas.width = grid.dc;
    canvas.height = grid.dr;
    const ctx = canvas.getContext('2d')!;
    const img = ctx.createImageData(grid.dc, grid.dr);
    for (let i = 0; i < grid.cells.length; i++) {
      const v = grid.cells[i];
      if (grid.wrapped && Number.isNaN(v) && i >= tensor.numel) {
        img.data[i * 4 + 3] = 0; // padding
        continue;
      }
      diverging(v / limit, img.data, i * 4);
    }
    ctx.putImageData(img, 0, 0);
  }, [grid, limit, tensor]);

  // Square cells where possible; flat matrices get a minimum row height and
  // tall ones are capped to the viewport.
  const aspect = grid.dr / grid.dc;
  const cssHeight = `min(max(${aspect * 100}cqw, ${Math.min(grid.dr * 10, 160)}px), 70vh)`;

  const onMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const cx = Math.floor(((e.clientX - rect.left) / rect.width) * grid.dc);
    const cy = Math.floor(((e.clientY - rect.top) / rect.height) * grid.dr);
    if (cx < 0 || cy < 0 || cx >= grid.dc || cy >= grid.dr) return setHover(null);
    const r0 = Math.floor((cy * grid.rows) / grid.dr);
    const r1 = Math.floor(((cy + 1) * grid.rows) / grid.dr) - 1;
    const c0 = Math.floor((cx * grid.cols) / grid.dc);
    const c1 = Math.floor(((cx + 1) * grid.cols) / grid.dc) - 1;
    setHover({ r0, r1, c0, c1, v: grid.cells[cy * grid.dc + cx] });
  };

  const [rows, cols] = as2D(tensor.shape);
  const downsampled = grid.dr < grid.rows || grid.dc < grid.cols;
  const fmtRange = (a: number, b: number) => (a === b ? String(a) : `${a}–${b}`);
  let hoverText = 'Hover to read values';
  if (hover) {
    if (grid.wrapped) {
      const idx = hover.r0 * 64 + hover.c0;
      hoverText = idx < tensor.numel ? `[${idx}] = ${formatValue(hover.v)}` : '';
    } else {
      hoverText = `[${fmtRange(hover.r0, hover.r1)}, ${fmtRange(hover.c0, hover.c1)}] ${
        downsampled ? 'max |w| ' : ''
      }= ${formatValue(hover.v)}`;
    }
  }

  return (
    <div className="viz">
      <div className="viz-toolbar">
        <span className="muted">
          {tensor.ndim > 2
            ? `Flattened to ${rows} × ${cols} (dim 0 × rest)`
            : grid.wrapped
              ? `${tensor.numel} values, wrapped 64 per row`
              : `${rows} rows × ${cols} cols`}
          {downsampled && ` · downsampled to ${grid.dr}×${grid.dc}`}
        </span>
        <span className="spacer" />
        {scaleControl}
      </div>
      <div className="heatmap-wrap">
        <canvas
          ref={canvasRef}
          className="heatmap"
          style={{ height: cssHeight }}
          onMouseMove={onMove}
          onMouseLeave={() => setHover(null)}
        />
      </div>
      <div className="viz-footer">
        <span className="mono hover-readout">{hoverText}</span>
        <ColorBar limit={limit} />
      </div>
    </div>
  );
}
