import type { LayerCategory } from './model';

export const CATEGORY_COLORS: Record<LayerCategory, string> = {
  conv: '#f0a04b',
  linear: '#5b9cf5',
  norm: '#a48cf0',
  activation: '#4fc98a',
  embedding: '#e56fb0',
  recurrent: '#e2c044',
  attention: '#ef6f6c',
  pool: '#3fc1c9',
  dropout: '#8a94a6',
  container: '#6b7489',
  other: '#8a94a6',
};

const NEG = [72, 140, 255];
const MID = [22, 25, 32];
const POS = [255, 138, 56];

/**
 * Diverging colormap for a dark background: blue (negative) → near-black (0)
 * → orange (positive). `x` is expected in [-1, 1].
 */
export function diverging(x: number, out: Uint8ClampedArray, o: number) {
  if (!Number.isFinite(x)) {
    out[o] = 255;
    out[o + 1] = 0;
    out[o + 2] = 255;
    out[o + 3] = 255;
    return;
  }
  const c = x < 0 ? NEG : POS;
  const t = Math.pow(Math.min(1, Math.abs(x)), 0.7);
  out[o] = MID[0] + (c[0] - MID[0]) * t;
  out[o + 1] = MID[1] + (c[1] - MID[1]) * t;
  out[o + 2] = MID[2] + (c[2] - MID[2]) * t;
  out[o + 3] = 255;
}

export function divergingCss(x: number): string {
  const px = new Uint8ClampedArray(4);
  diverging(x, px, 0);
  return `rgb(${px[0]}, ${px[1]}, ${px[2]})`;
}
