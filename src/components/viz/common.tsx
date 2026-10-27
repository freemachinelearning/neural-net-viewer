import { useState } from 'react';
import type { Tensor } from '../../lib/tensor';
import { formatValue } from '../../lib/tensor';

export type ScaleMode = 'max' | 'sigma';

/** Symmetric color-scale limit for a tensor: its max |value|, or mean ± 3σ to reduce outlier wash-out. */
export function useScale(tensor: Tensor): [number, React.ReactNode] {
  const [mode, setMode] = useState<ScaleMode>('sigma');
  const s = tensor.stats();
  const limit =
    (mode === 'max' ? s.absMax : Math.min(s.absMax, Math.abs(s.mean) + 3 * s.std)) || 1;
  const control = (
    <div className="segmented" title="Color scale range">
      <button className={mode === 'sigma' ? 'active' : ''} onClick={() => setMode('sigma')}>
        ±3σ
      </button>
      <button className={mode === 'max' ? 'active' : ''} onClick={() => setMode('max')}>
        ±max
      </button>
    </div>
  );
  return [limit, control];
}

export function ColorBar({ limit }: { limit: number }) {
  return (
    <div className="colorbar">
      <span className="mono">{formatValue(-limit)}</span>
      <div className="colorbar-gradient" />
      <span className="mono">{formatValue(limit)}</span>
    </div>
  );
}
