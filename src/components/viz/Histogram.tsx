import { useMemo, useState } from 'react';
import type { Tensor } from '../../lib/tensor';
import { formatValue, histogram } from '../../lib/tensor';

const W = 440;
const H = 190;
const PAD = { l: 8, r: 8, t: 10, b: 26 };

export default function Histogram({ tensor }: { tensor: Tensor }) {
  const [log, setLog] = useState(false);
  const [hover, setHover] = useState<number | null>(null);
  const { counts, min, max } = useMemo(() => histogram(tensor, 64), [tensor]);
  const stats = tensor.stats();

  const bins = counts.length;
  const peak = Math.max(...counts.map((c) => (log ? Math.log10(c + 1) : c)), 1);
  const iw = W - PAD.l - PAD.r;
  const ih = H - PAD.t - PAD.b;
  const bw = iw / bins;
  const x = (v: number) => PAD.l + ((v - min) / (max - min)) * iw;
  const binLo = (i: number) => min + ((max - min) * i) / bins;

  const ticks = [min, max];
  if (min < 0 && max > 0) ticks.splice(1, 0, 0);

  return (
    <div className="viz">
      <div className="viz-toolbar">
        <span className="muted">{stats.count.toLocaleString()} values · 64 bins</span>
        <span className="spacer" />
        <div className="segmented">
          <button className={!log ? 'active' : ''} onClick={() => setLog(false)}>
            linear
          </button>
          <button className={log ? 'active' : ''} onClick={() => setLog(true)}>
            log
          </button>
        </div>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="histogram" onMouseLeave={() => setHover(null)}>
        {counts.map((c, i) => {
          const h = ((log ? Math.log10(c + 1) : c) / peak) * ih;
          const mid = binLo(i + 0.5);
          return (
            <g key={i} onMouseEnter={() => setHover(i)}>
              <rect x={PAD.l + i * bw} y={PAD.t} width={bw} height={ih} fill="transparent" />
              <rect
                x={PAD.l + i * bw + 0.5}
                y={PAD.t + ih - h}
                width={Math.max(0.5, bw - 1)}
                height={h}
                rx={1}
                className={`hist-bar ${mid < 0 ? 'neg' : 'pos'} ${hover === i ? 'hover' : ''}`}
              />
            </g>
          );
        })}
        <line x1={PAD.l} x2={W - PAD.r} y1={PAD.t + ih} y2={PAD.t + ih} className="axis" />
        <line x1={x(stats.mean)} x2={x(stats.mean)} y1={PAD.t} y2={PAD.t + ih} className="mean-line" />
        <text x={x(stats.mean) + 4} y={PAD.t + 10} className="axis-label">
          mean
        </text>
        {ticks.map((t, i) => (
          <text
            key={i}
            x={x(t)}
            y={H - 8}
            className="axis-label"
            textAnchor={i === 0 ? 'start' : i === ticks.length - 1 ? 'end' : 'middle'}
          >
            {formatValue(t)}
          </text>
        ))}
      </svg>
      <div className="viz-footer">
        <span className="mono hover-readout">
          {hover !== null
            ? `[${formatValue(binLo(hover))}, ${formatValue(binLo(hover + 1))}) → ${counts[hover].toLocaleString()}`
            : 'Hover a bar for its count'}
        </span>
      </div>
    </div>
  );
}
