import { useState } from 'react';
import type { Tensor } from '../../lib/tensor';
import { as2D, contiguousStride, formatValue } from '../../lib/tensor';
import { divergingCss } from '../../lib/colors';

const PAGE_ROWS = 24;
const PAGE_COLS = 8;

/** Paged table of raw values, with ndim>2 tensors flattened to [dim0, rest]. */
export default function ValuesTable({ tensor }: { tensor: Tensor }) {
  const [rows, cols] = as2D(tensor.shape);
  const [r0, setR0] = useState(0);
  const [c0, setC0] = useState(0);
  const values = tensor.values();
  const limit = tensor.stats().absMax || 1;
  const strides = contiguousStride(tensor.shape);

  // Index label for a flat offset, e.g. "[3, 1, 0, 2]".
  const fullIndex = (flat: number) =>
    `[${strides.map((s, d) => Math.floor(flat / s) % tensor.shape[d]).join(', ')}]`;

  const rIdx = Array.from({ length: Math.min(PAGE_ROWS, rows - r0) }, (_, i) => r0 + i);
  const cIdx = Array.from({ length: Math.min(PAGE_COLS, cols - c0) }, (_, i) => c0 + i);

  const pager = (
    label: string,
    pos: number,
    size: number,
    total: number,
    set: (n: number) => void,
  ) => (
    <div className="pager">
      <span className="muted">{label}</span>
      <button disabled={pos === 0} onClick={() => set(Math.max(0, pos - size))}>
        ‹
      </button>
      <input
        type="number"
        min={0}
        max={Math.max(0, total - 1)}
        value={pos}
        onChange={(e) => set(Math.max(0, Math.min(total - 1, Number(e.target.value) || 0)))}
      />
      <span className="muted">
        –{Math.min(total, pos + size) - 1} of {total}
      </span>
      <button disabled={pos + size >= total} onClick={() => set(pos + size)}>
        ›
      </button>
    </div>
  );

  if (tensor.numel === 1) {
    return (
      <div className="viz">
        <div className="scalar mono">{formatValue(values[0])}</div>
      </div>
    );
  }

  return (
    <div className="viz">
      <div className="viz-toolbar wrap">
        {rows > 1 && pager('rows', r0, PAGE_ROWS, rows, setR0)}
        {pager(rows > 1 ? 'cols' : 'index', c0, PAGE_COLS, cols, setC0)}
      </div>
      <div className="values-scroll">
        <table className="values">
          <thead>
            <tr>
              <th />
              {cIdx.map((c) => (
                <th key={c}>{c}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rIdx.map((r) => (
              <tr key={r}>
                <th>{r}</th>
                {cIdx.map((c) => {
                  const flat = r * cols + c;
                  const v = values[flat];
                  return (
                    <td
                      key={c}
                      title={`${fullIndex(flat)} = ${v}`}
                      style={{ boxShadow: `inset 3px 0 0 ${divergingCss(v / limit)}` }}
                    >
                      {formatValue(v)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
