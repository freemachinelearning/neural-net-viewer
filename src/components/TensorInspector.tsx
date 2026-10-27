import { useState } from 'react';
import type { ParamEntry } from '../lib/model';
import { formatBytes, formatValue } from '../lib/tensor';
import Heatmap from './viz/Heatmap';
import Histogram from './viz/Histogram';
import KernelGrid from './viz/KernelGrid';
import Connections from './viz/Connections';
import ValuesTable from './viz/ValuesTable';

type Tab = 'heatmap' | 'histogram' | 'kernels' | 'connections' | 'values';

const TAB_LABELS: Record<Tab, string> = {
  kernels: 'Filters',
  connections: 'Neurons',
  heatmap: 'Heatmap',
  histogram: 'Distribution',
  values: 'Values',
};

export default function TensorInspector({ entry }: { entry: ParamEntry }) {
  const t = entry.tensor;
  const tabs: Tab[] = [];
  const isConv = (t.ndim === 4 || t.ndim === 3) && /weight/.test(entry.name);
  const isMatrix = t.ndim === 2 && t.shape[0] > 1 && t.shape[1] > 1;
  if (isConv) tabs.push('kernels');
  if (isMatrix && /weight/.test(entry.name)) tabs.push('connections');
  if (t.numel > 1) tabs.push('heatmap', 'histogram');
  tabs.push('values');
  const [tab, setTab] = useState<Tab>(tabs[0]);

  let stats: ReturnType<typeof t.stats> | null = null;
  let error: string | null = null;
  try {
    stats = t.dtype === 'unknown' ? null : t.stats();
  } catch (e) {
    error = e instanceof Error ? e.message : String(e);
  }

  return (
    <section className="inspector">
      <h3>
        <span className="mono">{entry.path}</span>
      </h3>
      <div className="stat-grid">
        <Stat label="shape" value={`[${t.shape.join(', ')}]`} />
        <Stat label="dtype" value={t.dtype} />
        <Stat label="size" value={formatBytes(t.nbytes)} />
        {stats && (
          <>
            <Stat label="min" value={formatValue(stats.min)} />
            <Stat label="max" value={formatValue(stats.max)} />
            <Stat label="mean" value={formatValue(stats.mean)} />
            <Stat label="std" value={formatValue(stats.std)} />
            <Stat label="zeros" value={`${((stats.zeros / Math.max(1, stats.count)) * 100).toFixed(1)}%`} />
            <Stat label="L2 norm" value={formatValue(stats.l2)} />
            {stats.nonFinite > 0 && <Stat label="NaN/Inf" value={String(stats.nonFinite)} warn />}
          </>
        )}
      </div>
      {error && <div className="error-banner">{error}</div>}
      {stats && (
        <>
          <div className="tabs">
            {tabs.map((k) => (
              <button key={k} className={k === tab ? 'active' : ''} onClick={() => setTab(k)}>
                {TAB_LABELS[k]}
              </button>
            ))}
          </div>
          <div className="tab-body">
            {tab === 'heatmap' && <Heatmap tensor={t} />}
            {tab === 'histogram' && <Histogram tensor={t} />}
            {tab === 'kernels' && <KernelGrid tensor={t} />}
            {tab === 'connections' && <Connections tensor={t} />}
            {tab === 'values' && <ValuesTable tensor={t} />}
          </div>
        </>
      )}
    </section>
  );
}

function Stat({ label, value, warn }: { label: string; value: string; warn?: boolean }) {
  return (
    <div className={`stat ${warn ? 'warn' : ''}`}>
      <div className="stat-label">{label}</div>
      <div className="stat-value mono">{value}</div>
    </div>
  );
}
