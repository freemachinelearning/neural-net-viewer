import { useEffect, useMemo, useRef, useState } from 'react';
import type { LayerCategory, ModelInfo, ModuleNode } from '../lib/model';
import { categorize } from '../lib/model';
import { CATEGORY_COLORS } from '../lib/colors';
import { formatNumber } from '../lib/tensor';
import { ancestors } from './ModuleTree';

interface Props {
  model: ModelInfo;
  selected: string;
  onSelect: (path: string) => void;
}

/** A vertical, nested block diagram of the network in definition order. */
export default function ArchitectureView({ model, selected, onSelect }: Props) {
  const [collapsed, setCollapsed] = useState<Set<string>>(() => {
    // Keep big models readable: fold containers below depth 2.
    const s = new Set<string>();
    if (model.index.size > 150) {
      for (const n of model.index.values()) if (n.depth >= 2 && n.children.length) s.add(n.path);
    }
    return s;
  });
  const selectedRef = useRef<HTMLDivElement>(null);

  const maxLeafParams = useMemo(
    () => Math.max(1, ...model.leaves.map((l) => l.totalParams)),
    [model],
  );
  const categories = useMemo(() => {
    const s = new Set<LayerCategory>();
    for (const l of model.leaves) s.add(categorize(l.type));
    return [...s];
  }, [model]);

  useEffect(() => {
    setCollapsed((prev) => {
      const hidden = ancestors(selected).filter((a) => prev.has(a));
      if (!hidden.length) return prev;
      const next = new Set(prev);
      hidden.forEach((a) => next.delete(a));
      return next;
    });
  }, [selected]);

  useEffect(() => {
    selectedRef.current?.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  }, [selected]);

  const toggle = (path: string) =>
    setCollapsed((prev) => {
      const next = new Set(prev);
      if (next.has(path)) next.delete(path);
      else next.add(path);
      return next;
    });

  const render = (n: ModuleNode): React.ReactNode => {
    const isSel = n.path === selected;
    const color = CATEGORY_COLORS[categorize(n.type)];

    if (n.children.length === 0) {
      const share = Math.sqrt(n.totalParams / maxLeafParams);
      return (
        <div
          key={n.path}
          ref={isSel ? selectedRef : undefined}
          className={`layer ${isSel ? 'selected' : ''}`}
          style={{ '--accent': color } as React.CSSProperties}
          onClick={(e) => {
            e.stopPropagation();
            onSelect(n.path);
          }}
        >
          <div className="layer-main">
            <span className="layer-type">{n.type}</span>
            {n.summary && <span className="layer-summary">{n.summary}</span>}
          </div>
          <div className="layer-sub">
            <span className="layer-name">{n.path || 'model'}</span>
            <span>{n.totalParams ? formatNumber(n.totalParams) : '—'}</span>
          </div>
          {n.totalParams > 0 && (
            <div className="layer-bar">
              <span style={{ width: `${Math.max(2, share * 100)}%` }} />
            </div>
          )}
        </div>
      );
    }

    const isCollapsed = collapsed.has(n.path);
    const isRoot = n.path === '';
    return (
      <div
        key={n.path}
        ref={isSel ? selectedRef : undefined}
        className={`block ${isSel ? 'selected' : ''} ${isRoot ? 'root' : ''}`}
        onClick={(e) => {
          e.stopPropagation();
          onSelect(n.path);
        }}
      >
        <div className="block-header">
          {!isRoot && (
            <button
              className="twisty-btn"
              onClick={(e) => {
                e.stopPropagation();
                toggle(n.path);
              }}
            >
              {isCollapsed ? '▸' : '▾'}
            </button>
          )}
          <span className="block-name">{n.name || 'model'}</span>
          <span className="block-type">{n.type}</span>
          <span className="block-count">{formatNumber(n.totalParams)} params</span>
        </div>
        {isCollapsed ? (
          <div className="block-collapsed">
            {n.children.length} submodule{n.children.length === 1 ? '' : 's'} ·{' '}
            {n.children
              .slice(0, 6)
              .map((c) => c.type)
              .join(' → ')}
            {n.children.length > 6 ? ' → …' : ''}
          </div>
        ) : (
          <div className="block-body">
            {n.params.length > 0 && (
              <div className="block-own-params">
                own tensors: {n.params.map((p) => p.name).join(', ')}
              </div>
            )}
            {n.children.map((c, i) => (
              <div key={c.path} className="flow-item">
                {i > 0 && <div className="connector" />}
                {render(c)}
              </div>
            ))}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="diagram">
      <div className="pane-header legend">
        {categories.map((c) => (
          <span key={c} className="legend-item">
            <span className="dot" style={{ background: CATEGORY_COLORS[c] }} />
            {c}
          </span>
        ))}
        <span className="spacer" />
        <button className="link" onClick={() => setCollapsed(new Set())}>
          Expand all
        </button>
        <button
          className="link"
          onClick={() =>
            setCollapsed(
              new Set([...model.index.values()].filter((n) => n.depth >= 1 && n.children.length).map((n) => n.path)),
            )
          }
        >
          Collapse all
        </button>
      </div>
      <div className="diagram-body">
        <div className="io-node">input</div>
        <div className="connector" />
        {render(model.root)}
        <div className="connector" />
        <div className="io-node">output</div>
      </div>
    </div>
  );
}
