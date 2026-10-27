import { useEffect, useMemo, useRef, useState } from 'react';
import type { ModelInfo, ModuleNode } from '../lib/model';
import { categorize } from '../lib/model';
import { CATEGORY_COLORS } from '../lib/colors';
import { formatNumber } from '../lib/tensor';

interface Props {
  model: ModelInfo;
  selected: string;
  onSelect: (path: string) => void;
}

export function ancestors(path: string): string[] {
  const parts = path ? path.split('.') : [];
  return parts.map((_, i) => parts.slice(0, i).join('.'));
}

export default function ModuleTree({ model, selected, onSelect }: Props) {
  const [expanded, setExpanded] = useState<Set<string>>(() => {
    const s = new Set<string>(['']);
    model.root.children.forEach((c) => s.add(c.path));
    return s;
  });
  const [query, setQuery] = useState('');
  const selectedRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setExpanded((prev) => {
      const missing = ancestors(selected).filter((a) => !prev.has(a));
      return missing.length ? new Set([...prev, ...missing]) : prev;
    });
  }, [selected]);

  useEffect(() => {
    selectedRef.current?.scrollIntoView({ block: 'nearest' });
  }, [selected, expanded]);

  // When searching, show matches plus their ancestors, fully expanded.
  const visible = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return null;
    const keep = new Set<string>();
    for (const n of model.index.values()) {
      if (n.path.toLowerCase().includes(q) || n.type.toLowerCase().includes(q)) {
        keep.add(n.path);
        ancestors(n.path).forEach((a) => keep.add(a));
      }
    }
    return keep;
  }, [query, model]);

  const toggle = (path: string) =>
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(path)) next.delete(path);
      else next.add(path);
      return next;
    });

  const render = (n: ModuleNode): React.ReactNode => {
    if (visible && !visible.has(n.path)) return null;
    const open = visible ? true : expanded.has(n.path);
    const hasKids = n.children.length > 0;
    const isSel = n.path === selected;
    return (
      <div key={n.path}>
        <div
          ref={isSel ? selectedRef : undefined}
          className={`tree-row ${isSel ? 'selected' : ''}`}
          style={{ paddingLeft: 8 + n.depth * 14 }}
          onClick={() => onSelect(n.path)}
        >
          <span
            className={`twisty ${hasKids ? '' : 'leaf'}`}
            onClick={(e) => {
              e.stopPropagation();
              if (hasKids) toggle(n.path);
            }}
          >
            {hasKids ? (open ? '▾' : '▸') : ''}
          </span>
          <span className="dot" style={{ background: CATEGORY_COLORS[categorize(n.type)] }} />
          <span className="tree-name">{n.name || 'model'}</span>
          <span className="tree-type">{n.type}</span>
          {n.totalParams > 0 && <span className="tree-count">{formatNumber(n.totalParams)}</span>}
        </div>
        {hasKids && open && n.children.map(render)}
      </div>
    );
  };

  return (
    <div className="tree">
      <div className="pane-header">
        <input
          className="search"
          placeholder="Filter layers…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
      </div>
      <div className="tree-body">{render(model.root)}</div>
    </div>
  );
}
