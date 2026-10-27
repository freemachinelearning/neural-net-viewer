import { useState } from 'react';
import type { ModuleNode } from '../lib/model';
import { categorize } from '../lib/model';
import { CATEGORY_COLORS } from '../lib/colors';
import { formatBytes, formatNumber } from '../lib/tensor';
import TensorInspector from './TensorInspector';

interface Props {
  node: ModuleNode;
  onSelect: (path: string) => void;
}

export default function ModuleDetails({ node, onSelect }: Props) {
  const [paramIdx, setParamIdx] = useState(0);
  const param = node.params[paramIdx];
  const crumbs = node.path ? node.path.split('.') : [];

  return (
    <div className="details">
      <div className="pane-header crumbs">
        <button className="crumb" onClick={() => onSelect('')}>
          model
        </button>
        {crumbs.map((c, i) => (
          <span key={i}>
            <span className="muted"> / </span>
            <button className="crumb" onClick={() => onSelect(crumbs.slice(0, i + 1).join('.'))}>
              {c}
            </button>
          </span>
        ))}
      </div>
      <div className="details-body">
        <div className="details-title">
          <span className="dot big" style={{ background: CATEGORY_COLORS[categorize(node.type)] }} />
          <h2>{node.type}</h2>
          {node.summary && <span className="layer-summary">{node.summary}</span>}
        </div>
        {node.inferred && node.path !== '' && (
          <p className="hint">
            Layer type inferred from parameter names and shapes (this file is a state_dict, which
            stores weights but not the module classes).
          </p>
        )}

        <div className="kv-grid">
          <div>Parameters</div>
          <div>{node.totalParams.toLocaleString()}</div>
          <div>Memory</div>
          <div>{formatBytes(node.totalBytes)}</div>
          {node.attrs.map(([k, v]) => (
            <div key={k} style={{ display: 'contents' }}>
              <div>{k}</div>
              <div className="mono">{v}</div>
            </div>
          ))}
        </div>

        {node.children.length > 0 && (
          <section>
            <h3>Submodules ({node.children.length})</h3>
            <div className="child-list">
              {node.children.map((c) => (
                <button key={c.path} className="child" onClick={() => onSelect(c.path)}>
                  <span className="dot" style={{ background: CATEGORY_COLORS[categorize(c.type)] }} />
                  <span className="mono">{c.name}</span>
                  <span className="muted">{c.type}</span>
                  <span className="spacer" />
                  <span className="muted">{c.totalParams ? formatNumber(c.totalParams) : ''}</span>
                </button>
              ))}
            </div>
          </section>
        )}

        {node.params.length > 0 && (
          <section>
            <h3>Tensors</h3>
            <table className="param-table">
              <thead>
                <tr>
                  <th>name</th>
                  <th>shape</th>
                  <th>dtype</th>
                  <th className="num">count</th>
                </tr>
              </thead>
              <tbody>
                {node.params.map((p, i) => (
                  <tr
                    key={p.name}
                    className={i === paramIdx ? 'selected' : ''}
                    onClick={() => setParamIdx(i)}
                  >
                    <td className="mono">
                      {p.name}
                      {p.kind === 'buffer' && <span className="tag">buffer</span>}
                    </td>
                    <td className="mono">[{p.tensor.shape.join(', ')}]</td>
                    <td className="muted">{p.tensor.dtype}</td>
                    <td className="num">{formatNumber(p.tensor.numel)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>
        )}

        {param && <TensorInspector key={param.path} entry={param} />}

        {node.params.length === 0 && node.children.length === 0 && (
          <p className="hint">This layer has no weights (e.g. an activation or pooling op).</p>
        )}
      </div>
    </div>
  );
}
