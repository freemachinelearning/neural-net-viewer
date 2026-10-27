import { useCallback, useMemo, useState } from 'react';
import { loadFile, type LoadedFile } from './lib/loaders';
import { buildModel, findCandidates, type Candidate } from './lib/model';
import { demoModel } from './lib/demo';
import { formatBytes, formatNumber } from './lib/tensor';
import FileDrop from './components/FileDrop';
import ModuleTree from './components/ModuleTree';
import ArchitectureView from './components/ArchitectureView';
import ModuleDetails from './components/ModuleDetails';

interface Loaded {
  name: string;
  size: number;
  file: LoadedFile;
  candidates: Candidate[];
}

export default function App() {
  const [loaded, setLoaded] = useState<Loaded | null>(null);
  const [candidateIdx, setCandidateIdx] = useState(0);
  const [selected, setSelected] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const open = useCallback((name: string, size: number, file: LoadedFile) => {
    const candidates = findCandidates(file);
    if (!candidates.length) throw new Error('No tensors found in this file.');
    setLoaded({ name, size, file, candidates });
    setCandidateIdx(0);
    setSelected('');
    setError(null);
  }, []);

  const handleFile = useCallback(
    async (f: File) => {
      setBusy(true);
      setError(null);
      // Let the "Parsing…" state paint before the synchronous parse blocks the thread.
      await new Promise((r) => setTimeout(r, 30));
      try {
        open(f.name, f.size, await loadFile(f));
      } catch (e) {
        console.error(e);
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setBusy(false);
      }
    },
    [open],
  );

  const model = useMemo(
    () => (loaded ? buildModel(loaded.candidates[candidateIdx]) : null),
    [loaded, candidateIdx],
  );

  if (!loaded || !model) {
    return (
      <FileDrop
        busy={busy}
        error={error}
        onFile={handleFile}
        onDemo={() => open('demo_cnn.pt', 0, demoModel())}
      />
    );
  }

  const node = model.index.get(selected) ?? model.root;

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark" />
          Model Viewer
        </div>
        <div className="file-meta">
          <span className="file-name">{loaded.name}</span>
          <span className="chip">{loaded.file.format}</span>
          <span className="chip">{model.source === 'module' ? 'full nn.Module' : 'state_dict'}</span>
          <span>{formatNumber(model.root.totalParams)} params</span>
          <span className="muted">{formatBytes(model.root.totalBytes)}</span>
        </div>
        <div className="topbar-actions">
          {loaded.candidates.length > 1 && (
            <label className="candidate-select">
              <span className="muted">Showing</span>
              <select value={candidateIdx} onChange={(e) => {
                setCandidateIdx(Number(e.target.value));
                setSelected('');
              }}>
                {loaded.candidates.map((c, i) => (
                  <option key={i} value={i}>
                    {c.label} ({c.tensorCount} tensors)
                  </option>
                ))}
              </select>
            </label>
          )}
          <label className="btn">
            Open file…
            <input
              type="file"
              accept=".pt,.pth,.bin,.ckpt,.safetensors"
              hidden
              onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
            />
          </label>
        </div>
      </header>
      {busy && <div className="busy-bar" />}
      {error && <div className="error-banner">{error}</div>}
      <main className="workspace">
        <aside className="pane tree-pane">
          <ModuleTree model={model} selected={node.path} onSelect={setSelected} />
        </aside>
        <section className="pane diagram-pane">
          <ArchitectureView model={model} selected={node.path} onSelect={setSelected} />
        </section>
        <aside className="pane details-pane">
          <ModuleDetails key={node.path} node={node} onSelect={setSelected} />
        </aside>
      </main>
    </div>
  );
}
