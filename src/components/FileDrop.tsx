import { useState } from 'react';

interface Props {
  busy: boolean;
  error: string | null;
  onFile: (f: File) => void;
  onDemo: () => void;
}

export default function FileDrop({ busy, error, onFile, onDemo }: Props) {
  const [over, setOver] = useState(false);
  return (
    <div className="landing">
      <div
        className={`dropzone ${over ? 'over' : ''}`}
        onDragOver={(e) => {
          e.preventDefault();
          setOver(true);
        }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setOver(false);
          const f = e.dataTransfer.files[0];
          if (f) onFile(f);
        }}
      >
        <div className="brand large">
          <span className="brand-mark" />
          Model Viewer
        </div>
        <p className="lede">
          Explore the layers and weights of a PyTorch model. Files are parsed entirely in your
          browser — nothing is uploaded.
        </p>
        {busy ? (
          <div className="parsing">Parsing model…</div>
        ) : (
          <>
            <label className="btn primary">
              Choose a model file
              <input
                type="file"
                accept=".pt,.pth,.bin,.ckpt,.safetensors"
                hidden
                onChange={(e) => e.target.files?.[0] && onFile(e.target.files[0])}
              />
            </label>
            <p className="muted">or drop it here</p>
            <button className="link" onClick={onDemo}>
              Try a demo CNN instead
            </button>
          </>
        )}
        {error && <div className="error-banner">{error}</div>}
        <div className="formats">
          <div>
            <b>Supported:</b> <code>torch.save(model.state_dict())</code>,{' '}
            <code>torch.save(model)</code>, training checkpoints (dicts containing a state_dict),
            TorchScript archives, and <code>.safetensors</code>.
          </div>
        </div>
      </div>
    </div>
  );
}
