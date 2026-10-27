import { PyObject } from './pickle';
import { Tensor } from './tensor';
import type { LoadedFile } from './loaders';

export interface ParamEntry {
  name: string;
  /** Fully qualified name, e.g. "features.0.weight". */
  path: string;
  tensor: Tensor;
  kind: 'parameter' | 'buffer';
}

export interface ModuleNode {
  /** Dotted path from the root; the root itself is "". */
  path: string;
  name: string;
  type: string;
  /** True when `type` was guessed from parameter names/shapes. */
  inferred: boolean;
  summary: string;
  attrs: [string, string][];
  params: ParamEntry[];
  children: ModuleNode[];
  totalParams: number;
  totalBytes: number;
  depth: number;
}

export interface Candidate {
  label: string;
  kind: 'module' | 'tensors';
  tensorCount: number;
  value: unknown;
}

export interface ModelInfo {
  root: ModuleNode;
  /** Every node indexed by path. */
  index: Map<string, ModuleNode>;
  /** Modules in definition order that have no submodules. */
  leaves: ModuleNode[];
  source: 'module' | 'state_dict';
}

// ---------------------------------------------------------------- candidates

const isModule = (x: unknown): x is PyObject =>
  x instanceof PyObject &&
  ((x.state instanceof Map && x.state.has('_modules')) || x.cls.module.startsWith('__torch__'));

/** Finds the things in a loaded file that could be shown as a model. */
export function findCandidates(file: LoadedFile): Candidate[] {
  const out: Candidate[] = [];
  const visit = (value: unknown, label: string, depth: number) => {
    if (isModule(value)) {
      out.push({ label, kind: 'module', tensorCount: flatten(value).length, value });
      return;
    }
    if (!(value instanceof Map)) return;
    const tensors = flatten(value).length;
    if (tensors === 0) return;
    const hasDirectTensors = [...value.values()].some((v) => v instanceof Tensor);
    if (hasDirectTensors || depth === 0) out.push({ label, kind: 'tensors', tensorCount: tensors, value });
    if (depth < 2 && !hasDirectTensors) {
      for (const [k, v] of value) visit(v, label === '(file)' ? String(k) : `${label}.${String(k)}`, depth + 1);
    }
  };
  visit(file.root, '(file)', 0);

  // Prefer real modules, then the largest tensor collection that isn't just "everything".
  return out.sort((a, b) => {
    if (a.kind !== b.kind) return a.kind === 'module' ? -1 : 1;
    const aRoot = a.label === '(file)' && out.length > 1 ? 1 : 0;
    const bRoot = b.label === '(file)' && out.length > 1 ? 1 : 0;
    if (aRoot !== bRoot) return aRoot - bRoot;
    return b.tensorCount - a.tensorCount;
  });
}

/** Flattens nested dicts / lists / modules into [dotted name, tensor] pairs. */
function flatten(value: unknown, prefix = '', seen = new Set<unknown>()): [string, Tensor][] {
  if (value instanceof Tensor) return [[prefix, value]];
  if (value === null || typeof value !== 'object' || seen.has(value)) return [];
  seen.add(value);
  const join = (k: unknown) => (prefix ? `${prefix}.${String(k)}` : String(k));
  const out: [string, Tensor][] = [];
  if (value instanceof Map) {
    for (const [k, v] of value) out.push(...flatten(v, join(k), seen));
  } else if (Array.isArray(value)) {
    value.forEach((v, i) => out.push(...flatten(v, join(i), seen)));
  } else if (value instanceof PyObject) {
    if (value.state instanceof Map) {
      for (const key of ['_parameters', '_buffers', '_modules']) {
        const m = value.state.get(key);
        if (m instanceof Map) for (const [k, v] of m) out.push(...flatten(v, join(k), seen));
      }
      if (!value.state.has('_modules')) out.push(...flatten(value.state, prefix, seen));
    }
    if (value.items) out.push(...flatten(value.items, prefix, seen));
  }
  return out;
}

// ---------------------------------------------------------------- building

export function buildModel(c: Candidate): ModelInfo {
  const root =
    c.kind === 'module'
      ? fromModule(c.value as PyObject, '', '', 0, new Set())
      : fromStateDict(flatten(c.value));
  const index = new Map<string, ModuleNode>();
  const leaves: ModuleNode[] = [];
  const walk = (n: ModuleNode) => {
    index.set(n.path, n);
    if (n.children.length === 0) leaves.push(n);
    n.children.forEach(walk);
  };
  walk(root);
  return { root, index, leaves, source: c.kind === 'module' ? 'module' : 'state_dict' };
}

function fromModule(obj: PyObject, name: string, path: string, depth: number, seen: Set<PyObject>): ModuleNode {
  seen.add(obj);
  const node = emptyNode(name, path, depth);
  node.type = obj.cls.name;
  const join = (k: string) => (path ? `${path}.${k}` : k);
  const state = obj.state instanceof Map ? (obj.state as Map<unknown, unknown>) : new Map();

  if (state.has('_modules')) {
    // Regular eager-mode nn.Module pickled via torch.save(model).
    for (const [key, kind] of [
      ['_parameters', 'parameter'],
      ['_buffers', 'buffer'],
    ] as const) {
      const m = state.get(key);
      if (!(m instanceof Map)) continue;
      for (const [k, v] of m) {
        if (v instanceof Tensor) node.params.push({ name: String(k), path: join(String(k)), tensor: v, kind });
      }
    }
    const mods = state.get('_modules');
    if (mods instanceof Map) {
      for (const [k, v] of mods) {
        if (v instanceof PyObject && !seen.has(v)) node.children.push(fromModule(v, String(k), join(String(k)), depth + 1, seen));
      }
    }
    for (const [k, v] of state) {
      const key = String(k);
      if (key.startsWith('_') || key === 'training') continue;
      const s = attrString(v);
      if (s !== null) node.attrs.push([key, s]);
    }
  } else {
    // TorchScript: attributes live directly on the object.
    for (const [k, v] of state) {
      const key = String(k);
      if (v instanceof Tensor) {
        node.params.push({ name: key, path: join(key), tensor: v, kind: v.requiresGrad || v.isParameter ? 'parameter' : 'buffer' });
      } else if (isModule(v) && !seen.has(v)) {
        node.children.push(fromModule(v, key, join(key), depth + 1, seen));
      } else if (!key.startsWith('_') && key !== 'training') {
        const s = attrString(v);
        if (s !== null) node.attrs.push([key, s]);
      }
    }
  }

  node.summary = summarize(node.type, node.params, node.attrs);
  finalize(node);
  return node;
}

function fromStateDict(entries: [string, Tensor][]): ModuleNode {
  const root = emptyNode('', '', 0);
  for (const [fullName, tensor] of entries) {
    const parts = fullName.split('.');
    const pname = parts.pop()!;
    let node = root;
    for (const part of parts) {
      let child = node.children.find((c) => c.name === part);
      if (!child) {
        child = emptyNode(part, node.path ? `${node.path}.${part}` : part, node.depth + 1);
        node.children.push(child);
      }
      node = child;
    }
    const isBuffer =
      /^(running_mean|running_var|num_batches_tracked)$/.test(pname) ||
      (!tensor.isParameter && /buffer|mask|position_ids|inv_freq/.test(pname));
    const kind = isBuffer ? 'buffer' : 'parameter';
    node.params.push({ name: pname, path: fullName, tensor, kind });
  }
  const infer = (n: ModuleNode) => {
    n.children.forEach(infer);
    const t = inferType(n);
    n.type = t;
    n.inferred = true;
    n.summary = summarize(t, n.params, n.attrs);
    finalize(n);
  };
  infer(root);
  if (root.children.length || root.type === 'Module') root.type = 'Model';
  return root;
}

function emptyNode(name: string, path: string, depth: number): ModuleNode {
  return {
    path,
    name,
    type: 'Module',
    inferred: false,
    summary: '',
    attrs: [],
    params: [],
    children: [],
    totalParams: 0,
    totalBytes: 0,
    depth,
  };
}

function finalize(n: ModuleNode) {
  n.totalParams = n.params.filter((p) => p.kind === 'parameter').reduce((a, p) => a + p.tensor.numel, 0);
  n.totalBytes = n.params.reduce((a, p) => a + p.tensor.nbytes, 0);
  for (const c of n.children) {
    n.totalParams += c.totalParams;
    n.totalBytes += c.totalBytes;
  }
}

function attrString(v: unknown): string | null {
  if (v === null) return 'None';
  if (typeof v === 'number') return Number.isInteger(v) ? String(v) : String(+v.toPrecision(6));
  if (typeof v === 'boolean') return v ? 'True' : 'False';
  if (typeof v === 'string') return v.length > 80 ? null : JSON.stringify(v);
  if (Array.isArray(v) && v.length <= 8 && v.every((x) => typeof x === 'number' || typeof x === 'string')) {
    return `(${v.join(', ')})`;
  }
  return null;
}

// ---------------------------------------------------------------- inference

function inferType(n: ModuleNode): string {
  const p = new Map(n.params.map((e) => [e.name, e.tensor]));
  const w = p.get('weight');
  const nm = n.name.toLowerCase();

  if (n.children.length && n.params.length === 0) {
    const numeric = n.children.every((c) => /^\d+$/.test(c.name));
    return numeric ? 'Sequential' : 'Module';
  }
  if (p.has('in_proj_weight')) return 'MultiheadAttention';
  const ih = p.get('weight_ih_l0');
  const hh = p.get('weight_hh_l0');
  if (ih && hh) {
    const ratio = ih.shape[0] / hh.shape[1];
    return ratio === 4 ? 'LSTM' : ratio === 3 ? 'GRU' : 'RNN';
  }
  if (p.has('running_mean')) return 'BatchNorm';
  if (!w) return n.params.length ? 'Parameters' : 'Module';
  if (w.ndim >= 3 && w.ndim <= 5) return `Conv${w.ndim - 2}d`;
  if (w.ndim === 2) return /emb|wte|wpe|token/.test(nm) ? 'Embedding' : 'Linear';
  if (w.ndim === 1) {
    if (/group/.test(nm)) return 'GroupNorm';
    if (/rms/.test(nm)) return 'RMSNorm';
    return /bn|batch/.test(nm) ? 'BatchNorm' : 'LayerNorm';
  }
  return 'Module';
}

const shapeStr = (s: number[]) => s.join('×');

function summarize(type: string, params: ParamEntry[], attrs: [string, string][]): string {
  const p = new Map(params.map((e) => [e.name, e.tensor]));
  const a = new Map(attrs);
  const w = p.get('weight');
  const t = type.replace(/^_+/, '');

  if (/^Linear|LazyLinear|NonDynamicallyQuantizableLinear/.test(t) && w?.ndim === 2) {
    return `${w.shape[1]} → ${w.shape[0]}`;
  }
  if (/^Conv(Transpose)?\dd$/.test(t) && w && w.ndim >= 3) {
    const groups = Number(a.get('groups') ?? 1);
    const transposed = t.includes('Transpose');
    const [o, i] = transposed ? [w.shape[1] * groups, w.shape[0]] : [w.shape[0], w.shape[1] * groups];
    let s = `${i} → ${o}, k=${shapeStr(w.shape.slice(2))}`;
    const stride = a.get('stride');
    if (stride && !/^\(1(, 1)*\)$/.test(stride)) s += `, s=${stride.replace(/[()]/g, '')}`;
    return s;
  }
  if (t === 'Embedding' && w?.ndim === 2) return `${w.shape[0]} × ${w.shape[1]}`;
  if (/Norm/.test(t) && w) return shapeStr(w.shape);
  if (/Norm/.test(t) && p.get('running_mean')) return shapeStr(p.get('running_mean')!.shape);
  if (/^(LSTM|GRU|RNN)$/.test(t)) {
    const ih = p.get('weight_ih_l0');
    const hh = p.get('weight_hh_l0');
    const layers = params.filter((e) => /^weight_ih_l\d+$/.test(e.name)).length;
    if (ih && hh) return `${ih.shape[1]} → ${hh.shape[1]}${layers > 1 ? `, ${layers} layers` : ''}`;
  }
  if (t === 'MultiheadAttention') {
    const ipw = p.get('in_proj_weight');
    const heads = a.get('num_heads');
    if (ipw) return `d=${ipw.shape[1]}${heads ? `, heads=${heads}` : ''}`;
  }
  if (/Dropout/.test(t) && a.has('p')) return `p=${a.get('p')}`;
  if (/Pool/.test(t) && a.has('kernel_size')) return `k=${a.get('kernel_size')!.replace(/[()]/g, '')}`;
  if (/Pool/.test(t) && a.has('output_size')) return `out=${a.get('output_size')!.replace(/[()]/g, '')}`;
  if (w) return shapeStr(w.shape);
  return '';
}

// ---------------------------------------------------------------- styling

export type LayerCategory = 'linear' | 'conv' | 'norm' | 'activation' | 'embedding' | 'recurrent' | 'attention' | 'pool' | 'dropout' | 'container' | 'other';

export function categorize(type: string): LayerCategory {
  if (/Conv/.test(type)) return 'conv';
  if (/Linear|Bilinear/.test(type)) return 'linear';
  if (/Norm/.test(type)) return 'norm';
  if (/Embedding/.test(type)) return 'embedding';
  if (/LSTM|GRU|RNN/.test(type)) return 'recurrent';
  if (/Attention|Transformer/.test(type)) return 'attention';
  if (/Pool/.test(type)) return 'pool';
  if (/Dropout/.test(type)) return 'dropout';
  if (/ReLU|GELU|SiLU|Sigmoid|Tanh|Softmax|ELU|Mish|Swish|Hardswish|Softplus|Identity|Flatten/.test(type)) return 'activation';
  if (/Sequential|ModuleList|ModuleDict|Model|Module|Block|Layer|Stage/.test(type)) return 'container';
  return 'other';
}
