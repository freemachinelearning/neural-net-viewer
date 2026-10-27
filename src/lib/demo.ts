import { Tensor } from './tensor';
import type { LoadedFile } from './loaders';

// Deterministic PRNG so the demo looks the same on every load.
function mulberry32(seed: number) {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const rand = mulberry32(42);
const gauss = () => Math.sqrt(-2 * Math.log(rand() + 1e-12)) * Math.cos(2 * Math.PI * rand());

function tensor(shape: number[], fill: (i: number) => number): Tensor {
  const n = shape.reduce((a, b) => a * b, 1);
  const data = new Float32Array(n);
  for (let i = 0; i < n; i++) data[i] = fill(i);
  const t = new Tensor('float32', shape, new Uint8Array(data.buffer));
  t.isParameter = true;
  return t;
}

const kaiming = (shape: number[]) => {
  const fanIn = shape.slice(1).reduce((a, b) => a * b, 1);
  const std = Math.sqrt(2 / fanIn);
  return tensor(shape, () => gauss() * std);
};

/** Gabor-like edge detectors, the kind of filters a trained first conv layer learns. */
function gaborFilters(out: number, k: number) {
  return tensor([out, 1, k, k], (i) => {
    const f = Math.floor(i / (k * k));
    const y = Math.floor((i % (k * k)) / k) - (k - 1) / 2;
    const x = (i % k) - (k - 1) / 2;
    const theta = (f / out) * Math.PI;
    const freq = 0.35 + (f % 3) * 0.2;
    const xr = x * Math.cos(theta) + y * Math.sin(theta);
    const yr = -x * Math.sin(theta) + y * Math.cos(theta);
    const env = Math.exp(-(xr * xr + yr * yr) / (2 * 1.6 * 1.6));
    return env * Math.cos(2 * Math.PI * freq * xr + (f % 2) * 1.2) * 0.5 + gauss() * 0.02;
  });
}

/** A small MNIST-style CNN expressed as a state_dict. */
export function demoModel(): LoadedFile {
  const sd = new Map<string, Tensor>();
  const bn = (name: string, c: number) => {
    sd.set(`${name}.weight`, tensor([c], () => 1 + gauss() * 0.1));
    sd.set(`${name}.bias`, tensor([c], () => gauss() * 0.05));
    const rm = tensor([c], () => gauss() * 0.2);
    const rv = tensor([c], () => 0.5 + rand());
    rm.isParameter = rv.isParameter = false;
    sd.set(`${name}.running_mean`, rm);
    sd.set(`${name}.running_var`, rv);
  };
  sd.set('features.0.weight', gaborFilters(16, 5));
  sd.set('features.0.bias', tensor([16], () => gauss() * 0.05));
  bn('features.1', 16);
  sd.set('features.4.weight', kaiming([32, 16, 3, 3]));
  sd.set('features.4.bias', tensor([32], () => gauss() * 0.05));
  bn('features.5', 32);
  // Sparse-ish hidden layer, as if L1-regularized.
  sd.set('classifier.1.weight', tensor([128, 32 * 7 * 7], () => (rand() < 0.7 ? 0 : gauss() * 0.08)));
  sd.set('classifier.1.bias', tensor([128], () => gauss() * 0.02));
  // Output layer with a few strongly-connected neurons per class.
  sd.set(
    'classifier.4.weight',
    tensor([10, 128], (i) => {
      const cls = Math.floor(i / 128);
      const j = i % 128;
      return (j % 10 === cls ? 0.6 : 0) + gauss() * 0.1;
    }),
  );
  sd.set('classifier.4.bias', tensor([10], () => gauss() * 0.02));
  return { format: 'pytorch', root: sd };
}
