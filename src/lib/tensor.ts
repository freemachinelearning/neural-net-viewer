export type DType =
  | 'float64'
  | 'float32'
  | 'float16'
  | 'bfloat16'
  | 'int64'
  | 'int32'
  | 'int16'
  | 'int8'
  | 'uint8'
  | 'bool'
  | 'unknown';

export const DTYPE_BYTES: Record<DType, number> = {
  float64: 8,
  float32: 4,
  float16: 2,
  bfloat16: 2,
  int64: 8,
  int32: 4,
  int16: 2,
  int8: 1,
  uint8: 1,
  bool: 1,
  unknown: 1,
};

export interface TensorStats {
  count: number;
  min: number;
  max: number;
  mean: number;
  std: number;
  absMax: number;
  zeros: number;
  nonFinite: number;
  l2: number;
}

/**
 * A tensor backed by raw little-endian bytes. Values are decoded lazily into a
 * contiguous Float32Array the first time they are needed.
 */
export class Tensor {
  private cache: Float32Array | null = null;
  private statsCache: TensorStats | null = null;
  requiresGrad = false;
  isParameter = false;

  constructor(
    public dtype: DType,
    public shape: number[],
    /** Raw bytes of the whole underlying storage. */
    private storage: Uint8Array,
    private storageOffset = 0,
    private stride: number[] | null = null,
  ) {}

  get numel() {
    return this.shape.reduce((a, b) => a * b, 1);
  }

  get nbytes() {
    return this.numel * DTYPE_BYTES[this.dtype];
  }

  get ndim() {
    return this.shape.length;
  }

  /** Values as a contiguous, row-major Float32Array. */
  values(): Float32Array {
    if (this.cache) return this.cache;
    const flat = decodeStorage(this.dtype, this.storage);
    const n = this.numel;
    const contiguous = contiguousStride(this.shape);
    const stride = this.stride ?? contiguous;
    let out: Float32Array;
    if (stride.every((s, i) => s === contiguous[i]) || n <= 1) {
      out = flat.subarray(this.storageOffset, this.storageOffset + n);
    } else {
      out = new Float32Array(n);
      const idx = new Array(this.shape.length).fill(0);
      for (let i = 0; i < n; i++) {
        let off = this.storageOffset;
        for (let d = 0; d < idx.length; d++) off += idx[d] * stride[d];
        out[i] = flat[off];
        for (let d = idx.length - 1; d >= 0; d--) {
          if (++idx[d] < this.shape[d]) break;
          idx[d] = 0;
        }
      }
    }
    this.cache = out;
    return out;
  }

  stats(): TensorStats {
    if (this.statsCache) return this.statsCache;
    const v = this.values();
    let min = Infinity,
      max = -Infinity,
      sum = 0,
      sumSq = 0,
      zeros = 0,
      nonFinite = 0,
      count = 0;
    for (let i = 0; i < v.length; i++) {
      const x = v[i];
      if (!Number.isFinite(x)) {
        nonFinite++;
        continue;
      }
      if (x < min) min = x;
      if (x > max) max = x;
      if (x === 0) zeros++;
      sum += x;
      sumSq += x * x;
      count++;
    }
    const mean = count ? sum / count : 0;
    const variance = count ? Math.max(0, sumSq / count - mean * mean) : 0;
    this.statsCache = {
      count: v.length,
      min: count ? min : 0,
      max: count ? max : 0,
      mean,
      std: Math.sqrt(variance),
      absMax: count ? Math.max(Math.abs(min), Math.abs(max)) : 0,
      zeros,
      nonFinite,
      l2: Math.sqrt(sumSq),
    };
    return this.statsCache;
  }
}

export function contiguousStride(shape: number[]): number[] {
  const s = new Array(shape.length);
  let acc = 1;
  for (let i = shape.length - 1; i >= 0; i--) {
    s[i] = acc;
    acc *= shape[i];
  }
  return s;
}

let halfTable: Float32Array | null = null;
function getHalfTable() {
  if (halfTable) return halfTable;
  halfTable = new Float32Array(65536);
  for (let h = 0; h < 65536; h++) {
    const s = h & 0x8000 ? -1 : 1;
    const e = (h >> 10) & 0x1f;
    const f = h & 0x3ff;
    if (e === 0) halfTable[h] = s * Math.pow(2, -14) * (f / 1024);
    else if (e === 31) halfTable[h] = f ? NaN : s * Infinity;
    else halfTable[h] = s * Math.pow(2, e - 15) * (1 + f / 1024);
  }
  return halfTable;
}

/** Ensures the buffer is aligned for a typed-array view of the given element size. */
function aligned(bytes: Uint8Array, size: number): Uint8Array {
  return bytes.byteOffset % size === 0 ? bytes : bytes.slice();
}

function decodeStorage(dtype: DType, raw: Uint8Array): Float32Array {
  const size = DTYPE_BYTES[dtype];
  const b = aligned(raw, size);
  const n = Math.floor(b.byteLength / size);
  switch (dtype) {
    case 'float32':
      return new Float32Array(b.buffer, b.byteOffset, n);
    case 'float64':
      return Float32Array.from(new Float64Array(b.buffer, b.byteOffset, n));
    case 'float16': {
      const t = getHalfTable();
      const u = new Uint16Array(b.buffer, b.byteOffset, n);
      const out = new Float32Array(n);
      for (let i = 0; i < n; i++) out[i] = t[u[i]];
      return out;
    }
    case 'bfloat16': {
      const u = new Uint16Array(b.buffer, b.byteOffset, n);
      const bits = new Uint32Array(n);
      for (let i = 0; i < n; i++) bits[i] = u[i] << 16;
      return new Float32Array(bits.buffer);
    }
    case 'int64': {
      const src = new BigInt64Array(b.buffer, b.byteOffset, n);
      const out = new Float32Array(n);
      for (let i = 0; i < n; i++) out[i] = Number(src[i]);
      return out;
    }
    case 'int32':
      return Float32Array.from(new Int32Array(b.buffer, b.byteOffset, n));
    case 'int16':
      return Float32Array.from(new Int16Array(b.buffer, b.byteOffset, n));
    case 'int8':
      return Float32Array.from(new Int8Array(b.buffer, b.byteOffset, n));
    case 'uint8':
    case 'bool':
    case 'unknown':
      return Float32Array.from(b.subarray(0, n));
  }
}

export function histogram(t: Tensor, bins = 60) {
  const { min, max } = t.stats();
  const v = t.values();
  const counts = new Array(bins).fill(0);
  const span = max - min || 1;
  for (let i = 0; i < v.length; i++) {
    const x = v[i];
    if (!Number.isFinite(x)) continue;
    let b = Math.floor(((x - min) / span) * bins);
    if (b >= bins) b = bins - 1;
    counts[b]++;
  }
  return { counts, min, max: min === max ? min + 1 : max };
}

/** Collapse any tensor into a 2D [rows, cols] matrix view for display. */
export function as2D(shape: number[]): [number, number] {
  if (shape.length === 0) return [1, 1];
  if (shape.length === 1) return [1, shape[0]];
  const rows = shape[0];
  return [rows, shape.slice(1).reduce((a, b) => a * b, 1)];
}

export function formatNumber(n: number): string {
  if (n >= 1e9) return (n / 1e9).toFixed(2) + 'B';
  if (n >= 1e6) return (n / 1e6).toFixed(2) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
  return String(n);
}

export function formatBytes(n: number): string {
  if (n >= 1 << 30) return (n / (1 << 30)).toFixed(2) + ' GB';
  if (n >= 1 << 20) return (n / (1 << 20)).toFixed(2) + ' MB';
  if (n >= 1 << 10) return (n / (1 << 10)).toFixed(1) + ' KB';
  return n + ' B';
}

export function formatValue(x: number): string {
  if (!Number.isFinite(x)) return String(x);
  if (x === 0) return '0';
  const a = Math.abs(x);
  if (a >= 1e4 || a < 1e-3) return x.toExponential(3);
  return x.toPrecision(4);
}
