import { unzipSync } from 'fflate';
import { PyClass, Unpickler, type PyCallable } from './pickle';
import { Tensor, type DType } from './tensor';

export interface LoadedFile {
  format: 'pytorch' | 'torchscript' | 'safetensors';
  /** The unpickled root object (module, state_dict, checkpoint dict, ...). */
  root: unknown;
}

const STORAGE_DTYPES: Record<string, DType> = {
  DoubleStorage: 'float64',
  FloatStorage: 'float32',
  HalfStorage: 'float16',
  BFloat16Storage: 'bfloat16',
  LongStorage: 'int64',
  IntStorage: 'int32',
  ShortStorage: 'int16',
  CharStorage: 'int8',
  ByteStorage: 'uint8',
  BoolStorage: 'bool',
};

const TORCH_DTYPES: Record<string, DType> = {
  float64: 'float64',
  double: 'float64',
  float32: 'float32',
  float: 'float32',
  float16: 'float16',
  half: 'float16',
  bfloat16: 'bfloat16',
  int64: 'int64',
  long: 'int64',
  int32: 'int32',
  int: 'int32',
  int16: 'int16',
  short: 'int16',
  int8: 'int8',
  uint8: 'uint8',
  bool: 'bool',
};

class StorageRef {
  constructor(
    public dtype: DType,
    public bytes: Uint8Array,
  ) {}
}

const fn = (f: (args: unknown[]) => unknown): PyCallable => ({ __call__: f });

function rebuildTensor(args: unknown[]): Tensor {
  const [storage, offset, size, stride, requiresGrad] = args as [
    StorageRef,
    number,
    number[],
    number[],
    boolean,
  ];
  if (!(storage instanceof StorageRef)) throw new Error('Tensor without storage');
  const t = new Tensor(storage.dtype, [...size], storage.bytes, offset, [...stride]);
  t.requiresGrad = !!requiresGrad;
  return t;
}

function markParameter(t: unknown, requiresGrad: unknown) {
  if (t instanceof Tensor) {
    t.isParameter = true;
    t.requiresGrad = requiresGrad !== false;
  }
  return t;
}

/** Map pickle GLOBAL references onto JS implementations where we need real behaviour. */
function torchGlobals(module: string, name: string): unknown {
  const full = `${module}.${name}`;
  if (module === 'torch' && name in STORAGE_DTYPES) return { storageDtype: STORAGE_DTYPES[name] };
  if (module === 'torch' && name in TORCH_DTYPES) return { torchDtype: TORCH_DTYPES[name] };
  switch (full) {
    case 'torch._utils._rebuild_tensor_v2':
    case 'torch._utils._rebuild_tensor':
      return fn(rebuildTensor);
    case 'torch._utils._rebuild_parameter':
    case 'torch._utils._rebuild_parameter_with_state':
      return fn(([data, rg]) => markParameter(data, rg));
    case 'torch.nn.parameter.Parameter':
      return fn(([data, rg]) => markParameter(data, rg));
    case 'torch._tensor._rebuild_from_type_v2':
      // (func, new_type, args, state): rebuild with func(*args), ignore subclass.
      return fn(([f, newType, a]) => {
        const t = (f as PyCallable).__call__(a as unknown[]);
        if (newType instanceof PyClass && newType.name === 'Parameter') markParameter(t, true);
        return t;
      });
    case 'collections.OrderedDict':
    case 'builtins.dict':
    case '__builtin__.dict':
      return fn(() => new Map());
    case 'builtins.set':
    case '__builtin__.set':
      return fn(([items]) => new Set((items as unknown[]) ?? []));
    case 'builtins.list':
      return fn(([items]) => [...((items as unknown[]) ?? [])]);
    case 'builtins.tuple':
      return fn(([items]) => [...((items as unknown[]) ?? [])]);
    case 'torch.Size':
      return fn(([dims]) => [...(dims as number[])]);
    case 'torch.device':
      return fn(([d]) => `device(${String(d)})`);
    case 'builtins.slice':
      return fn((a) => `slice(${a.join(', ')})`);
    case '_codecs.encode':
      return fn(([s]) => s);
    case 'numpy.core.multiarray.scalar':
    case 'numpy._core.multiarray.scalar':
      return fn(([, bytes]) => bytes);
  }
  return new PyClass(module, name);
}

export async function loadFile(file: File): Promise<LoadedFile> {
  const buf = new Uint8Array(await file.arrayBuffer());
  return loadBytes(buf, file.name);
}

export function loadBytes(buf: Uint8Array, filename = ''): LoadedFile {
  // Zip magic "PK\x03\x04" → torch.save (>=1.6) or TorchScript archive.
  if (buf[0] === 0x50 && buf[1] === 0x4b) return loadTorchZip(buf);
  if (filename.endsWith('.safetensors') || looksLikeSafetensors(buf)) return loadSafetensors(buf);
  if (buf[0] === 0x80) {
    throw new Error(
      'This looks like a legacy (pre-1.6) torch.save file, which is not supported. ' +
        'Re-save it with a recent PyTorch: torch.save(torch.load(path), new_path).',
    );
  }
  throw new Error('Unrecognized file format. Expected a .pt/.pth zip archive or .safetensors file.');
}

function loadTorchZip(buf: Uint8Array): LoadedFile {
  const files = unzipSync(buf);
  const names = Object.keys(files);
  const pklName = names.find((n) => /(^|\/)data\.pkl$/.test(n));
  if (!pklName) throw new Error('Archive has no data.pkl — not a PyTorch checkpoint.');
  const prefix = pklName.slice(0, -'data.pkl'.length);
  const isScript = names.some((n) => n.startsWith(prefix + 'code/'));

  const byteorder = files[prefix + 'byteorder'];
  if (byteorder && new TextDecoder().decode(byteorder).trim() === 'big') {
    throw new Error('Big-endian checkpoints are not supported.');
  }

  const storages = new Map<string, StorageRef>();
  const persistentLoad = (pid: unknown) => {
    const tuple = pid as unknown[];
    if (!Array.isArray(tuple) || tuple[0] !== 'storage') {
      throw new Error(`Unsupported persistent id ${JSON.stringify(pid)}`);
    }
    const [, storageType, key] = tuple as [string, unknown, string];
    const cached = storages.get(key);
    if (cached) return cached;
    let dtype: DType = 'unknown';
    if (storageType && typeof storageType === 'object') {
      if ('storageDtype' in storageType) dtype = (storageType as { storageDtype: DType }).storageDtype;
      else if ('torchDtype' in storageType) dtype = (storageType as { torchDtype: DType }).torchDtype;
    }
    const bytes = files[`${prefix}data/${key}`];
    if (!bytes) throw new Error(`Missing tensor storage data/${key}`);
    const ref = new StorageRef(dtype, bytes);
    storages.set(key, ref);
    return ref;
  };

  const root = new Unpickler(files[pklName], torchGlobals, persistentLoad).load();
  return { format: isScript ? 'torchscript' : 'pytorch', root };
}

const SAFETENSOR_DTYPES: Record<string, DType> = {
  F64: 'float64',
  F32: 'float32',
  F16: 'float16',
  BF16: 'bfloat16',
  I64: 'int64',
  I32: 'int32',
  I16: 'int16',
  I8: 'int8',
  U8: 'uint8',
  BOOL: 'bool',
};

function looksLikeSafetensors(buf: Uint8Array) {
  if (buf.length < 10) return false;
  const n = Number(new DataView(buf.buffer, buf.byteOffset).getBigUint64(0, true));
  return n > 0 && n < buf.length && buf[8] === 0x7b; // '{'
}

function loadSafetensors(buf: Uint8Array): LoadedFile {
  const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const headerLen = Number(view.getBigUint64(0, true));
  const header = JSON.parse(new TextDecoder().decode(buf.subarray(8, 8 + headerLen))) as Record<
    string,
    { dtype: string; shape: number[]; data_offsets: [number, number] }
  >;
  const base = 8 + headerLen;
  const root = new Map<string, unknown>();
  for (const [name, info] of Object.entries(header)) {
    if (name === '__metadata__') continue;
    const [start, end] = info.data_offsets;
    const bytes = buf.subarray(base + start, base + end);
    const t = new Tensor(SAFETENSOR_DTYPES[info.dtype] ?? 'unknown', info.shape, bytes);
    t.isParameter = true;
    root.set(name, t);
  }
  return { format: 'safetensors', root };
}
