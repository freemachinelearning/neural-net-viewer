// A minimal Python pickle virtual machine (protocols 0–5), enough to read
// the data.pkl that torch.save() writes. Unknown classes are not executed;
// they become PyObject placeholders that record their constructor args and state.

export class PyClass {
  constructor(public module: string, public name: string) {}
  get fullName() {
    return `${this.module}.${this.name}`;
  }
}

export class PyObject {
  args: unknown[] = [];
  state: unknown = undefined;
  // Items added via SETITEM(S) / APPEND(S) on dict or list subclasses.
  items: Map<unknown, unknown> | null = null;
  list: unknown[] | null = null;
  constructor(public cls: PyClass) {}
}

export type GlobalResolver = (module: string, name: string) => unknown;
export type PersistentLoader = (pid: unknown) => unknown;
/** A callable produced by the resolver. Receives the REDUCE args. */
export type PyCallable = { __call__: (args: unknown[]) => unknown };

export function isCallable(x: unknown): x is PyCallable {
  return typeof x === 'object' && x !== null && '__call__' in x;
}

export class Unpickler {
  private pos = 0;
  private stack: unknown[] = [];
  private metastack: unknown[][] = [];
  private memo = new Map<number, unknown>();
  private view: DataView;
  private decoder = new TextDecoder('utf-8');
  private latin1 = new TextDecoder('latin1');

  constructor(
    private data: Uint8Array,
    private resolveGlobal: GlobalResolver = (m, n) => new PyClass(m, n),
    private persistentLoad: PersistentLoader = (pid) => {
      throw new Error(`Unsupported persistent id: ${String(pid)}`);
    },
  ) {
    this.view = new DataView(data.buffer, data.byteOffset, data.byteLength);
  }

  load(): unknown {
    for (;;) {
      if (this.pos >= this.data.length) throw new Error('Pickle ended without STOP');
      const op = this.data[this.pos++];
      switch (op) {
        case 0x80: // PROTO
          this.pos++;
          break;
        case 0x95: // FRAME
          this.pos += 8;
          break;
        case 0x2e: // STOP
          return this.stack.pop();
        case 0x28: // MARK
          this.metastack.push(this.stack);
          this.stack = [];
          break;
        case 0x30: // POP
          if (this.stack.length) this.stack.pop();
          else this.popMark();
          break;
        case 0x31: // POP_MARK
          this.popMark();
          break;
        case 0x32: // DUP
          this.stack.push(this.stack[this.stack.length - 1]);
          break;

        // ---- numbers
        case 0x4a: // BININT
          this.stack.push(this.view.getInt32(this.pos, true));
          this.pos += 4;
          break;
        case 0x4b: // BININT1
          this.stack.push(this.data[this.pos++]);
          break;
        case 0x4d: // BININT2
          this.stack.push(this.view.getUint16(this.pos, true));
          this.pos += 2;
          break;
        case 0x8a: {
          // LONG1
          const n = this.data[this.pos++];
          this.stack.push(this.readLong(n));
          break;
        }
        case 0x8b: {
          // LONG4
          const n = this.view.getInt32(this.pos, true);
          this.pos += 4;
          this.stack.push(this.readLong(n));
          break;
        }
        case 0x49: {
          // INT
          const s = this.readLine();
          if (s === '01') this.stack.push(true);
          else if (s === '00') this.stack.push(false);
          else this.stack.push(parseInt(s, 10));
          break;
        }
        case 0x4c: // LONG
          this.stack.push(parseInt(this.readLine().replace(/L$/, ''), 10));
          break;
        case 0x47: // BINFLOAT (big-endian double)
          this.stack.push(this.view.getFloat64(this.pos, false));
          this.pos += 8;
          break;
        case 0x46: // FLOAT
          this.stack.push(parseFloat(this.readLine()));
          break;

        // ---- singletons
        case 0x4e: // NONE
          this.stack.push(null);
          break;
        case 0x88: // NEWTRUE
          this.stack.push(true);
          break;
        case 0x89: // NEWFALSE
          this.stack.push(false);
          break;

        // ---- strings / bytes
        case 0x58: {
          // BINUNICODE
          const n = this.view.getUint32(this.pos, true);
          this.pos += 4;
          this.stack.push(this.readUtf8(n));
          break;
        }
        case 0x8c: {
          // SHORT_BINUNICODE
          const n = this.data[this.pos++];
          this.stack.push(this.readUtf8(n));
          break;
        }
        case 0x8d: {
          // BINUNICODE8
          const n = Number(this.view.getBigUint64(this.pos, true));
          this.pos += 8;
          this.stack.push(this.readUtf8(n));
          break;
        }
        case 0x56: // UNICODE (raw-unicode-escape, line terminated)
          this.stack.push(this.readLine());
          break;
        case 0x53: {
          // STRING (quoted repr)
          const s = this.readLine();
          this.stack.push(s.slice(1, -1));
          break;
        }
        case 0x54: {
          // BINSTRING
          const n = this.view.getInt32(this.pos, true);
          this.pos += 4;
          this.stack.push(this.readLatin1(n));
          break;
        }
        case 0x55: {
          // SHORT_BINSTRING
          const n = this.data[this.pos++];
          this.stack.push(this.readLatin1(n));
          break;
        }
        case 0x42: {
          // BINBYTES
          const n = this.view.getUint32(this.pos, true);
          this.pos += 4;
          this.stack.push(this.readBytes(n));
          break;
        }
        case 0x43: {
          // SHORT_BINBYTES
          const n = this.data[this.pos++];
          this.stack.push(this.readBytes(n));
          break;
        }
        case 0x8e: // BINBYTES8
        case 0x96: {
          // BYTEARRAY8
          const n = Number(this.view.getBigUint64(this.pos, true));
          this.pos += 8;
          this.stack.push(this.readBytes(n));
          break;
        }

        // ---- containers
        case 0x5d: // EMPTY_LIST
          this.stack.push([]);
          break;
        case 0x61: {
          // APPEND
          const v = this.stack.pop();
          this.appendTo(this.top(), [v]);
          break;
        }
        case 0x65: {
          // APPENDS
          const items = this.popMark();
          this.appendTo(this.top(), items);
          break;
        }
        case 0x6c: // LIST
          this.push(this.popMark());
          break;
        case 0x29: // EMPTY_TUPLE
          this.stack.push([]);
          break;
        case 0x74: // TUPLE
          this.push(this.popMark());
          break;
        case 0x85: // TUPLE1
          this.stack.push([this.stack.pop()]);
          break;
        case 0x86: {
          // TUPLE2
          const b = this.stack.pop();
          const a = this.stack.pop();
          this.stack.push([a, b]);
          break;
        }
        case 0x87: {
          // TUPLE3
          const c = this.stack.pop();
          const b = this.stack.pop();
          const a = this.stack.pop();
          this.stack.push([a, b, c]);
          break;
        }
        case 0x7d: // EMPTY_DICT
          this.stack.push(new Map());
          break;
        case 0x64: {
          // DICT
          const items = this.popMark();
          const m = new Map();
          for (let i = 0; i < items.length; i += 2) m.set(dictKey(items[i]), items[i + 1]);
          this.stack.push(m);
          break;
        }
        case 0x73: {
          // SETITEM
          const v = this.stack.pop();
          const k = this.stack.pop();
          this.setItems(this.top(), [k, v]);
          break;
        }
        case 0x75: {
          // SETITEMS
          const items = this.popMark();
          this.setItems(this.top(), items);
          break;
        }
        case 0x8f: // EMPTY_SET
          this.stack.push(new Set());
          break;
        case 0x90: {
          // ADDITEMS
          const items = this.popMark();
          const s = this.top() as Set<unknown>;
          for (const it of items) s.add(it);
          break;
        }
        case 0x91: // FROZENSET
          this.push(new Set(this.popMark()));
          break;

        // ---- objects
        case 0x63: {
          // GLOBAL
          const module = this.readLine();
          const name = this.readLine();
          this.stack.push(this.resolveGlobal(module, name));
          break;
        }
        case 0x93: {
          // STACK_GLOBAL
          const name = this.stack.pop() as string;
          const module = this.stack.pop() as string;
          this.stack.push(this.resolveGlobal(module, name));
          break;
        }
        case 0x52: {
          // REDUCE
          const args = this.stack.pop() as unknown[];
          const fn = this.stack.pop();
          this.stack.push(this.call(fn, args));
          break;
        }
        case 0x81: {
          // NEWOBJ
          const args = this.stack.pop() as unknown[];
          const cls = this.stack.pop();
          this.stack.push(this.call(cls, args));
          break;
        }
        case 0x92: {
          // NEWOBJ_EX
          this.stack.pop(); // kwargs
          const args = this.stack.pop() as unknown[];
          const cls = this.stack.pop();
          this.stack.push(this.call(cls, args));
          break;
        }
        case 0x6f: {
          // OBJ
          const items = this.popMark();
          const cls = items.shift();
          this.stack.push(this.call(cls, items));
          break;
        }
        case 0x69: {
          // INST
          const module = this.readLine();
          const name = this.readLine();
          const args = this.popMark();
          this.stack.push(this.call(this.resolveGlobal(module, name), args));
          break;
        }
        case 0x62: {
          // BUILD
          const state = this.stack.pop();
          const obj = this.top();
          if (obj instanceof PyObject) obj.state = state;
          else if (obj && typeof obj === 'object' && 'setState' in obj) {
            (obj as { setState(s: unknown): void }).setState(state);
          }
          break;
        }

        // ---- memo
        case 0x71: // BINPUT
          this.memo.set(this.data[this.pos++], this.top());
          break;
        case 0x72: // LONG_BINPUT
          this.memo.set(this.view.getUint32(this.pos, true), this.top());
          this.pos += 4;
          break;
        case 0x70: // PUT
          this.memo.set(parseInt(this.readLine(), 10), this.top());
          break;
        case 0x94: // MEMOIZE
          this.memo.set(this.memo.size, this.top());
          break;
        case 0x68: // BINGET
          this.stack.push(this.memo.get(this.data[this.pos++]));
          break;
        case 0x6a: // LONG_BINGET
          this.stack.push(this.memo.get(this.view.getUint32(this.pos, true)));
          this.pos += 4;
          break;
        case 0x67: // GET
          this.stack.push(this.memo.get(parseInt(this.readLine(), 10)));
          break;

        // ---- persistent ids
        case 0x51: // BINPERSID
          this.stack.push(this.persistentLoad(this.stack.pop()));
          break;
        case 0x50: // PERSID
          this.stack.push(this.persistentLoad(this.readLine()));
          break;

        default:
          throw new Error(
            `Unsupported pickle opcode 0x${op.toString(16)} at offset ${this.pos - 1}`,
          );
      }
    }
  }

  // Note: `this.stack.push(this.popMark())` would push onto the pre-MARK stack,
  // because the receiver is evaluated before popMark() swaps it.
  private push(v: unknown) {
    this.stack.push(v);
  }

  private top() {
    return this.stack[this.stack.length - 1];
  }

  private popMark(): unknown[] {
    const items = this.stack;
    const prev = this.metastack.pop();
    if (!prev) throw new Error('Pickle MARK stack underflow');
    this.stack = prev;
    return items;
  }

  private call(fn: unknown, args: unknown[]): unknown {
    if (isCallable(fn)) return fn.__call__(args);
    if (fn instanceof PyClass) {
      const o = new PyObject(fn);
      o.args = args;
      return o;
    }
    throw new Error(`Cannot call non-callable pickle value: ${String(fn)}`);
  }

  private appendTo(target: unknown, items: unknown[]) {
    if (Array.isArray(target)) target.push(...items);
    else if (target instanceof PyObject) (target.list ??= []).push(...items);
    else throw new Error('APPEND on non-list');
  }

  private setItems(target: unknown, items: unknown[]) {
    let m: Map<unknown, unknown>;
    if (target instanceof Map) m = target;
    else if (target instanceof PyObject) m = target.items ??= new Map();
    else throw new Error('SETITEM on non-dict');
    for (let i = 0; i < items.length; i += 2) m.set(dictKey(items[i]), items[i + 1]);
  }

  private readLine(): string {
    const start = this.pos;
    while (this.data[this.pos] !== 0x0a) this.pos++;
    const s = this.latin1.decode(this.data.subarray(start, this.pos));
    this.pos++;
    return s;
  }

  private readUtf8(n: number) {
    const s = this.decoder.decode(this.data.subarray(this.pos, this.pos + n));
    this.pos += n;
    return s;
  }

  private readLatin1(n: number) {
    const s = this.latin1.decode(this.data.subarray(this.pos, this.pos + n));
    this.pos += n;
    return s;
  }

  private readBytes(n: number) {
    const b = this.data.slice(this.pos, this.pos + n);
    this.pos += n;
    return b;
  }

  private readLong(n: number): number {
    if (n === 0) return 0;
    let v = 0n;
    for (let i = n - 1; i >= 0; i--) v = (v << 8n) | BigInt(this.data[this.pos + i]);
    if (this.data[this.pos + n - 1] & 0x80) v -= 1n << BigInt(n * 8);
    this.pos += n;
    return Number(v);
  }
}

// Tuples become arrays in JS, which Map compares by identity. Stringify them
// so keys like (0, 1) remain usable; everything else passes through.
function dictKey(k: unknown): unknown {
  return Array.isArray(k) ? `(${k.map(String).join(', ')})` : k;
}
