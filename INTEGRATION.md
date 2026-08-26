# Integrating dedup

`dedup` ships as a Rust crate with two consumption modes:

- **Rust hosts** — depend on the crate directly (`rlib`), call the batch and streaming APIs natively.
- **JavaScript hosts** — build the `wasm32-unknown-unknown` target and talk to it through a thin FFI bridge (~470 lines, no runtime dependency on generated wasm-bindgen glue).

This document covers the reference integration in the [r](https://git.sly.so/kade/r) coding-agent monorepo and the steps to embed dedup in other harnesses.

## Reference integration: r

The r monorepo integrates dedup in three tiers:

```
~/src/dedup                        this crate (Rust + WASM)
  └─ packages/dedup-wasm           JS bridge package (@entropy-tamer/r-dedup-wasm)
       └─ packages/coding-agent    harness wiring (three modules)
```

### Tier 1: bridge package — `packages/dedup-wasm`

A standalone npm package that owns the WASM binary and FFI. Key properties:

- **Manual marshaling** — strings cross the boundary as pointer/length pairs via `__wbindgen_malloc` / `__wbindgen_realloc` / `__wbindgen_free`, implemented directly in [`src/dedup-wasm.ts`](https://git.sly.so/kade/r/src/branch/main/packages/dedup-wasm/src/dedup-wasm.ts). No generated `_bg.js` glue at runtime.
- **Lazy init with caching** — `initDedupWasm()` loads and instantiates the module once; every exported function awaits a shared init promise.
- **Exports**: batch functions (`deduplicateText`, `getRepetitionRatio`, `jaccardSimilarity`, `tokenize`, `resolvePreset`, `checkRepetition`) plus a `StreamingDedupMonitor` class wrapping the Rust streaming state.
- **Build pipeline** — `pnpm run build:wasm` compiles this crate with wasm-bindgen (`--target no-modules`) and copies only `dedup_wasm_bg.wasm` into the package's `wasm/` directory.

### Tier 2: in-turn deduplication — `coding-agent/src/core/deduplication.ts`

Batch API wrapper preserving a stable TS surface over the WASM calls. It eager-initializes WASM at module load so the synchronous-looking API is ready when real work arrives, and falls back gracefully (logged warning, retry on next call) if initialization fails.

### Tier 3: streaming monitor — `coding-agent/src/core/stream-deduplication.ts`

`StreamDeduplicationMonitor` wraps the WASM `StreamingDedupMonitor` and adds a callback layer. The host agent session:

1. Creates a monitor per assistant turn (`createStreamDeduplicationMonitor({...})`)
2. Feeds every `text_delta` into `feedChunk(chunk)` — results are immediate; no polling
3. Acts on the recommended action: `none` continues, `steer` injects a corrective message, `abort` terminates generation
4. Calls `reset()` between turns

All gating (warmup, content length, chunk delta) lives inside the Rust state machine; the TS side holds no timers.

### Cross-turn loop detection — `coding-agent/src/core/loop-detection.ts`

Uses `tokenize` and `jaccardSimilarity` from the same WASM module to compare turn signatures across turns: tool-call serializations plus hashed text content. Consecutive similar turns escalate through corrective-message tiers before aborting.

## Embedding in other harnesses

### Rust harnesses (e.g. Codex / `codex-rs`)

Codex is a Rust workspace, so no WASM layer is involved — use the crate natively.

1. **Add the dependency** to the relevant workspace member's `Cargo.toml`:
   ```toml
   [dependencies]
   dedup-wasm = { git = "https://git.sly.so/kade/dedup" }
   ```
   Or vendor the crate and add it to the workspace.
2. **Wire the batch API** wherever completed assistant messages are post-processed:
   ```rust
   let settings = resolve_dedup_preset(DedupPresetName::Default);
   let check = check_repetition(&message_text, &settings);
   ```
3. **Wire the streaming monitor** into the response-stream handler. Where each text chunk arrives:
   ```rust
   let mut monitor = StreamingState::new(settings);
   // per text chunk:
   let check = monitor.feed_chunk(&chunk_text);
   // on stream end or repeated==true:
   match monitor.determine_action(&check) {
       StreamAction::None => {}
       StreamAction::Steer => { /* append corrective message to conversation */ }
       StreamAction::Abort => { /* cancel generation */ }
   }
   ```
   Create one `StreamingState` per generation turn; drop or reset between turns.
4. **Map actions to harness semantics.** Codex-style harnesses typically have a turn-interrupt mechanism — map `Abort` to it, and implement `Steer` as enqueueing a system/user corrective message before the next model call.
5. **Configuration surface.** Expose `DedupSettings` (or just preset names) through whatever settings format the harness uses; see the preset table in [README.md](./README.md) for defaults.

### JavaScript/Node harnesses (e.g. Claude Code)

Claude Code is a Node application, so use the same pattern as the r bridge package:

1. **Build the WASM artifact**:
   ```bash
   cargo build --manifest-path ~/src/dedup/Cargo.toml --target wasm32-unknown-unknown --release
   wasm-bindgen --out-dir pkg --target no-modules \
     ~/src/dedup/target/wasm32-unknown-unknown/release/dedup_wasm.wasm
   ```
   Ship `dedup_wasm_bg.wasm` alongside your app.
2. **Copy or adapt the bridge.** The smallest path is to vendor `packages/dedup-wasm/src/dedup-wasm.ts` + `types.ts` from the r monorepo — it implements all required imports (`__wbindgen_throw`, `getRandomValues`, externref table setup, `__wbindgen_start`) with no external dependencies beyond Node built-ins. Requirements it encodes:
   - pointer/length string marshaling via malloc/realloc/free
   - exact import names including the `__wbg_` prefix
   - `crypto.getRandomValues` backed by `node:crypto.webcrypto`
   - `__wbindgen_start()` after instantiation
   See [ARCHITECTURE.md § WASM Portability Notes](./ARCHITECTURE.md#wasm-portability-notes) for why each piece exists.
3. **Init once, then use anywhere**:
   ```ts
   await initDedupWasm();
   const check = JSON.parse(checkRepetition(text, JSON.stringify(settings)));
   ```
4. **Hook the streaming path.** In the message-stream handler, feed each content-block delta into `StreamingDedupMonitor.feedChunk(chunk)` and act on `recommendedAction`. Keep the action policy: steer on sentence-level matches, abort only on accumulated near-exact repetition.

### Minimal checklist for any host

- [ ] One `StreamingState` / `StreamingDedupMonitor` per assistant turn, reset between turns
- [ ] Feed raw text deltas (not line-buffered) for lowest detection latency
- [ ] Treat `steer` as non-fatal; reserve `abort` for confirmed loops
- [ ] Surface presets (`default` / `relaxed` / `sensitive`) rather than raw thresholds where possible
- [ ] Degrade gracefully: if WASM/native init fails, disable monitoring rather than failing the session

## Versioning

The crate and the bridge package evolve together but version independently: the crate is at `0.1.0` in `Cargo.toml`; the npm bridge package (`@entropy-tamer/r-dedup-wasm`) carries its own version in the r monorepo. When changing the export surface in `lib.rs`, update the bridge's FFI and types in the same change set.
