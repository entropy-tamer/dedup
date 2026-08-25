# dedup — Text Deduplication in WASM

High-performance text deduplication compiled to WebAssembly. Replaces O(n²) n-gram comparison in TypeScript with constant-time Rust, eliminating main-thread hangs during assistant generation on laptop-class machines.

## Features

- **In-turn dedup**: Detects near-exact repetition within a single message using Rabin-Karp rolling hash + Jaccard similarity
- **Streaming orchestration**: Real-time detection during generation via `feed_chunk()` — no polling, no setInterval
- **Three presets**: Default, relaxed (fewer false positives), sensitive (catches more repetition)

## Architecture

```mermaid
graph TD
    A[Assistant output] --> B[StreamingDedup Monitor]
    B --> C{In-turn check}
    C -- near-exact match --> D[N-gram confirmation]
    C -- semantic similarity --> E[Sentence corroboration]
    D & E --> F{Action determination<br/>none / steer / abort?}

```

See [ARCHITECTURE.md](./ARCHITECTURE.md) for full details on each layer, data flow diagrams, and module dependency graphs.

## Build

```bash
# Compile to native (for tests on host OS)
cargo test --release

# Compile to WebAssembly target
rustup target add wasm32-unknown-unknown
cargo build --target wasm32-unknown-unknown --release

# Generate WASM glue for Node.js interop
wasm-bindgen target/wasm32-unknown-unknown/release/dedup_wasm.wasm \
  --out-dir pkg --target nodejs
```

## API (Rust)

### Batch API — check, deduplicate, and compute repetition ratio

```rust
// Check for repetition in text (two-pass: n-gram + sentence)
let settings = DedupSettings::default();
let result = check_repetition(&text, &settings);
// → { repeated: bool, sections: Vec<RepeatedSection> }

// Remove repeated sections from text
let deduped = deduplicate_text(&text, &settings);

// Get the ratio of repetitive content (0.0 to 1.0)
let ratio = get_repetition_ratio(&text, &settings);
```

### Streaming API — incremental detection during generation

```rust
use std::io;

let settings = DedupSettings::default();
let mut monitor = StreamingState::new(settings);

for chunk in assistant_stream {
    let check = monitor.feed_chunk(chunk);
    if check.repeated {
        // n-gram sections contribute toward abort threshold,
        // sentence-level matches only steer (never abort).
    }
}

// After the stream ends, determine final action:
let action = monitor.determine_action(&last_check);
match action {
    StreamAction::Abort => io::eprintln!("Stream aborted — model in degenerate loop"),
    StreamAction::Steer  => /* inject corrective message into prompt */,
    StreamAction::None   => {}, // continue streaming normally
}
```

### Presets

```rust
use crate::presets::{DedupPresetName, resolve_dedup_preset};

let settings = resolve_dedup_preset(DedupPresetName::Relaxed);
// → DedupSettings with relaxed thresholds:
//    ngram_similarity_threshold=0.92, step_size=15, min_repeat_length=80
```

## Performance characteristics

| Operation | Time Complexity | Typical latency (10KB text) |
|-----------|----------------|----------------------------|
| N-gram comparison | O(n² / step_size) per pass | < 5ms (native), ~1ms (WASM) |
| Streaming feed_chunk() | Incremental — only new content re-analyzed | ~2ms/chunk at streaming rate |

The WASM version runs on the main thread without blocking because:
- The warmup gate blocks checks for the first 3 seconds of content (configurable)
- Content-length gating skips analysis until sufficient text is accumulated (~400 chars)
- Chunk-delta gating avoids re-analysis when no new characters arrived since last check

## Test coverage

```bash
# Run all Rust tests (in-turn + streaming + WASM bindings)
cargo test --release

# Expected output: 64 passing tests across 3 test targets
# lib unit tests   — in-turn check, deduplication, presets, wasm bindings
# dedup_tests      — integration tests (batch, streaming, scenario regressions)
```

## License

MIT. See [LICENSE](./LICENSE).
