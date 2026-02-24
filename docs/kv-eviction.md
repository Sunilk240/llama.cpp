# Smart KV Eviction (Experimental)

Smart KV eviction is an experimental feature that intelligently removes KV cache entries when the cache is full, enabling infinite-length text generation without the quality degradation of context shifting.

## Quick Start

```bash
# StreamingLLM mode: keep sink tokens + recent window, evict oldest middle tokens
./llama-cli -m model.gguf --kv-eviction 1 --ctx-size 2048 -p "Hello" -n 5000

# Scored mode: evict least-recently-accessed tokens first (better quality)
./llama-cli -m model.gguf --kv-eviction 2 --ctx-size 2048 -p "Hello" -n 5000

# Custom sink + protected tokens (e.g. protect a 128-token system prompt)
./llama-cli -m model.gguf --kv-eviction 1 --kv-sink-tokens 8 --kv-protected-tokens 128

# Works with the server too
./llama-server -m model.gguf --kv-eviction 1 --port 8080

# Disabled by default (falls back to context shift)
./llama-cli -m model.gguf --kv-eviction 0 -p "Hello" -n 100
```

## How It Works

### The Problem

When the KV cache fills up during long generation, llama.cpp uses **context shifting** — discarding the oldest half of the cache and shifting positions. This causes quality degradation because the model loses important middle context.

### The Solution

Smart KV eviction selectively removes individual cache entries based on their importance, preserving:

1. **Sink tokens**: First N positions (attention sinks essential for stability)
2. **Protected tokens**: System prompt or other critical prefix tokens
3. **Recent window**: Last 25% of each sequence (actively being generated)

Only the "middle" tokens — least important for ongoing generation — are evicted.

### Two Modes

| Mode | Flag | Strategy | Best For |
|------|------|----------|----------|
| **Streaming** | `--kv-eviction 1` | Evict oldest middle tokens (position-based) | Simple, fast, reliable |
| **Scored** | `--kv-eviction 2` | Evict least-recently-accessed tokens | Better quality for long conversations |

### Eviction Trigger

Eviction happens automatically inside `init_batch()` when `prepare()` fails to find free slots:

```
prepare() fails → evict_cells(32+) → retry prepare() → success
```

If eviction frees enough cells, generation continues seamlessly. If not (all cells protected), it falls back to the standard failure path.

## CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--kv-eviction MODE` | `0` (disabled) | `0`=none, `1`=streaming, `2`=scored |
| `--kv-sink-tokens N` | `4` | Initial positions to always keep (0-256) |
| `--kv-protected-tokens N` | `0` | Positions to protect from eviction (e.g. system prompt length) |

## Architecture

```
CLI (--kv-eviction MODE)
  → common_params.kv_eviction_mode / kv_sink_tokens / kv_protected_tokens
  → llama_context_params.kv_eviction_mode / kv_sink_tokens / kv_protected_tokens
  → llama_context constructor:
      → dynamic_cast<llama_kv_cache*> → set_eviction_policy()
      → dynamic_cast<llama_kv_cache_iswa*> → get_base() → set_eviction_policy()
  → Runtime:
      init_batch() → prepare() fails → evict_cells() → retry prepare()
      apply_ubatch() → update access scores (scored mode only)
```

## Safety Features

- **Shared cells preserved**: Cells used by multiple sequences (`seq_cp`) are never evicted
- **Per-sequence recent window**: Protects `max(32, seq_len/4)` most recent positions per sequence
- **Input validation**: Range checks on all parameters (sink 0-256, protected ≥ 0)
- **ISWA compatibility**: Only the base cache is evicted; SWA cache uses native sliding window
- **Zero impact when disabled**: Default `--kv-eviction 0` touches no code paths at runtime

## Files

| File | Purpose |
|------|---------|
| `src/llama-kv-cache.h` | Eviction enum, state vars, `evict_cells()` + `set_eviction_policy()` declarations |
| `src/llama-kv-cache.cpp` | Core eviction logic, scoring in `apply_ubatch()`, `init_batch()` hook |
| `include/llama.h` | 3 params in `llama_context_params` |
| `common/common.h` | 3 params in `common_params` |
| `common/common.cpp` | Param wiring |
| `common/arg.cpp` | CLI flag parsing with validation |
| `src/llama-context.cpp` | Constructor wiring (single + ISWA dual-cache) |

## Background

Based on two research papers:

- **StreamingLLM** (Xiao et al., 2023): Discovered "attention sinks" — first few tokens receive disproportionate attention. Keeping sinks + recent window preserves generation quality.
- **H2O** (Zhang et al., 2023): Heavy-Hitter Oracle — tracks cumulative attention scores to identify important tokens. Our "scored" mode is a lightweight approximation using access timestamps.

## Limitations

- No per-layer scoring (all layers share the same eviction decision)
- Scored mode uses access recency, not true attention weights (lightweight approximation)
- Not yet tested with speculative decoding
- Eviction granularity is per-cell (not per-block for PagedAttention mode)
