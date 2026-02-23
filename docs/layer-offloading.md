# Three-Tier Prefetched Layer Offloading (Experimental)

Layer offloading is an experimental feature that enables running models larger than GPU VRAM by dynamically swapping layer weights between Disk, CPU, and GPU using a sliding window with async prefetching.

## Quick Start

```bash
# Auto-detect window size based on free VRAM
./llama-cli -m model.gguf -ngl 5 --layer-window auto -p "Hello" -n 100

# Manual window size (4 CPU layers windowed at a time)
./llama-cli -m model.gguf -ngl 5 --layer-window 4 -p "Hello" -n 100

# Disable async prefetching (sync transfer only)
./llama-cli -m model.gguf -ngl 5 --layer-window auto --no-layer-prefetch -p "Hello" -n 100

# Works with the server too
./llama-server -m model.gguf -ngl 5 --layer-window auto --port 8080
```

## How It Works

### The Problem

When `-ngl` is less than the total layer count, remaining layers compute on CPU, which is significantly slower. Users with limited VRAM are stuck with slow CPU inference for those layers.

### The Solution

Layer offloading keeps only a **window** of N layers in GPU staging buffers at a time. Before each computation:

1. **Swap in**: CPU-tier layer weights are temporarily redirected to GPU staging via pointer swaps
2. **Transfer**: Weight data is copied from CPU → GPU staging buffer
3. **Compute**: GPU computes using the staged weights (fast!)
4. **Swap back**: Pointers are restored to original CPU locations

### Three Tiers

| Tier | Location | Behavior |
|------|----------|----------|
| **GPU** | VRAM (permanent) | Controlled by `-ngl`, always on GPU |
| **CPU** | System RAM | Windowed into GPU staging as needed |
| **Disk** | GGUF file | Loaded into CPU cache on demand (LRU eviction) |

### Interaction with `-ngl`

```
Model: 48 layers total, -ngl 5

Layer 0-42:  CPU tier  ─── windowed through GPU staging ───→ GPU compute
Layer 43-47: GPU tier  ─── permanently on GPU ──────────────→ GPU compute
```

- `-ngl` controls how many layers are **permanently** on GPU
- `--layer-window` controls how CPU-tier layers are **temporarily** windowed into GPU
- If `-ngl` covers all layers, `--layer-window` has no effect

## Key Design Decisions

### Graph Reuse Preservation

Only `tensor->data` and `tensor->buffer` pointers are swapped — the `ggml_tensor` graph nodes never change. This preserves graph topology for llama.cpp's graph reuse optimization.

### Double-Buffered Staging

Two staging buffer slots allow computing from one while loading data into the other, enabling overlap of data transfer and computation.

### Automatic Tensor Handling

A `for_each_layer_tensor()` helper uses `static_assert`-validated contiguous pointer iteration over all ~178 tensor fields in `llama_layer`, automatically adapting to upstream struct changes.

## CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--layer-window N` | `0` (disabled) | `auto` = detect from free VRAM, or exact number |
| `--no-layer-prefetch` | enabled | Disable async prefetching of next window |

Environment variables: `LLAMA_ARG_LAYER_WINDOW`

## Architecture

```
CLI (--layer-window N)
  → common_params.layer_window
  → llama_model_params.layer_window / .layer_prefetch
  → load_tensors(): init window, assign tiers, allocate staging
  → process_ubatch(): swap_layer_to_gpu → graph_compute → swap_layer_to_cpu

Phase A (scheduler-level):
  ggml_backend_sched_compute_splits()
    → prefetch next split's inputs during current split compute

Phase B (layer-level):
  llama_layer_window
    → sliding window of N layers on double-buffered GPU staging
    → pointer swap preserves graph topology

Phase C (disk tier):
  llama_layer_window::disk_io
    → read layers from GGUF via _fseeki64/pread
    → LRU CPU cache with eviction
```

## Files

| File | Purpose |
|------|---------|
| `src/llama-layer-window.h` | Enums, structs, window manager API |
| `src/llama-layer-window.cpp` | Core implementation: init, swap, auto-detect, disk I/O, LRU |
| `ggml/src/ggml-backend.cpp` | Phase A: scheduler-level async prefetch |
| `src/llama-model.cpp` | Window init in `load_tensors()`, destructor cleanup, tensor offset recording |
| `src/llama-context.cpp` | Swap hooks in `process_ubatch()` |
| `include/llama.h` | Public API params |
| `common/common.h` | Common params |
| `common/arg.cpp` | CLI flag parsing |

## Limitations

- Pinned memory (`cudaMallocHost`) not yet used (fallback to `malloc`)
- Disk tier I/O thread is not yet producer-consumer queued (synchronous reads)
- Window centering uses fixed mid-layer heuristic (TODO: graph-aware layer detection)
- Not yet tested with speculative decoding or multi-GPU split modes
