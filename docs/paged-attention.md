# PagedAttention (Experimental)

PagedAttention is an experimental memory management feature for the KV cache that reduces memory fragmentation and enables efficient context shifting for multi-sequence workloads.

## Quick Start

```bash
# Enable PagedAttention
./llama-cli -m model.gguf --kv-cache-paged -p "Hello" -n 100

# Disable (default)
./llama-cli -m model.gguf --no-kv-cache-paged -p "Hello" -n 100

# Works with the server too (transparent)
./llama-server -m model.gguf --kv-cache-paged --port 8080
```

## How It Works

Instead of allocating KV cache slots as a contiguous ring buffer, PagedAttention divides the cache into fixed-size **blocks** (default: 32 tokens). Sequences map logical token positions to physical blocks via a **block table**, similar to OS virtual memory paging.

**Key benefits:**
- **Zero fragmentation**: Blocks are allocated on demand, no wasted gaps
- **O(1) context shift**: Remove/remap blocks instead of moving data
- **Copy-on-Write (CoW)**: Shared sequences (beam search) share blocks until written to

## Block Size

The default block size is **32 tokens**, which aligns with:
- F16 KV cache
- Q8_0 quantized KV cache (quant block = 32)
- Q4_0 quantized KV cache (quant block = 32)

> **Note**: Q4_K quantization uses a quant block of 256, which may require a larger block size for optimal alignment. This is a future optimization.

## Architecture

```
CLI (--kv-cache-paged)
  → common_params.kv_cache_paged
  → llama_context_params.kv_cache_paged
  → llama_cparams.kv_cache_paged
  → llama_memory_params.kv_cache_paged
  → llama_kv_cache constructor (paged=true)
      → BlockAllocator (manages free block pool)
      → BlockTable (maps seq_id → block list)
```

## Files

| File | Purpose |
|------|---------|
| `src/llama-kv-cache-paged.h` | `llama_block_allocator` + `llama_block_table` structs |
| `src/llama-kv-cache-paged.cpp` | Block allocator/table implementation |
| `src/llama-kv-cache.h` | PA members in `llama_kv_cache` |
| `src/llama-kv-cache.cpp` | PA fast path in `find_slot()`, CoW in `seq_cp()`, block management in `seq_rm()`/`clear()`/`seq_keep()` |
| `tests/test-kv-cache-paged.cpp` | 17 unit tests |

## Performance

Tested with SmolLM2-135M-Instruct-Q8_0 (CPU):

| Mode | Prompt (t/s) | Generation (t/s) |
|------|-------------|-------------------|
| Baseline | 374.8 | 24.3 |
| PagedAttention | 512.5 (avg) | 27.6 (avg) |

**No throughput regression** — PA matches or exceeds baseline performance.

## Limitations

- Block size is fixed at 32 (not yet configurable)
- Q4_K quantization may not align optimally with block_size=32
- No custom paged flash attention kernel yet (future: `ggml_paged_flash_attn_ext`)
- Memory benchmark with multi-slot server not yet validated
