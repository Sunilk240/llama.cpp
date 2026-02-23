#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <cstddef>
#include <vector>
#include <utility>

struct llama_hparams;
struct llama_model;
struct llama_layer;

// Which tier a layer belongs to
enum llama_layer_tier {
    LLAMA_TIER_GPU  = 0,  // Permanently on GPU (fits in VRAM)
    LLAMA_TIER_CPU  = 1,  // On CPU, windowed into GPU staging as needed
    LLAMA_TIER_DISK = 2,  // On disk (Phase C)
};

// Transfer state for async operations
enum llama_layer_transfer_state {
    LLAMA_XFER_IDLE    = 0,  // Not in transfer
    LLAMA_XFER_LOADING = 1,  // Async transfer in progress
    LLAMA_XFER_READY   = 2,  // Transfer complete, data available at target
};

// Saved pointer for restoring original tensor state after swap
struct llama_tensor_saved_ptr {
    struct ggml_tensor *    tensor;       // the tensor node (never changes)
    void *                  orig_data;    // original ->data pointer
    ggml_backend_buffer_t   orig_buffer;  // original ->buffer pointer
};

// Per-layer metadata for the window manager
struct llama_layer_window_entry {
    int32_t il         = -1;                            // layer index
    llama_layer_tier tier = LLAMA_TIER_CPU;              // which tier this layer belongs to
    llama_layer_transfer_state xfer_state = LLAMA_XFER_IDLE; // current transfer status

    size_t weight_bytes = 0;                             // total bytes of all tensors in this layer

    // Staging buffer indices (into the double-buffer pool)
    int staging_slot = -1;                               // -1 if not staged, 0 or 1 for double-buffer

    // Saved original pointers for swap-back (populated by swap_layer_to_gpu)
    std::vector<llama_tensor_saved_ptr> saved_ptrs;
};

// Configuration for the layer window
struct llama_layer_window_params {
    int32_t n_window         = 0;     // -1 = auto, 0 = disabled, >0 = manual window size
    bool    prefetch_enabled = true;  // enable async prefetching (default: true)
};

// The Layer Window Manager
//
// Manages a sliding window of N layers on GPU staging buffers, swapping tensor
// data pointers between CPU memory and GPU staging without modifying graph topology.
// This preserves graph reuse: only ->data and ->buffer are swapped, not ggml_tensor nodes.
struct llama_layer_window {
    llama_layer_window_params params;

    int32_t n_layer      = 0;  // total layers in the model
    int32_t n_window     = 0;  // computed window size (after auto-detect)
    int32_t n_gpu_static = 0;  // layers that permanently fit in GPU
    bool    use_pinned   = false;  // whether pinned memory was successfully allocated

    // Per-layer tracking
    std::vector<llama_layer_window_entry> entries;

    // Double-buffered staging areas
    // staging[0] and staging[1] alternate: compute from one while loading into the other
    struct staging_buffer {
        void * host_ptr = nullptr;   // host memory (pinned via cudaMallocHost or fallback malloc)
        size_t size     = 0;         // bytes allocated
        bool   pinned   = false;     // true if allocated via cudaMallocHost
    } staging[2];
    int active_slot = 0;             // which staging slot is currently being computed from

    // GPU-side buffer that holds the actual data the compute kernels read from
    ggml_backend_buffer_t staging_gpu_buffer[2] = { nullptr, nullptr };

    // Initialization (sets up per-layer entries, does NOT allocate staging)
    void init(int32_t n_layer_total);

    // Compute per-layer weight sizes from the model's actual tensors
    void compute_layer_sizes(const struct llama_model & model);

    // Allocate staging buffers (call AFTER compute_layer_sizes)
    void allocate_staging_buffers(ggml_backend_t gpu_backend);

    // Auto-detect window size based on available memory
    // NOTE: kv_cache_size must be estimated BEFORE calling this
    int32_t auto_detect_window(
        size_t free_vram,
        size_t kv_cache_size,
        size_t activation_size
    );

    // Get the range of layers that should be in GPU for a given current layer
    // Returns [start_il, end_il) range
    std::pair<int32_t, int32_t> get_window_range(int32_t current_il) const;

    // Check if a layer is currently available on GPU
    bool is_on_gpu(int32_t il) const;

    // Swap tensor data pointers for a layer (CPU buffer <-> GPU staging)
    // CRITICAL: Preserves ggml_tensor nodes, only swaps ->data and ->buffer
    // This keeps graph topology unchanged for graph reuse
    void swap_layer_to_gpu(int32_t il, struct llama_layer & layer);
    void swap_layer_to_cpu(int32_t il, struct llama_layer & layer);

    // Cleanup (frees pinned memory and GPU staging buffers)
    void free();

    // Is windowing active?
    bool enabled() const { return n_window > 0 && n_window < n_layer; }
};
