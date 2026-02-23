#include "llama-layer-window.h"
#include "llama-model.h"
#include "llama-impl.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>

#ifdef _WIN32
#include <io.h>      // _fseeki64
#else
#include <unistd.h>  // pread
#endif

//
// Helper: iterate all non-null ggml_tensor* fields in a llama_layer.
//
// All ggml_tensor* fields from attn_norm through indexer_attn_q_b are laid out
// contiguously in llama_layer (same pointer type, no intervening non-pointer fields).
// See llama-model.h for the struct definition.
//
// Sub-structs (posnet, convnext, shortconv, nextn) at the end are for TTS/exotic
// architectures and are NOT iterated here — they can be added if needed.
//
static_assert(
    (offsetof(llama_layer, indexer_attn_q_b) - offsetof(llama_layer, attn_norm))
        % sizeof(ggml_tensor *) == 0,
    "llama_layer tensor fields are not contiguously packed"
);

static constexpr size_t LAYER_TENSOR_FIELD_COUNT =
    (offsetof(llama_layer, indexer_attn_q_b) - offsetof(llama_layer, attn_norm))
        / sizeof(ggml_tensor *) + 1;

template<typename Fn>
static void for_each_layer_tensor(llama_layer & layer, Fn && fn) {
    ggml_tensor ** first = &layer.attn_norm;
    for (size_t i = 0; i < LAYER_TENSOR_FIELD_COUNT; ++i) {
        if (first[i]) {
            fn(first[i]);
        }
    }
}

template<typename Fn>
static void for_each_layer_tensor_const(const llama_layer & layer, Fn && fn) {
    ggml_tensor * const * first = &layer.attn_norm;
    for (size_t i = 0; i < LAYER_TENSOR_FIELD_COUNT; ++i) {
        if (first[i]) {
            fn(first[i]);
        }
    }
}

// ---- llama_layer_window implementation ----

void llama_layer_window::init(int32_t n_layer_total) {
    n_layer = n_layer_total;
    entries.resize(n_layer);
    for (int32_t i = 0; i < n_layer; i++) {
        entries[i].il         = i;
        entries[i].tier       = LLAMA_TIER_CPU;  // default: all CPU-tier
        entries[i].xfer_state = LLAMA_XFER_IDLE;
        entries[i].staging_slot = -1;
    }
}

void llama_layer_window::compute_layer_sizes(const struct llama_model & model) {
    for (int32_t il = 0; il < n_layer && il < (int32_t)model.layers.size(); il++) {
        size_t total = 0;
        for_each_layer_tensor_const(model.layers[il], [&](const ggml_tensor * t) {
            total += ggml_nbytes(t);
        });
        entries[il].weight_bytes = total;
    }
}

void llama_layer_window::allocate_staging_buffers(ggml_backend_t gpu_backend) {
    // Find the largest CPU-tier layer
    size_t max_layer_size = 0;
    for (const auto & e : entries) {
        if (e.tier == LLAMA_TIER_CPU) {
            max_layer_size = std::max(max_layer_size, e.weight_bytes);
        }
    }

    if (max_layer_size == 0) {
        LLAMA_LOG_WARN("%s: no CPU-tier layers, skipping staging allocation\n", __func__);
        return;
    }

    for (int i = 0; i < 2; i++) {
        // Host staging: regular malloc
        // TODO: use ggml_backend_dev_host_buffer_type() for pinned memory on CUDA
        staging[i].host_ptr = malloc(max_layer_size);
        GGML_ASSERT(staging[i].host_ptr && "failed to allocate host staging buffer");
        staging[i].size   = max_layer_size;
        staging[i].pinned = false;

        // GPU-side staging buffer
        staging_gpu_buffer[i] = ggml_backend_alloc_buffer(gpu_backend, max_layer_size);
        GGML_ASSERT(staging_gpu_buffer[i] && "failed to allocate GPU staging buffer");
    }

    use_pinned = staging[0].pinned;

    LLAMA_LOG_INFO("%s: staging buffers: 2 x %.1f MiB host (%s) + 2 x %.1f MiB device\n",
        __func__,
        max_layer_size / (1024.0 * 1024.0),
        use_pinned ? "pinned" : "unpinned",
        max_layer_size / (1024.0 * 1024.0));
}

void llama_layer_window::free() {
    for (int i = 0; i < 2; i++) {
        if (staging_gpu_buffer[i]) {
            ggml_backend_buffer_free(staging_gpu_buffer[i]);
            staging_gpu_buffer[i] = nullptr;
        }
        if (staging[i].host_ptr) {
            ::free(staging[i].host_ptr);
            staging[i].host_ptr = nullptr;
            staging[i].size   = 0;
            staging[i].pinned = false;
        }
    }
    // Phase C: clean up disk cache
    disk.free_cache();
    entries.clear();
    n_layer  = 0;
    n_window = 0;
}

int32_t llama_layer_window::auto_detect_window(
        size_t free_vram, size_t kv_cache_size, size_t activation_size) {
    // Reserve VRAM for KV cache, activations, and 256 MiB safety margin
    const size_t safety_margin = 256ULL << 20;
    const size_t reserved = kv_cache_size + activation_size + safety_margin;

    if (free_vram <= reserved) {
        LLAMA_LOG_WARN("%s: free VRAM (%.0f MiB) <= reserved (%.0f MiB), disabling window\n",
            __func__, free_vram / (1024.0 * 1024.0), reserved / (1024.0 * 1024.0));
        n_window = 0;
        return 0;
    }

    const size_t available = free_vram - reserved;

    // Find max layer size among CPU-tier layers
    size_t max_layer = 0;
    int32_t n_cpu = 0;
    for (const auto & e : entries) {
        if (e.tier == LLAMA_TIER_CPU) {
            max_layer = std::max(max_layer, e.weight_bytes);
            n_cpu++;
        }
    }

    if (max_layer == 0 || n_cpu == 0) {
        n_window = 0;
        return 0;
    }

    // Double-buffered: need 2 × max_layer per window slot
    n_window = std::max<int32_t>(1, (int32_t)(available / (2 * max_layer)));
    n_window = std::min(n_window, n_cpu);

    LLAMA_LOG_INFO("%s: auto-detected window size: %d layers "
        "(%.0f MiB avail, %.1f MiB/layer, %d CPU-tier layers)\n",
        __func__, n_window,
        available / (1024.0 * 1024.0),
        max_layer / (1024.0 * 1024.0),
        n_cpu);

    return n_window;
}

std::pair<int32_t, int32_t> llama_layer_window::get_window_range(int32_t current_il) const {
    if (!enabled()) {
        return { 0, n_layer };
    }

    // Center window on current layer, clamp to [0, n_layer)
    int32_t half  = n_window / 2;
    int32_t start = current_il - half;
    int32_t end   = start + n_window;

    if (start < 0) {
        start = 0;
        end   = std::min(n_window, n_layer);
    }
    if (end > n_layer) {
        end   = n_layer;
        start = std::max(0, end - n_window);
    }

    return { start, end };
}

bool llama_layer_window::is_on_gpu(int32_t il) const {
    if (il < 0 || il >= n_layer) return false;
    const auto & e = entries[il];
    return e.tier == LLAMA_TIER_GPU || e.staging_slot >= 0;
}

void llama_layer_window::swap_layer_to_gpu(int32_t il, struct llama_layer & layer) {
    auto & entry = entries[il];

    // Already on GPU permanently — nothing to do
    if (entry.tier == LLAMA_TIER_GPU) return;

    // Already swapped into staging
    if (entry.staging_slot >= 0) return;

    int slot = active_slot;
    entry.staging_slot = slot;
    entry.saved_ptrs.clear();

    // CRITICAL: We only swap ->data and ->buffer pointers.
    // The ggml_tensor nodes themselves stay unchanged in the graph,
    // preserving graph topology for graph reuse (Warning #1 from research).
    void * base = ggml_backend_buffer_get_base(staging_gpu_buffer[slot]);
    ggml_backend_buffer_t buf = staging_gpu_buffer[slot];
    size_t offset = 0;

    for_each_layer_tensor(layer, [&](ggml_tensor * t) {
        // Save original pointers for restore in swap_layer_to_cpu
        entry.saved_ptrs.push_back({ t, t->data, t->buffer });
        // Redirect tensor to GPU staging
        t->data   = (char *)base + offset;
        t->buffer = buf;
        offset   += ggml_nbytes(t);
    });
}

void llama_layer_window::swap_layer_to_cpu(int32_t il, struct llama_layer & layer) {
    auto & entry = entries[il];

    if (entry.tier == LLAMA_TIER_GPU) return;
    if (entry.staging_slot < 0)      return;  // not swapped

    // Restore all tensor data/buffer pointers to their original values
    for (auto & sp : entry.saved_ptrs) {
        sp.tensor->data   = sp.orig_data;
        sp.tensor->buffer = sp.orig_buffer;
    }

    entry.saved_ptrs.clear();
    entry.staging_slot = -1;

    GGML_UNUSED(layer);
}

// ---- Phase C: Disk I/O implementation ----

void llama_layer_window::disk_io::init(int32_t n_layer) {
    layer_offsets.resize(n_layer);
    access_counter = 0;
}

void llama_layer_window::disk_io::load_layer_from_disk(int32_t il, void * dst) {
    if (!model_file || il < 0 || il >= (int32_t)layer_offsets.size()) {
        LLAMA_LOG_ERROR("%s: invalid layer %d or no file handle\n", __func__, il);
        return;
    }

    const auto & offsets = layer_offsets[il];
    size_t write_offset = 0;

    for (const auto & [file_off, size] : offsets.tensor_offsets) {
#ifdef _WIN32
        _fseeki64(model_file, (__int64)file_off, SEEK_SET);
        size_t read = fread((char *)dst + write_offset, 1, size, model_file);
        if (read != size) {
            LLAMA_LOG_ERROR("%s: short read for layer %d: expected %zu, got %zu\n",
                __func__, il, size, read);
        }
#else
        ssize_t read = pread(fileno(model_file), (char *)dst + write_offset, size, file_off);
        if (read < 0 || (size_t)read != size) {
            LLAMA_LOG_ERROR("%s: pread failed for layer %d: expected %zu, got %zd\n",
                __func__, il, size, read);
        }
#endif
        write_offset += size;
    }
}

void llama_layer_window::disk_io::evict_lru() {
    if (cpu_cache.empty()) return;

    // Sort by last_access ascending (oldest first)
    std::sort(cpu_cache.begin(), cpu_cache.end(),
        [](const cpu_cache_entry & a, const cpu_cache_entry & b) {
            return a.last_access < b.last_access;
        });

    // Compute total cache usage
    size_t total = 0;
    for (const auto & e : cpu_cache) total += e.size;

    // Evict oldest entries until under budget
    while (total > cpu_cache_budget && !cpu_cache.empty()) {
        auto & oldest = cpu_cache.front();
        total -= oldest.size;
        ::free(oldest.data);
        LLAMA_LOG_DEBUG("%s: evicted layer %d (%.1f MiB), total now %.1f MiB\n",
            __func__, oldest.il,
            oldest.size / (1024.0 * 1024.0),
            total / (1024.0 * 1024.0));
        cpu_cache.erase(cpu_cache.begin());
    }
}

void llama_layer_window::disk_io::free_cache() {
    for (auto & e : cpu_cache) {
        if (e.data) {
            ::free(e.data);
            e.data = nullptr;
        }
    }
    cpu_cache.clear();
    layer_offsets.clear();

    if (model_file) {
        fclose(model_file);
        model_file = nullptr;
    }

    stop = true;
    if (io_thread.joinable()) {
        io_thread.join();
    }
}

// ---- Phase C: 3-tier auto-detection ----

static constexpr size_t TIER_SAFETY_MARGIN = 256ULL << 20;  // 256 MiB

void llama_layer_window::auto_detect_tiers(
        const std::vector<ggml_backend_dev_t> & devices,
        size_t cpu_available) {
    // Query GPU free memory
    size_t gpu_free = 0;
    for (auto * dev : devices) {
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU ||
            ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_IGPU) {
            size_t free_dev = 0, total_dev = 0;
            ggml_backend_dev_memory(dev, &free_dev, &total_dev);
            gpu_free += free_dev;
        }
    }

    size_t gpu_budget = (gpu_free > TIER_SAFETY_MARGIN) ? gpu_free - TIER_SAFETY_MARGIN : 0;
    size_t cpu_budget = (cpu_available > TIER_SAFETY_MARGIN) ? cpu_available - TIER_SAFETY_MARGIN : 0;

    n_gpu_static = 0;
    int32_t n_cpu  = 0;
    int32_t n_disk = 0;

    // Assign layers from the end (output layers benefit most from GPU)
    for (int il = n_layer - 1; il >= 0; il--) {
        if (entries[il].weight_bytes <= gpu_budget) {
            entries[il].tier = LLAMA_TIER_GPU;
            gpu_budget -= entries[il].weight_bytes;
            n_gpu_static++;
        } else if (entries[il].weight_bytes <= cpu_budget) {
            entries[il].tier = LLAMA_TIER_CPU;
            cpu_budget -= entries[il].weight_bytes;
            n_cpu++;
        } else {
            entries[il].tier = LLAMA_TIER_DISK;
            n_disk++;
        }
    }

    LLAMA_LOG_INFO("%s: tier assignment: %d GPU, %d CPU, %d Disk\n",
        __func__, n_gpu_static, n_cpu, n_disk);
}
