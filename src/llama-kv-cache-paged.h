#pragma once

#include "llama.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <unordered_map>
#include <vector>

// PagedAttention Block Allocator
//
// Manages a pool of fixed-size physical blocks for KV cache storage.
// Inspired by vLLM's BlockPool (vllm/v1/core/block_pool.py).
//
// Each block holds `block_size` KV cache cells (tokens).
// Blocks are allocated from a free list and tracked via reference counting
// to support Copy-on-Write (CoW) sharing between sequences.

struct llama_block_allocator {
    uint32_t block_size;   // tokens per block (e.g. 32)
    uint32_t num_blocks;   // total physical blocks available

    std::vector<uint32_t> free_list;   // stack of free block IDs
    std::vector<uint32_t> ref_count;   // reference count per block

    // Initialize with total cell count and block size.
    // num_blocks = total_cells / block_size
    llama_block_allocator(uint32_t total_cells, uint32_t blk_size);

    // Allocate one block from the free list. Returns block ID.
    // Increments ref_count to 1.
    // Asserts: free_list is not empty.
    uint32_t allocate();

    // Decrement ref_count for a block. If it reaches 0, return to free list.
    // This enables CoW: shared blocks are only truly freed when all refs are gone.
    void free_block(uint32_t block_id);

    // Increment ref_count for a block (used for CoW sharing).
    void inc_ref(uint32_t block_id);

    // Check if n blocks can be allocated.
    bool can_allocate(uint32_t n_blocks) const;

    // Number of currently free blocks.
    uint32_t num_free() const;

    // Total number of blocks.
    uint32_t total() const;
};

// PagedAttention Block Table
//
// Maps logical token positions to physical block locations for each sequence.
// Inspired by vLLM's BlockTable (vllm/v1/worker/block_table.py).
//
// Translation formula (matches vLLM's compute_slot_mapping):
//   physical_cell = tables[seq][pos / block_size] * block_size + (pos % block_size)
//
// This is the core of PagedAttention: instead of requiring contiguous KV cache
// cells for a sequence, we map logical positions to scattered physical blocks.

struct llama_block_table {
    uint32_t block_size = 0;

    // seq_id → list of physical block IDs (in logical order)
    std::unordered_map<llama_seq_id, std::vector<uint32_t>> tables;

    // Convert a logical position for a sequence to a physical cell index.
    // Formula: tables[seq][pos / block_size] * block_size + (pos % block_size)
    uint32_t logical_to_physical(llama_seq_id seq, llama_pos pos) const;

    // Append a new block to a sequence's block list.
    void append_block(llama_seq_id seq, uint32_t block_id);

    // Check if the sequence needs a new block to store `new_total_tokens` tokens.
    // Returns true when current capacity < new_total_tokens.
    bool needs_new_block(llama_seq_id seq, uint32_t new_total_tokens) const;

    // Current capacity of a sequence in tokens (num_blocks * block_size).
    uint32_t capacity(llama_seq_id seq) const;

    // Number of blocks allocated to a sequence.
    uint32_t num_blocks_for(llama_seq_id seq) const;

    // Check if a sequence exists in the table.
    bool has_seq(llama_seq_id seq) const;

    // Get the physical block ID for a given logical position.
    // Returns the block ID at tables[seq][pos / block_size].
    uint32_t get_block_id(llama_seq_id seq, llama_pos pos) const;

    // Replace the block at a given logical index with a new block ID.
    // Used for CoW: when writing to a shared block, allocate new block, copy data, replace entry.
    void replace_block(llama_seq_id seq, uint32_t logical_block_idx, uint32_t new_block_id);

    // Copy-on-Write: share all blocks from src to dst.
    // Increments ref_count for all shared blocks via the allocator.
    void share(llama_seq_id src, llama_seq_id dst, llama_block_allocator & alloc);

    // Free all blocks for a sequence. Decrements ref_counts via the allocator.
    // Removes the sequence from the table.
    void free_seq(llama_seq_id seq, llama_block_allocator & alloc);

    // Remove blocks covering token positions [pos_start, pos_end) for a sequence.
    // Used for context shift — O(1) block remapping instead of data movement.
    // Frees fully removed blocks via the allocator.
    void remove_blocks_range(llama_seq_id seq, uint32_t pos_start, uint32_t pos_end,
                             llama_block_allocator & alloc);

    // Clear all tables (used on reset).
    void clear(llama_block_allocator & alloc);
};
