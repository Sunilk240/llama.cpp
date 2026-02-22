#include "llama-kv-cache-paged.h"

#include <algorithm>
#include <cassert>
#include <stdexcept>

//
// llama_block_allocator
//

llama_block_allocator::llama_block_allocator(uint32_t total_cells, uint32_t blk_size)
    : block_size(blk_size)
    , num_blocks(total_cells / blk_size)
    , ref_count(num_blocks, 0)
{
    assert(blk_size > 0);
    assert(total_cells >= blk_size);

    // Initialize free list with all block IDs (0 to num_blocks-1)
    // We push in reverse so that block 0 is allocated first (stack = LIFO)
    free_list.reserve(num_blocks);
    for (uint32_t i = num_blocks; i > 0; --i) {
        free_list.push_back(i - 1);
    }
}

uint32_t llama_block_allocator::allocate() {
    assert(!free_list.empty() && "BlockAllocator: no free blocks available");

    uint32_t block_id = free_list.back();
    free_list.pop_back();

    assert(ref_count[block_id] == 0);
    ref_count[block_id] = 1;

    return block_id;
}

void llama_block_allocator::free_block(uint32_t block_id) {
    assert(block_id < num_blocks);
    assert(ref_count[block_id] > 0);

    ref_count[block_id]--;

    if (ref_count[block_id] == 0) {
        free_list.push_back(block_id);
    }
}

void llama_block_allocator::inc_ref(uint32_t block_id) {
    assert(block_id < num_blocks);
    assert(ref_count[block_id] > 0 && "BlockAllocator: cannot inc_ref a free block");

    ref_count[block_id]++;
}

bool llama_block_allocator::can_allocate(uint32_t n_blocks) const {
    return free_list.size() >= n_blocks;
}

uint32_t llama_block_allocator::num_free() const {
    return (uint32_t) free_list.size();
}

uint32_t llama_block_allocator::total() const {
    return num_blocks;
}

//
// llama_block_table
//

uint32_t llama_block_table::logical_to_physical(llama_seq_id seq, llama_pos pos) const {
    assert(pos >= 0);

    auto it = tables.find(seq);
    assert(it != tables.end() && "BlockTable: sequence not found");

    const auto & blocks = it->second;
    uint32_t logical_block = (uint32_t) pos / block_size;

    assert(logical_block < blocks.size() && "BlockTable: position exceeds allocated blocks");

    // vLLM formula: physical_cell = block_table[logical_block] * block_size + (pos % block_size)
    return blocks[logical_block] * block_size + ((uint32_t) pos % block_size);
}

void llama_block_table::append_block(llama_seq_id seq, uint32_t block_id) {
    tables[seq].push_back(block_id);
}

bool llama_block_table::needs_new_block(llama_seq_id seq, uint32_t new_total_tokens) const {
    return new_total_tokens > capacity(seq);
}

uint32_t llama_block_table::capacity(llama_seq_id seq) const {
    auto it = tables.find(seq);
    if (it == tables.end()) {
        return 0;
    }
    return (uint32_t) it->second.size() * block_size;
}

uint32_t llama_block_table::num_blocks_for(llama_seq_id seq) const {
    auto it = tables.find(seq);
    if (it == tables.end()) {
        return 0;
    }
    return (uint32_t) it->second.size();
}

bool llama_block_table::has_seq(llama_seq_id seq) const {
    return tables.find(seq) != tables.end();
}

uint32_t llama_block_table::get_block_id(llama_seq_id seq, llama_pos pos) const {
    assert(pos >= 0);

    auto it = tables.find(seq);
    assert(it != tables.end() && "BlockTable: sequence not found");

    const auto & blocks = it->second;
    uint32_t logical_block = (uint32_t) pos / block_size;

    assert(logical_block < blocks.size() && "BlockTable: position exceeds allocated blocks");

    return blocks[logical_block];
}

void llama_block_table::replace_block(llama_seq_id seq, uint32_t logical_block_idx, uint32_t new_block_id) {
    auto it = tables.find(seq);
    assert(it != tables.end() && "BlockTable: sequence not found for replace_block");
    assert(logical_block_idx < it->second.size() && "BlockTable: logical_block_idx out of range");

    it->second[logical_block_idx] = new_block_id;
}

void llama_block_table::share(llama_seq_id src, llama_seq_id dst, llama_block_allocator & alloc) {
    auto it = tables.find(src);
    assert(it != tables.end() && "BlockTable: source sequence not found for share");

    // Copy block list from src to dst
    tables[dst] = it->second;

    // Increment ref_count for each shared block
    for (uint32_t block_id : tables[dst]) {
        alloc.inc_ref(block_id);
    }
}

void llama_block_table::free_seq(llama_seq_id seq, llama_block_allocator & alloc) {
    auto it = tables.find(seq);
    if (it == tables.end()) {
        return;
    }

    // Decrement ref_count for each block
    for (uint32_t block_id : it->second) {
        alloc.free_block(block_id);
    }

    tables.erase(it);
}

void llama_block_table::remove_blocks_range(
        llama_seq_id seq, uint32_t pos_start, uint32_t pos_end,
        llama_block_allocator & alloc) {
    auto it = tables.find(seq);
    if (it == tables.end()) {
        return;
    }

    auto & blocks = it->second;

    // Calculate which blocks are fully covered by the removal range
    uint32_t block_start = pos_start / block_size;
    uint32_t block_end   = (pos_end + block_size - 1) / block_size; // round up

    // Clamp to actual block list size
    block_end = std::min(block_end, (uint32_t) blocks.size());

    if (block_start >= block_end) {
        return;
    }

    // Free blocks in the removal range
    for (uint32_t i = block_start; i < block_end; ++i) {
        alloc.free_block(blocks[i]);
    }

    // Erase the blocks from the list (shifts later blocks left)
    blocks.erase(blocks.begin() + block_start, blocks.begin() + block_end);
}

void llama_block_table::clear(llama_block_allocator & alloc) {
    for (auto & [seq, blocks] : tables) {
        for (uint32_t block_id : blocks) {
            alloc.free_block(block_id);
        }
    }
    tables.clear();
}
