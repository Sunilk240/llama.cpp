// Test suite for PagedAttention data structures (BlockAllocator + BlockTable)
//
// Validates: allocation, ref counting, CoW sharing, logical→physical translation,
// block range removal (context shift), and edge cases.

#include "llama-kv-cache-paged.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <vector>

#define ASSERT_EQ(a, b) do { \
    auto _a = (a); auto _b = (b); \
    if (_a != _b) { \
        fprintf(stderr, "FAIL: %s == %s (%u != %u) at %s:%d\n", \
                #a, #b, (unsigned)_a, (unsigned)_b, __FILE__, __LINE__); \
        abort(); \
    } \
} while(0)

#define ASSERT_TRUE(x) do { \
    if (!(x)) { \
        fprintf(stderr, "FAIL: %s at %s:%d\n", #x, __FILE__, __LINE__); \
        abort(); \
    } \
} while(0)

#define ASSERT_FALSE(x) ASSERT_TRUE(!(x))

// ============================================================
// BlockAllocator Tests
// ============================================================

static void test_allocator_basic() {
    // 128 cells / 32 per block = 4 blocks
    llama_block_allocator alloc(128, 32);

    ASSERT_EQ(alloc.total(),    4u);
    ASSERT_EQ(alloc.num_free(), 4u);

    // Allocate first block
    uint32_t b0 = alloc.allocate();
    ASSERT_EQ(alloc.num_free(), 3u);
    ASSERT_EQ(alloc.ref_count[b0], 1u);

    // Allocate remaining blocks
    uint32_t b1 = alloc.allocate();
    uint32_t b2 = alloc.allocate();
    uint32_t b3 = alloc.allocate();
    ASSERT_EQ(alloc.num_free(), 0u);

    // All block IDs should be unique
    std::set<uint32_t> ids = {b0, b1, b2, b3};
    ASSERT_EQ(ids.size(), 4u);

    // Cannot allocate when empty
    ASSERT_FALSE(alloc.can_allocate(1));

    // Free one block
    alloc.free_block(b2);
    ASSERT_EQ(alloc.num_free(), 1u);
    ASSERT_EQ(alloc.ref_count[b2], 0u);

    // Can allocate again
    ASSERT_TRUE(alloc.can_allocate(1));
    uint32_t b4 = alloc.allocate();
    ASSERT_EQ(b4, b2); // Should get the same block back (LIFO)
}

static void test_allocator_ref_counting() {
    llama_block_allocator alloc(64, 32); // 2 blocks

    uint32_t b0 = alloc.allocate();
    ASSERT_EQ(alloc.ref_count[b0], 1u);

    // Inc ref (CoW sharing)
    alloc.inc_ref(b0);
    ASSERT_EQ(alloc.ref_count[b0], 2u);

    // First free → ref_count=1, block NOT returned to free list
    alloc.free_block(b0);
    ASSERT_EQ(alloc.ref_count[b0], 1u);
    ASSERT_EQ(alloc.num_free(), 1u); // only the other block is free

    // Second free → ref_count=0, block returned to free list
    alloc.free_block(b0);
    ASSERT_EQ(alloc.ref_count[b0], 0u);
    ASSERT_EQ(alloc.num_free(), 2u); // both blocks free
}

static void test_allocator_can_allocate() {
    llama_block_allocator alloc(96, 32); // 3 blocks

    ASSERT_TRUE(alloc.can_allocate(1));
    ASSERT_TRUE(alloc.can_allocate(3));
    ASSERT_FALSE(alloc.can_allocate(4));

    alloc.allocate();
    alloc.allocate();
    ASSERT_TRUE(alloc.can_allocate(1));
    ASSERT_FALSE(alloc.can_allocate(2));
}

static void test_allocator_free_all() {
    llama_block_allocator alloc(128, 32); // 4 blocks

    std::vector<uint32_t> blocks;
    for (int i = 0; i < 4; i++) {
        blocks.push_back(alloc.allocate());
    }
    ASSERT_EQ(alloc.num_free(), 0u);

    // Free all
    for (uint32_t b : blocks) {
        alloc.free_block(b);
    }
    ASSERT_EQ(alloc.num_free(), 4u);
}

// ============================================================
// BlockTable Tests
// ============================================================

static void test_table_logical_to_physical() {
    llama_block_allocator alloc(256, 32); // 8 blocks
    llama_block_table table;
    table.block_size = 32;

    // Sequence 0: allocate 2 blocks
    uint32_t b0 = alloc.allocate(); // block ID 0
    uint32_t b1 = alloc.allocate(); // block ID 1
    table.append_block(0, b0);
    table.append_block(0, b1);

    // Position 0 → block 0, offset 0 → physical cell 0*32+0 = 0
    ASSERT_EQ(table.logical_to_physical(0, 0), b0 * 32 + 0);

    // Position 31 → block 0, offset 31 → physical cell 0*32+31 = 31
    ASSERT_EQ(table.logical_to_physical(0, 31), b0 * 32 + 31);

    // Position 32 → block 1, offset 0 → physical cell 1*32+0 = 32
    ASSERT_EQ(table.logical_to_physical(0, 32), b1 * 32 + 0);

    // Position 50 → block 1, offset 18 → physical cell 1*32+18 = 50
    ASSERT_EQ(table.logical_to_physical(0, 50), b1 * 32 + 18);
}

static void test_table_logical_to_physical_noncontiguous() {
    // Test when physical blocks are NOT contiguous (the whole point of PA!)
    llama_block_allocator alloc(256, 32); // 8 blocks

    // Allocate blocks for seq 0
    uint32_t s0_b0 = alloc.allocate(); // block 0
    uint32_t s0_b1 = alloc.allocate(); // block 1

    // Allocate block for seq 1 (this goes in between!)
    uint32_t s1_b0 = alloc.allocate(); // block 2

    // Allocate another block for seq 0
    uint32_t s0_b2 = alloc.allocate(); // block 3

    llama_block_table table;
    table.block_size = 32;

    table.append_block(0, s0_b0); // logical block 0 → physical 0
    table.append_block(0, s0_b1); // logical block 1 → physical 1
    table.append_block(0, s0_b2); // logical block 2 → physical 3 (GAP!)
    table.append_block(1, s1_b0); // seq 1: logical block 0 → physical 2

    // Seq 0, pos 64 → logical block 2, offset 0 → physical cell 3*32+0 = 96
    ASSERT_EQ(table.logical_to_physical(0, 64), s0_b2 * 32 + 0);

    // Seq 1, pos 5 → logical block 0, offset 5 → physical cell 2*32+5 = 69
    ASSERT_EQ(table.logical_to_physical(1, 5), s1_b0 * 32 + 5);
}

static void test_table_needs_new_block() {
    llama_block_table table;
    table.block_size = 32;

    // Empty seq needs block at token 1
    ASSERT_TRUE(table.needs_new_block(0, 1));

    // Add one block → capacity = 32
    table.append_block(0, 0);
    ASSERT_FALSE(table.needs_new_block(0, 1));
    ASSERT_FALSE(table.needs_new_block(0, 32));
    ASSERT_TRUE(table.needs_new_block(0, 33)); // over capacity
}

static void test_table_capacity() {
    llama_block_table table;
    table.block_size = 32;

    ASSERT_EQ(table.capacity(0), 0u);
    ASSERT_EQ(table.capacity(99), 0u); // non-existent seq

    table.append_block(0, 0);
    ASSERT_EQ(table.capacity(0), 32u);

    table.append_block(0, 1);
    ASSERT_EQ(table.capacity(0), 64u);
}

static void test_table_share_cow() {
    llama_block_allocator alloc(256, 32); // 8 blocks
    llama_block_table table;
    table.block_size = 32;

    // Seq 0 gets 2 blocks
    uint32_t b0 = alloc.allocate();
    uint32_t b1 = alloc.allocate();
    table.append_block(0, b0);
    table.append_block(0, b1);

    ASSERT_EQ(alloc.ref_count[b0], 1u);
    ASSERT_EQ(alloc.ref_count[b1], 1u);

    // Share seq 0 → seq 1 (CoW)
    table.share(0, 1, alloc);

    // Both seqs should have same blocks
    ASSERT_EQ(table.logical_to_physical(0, 0), table.logical_to_physical(1, 0));
    ASSERT_EQ(table.logical_to_physical(0, 40), table.logical_to_physical(1, 40));

    // Ref counts should be 2
    ASSERT_EQ(alloc.ref_count[b0], 2u);
    ASSERT_EQ(alloc.ref_count[b1], 2u);

    // Free seq 1 → ref counts back to 1
    table.free_seq(1, alloc);
    ASSERT_EQ(alloc.ref_count[b0], 1u);
    ASSERT_EQ(alloc.ref_count[b1], 1u);
    ASSERT_FALSE(table.has_seq(1));

    // Free seq 0 → ref counts = 0, blocks returned to free list
    table.free_seq(0, alloc);
    ASSERT_EQ(alloc.ref_count[b0], 0u);
    ASSERT_EQ(alloc.ref_count[b1], 0u);
    ASSERT_EQ(alloc.num_free(), 8u);
}

static void test_table_free_seq() {
    llama_block_allocator alloc(128, 32); // 4 blocks
    llama_block_table table;
    table.block_size = 32;

    uint32_t b0 = alloc.allocate();
    uint32_t b1 = alloc.allocate();
    table.append_block(0, b0);
    table.append_block(0, b1);

    ASSERT_EQ(alloc.num_free(), 2u);

    table.free_seq(0, alloc);
    ASSERT_EQ(alloc.num_free(), 4u);
    ASSERT_FALSE(table.has_seq(0));

    // Freeing again should be safe (no-op)
    table.free_seq(0, alloc);
    ASSERT_EQ(alloc.num_free(), 4u);
}

static void test_table_remove_blocks_range() {
    // Context shift test: remove middle blocks
    llama_block_allocator alloc(256, 32); // 8 blocks
    llama_block_table table;
    table.block_size = 32;

    // Seq 0 gets 4 blocks (128 tokens)
    uint32_t b0 = alloc.allocate();
    uint32_t b1 = alloc.allocate();
    uint32_t b2 = alloc.allocate();
    uint32_t b3 = alloc.allocate();
    table.append_block(0, b0);
    table.append_block(0, b1);
    table.append_block(0, b2);
    table.append_block(0, b3);

    ASSERT_EQ(table.num_blocks_for(0), 4u);
    ASSERT_EQ(alloc.num_free(), 4u);

    // Remove blocks covering positions [32, 96) → blocks 1 and 2
    table.remove_blocks_range(0, 32, 96, alloc);

    // Should have 2 blocks remaining (b0 and b3)
    ASSERT_EQ(table.num_blocks_for(0), 2u);
    ASSERT_EQ(alloc.num_free(), 6u); // 4 original free + 2 freed

    // b1 and b2 should be free
    ASSERT_EQ(alloc.ref_count[b1], 0u);
    ASSERT_EQ(alloc.ref_count[b2], 0u);

    // b0 and b3 still allocated
    ASSERT_EQ(alloc.ref_count[b0], 1u);
    ASSERT_EQ(alloc.ref_count[b3], 1u);
}

static void test_table_clear() {
    llama_block_allocator alloc(128, 32);
    llama_block_table table;
    table.block_size = 32;

    table.append_block(0, alloc.allocate());
    table.append_block(0, alloc.allocate());
    table.append_block(1, alloc.allocate());

    ASSERT_EQ(alloc.num_free(), 1u);

    table.clear(alloc);
    ASSERT_EQ(alloc.num_free(), 4u);
    ASSERT_FALSE(table.has_seq(0));
    ASSERT_FALSE(table.has_seq(1));
}

// ============================================================
// Edge Case Tests
// ============================================================

static void test_edge_single_token() {
    llama_block_allocator alloc(32, 32); // 1 block
    llama_block_table table;
    table.block_size = 32;

    ASSERT_TRUE(table.needs_new_block(0, 1));
    uint32_t b = alloc.allocate();
    table.append_block(0, b);

    ASSERT_EQ(table.logical_to_physical(0, 0), b * 32);
    ASSERT_FALSE(table.needs_new_block(0, 1));
    ASSERT_TRUE(table.needs_new_block(0, 33)); // next block needed at 33
}

static void test_edge_exact_block_boundary() {
    llama_block_allocator alloc(64, 32); // 2 blocks
    llama_block_table table;
    table.block_size = 32;

    uint32_t b0 = alloc.allocate();
    table.append_block(0, b0);

    // Exactly 32 tokens fits in 1 block
    ASSERT_FALSE(table.needs_new_block(0, 32));

    // 33 tokens needs 2nd block
    ASSERT_TRUE(table.needs_new_block(0, 33));

    uint32_t b1 = alloc.allocate();
    table.append_block(0, b1);

    // Position 32 should map to second block
    ASSERT_EQ(table.logical_to_physical(0, 32), b1 * 32 + 0);
}

static void test_edge_block_size_16() {
    // Test with block_size=16 (vLLM supports 8, 16, 32)
    llama_block_allocator alloc(64, 16); // 4 blocks
    llama_block_table table;
    table.block_size = 16;

    ASSERT_EQ(alloc.total(), 4u);

    uint32_t b0 = alloc.allocate();
    uint32_t b1 = alloc.allocate();
    table.append_block(0, b0);
    table.append_block(0, b1);

    // Position 15 → block 0, offset 15
    ASSERT_EQ(table.logical_to_physical(0, 15), b0 * 16 + 15);

    // Position 16 → block 1, offset 0
    ASSERT_EQ(table.logical_to_physical(0, 16), b1 * 16 + 0);
}

static void test_multiple_sequences() {
    llama_block_allocator alloc(256, 32); // 8 blocks
    llama_block_table table;
    table.block_size = 32;

    // Allocate interleaved blocks for 3 sequences
    for (int seq = 0; seq < 3; seq++) {
        uint32_t b = alloc.allocate();
        table.append_block(seq, b);
    }

    // Each seq has 1 block, they should be different physical blocks
    uint32_t p0 = table.logical_to_physical(0, 0);
    uint32_t p1 = table.logical_to_physical(1, 0);
    uint32_t p2 = table.logical_to_physical(2, 0);

    // Physical cells should be in different blocks
    ASSERT_TRUE(p0 / 32 != p1 / 32);
    ASSERT_TRUE(p1 / 32 != p2 / 32);
    ASSERT_TRUE(p0 / 32 != p2 / 32);
}

// ============================================================
// Integration-style test: simulate a mini inference run
// ============================================================

static void test_mini_inference_simulation() {
    // Simulate: 2 sequences, each growing from 0 to ~80 tokens
    llama_block_allocator alloc(256, 32); // 8 blocks
    llama_block_table table;
    table.block_size = 32;

    for (int seq = 0; seq < 2; seq++) {
        for (int pos = 0; pos < 80; pos++) {
            // Check if we need a new block
            if (table.needs_new_block(seq, pos + 1)) {
                ASSERT_TRUE(alloc.can_allocate(1));
                uint32_t b = alloc.allocate();
                table.append_block(seq, b);
            }

            // Verify translation works
            uint32_t phys = table.logical_to_physical(seq, pos);
            ASSERT_TRUE(phys < 256); // within total cells
        }
    }

    // Each seq should have 3 blocks (80 tokens / 32 = 2.5, rounded up = 3)
    ASSERT_EQ(table.num_blocks_for(0), 3u);
    ASSERT_EQ(table.num_blocks_for(1), 3u);

    // 6 blocks used, 2 free
    ASSERT_EQ(alloc.num_free(), 2u);

    // Now simulate context shift for seq 0: remove blocks [32, 64)
    table.remove_blocks_range(0, 32, 64, alloc);
    ASSERT_EQ(table.num_blocks_for(0), 2u);
    ASSERT_EQ(alloc.num_free(), 3u);

    // Free seq 1
    table.free_seq(1, alloc);
    ASSERT_EQ(alloc.num_free(), 6u);

    // Free seq 0
    table.free_seq(0, alloc);
    ASSERT_EQ(alloc.num_free(), 8u);
}

// ============================================================
// Test runner
// ============================================================

static void run(const char * name, void (*f)()) {
    printf("  %-48s ", name);
    fflush(stdout);
    f();
    printf("PASSED\n");
}

int main() {
    printf("PagedAttention Data Structure Tests\n");
    printf("====================================\n");

    printf("\nBlockAllocator:\n");
    run("basic allocation and free",      test_allocator_basic);
    run("reference counting (CoW)",       test_allocator_ref_counting);
    run("can_allocate boundary",          test_allocator_can_allocate);
    run("free all blocks",               test_allocator_free_all);

    printf("\nBlockTable:\n");
    run("logical→physical translation",   test_table_logical_to_physical);
    run("non-contiguous blocks",          test_table_logical_to_physical_noncontiguous);
    run("needs_new_block boundary",       test_table_needs_new_block);
    run("capacity tracking",             test_table_capacity);
    run("CoW share + ref counting",       test_table_share_cow);
    run("free_seq cleanup",              test_table_free_seq);
    run("remove_blocks_range (ctx shift)", test_table_remove_blocks_range);
    run("clear all",                     test_table_clear);

    printf("\nEdge Cases:\n");
    run("single token",                  test_edge_single_token);
    run("exact block boundary",          test_edge_exact_block_boundary);
    run("block_size=16",                 test_edge_block_size_16);
    run("multiple sequences",            test_multiple_sequences);

    printf("\nIntegration:\n");
    run("mini inference simulation",      test_mini_inference_simulation);

    printf("\n====================================\n");
    printf("All %d tests PASSED!\n", 17);
    return 0;
}
