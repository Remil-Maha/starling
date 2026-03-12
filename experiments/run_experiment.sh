#!/bin/bash
# =============================================================================
# Starling I/O Metrics Experiment - Run Script
# =============================================================================
#
# Prerequisites:
#   1. Build Starling with the experiments target:
#        mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make run_io_metrics -j$(nproc)
#
#   2. Build the disk index (if not already done):
#        ./build/tests/build_disk_index --data_type float --dist_fn l2 \
#            --data_path /path/to/corpus_vectors.fbin \
#            --index_path_prefix /path/to/index/nq \
#            -R 64 -L 100 -B 0.3 -M 15
#
#   3. Run graph partitioning (for Starling page search):
#        ./build/graph_partition/partitioner --data_type float \
#            --index_file /path/to/index/nq_disk.index \
#            --gp_file /path/to/index/nq_partition.bin \
#            --block_size 1
#
#   4. Set the variables below and run this script.
#
# =============================================================================

set -e

# ============ CONFIGURATION - MODIFY THESE ============

# Path to query vectors in .fbin format
QUERY_FILE="/path/to/query_vectors.fbin"

# Path prefix of the built index (same as --index_path_prefix during build)
INDEX_PREFIX="/path/to/index/nq"

# Path to the disk index file (_disk.index)
DISK_FILE="/path/to/index/nq_disk.index"

# Output directory for results
RESULT_DIR="$(dirname "$0")/results"

# Search parameters
L=100           # Search list size
K=10            # Recall@K
W=4             # Beam width
NUM_QUERIES=50  # Number of queries to test
NUM_THREADS=1   # Use 1 for instrumented search (required)

# Search mode: 1 = Starling (page search), 0 = DiskANN (beam search)
USE_PAGE_SEARCH=1

# Distance function: l2, mips, or cosine
DIST_FN="l2"

# In-memory navigation graph L (0 to disable)
MEM_L=0
MEM_INDEX_PATH=""

# ============ END CONFIGURATION ============

mkdir -p "$RESULT_DIR"

BUILD_DIR="$(dirname "$0")/../build"

echo "Running I/O Metrics Experiment..."
echo "  Query file: $QUERY_FILE"
echo "  Index prefix: $INDEX_PREFIX"
echo "  Disk file: $DISK_FILE"
echo "  Results dir: $RESULT_DIR"
echo ""

"$BUILD_DIR/experiments/run_io_metrics" \
    --data_type float \
    --dist_fn "$DIST_FN" \
    --index_path_prefix "$INDEX_PREFIX" \
    --disk_file_path "$DISK_FILE" \
    --query_file "$QUERY_FILE" \
    --result_dir "$RESULT_DIR" \
    -K "$K" \
    -L "$L" \
    -W "$W" \
    -Q "$NUM_QUERIES" \
    -T "$NUM_THREADS" \
    --use_page_search "$USE_PAGE_SEARCH" \
    --mem_L "$MEM_L" \
    --mem_index_path "$MEM_INDEX_PATH"

echo ""
echo "=== Results ==="
echo "  Summary:      $RESULT_DIR/summary.csv"
echo "  Per-query:    $RESULT_DIR/per_query.csv"
echo "  Block access: $RESULT_DIR/block_access.csv"
