#!/bin/bash
#
# Benchmark script for PTO-Gromacs non-bonded interaction optimization
# Usage: ./benchmark_nonbonded.sh [output_dir] [repeat]
#

OUTPUT_DIR=${1:-./results}
REPEAT=${2:-5}

mkdir -p $OUTPUT_DIR

echo "=== PTO-Gromacs Non-bonded Benchmark ==="
echo "Output directory: $OUTPUT_DIR"
echo "Repeat: $REPEAT times"
echo

# Get system information
echo "System Information:"
echo "------------------"
uname -a >> $OUTPUT_DIR/system-info.txt
lscpu | grep "Model name\|CPU MHz\|CPU cores\|Thread(s) per core" >> $OUTPUT_DIR/system-info.txt
cat /proc/cpuinfo | grep -m1 "features" >> $OUTPUT_DIR/system-info.txt
echo "Done. System info saved to $OUTPUT_DIR/system-info.txt"
echo

# Run baseline (reference implementation)
echo "Running baseline..."
for i in $(seq 1 $REPEAT); do
    ./code/build/test_nonbonded --benchmark --reference >> $OUTPUT_DIR/baseline-timings.txt
done
echo "Baseline done."
echo

# Run PTO optimized implementation
echo "Running PTO optimized version..."
for i in $(seq 1 $REPEAT); do
    ./code/build/test_nonbonded --benchmark --pto >> $OUTPUT_DIR/pto-timings.txt
done
echo "PTO done."
echo

# Run with different tile sizes for tuning
echo "Running tile size sweep..."
for tile_size in 16 32 64 96 128; do
    echo "  Tile size: $tile_size"
    ./code/build/test_nonbonded --benchmark --pto --tile-size=$tile_size >> $OUTPUT_DIR/tile-size-sweep.txt
done
echo "Tile size sweep done."
echo

# Generate report
echo "Generating benchmark report..."
python3 ./benchmarks/scripts/analyze_benchmark.py \
    --baseline $OUTPUT_DIR/baseline-timings.txt \
    --pto $OUTPUT_DIR/pto-timings.txt \
    --tile-sweep $OUTPUT_DIR/tile-size-sweep.txt \
    --output $OUTPUT_DIR/benchmark-report.md

echo "=== Benchmark completed ==="
echo "Report: $OUTPUT_DIR/benchmark-report.md"
