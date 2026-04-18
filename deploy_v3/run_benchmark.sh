#!/bin/bash
# PTO-GROMACS v3 鲲鹏930 部署和测试脚本
# 
# 使用方法:
#   scp -r deploy_v3/ kunpeng:~/pto_v3/
#   ssh kunpeng "cd ~/pto_v3 && bash run_benchmark.sh"
#
# 或者如果kunpeng不可达，通过跳板机:
#   scp -r deploy_v3/ jump:/tmp/pto_v3/
#   ssh jump "scp -r /tmp/pto_v3/ xuefenghao@192.168.90.45:~/pto_v3/"

set -e

echo "===== PTO-GROMACS v3 鲲鹏930 编译和测试 ====="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Arch: $(uname -m)"
echo "CPU: $(cat /proc/cpuinfo | grep 'model name' | head -1)"
echo "Cores: $(nproc)"
echo ""

# 检查SVE支持
if grep -q 'sve' /proc/cpuinfo 2>/dev/null || cat /proc/cpuinfo | grep -i 'features' | head -1 | grep -q 'sve'; then
    echo "SVE: Supported"
else
    echo "SVE: Checking via auxiliary vector..."
    if [ -f /proc/sys/auxv ]; then
        echo "SVE: Checking auxiliary vector"
    fi
fi

# 检查编译器
CC=${CC:-gcc}
echo "Compiler: $($CC --version | head -1)"
echo ""

# 查找GRO测试文件
GRO_FILE=""
for f in ../tests/results/md_test/water_24.gro ../tests/results/md_test/water_16.gro ./water_24.gro ./spc216.gro; do
    if [ -f "$f" ]; then
        GRO_FILE="$f"
        break
    fi
done

if [ -z "$GRO_FILE" ]; then
    echo "ERROR: No .gro test file found. Please provide a GRO file."
    echo "Usage: $0 <file.gro> [cutoff] [nsteps] [tile_size]"
    exit 1
fi

# 编译
echo "===== Compiling ====="
echo "CMD: $CC -O3 -march=armv8-a+sve -msve-vector-bits=256 -ffast-math -fopenmp pto_e2e_v3.c -o pto_e2e_v3 -lm"
$CC -O3 -march=armv8-a+sve -msve-vector-bits=256 -ffast-math -fopenmp pto_e2e_v3.c -o pto_e2e_v3 -lm 2>&1 || {
    echo "SVE 256-bit compilation failed, trying default SVE..."
    $CC -O3 -march=armv8-a+sve -ffast-math -fopenmp pto_e2e_v3.c -o pto_e2e_v3 -lm 2>&1 || {
        echo "SVE compilation failed, trying ARMv8..."
        $CC -O3 -march=armv8-a -ffast-math -fopenmp pto_e2e_v3.c -o pto_e2e_v3 -lm
    }
}
echo ""

# 运行测试
echo "===== Running Benchmark ====="
echo ""

# Test with different tile sizes
for TS in 32 64 128; do
    echo "--- Tile size: $TS ---"
    ./pto_e2e_v3 "$@" "$GRO_FILE" 1.0 200 "$TS" 2>&1 || ./pto_e2e_v3 "$GRO_FILE" 1.0 200 "$TS"
    echo ""
done

echo "===== Done ====="
