#
# GROMACS PTO Makefile - 鲲鹏930 SVE 版本
#
# 简化实现，只保留 SVE 路径
#
# 使用方法：
#   make                    # 编译
#   make e2e                # 编译端到端测试
#   make clean              # 清理
#

CC ?= gcc

# ===== 鲲鹏930 SVE 优化配置 =====
CFLAGS  = -O3
CFLAGS += -march=armv8.6-a+sve
CFLAGS += -msve-vector-bits=256
CFLAGS += -ffast-math -funroll-loops -fipa-pta
CFLAGS += -falign-functions=64 -falign-loops=16
CFLAGS += -flto=auto
CFLAGS += -fopenmp -fopenmp-simd
CFLAGS += -Wall -Wextra -I.
CFLAGS += -DARM64=1 -DHAVE_SVE=1 -DSVE_VECTOR_BITS=256
CFLAGS += -DTILE_SIZE_DEFAULT=512 -DTILE_SIZE_MAX=1024

LDFLAGS = -lm -fopenmp

# SVE 核心源文件
SOURCES = gromacs_pto_tiling.c gromacs_pto_sve.c
HEADERS = gromacs_pto_arm.h
OBJECTS = $(SOURCES:.c=.o)

# 端到端测试
E2E_SOURCES = pto_e2e_v3.c
E2E_BENCHMARK_SOURCES = pto_gromacs_e2e_benchmark.c

all: $(OBJECTS) e2e

%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

e2e: $(OBJECTS) $(E2E_SOURCES)
	$(CC) $(CFLAGS) $(E2E_SOURCES) $(OBJECTS) -o pto_e2e_v3 $(LDFLAGS)

benchmark: $(OBJECTS) $(E2E_BENCHMARK_SOURCES)
	$(CC) $(CFLAGS) $(E2E_BENCHMARK_SOURCES) $(OBJECTS) -o pto_e2e_benchmark $(LDFLAGS)

# ===== PTO-ISA v8 (Mega-Kernel, Zero-Spill) =====
v8: pto_e2e_v8_megakernel.cpp pto_gromacs_core.hpp
	$(CXX) $(PTO_CXXFLAGS) $< -o pto_e2e_v8_megakernel $(PTO_LDFLAGS)

# ===== PTO-ISA v7 (C++ + SVE backend) =====
CXX ?= g++
PTO_CXXFLAGS = -O3 -march=armv8.6-a+sve -msve-vector-bits=256
PTO_CXXFLAGS += -ffast-math -funroll-loops -fopenmp
PTO_CXXFLAGS += -std=c++17 -Wall -Wextra -I.
PTO_CXXFLAGS += -DHAVE_SVE=1 -DARM64=1 -DSVE_VECTOR_BITS=256
PTO_LDFLAGS = -lm -fopenmp

ptoisa: pto_e2e_v6_ptoisa.cpp pto_gromacs_core.hpp
	$(CXX) $(PTO_CXXFLAGS) $< -o pto_e2e_v6_ptoisa $(PTO_LDFLAGS)

clean:
	rm -f $(OBJECTS) pto_e2e_v3 pto_e2e_benchmark pto_e2e_v6_ptoisa *.o

info:
	@echo "========================================"
	@echo "GROMACS PTO - SVE Only (鲲鹏930)"
	@echo "========================================"
	@echo "Sources: $(SOURCES)"
	@echo "Headers: $(HEADERS)"
	@echo "CFLAGS:  $(CFLAGS)"
	@echo "========================================"

.PHONY: all e2e benchmark ptoisa clean info
