#
# GROMACS PTO Makefile
#
# 第一阶段：非键相互作用优化
#
# 支持平台：
#   - ARM SVE/SME (默认)
#   - x86 AVX/AVX2
#
# 使用方法：
#   make            # 编译 ARM 版本（默认）
#   make x86        # 编译 x86 版本
#   make all        # 编译当前平台
#   make test       # 运行测试

# 编译器设置
CC ?= gcc
CXX ?= g++

# 默认平台检测
UNAME_M := $(shell uname -m)
ifeq ($(UNAME_M),x86_64)
    DEFAULT_PLATFORM = x86
else ifeq ($(UNAME_M),aarch64)
    DEFAULT_PLATFORM = arm
else
    DEFAULT_PLATFORM = x86
endif

# 平台选择（可通过 make PLATFORM=x86 覆盖）
PLATFORM ?= $(DEFAULT_PLATFORM)

# ===== ARM SVE/SME 配置 =====
ifeq ($(PLATFORM),arm)
    CFLAGS ?= -O3 -march=armv9-a+sve+sme
    CFLAGS += -Wall -Wextra -I.
    
    SOURCES = \
        gromacs_pto_tiling.c \
        gromacs_pto_sve.c \
        gromacs_pto_sme.c
    
    HEADERS = gromacs_pto_arm.h
    TEST_DIR = tests
    TEST_SOURCES = tests/test_nonbonded.c
endif

# ===== x86 AVX/AVX2 配置 =====
ifeq ($(PLATFORM),x86)
    CFLAGS ?= -O3 -march=x86-64 -mavx -mavx2 -mfma
    CFLAGS += -Wall -Wextra -I.

    SOURCES = \
        gromacs_pto_x86.c \
        gromacs_pto_avx.c \
        gromacs_pto_pypto.c
    
    HEADERS = gromacs_pto_x86.h
    TEST_DIR = tests
    TEST_SOURCES = tests/test_nonbonded_x86.c
endif

# 链接选项
LDFLAGS ?= -lm

# OpenMP并行支持
CFLAGS += -fopenmp
LDFLAGS += -fopenmp

# 对象文件
OBJECTS = $(SOURCES:.c=.o)
TEST_OBJECTS = $(TEST_SOURCES:.c=.o)

# 测试目标
TESTS = $(TEST_DIR)/test_nonbonded $(TEST_DIR)/benchmark_nonbonded $(TEST_DIR)/failure_tests

# 默认目标
all: $(OBJECTS) tests

# x86 目标（快捷方式）
x86:
	$(MAKE) PLATFORM=x86 all

# ARM 目标（快捷方式）
arm:
	$(MAKE) PLATFORM=arm all

# 编译对象
x86-%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

# 编译对象
%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

# 测试: 非键相互作用
tests/test_nonbonded: $(TEST_SOURCES) $(OBJECTS)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

# 性能基准测试
tests/benchmark_nonbonded: tests/benchmark_nonbonded.c $(OBJECTS)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

# 失败测试用例（验证修复）
tests/failure_tests: tests/failure_tests.c $(OBJECTS)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

tests: $(TESTS)

# 运行所有测试
test: tests/test_nonbonded
	./tests/test_nonbonded

# 清理
clean:
	rm -f $(OBJECTS)
	rm -f $(TESTS)
	rm -f *.o

# 安装到GROMACS项目
# 需要配置GROMACS源码路径
# GROMACS_SRC ?= /path/to/gromacs-2026.0
# install:
# 	cp $(HEADERS) $(GROMACS_SRC)/src/gromacs/nonbonded/
# 	cp $(SOURCES) $(GROMACS_SRC)/src/gromacs/nonbonded/

# 信息
info:
	@echo "GROMACS PTO for x86 AVX/AVX2"
	@echo "============================"
	@echo "Compiler: $(CC)"
	@echo "CFLAGS: $(CFLAGS)"
	@echo "Sources: $(SOURCES)"
	@echo "Objects: $(OBJECTS)"
	@echo "Tests: $(TESTS)"

.PHONY: all clean test tests info install x86 arm
