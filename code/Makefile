#
# GROMACS PTO ARM SVE/SME Makefile
#
# 第一阶段：非键相互作用优化
#

# 编译器设置 - 需要AArch64 GCC支持SVE/SME
CC ?= gcc
CXX ?= g++

# 编译选项
# 对于支持SVE/SME的ARMv9，使用: -march=armv9-a -O3
CFLAGS ?= -O3 -march=armv9-a+sve+sme
CFLAGS += -Wall -Wextra -I.

# 可选：如果只测试SVE，不需要SME
# CFLAGS += -O3 -march=armv8-a+sve

# 链接选项
LDFLAGS ?= -lm

# 打开OpenMP并行支持
CFLAGS += -fopenmp
LDFLAGS += -fopenmp

# 源文件
SOURCES = \
	gromacs_pto_tiling.c \
	gromacs_pto_sve.c \
	gromacs_pto_sme.c

# 对象文件
OBJECTS = $(SOURCES:.c=.o)

# 头文件
HEADERS = gromacs_pto_arm.h

# 测试目标
TESTS = tests/test_nonbonded tests/benchmark_nonbonded

# 默认目标
all: $(OBJECTS) tests

# 编译对象
%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

# 测试: 非键相互作用
tests/test_nonbonded: tests/test_nonbonded.c $(OBJECTS)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

# 性能基准测试
tests/benchmark_nonbonded: tests/benchmark_nonbonded.c $(OBJECTS)
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
	@echo "GROMACS PTO for ARM SVE/SME"
	@echo "============================"
	@echo "Compiler: $(CC)"
	@echo "CFLAGS: $(CFLAGS)"
	@echo "Sources: $(SOURCES)"
	@echo "Objects: $(OBJECTS)"
	@echo "Tests: $(TESTS)"

.PHONY: all clean test tests info install
