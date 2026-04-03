# Makefile for x86 GROMACS Tests

CC = gcc
CFLAGS = -O2 -march=native -Wall -Wextra -std=c11
LDFLAGS = -lm

# SIMD flags based on CPU support
SIMD_FLAGS = -msse2
ifeq ($(shell grep -q avx /proc/cpuinfo && echo yes),yes)
    SIMD_FLAGS += -mavx
endif
ifeq ($(shell grep -q avx2 /proc/cpuinfo && echo yes),yes)
    SIMD_FLAGS += -mavx2 -mfma
endif

CFLAGS += $(SIMD_FLAGS)

# Source files
TEST_FRAMEWORK = test_framework_x86.c
TEST_FRAMEWORK_H = test_framework_x86.h

# Targets
all: test_unit benchmark_nonbonded

test_unit: test_unit_x86.c $(TEST_FRAMEWORK) $(TEST_FRAMEWORK_H)
	$(CC) $(CFLAGS) -o $@ test_unit_x86.c $(TEST_FRAMEWORK) $(LDFLAGS)

benchmark_nonbonded: benchmark_nonbonded_x86.c $(TEST_FRAMEWORK) $(TEST_FRAMEWORK_H)
	$(CC) $(CFLAGS) -o $@ benchmark_nonbonded_x86.c $(TEST_FRAMEWORK) $(LDFLAGS)

clean:
	rm -f test_unit benchmark_nonbonded *.o

run-test: test_unit
	./test_unit

run-bench: benchmark_nonbonded
	./benchmark_nonbonded

# GROMACS tests (requires GROMACS installed)
gmx-test:
	bash run_e2e_test.sh

.PHONY: all clean run-test run-bench gmx-test
