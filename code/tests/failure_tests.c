/*
 * PTO-Gromacs - Failure Test Cases
 *
 * This file contains test cases that verify our fixes for known issues.
 * Each test corresponds to a failure case documented in:
 * docs/technical-reports/failure-test-cases-and-fixes.md
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../../gromacs_pto_arm.h"
#include "../../gromacs_pto_tiling.h"

/* Test tolerance */
#define TOLERANCE 1e-3f
#define TOLERANCE_STRICT 1e-6f

/* Test result tracking */
static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;

#define TEST_ASSERT(cond, msg) \
    do { \
        tests_run++; \
        if (!(cond)) { \
            printf("FAIL: %s\n", msg); \
            tests_failed++; \
        } else { \
            printf("PASS: %s\n", msg); \
            tests_passed++; \
        } \
    } while (0)

/* Test 1: Tile allocation for large tile sizes should not overflow */
static int test_large_tile_allocation(void) {
    printf("\n=== Test 1: Large tile allocation (fixes stack overflow) ===\n");
    
    PTOTile *tile = pto_tile_create(128);
    TEST_ASSERT(tile != NULL, "Tile creation for 128 atoms should succeed");
    
    if (tile != NULL) {
        /* Check we can access all elements */
        for (int i = 0; i < 128; i++) {
            tile->coords[i][0] = 1.0f;
            tile->coords[i][1] = 2.0f;
            tile->coords[i][2] = 3.0f;
            tile->forces[i][0] = 0.0f;
        }
        
        /* Verify writes */
        int ok = 1;
        for (int i = 0; i < 128; i++) {
            if (tile->coords[i][0] != 1.0f || tile->coords[i][1] != 2.0f) {
                ok = 0;
                break;
            }
        }
        TEST_ASSERT(ok, "All elements should be accessible and correct");
        
        pto_tile_destroy(tile);
    }
    
    /* Test cache checking */
    int fits_64 = pto_check_tile_fits_in_cache(64, 512);
    int fits_128 = pto_check_tile_fits_in_cache(128, 512);
    int fits_1024 = pto_check_tile_fits_in_cache(1024, 512);
    
    TEST_ASSERT(fits_64, "64 atoms should fit in 512KB cache");
    TEST_ASSERT(fits_128, "128 atoms should fit in 512KB cache");
    TEST_ASSERT(!fits_1024, "1024 atoms should NOT fit in 512KB cache");
    
    return 0;
}

/* Test 2: Force conservation with symmetric computation */
static int test_force_conservation(void) {
    printf("\n=== Test 2: Force conservation (Newton's third law) ===\n");
    
    /* Two-body test */
    float coords[2][3] = {
        {0.0f, 0.0f, 0.0f},
        {1.0f, 0.0f, 0.0f}
    };
    float forces[2][3] = {
        {0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f}
    };
    
    /* Compute interaction with symmetric force accumulation */
    pto_nonbonded_pair_compute(coords[0], coords[1], forces[0], forces[1], 1.0f, 1.0f);
    
    /* Newton's third law: F_i = -F_j */
    float sum_fx = forces[0][0] + forces[1][0];
    float sum_fy = forces[0][1] + forces[1][1];
    float sum_fz = forces[0][2] + forces[1][2];
    
    printf("  Force sum: fx=%.16f, fy=%.16f, fz=%.16f\n", sum_fx, sum_fy, sum_fz);
    
    TEST_ASSERT(fabsf(sum_fx) < TOLERANCE_STRICT, "Sum of x-forces should be zero");
    TEST_ASSERT(fabsf(sum_fy) < TOLERANCE_STRICT, "Sum of y-forces should be zero");
    TEST_ASSERT(fabsf(sum_fz) < TOLERANCE_STRICT, "Sum of z-forces should be zero");
    
    return 0;
}

/* Test 3: Periodic boundary condition with cross-boundary pairs */
static int test_periodic_boundary_cross_tile(void) {
    printf("\n=== Test 3: Periodic boundary condition (cross-boundary pairs) ===\n");
    
    float box[3][3] = {
        {2.0f, 0.0f, 0.0f},
        {0.0f, 2.0f, 0.0f},
        {0.0f, 0.0f, 2.0f}
    };
    
    /* Atoms near boundary: one at 0.1, one at 1.9 (should be distance 0.2 after PBC) */
    float coords[2][3] = {
        {0.1f, 0.0f, 0.0f},
        {1.9f, 0.0f, 0.0f}
    };
    
    float box_half[3];
    for (int i = 0; i < 3; i++) {
        box_half[i] = box[i][i] * 0.5f;
    }
    
    /* Apply minimum image convention */
    float dx = coords[1][0] - coords[0][0];
    pto_minimum_image(&dx, box[0][0], box_half[0]);
    
    float expected_dx = -0.2f;
    printf("  dx after PBC: %.6f (expected %.6f)\n", dx, expected_dx);
    
    TEST_ASSERT(fabsf(dx - expected_dx) < TOLERANCE, "dx should be -0.2 after PBC");
    
    /* Distance should be 0.2, not 1.8 */
    float r_sq = dx * dx;
    TEST_ASSERT(r_sq < 0.05f, "Squared distance should be small (~0.04)");
    
    return 0;
}

/* Test 4: Adaptive tile partitioning with heterogeneous density */
static int test_adaptive_tile_partitioning(void) {
    printf("\n=== Test 4: Adaptive tile partitioning (load balancing) ===\n");
    
    /* Create a system with heterogeneous density:
     * - One dense region (cluster of 100 atoms near origin)
     * - One sparse region (only 10 atoms)
     */
    const int n_atoms = 110;
    float *coords = (float*)malloc(n_atoms * 3 * sizeof(float));
    
    /* Dense cluster in center: 100 atoms */
    for (int i = 0; i < 100; i++) {
        coords[i*3 + 0] = 0.1f + (i % 10) * 0.1f;
        coords[i*3 + 1] = 0.1f + (i / 10) * 0.1f;
        coords[i*3 + 2] = 0.0f;
    }
    
    /* Sparse region: 10 atoms spread out */
    for (int i = 100; i < 110; i++) {
        coords[i*3 + 0] = 2.0f + (i - 100) * 1.0f;
        coords[i*3 + 1] = 0.0f;
        coords[i*3 + 2] = 0.0f;
    }
    
    float box[3][3] = {
        {12.0f, 0.0f, 0.0f},
        {12.0f, 0.0f, 0.0f},
        {12.0f, 0.0f, 0.0f}
    };
    
    PTOTilePartition partition;
    int result = pto_adaptive_tile_partition(coords, n_atoms, box, 32, &partition);
    
    TEST_ASSERT(result == 0, "Adaptive partitioning should succeed");
    TEST_ASSERT(partition.n_tiles >= 2, "Should create at least 2 tiles for heterogeneous density");
    
    /* Check that dense region is not split into too many tiny tiles */
    int max_atoms = 0;
    int min_atoms = n_atoms;
    int total_atoms = 0;
    for (int i = 0; i < partition.n_tiles; i++) {
        int na = partition.tiles[i].n_atoms;
        total_atoms += na;
        if (na > max_atoms) max_atoms = na;
        if (na < min_atoms) min_atoms = na;
    }
    
    TEST_ASSERT(total_atoms == n_atoms, "Total atoms should be conserved");
    printf("  Tiles: %d, min_atoms: %d, max_atoms: %d\n", partition.n_tiles, min_atoms, max_atoms);
    
    /* Max tile shouldn't be much bigger than target */
    TEST_ASSERT(max_atoms <= 2 * 32, "No tile should be more than 2x target size");
    
    pto_partition_destroy(&partition);
    free(coords);
    
    return 0;
}

/* Test 5: SVE vector length agnostic computation */
static int test_sve_vector_length_agnostic(void) {
    printf("\n=== Test 5: SVE vector length agnostic computation ===\n");
    
    /* Test that we can handle any SVE length */
    int vec_width = pto_sve_vector_width_words();
    printf("  SVE vector width: %d words\n", vec_width);
    
    TEST_ASSERT(vec_width > 0, "Vector width should be positive");
    TEST_ASSERT((vec_width & (vec_width - 1)) == 0, "Vector width should be power of two");
    
    /* Test alignment computation */
    for (int n = 1; n <= 128; n++) {
        int n_iter = pto_sve_iterations(n);
        int expected = (n + vec_width - 1) / vec_width;
        TEST_ASSERT(n_iter == expected, "Iteration count calculation should be correct");
    }
    
    return 0;
}

/* Test 6: Energy computation within tolerance */
static int test_energy_correctness(void) {
    printf("\n=== Test 6: Energy computation correctness ===\n");
    
    /* Reference system: two Ar atoms at 0.4 nm distance
     * LJ parameters: sigma=0.34 nm, epsilon=0.996 kJ/mol
     */
    float sigma = 0.34f;
    float epsilon = 0.996f;
    float r = 0.4f;
    
    float expected_energy = 4.0f * epsilon * (
        powf(sigma / r, 12) - powf(sigma / r, 6)
    );
    
    float computed_energy = pto_lj_energy(r, sigma, epsilon);
    
    printf("  Expected energy: %.6f kJ/mol\n", expected_energy);
    printf("  Computed energy: %.6f kJ/mol\n", computed_energy);
    printf("  Difference: %.6e\n", fabsf(computed_energy - expected_energy));
    
    TEST_ASSERT(fabsf(computed_energy - expected_energy) < TOLERANCE, 
                "LJ energy computation should match reference");
    
    return 0;
}

/* Main test runner */
int main(int argc, char **argv) {
    printf("PTO-Gromacs Failure Test Cases\n");
    printf("===============================\n");
    printf("Running tests for fixed failure cases...\n");
    
    test_large_tile_allocation();
    test_force_conservation();
    test_periodic_boundary_cross_tile();
    test_adaptive_tile_partitioning();
    test_sve_vector_length_agnostic();
    test_energy_correctness();
    
    printf("\n==================\n");
    printf("Test Summary:\n");
    printf("  Total:  %d\n", tests_run);
    printf("  Passed: %d\n", tests_passed);
    printf("  Failed: %d\n", tests_failed);
    printf("\n");
    
    if (tests_failed == 0) {
        printf("✅ All tests passed!\n");
        return 0;
    } else {
        printf("❌ Some tests FAILED!\n");
        return 1;
    }
}
