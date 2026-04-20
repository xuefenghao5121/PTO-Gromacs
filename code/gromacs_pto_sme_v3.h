/*
 * GROMACS PTO SME v3 Header
 */

#ifndef GROMACS_PTO_SME_V3_H
#define GROMACS_PTO_SME_V3_H

#include <stdbool.h>
#include "gromacs_pto_arm.h"

/* SME v3 API */
bool gmx_pto_sme_v3_is_available(void);
bool gmx_pto_sme_v3_enable(void);
void gmx_pto_sme_v3_disable(void);
bool gmx_pto_sme_v3_init(void);
void gmx_pto_sme_v3_cleanup(void);

void gmx_pto_sme_v3_load_coords(sme_tile_v3_t *tile, 
                                  const float *coords, int n);
void gmx_pto_sme_v3_fmmla_distance(sme_tile_v3_t *tile);
void gmx_pto_sme_v3_compute_fused(sme_tile_v3_t *tile, float cutoff_sq,
                                    float lj_epsilon, float lj_sigma_sq);
void gmx_pto_sme_v3_reduce_forces(sme_tile_v3_t *tile, float *forces, int n);
int gmx_pto_sme_v3_nonbonded_compute(gmx_pto_nonbonded_context_t *context,
                                       gmx_pto_atom_data_t *atom_data);
void gmx_pto_sme_v3_benchmark(gmx_pto_nonbonded_context_t *context,
                                gmx_pto_atom_data_t *atom_data,
                                int n_steps);
void gmx_pto_sme_v3_print_info(void);

#endif
