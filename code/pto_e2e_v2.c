/*
 * PTO-GROMACS 端到端性能对比测试 v2
 * 
 * 修复: 线程私有力缓冲区 + 消除标量gather拷贝
 *
 * 编译:
 *   gcc -O3 -march=armv8-a+sve -msve-vector-bits=256 -ffast-math -fopenmp \
 *       pto_e2e_benchmark_v2.c -o pto_e2e_v2 -lm -fopenmp
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <arm_sve.h>

/* ============ GRO file parser ============ */
typedef struct { int natoms; float *x; float box[3]; } GroData;

static GroData* read_gro(const char *fn) {
    FILE *fp = fopen(fn,"r"); if(!fp) return NULL;
    fprintf(stderr,"read_gro start\n"); GroData *g = calloc(1,sizeof(GroData)); char line[256];
    fgets(line,sizeof(line),fp); fgets(line,sizeof(line),fp);
    fprintf(stderr,"line=[%s]\n",line); g->natoms = atoi(line);
    g->x = malloc(g->natoms*3*sizeof(float));
    for(int i=0;i<g->natoms;i++){
        fgets(line,sizeof(line),fp);
        char *p = line+20;
        g->x[i*3+0]=strtof(p,&p); g->x[i*3+1]=strtof(p,&p); g->x[i*3+2]=strtof(p,&p);
    }
    fgets(line,sizeof(line),fp);
    sscanf(line,"%f%f%f",&g->box[0],&g->box[1],&g->box[2]);
    fclose(fp); return g;
}

/* ============ Neighbor list ============ */
typedef struct { int *start; int *jatoms; int *count; int total_pairs; } NBList;

static NBList* build_nblist(float *x, int n, float box[3], float cut) {
    NBList *nl = calloc(1,sizeof(NBList));
    nl->start = malloc(n*sizeof(int));
    nl->count = calloc(n,sizeof(int));
    float csq = cut*cut;
    int total=0;
    for(int i=0;i<n;i++) for(int j=i+1;j<n;j++){
        float dx=x[i*3]-x[j*3], dy=x[i*3+1]-x[j*3+1], dz=x[i*3+2]-x[j*3+2];
        dx-=box[0]*roundf(dx/box[0]); dy-=box[1]*roundf(dy/box[1]); dz-=box[2]*roundf(dz/box[2]);
        if(dx*dx+dy*dy+dz*dz < csq) { nl->count[i]++; total++; }
    }
    nl->start[0]=0;
    for(int i=1;i<n;i++) nl->start[i]=nl->start[i-1]+nl->count[i-1];
    nl->jatoms = malloc(total*sizeof(int));
    nl->total_pairs = total;
    int *pos = calloc(n,sizeof(int));
    for(int i=0;i<n;i++) for(int j=i+1;j<n;j++){
        float dx=x[i*3]-x[j*3], dy=x[i*3+1]-x[j*3+1], dz=x[i*3+2]-x[j*3+2];
        dx-=box[0]*roundf(dx/box[0]); dy-=box[1]*roundf(dy/box[1]); dz-=box[2]*roundf(dz/box[2]);
        if(dx*dx+dy*dy+dz*dz < csq) nl->jatoms[nl->start[i]+pos[i]++]=j;
    }
    free(pos); return nl;
}

/* ============ Scalar baseline ============ */
static double scalar_nb(float *x, float *f, int n, float box[3], NBList *nl, float cut) {
    float csq=cut*cut, eps=0.5f, ssq=0.09f;
    struct timespec t0; clock_gettime(CLOCK_MONOTONIC,&t0);
    memset(f,0,n*3*sizeof(float));
    #pragma omp parallel
    {
        float *lf = calloc(n*3,sizeof(float));
        #pragma omp for schedule(dynamic,64)
        for(int i=0;i<n;i++){
            float fx=0,fy=0,fz=0, xi=x[i*3],yi=x[i*3+1],zi=x[i*3+2];
            for(int k=0;k<nl->count[i];k++){
                int j=nl->jatoms[nl->start[i]+k];
                float dx=xi-x[j*3], dy=yi-x[j*3+1], dz=zi-x[j*3+2];
                dx-=box[0]*roundf(dx/box[0]); dy-=box[1]*roundf(dy/box[1]); dz-=box[2]*roundf(dz/box[2]);
                float rsq=dx*dx+dy*dy+dz*dz;
                if(rsq<csq && rsq>1e-8f){
                    float ir=1.0f/rsq, s2=ssq*ir, s6=s2*s2*s2, s12=s6*s6;
                    float fr=24.0f*eps*(2.0f*s12-s6)*ir;
                    fx+=fr*dx; fy+=fr*dy; fz+=fr*dz;
                    lf[j*3]-=fr*dx; lf[j*3+1]-=fr*dy; lf[j*3+2]-=fr*dz;
                }
            }
            lf[i*3]+=fx; lf[i*3+1]+=fy; lf[i*3+2]+=fz;
        }
        #pragma omp critical
        for(int k=0;k<n*3;k++) f[k]+=lf[k];
        free(lf);
    }
    struct timespec t1; clock_gettime(CLOCK_MONOTONIC,&t1);
    return (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)*1e-9;
}

/* ============ PTO-SVE optimized (v2: thread-local + fused) ============ */
static double pto_sve_nb(float *x, float *f, int n, float box[3], NBList *nl, float cut) {
    float csq=cut*cut, eps=0.5f, ssq=0.09f;
    const int vl = 8;
    struct timespec t0; clock_gettime(CLOCK_MONOTONIC,&t0);
    memset(f,0,n*3*sizeof(float));
    
    #pragma omp parallel
    {
        float *lf = calloc(n*3,sizeof(float));
        svbool_t all_p = svptrue_b32();
        
        #pragma omp for schedule(dynamic,64)
        for(int i=0;i<n;i++){
            float xi=x[i*3],yi=x[i*3+1],zi=x[i*3+2];
            svfloat32_t xi_v=svdup_f32(xi), yi_v=svdup_f32(yi), zi_v=svdup_f32(zi);
            svfloat32_t fx_a=svdup_f32(0), fy_a=svdup_f32(0), fz_a=svdup_f32(0);
            
            int nn = nl->count[i];
            for(int k=0;k<nn;k+=vl){
                int rem = nn-k;
                svbool_t pg = svwhilelt_b32(0, rem);
                
                /* Gather j-coords via contiguous index array + manual gather */
                float jx[8],jy[vl],jz[vl];
                for(int m=0;m<rem;m++){
                    int j=nl->jatoms[nl->start[i]+k+m];
                    float dx=xi-x[j*3], dy=yi-x[j*3+1], dz=zi-x[j*3+2];
                    dx-=box[0]*roundf(dx/box[0]); dy-=box[1]*roundf(dy/box[1]); dz-=box[2]*roundf(dz/box[2]);
                    jx[m]=dx; jy[m]=dy; jz[m]=dz;
                }
                svfloat32_t dx_v=svld1_f32(pg,jx), dy_v=svld1_f32(pg,jy), dz_v=svld1_f32(pg,jz);
                
                /* rsq (PBC already applied) */
                svfloat32_t rsq = svadd_f32_x(pg, svadd_f32_x(pg,
                    svmul_f32_x(pg,dx_v,dx_v), svmul_f32_x(pg,dy_v,dy_v)),
                    svmul_f32_x(pg,dz_v,dz_v));
                
                svbool_t valid = svand_b_z(pg,
                    svcmplt_f32(pg,rsq,svdup_f32(csq)),
                    svcmpgt_f32(pg,rsq,svdup_f32(1e-8f)));
                
                if(svptest_any(all_p,valid)){
                    /* Fused LJ computation */
                    svfloat32_t ir = svdiv_f32_z(valid,svdup_f32(1.0f),rsq);
                    svfloat32_t s2=svmul_f32_z(valid,svdup_f32(ssq),ir);
                    svfloat32_t s6=svmul_f32_z(valid,svmul_f32_z(valid,s2,s2),s2);
                    svfloat32_t s12=svmul_f32_z(valid,s6,s6);
                    svfloat32_t fr=svmul_f32_z(valid,svdup_f32(24.0f*eps),
                        svmul_f32_z(valid,svsub_f32_z(valid,svmul_f32_z(valid,svdup_f32(2.0f),s12),s6),ir));
                    
                    svfloat32_t fx_v=svmul_f32_z(valid,fr,dx_v);
                    svfloat32_t fy_v=svmul_f32_z(valid,fr,dy_v);
                    svfloat32_t fz_v=svmul_f32_z(valid,fr,dz_v);
                    
                    /* Accumulate i-forces in registers */
                    fx_a=svadd_f32_m(valid,fx_a,fx_v);
                    fy_a=svadd_f32_m(valid,fy_a,fy_v);
                    fz_a=svadd_f32_m(valid,fz_a,fz_v);
                    
                    /* Scatter j-forces to thread-local buffer (no atomic!) */
                    float fxo[8],fyo[vl],fzo[vl];
                    svst1_f32(pg,fxo,fx_v); svst1_f32(pg,fyo,fy_v); svst1_f32(pg,fzo,fz_v);
                    for(int m=0;m<rem;m++){
                        int j=nl->jatoms[nl->start[i]+k+m];
                        lf[j*3]-=fxo[m]; lf[j*3+1]-=fyo[m]; lf[j*3+2]-=fzo[m];
                    }
                }
            }
            /* Reduce i-forces */
            lf[i*3]+=svaddv_f32(all_p,fx_a);
            lf[i*3+1]+=svaddv_f32(all_p,fy_a);
            lf[i*3+2]+=svaddv_f32(all_p,fz_a);
        }
        
        /* Single global reduction (no atomics) */
        #pragma omp critical
        for(int k=0;k<n*3;k++) f[k]+=lf[k];
        free(lf);
    }
    struct timespec t1; clock_gettime(CLOCK_MONOTONIC,&t1);
    return (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)*1e-9;
}

int main(int argc, char *argv[]) {
    if(argc<2){ printf("Usage: %s <file.gro> [cutoff] [nsteps]\n",argv[0]); return 1; }
    float cut = argc>2?atof(argv[2]):1.0f;
    int steps = argc>3?atoi(argv[3]):200;
    
    GroData *g = read_gro(argv[1]);
    if(!g) return 1;
    
    printf("===== PTO-GROMACS 端到端 Benchmark v2 =====\n");
    printf("File: %s | Atoms: %d | Box: %.1fx%.1fx%.1f nm\n", argv[1], g->natoms, g->box[0], g->box[1], g->box[2]);
    printf("Cutoff: %.2f nm | Steps: %d | SVE: %d-bit (%d floats) | Threads: %d\n\n",
           cut, steps, svcntb()*8, svcntw(), omp_get_max_threads());
    
    printf("Building nblist...\n"); NBList *nl = build_nblist(g->x, g->natoms, g->box, cut);
    printf("NBlist built.\n"); printf("Neighbor pairs: %d (%.1f neighbors/atom)\n\n", nl->total_pairs, 2.0f*nl->total_pairs/g->natoms);
    
    float *f1=malloc(g->natoms*3*sizeof(float)), *f2=malloc(g->natoms*3*sizeof(float));
    
    /* Warmup */
    scalar_nb(g->x,f1,g->natoms,g->box,nl,cut);
    pto_sve_nb(g->x,f2,g->natoms,g->box,nl,cut);
    
    /* Scalar */
    double ts=0;
    for(int s=0;s<steps;s++) ts+=scalar_nb(g->x,f1,g->natoms,g->box,nl,cut);
    printf("Scalar:  %.3f ms/step | %.1f M pairs/s\n", ts/steps*1000, (double)nl->total_pairs*steps/ts/1e6);
    
    /* PTO-SVE */
    double tp=0;
    for(int s=0;s<steps;s++) tp+=pto_sve_nb(g->x,f2,g->natoms,g->box,nl,cut);
    printf("PTO-SVE: %.3f ms/step | %.1f M pairs/s\n", tp/steps*1000, (double)nl->total_pairs*steps/tp/1e6);
    
    /* Force validation */
    float maxd=0;
    for(int k=0;k<g->natoms*3;k++){ float d=fabsf(f1[k]-f2[k]); if(d>maxd)maxd=d; }
    
    printf("\n===== 结果 =====\n");
    printf("Speedup: %.2fx\n", ts/tp);
    printf("Force max diff: %.6e\n", maxd);
    printf("Scalar: %.1f ns/day (nonbonded only) | PTO-SVE: %.1f ns/day\n",
           0.002*steps/ts*86400, 0.002*steps/tp*86400);
    
    free(f1);free(f2);free(nl->start);free(nl->jatoms);free(nl->count);free(nl);free(g->x);free(g);
    return 0;
}
