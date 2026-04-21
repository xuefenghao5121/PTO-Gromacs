/*
 * PTO-GROMACS Large Scale Benchmark - Cell-list NL version
 * Modified from pto_e2e_v3.c with O(N) cell-list neighbor building
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#ifdef __AVX__
#include <immintrin.h>
#define HAS_AVX 1
#else
#define HAS_AVX 0
#endif

/* ============ GRO parser ============ */
typedef struct { int natoms; float *x; float box[3]; } GroData;
static GroData* read_gro(const char *fn) {
    FILE *fp = fopen(fn,"r"); if(!fp) return NULL;
    GroData *g = calloc(1,sizeof(GroData)); char line[256];
    fgets(line,sizeof(line),fp); fgets(line,sizeof(line),fp);
    g->natoms = atoi(line);
    g->x = (float*)malloc(g->natoms*3*sizeof(float));
    for(int i=0;i<g->natoms;i++){
        fgets(line,sizeof(line),fp);
        char *p = line+20;
        g->x[i*3+0]=strtof(p,&p); g->x[i*3+1]=strtof(p,&p); g->x[i*3+2]=strtof(p,&p);
    }
    fgets(line,sizeof(line),fp);
    sscanf(line,"%f%f%f",&g->box[0],&g->box[1],&g->box[2]);
    fclose(fp); return g;
}

/* ============ Neighbor list (cell-list O(N)) ============ */
typedef struct { int *start; int *jatoms; int *count; int total_pairs; } NBList;

static NBList* build_nblist_cell(const float *x, int n, const float box[3], float cut) {
    NBList *nl = calloc(1,sizeof(NBList));
    nl->start = malloc(n*sizeof(int));
    nl->count = calloc(n,sizeof(int));
    float csq = cut*cut;
    
    /* Build cell lists */
    int ncx = (int)(box[0]/cut); if(ncx<1) ncx=1;
    int ncy = (int)(box[1]/cut); if(ncy<1) ncy=1;
    int ncz = (int)(box[2]/cut); if(ncz<1) ncz=1;
    int ncells = ncx*ncy*ncz;
    int *cell_head = malloc(ncells*sizeof(int));
    int *cell_next = malloc(n*sizeof(int));
    for(int c=0;c<ncells;c++) cell_head[c]=-1;
    for(int i=0;i<n;i++){
        int cx=(int)((x[i*3]/box[0]+1.0f)*ncx)%ncx;
        int cy=(int)((x[i*3+1]/box[1]+1.0f)*ncy)%ncy;
        int cz=(int)((x[i*3+2]/box[2]+1.0f)*ncz)%ncz;
        if(cx<0)cx+=ncx; if(cy<0)cy+=ncy; if(cz<0)cz+=ncz;
        int c = cx + cy*ncx + cz*ncx*ncy;
        cell_next[i] = cell_head[c];
        cell_head[c] = i;
    }
    
    /* Count pairs */
    int total=0;
    for(int i=0;i<n;i++){
        int cx=(int)((x[i*3]/box[0]+1.0f)*ncx)%ncx;
        int cy=(int)((x[i*3+1]/box[1]+1.0f)*ncy)%ncy;
        int cz=(int)((x[i*3+2]/box[2]+1.0f)*ncz)%ncz;
        if(cx<0)cx+=ncx; if(cy<0)cy+=ncy; if(cz<0)cz+=ncz;
        for(int dcx=-1;dcx<=1;dcx++) for(int dcy=-1;dcy<=1;dcy++) for(int dcz=-1;dcz<=1;dcz++){
            int ncx2=(cx+dcx+ncx)%ncx, ncy2=(cy+dcy+ncy)%ncy, ncz2=(cz+dcz+ncz)%ncz;
            int c = ncx2+ncy2*ncx+ncz2*ncx*ncy;
            for(int j=cell_head[c]; j>=0; j=cell_next[j]){
                if(j<=i) continue;
                float dx=x[i*3]-x[j*3],dy=x[i*3+1]-x[j*3+1],dz=x[i*3+2]-x[j*3+2];
                dx-=box[0]*roundf(dx/box[0]); dy-=box[1]*roundf(dy/box[1]); dz-=box[2]*roundf(dz/box[2]);
                if(dx*dx+dy*dy+dz*dz<csq) { nl->count[i]++; total++; }
            }
        }
    }
    
    nl->start[0]=0;
    for(int i=1;i<n;i++) nl->start[i]=nl->start[i-1]+nl->count[i-1];
    nl->jatoms = malloc(total*sizeof(int));
    nl->total_pairs = total;
    
    int *pos = calloc(n,sizeof(int));
    for(int i=0;i<n;i++){
        int cx=(int)((x[i*3]/box[0]+1.0f)*ncx)%ncx;
        int cy=(int)((x[i*3+1]/box[1]+1.0f)*ncy)%ncy;
        int cz=(int)((x[i*3+2]/box[2]+1.0f)*ncz)%ncz;
        if(cx<0)cx+=ncx; if(cy<0)cy+=ncy; if(cz<0)cz+=ncz;
        for(int dcx=-1;dcx<=1;dcx++) for(int dcy=-1;dcy<=1;dcy++) for(int dcz=-1;dcz<=1;dcz++){
            int ncx2=(cx+dcx+ncx)%ncx, ncy2=(cy+dcy+ncy)%ncy, ncz2=(cz+dcz+ncz)%ncz;
            int c = ncx2+ncy2*ncx+ncz2*ncx*ncy;
            for(int j=cell_head[c]; j>=0; j=cell_next[j]){
                if(j<=i) continue;
                float dx=x[i*3]-x[j*3],dy=x[i*3+1]-x[j*3+1],dz=x[i*3+2]-x[j*3+2];
                dx-=box[0]*roundf(dx/box[0]); dy-=box[1]*roundf(dy/box[1]); dz-=box[2]*roundf(dz/box[2]);
                if(dx*dx+dy*dy+dz*dz<csq) nl->jatoms[nl->start[i]+pos[i]++]=j;
            }
        }
    }
    free(pos); free(cell_head); free(cell_next);
    return nl;
}

/* ============ Scalar NB kernel ============ */
static double scalar_nb(const float *x, float *f, int n, const float box[3], const NBList *nl, float cut) {
    double t0 = omp_get_wtime();
    float csq = cut*cut;
    float inv_cut6 = 1.0f/(cut*cut*cut*cut*cut*cut);
    memset(f, 0, n*3*sizeof(float));
    
    #pragma omp parallel for schedule(dynamic,256)
    for(int i=0;i<n;i++){
        float fx=0,fy=0,fz=0;
        float xi=x[i*3],yi=x[i*3+1],zi=x[i*3+2];
        for(int k=0;k<nl->count[i];k++){
            int j=nl->jatoms[nl->start[i]+k];
            float dx=xi-x[j*3], dy=yi-x[j*3+1], dz=zi-x[j*3+2];
            dx-=box[0]*roundf(dx/box[0]); dy-=box[1]*roundf(dy/box[1]); dz-=box[2]*roundf(dz/box[2]);
            float rsq=dx*dx+dy*dy+dz*dz;
            if(rsq>=csq||rsq<1e-6f) continue;
            float inv_r2=1.0f/rsq;
            float inv_r6=inv_r2*inv_r2*inv_r2;
            float fmag=48.0f*inv_r6*(inv_r6-0.5f)*inv_r2;
            fx+=fmag*dx; fy+=fmag*dy; fz+=fmag*dz;
        }
        f[i*3]+=fx; f[i*3+1]+=fy; f[i*3+2]+=fz;
    }
    return omp_get_wtime()-t0;
}

/* ============ PTO Tile NB kernel (AVX2) ============ */
static double pto_tile_avx(const float *x, float *f, int n, const float box[3], const NBList *nl, float cut, int tile_size) {
    double t0 = omp_get_wtime();
    float csq = cut*cut;
    int num_tiles = (n + tile_size - 1) / tile_size;
    memset(f, 0, n*3*sizeof(float));
    
    #pragma omp parallel
    {
        float *lfx = calloc(n, sizeof(float));
        float *lfy = calloc(n, sizeof(float));
        float *lfz = calloc(n, sizeof(float));
        
        #pragma omp for schedule(dynamic,64)
        for(int i=0;i<n;i++){
            float xi=x[i*3], yi=x[i*3+1], zi=x[i*3+2];
            
            /* Group neighbors by tile */
            for(int ti=0; ti<num_tiles; ti++){
                int tbeg = ti*tile_size;
                int tend = tbeg+tile_size; if(tend>n) tend=n;
                
                /* Collect j-atoms in this tile */
                int tile_js[1024]; /* stack buffer */
                int tc=0;
                for(int k=0;k<nl->count[i];k++){
                    int j=nl->jatoms[nl->start[i]+k];
                    if(j>=tbeg && j<tend){
                        tile_js[tc++]=j;
                    }
                }
                if(tc==0) continue;
                
                /* Process tile with AVX2 */
                int v=0;
#if HAS_AVX
                __m256 vxi = _mm256_set1_ps(xi);
                __m256 vyi = _mm256_set1_ps(yi);
                __m256 vzi = _mm256_set1_ps(zi);
                __m256 vfx = _mm256_setzero_ps();
                __m256 vfy = _mm256_setzero_ps();
                __m256 vfz = _mm256_setzero_ps();
                __m256 vinv_cut6 = _mm256_set1_ps(1.0f/(cut*cut*cut*cut*cut*cut));
                
                for(; v+7<tc; v+=8){
                    int idx[8]; for(int m=0;m<8;m++) idx[m]=tile_js[v+m];
                    __m256 vjx = _mm256_set_ps(x[idx[7]*3],x[idx[6]*3],x[idx[5]*3],x[idx[4]*3],x[idx[3]*3],x[idx[2]*3],x[idx[1]*3],x[idx[0]*3]);
                    __m256 vjy = _mm256_set_ps(x[idx[7]*3+1],x[idx[6]*3+1],x[idx[5]*3+1],x[idx[4]*3+1],x[idx[3]*3+1],x[idx[2]*3+1],x[idx[1]*3+1],x[idx[0]*3+1]);
                    __m256 vjz = _mm256_set_ps(x[idx[7]*3+2],x[idx[6]*3+2],x[idx[5]*3+2],x[idx[4]*3+2],x[idx[3]*3+2],x[idx[2]*3+2],x[idx[1]*3+2],x[idx[0]*3+2]);
                    __m256 dx = _mm256_sub_ps(vxi, vjx);
                    __m256 dy = _mm256_sub_ps(vyi, vjy);
                    __m256 dz = _mm256_sub_ps(vzi, vjz);
                    __m256 rsq = _mm256_fmadd_ps(dx,dx,_mm256_fmadd_ps(dy,dy,_mm256_mul_ps(dz,dz)));
                    __m256 inv_r2 = _mm256_div_ps(_mm256_set1_ps(1.0f), rsq);
                    __m256 inv_r6 = _mm256_mul_ps(inv_r2, _mm256_mul_ps(inv_r2, inv_r2));
                    __m256 fmag = _mm256_mul_ps(_mm256_set1_ps(48.0f), _mm256_mul_ps(inv_r6, _mm256_mul_ps(_mm256_sub_ps(inv_r6, _mm256_set1_ps(0.5f)), inv_r2)));
                    vfx = _mm256_fmadd_ps(fmag, dx, vfx);
                    vfy = _mm256_fmadd_ps(fmag, dy, vfy);
                    vfz = _mm256_fmadd_ps(fmag, dz, vfz);
                }
                /* Horizontal reduce */
                float hfx[8],hfy[8],hfz[8];
                _mm256_storeu_ps(hfx,vfx); _mm256_storeu_ps(hfy,vfy); _mm256_storeu_ps(hfz,vfz);
                float sfx=hfx[0]+hfx[1]+hfx[2]+hfx[3]+hfx[4]+hfx[5]+hfx[6]+hfx[7];
                float sfy=hfy[0]+hfy[1]+hfy[2]+hfy[3]+hfy[4]+hfy[5]+hfy[6]+hfy[7];
                float sfz=hfz[0]+hfz[1]+hfz[2]+hfz[3]+hfz[4]+hfz[5]+hfz[6]+hfz[7];
                lfx[i]+=sfx; lfy[i]+=sfy; lfz[i]+=sfz;
#endif
                /* Remainder loop */
                float fx=0,fy=0,fz=0;
                for(;v<tc;v++){
                    int j=tile_js[v];
                    float dx=xi-x[j*3],dy=yi-x[j*3+1],dz=zi-x[j*3+2];
                    dx-=box[0]*roundf(dx/box[0]);dy-=box[1]*roundf(dy/box[1]);dz-=box[2]*roundf(dz/box[2]);
                    float rsq=dx*dx+dy*dy+dz*dz;
                    if(rsq>=csq||rsq<1e-6f) continue;
                    float inv_r2=1.0f/rsq, inv_r6=inv_r2*inv_r2*inv_r2;
                    float fmag=48.0f*inv_r6*(inv_r6-0.5f)*inv_r2;
                    fx+=fmag*dx;fy+=fmag*dy;fz+=fmag*dz;
                }
                lfx[i]+=fx; lfy[i]+=fy; lfz[i]+=fz;
            }
        }
        
        /* Reduce */
        #pragma omp critical
        for(int i=0;i<n;i++){
            f[i*3]+=lfx[i]; f[i*3+1]+=lfy[i]; f[i*3+2]+=lfz[i];
        }
        free(lfx); free(lfy); free(lfz);
    }
    return omp_get_wtime()-t0;
}

/* ============ PTO Tile NB kernel (Scalar tile) ============ */
static double pto_tile_scalar(const float *x, float *f, int n, const float box[3], const NBList *nl, float cut, int tile_size) {
    double t0 = omp_get_wtime();
    float csq = cut*cut;
    memset(f, 0, n*3*sizeof(float));
    
    #pragma omp parallel
    {
        float *lfx = calloc(n, sizeof(float));
        float *lfy = calloc(n, sizeof(float));
        float *lfz = calloc(n, sizeof(float));
        
        #pragma omp for schedule(dynamic,64)
        for(int i=0;i<n;i++){
            float xi=x[i*3],yi=x[i*3+1],zi=x[i*3+2];
            float fx=0,fy=0,fz=0;
            for(int k=0;k<nl->count[i];k++){
                int j=nl->jatoms[nl->start[i]+k];
                float dx=xi-x[j*3], dy=yi-x[j*3+1], dz=zi-x[j*3+2];
                dx-=box[0]*roundf(dx/box[0]); dy-=box[1]*roundf(dy/box[1]); dz-=box[2]*roundf(dz/box[2]);
                float rsq=dx*dx+dy*dy+dz*dz;
                if(rsq>=csq||rsq<1e-6f) continue;
                float inv_r2=1.0f/rsq, inv_r6=inv_r2*inv_r2*inv_r2;
                float fmag=48.0f*inv_r6*(inv_r6-0.5f)*inv_r2;
                fx+=fmag*dx; fy+=fmag*dy; fz+=fmag*dz;
            }
            lfx[i]+=fx; lfy[i]+=fy; lfz[i]+=fz;
        }
        
        #pragma omp critical
        for(int i=0;i<n;i++){
            f[i*3]+=lfx[i]; f[i*3+1]+=lfy[i]; f[i*3+2]+=lfz[i];
        }
        free(lfx); free(lfy); free(lfz);
    }
    return omp_get_wtime()-t0;
}

int main(int argc, char *argv[]) {
    if(argc<2){ printf("Usage: %s <file.gro> [cutoff] [nsteps] [tile_size]\n",argv[0]); return 1; }
    float cut = argc>2?atof(argv[2]):1.0f;
    int steps = argc>3?atoi(argv[3]):100;
    int tile_size = argc>4?atoi(argv[4]):64;
    
    GroData *g = read_gro(argv[1]);
    if(!g) { fprintf(stderr,"Cannot read %s\n",argv[1]); return 1; }
    int n = g->natoms;
    
    printf("================================================================\n");
    printf("  PTO-GROMACS Large Scale Benchmark (Cell-list NL)\n");
    printf("================================================================\n");
    printf("Atoms: %d | Box: %.1fx%.1fx%.1f | Cut: %.2f | Steps: %d | Tile: %d\n",
           n, g->box[0],g->box[1],g->box[2], cut, steps, tile_size);
#if HAS_AVX
    printf("Mode: x86 AVX2+FMA (256-bit, 8 floats)\n");
#else
    printf("Mode: x86 Scalar\n");
#endif
    printf("Threads: %d\n\n", omp_get_max_threads());
    
    double t_nl_start = omp_get_wtime();
    printf("Building neighbor list (cell-list)..."); fflush(stdout);
    NBList *nl = build_nblist_cell(g->x, n, g->box, cut);
    double t_nl = omp_get_wtime()-t_nl_start;
    printf(" %d pairs (%.1f nbs/atom) [%.2f s]\n\n", nl->total_pairs, 2.0f*nl->total_pairs/n, t_nl);
    
    float *f_scalar = malloc(n*3*sizeof(float));
    float *f_pto = malloc(n*3*sizeof(float));
    
    /* Scalar baseline */
    printf("Scalar baseline:     "); fflush(stdout);
    scalar_nb(g->x, f_scalar, n, g->box, nl, cut);
    double ts=0;
    for(int s=0;s<steps;s++) ts+=scalar_nb(g->x, f_scalar, n, g->box, nl, cut);
    printf("%.3f ms/step\n", ts/steps*1000);
    
    /* PTO Tile Scalar */
    printf("PTO Tile Scalar:     "); fflush(stdout);
    pto_tile_scalar(g->x, f_pto, n, g->box, nl, cut, tile_size);
    double tps=0;
    for(int s=0;s<steps;s++) tps+=pto_tile_scalar(g->x, f_pto, n, g->box, nl, cut, tile_size);
    printf("%.3f ms/step (%.2fx)\n", tps/steps*1000, ts/tps);
    
#if HAS_AVX
    /* PTO Tile AVX2 */
    printf("PTO Tile AVX2+FMA:   "); fflush(stdout);
    pto_tile_avx(g->x, f_pto, n, g->box, nl, cut, tile_size);
    double tpa=0;
    for(int s=0;s<steps;s++) tpa+=pto_tile_avx(g->x, f_pto, n, g->box, nl, cut, tile_size);
    printf("%.3f ms/step (%.2fx)\n", tpa/steps*1000, ts/tpa);
#endif
    
    /* Force validation */
    float max_diff = 0;
    for(int k=0;k<n*3;k++){float d=fabsf(f_scalar[k]-f_pto[k]); if(d>max_diff) max_diff=d;}
    
    printf("\n================================================================\n");
    printf("  性能报告\n");
    printf("================================================================\n");
    printf("Atoms: %d | Pairs: %d | Threads: %d | NL build: %.2f s\n", n, nl->total_pairs, omp_get_max_threads(), t_nl);
    printf("Scalar baseline:  %.3f ms/step (1.00x)\n", ts/steps*1000);
    printf("PTO Tile Scalar:  %.3f ms/step (%.2fx)\n", tps/steps*1000, ts/tps);
#if HAS_AVX
    printf("PTO Tile AVX2:    %.3f ms/step (%.2fx)\n", tpa/steps*1000, ts/tpa);
    printf("AVX2 vs Scalar:   %.2fx speedup\n", tps/tpa);
#endif
    printf("Force max diff: %.6e\n", max_diff);
    printf("================================================================\n");
    
    free(f_scalar); free(f_pto);
    free(nl->start); free(nl->jatoms); free(nl->count); free(nl);
    free(g->x); free(g);
    return 0;
}
